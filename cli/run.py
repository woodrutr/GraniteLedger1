from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import typer

from engine.outputs import EngineOutputs
from engine.run_loop import run_end_to_end_from_frames
from engine.sensitivity import run_sensitivity_audit
from gui.app import (
    _build_default_frames,
    _build_policy_frame,
    _cached_forecast_frame,
    _ensure_years_in_demand,
    _load_config_data,
    _resolve_forecast_base_path,
    _years_from_config,
)
from gui.demand_helpers import (
    _ScenarioSelection,
    _default_scenario_manifests,
    _manifests_from_selection,
)

try:  # pragma: no cover - optional metadata for manifest reporting
    from gui.app import _GIT_COMMIT
except ImportError:  # pragma: no cover - CLI tests provide minimal stubs
    _GIT_COMMIT = None  # type: ignore[assignment]

try:  # pragma: no cover - optional type alias exposed by the GUI module
    from gui.app import FramesType as _FramesType
except ImportError:  # pragma: no cover - CLI tests provide minimal stubs
    from typing import Any as _FramesType

FramesType = _FramesType


app = typer.Typer(
    help='Run the Granite Ledger simulation engine end-to-end.',
    invoke_without_command=True,
)


def _forecast_manifests_from_config(config: Mapping[str, Any]) -> tuple[list[_ScenarioSelection], str]:
    """Return forecast manifests resolved from the CLI configuration."""

    base_path = _resolve_forecast_base_path()
    frame = _cached_forecast_frame(base_path)

    manifests: list[_ScenarioSelection] = []
    modules_cfg = config.get("modules") if isinstance(config, Mapping) else None
    if isinstance(modules_cfg, Mapping):
        demand_cfg = modules_cfg.get("demand")
        if isinstance(demand_cfg, Mapping):
            selection_cfg = demand_cfg.get("load_forecasts")
            if isinstance(selection_cfg, Mapping) and selection_cfg:
                manifests = _manifests_from_selection(
                    selection_cfg,
                    frame=frame,
                    base_path=base_path,
                )

    if not manifests:
        try:
            manifests = _default_scenario_manifests(frame, base_path=base_path)
        except Exception:  # pragma: no cover - defensive guard
            manifests = []

    return manifests, base_path


def _manifest_records(
    manifests: Sequence[_ScenarioSelection],
    *,
    base_path: str | Path | None,
) -> list[dict[str, Any]]:
    """Convert scenario selections into manifest records."""

    root = Path(base_path) if base_path else None
    records: list[dict[str, Any]] = []
    for manifest in manifests:
        path_value: str | None = None
        if root is not None:
            try:
                path_value = str((root / manifest.iso / manifest.scenario).resolve())
            except OSError:  # pragma: no cover - best-effort path resolution
                path_value = str(root / manifest.iso / manifest.scenario)
        records.append(
            {
                "iso": manifest.iso,
                "scenario": manifest.scenario,
                "zones": list(manifest.zones),
                "years": list(manifest.years),
                "manifest": f"{manifest.iso}::{manifest.scenario}",
                "path": path_value,
            }
        )
    return records


def _parse_years_option(value: str) -> list[int]:
    """Parse the ``--years`` option into a sorted list of integers."""

    years: set[int] = set()
    for part in value.split(','):
        token = part.strip()
        if not token:
            continue
        if '-' in token:
            start_str, end_str = token.split('-', 1)
            try:
                start = int(start_str.strip())
                end = int(end_str.strip())
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Invalid year range '{token}'") from exc
            step = 1 if end >= start else -1
            years.update(range(start, end + step, step))
        else:
            try:
                years.add(int(token))
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Invalid year '{token}'") from exc

    if not years:
        raise ValueError('No valid years supplied')

    return sorted(years)


def _resolve_years(years_option: str | None, config: Mapping[str, Any]) -> list[int]:
    """Determine the simulation years from CLI overrides or configuration."""

    if years_option:
        return _parse_years_option(years_option)

    years = _years_from_config(config)
    if not years:
        raise ValueError('No simulation years specified; supply --years to override the config')
    return years


def _coerce_bool(value: Any, *, default: bool = True) -> bool:
    """Interpret booleans from common TOML representations."""

    if value in (None, ''):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(int(value))
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'true', 't', 'yes', 'y', '1', 'on'}:
            return True
        if normalized in {'false', 'f', 'no', 'n', '0', 'off'}:
            return False
    return default


def _extract_policy_flags(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return policy enablement flags derived from the configuration mapping."""

    market_cfg = config.get('allowance_market')
    if not isinstance(market_cfg, Mapping):
        market_cfg = {}

    carbon_enabled = _coerce_bool(market_cfg.get('enabled'), default=True)
    ccr1_enabled = _coerce_bool(market_cfg.get('ccr1_enabled'), default=True)
    ccr2_enabled = _coerce_bool(market_cfg.get('ccr2_enabled'), default=True)

    banking_key = 'allowance_banking_enabled'
    if banking_key not in market_cfg:
        banking_key = 'bank_enabled'
    banking_enabled = _coerce_bool(market_cfg.get(banking_key), default=True)

    if not carbon_enabled:
        banking_enabled = False

    control_period: int | None
    raw_control = market_cfg.get('control_period_years')
    if raw_control in (None, ''):
        control_period = None
    else:
        try:
            control_period = int(raw_control)
        except (TypeError, ValueError):
            control_period = None
        if control_period is not None and control_period <= 0:
            control_period = None

    floor_mode = str(market_cfg.get('floor_escalator_mode', 'fixed')).strip().lower()
    if floor_mode not in {'percent', 'fixed'}:
        floor_mode = 'fixed'

    floor_value = market_cfg.get('floor_escalator_value')
    try:
        floor_growth = float(floor_value)
    except (TypeError, ValueError):
        floor_growth = 0.0

    ccr1_growth = market_cfg.get('ccr1_escalator_pct', 0.0)
    try:
        ccr1_growth = float(ccr1_growth)
    except (TypeError, ValueError):
        ccr1_growth = 0.0

    ccr2_growth = market_cfg.get('ccr2_escalator_pct', 0.0)
    try:
        ccr2_growth = float(ccr2_growth)
    except (TypeError, ValueError):
        ccr2_growth = 0.0

    return {
        'carbon_policy_enabled': carbon_enabled,
        'ccr1_enabled': ccr1_enabled,
        'ccr2_enabled': ccr2_enabled,
        'banking_enabled': banking_enabled,
        'control_period_years': control_period,
        'floor_escalator_mode': floor_mode,
        'floor_escalator_value': floor_growth,
        'ccr1_escalator_pct': ccr1_growth,
        'ccr2_escalator_pct': ccr2_growth,
    }


def _normalize_run_id_label(value: str) -> str:
    sanitized = [
        char if char.isalnum() or char in {"-", "_"} else "-"
        for char in str(value).strip()
    ]
    normalized = "".join(sanitized).strip("-_")
    return normalized or "run"


def _derive_run_id(config: Mapping[str, Any], years: Sequence[int]) -> str:
    explicit = config.get("output_name")
    if isinstance(explicit, str) and explicit.strip():
        return _normalize_run_id_label(explicit)

    payload = {"years": list(map(int, years)), "modules": config.get("modules", {})}
    digest_source = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha1(digest_source).hexdigest()[:8]
    year_label = "-".join(str(year) for year in years) if years else "all-years"
    return f"cli_{year_label}_{digest}"


def _run_engine(
    config: Mapping[str, Any],
    years: Sequence[int],
    *,
    use_network: bool = False,
) -> tuple[Frames, EngineOutputs, Sequence[Mapping[str, Any]]]:
    """Execute the dispatch/allowance engine for ``years`` using ``config``."""

    flags = _extract_policy_flags(config)
    carbon_enabled = bool(flags['carbon_policy_enabled'])
    ccr1_enabled = bool(flags['ccr1_enabled'])
    ccr2_enabled = bool(flags['ccr2_enabled'])
    banking_enabled = bool(flags['banking_enabled'])
    control_period = flags.get('control_period_years')

    manifests, forecast_base_path = _forecast_manifests_from_config(config)

    frames = _build_default_frames(
        years,
        carbon_policy_enabled=carbon_enabled,
        banking_enabled=banking_enabled,
        forecast_bundles=manifests,
    )
    frames = _ensure_years_in_demand(frames, years)
    policy_frame = _build_policy_frame(
        config,
        years,
        carbon_enabled,
        ccr1_enabled=ccr1_enabled,
        ccr2_enabled=ccr2_enabled,
        control_period_years=control_period,
        banking_enabled=banking_enabled,
        floor_escalator_mode=flags.get('floor_escalator_mode'),
        floor_escalator_value=flags.get('floor_escalator_value'),
        ccr1_escalator_pct=flags.get('ccr1_escalator_pct'),
        ccr2_escalator_pct=flags.get('ccr2_escalator_pct'),
    )
    frames = frames.with_frame("policy", policy_frame)

    enable_ccr = carbon_enabled and (ccr1_enabled or ccr2_enabled)

    outputs = run_end_to_end_from_frames(
        frames,
        years=years,
        price_initial=0.0,
        enable_floor=carbon_enabled,
        enable_ccr=enable_ccr,
        use_network=use_network,
        states=config.get("states") if isinstance(config, Mapping) else None,
    )

    return frames, outputs, _manifest_records(manifests, base_path=forecast_base_path)


def _write_outputs(outputs: EngineOutputs, out_dir: Path, run_id: str) -> Path:
    """Persist model outputs inside ``out_dir/run_id`` using CLI naming."""

    target_dir = out_dir / run_id
    target_dir.mkdir(parents=True, exist_ok=True)
    outputs.to_csv(
        target_dir,
        annual_filename='allowance.csv',
        emissions_filename='emissions.csv',
        price_filename='prices.csv',
        flows_filename='flows.csv',
    )

    # Expose top-level CSV copies for compatibility with legacy tooling expectations.
    for filename in ('allowance.csv', 'emissions.csv', 'prices.csv', 'flows.csv'):
        source = target_dir / filename
        destination = out_dir / filename
        try:
            shutil.copyfile(source, destination)
        except FileNotFoundError:  # pragma: no cover - defensive guard
            continue

    summary_table = outputs.emissions_summary_table()
    if summary_table.empty:
        typer.secho('Regional emissions summary: no data available.', fg=typer.colors.YELLOW)
    else:
        typer.secho('Regional emissions summary (tons):', fg=typer.colors.BLUE)
        typer.echo(summary_table.to_string(index=False))

    return target_dir


def build_manifest(
    *,
    run_id: str,
    run_directory: Path,
    config_path: Path | None,
    config: Mapping[str, Any],
    years: Sequence[int],
    use_network: bool,
    frames: Frames,
    outputs: EngineOutputs,
    forecast_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Construct a JSON-serialisable manifest describing the CLI run."""

    demand_summary: dict[str, Any] = {"regions": [], "years": [], "rows": 0}
    try:
        demand_df = frames.demand()
    except Exception:  # pragma: no cover - defensive guard
        demand_df = None
    if demand_df is not None and not demand_df.empty:
        demand_summary = {
            "regions": sorted({str(region) for region in demand_df["region"].unique()}),
            "years": sorted({int(year) for year in demand_df["year"].unique()}),
            "rows": int(len(demand_df)),
        }

    outputs_summary = {
        "allowance_rows": int(len(outputs.annual)),
        "emissions_rows": int(len(outputs.emissions_by_region)),
        "price_rows": int(len(outputs.price_by_region)),
        "flow_rows": int(len(outputs.flows)),
    }

    forecast_records = [dict(record) for record in forecast_manifests]
    load_forecast_records = [dict(record) for record in forecast_records]

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "run_directory": str(run_directory),
        "config": {
            "path": str(config_path) if config_path else None,
            "output_name": config.get("output_name"),
        },
        "simulation": {
            "years": [int(year) for year in years],
            "use_network": bool(use_network),
        },
        "demand": demand_summary,
        "outputs": outputs_summary,
        "forecast_bundles": forecast_records,
        "load_forecasts": load_forecast_records,
    }

    modules_cfg = config.get("modules")
    if isinstance(modules_cfg, Mapping):
        manifest["modules"] = {key: value for key, value in modules_cfg.items()}

    return manifest


def _render_manifest_section(value: Any, indent_level: int = 0) -> list[str]:
    indent = "  " * indent_level
    if isinstance(value, Mapping):
        lines: list[str] = []
        for key, item in value.items():
            label = str(key).replace('_', ' ')
            if isinstance(item, (Mapping, list, tuple)):
                lines.append(f"{indent}- **{label}:**")
                lines.extend(_render_manifest_section(item, indent_level + 1))
            else:
                lines.append(f"{indent}- **{label}:** {item}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (Mapping, list, tuple)):
                lines.append(f"{indent}-")
                lines.extend(_render_manifest_section(item, indent_level + 1))
            else:
                lines.append(f"{indent}- {item}")
        return lines or [f"{indent}- []"]
    if isinstance(value, tuple):
        return _render_manifest_section(list(value), indent_level)
    return [f"{indent}{value}"]


def _manifest_to_markdown(manifest: Mapping[str, Any]) -> str:
    lines = ["# Run Manifest", ""]
    lines.extend(_render_manifest_section(manifest))
    return "\n".join(lines).rstrip() + "\n"


def _build_model_documentation(manifest: Mapping[str, Any]) -> str:
    lines = ["# Model Documentation", "", "## Run Summary"]
    run_id = manifest.get("run_id")
    if run_id:
        lines.append(f"- **Run ID:** {run_id}")

    simulation = manifest.get("simulation", {})
    if isinstance(simulation, Mapping):
        years = simulation.get("years")
        if years:
            lines.append(f"- **Years:** {', '.join(str(year) for year in years)}")
        if simulation.get("use_network"):
            lines.append("- **Network dispatch:** Enabled")
        else:
            lines.append("- **Network dispatch:** Disabled")

    demand = manifest.get("demand", {})
    if isinstance(demand, Mapping) and demand.get("regions"):
        lines.append("- **Demand regions:** " + ", ".join(demand.get("regions", [])))

    lines.append("")
    lines.append("## Forecast Bundles")
    bundles = manifest.get("load_forecasts")
    if not isinstance(bundles, list):
        bundles = manifest.get("forecast_bundles", [])
    if isinstance(bundles, list) and bundles:
        for record in bundles:
            if not isinstance(record, Mapping):
                continue
            label = record.get("manifest_name") or "bundle"
            lines.append(f"- **{label}:**")
            zones = record.get("zones") or record.get("regions", [])
            details = {
                "Path": record.get("path"),
                "ISO": record.get("iso"),
                "Scenario": record.get("scenario"),
                "Zones": ", ".join(zones) if isinstance(zones, Sequence) else zones,
            }
            for key, value in details.items():
                if value:
                    lines.append(f"  - **{key}:** {value}")
    else:
        lines.append("- No ISO forecast bundles were applied.")

    lines.append("")
    lines.append("## Outputs")
    outputs = manifest.get("outputs", {})
    if isinstance(outputs, Mapping):
        for key, value in outputs.items():
            lines.append(f"- **{key.replace('_', ' ')}:** {value}")

    return "\n".join(lines).rstrip() + "\n"


def _write_documentation(
    *,
    run_id: str,
    run_directory: Path,
    config_path: Path | None,
    config: Mapping[str, Any],
    years: Sequence[int],
    use_network: bool,
    frames: Frames,
    outputs: EngineOutputs,
    forecast_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    """Write manifest and documentation artifacts for the run."""

    run_directory.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        run_id=run_id,
        run_directory=run_directory,
        config_path=config_path,
        config=config,
        years=years,
        use_network=use_network,
        frames=frames,
        outputs=outputs,
        forecast_manifests=forecast_manifests,
    )

    manifest_json = run_directory / 'run_manifest.json'
    manifest_json.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

    manifest_markdown = run_directory / 'run_manifest.md'
    manifest_markdown.write_text(_manifest_to_markdown(manifest), encoding='utf-8')

    documentation_path = run_directory / 'model_documentation.md'
    documentation_path.write_text(_build_model_documentation(manifest), encoding='utf-8')

    return {
        'manifest_json': manifest_json,
        'manifest_markdown': manifest_markdown,
        'documentation': documentation_path,
    }


def _execute_main(
    config: Path | None = typer.Option(
        None,
        '--config',
        '-c',
        help='Path to the TOML configuration file (defaults to run_config.toml).',
    ),
    years: str | None = typer.Option(
        None,
        '--years',
        help='Override simulation years (e.g. "2025,2026" or "2025-2027").',
    ),
    use_network: bool = typer.Option(
        False,
        '--use-network',
        help='Enable network dispatch with transmission constraints.',
    ),
    out: Path = typer.Option(
        Path('output'),
        '--out',
        '-o',
        help='Directory where CSV outputs will be written.',
    ),
    ) -> None:
    """Run the policy model end-to-end and export CSV outputs."""

    try:
        config_data = _load_config_data(config)
    except Exception as exc:  # pragma: no cover - defensive guard
        typer.secho(f'Failed to load configuration: {exc}', err=True, fg=typer.colors.RED)
        raise typer.Exit(1)

    try:
        simulation_years = _resolve_years(years, config_data)
    except Exception as exc:
        typer.secho(f'Invalid year selection: {exc}', err=True, fg=typer.colors.RED)
        raise typer.Exit(2)

    try:
        frames, outputs, forecast_manifests = _run_engine(
            config_data, simulation_years, use_network=use_network
        )
    except ModuleNotFoundError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(3)
    except Exception as exc:  # pragma: no cover - defensive guard
        typer.secho(f'Model execution failed: {exc}', err=True, fg=typer.colors.RED)
        raise typer.Exit(3)

    run_id = _derive_run_id(config_data, simulation_years)
    try:
        run_directory = _write_outputs(outputs, out, run_id)
    except Exception as exc:  # pragma: no cover - defensive guard
        typer.secho(f"Failed to write outputs: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(4)

    try:
        doc_paths = _write_documentation(
            run_id=run_id,
            run_directory=run_directory,
            config_path=config,
            config=config_data,
            years=simulation_years,
            use_network=use_network,
            frames=frames,
            outputs=outputs,
            forecast_manifests=forecast_manifests,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        typer.secho(f"Failed to write documentation: {exc}", err=True, fg=typer.colors.YELLOW)
    else:
        typer.secho(
            f"Documentation saved to {doc_paths['manifest_json'].parent}",
            fg=typer.colors.BLUE,
        )

    typer.secho(
        (
            "Saved allowance results for years "
            f"{', '.join(map(str, simulation_years))} to {run_directory.resolve()}"
        ),
        fg=typer.colors.GREEN,
    )


@app.callback()
def _default_entrypoint(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None,
        '--config',
        '-c',
        help='Path to the TOML configuration file (defaults to run_config.toml).',
    ),
    years: str | None = typer.Option(
        None,
        '--years',
        help='Override simulation years (e.g. "2025,2026" or "2025-2027").',
    ),
    use_network: bool = typer.Option(
        False,
        '--use-network',
        help='Enable network dispatch with transmission constraints.',
    ),
    out: Path = typer.Option(
        Path('output'),
        '--out',
        '-o',
        help='Directory where CSV outputs will be written.',
    ),
) -> None:
    """Execute the default run command when no explicit subcommand is provided."""

    ctx.obj = {
        'config': config,
        'years': years,
        'use_network': use_network,
        'out': out,
    }

    if ctx.invoked_subcommand is None:
        _execute_main(config, years, use_network, out)


@app.command('main')
def main_command(
    config: Path | None = typer.Option(
        None,
        '--config',
        '-c',
        help='Path to the TOML configuration file (defaults to run_config.toml).',
    ),
    years: str | None = typer.Option(
        None,
        '--years',
        help='Override simulation years (e.g. "2025,2026" or "2025-2027").',
    ),
    use_network: bool = typer.Option(
        False,
        '--use-network',
        help='Enable network dispatch with transmission constraints.',
    ),
    out: Path = typer.Option(
        Path('output'),
        '--out',
        '-o',
        help='Directory where CSV outputs will be written.',
    ),
) -> None:
    """Explicit command alias for the primary model run."""

    _execute_main(config, years, use_network, out)


@app.command()
def sensitivity(
    ctx: typer.Context,
    demand_delta: float = typer.Option(
        0.1,
        '--demand-delta',
        help='Fractional change applied to demand for sensitivity scenarios.',
    ),
    gas_delta: float = typer.Option(
        0.1,
        '--gas-delta',
        help='Fractional change applied to gas fuel prices for sensitivity scenarios.',
    ),
    cap_delta: float = typer.Option(
        0.1,
        '--cap-delta',
        help='Fractional change applied to allowance caps for sensitivity scenarios.',
    ),
) -> None:
    """Run the automated sensitivity audit using the provided configuration."""

    options = ctx.ensure_object(dict)
    config = options.get('config')
    years_option = options.get('years')
    use_network = bool(options.get('use_network', False))

    try:
        config_data = _load_config_data(config)
    except Exception as exc:  # pragma: no cover - defensive guard
        typer.secho(f'Failed to load configuration: {exc}', err=True, fg=typer.colors.RED)
        raise typer.Exit(1)

    try:
        simulation_years = _resolve_years(years_option, config_data)
    except Exception as exc:
        typer.secho(f'Invalid year selection: {exc}', err=True, fg=typer.colors.RED)
        raise typer.Exit(2)

    flags = _extract_policy_flags(config_data)
    carbon_enabled = bool(flags['carbon_policy_enabled'])
    banking_enabled = bool(flags['banking_enabled'])
    ccr1_enabled = bool(flags['ccr1_enabled'])
    ccr2_enabled = bool(flags['ccr2_enabled'])
    control_period = flags.get('control_period_years')

    frames = _build_default_frames(
        simulation_years,
        carbon_policy_enabled=carbon_enabled,
        banking_enabled=banking_enabled,
    )
    frames = _ensure_years_in_demand(frames, simulation_years)
    policy_frame = _build_policy_frame(
        config_data,
        simulation_years,
        carbon_enabled,
        ccr1_enabled=ccr1_enabled,
        ccr2_enabled=ccr2_enabled,
        control_period_years=control_period,
        banking_enabled=banking_enabled,
        floor_escalator_mode=flags.get('floor_escalator_mode'),
        floor_escalator_value=flags.get('floor_escalator_value'),
        ccr1_escalator_pct=flags.get('ccr1_escalator_pct'),
        ccr2_escalator_pct=flags.get('ccr2_escalator_pct'),
    )
    frames = frames.with_frame('policy', policy_frame)

    enable_ccr = carbon_enabled and (ccr1_enabled or ccr2_enabled)

    report = run_sensitivity_audit(
        frames,
        simulation_years,
        demand_delta=demand_delta,
        gas_delta=gas_delta,
        cap_delta=cap_delta,
        enable_floor=carbon_enabled,
        enable_ccr=enable_ccr,
        use_network=use_network,
    )

    summary_table = report.to_dataframe()
    if summary_table.empty:
        typer.secho('No sensitivity results were generated.', fg=typer.colors.YELLOW)
        return

    typer.secho('Sensitivity scenario summary:', fg=typer.colors.BLUE)
    typer.echo(summary_table.to_string(index=False))

    results = report.summary()
    baseline = results.get('baseline', {})

    def _compare(label: str, expected: bool) -> None:
        status = 'OK' if expected else 'CHECK'
        color = typer.colors.GREEN if expected else typer.colors.YELLOW
        typer.secho(f'{label}: {status}', fg=color)

    demand_up = results.get('demand_up', {})
    demand_down = results.get('demand_down', {})
    gas_up = results.get('gas_up', {})
    gas_down = results.get('gas_down', {})
    cap_up = results.get('cap_up', {})
    cap_down = results.get('cap_down', {})

    baseline_emissions = float(baseline.get('total_emissions', 0.0))
    baseline_price = float(baseline.get('avg_allowance_price', 0.0))

    _compare(
        'Demand +10% increases total emissions',
        float(demand_up.get('total_emissions', baseline_emissions)) > baseline_emissions,
    )
    _compare(
        'Demand -10% reduces total emissions',
        float(demand_down.get('total_emissions', baseline_emissions)) < baseline_emissions,
    )
    _compare(
        'Gas price +10% increases allowance price',
        float(gas_up.get('avg_allowance_price', baseline_price)) >= baseline_price,
    )
    _compare(
        'Gas price -10% lowers allowance price',
        float(gas_down.get('avg_allowance_price', baseline_price)) <= baseline_price,
    )
    _compare(
        'Cap +10% reduces allowance price',
        float(cap_up.get('avg_allowance_price', baseline_price)) <= baseline_price,
    )
    _compare(
        'Cap -10% increases allowance price',
        float(cap_down.get('avg_allowance_price', baseline_price)) >= baseline_price,
    )


if __name__ == '__main__':  # pragma: no cover - CLI entry point
    app()
