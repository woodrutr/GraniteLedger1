"""Engine orchestration and validation helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union, List

from engine.data_loaders.ei_units import load_ei_units
from common.validators import validate_demand_table, DemandValidationError
from engine.frames import FramePipeline, ModelInputBundle
from engine.inputs.demand_source import resolve_demand_frame
from engine.outputs import EngineOutputs
from engine.run_loop import run_end_to_end
from normalization import normalize_token

try:  # pandas is only used for type checking when available
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    pd = None  # type: ignore

try:
    from granite_io.frames_api import Frames, PolicySpec
    from policy.allowance_annual import ConfigError as PolicyConfigError
except Exception:  # pragma: no cover - fallback for import cycles during tests
    Frames = object  # type: ignore
    PolicySpec = object  # type: ignore
    PolicyConfigError = Exception  # type: ignore


FramesType = Union["Frames", Mapping[str, "pd.DataFrame"]]  # type: ignore[name-defined]

_PIPELINE = FramePipeline()


def build_load_frame(config: Mapping[str, Any]) -> "pd.DataFrame":
    """Return the consolidated load forecast frame filtered by scenario."""

    if pd is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError("pandas is required to build load frames")

    from engine.data_loaders.load_forecasts import load_forecasts

    csv_path = config.get("load_forecast_csv")
    frame = load_forecasts(csv_path)
    if not isinstance(frame, pd.DataFrame):
        return pd.DataFrame(columns=["scenario"])

    raw_scenario = str(config.get("scenario", "isone_2025_forecast")).strip()
    scenario_token = normalize_token(raw_scenario)
    if not raw_scenario:
        return frame

    working = frame.copy()
    working["scenario"] = working["scenario"].astype(str)
    if not scenario_token:
        return working

    scenario_matches = working["scenario"].map(normalize_token)
    mask = scenario_matches == scenario_token
    return working[mask]


def prepare_units(config: Mapping[str, Any]) -> "pd.DataFrame":
    """Load and register EI-format unit data from the provided configuration."""

    if pd is None:  # pragma: no cover - optional dependency guard
        raise RuntimeError("pandas is required to prepare unit data")

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping of configuration values")

    try:
        path = config["ei_units_csv"]
    except KeyError as exc:
        raise KeyError("config is missing the 'ei_units_csv' entry") from exc

    df = load_ei_units(path)

    try:
        import engine as engine_pkg
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("Failed to import the engine package for unit registration") from exc

    engine_pkg.set_units(df)
    return df


def _coerce_years(cfg: Mapping[str, Any] | None) -> Optional[List[int]]:
    if not isinstance(cfg, Mapping):
        return None

    years = cfg.get("years")
    if isinstance(years, Iterable) and not isinstance(years, (str, bytes)):
        normalized: set[int] = set()
        for raw in years:
            try:
                normalized.add(int(raw))
            except Exception:
                continue
        if normalized:
            return sorted(normalized)

    start = cfg.get("start_year")
    end = cfg.get("end_year")
    try:
        if start is not None and end is not None:
            a, b = int(start), int(end)
            if b < a:
                a, b = b, a
            return list(range(a, b + 1))
    except Exception:
        return None

    return None


def _cfg_bool(cfg: Mapping[str, Any], key: str, default: bool) -> bool:
    value = cfg.get(key, default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    try:
        return bool(value)
    except Exception:
        return bool(default)


def _cfg_float(cfg: Mapping[str, Any], key: str, default: float) -> float:
    try:
        value = cfg.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _first_present(cfg: Mapping[str, Any], keys: Iterable[str], default: Any) -> Any:
    for key in keys:
        if key in cfg:
            return cfg[key]
    return default


def _inputs_have_demand(inputs: FramesType) -> bool:
    if hasattr(inputs, "optional_frame") and callable(getattr(inputs, "optional_frame")):
        try:
            frame = inputs.optional_frame("demand")  # type: ignore[attr-defined]
        except Exception:
            frame = None
        else:
            if frame is not None:
                empty = getattr(frame, "empty", None)
                if empty is not None:
                    return not bool(empty)
                return True
    if hasattr(inputs, "has_frame") and callable(getattr(inputs, "has_frame")):
        try:
            if inputs.has_frame("demand"):  # type: ignore[attr-defined]
                return True
        except Exception:
            pass
    if isinstance(inputs, Mapping):
        for key, value in inputs.items():
            if str(key).strip().lower() != "demand":
                continue
            if pd is not None and isinstance(value, pd.DataFrame):
                return not value.empty
            empty = getattr(value, "empty", None)
            if empty is not None:
                return not bool(empty)
            if value is not None:
                return True
    return False


def _inject_demand_frame(inputs: FramesType, demand_df: "pd.DataFrame") -> FramesType:
    if hasattr(inputs, "with_frame") and callable(getattr(inputs, "with_frame")):
        try:
            return inputs.with_frame("demand", demand_df)  # type: ignore[attr-defined]
        except Exception:
            pass
    if isinstance(inputs, Mapping):
        updated: dict[str, object] = {}
        for key, value in inputs.items():
            updated[key] = value
        updated["demand"] = demand_df
        return updated  # type: ignore[return-value]
    return inputs


def _build_bundle(
    inputs: FramesType,
    *,
    policy_enabled: bool,
    banking_enabled: bool,
    years: Sequence[int] | None,
    carbon_price_schedule: Mapping[Any, Any] | float | None,
    carbon_price_value: float | None,
) -> ModelInputBundle:
    input_meta: dict[str, object] = {}
    if isinstance(inputs, Frames):
        frame_mapping: dict[str, "pd.DataFrame"] = {name: inputs.frame(name) for name in inputs}
        meta_mapping = getattr(inputs, "_meta", {})
        if isinstance(meta_mapping, Mapping):
            input_meta.update(meta_mapping)
    elif isinstance(inputs, Mapping):
        frame_mapping = {}
        for name, value in inputs.items():
            key = str(name)
            if isinstance(value, pd.DataFrame):
                frame_mapping[key] = value
            else:
                input_meta[key] = value
    else:
        raise TypeError("inputs must be a Frames instance or mapping of dataframes")

    bundle = _PIPELINE.build_bundle(
        frame_mapping,
        years=years,
        carbon_policy_enabled=policy_enabled,
        banking_enabled=banking_enabled,
        carbon_price_schedule=carbon_price_schedule,
        carbon_price_value=carbon_price_value,
    )
    if input_meta:
        merged_meta = dict(bundle.meta or {})
        merged_meta.update(input_meta)
        return ModelInputBundle(
            frames=bundle.frames,
            vectors=bundle.vectors,
            years=bundle.years,
            policy=bundle.policy,
            meta=merged_meta,
        )
    return bundle


def _validate_demand(frames: "Frames", years: Sequence[int]) -> None:
    try:
        demand = frames.demand()
    except KeyError as exc:
        raise ValueError("Demand data is required to run the model") from exc

    if demand.empty:
        raise ValueError("Demand data is empty; provide demand for each modeled year")

    available_years = {int(year) for year in demand["year"].unique()}
    missing_years = [year for year in years if year not in available_years]
    if missing_years:
        missing = ", ".join(str(year) for year in missing_years)
        raise ValueError(f"Demand data is missing rows for years: {missing}")


def _validate_peak_demand(frames: "Frames", years: Sequence[int]) -> None:
    try:
        peak = frames.peak_demand()
    except KeyError:
        return

    if peak.empty:
        return

    available_years = {int(year) for year in peak["year"].unique()}
    missing_years = [year for year in years if year not in available_years]
    if missing_years:
        missing = ", ".join(str(year) for year in missing_years)
        raise ValueError(f"Peak demand data is missing rows for years: {missing}")


def _validate_units(frames: "Frames") -> None:
    try:
        units = frames.units()
    except KeyError as exc:
        raise ValueError("Unit inventory is required to run the model") from exc

    if units.empty:
        raise ValueError("Unit inventory is empty; provide generating units for all regions")


def _validate_transmission(frames: "Frames", *, use_network: bool) -> None:
    if not use_network:
        return

    transmission = frames.transmission()
    if transmission.empty:
        raise ValueError(
            "Network dispatch requested but no transmission interfaces were provided"
        )

    positive = transmission["limit_mw"].astype(float) > 0.0
    if not bool(positive.any()):
        raise ValueError(
            "Transmission interfaces require positive limit_mw values when network dispatch is enabled"
        )


def _validate_policy(frames: "Frames", years: Sequence[int], *, enabled: bool) -> None:
    if not enabled:
        return

    try:
        policy_spec: PolicySpec = frames.policy()  # type: ignore[assignment]
    except PolicyConfigError as exc:
        raise ValueError(f"Carbon policy configuration is invalid: {exc}") from exc
    except KeyError as exc:
        raise ValueError("Carbon policy is enabled but no policy frame was provided") from exc

    cap_series = getattr(policy_spec, "cap", None)
    if cap_series is None:
        raise ValueError("Carbon policy configuration is missing cap data")

    try:
        policy_obj = policy_spec.to_policy()  # type: ignore[attr-defined]
    except Exception as exc:
        raise ValueError(f"Carbon policy configuration could not be instantiated: {exc}") from exc

    cap_index = getattr(getattr(policy_obj, "cap", None), "index", [])
    available_years: set[int] = set()
    for label in cap_index:
        try:
            year_value = policy_obj.compliance_year_for(label)  # type: ignore[attr-defined]
        except Exception:
            try:
                year_value = int(label)
            except Exception:
                continue
        try:
            available_years.add(int(year_value))
        except Exception:
            continue

    if not available_years:
        raise ValueError("Carbon policy configuration does not contain any annual caps")

    missing_years = [year for year in years if year not in available_years]
    if missing_years:
        missing = ", ".join(str(year) for year in missing_years)
        raise ValueError(
            f"Carbon policy configuration is missing entries for years: {missing}"
        )


def _validate_inputs(
    bundle: ModelInputBundle,
    years: Sequence[int],
    *,
    policy_enabled: bool,
    use_network: bool,
) -> None:
    frames = bundle.frames
    _validate_demand(frames, years)
    _validate_peak_demand(frames, years)
    _validate_units(frames)
    _validate_transmission(frames, use_network=use_network)
    _validate_policy(frames, years, enabled=policy_enabled)


def run_policy_simulation(
    config: Mapping[str, Any] | None,
    inputs: FramesType,
) -> EngineOutputs:
    """Validate inputs and execute the integrated policy + dispatch engine."""

    cfg: Mapping[str, Any] = config or {}

    demand_df: "pd.DataFrame" | None = None
    table = cfg.get("demand_table") if isinstance(cfg, Mapping) else None
    has_table = pd is not None and isinstance(table, pd.DataFrame) and not table.empty
    if has_table:
        demand_df = resolve_demand_frame(cfg)
    elif not _inputs_have_demand(inputs):
        demand_df = resolve_demand_frame(cfg)

    years_hint = _coerce_years(cfg)

    if demand_df is not None:
        try:
            units_path = cfg.get("ei_units_csv") if isinstance(cfg, Mapping) else None
            ei_df = load_ei_units(units_path) if units_path else load_ei_units()
        except Exception:
            ei_df = None
        try:
            demand_df, _ = validate_demand_table(
                demand_df,
                ei_df,
                required_years=years_hint,
            )
        except DemandValidationError as exc:
            details = f" ({exc.details})" if exc.details else ""
            raise ValueError(f"Invalid demand table: {exc.message}{details}") from exc

    inputs_with_demand: FramesType = (
        _inject_demand_frame(inputs, demand_df) if demand_df is not None else inputs
    )

    years = years_hint
    if not years:
        raise ValueError(
            "Model years must be specified via 'years' or 'start_year'/'end_year'"
        )

    policy_enabled = _cfg_bool(cfg, "policy_enabled", True)
    banking_enabled = policy_enabled and _cfg_bool(cfg, "banking_enabled", True)
    use_network = _cfg_bool(cfg, "use_network", False)
    carbon_price_schedule = cfg.get("carbon_price_schedule")
    carbon_price_value = _cfg_float(cfg, "carbon_price_value", 0.0)

    bundle = _build_bundle(
        inputs_with_demand,
        policy_enabled=policy_enabled,
        banking_enabled=banking_enabled,
        years=years,
        carbon_price_schedule=carbon_price_schedule,
        carbon_price_value=carbon_price_value,
    )

    _validate_inputs(
        bundle,
        years,
        policy_enabled=policy_enabled,
        use_network=use_network,
    )

    enable_floor = policy_enabled and _cfg_bool(cfg, "enable_floor", True)
    enable_ccr = policy_enabled and _cfg_bool(cfg, "enable_ccr", True)

    price_initial = _first_present(cfg, ("price_initial", "initial_price"), 0.0)

    schedule_override = carbon_price_schedule
    if not schedule_override:
        schedule_override = bundle.vectors.as_price_schedule("carbon_price")

    outputs = run_end_to_end(
        bundle,
        years=years,
        price_initial=price_initial,
        tol=_cfg_float(cfg, "tol", 1e-3),
        max_iter=int(_first_present(cfg, ("max_iter",), 25)),
        relaxation=_cfg_float(cfg, "relaxation", 0.5),
        enable_floor=enable_floor,
        enable_ccr=enable_ccr,
        price_cap=_cfg_float(cfg, "price_cap", 1000.0),
        use_network=use_network,
        carbon_price_schedule=schedule_override,
        carbon_price_value=carbon_price_value,
        deep_carbon_pricing=_cfg_bool(cfg, "deep_carbon_pricing", False),
        progress_cb=cfg.get("progress_cb"),
        stage_cb=cfg.get("stage_cb"),
        states=cfg.get("states"),
    )

    return outputs


def build_frames(
    *,
    load_root: str,
    selection: Mapping[str, Any] | None = None,
    **_: Any,
) -> "Frames":
    """Compatibility wrapper around :func:`granite_io.frames_api.build_frames`."""

    if not load_root:
        raise ValueError("load_root must be provided to build_frames")

    from engine.io.peak_demand import load_peak_demand
    from engine.io.region_mapping import load_state_zone_maps

    if selection is not None and not isinstance(selection, Mapping):
        raise TypeError("selection must be a mapping when provided")

    root_path = Path(load_root)
    if not root_path.exists():
        raise FileNotFoundError(f"load_root not found: {load_root}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"load_root is not a directory: {load_root}")

    selection_dict: Mapping[str, Any] = selection or {}
    load_selection_raw = selection_dict.get("load") if isinstance(selection_dict, Mapping) else None
    if load_selection_raw is None:
        raise ValueError("selection.load is required to determine the load scenario")
    if not isinstance(load_selection_raw, Mapping):
        raise TypeError("selection.load must be a mapping of load parameters")

    scenario_value = str(load_selection_raw.get("scenario", "")).strip()
    if not scenario_value:
        raise ValueError("selection.load.scenario is required")
    if not normalize_token(scenario_value):
        raise ValueError("selection.load.scenario is required")

    iso_values = load_selection_raw.get("isos")
    if isinstance(iso_values, str):
        iso_list = [iso_values.strip()]
    elif isinstance(iso_values, Iterable):
        iso_list = [
            str(value).strip()
            for value in iso_values
            if not isinstance(value, (bytes, bytearray)) and str(value).strip()
        ]
    else:
        iso_list = []

    iso_tokens: set[str] = set()
    for raw_iso in iso_list:
        token = normalize_token(raw_iso)
        if token:
            iso_tokens.add(token)

    load_config: dict[str, Any] = {
        "load_forecast_csv": load_selection_raw.get("load_forecast_csv", root_path),
        "scenario": scenario_value,
    }

    load_frame = build_load_frame(load_config)
    if load_frame.empty:
        raise RuntimeError(
            "No load forecast data matched the requested scenario configuration"
        )

    if iso_tokens:
        load_frame = load_frame[
            load_frame["iso"].astype(str).map(normalize_token).isin(iso_tokens)
        ]

    if load_frame.empty:
        raise ValueError(
            "No load forecast data matched the requested ISO/scenario selection"
        )

    load_frame = load_frame.copy()
    load_frame["year"] = pd.to_numeric(load_frame["year"], errors="coerce").astype("Int64")
    load_frame["load_gwh"] = pd.to_numeric(load_frame["load_gwh"], errors="coerce")
    load_frame = load_frame.dropna(subset=["year", "load_gwh"])
    load_frame["year"] = load_frame["year"].astype(int)
    load_frame["load_gwh"] = load_frame["load_gwh"].astype(float)

    load_frame = load_frame.rename(columns={"region_id": "zone"})
    ordered_columns = ["iso", "zone", "scenario", "year", "load_gwh"]
    remaining_columns = [col for col in load_frame.columns if col not in ordered_columns]
    load_frame = load_frame.loc[:, ordered_columns + remaining_columns]

    demand_frame = load_frame[["zone", "year", "load_gwh"]].rename(
        columns={"zone": "region", "load_gwh": "demand_gwh"}
    )
    demand_frame["demand_mwh"] = demand_frame["demand_gwh"] * 1_000.0
    demand_frame = demand_frame.drop(columns=["demand_gwh"]).loc[
        :, ["year", "region", "demand_mwh"]
    ]

    peak_frame = load_peak_demand(root_path, scenario_value)
    if peak_frame.empty:
        peak_frame = pd.DataFrame(columns=["year", "region", "peak_demand_mw"])
    else:
        peak_frame = peak_frame.loc[:, ["year", "region", "peak_demand_mw"]]
        peak_frame = peak_frame.sort_values(["year", "region"]).reset_index(drop=True)

    return {
        "load": load_frame.reset_index(drop=True),
        "demand": demand_frame.sort_values(["year", "region"]).reset_index(drop=True),
        "peak_demand": peak_frame,
        "state_zone_maps": load_state_zone_maps(),
    }


__all__ = ["run_policy_simulation", "build_frames", "prepare_units"]

