"""Sensitivity audit helpers for the allowance engine."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd

from engine.outputs import EngineOutputs
from engine.run_loop import run_end_to_end_from_frames
from io_loader import Frames


@dataclass(frozen=True)
class ScenarioResult:
    """Bundle model outputs with derived summary metrics."""

    label: str
    outputs: EngineOutputs
    summary: dict[str, float]


@dataclass(frozen=True)
class SensitivityAuditReport:
    """Collection of baseline and perturbed scenario results."""

    scenarios: dict[str, ScenarioResult]
    baseline_label: str = "baseline"

    def summary(self) -> dict[str, dict[str, float]]:
        """Return a mapping of scenario labels to summary metrics."""

        return {label: result.summary for label, result in self.scenarios.items()}

    def to_dataframe(self) -> pd.DataFrame:
        """Render the scenario summaries as a pandas DataFrame."""

        rows: list[dict[str, float | str]] = []
        for label, result in sorted(self.scenarios.items()):
            row: dict[str, float | str] = {"scenario": label}
            row.update(result.summary)
            rows.append(row)
        if not rows:
            return pd.DataFrame(columns=["scenario"])
        return pd.DataFrame(rows)

    def baseline(self) -> ScenarioResult:
        """Return the baseline scenario result."""

        try:
            return self.scenarios[self.baseline_label]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError(f"Baseline label '{self.baseline_label}' not found in scenarios") from exc


def _summarize_outputs(outputs: EngineOutputs) -> dict[str, float]:
    """Compute aggregate metrics used to validate sensitivities."""

    annual = outputs.annual

    def _series_mean(column: str) -> float:
        if column not in annual or annual[column].empty:
            return 0.0
        return float(pd.to_numeric(annual[column], errors="coerce").fillna(0.0).mean())

    def _series_sum(column: str) -> float:
        if column not in annual or annual[column].empty:
            return 0.0
        return float(pd.to_numeric(annual[column], errors="coerce").fillna(0.0).sum())

    if "cp_all" in annual.columns:
        allowance_column = "cp_all"
    elif "allowance_price" in annual.columns:
        allowance_column = "allowance_price"
    else:
        allowance_column = "cp_last" if "cp_last" in annual.columns else "p_co2"
    total_price_column = "cp_last" if "cp_last" in annual.columns else "p_co2"
    summary: dict[str, float] = {
        "avg_allowance_price": _series_mean(allowance_column),
        "avg_total_price": _series_mean(total_price_column),
        "total_emissions": _series_sum("emissions_tons"),
        "total_allowances": _series_sum("allowances_minted"),
        "shortage_years": 0.0,
    }

    if "shortage_flag" in annual.columns:
        shortage_series = pd.to_numeric(annual["shortage_flag"], errors="coerce").fillna(0.0)
        summary["shortage_years"] = float(shortage_series.sum())

    if "bank" in annual.columns and not annual.empty:
        summary["final_bank"] = float(
            pd.to_numeric(annual["bank"], errors="coerce").fillna(0.0).iloc[-1]
        )
    else:
        summary["final_bank"] = 0.0

    return summary


def _scale_demand(frames: Frames, factor: float) -> Frames:
    demand = frames.demand()
    demand["demand_mwh"] = pd.to_numeric(demand["demand_mwh"], errors="coerce").fillna(0.0) * float(factor)
    return frames.with_frame("demand", demand)


def _scale_gas_price(frames: Frames, factor: float) -> Frames:
    units = frames.units()
    fuel_column = units["fuel"].astype(str).str.lower()
    mask = fuel_column == "gas"
    if mask.any():
        units.loc[mask, "fuel_price_per_mmbtu"] = pd.to_numeric(
            units.loc[mask, "fuel_price_per_mmbtu"], errors="coerce"
        ).fillna(0.0) * float(factor)
    return frames.with_frame("units", units)


def _scale_cap(frames: Frames, factor: float) -> Frames:
    try:
        policy_df = frames.frame("policy")
    except KeyError:
        spec = frames.policy()
        years = list(spec.cap.index)
        records = []
        for year in years:
            records.append(
                {
                    "year": int(year),
                    "cap_tons": float(spec.cap.loc[year]),
                    "ccr1_qty": float(spec.ccr1_qty.loc[year]),
                    "ccr2_qty": float(spec.ccr2_qty.loc[year]),
                    "floor_dollars": float(spec.floor.loc[year]),
                    "ccr1_trigger": float(spec.ccr1_trigger.loc[year]),
                    "ccr2_trigger": float(spec.ccr2_trigger.loc[year]),
                    "cp_id": str(spec.cp_id.loc[year]),
                    "full_compliance": int(year) in spec.full_compliance_years,
                    "bank0": float(spec.bank0),
                    "annual_surrender_frac": float(spec.annual_surrender_frac),
                    "carry_pct": float(spec.carry_pct),
                    "policy_enabled": bool(spec.enabled),
                    "resolution": getattr(spec, "resolution", "annual"),
                }
            )
        policy_df = pd.DataFrame(records)
    if "cap_tons" in policy_df.columns:
        policy_df["cap_tons"] = pd.to_numeric(policy_df["cap_tons"], errors="coerce").fillna(0.0) * float(factor)
    if "ccr1_qty" in policy_df.columns:
        policy_df["ccr1_qty"] = pd.to_numeric(policy_df["ccr1_qty"], errors="coerce").fillna(0.0) * float(factor)
    if "ccr2_qty" in policy_df.columns:
        policy_df["ccr2_qty"] = pd.to_numeric(policy_df["ccr2_qty"], errors="coerce").fillna(0.0) * float(factor)
    return frames.with_frame("policy", policy_df)


def run_sensitivity_audit(
    frames: Frames | Mapping[str, pd.DataFrame],
    years: Iterable[int],
    *,
    demand_delta: float = 0.1,
    gas_delta: float = 0.1,
    cap_delta: float = 0.1,
    **engine_kwargs: Any,
) -> SensitivityAuditReport:
    """Run baseline and perturbed scenarios returning a structured report."""

    frames_obj = Frames.coerce(frames)
    baseline_kwargs = dict(engine_kwargs)
    baseline_kwargs.setdefault("price_initial", 0.0)
    baseline_outputs = run_end_to_end_from_frames(frames_obj, years=years, **baseline_kwargs)
    scenarios: dict[str, ScenarioResult] = {
        "baseline": ScenarioResult("baseline", baseline_outputs, _summarize_outputs(baseline_outputs))
    }

    scenario_inputs = {
        "demand_down": _scale_demand(frames_obj, 1.0 - demand_delta),
        "demand_up": _scale_demand(frames_obj, 1.0 + demand_delta),
        "gas_down": _scale_gas_price(frames_obj, 1.0 - gas_delta),
        "gas_up": _scale_gas_price(frames_obj, 1.0 + gas_delta),
        "cap_down": _scale_cap(frames_obj, 1.0 - cap_delta),
        "cap_up": _scale_cap(frames_obj, 1.0 + cap_delta),
    }

    for label, scenario_frames in scenario_inputs.items():
        outputs = run_end_to_end_from_frames(scenario_frames, years=years, **baseline_kwargs)
        scenarios[label] = ScenarioResult(label, outputs, _summarize_outputs(outputs))

    return SensitivityAuditReport(scenarios)


def _format_summary(report: SensitivityAuditReport) -> str:
    table = report.to_dataframe()
    if table.empty:
        return "No scenarios evaluated."
    return table.to_string(index=False)


def main(argv: list[str] | None = None) -> int:
    """Entry point for running the sensitivity audit as a script."""

    parser = argparse.ArgumentParser(description="Run the GraniteLedger sensitivity audit.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the TOML configuration file (defaults to run_config.toml).",
    )
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma-separated list or range of simulation years (e.g. 2025-2027).",
    )
    parser.add_argument(
        "--use-network",
        action="store_true",
        help="Enable network dispatch when running scenarios.",
    )
    parser.add_argument(
        "--demand-delta",
        type=float,
        default=0.1,
        help="Fractional change applied to demand for sensitivity scenarios.",
    )
    parser.add_argument(
        "--gas-delta",
        type=float,
        default=0.1,
        help="Fractional change applied to gas fuel prices for sensitivity scenarios.",
    )
    parser.add_argument(
        "--cap-delta",
        type=float,
        default=0.1,
        help="Fractional change applied to allowance caps for sensitivity scenarios.",
    )

    args = parser.parse_args(argv)

    from cli.run import (
        _build_default_frames,
        _build_policy_frame,
        _ensure_years_in_demand,
        _extract_policy_flags,
        _load_config_data,
        _resolve_years,
    )

    try:
        config_data = _load_config_data(args.config)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise SystemExit(f"Failed to load configuration: {exc}")

    try:
        years = _resolve_years(args.years, config_data)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise SystemExit(f"Invalid years specification: {exc}")

    flags = _extract_policy_flags(config_data)
    carbon_enabled = bool(flags['carbon_policy_enabled'])
    banking_enabled = bool(flags['banking_enabled'])
    ccr1_enabled = bool(flags['ccr1_enabled'])
    ccr2_enabled = bool(flags['ccr2_enabled'])

    frames = _build_default_frames(
        years,
        carbon_policy_enabled=carbon_enabled,
        banking_enabled=banking_enabled,
    )
    frames = _ensure_years_in_demand(frames, years)
    policy_frame = _build_policy_frame(
        config_data,
        years,
        carbon_enabled,
        ccr1_enabled=ccr1_enabled,
        ccr2_enabled=ccr2_enabled,
        control_period_years=flags.get('control_period_years'),
        banking_enabled=banking_enabled,
        floor_escalator_mode=flags.get('floor_escalator_mode'),
        floor_escalator_value=flags.get('floor_escalator_value'),
        ccr1_escalator_pct=flags.get('ccr1_escalator_pct'),
        ccr2_escalator_pct=flags.get('ccr2_escalator_pct'),
    )
    frames = frames.with_frame("policy", policy_frame)

    enable_ccr = carbon_enabled and (ccr1_enabled or ccr2_enabled)

    report = run_sensitivity_audit(
        frames,
        years,
        demand_delta=args.demand_delta,
        gas_delta=args.gas_delta,
        cap_delta=args.cap_delta,
        enable_floor=carbon_enabled,
        enable_ccr=enable_ccr,
        use_network=args.use_network,
    )

    print(_format_summary(report))
    return 0


if __name__ == "__main__":  # pragma: no cover - script execution
    raise SystemExit(main())
