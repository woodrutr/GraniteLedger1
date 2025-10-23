"""Deep model documentation export helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from pathlib import Path

import pandas as pd

from common.data_access.load_forecasts import repo_load_forecasts
from common.schemas.load_forecast import parse_load_forecast_csv
from engine.normalization import normalize_iso_name
from engine.outputs import EngineOutputs

try:  # pragma: no cover - optional import for type checking
    from granite_io.frames_api import Frames
except Exception:  # pragma: no cover - allow runtime injection
    Frames = Any  # type: ignore[assignment]

def _infer_iso_from_region(region: object) -> str:
    token = str(region or "").strip().upper()
    if not token:
        return ""
    prefix = token.split("_", 1)[0]
    norm = normalize_iso_name(prefix)
    return (norm or prefix or "").upper()


def _load_forecast_table(manifest: Mapping[str, Any] | None) -> pd.DataFrame:
    """
    Return a small table used in the deep doc. Always source data via the
    canonical CSV parser, then project legacy column names expected by deepdoc.
    """

    base: pd.DataFrame | None = None
    try:
        csv_path: Path | None = None
        if isinstance(manifest, Mapping):
            entry = manifest.get("load_forecasts") or manifest
            if isinstance(entry, Mapping):
                raw_path = entry.get("path")
                if raw_path:
                    candidate = Path(str(raw_path))
                    if candidate.is_dir():
                        csv_candidates = sorted(candidate.glob("*.csv"))
                        if csv_candidates:
                            csv_path = csv_candidates[0]
                    elif candidate.is_file():
                        csv_path = candidate
                frames = entry.get("frames") if isinstance(entry, Mapping) else None
                if isinstance(frames, Mapping):
                    for frame in frames.values():
                        if isinstance(frame, pd.DataFrame) and not frame.empty:
                            base = frame
                            break
            elif isinstance(entry, Sequence):
                for candidate_entry in entry:
                    if not isinstance(candidate_entry, Mapping):
                        continue
                    raw_path = candidate_entry.get("path")
                    if raw_path:
                        candidate = Path(str(raw_path))
                        if candidate.is_dir():
                            csv_candidates = sorted(candidate.glob("*.csv"))
                            if csv_candidates:
                                csv_path = csv_candidates[0]
                                break
                        elif candidate.is_file():
                            csv_path = candidate
                            break

        if base is None:
            if csv_path and csv_path.exists():
                base = parse_load_forecast_csv(csv_path)
            else:
                base = repo_load_forecasts()
    except Exception:
        return pd.DataFrame(
            columns=["Year", "Load_GWh", "iso_norm", "scenario_norm", "region_norm"]
        )

    if not isinstance(base, pd.DataFrame) or base.empty:
        return pd.DataFrame(
            columns=["Year", "Load_GWh", "iso_norm", "scenario_norm", "region_norm"]
        )

    work = base.copy()

    year_series = work.get("year")
    if year_series is None:
        year_series = work.get("Year")
    if year_series is None:
        year_series = pd.Series(pd.NA, index=work.index, dtype="float")
    work["Year"] = pd.to_numeric(year_series, errors="coerce").astype("Int64")

    load_series = work.get("load_gwh")
    if load_series is None:
        load_series = work.get("Load_GWh")
    if load_series is None and work.get("load_mwh") is not None:
        load_series = pd.to_numeric(work.get("load_mwh"), errors="coerce") / 1000.0
    if load_series is None:
        load_series = pd.Series(pd.NA, index=work.index, dtype="float")
    work["Load_GWh"] = pd.to_numeric(load_series, errors="coerce")

    region_series = work.get("region_id")
    if region_series is None:
        region_series = work.get("region")
    if region_series is None:
        region_series = pd.Series(pd.NA, index=work.index, dtype="string")
    work["region_norm"] = pd.Series(region_series, dtype="string")

    scenario_series = work.get("scenario")
    if scenario_series is None:
        scenario_series = work.get("scenario_name")
    if scenario_series is None:
        scenario_series = pd.Series(pd.NA, index=work.index, dtype="string")
    work["scenario_norm"] = pd.Series(scenario_series, dtype="string")
    work["iso_norm"] = work["region_norm"].map(_infer_iso_from_region).astype("string")

    out = work.dropna(subset=["Year", "Load_GWh", "region_norm"]).loc[
        :, ["Year", "Load_GWh", "iso_norm", "scenario_norm", "region_norm"]
    ]
    return out.reset_index(drop=True)


def _dispatch_overview(frames: Frames) -> Mapping[str, Any]:
    try:
        units = frames.units()
    except Exception:  # pragma: no cover - defensive guard
        return {}
    if units.empty:
        return {}
    id_column = "unique_id" if "unique_id" in units.columns else "unit_id"
    units_by_fuel = (
        units.groupby("fuel")[[id_column]].nunique()[id_column].to_dict()
    )
    capacity_records = (
        units.groupby(["fuel", "region"])["cap_mw"].sum().reset_index()
    )
    capacity_by_fuel_region = [
        {
            "fuel": str(row.fuel),
            "region": str(row.region),
            "cap_mw": float(row.cap_mw),
        }
        for row in capacity_records.itertuples(index=False)
    ]
    heat_rate = units.groupby("fuel")["hr_mmbtu_per_mwh"].mean().to_dict()
    return {
        "units_by_fuel": {str(fuel): int(count) for fuel, count in units_by_fuel.items()},
        "capacity_mw_by_fuel_region": capacity_by_fuel_region,
        "avg_heat_rate": {str(fuel): float(value) for fuel, value in heat_rate.items()},
    }


def _transmission_overview(frames: Frames) -> list[Mapping[str, Any]]:
    try:
        transmission = frames.transmission()
    except Exception:  # pragma: no cover - defensive guard
        return []
    if transmission.empty:
        return []
    return transmission.to_dict(orient="records")


def _emissions_by_fuel(outputs: EngineOutputs) -> list[Mapping[str, Any]]:
    emissions = getattr(outputs, "emissions_by_fuel", pd.DataFrame())
    if not isinstance(emissions, pd.DataFrame) or emissions.empty:
        return []
    return emissions.to_dict(orient="records")


def build_deep_doc(
    manifest: Mapping[str, Any],
    frames: Frames,
    outputs: EngineOutputs,
) -> dict[str, Any]:
    return {
        "load_forecasts": _load_forecast_table(manifest),
        "dispatch": _dispatch_overview(frames),
        "transmission": _transmission_overview(frames),
        "emissions_by_fuel": _emissions_by_fuel(outputs),
    }


def deep_doc_to_markdown(deep_doc: Mapping[str, Any]) -> str:
    lines = ["# Granite Ledger – Model Documentation"]

    forecasts = deep_doc.get("load_forecasts", [])
    if isinstance(forecasts, pd.DataFrame):
        if not forecasts.empty:
            lines.append("## Load Forecast Data")
            lines.append(forecasts.to_string(index=False))
    elif isinstance(forecasts, Sequence) and forecasts:
        lines.append("## Load Forecast Data")
        for entry in forecasts:
            if not isinstance(entry, Mapping):
                continue
            lines.append(
                f"- **{entry.get('iso')}:** {entry.get('manifest')} (" +
                f"{entry.get('path')})"
            )
            csvs = entry.get("csv_files", [])
            if csvs:
                lines.append("  - Files: " + ", ".join(csvs))
            coverage = entry.get("coverage")
            if isinstance(coverage, Mapping):
                start = coverage.get("first")
                end = coverage.get("last")
                if start is not None and end is not None:
                    span = f"{start}–{end}"
                else:
                    span = "Unknown"
                lines.append("  - Coverage: " + span)

    dispatch = deep_doc.get("dispatch", {})
    if isinstance(dispatch, Mapping) and dispatch:
        lines.append("## Dispatch Overview")
        units_by_fuel = dispatch.get("units_by_fuel", {})
        if isinstance(units_by_fuel, Mapping):
            lines.append("- **Units by Fuel:** " + ", ".join(f"{fuel}: {count}" for fuel, count in units_by_fuel.items()))
        capacity = dispatch.get("capacity_mw_by_fuel_region", [])
        if isinstance(capacity, Sequence) and capacity:
            lines.append("- **Capacity (MW) by Fuel & Region:**")
            for record in capacity:
                lines.append(
                    f"  - {record.get('fuel')} @ {record.get('region')}: {record.get('cap_mw')} MW"
                )
        heat_rate = dispatch.get("avg_heat_rate", {})
        if isinstance(heat_rate, Mapping):
            lines.append(
                "- **Average Heat Rate:** "
                + ", ".join(f"{fuel}: {value:.2f}" for fuel, value in heat_rate.items())
            )

    transmission = deep_doc.get("transmission", [])
    if isinstance(transmission, Sequence) and transmission:
        lines.append("## Transmission Interfaces")

        def _format_number(value: Any) -> str:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return str(value)
            if number.is_integer():
                return f"{int(number)}"
            return f"{number:.2f}".rstrip("0").rstrip(".")

        for record in transmission:
            if not isinstance(record, Mapping):
                continue
            header = (
                f"- {record.get('from_region')} → {record.get('to_region')}"
            )
            details: list[str] = []
            capacity = record.get("capacity_mw")
            if capacity is not None:
                details.append("forward " + _format_number(capacity) + " MW")
            reverse_capacity = record.get("reverse_capacity_mw")
            if reverse_capacity is not None:
                details.append("reverse " + _format_number(reverse_capacity) + " MW")
            efficiency = record.get("efficiency")
            if efficiency is not None:
                details.append("efficiency " + _format_number(efficiency))
            added_cost = record.get("added_cost_per_mwh")
            if added_cost is not None:
                details.append(
                    "added cost " + _format_number(added_cost) + " $/MWh"
                )
            in_service_year = record.get("in_service_year")
            if in_service_year is not None:
                details.append(
                    "in service " + _format_number(in_service_year)
                )
            if details:
                header += ": " + ", ".join(details)
            else:
                limit = record.get("limit_mw")
                if limit is not None:
                    header += " (" + _format_number(limit) + " MW)"
            lines.append(header)

    emissions_fuel = deep_doc.get("emissions_by_fuel", [])
    if isinstance(emissions_fuel, Sequence) and emissions_fuel:
        lines.append("## Emissions by Fuel")
        for record in emissions_fuel:
            if not isinstance(record, Mapping):
                continue
            year = record.get("year")
            fuel = record.get("fuel")
            tons = record.get("emissions_tons")
            lines.append(f"- {year}: {fuel} – {tons}")

    return "\n".join(lines).strip() + "\n"
