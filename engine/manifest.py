"""Utilities to build run manifests and human-readable summaries."""

from __future__ import annotations

from datetime import datetime, timezone
import platform
from typing import Any, Mapping, Sequence

import pandas as pd
from engine.constants import FLOW_TOL

try:  # pragma: no cover - optional import for type checking
    from granite_io.frames_api import Frames
except Exception:  # pragma: no cover - allow build_manifest to accept Mapping
    Frames = Any  # type: ignore[assignment]

from engine.outputs import EngineOutputs


def _to_year_list(frames: Frames) -> list[int]:
    try:
        demand = frames.demand()
    except Exception:  # pragma: no cover - defensive guard
        return []
    if demand.empty or "year" not in demand.columns:
        return []
    years = pd.to_numeric(demand["year"], errors="coerce").dropna().astype(int)
    return sorted(years.unique().tolist())


def _transmission_summary(frames: Frames) -> Mapping[str, Any]:
    try:
        transmission = frames.transmission()
    except Exception:  # pragma: no cover - defensive guard
        transmission = pd.DataFrame(columns=["from_region", "to_region", "limit_mw"])
    if transmission.empty:
        return {
            "interfaces": 0,
            "is_symmetric": True,
            "typical_limit_mw": 0.0,
        }
    interfaces = int(len(transmission))

    capacity_series = (
        pd.to_numeric(transmission.get("capacity_mw"), errors="coerce")
        if "capacity_mw" in transmission
        else pd.Series(dtype=float)
    )
    capacity_series = capacity_series.dropna()
    if capacity_series.empty and "limit_mw" in transmission:
        limit_series = pd.to_numeric(transmission["limit_mw"], errors="coerce").dropna()
        typical = float(limit_series.median()) if not limit_series.empty else 0.0
    else:
        typical = float(capacity_series.median()) if not capacity_series.empty else 0.0

    def _coerce_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    symmetric = True
    if {"capacity_mw", "reverse_capacity_mw"}.issubset(transmission.columns):
        for row in transmission.itertuples(index=False):
            forward = _coerce_float(getattr(row, "capacity_mw", None))
            reverse = _coerce_float(getattr(row, "reverse_capacity_mw", None))
            if forward is None or reverse is None or abs(forward - reverse) > FLOW_TOL:
                symmetric = False
                break
    else:
        value_column = (
            "capacity_mw"
            if "capacity_mw" in transmission.columns
            else "limit_mw"
        )
        pairs = {
            (str(row.from_region), str(row.to_region)): _coerce_float(
                getattr(row, value_column, None)
            )
            for row in transmission.itertuples(index=False)
        }
        for (from_region, to_region), value in pairs.items():
            if value is None:
                symmetric = False
                break
            reverse_key = (to_region, from_region)
            reverse_value = pairs.get(reverse_key)
            if reverse_value is None or abs(reverse_value - value) > FLOW_TOL:
                symmetric = False
                break
    return {
        "interfaces": interfaces,
        "is_symmetric": symmetric,
        "typical_limit_mw": typical,
    }


def _policy_summary(frames: Frames) -> Mapping[str, Any]:
    try:
        policy_spec = frames.policy()
    except Exception:  # pragma: no cover - defensive guard
        return {}
    cap_series = policy_spec.cap.astype(float)
    floor_series = policy_spec.floor.astype(float)
    decline = None
    percent_decline = None
    if len(cap_series) >= 2:
        start = float(cap_series.iloc[0])
        end = float(cap_series.iloc[-1])
        decline = end - start
        percent_decline = (end / start - 1.0) * 100.0 if start else None
    result: dict[str, Any] = {
        "cap_tons": {
            int(year): float(value)
            for year, value in policy_spec.cap.astype(float).items()
        },
        "price_floor": {
            int(year): float(value)
            for year, value in policy_spec.floor.astype(float).items()
        },
        "cap_decline_tons": decline,
        "cap_decline_percent": percent_decline,
        "ccr1_trigger": {
            int(year): float(value)
            for year, value in policy_spec.ccr1_trigger.astype(float).items()
        },
        "ccr1_qty": {
            int(year): float(value)
            for year, value in policy_spec.ccr1_qty.astype(float).items()
        },
        "ccr2_trigger": {
            int(year): float(value)
            for year, value in policy_spec.ccr2_trigger.astype(float).items()
        },
        "ccr2_qty": {
            int(year): float(value)
            for year, value in policy_spec.ccr2_qty.astype(float).items()
        },
        "banking_enabled": bool(getattr(policy_spec, "banking_enabled", True)),
        "control_period_years": getattr(policy_spec, "control_period_years", None),
    }
    return result


def _regions_from_config(run_config: Mapping[str, Any]) -> Sequence[str]:
    regions = run_config.get("regions") if isinstance(run_config, Mapping) else None
    if isinstance(regions, Mapping):
        return [str(key) for key, value in regions.items() if _is_nonzero(value)]
    if isinstance(regions, Sequence) and not isinstance(regions, (str, bytes)):
        return [str(entry) for entry in regions]
    return []


def _is_nonzero(value: Any) -> bool:
    try:
        return bool(float(value))
    except (TypeError, ValueError):
        return True


def _load_forecast_section(
    forecast_manifests: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    if not forecast_manifests:
        return [
            {
                "iso": "synthetic",
                "source": "Synthetic",
                "vintage": None,
                "scenario": "Development Stub",
                "manifest": "Synthetic – Development Stub",
                "zones": [],
                "path": None,
            }
        ]

    manifest_entries: dict[str, Mapping[str, Any]] = {}
    for record in forecast_manifests:
        if not isinstance(record, Mapping):
            continue
        iso_value = str(record.get("iso") or "").strip()
        if not iso_value:
            continue
        scenario_text = str(record.get("scenario") or "").strip()
        manifest_label = str(record.get("manifest") or f"{iso_value}::{scenario_text}")
        zones = record.get("zones") or record.get("regions") or []
        if isinstance(zones, Mapping):
            zone_list = sorted(str(key) for key in zones.keys())
        elif isinstance(zones, Sequence) and not isinstance(zones, (str, bytes, bytearray)):
            zone_list = sorted(str(zone) for zone in zones)
        else:
            zone_list = []
        entry: dict[str, Any] = {
            "iso": iso_value,
            "source": record.get("source"),
            "vintage": record.get("vintage"),
            "scenario": scenario_text or None,
            "manifest": manifest_label,
            "zones": zone_list,
            "path": record.get("path"),
        }
        manifest_entries.setdefault(iso_value, entry)

    return list(manifest_entries.values())


def _emissions_section(outputs: EngineOutputs) -> Mapping[str, Any]:
    emissions_total = {}
    if isinstance(outputs.emissions_total, Mapping):
        emissions_total = {
            int(year): float(value)
            for year, value in outputs.emissions_total.items()
        }
    bank: Mapping[int, float] = {}
    if hasattr(outputs, "annual"):
        annual_df = getattr(outputs, "annual")
        if isinstance(annual_df, pd.DataFrame) and "year" in annual_df.columns:
            bank_series = annual_df.get("bank")
            if bank_series is not None:
                bank_numeric = pd.to_numeric(bank_series, errors="coerce").fillna(0.0)
                year_series = pd.to_numeric(annual_df["year"], errors="coerce")
                bank = {
                    int(year): float(bank_numeric.iloc[idx])
                    for idx, year in enumerate(year_series)
                    if not pd.isna(year)
                }
    return {"total_tons": emissions_total, "bank_tons": bank}


def build_manifest(
    run_config: Mapping[str, Any],
    frames: Frames,
    outputs: EngineOutputs,
    *,
    forecast_manifests: Sequence[Mapping[str, Any]] | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    years = _to_year_list(frames)
    run_section = {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "years": {
            "min": years[0] if years else None,
            "max": years[-1] if years else None,
            "all": years,
        },
        "deep_carbon_pricing_enabled": bool(
            run_config.get("modules", {})
            .get("electricity_dispatch", {})
            .get("deep_carbon_pricing", False)
        ),
        "banking_enabled": bool(
            run_config.get("modules", {})
            .get("carbon_policy", {})
            .get("bank_enabled", True)
        ),
    }

    policy_section = _policy_summary(frames)
    if policy_section and "banking_enabled" in policy_section:
        run_section["banking_enabled"] = bool(policy_section.get("banking_enabled"))
    electricity_section = {
        "regions": list(_regions_from_config(run_config)),
        "transmission": dict(_transmission_summary(frames)),
    }
    load_forecasts_section = _load_forecast_section(forecast_manifests)
    emissions_section = _emissions_section(outputs)
    engine_section = {
        "python_version": platform.python_version(),
        "engine_version": None,
        "git_commit": git_commit,
    }

    manifest = {
        "run": run_section,
        "policy": policy_section,
        "electricity": electricity_section,
        "load_forecasts": load_forecasts_section,
        "emissions": emissions_section,
        "engine": engine_section,
    }
    return manifest


def _markdown_pair(label: str, value: Any) -> str:
    if value is None:
        return f"- **{label}:** _None_"
    if isinstance(value, bool):
        return f"- **{label}:** {'Yes' if value else 'No'}"
    return f"- **{label}:** {value}"


def manifest_to_markdown(manifest: Mapping[str, Any]) -> str:
    lines: list[str] = ["# Granite Ledger – Run Manifest"]

    run_section = manifest.get("run", {})
    if isinstance(run_section, Mapping):
        lines.append("## Run")
        years = run_section.get("years", {})
        if isinstance(years, Mapping):
            span = None
            if years.get("min") is not None and years.get("max") is not None:
                span = f"{years['min']}–{years['max']}"
            lines.append(_markdown_pair("Years", span or years.get("all", [])))
        lines.append(
            _markdown_pair(
                "Deep Carbon Pricing Enabled", run_section.get("deep_carbon_pricing_enabled")
            )
        )
        lines.append(_markdown_pair("Banking Enabled", run_section.get("banking_enabled")))

    policy = manifest.get("policy", {})
    if isinstance(policy, Mapping):
        lines.append("## Policy")
        decline = policy.get("cap_decline_tons")
        if decline is not None:
            lines.append(_markdown_pair("Cap Decline (tons)", f"{decline:,.0f}"))
        percent = policy.get("cap_decline_percent")
        if percent is not None:
            lines.append(_markdown_pair("Cap Decline (%)", f"{percent:.2f}"))
        floor = policy.get("price_floor")
        if isinstance(floor, Mapping):
            lines.append("- **Price Floor:** " + ", ".join(f"{year}: {value}" for year, value in floor.items()))
        lines.append(_markdown_pair("CCR1 Trigger", policy.get("ccr1_trigger")))
        lines.append(_markdown_pair("CCR2 Trigger", policy.get("ccr2_trigger")))

    electricity = manifest.get("electricity", {})
    if isinstance(electricity, Mapping):
        lines.append("## Electricity & Regions")
        regions = electricity.get("regions", [])
        if regions:
            lines.append("- **Regions:** " + ", ".join(regions))
        transmission = electricity.get("transmission", {})
        if isinstance(transmission, Mapping):
            lines.append(
                "- **Transmission:** "
                + ", ".join(f"{key.replace('_', ' ').title()}: {value}" for key, value in transmission.items())
            )

    forecast_section = manifest.get("load_forecasts", [])
    if isinstance(forecast_section, Sequence):
        lines.append("## Load Forecasts")
        for entry in forecast_section:
            if not isinstance(entry, Mapping):
                continue
            iso = entry.get("iso", "ISO")
            manifest_name = entry.get("manifest")
            scenario = entry.get("scenario")
            descriptor = manifest_name or scenario
            lines.append(f"- **{iso}:** {descriptor}")

    emissions = manifest.get("emissions", {})
    if isinstance(emissions, Mapping):
        lines.append("## Emissions & Bank")
        total = emissions.get("total_tons", {})
        if isinstance(total, Mapping) and total:
            lines.append("- **Total Emissions:** " + ", ".join(f"{year}: {value:,.0f}" for year, value in total.items()))
        bank = emissions.get("bank_tons", {})
        if isinstance(bank, Mapping) and bank:
            lines.append("- **Bank:** " + ", ".join(f"{year}: {value:,.0f}" for year, value in bank.items()))

    engine = manifest.get("engine", {})
    if isinstance(engine, Mapping):
        lines.append("## Engine")
        lines.append(_markdown_pair("Python", engine.get("python_version")))
        if engine.get("git_commit"):
            lines.append(_markdown_pair("Git Commit", engine.get("git_commit")))

    return "\n".join(lines).strip() + "\n"
