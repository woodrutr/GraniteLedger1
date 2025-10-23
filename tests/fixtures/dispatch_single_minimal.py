"""Fixtures supporting single-region dispatch tests."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from collections.abc import Iterable

from io_loader import Frames
from regions.registry import REGIONS

HOURS_PER_YEAR = 8760.0
DEFAULT_YEARS: tuple[int, ...] = (2025, 2026, 2027)
DEFAULT_REGION_ID: str = next(iter(REGIONS))


def baseline_units() -> pd.DataFrame:
    """Return a deterministic three-unit system used across the tests."""

    data = [
        {
            "unit_id": "wind-1",
            "unique_id": "wind-1",
            "fuel": "wind",
            "region": DEFAULT_REGION_ID,
            "cap_mw": 50.0,
            "availability": 0.5,
            "hr_mmbtu_per_mwh": 0.0,
            "vom_per_mwh": 0.0,
            "fuel_price_per_mmbtu": 0.0,
            "ef_ton_per_mwh": 0.0,
            "carbon_cost_per_mwh": 0.0,
            "co2_short_ton_per_mwh": 0.0,
        },
        {
            "unit_id": "coal-1",
            "unique_id": "coal-1",
            "fuel": "coal",
            "region": DEFAULT_REGION_ID,
            "cap_mw": 80.0,
            "availability": 0.9,
            "hr_mmbtu_per_mwh": 9.0,
            "vom_per_mwh": 1.5,
            "fuel_price_per_mmbtu": 2.0,
            "ef_ton_per_mwh": 1.0,
            "carbon_cost_per_mwh": 0.0,
            "co2_short_ton_per_mwh": 1.0,
        },
        {
            "unit_id": "gas-1",
            "unique_id": "gas-1",
            "fuel": "gas",
            "region": DEFAULT_REGION_ID,
            "cap_mw": 120.0,
            "availability": 0.9,
            "hr_mmbtu_per_mwh": 7.0,
            "vom_per_mwh": 2.0,
            "fuel_price_per_mmbtu": 2.5,
            "ef_ton_per_mwh": 0.5,
            "carbon_cost_per_mwh": 0.0,
            "co2_short_ton_per_mwh": 0.5,
        },
    ]
    return pd.DataFrame(data)


def baseline_frames(year: int = 2030, load_mwh: float = 1_000_000.0) -> Frames:
    """Minimal frames for single-region tests."""

    demand = pd.DataFrame(
        [{"year": year, "region": DEFAULT_REGION_ID, "demand_mwh": load_mwh}]
    )
    fuels = pd.DataFrame(
        [
            {"fuel": "wind", "covered": True},
            {"fuel": "coal", "covered": True},
            {"fuel": "gas", "covered": True},
        ]
    )
    transmission = pd.DataFrame(columns=["from_region", "to_region", "limit_mw"])

    return Frames(
        {
            "units": baseline_units(),
            "demand": demand,
            "fuels": fuels,
            "transmission": transmission,
        }
    )


def infeasible_frames(year: int = 2030) -> Frames:
    """Frames with load exceeding the total available generation."""

    base = baseline_frames(year=year)
    units = base.units()
    total_cap = float((units["cap_mw"] * units["availability"] * HOURS_PER_YEAR).sum())
    demand = base.demand()
    demand.loc[demand["year"] == year, "demand_mwh"] = total_cap + 10_000.0
    return base.with_frame("demand", demand)


def policy_frame(
    floor: float = 0.0,
    cap_scale: float = 1.0,
    *,
    years: Iterable[int] = DEFAULT_YEARS,
    carry_pct: float = 1.0,
    annual_surrender_frac: float = 0.5,
    policy_enabled: bool = True,
) -> pd.DataFrame:
    """Return a simple multi-year carbon policy configuration for integration tests."""

    year_sequence = tuple(int(year) for year in years)
    caps_base = [800_000.0 - 20_000.0 * idx for idx, _ in enumerate(year_sequence)]
    caps = [cap * cap_scale for cap in caps_base]
    ccr1_qty = 120_000.0 * cap_scale
    ccr2_qty = 180_000.0 * cap_scale
    bank0 = 200_000.0 * cap_scale

    records = []
    for idx, year in enumerate(year_sequence):
        records.append(
            {
                "year": year,
                "cap_tons": caps[idx],
                "floor_dollars": float(floor),
                "ccr1_trigger": 10.0,
                "ccr1_qty": ccr1_qty,
                "ccr2_trigger": 18.0,
                "ccr2_qty": ccr2_qty,
                "cp_id": "CP1",
                "full_compliance": year == year_sequence[-1],
                "bank0": bank0,
                "annual_surrender_frac": float(annual_surrender_frac),
                "carry_pct": float(carry_pct),
                "policy_enabled": bool(policy_enabled),
                "resolution": "annual",
            }
        )
    return pd.DataFrame(records)


def three_year_frames(
    loads: Iterable[float] | None = None,
    *,
    years: Iterable[int] = DEFAULT_YEARS,
    floor: float = 0.0,
    cap_scale: float = 1.0,
    carry_pct: float = 1.0,
    annual_surrender_frac: float = 0.5,
    policy_enabled: bool = True,
) -> Frames:
    """Build frames with demand and policy data for a multi-year integration run."""

    year_sequence = tuple(int(year) for year in years)
    if not year_sequence:
        raise ValueError("at least one model year must be provided")
    if loads is None:
        loads = (1_000_000.0, 950_000.0, 900_000.0)
    load_sequence = tuple(float(load) for load in loads)
    if len(load_sequence) != len(year_sequence):
        raise ValueError("loads must align with the supplied years")

    frames = baseline_frames(year=year_sequence[0], load_mwh=load_sequence[0])
    demand = pd.DataFrame(
        [
            {"year": year, "region": DEFAULT_REGION_ID, "demand_mwh": load}
            for year, load in zip(year_sequence, load_sequence, strict=False)
        ]
    )
    frames = frames.with_frame("demand", demand)

    units = frames.units()
    units.loc[units["fuel"] == "gas", "cap_mw"] = 200.0
    frames = frames.with_frame("units", units)

    frames = frames.with_frame(
        "policy",
        policy_frame(
            floor=floor,
            cap_scale=cap_scale,
            years=year_sequence,
            carry_pct=carry_pct,
            annual_surrender_frac=annual_surrender_frac,
            policy_enabled=policy_enabled,
        ),
    )
    return frames


def expansion_options() -> pd.DataFrame:
    """Return candidate capacity build options for capacity expansion tests."""

    return pd.DataFrame(
        [
            {
                "unit_id": "solar",  # zero-emission technology
                "unique_id": "solar",
                "fuel": "solar",
                "region": DEFAULT_REGION_ID,
                "cap_mw": 60.0,
                "availability": 0.4,
                "hr_mmbtu_per_mwh": 0.0,
                "vom_per_mwh": 2.0,
                "fuel_price_per_mmbtu": 0.0,
                "ef_ton_per_mwh": 0.0,
                "carbon_cost_per_mwh": 0.0,
                "co2_short_ton_per_mwh": 0.0,
                "capex_per_mw": 1_500_000.0,
                "fixed_om_per_mw": 30_000.0,
                "lifetime_years": 25,
                "max_builds": 3,
            },
            {
                "unit_id": "fast_gas",  # flexible build used to resolve shortages
                "unique_id": "fast_gas",
                "fuel": "gas",
                "region": DEFAULT_REGION_ID,
                "cap_mw": 100.0,
                "availability": 0.9,
                "hr_mmbtu_per_mwh": 7.0,
                "vom_per_mwh": 5.0,
                "fuel_price_per_mmbtu": 3.0,
                "ef_ton_per_mwh": 0.6,
                "carbon_cost_per_mwh": 0.0,
                "co2_short_ton_per_mwh": 0.6,
                "capex_per_mw": 900_000.0,
                "fixed_om_per_mw": 20_000.0,
                "lifetime_years": 20,
                "max_builds": 2,
            },
        ]
    )


def frames_with_expansion(year: int = 2030, load_mwh: float = 1_000_000.0) -> Frames:
    """Return frames augmented with expansion candidates."""

    base = baseline_frames(year=year, load_mwh=load_mwh)
    return base.with_frame("expansion", expansion_options())


def all_region_frames(
    year: int = 2025,
    demand_mwh: float = 250_000.0,
    *,
    availability: float = 0.9,
) -> Frames:
    """Return frames covering every registered region with simple generators."""

    demand_records: list[dict[str, object]] = []
    unit_records: list[dict[str, object]] = []
    for idx, region_id in enumerate(REGIONS.keys()):
        demand_records.append(
            {"year": int(year), "region": region_id, "demand_mwh": float(demand_mwh)}
        )
        cap_mw = float(demand_mwh) / (HOURS_PER_YEAR * availability)
        unit_records.append(
            {
                "unit_id": f"gas-{idx}",
                "unique_id": f"gas-{idx}",
                "fuel": "gas",
                "region": region_id,
                "cap_mw": cap_mw,
                "availability": availability,
                "hr_mmbtu_per_mwh": 7.0,
                "vom_per_mwh": 2.0,
                "fuel_price_per_mmbtu": 2.5,
                "ef_ton_per_mwh": 0.5,
                "carbon_cost_per_mwh": 0.0,
                "co2_short_ton_per_mwh": 0.5,
            }
        )

    demand = pd.DataFrame(demand_records)
    units = pd.DataFrame(unit_records)
    fuels = pd.DataFrame(
        [
            {"fuel": "gas", "covered": True},
        ]
    )
    transmission = pd.DataFrame(columns=["from_region", "to_region", "limit_mw"])

    policy = policy_frame(years=[year])

    return Frames(
        {
            "units": units,
            "demand": demand,
            "fuels": fuels,
            "transmission": transmission,
            "policy": policy,
        }
    )

