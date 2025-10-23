"""Fixtures supporting stress and sensitivity audits for the engine."""
from __future__ import annotations

from typing import Iterable, Sequence

import pytest

pd = pytest.importorskip("pandas")

from tests.fixtures.dispatch_single_minimal import baseline_frames

YEARS: Sequence[int] = (2025, 2026, 2027)


def policy_frame(
    *,
    floor: float = 0.0,
    cap_scale: float = 1.0,
    bank0: float = 200_000.0,
    carry_pct: float = 1.0,
    annual_surrender_frac: float = 0.5,
    policy_enabled: bool = True,
) -> pd.DataFrame:
    """Create a three-year policy frame with configurable scaling."""

    caps_base = [800_000.0, 780_000.0, 760_000.0]
    caps = [cap * cap_scale for cap in caps_base]
    ccr1_qty = 120_000.0 * cap_scale
    ccr2_qty = 180_000.0 * cap_scale

    records = []
    for idx, year in enumerate(YEARS):
        records.append(
            {
                "year": int(year),
                "cap_tons": float(caps[idx]),
                "floor_dollars": float(floor),
                "ccr1_trigger": 10.0,
                "ccr1_qty": float(ccr1_qty),
                "ccr2_trigger": 18.0,
                "ccr2_qty": float(ccr2_qty),
                "cp_id": "CP1",
                "full_compliance": year == YEARS[-1],
                "bank0": float(bank0),
                "annual_surrender_frac": float(annual_surrender_frac),
                "carry_pct": float(carry_pct),
                "policy_enabled": bool(policy_enabled),
                "resolution": "annual",
            }
        )
    return pd.DataFrame(records)


def audit_frames(
    loads: Iterable[float] | None = None,
    *,
    floor: float = 0.0,
    cap_scale: float = 1.0,
    bank0: float = 200_000.0,
    carry_pct: float = 1.0,
    annual_surrender_frac: float = 0.5,
    policy_enabled: bool = True,
    gas_price_multiplier: float = 1.0,
) -> "Frames":
    """Return Frames configured for engine audit scenarios."""

    from io_loader import Frames

    if loads is None:
        loads = [1_000_000.0, 950_000.0, 900_000.0]

    load_list = list(loads)
    if not load_list:
        raise ValueError('loads must contain at least one value')

    base = baseline_frames(year=YEARS[0], load_mwh=float(load_list[0]))
    demand_records = [
        {"year": int(year), "region": "default", "demand_mwh": float(load)}
        for year, load in zip(YEARS, load_list)
    ]
    demand = pd.DataFrame(demand_records)

    frames = base.with_frame("demand", demand)
    units = frames.units()
    gas_mask = units["fuel"].astype(str).str.lower() == "gas"
    if gas_mask.any():
        units.loc[gas_mask, "fuel_price_per_mmbtu"] = (
            pd.to_numeric(units.loc[gas_mask, "fuel_price_per_mmbtu"], errors="coerce")
            .fillna(0.0)
            * float(gas_price_multiplier)
        )
    frames = frames.with_frame("units", units)
    frames = frames.with_frame(
        "policy",
        policy_frame(
            floor=floor,
            cap_scale=cap_scale,
            bank0=bank0,
            carry_pct=carry_pct,
            annual_surrender_frac=annual_surrender_frac,
            policy_enabled=policy_enabled,
        ),
    )
    return Frames.coerce(frames)


__all__ = ["YEARS", "audit_frames", "policy_frame"]
