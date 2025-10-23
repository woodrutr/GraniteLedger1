"""Fixtures for annual allowance market tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pytest

pd = pytest.importorskip("pandas")

from policy.allowance_annual import RGGIPolicyAnnual


@dataclass
class LinearDispatch:
    """Simple callable dispatch stub with linear response to price."""

    base: Mapping[int, float]
    slope: Mapping[int, float] | float

    def __call__(self, year: int, price: float) -> float:
        slope = self.slope if isinstance(self.slope, (int, float)) else self.slope.get(year, 0.0)
        emission = self.base.get(year, 0.0) - float(slope) * price
        return max(0.0, emission)


def policy_three_year() -> RGGIPolicyAnnual:
    """Baseline three-year policy spanning a single control period."""

    years = [2025, 2026, 2027]
    cap = pd.Series({2025: 100.0, 2026: 90.0, 2027: 250.0})
    floor = pd.Series({year: 4.0 for year in years})
    ccr1_trigger = pd.Series({year: 7.0 for year in years})
    ccr1_qty = pd.Series({2025: 30.0, 2026: 30.0, 2027: 30.0})
    ccr2_trigger = pd.Series({year: 13.0 for year in years})
    ccr2_qty = pd.Series({2025: 60.0, 2026: 60.0, 2027: 60.0})
    cp_id = pd.Series({year: "CP1" for year in years})

    return RGGIPolicyAnnual(
        cap=cap,
        floor=floor,
        ccr1_trigger=ccr1_trigger,
        ccr1_qty=ccr1_qty,
        ccr2_trigger=ccr2_trigger,
        ccr2_qty=ccr2_qty,
        cp_id=cp_id,
        bank0=10.0,
        full_compliance_years={2027},
        annual_surrender_frac=0.5,
        carry_pct=1.0,
    )


def policy_for_shortage() -> RGGIPolicyAnnual:
    """Policy with limited allowances to exercise shortage behaviour."""

    years = [2025]
    cap = pd.Series({2025: 100.0})
    floor = pd.Series({2025: 4.0})
    ccr1_trigger = pd.Series({2025: 7.0})
    ccr1_qty = pd.Series({2025: 30.0})
    ccr2_trigger = pd.Series({2025: 13.0})
    ccr2_qty = pd.Series({2025: 60.0})
    cp_id = pd.Series({2025: "CP-short"})

    return RGGIPolicyAnnual(
        cap=cap,
        floor=floor,
        ccr1_trigger=ccr1_trigger,
        ccr1_qty=ccr1_qty,
        ccr2_trigger=ccr2_trigger,
        ccr2_qty=ccr2_qty,
        cp_id=cp_id,
        bank0=10.0,
        full_compliance_years={2025},
        annual_surrender_frac=0.5,
        carry_pct=1.0,
    )


def policy_frame_three_year() -> pd.DataFrame:
    """Return a DataFrame compatible with :class:`granite_io.frames_api.Frames`."""

    years = [2025, 2026, 2027]
    records = []
    for year in years:
        records.append(
            {
                "year": year,
                "cap_tons": 100.0 if year == 2025 else (90.0 if year == 2026 else 250.0),
                "floor_dollars": 4.0,
                "ccr1_trigger": 7.0,
                "ccr1_qty": 30.0,
                "ccr2_trigger": 13.0,
                "ccr2_qty": 60.0,
                "cp_id": "CP1",
                "full_compliance": year == 2027,
                "bank0": 10.0,
                "annual_surrender_frac": 0.5,
                "carry_pct": 1.0,
            }
        )
    return pd.DataFrame(records)
