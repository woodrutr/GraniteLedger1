import importlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pd = pytest.importorskip("pandas")

from engine import run_loop
from tests.carbon_price_utils import assert_aliases_match_canonical, with_carbon_vector_columns
from tests.fixtures import dispatch_single_minimal as fixtures


def _schedule(years, base, escalator_pct):
    values = {}
    price = float(base)
    for year in years:
        values[year] = price
        price *= 1.0 + float(escalator_pct) / 100.0
    return values


@pytest.mark.parametrize("deep", [False, True])
def test_exogenous_price_schedule_propagates(deep):
    years = [2025, 2026, 2027]
    frames = fixtures.three_year_frames(years=years)
    sched = _schedule(years, 45.0, 4.0)
    if hasattr(frames, "_meta"):
        frames._meta.setdefault(
            "carbon_price_schedule", {str(key): value for key, value in sched.items()}
        )
        frames._meta["carbon_price_default"] = 45.0
    setattr(frames, "deep_carbon_pricing_enabled", bool(deep))

    outputs = run_loop.run_end_to_end_from_frames(
        frames,
        years=years,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
        carbon_price_schedule=sched,
        deep_carbon_pricing=bool(deep),
    )
    df = with_carbon_vector_columns(outputs.annual)
    assert_aliases_match_canonical(df)
    df = df.set_index("year")

    assert {"cp_all", "cp_effective"}.issubset(df.columns)
    for year in years:
        if "cp_exempt" in df.columns:
            assert df.loc[year, "cp_exempt"] == pytest.approx(sched[year], rel=1e-6)
        allowance_value = df.loc[year, "cp_all"]
        effective_value = df.loc[year, "cp_effective"]
        if deep:
            assert effective_value == pytest.approx(allowance_value + sched[year], rel=1e-6)
        else:
            assert effective_value == pytest.approx(max(allowance_value, sched[year]), rel=1e-6)



def test_invalid_carbon_price_schedule_raises_value_error():
    years = [2025, 2026]
    frames = fixtures.three_year_frames(years=years, loads=(1_000_000.0, 950_000.0))

    with pytest.raises(ValueError, match="Carbon price schedule contains no valid year/price pairs"):
        run_loop.run_end_to_end_from_frames(
            frames,
            years=years,
            carbon_price_schedule={"FY25": 45.0},
        )
