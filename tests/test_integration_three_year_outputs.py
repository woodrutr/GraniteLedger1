from __future__ import annotations
import importlib
import pytest
pd = pytest.importorskip("pandas")
engine_run_loop = importlib.import_module("engine.run_loop")
run_end_to_end_from_frames = engine_run_loop.run_end_to_end_from_frames
from tests.fixtures.dispatch_single_minimal import DEFAULT_YEARS, three_year_frames


def test_three_year_run_produces_non_empty_outputs() -> None:
    years = list(DEFAULT_YEARS)
    frames = three_year_frames(years=years)
    outputs = run_end_to_end_from_frames(
        frames,
        years=years,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    assert not outputs.annual.empty
    assert not outputs.emissions_by_region.empty
    assert not outputs.price_by_region.empty
    assert list(outputs.flows.columns) == ["year", "from_region", "to_region", "flow_mwh"]
    assert isinstance(outputs.limiting_factors, list)
    assert all(isinstance(item, str) for item in outputs.limiting_factors)
