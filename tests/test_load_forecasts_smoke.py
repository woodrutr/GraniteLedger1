from pathlib import Path

from engine.data_loaders.load_forecasts import load_demand_forecasts_selection


def test_demand_forecasts_have_multiple_regions() -> None:
    base_path = Path(__file__).resolve().parents[1] / "input" / "electricity" / "load_forecasts"
    demand_df = load_demand_forecasts_selection(base_path=base_path)
    assert not demand_df.empty, "Demand dataframe should not be empty"

    totals = demand_df.groupby("region_id")["demand_mwh"].sum().astype(float)
    positive_regions = [region for region, value in totals.items() if value > 0.0]
    assert len(positive_regions) >= 3, "Expected positive demand in at least three regions"
