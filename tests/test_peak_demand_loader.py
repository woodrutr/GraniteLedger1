from pathlib import Path

from engine.io.peak_demand import load_peak_demand


def test_load_peak_demand_filters_scenario(tmp_path: Path) -> None:
    csv_path = tmp_path / "peak_demand.csv"
    csv_path.write_text(
        "Year,Region_ID,Scenario,Peak_Demand_MW\n"
        "2030,PJM_DOM,base,100\n"
        "2030,PJM_DOM,high,125\n"
        "2031,PJM_DOM,base,110\n"
    )

    frame = load_peak_demand(csv_path, scenario="base")
    assert list(frame.columns) == ["year", "region", "peak_demand_mw", "scenario"]
    assert frame.shape[0] == 2
    assert set(frame["year"]) == {2030, 2031}
    assert frame["peak_demand_mw"].tolist() == [100.0, 110.0]
    assert frame["region"].tolist() == ["PJM_DOM", "PJM_DOM"]


def test_load_peak_demand_returns_empty_when_missing(tmp_path: Path) -> None:
    root = tmp_path / "load_forecasts"
    root.mkdir()

    frame = load_peak_demand(root, scenario="baseline")
    assert frame.empty
    assert list(frame.columns) == ["year", "region", "peak_demand_mw", "scenario"]
