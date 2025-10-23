import pandas as pd
from pathlib import Path

from gui import helpers as gui_helpers


def test_build_demand_filters_by_year_from_timestamp(monkeypatch):
    data = pd.DataFrame(
        {
            "state": ["NY", "NY"],
            "iso": ["PJM", "PJM"],
            "region_id": ["PJM_NYUP", "PJM_NYUP"],
            "scenario": ["Test", "Test"],
            "timestamp": pd.to_datetime(["2030-01-01", "2031-01-01"]),
            "load_gwh": [1.0, 2.0],
        }
    )

    def fake_load_demand_forecasts(selection, root):  # pragma: no cover - helper stub
        return data.copy()

    monkeypatch.setattr(
        gui_helpers, "load_demand_forecasts_selection", fake_load_demand_forecasts
    )
    monkeypatch.setattr(gui_helpers, "_root", lambda: Path("/tmp"))

    result = gui_helpers.build_demand([2030], {"NY": {"iso": "PJM", "scenario": "Test"}})

    assert list(result["year"]) == [2030]
    assert pd.api.types.is_integer_dtype(result["year"])
    assert result["load_gwh"].tolist() == [1.0]
