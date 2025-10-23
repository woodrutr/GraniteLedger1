from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.io import load_forecasts_strict as strict
from regions.registry import REGIONS
import gui.app as app


class _DummyContainer:
    def __init__(self) -> None:
        self.selectbox_calls: list[dict[str, object]] = []

    def subheader(self, *_args: object, **_kwargs: object) -> None:
        return None

    def info(self, *_args: object, **_kwargs: object) -> None:
        return None

    def selectbox(
        self,
        label: str,
        *,
        options: list[str],
        index: int,
        key: str,
        help: str,
    ) -> str:
        self.selectbox_calls.append(
            {
                "label": label,
                "options": list(options),
                "index": index,
                "key": key,
                "help": help,
            }
        )
        return options[index]


def _write_zone(path: Path, rows: list[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Year,Load_GWh\n")
        for year, load in rows:
            handle.write(f"{year},{load}\n")


def test_strict_loader_resolves_regions(tmp_path: Path) -> None:
    nyiso_path = tmp_path / "NYISO" / "NYISO_GoldBook_2025_Baseline"
    pjm_path = tmp_path / "PJM" / "PJM_2024_Baseline"
    _write_zone(nyiso_path / "NYISO_J.csv", [(2025, 100.0)])
    _write_zone(pjm_path / "PJM_ATSI.csv", [(2024, 200.0)])

    cache_path = tmp_path / "cache.parquet"
    frame = strict.build_table(base_path=tmp_path, cache_path=cache_path, use_cache=False)

    assert set(frame["region_id"].astype(str)) == {"NYISO_J", "PJM_ATSI"}
    assert set(frame["region_id"].astype(str)).issubset({str(key) for key in REGIONS})

def test_missing_region_raises_clear_error(tmp_path: Path) -> None:
    bad_zone = tmp_path / "NYISO" / "NYISO_GoldBook_2025_Baseline" / "Unknown.csv"
    _write_zone(bad_zone, [(2025, 10.0)])

    with pytest.raises(strict.ValidationError) as excinfo:
        strict.build_table(base_path=tmp_path, use_cache=False)

    assert "Zone UNKNOWN missing from REGION registry" in str(excinfo.value)


def test_gui_dropdown_uses_folder_names_and_state(monkeypatch: pytest.MonkeyPatch) -> None:
    session_state: dict[str, object] = {"forecast_selections": {"nyiso": "Gold Book Low"}}
    dummy_st = SimpleNamespace(session_state=session_state)
    monkeypatch.setattr(app, "st", dummy_st)

    frame = pd.DataFrame(
        {
            "iso": pd.Categorical(["NYISO", "NYISO"]),
            "scenario": pd.Categorical([
                "Gold Book High",
                "Gold Book Low",
            ]),
            "zone": pd.Categorical(["ZoneA", "ZoneA"]),
            "region_id": pd.Categorical(["NYISO_A", "NYISO_A"]),
            "Year": [2025, 2026],
            "Load_GWh": [1.0, 2.0],
            "iso_norm": ["nyiso", "nyiso"],
            "scenario_norm": ["gold_book_high", "gold_book_low"],
            "region_norm": ["nyiso_a", "nyiso_a"],
        }
    )

    monkeypatch.setattr(app, "_resolve_forecast_base_path", lambda: "root")
    monkeypatch.setattr(app, "_cached_forecast_frame", lambda base_path: frame)

    selection_calls: dict[str, object] = {}

    def fake_select(selection, *, base_path=None, frame=None):  # type: ignore[no-untyped-def]
        selection_calls["selection"] = dict(selection)
        selection_calls["base_path"] = base_path
        selection_calls["frame"] = frame
        return []

    monkeypatch.setattr(app, "select_forecast_bundles", fake_select)
    monkeypatch.setattr(app, "_regions_available_zones", lambda *_args, **_kwargs: [])

    container = _DummyContainer()
    run_config: dict[str, object] = {"modules": {}}

    settings = app._render_demand_module_section(
        container,
        run_config,
        regions=[],
        years=None,
    )

    assert container.selectbox_calls, "selectbox should be invoked"
    assert container.selectbox_calls[0]["options"] == ["Gold Book High", "Gold Book Low"]
    assert container.selectbox_calls[0]["index"] == 1

    assert settings.load_forecasts == {"NYISO": "Gold Book Low"}
    assert session_state["forecast_selections"] == {"NYISO": "Gold Book Low"}
    assert selection_calls["selection"] == {"NYISO": "Gold Book Low"}
    assert selection_calls["frame"] is frame
