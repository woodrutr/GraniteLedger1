from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import regions.load_forecasts as regions_forecasts
import gui.app as app


class _DummyContainer:
    def __init__(self) -> None:
        self.selectbox_calls: list[dict[str, object]] = []

    def subheader(self, *_args: object, **_kwargs: object) -> None:
        return None

    def info(self, *_args: object, **_kwargs: object) -> None:
        return None

    def markdown(self, *_args: object, **_kwargs: object) -> None:
        return None

    def file_uploader(self, *_args: object, **_kwargs: object):
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


def test_regions_load_forecasts_module_round_trip(tmp_path: Path) -> None:
    csv_path = tmp_path / "load_forecasts.csv"
    csv_path.write_text(
        """region_id,scenario_name,year,load_gwh
ISO-NE_CT,Example_2025_Baseline,2025,10
ISO-NE_CT,Example_2025_Baseline,2026,20
""",
        encoding="utf-8",
    )

    frame = regions_forecasts.load_forecasts(csv_path)

    assert list(frame.columns) == [
        "iso",
        "zone",
        "region_id",
        "scenario",
        "year",
        "load_mwh",
    ]
    assert frame["iso"].tolist() == ["ISO-NE", "ISO-NE"]
    assert frame["zone"].tolist() == ["CT", "CT"]
    assert frame["scenario"].unique().tolist() == ["example_2025_baseline"]
    assert frame["load_mwh"].tolist() == [10_000.0, 20_000.0]


def test_regions_available_iso_scenarios(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"iso": ["iso_a"], "scenario": ["Baseline"], "zone": ["Z1"]})
    captured: dict[str, object] = {}

    def fake_load(base_path=None):  # type: ignore[no-untyped-def]
        captured["base_path"] = base_path
        return frame

    def fake_index(df):  # type: ignore[no-untyped-def]
        captured["frame"] = df
        return {"iso_a": ["Baseline"]}

    monkeypatch.setattr(app, "_regions_load_forecasts_frame", fake_load)
    monkeypatch.setattr(app, "_regions_scenario_index", fake_index)

    result = app._regions_available_iso_scenarios(base_path="/tmp/root")

    assert result == {"iso_a": ["Baseline"]}
    assert captured["base_path"] == "/tmp/root"
    assert captured["frame"] is frame


def test_regions_available_zones(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame({"iso": ["iso_a"], "scenario": ["Baseline"], "zone": ["Z1"]})
    captured: dict[str, object] = {}

    def fake_load(base_path=None):  # type: ignore[no-untyped-def]
        captured["base_path"] = base_path
        return frame

    def fake_zones(df, iso, scenario):  # type: ignore[no-untyped-def]
        captured["frame"] = df
        captured["iso"] = iso
        captured["scenario"] = scenario
        return ["Z1"]

    monkeypatch.setattr(app, "_regions_load_forecasts_frame", fake_load)
    monkeypatch.setattr(app, "_regions_zones_for", fake_zones)

    result = app._regions_available_zones("/tmp/root", "iso_a", "Baseline")

    assert result == ["Z1"]
    assert captured == {
        "base_path": "/tmp/root",
        "frame": frame,
        "iso": "iso_a",
        "scenario": "Baseline",
    }


def test_gui_load_forecast_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "st", None)

    frame = pd.DataFrame(
        {
            "iso": pd.Categorical(["iso_a", "iso_a"]),
            "scenario": pd.Categorical(["Baseline", "Baseline"]),
            "zone": pd.Categorical(["Zone1", "Zone2"]),
            "region_id": pd.Categorical(["ISO_A_ZONE1", "ISO_A_ZONE2"]),
            "Year": [2025, 2026],
            "Load_GWh": [1.0, 2.0],
            "iso_norm": ["iso_a", "iso_a"],
            "scenario_norm": ["baseline", "baseline"],
            "region_norm": ["iso_a_zone1", "iso_a_zone2"],
        }
    )

    monkeypatch.setattr(app, "_resolve_forecast_base_path", lambda: "root")
    monkeypatch.setattr(app, "_cached_forecast_frame", lambda base_path: frame)
    zone_calls: list[tuple[str, str, str]] = []

    def fake_available_zones(base_path, iso, scenario):  # type: ignore[no-untyped-def]
        zone_calls.append((base_path, iso, scenario))
        return ["Zone1", "Zone2"]

    monkeypatch.setattr(app, "_regions_available_zones", fake_available_zones)

    recorded: dict[str, object] = {}

    def fake_select(
        selection, *, base_path=None, frame=None
    ):  # type: ignore[no-untyped-def]
        recorded["selection"] = dict(selection)
        recorded["base_path"] = base_path
        recorded["frame"] = frame
        return ["bundle"]

    monkeypatch.setattr(app, "select_forecast_bundles", fake_select)

    run_config: dict[str, object] = {}
    container = SimpleNamespace()

    settings = app._render_demand_module_section(
        container,
        run_config,
        regions=[],
        years=None,
    )

    assert settings.load_forecasts == {"iso_a": "Baseline"}
    assert recorded == {
        "selection": {"iso_a": "Baseline"},
        "base_path": Path("root"),
        "frame": frame,
    }
    assert zone_calls == []
    assert "demand" in run_config.get("modules", {})


def test_select_bundles_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "st", None)

    frame = pd.DataFrame(
        {
            "iso": ["ISO_A", "ISO_A", "ISO_B", "ISO_B"],
            "scenario": ["Baseline", "High", "Baseline", "High"],
            "zone": ["Z1", "Z2", "Z3", "Z4"],
        }
    )

    monkeypatch.setattr(app, "_resolve_forecast_base_path", lambda: "root")
    monkeypatch.setattr(app, "_cached_forecast_frame", lambda base_path: frame)
    select_calls: list[dict[str, object]] = []

    def fake_select(
        selection, *, base_path=None, frame=None
    ):  # type: ignore[no-untyped-def]
        select_calls.append(
            {
                "selection": dict(selection),
                "base_path": base_path,
                "frame": frame,
            }
        )
        return ["bundle"]

    monkeypatch.setattr(app, "select_forecast_bundles", fake_select)
    monkeypatch.setattr(app, "_regions_available_zones", lambda *args, **kwargs: ["Z1"])  # type: ignore[arg-type]

    settings = app._render_demand_module_section(
        SimpleNamespace(),
        {},
        regions=[],
        years=None,
    )

    assert settings.load_forecasts == {"ISO_A": "Baseline", "ISO_B": "Baseline"}
    assert select_calls == [
        {
            "selection": {"ISO_A": "Baseline", "ISO_B": "Baseline"},
            "base_path": Path("root"),
            "frame": frame,
        }
    ]


def test_gui_load_forecast_dropdown_options(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_st = SimpleNamespace(session_state={})
    monkeypatch.setattr(app, "st", dummy_st)

    frame = pd.DataFrame(
        {
            "iso": pd.Categorical(
                [
                    "NYISO",
                    "NYISO",
                    "PJM",
                    "PJM",
                ]
            ),
            "scenario": pd.Categorical(
                [
                    "NYISO Gold Book 2025 – High",
                    "NYISO Gold Book 2025 – Low",
                    "PJM 2025 Base",
                    "PJM 2025 High",
                ]
            ),
            "zone": pd.Categorical([
                "ZoneA",
                "ZoneB",
                "Zone1",
                "Zone2",
            ]),
        }
    )

    monkeypatch.setattr(app, "_resolve_forecast_base_path", lambda: "root")
    monkeypatch.setattr(app, "_cached_forecast_frame", lambda base_path: frame)
    monkeypatch.setattr(app, "_regions_available_zones", lambda *_args, **_kwargs: ["ZoneA"])  # type: ignore[arg-type]

    recorded: dict[str, object] = {}

    def fake_select(selection, *, base_path=None, frame=None):  # type: ignore[no-untyped-def]
        recorded["selection"] = dict(selection)
        recorded["base_path"] = base_path
        recorded["frame"] = frame
        return []

    monkeypatch.setattr(app, "select_forecast_bundles", fake_select)

    container = _DummyContainer()
    settings = app._render_demand_module_section(
        container,
        {},
        regions=[],
        years=None,
    )

    option_sets = {tuple(call["options"]) for call in container.selectbox_calls}
    assert option_sets == {
        ("NYISO Gold Book 2025 – High", "NYISO Gold Book 2025 – Low"),
        ("PJM 2025 Base", "PJM 2025 High"),
    }
    state_labels = {
        call["label"].split(" ", 1)[0]
        for call in container.selectbox_calls
        if call["label"].endswith("load forecast scenario")
    }
    assert settings.load_forecasts == recorded["selection"]
    assert set(settings.load_forecasts) == state_labels
    assert recorded["base_path"] == Path("root")
    recorded_frame = recorded["frame"]
    if hasattr(recorded_frame, "_frame"):
        recorded_frame = recorded_frame._frame
    assert isinstance(recorded_frame, pd.DataFrame)
    pd.testing.assert_frame_equal(recorded_frame, frame)
    assert dummy_st.session_state["forecast_selections"] == settings.load_forecasts
    assert dummy_st.session_state["forecast_iso_labels"] == {
        state: ("NYISO" if state == "NY" else "PJM")
        for state in state_labels
    }
    assert settings.load_forecasts["NY"] == "NYISO::NYISO Gold Book 2025 – High"
    for state in sorted(state_labels - {"NY"}):
        assert settings.load_forecasts[state] == "PJM::PJM 2025 Base"


def test_zones_dropdown_stability(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {
            "iso": ["ISO_A", "ISO_A", "ISO_A"],
            "scenario": ["Baseline", "Baseline", "High"],
            "zone": ["Z1", "Z2", "Z3"],
        }
    )

    def explode(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    monkeypatch.setattr(app, "_regions_available_zones", explode)

    summary = app._summarize_forecasts(frame, base_path="root")

    assert summary == [
        {"iso": "ISO_A", "scenario": "Baseline", "zones": ["Z1", "Z2"], "zone_count": 2},
        {"iso": "ISO_A", "scenario": "High", "zones": ["Z3"], "zone_count": 1},
    ]
