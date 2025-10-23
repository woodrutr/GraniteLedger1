from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pandas.api.types import CategoricalDtype
import pytest

from engine.io import load_forecasts_strict as strict
from engine.data_loaders import load_forecasts
import gui.app as app


def _write_csv(path: Path, rows: list[tuple[int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("Year,Load_GWh\n")
        for year, load in rows:
            handle.write(f"{year},{load}\n")


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
            {"label": label, "options": list(options), "index": index, "key": key, "help": help}
        )
        return options[index]


def test_build_table_returns_expected_structure(tmp_path: Path) -> None:
    csv_path = tmp_path / "NyISO" / "GoldBook_2025_Baseline" / "NYISO_A.csv"
    _write_csv(csv_path, [(2025, 123.4)])

    frame = strict.build_table(base_path=tmp_path, cache_path=tmp_path / "cache.parquet")

    assert list(frame.columns) == [
        "iso",
        "scenario",
        "zone",
        "region_id",
        "Year",
        "Load_GWh",
        "iso_norm",
        "scenario_norm",
        "region_norm",
    ]

    assert frame.loc[0, "iso"] == "NyISO"
    assert frame.loc[0, "scenario"] == "GoldBook_2025_Baseline"
    assert frame.loc[0, "zone"] == "NYISO_A"
    assert frame.loc[0, "Year"] == 2025
    assert frame.loc[0, "Load_GWh"] == pytest.approx(123.4)
    assert frame.loc[0, "region_id"] == "NYISO_A"
    assert frame.loc[0, "iso_norm"] == "nyiso"
    assert frame.loc[0, "scenario_norm"] == "goldbook_2025_baseline"
    assert frame.loc[0, "region_norm"] == "nyiso_a"

    for column in ("iso", "scenario", "zone", "region_id"):
        assert isinstance(frame[column].dtype, CategoricalDtype)


def test_validation_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "nyiso" / "baseline" / "NYISO_A.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Year\n2025\n", encoding="utf-8")

    with pytest.raises(strict.ValidationError) as excinfo:
        strict.build_table(base_path=tmp_path, cache_path=tmp_path / "cache.parquet", use_cache=False)

    err = excinfo.value
    assert err.file == path
    assert err.row == 1
    assert err.column is None
    assert "expected columns" in err.reason


def test_validation_bad_value(tmp_path: Path) -> None:
    path = tmp_path / "nyiso" / "baseline" / "NYISO_A.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Year,Load_GWh\n2025,not_a_number\n", encoding="utf-8")

    with pytest.raises(strict.ValidationError) as excinfo:
        strict.build_table(base_path=tmp_path, cache_path=tmp_path / "cache.parquet", use_cache=False)

    err = excinfo.value
    assert err.row == 2
    assert err.column == "Load_GWh"


def test_validation_extra_column(tmp_path: Path) -> None:
    path = tmp_path / "nyiso" / "baseline" / "NYISO_A.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Year,Load_GWh,Extra\n2025,1,1\n", encoding="utf-8")

    with pytest.raises(strict.ValidationError) as excinfo:
        strict.build_table(base_path=tmp_path, cache_path=tmp_path / "cache.parquet", use_cache=False)

    err = excinfo.value
    assert err.row == 1
    assert err.column is None
    assert "expected columns" in err.reason


def test_validation_duplicate_year(tmp_path: Path) -> None:
    csv_path = tmp_path / "nyiso" / "baseline" / "NYISO_A.csv"
    _write_csv(csv_path, [(2020, 1.0), (2020, 1.1)])

    with pytest.raises(strict.ValidationError) as excinfo:
        strict.build_table(base_path=tmp_path, cache_path=tmp_path / "cache.parquet", use_cache=False)

    err = excinfo.value
    assert err.row == 3
    assert err.column == "Year"
    assert "strictly increasing" in err.reason


def test_validation_non_monotone_year(tmp_path: Path) -> None:
    csv_path = tmp_path / "nyiso" / "baseline" / "NYISO_A.csv"
    _write_csv(csv_path, [(2021, 1.0), (2020, 1.1)])

    with pytest.raises(strict.ValidationError) as excinfo:
        strict.build_table(base_path=tmp_path, cache_path=tmp_path / "cache.parquet", use_cache=False)

    err = excinfo.value
    assert err.row == 3
    assert err.column == "Year"
    assert "strictly increasing" in err.reason


def test_validation_negative_load(tmp_path: Path) -> None:
    csv_path = tmp_path / "nyiso" / "baseline" / "NYISO_A.csv"
    _write_csv(csv_path, [(2020, -1.0)])

    with pytest.raises(strict.ValidationError) as excinfo:
        strict.build_table(base_path=tmp_path, cache_path=tmp_path / "cache.parquet", use_cache=False)

    err = excinfo.value
    assert err.row == 2
    assert err.column == "Load_GWh"
    assert "non-negative" in err.reason


def test_validation_empty_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "nyiso" / "baseline" / "NYISO_A.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("Year,Load_GWh\n", encoding="utf-8")

    with pytest.raises(strict.ValidationError) as excinfo:
        strict.build_table(base_path=tmp_path, cache_path=tmp_path / "cache.parquet", use_cache=False)

    err = excinfo.value
    assert err.row is None
    assert err.column is None
    assert "no data rows" in err.reason


def test_cache_hit_skips_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "nyiso" / "baseline" / "NYISO_A.csv"
    _write_csv(csv_path, [(2025, 1.0)])
    cache_path = tmp_path / "cache.parquet"

    strict.build_table(base_path=tmp_path, cache_path=cache_path)

    def fail(_: Path) -> list[tuple[int, float]]:
        raise AssertionError("cache should satisfy request")

    monkeypatch.setattr(strict, "_read_csv_records", fail)
    frame = strict.build_table(base_path=tmp_path, cache_path=cache_path)
    assert frame.loc[0, "Load_GWh"] == pytest.approx(1.0)


def test_cache_invalidated_on_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "nyiso" / "baseline" / "NYISO_A.csv"
    _write_csv(csv_path, [(2025, 1.0)])
    cache_path = tmp_path / "cache.parquet"

    strict.build_table(base_path=tmp_path, cache_path=cache_path)

    original_reader = strict._read_csv_records

    _write_csv(csv_path, [(2025, 2.5)])

    calls: dict[str, int] = {"count": 0}

    def tracked(path: Path) -> list[tuple[int, float]]:
        calls["count"] += 1
        return original_reader(path)

    monkeypatch.setattr(strict, "_read_csv_records", tracked)
    frame = strict.build_table(base_path=tmp_path, cache_path=cache_path)

    assert calls["count"] == 1
    assert frame.loc[0, "Load_GWh"] == pytest.approx(2.5)


def test_multi_root_union_and_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root_a = tmp_path / "input" / "electricity" / "load_forecasts"
    root_b = tmp_path / "inputs" / "electricity" / "loadforecasts"

    _write_csv(root_a / "nyiso" / "baseline" / "NYISO_A.csv", [(2000, 1.0)])
    _write_csv(root_b / "pjm" / "baseline" / "PJM_ATSI.csv", [(2001, 2.0)])

    cache_path = tmp_path / "cache.parquet"

    frame = strict.build_table(base_path=[root_a, root_b], cache_path=cache_path)
    assert set(frame["Year"]) == {2000, 2001}

    _write_csv(root_b / "pjm" / "baseline" / "PJM_ATSI.csv", [(2001, 4.5)])

    calls: dict[str, int] = {"count": 0}

    original_reader = strict._read_csv_records

    def tracked(path: Path) -> list[tuple[int, float]]:
        calls["count"] += 1
        return original_reader(path)

    monkeypatch.setattr(strict, "_read_csv_records", tracked)

    frame = strict.build_table(base_path=[root_a, root_b], cache_path=cache_path)
    assert calls["count"] == 2
    zone_load = dict(zip(frame["zone"].astype(str), frame["Load_GWh"]))
    assert zone_load["NYISO_A"] == pytest.approx(1.0)
    assert zone_load["PJM_ATSI"] == pytest.approx(4.5)


def test_gui_dropdown_uses_full_scenario_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "st", SimpleNamespace(session_state={}))

    frame = pd.DataFrame(
        {
            "iso": pd.Categorical(["nyiso", "nyiso"], categories=["nyiso"]),
            "scenario": pd.Categorical(
                ["Gold Book High", "Gold Book Low"],
                categories=["Gold Book High", "Gold Book Low"],
            ),
            "zone": pd.Categorical(["ZoneA", "ZoneA"], categories=["ZoneA"]),
            "region_id": pd.Categorical(["NYISO_A", "NYISO_A"], categories=["NYISO_A"]),
            "Year": [2025, 2026],
            "Load_GWh": [1.0, 2.0],
            "iso_norm": ["nyiso", "nyiso"],
            "scenario_norm": ["gold_book_high", "gold_book_low"],
            "region_norm": ["nyiso_a", "nyiso_a"],
        }
    )

    monkeypatch.setattr(app, "_resolve_forecast_base_path", lambda: "root")
    monkeypatch.setattr(app, "_cached_forecast_frame", lambda base_path: frame)
    select_calls: dict[str, object] = {}

    def fake_select(selection, *, base_path=None, frame=None):  # type: ignore[no-untyped-def]
        select_calls["selection"] = dict(selection)
        select_calls["base_path"] = base_path
        select_calls["frame"] = frame
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
    assert settings.load_forecasts == {"nyiso": "Gold Book High"}
    assert select_calls["selection"] == {"nyiso": "Gold Book High"}


def test_validate_forecasts_raises_clear_error(tmp_path: Path) -> None:
    root = tmp_path / "load_forecasts"
    bad_csv = root / "nyiso" / "Scenario" / "ZoneA.csv"
    bad_csv.parent.mkdir(parents=True, exist_ok=True)
    bad_csv.write_text("Bad,Header\n1,2\n", encoding="utf-8")

    with pytest.raises(strict.ValidationError) as excinfo:
        load_forecasts.validate_forecasts(root)

    err = excinfo.value
    assert err.file == bad_csv
    assert err.row == 1
    assert err.column is None
    assert "expected columns" in err.reason


def test_available_iso_scenarios_handles_unknown_region(tmp_path: Path) -> None:
    csv_path = tmp_path / "nyiso" / "baseline" / "Mystery Region.csv"
    _write_csv(csv_path, [(2025, 1.0)])

    manifests = load_forecasts.available_iso_scenarios("nyiso", base_path=tmp_path)

    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.zone == "mystery_region"
    assert manifest.path == csv_path.resolve()


def test_available_iso_scenarios_includes_unrecognized_scenarios(tmp_path: Path) -> None:
    baseline_csv = tmp_path / "nyiso" / "baseline" / "NYISO_A.csv"
    stress_csv = tmp_path / "nyiso" / "stress_test" / "NYISO_A.csv"

    _write_csv(baseline_csv, [(2025, 1.0)])
    _write_csv(stress_csv, [(2026, 2.0)])

    manifests = load_forecasts.available_iso_scenarios("nyiso", base_path=tmp_path)

    assert [manifest.scenario for manifest in manifests] == ["baseline", "stress_test"]
    assert manifests[1].path == stress_csv.resolve()
