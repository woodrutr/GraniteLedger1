import logging
from pathlib import Path

import pandas as pd
import pytest

from engine import settings as engine_settings
import engine.data_loaders.load_forecasts as load_forecasts
from engine.io.load_forecast import load_load_forecasts


def _write_csv(path: Path, *, years: list[int], demand: list[float], region: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"year": years, "demand_mwh": demand}
    if region is not None:
        data["region"] = region
    pd.DataFrame(data).to_csv(path, index=False)


def test_load_forecasts_normalizes_region_ids(tmp_path: Path) -> None:
    csv_path = tmp_path / "load_forecasts.csv"
    pd.DataFrame(
        {
            "region_id": ["nyiso-n", "pjm-md"],
            "year": [2025, 2026],
            "load_gwh": [1.5, 2.0],
        }
    ).to_csv(csv_path, index=False)

    frame = load_forecasts.load_forecasts(csv_path)

    assert list(frame.columns) == [
        "region_id",
        "scenario_name",
        "scenario",
        "year",
        "load_mwh",
        "load_gwh",
    ]
    assert frame["region_id"].tolist() == ["NYISO_N", "PJM_MD"]
    assert frame["scenario_name"].tolist() == ["DEFAULT", "DEFAULT"]
    assert frame["load_mwh"].tolist() == [1500.0, 2000.0]


def test_load_load_forecasts_from_manifest(tmp_path: Path) -> None:
    root = tmp_path / "electricity" / "load_forecasts"
    root.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(
        {
            "region_id": ["ISO-NE_CT CT", "SOCO_SYS AL"],
            "scenario_name": ["isone_2025_baseline", "southeast_2025_reference"],
            "year": [2025, 2025],
            "load_gwh": [123.4, 567.8],
        }
    )
    manifest.to_csv(root / "load_forecasts.csv", index=False)

    frame = load_load_forecasts(root)

    assert set(frame.columns) == {"iso", "region_id", "scenario", "year", "load_gwh"}
    assert frame["scenario"].tolist() == ["isone_2025_baseline", "southeast_2025_reference"]
    assert frame["region_id"].tolist() == ["ISO-NE_CT", "SOCO_SYS"]
    assert set(frame["iso"]) == {"ISO-NE", "SOUTHEAST"}


def test_manifest_timestamp_grouping(tmp_path: Path) -> None:
    root = tmp_path / "electricity" / "load_forecasts"
    root.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(
        {
            "iso_zone": ["NYISO_A", "NYISO_A", "NYISO_B"],
            "scenario": ["baseline", "baseline", "baseline"],
            "timestamp": [
                "2025-01-01T00:00:00",
                "2025-06-01T00:00:00",
                "2025-01-01T00:00:00",
            ],
            "load_mwh": [1_000.0, 2_000.0, 3_000.0],
        }
    )
    manifest.to_csv(root / "load_forecasts.csv", index=False)

    frame = load_load_forecasts(root)

    assert {"iso", "region_id", "scenario", "year", "load_gwh", "load_mwh"} <= set(frame.columns)
    assert frame["scenario"].unique().tolist() == ["baseline"]
    assert frame["iso"].unique().tolist() == ["NYISO"]

    zone_a = frame[frame["region_id"] == "NYISO_A"].reset_index(drop=True)
    assert zone_a.at[0, "year"] == 2025
    assert zone_a.at[0, "load_mwh"] == pytest.approx(3_000.0)
    assert zone_a.at[0, "load_gwh"] == pytest.approx(3.0)

    zone_b = frame[frame["region_id"] == "NYISO_B"].reset_index(drop=True)
    assert zone_b.at[0, "year"] == 2025
    assert zone_b.at[0, "load_mwh"] == pytest.approx(3_000.0)
    assert zone_b.at[0, "load_gwh"] == pytest.approx(3.0)


def test_parse_folder_token_gold_book():
    source, vintage, scenario = load_forecasts.parse_folder_token("GoldBook_2025_Baseline")
    assert source == "Gold Book"
    assert vintage == 2025
    assert scenario == "Baseline"


def test_parse_folder_token_iso_ne_celt():
    source, vintage, scenario = load_forecasts.parse_folder_token("ISO-NE_CELT_2025_High")
    assert source == "ISO NE CELT"
    assert vintage == 2025
    assert scenario == "High"


def test_normalize_iso_aliases():
    from engine.normalization import normalize_iso_name

    assert normalize_iso_name("ISO NE") == "iso_ne"
    assert normalize_iso_name("isone") == "iso_ne"
    assert normalize_iso_name("NY ISO") == "nyiso"


@pytest.fixture
def forecast_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "electricity" / "load_forecasts"
    scenario_dir = root / "nyiso" / "GoldBook_2025_Baseline"
    _write_csv(scenario_dir / "NYISO_A.csv", years=[2025], demand=[123.0])
    monkeypatch.setattr(load_forecasts, "_default_input_root", lambda: root)
    monkeypatch.setattr(
        load_forecasts,
        "_region_registry",
        lambda: {"NYISO_A": "NYISO Zone A", "NYISO_B": "NYISO Zone B"},
    )
    return root


@pytest.fixture
def multi_iso_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "electricity" / "load_forecasts"
    nyiso_dir = root / "nyiso" / "GoldBook_2025_Baseline"
    _write_csv(nyiso_dir / "NYISO_A.csv", years=[2025], demand=[123.0])

    iso_ne_base = root / "iso-ne" / "ISO-NE_CELT_2025_Baseline"
    _write_csv(iso_ne_base / "ISO-NE_CT.csv", years=[2025], demand=[456.0])

    iso_ne_high = root / "iso-ne" / "ISO-NE_CELT_2025_High"
    _write_csv(iso_ne_high / "ISO-NE_CT.csv", years=[2025], demand=[789.0])

    monkeypatch.setattr(load_forecasts, "_default_input_root", lambda: root)
    monkeypatch.setattr(
        load_forecasts,
        "_region_registry",
        lambda: {
            "NYISO_A": "NYISO Zone A",
            "ISO-NE_CT": "ISO-NE Connecticut",
        },
    )
    return root
def test_load_zone_forecast_and_missing_warning(
    forecast_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        frame = load_forecasts.load_zone_forecast("nyiso", "NYISO_A", "GoldBook_2025_Baseline")
    assert list(frame.columns) == ["year", "demand_mwh"]
    assert frame["year"].tolist() == [2025]
    assert frame["demand_mwh"].tolist() == [123.0]

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        missing = load_forecasts.load_zone_forecast("nyiso", "NYISO_Z", "GoldBook_2025_Baseline")
    assert missing.empty
    assert any(
        "Zone 'NYISO_Z' missing in ISO 'nyiso' selection 'GoldBook_2025_Baseline'" in record.getMessage()
        for record in caplog.records
    )


def test_load_zone_forecast_detects_mwh_and_gwh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "electricity" / "load_forecasts"
    scenario_dir = root / "nyiso" / "GoldBook_2025_Baseline"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "NYISO_A.csv").write_text("Year,Demand_MWh\n2025,1.5\n", encoding="utf-8")
    (scenario_dir / "NYISO_B.csv").write_text("Year,Load_GWh\n2025,2.5\n", encoding="utf-8")

    monkeypatch.setattr(load_forecasts, "_default_input_root", lambda: root)
    monkeypatch.setattr(
        load_forecasts,
        "_region_registry",
        lambda: {"NYISO_A": "NYISO Zone A", "NYISO_B": "NYISO Zone B"},
    )

    mwh_frame = load_forecasts.load_zone_forecast("nyiso", "NYISO_A", "GoldBook_2025_Baseline")
    assert mwh_frame["demand_mwh"].iloc[0] == pytest.approx(1.5)

    gwh_frame = load_forecasts.load_zone_forecast("nyiso", "NYISO_B", "GoldBook_2025_Baseline")
    assert gwh_frame["demand_mwh"].iloc[0] == pytest.approx(2500.0)


import logging
from pathlib import Path

def test_available_iso_scenarios_returns_manifest(forecast_root: Path) -> None:
    scenarios = load_forecasts.available_iso_scenarios("nyiso")
    assert len(scenarios) == 1
    manifest = scenarios[0]
    assert manifest.iso == "nyiso"
    assert manifest.scenario == "baseline"
    assert manifest.zone == "NYISO_A"
    assert manifest.path.name == "NYISO_A.csv"


def test_load_demand_forecasts_uses_manifest_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "electricity" / "load_forecasts"
    root.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(
        {
            "state": ["MA", "MA"],
            "region_id": ["ISO-NE_MA", "ISO-NE_MA"],
            "scenario_name": ["celt_2025_baseline", "celt_2025_baseline"],
            "year": [2025, 2026],
            "load_gwh": [100.0, 110.0],
        }
    )
    manifest.to_csv(root / "load_forecasts.csv", index=False)

    monkeypatch.setattr(load_forecasts, "_load_region_state_map", lambda *_, **__: {})

    selection = {"MA": {"iso": "ISO-NE", "scenario": "celt_2025_baseline"}}

    demand = load_forecasts.load_demand_forecasts_selection(selection, root=root)

    assert not demand.empty
    assert demand["state"].unique().tolist() == ["MA"]
    assert demand["region_id"].unique().tolist() == ["ISO_NE_MA"]
    assert demand["load_gwh"].tolist() == [100.0, 110.0]


def test_load_demand_forecasts_splits_multi_state_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "iso": ["PJM", "PJM"],
            "region_id": ["PJM_PEPCO", "PJM_PEPCO"],
            "scenario": ["baseline", "baseline"],
            "year": [2025, 2026],
            "load_gwh": [100.0, 110.0],
        }
    )

    monkeypatch.setattr(
        load_forecasts, "_load_consolidated_forecasts", lambda *_args, **_kwargs: frame.copy()
    )
    monkeypatch.setattr(
        load_forecasts, "_load_manifest_frame", lambda *_args, **_kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(
        load_forecasts,
        "_load_state_region_shares",
        lambda *_args, **_kwargs: {
            "DC": {"PJM_PEPCO": 0.4},
            "MD": {"PJM_PEPCO": 0.6},
        },
    )

    selection = {
        "DC": {"iso": "PJM", "scenario": "baseline"},
        "MD": {"iso": "PJM", "scenario": "baseline"},
    }

    demand = load_forecasts.load_demand_forecasts_selection(selection, root=Path("/tmp"))

    assert sorted(demand["state"].unique().tolist()) == ["DC", "MD"]

    totals = demand.groupby("state")["load_gwh"].sum().to_dict()
    assert totals["DC"] == pytest.approx(100.0 * 0.4 + 110.0 * 0.4)
    assert totals["MD"] == pytest.approx(100.0 * 0.6 + 110.0 * 0.6)
def test_load_iso_scenario_table_infers_load_column(tmp_path: Path) -> None:
    root = tmp_path / "electricity" / "load_forecasts"
    scenario_dir = root / "iso_ne" / "Baseline"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    (scenario_dir / "ISO-NE_MA.csv").write_text(
        "year,ISO-NE_MA\n2025,123\n2026,456\n", encoding="utf-8"
    )

    table = load_forecasts.load_iso_scenario_table(
        "iso_ne", "Baseline", base_path=root
    )

    assert table["iso"].unique().tolist() == ["ISO-NE"]
    assert table["zone"].unique().tolist() == ["ISO_NE_MA"]
    assert table["load_gwh"].tolist() == [123.0, 456.0]


def test_default_input_root_tracks_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_root = tmp_path / "first" / "electricity" / "load_forecasts"
    second_root = tmp_path / "second" / "electricity" / "load_forecasts"

    _write_csv(
        first_root / "nyiso" / "GoldBook_2025_Baseline" / "NYISO_A.csv",
        years=[2025],
        demand=[1.0],
    )
    _write_csv(
        second_root / "nyiso" / "GoldBook_2026_Baseline" / "NYISO_A.csv",
        years=[2026],
        demand=[2.0],
    )

    state = {"root": first_root}

    def dynamic_root() -> Path:
        return state["root"]

    monkeypatch.setattr(engine_settings, "input_root", dynamic_root)
    monkeypatch.setattr(load_forecasts, "_input_root", dynamic_root)
    monkeypatch.setattr(
        load_forecasts,
        "_region_registry",
        lambda: {"NYISO_A": "NYISO Zone A"},
    )

    first_manifests = load_forecasts.available_iso_scenarios("nyiso")
    assert first_manifests
    assert all(manifest.path.is_relative_to(first_root) for manifest in first_manifests)

    state["root"] = second_root

    second_manifests = load_forecasts.available_iso_scenarios("nyiso")
    assert second_manifests
    assert all(manifest.path.is_relative_to(second_root) for manifest in second_manifests)


def _write_forecast_csv(
    path: Path,
    rows: list[tuple[int, float]],
    headers: tuple[str, str] = ("Year", "Load_GWh"),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    year_header, load_header = headers
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{year_header},{load_header}\n")
        for year, load in rows:
            handle.write(f"{year},{load}\n")


def test_load_iso_scenario_happy_path(tmp_path: Path) -> None:
    root = tmp_path / "electricity" / "load_forecasts"
    scenario_dir = root / "nyiso" / "goldbook_2025_baseline"
    _write_forecast_csv(
        scenario_dir / "NYISO_A.csv",
        [(2025, 100.0), (2026, 110.5)],
    )
    _write_forecast_csv(
        scenario_dir / "NYISO_B.csv",
        [(2025, 200.0), (2026, 210.25)],
    )

    frame = load_load_forecasts(str(root))

    assert list(frame.columns) == ["iso", "region_id", "scenario", "year", "load_gwh"]
    assert str(frame["year"].dtype) == "Int64"
    assert pd.api.types.is_float_dtype(frame["load_gwh"])
    assert len(frame) == 4
    assert frame["scenario"].unique().tolist() == ["goldbook_2025_baseline"]
    assert set(frame["region_id"].unique()) == {"NYISO_A", "NYISO_B"}


def test_missing_folder_raises(tmp_path: Path) -> None:
    missing_root = tmp_path / "electricity" / "load_forecasts"
    with pytest.raises(FileNotFoundError):
        load_load_forecasts(str(missing_root))


def test_schema_normalization(tmp_path: Path) -> None:
    root = tmp_path / "electricity" / "load_forecasts"
    scenario_dir = root / "nyiso" / "goldbook_2025_baseline"
    _write_forecast_csv(
        scenario_dir / "nyiso_a.csv",
        [(2025, 1.0)],
        headers=(" year ", "load_gwh"),
    )
    _write_forecast_csv(
        scenario_dir / "nyiso_b.csv",
        [(2026, 2.5)],
        headers=("YEAR", "Demand_GWh"),
    )

    frame = load_load_forecasts(root)

    assert list(frame.columns) == ["iso", "region_id", "scenario", "year", "load_gwh"]
    assert frame["year"].tolist() == [2025, 2026]
    assert pd.api.types.is_float_dtype(frame["load_gwh"])
    assert frame.loc[frame["region_id"] == "NYISO_A", "load_gwh"].iloc[0] == pytest.approx(1.0)
    assert frame.loc[frame["region_id"] == "NYISO_B", "load_gwh"].iloc[0] == pytest.approx(2.5)


def test_uppercase_extensions_discovered(tmp_path: Path) -> None:
    root = tmp_path / "electricity" / "load_forecasts"
    scenario_dir = root / "nyiso" / "goldbook_2025_baseline"
    _write_forecast_csv(
        scenario_dir / "NYISO_A.CSV",
        [(2025, 1.0)],
    )

    frame = load_load_forecasts(root)

    assert not frame.empty
    assert frame["region_id"].tolist() == ["NYISO_A"]


def test_frames_build(tmp_path: Path) -> None:
    root = tmp_path / "electricity" / "load_forecasts"
    scenario_dir = root / "nyiso" / "goldbook_2025_baseline"
    _write_forecast_csv(
        scenario_dir / "NYISO_A.csv",
        [(2025, 1.0), (2026, 1.5)],
    )
    _write_forecast_csv(
        scenario_dir / "NYISO_B.csv",
        [(2025, 2.0), (2026, 2.5)],
    )

    import sys
    import types

    created_modules: list[str] = []

    if "engine.outputs" not in sys.modules:
        outputs_stub = types.ModuleType("engine.outputs")
        outputs_stub.EngineOutputs = object  # type: ignore[attr-defined]
        sys.modules["engine.outputs"] = outputs_stub
        created_modules.append("engine.outputs")
    else:
        outputs_stub = sys.modules["engine.outputs"]
        if not hasattr(outputs_stub, "EngineOutputs"):
            setattr(outputs_stub, "EngineOutputs", object)

    if "engine.outputs.postprocess" not in sys.modules:
        postprocess_stub = types.ModuleType("engine.outputs.postprocess")
        postprocess_stub.apply_determinism = lambda value, *_, **__: value  # type: ignore[attr-defined]
        sys.modules["engine.outputs.postprocess"] = postprocess_stub
        created_modules.append("engine.outputs.postprocess")
    else:
        postprocess_stub = sys.modules["engine.outputs.postprocess"]
        if not hasattr(postprocess_stub, "apply_determinism"):
            setattr(
                postprocess_stub,
                "apply_determinism",
                lambda value, *_, **__: value,
            )

    from engine.orchestrate import build_frames

    try:
        frames_obj = build_frames(
            load_root=str(root),
            selection={
                "load": {
                    "scenario": "goldbook_2025_baseline",
                    "isos": ["nyiso"],
                }
            },
        )
    finally:
        for module_name in created_modules:
            sys.modules.pop(module_name, None)

    assert set(frames_obj) == {"load", "demand", "peak_demand", "state_zone_maps"}
    load_df = frames_obj["load"]
    demand_df = frames_obj["demand"]
    peak_df = frames_obj["peak_demand"]

    assert list(load_df.columns)[:5] == ["iso", "zone", "scenario", "year", "load_gwh"]
    assert load_df.shape[0] == 4

    assert list(demand_df.columns) == ["year", "region", "demand_mwh"]
    assert demand_df.shape[0] == 4
    assert list(peak_df.columns) == ["year", "region", "peak_demand_mw"]
    assert peak_df.empty
    merged = load_df.rename(columns={"zone": "region"}).merge(
        demand_df, on=["year", "region"], how="inner"
    )
    assert not merged.empty
    pd.testing.assert_series_equal(
        merged["demand_mwh"].reset_index(drop=True),
        (merged["load_gwh"] * 1_000.0).reset_index(drop=True),
        check_names=False,
    )
    assert set(demand_df["region"]) == {"NYISO_A", "NYISO_B"}
    assert isinstance(frames_obj["state_zone_maps"], dict)


def test_frames_build_accepts_normalized_scenario(tmp_path: Path) -> None:
    root = tmp_path / "electricity" / "load_forecasts"
    scenario_dir = root / "nyiso" / "GoldBook_2025_Baseline"
    _write_forecast_csv(
        scenario_dir / "NYISO_A.csv",
        [(2025, 1.0), (2026, 1.5)],
    )
    _write_forecast_csv(
        scenario_dir / "NYISO_B.csv",
        [(2025, 2.0), (2026, 2.5)],
    )

    import sys
    import types

    created_modules: list[str] = []

    if "engine.outputs" not in sys.modules:
        outputs_stub = types.ModuleType("engine.outputs")
        outputs_stub.EngineOutputs = object  # type: ignore[attr-defined]
        sys.modules["engine.outputs"] = outputs_stub
        created_modules.append("engine.outputs")
    else:
        outputs_stub = sys.modules["engine.outputs"]
        if not hasattr(outputs_stub, "EngineOutputs"):
            setattr(outputs_stub, "EngineOutputs", object)

    if "engine.outputs.postprocess" not in sys.modules:
        postprocess_stub = types.ModuleType("engine.outputs.postprocess")
        postprocess_stub.apply_determinism = lambda value, *_, **__: value  # type: ignore[attr-defined]
        sys.modules["engine.outputs.postprocess"] = postprocess_stub
        created_modules.append("engine.outputs.postprocess")
    else:
        postprocess_stub = sys.modules["engine.outputs.postprocess"]
        if not hasattr(postprocess_stub, "apply_determinism"):
            setattr(
                postprocess_stub,
                "apply_determinism",
                lambda value, *_, **__: value,
            )

    from engine.orchestrate import build_frames

    try:
        frames_obj = build_frames(
            load_root=str(root),
            selection={
                "load": {
                    "scenario": "Goldbook 2025 Baseline",
                    "isos": ["NYISO"],
                }
            },
        )
    finally:
        for module_name in created_modules:
            sys.modules.pop(module_name, None)

    assert set(frames_obj) == {"load", "demand", "peak_demand", "state_zone_maps"}
    load_df = frames_obj["load"]
    demand_df = frames_obj["demand"]

    assert not load_df.empty
    assert not demand_df.empty
    assert set(load_df["zone"]) == {"NYISO_A", "NYISO_B"}
    assert load_df["scenario"].str.contains("GoldBook_2025_Baseline").all()


def test_frames_build_accepts_iso_alias_with_underscore(tmp_path: Path) -> None:
    root = tmp_path / "electricity" / "load_forecasts"
    scenario_dir = root / "iso-ne" / "Celt_2025_Baseline"
    _write_forecast_csv(
        scenario_dir / "ISO-NE_CT.csv",
        [(2025, 3.0), (2026, 3.5)],
    )

    import sys
    import types

    created_modules: list[str] = []

    if "engine.outputs" not in sys.modules:
        outputs_stub = types.ModuleType("engine.outputs")
        outputs_stub.EngineOutputs = object  # type: ignore[attr-defined]
        sys.modules["engine.outputs"] = outputs_stub
        created_modules.append("engine.outputs")
    else:
        outputs_stub = sys.modules["engine.outputs"]
        if not hasattr(outputs_stub, "EngineOutputs"):
            setattr(outputs_stub, "EngineOutputs", object)

    if "engine.outputs.postprocess" not in sys.modules:
        postprocess_stub = types.ModuleType("engine.outputs.postprocess")
        postprocess_stub.apply_determinism = lambda value, *_, **__: value  # type: ignore[attr-defined]
        sys.modules["engine.outputs.postprocess"] = postprocess_stub
        created_modules.append("engine.outputs.postprocess")
    else:
        postprocess_stub = sys.modules["engine.outputs.postprocess"]
        if not hasattr(postprocess_stub, "apply_determinism"):
            setattr(
                postprocess_stub,
                "apply_determinism",
                lambda value, *_, **__: value,
            )

    from engine.orchestrate import build_frames

    try:
        frames_obj = build_frames(
            load_root=str(root),
            selection={
                "load": {
                    "scenario": "CELT 2025 Baseline",
                    "isos": ["iso_ne"],
                }
            },
        )
    finally:
        for module_name in created_modules:
            sys.modules.pop(module_name, None)

    load_df = frames_obj["load"]
    demand_df = frames_obj["demand"]

    assert not load_df.empty
    assert not demand_df.empty
    assert load_df["iso"].eq("ISO-NE").all()
    assert set(demand_df["region"]) == {"ISO_NE_CT"}
