import importlib
import logging
import io
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from regions.registry import REGIONS
from tests.carbon_price_utils import (
    CarbonPriceFixture,
    assert_aliases_match_canonical,
    with_carbon_vector_columns,
)
from tests.fixtures.dispatch_single_minimal import DEFAULT_REGION_ID, baseline_frames

SECOND_REGION_ID = list(REGIONS)[1]
THIRD_REGION_ID = list(REGIONS)[2]
FOURTH_REGION_ID = list(REGIONS)[3]
from gui.app import (
    DEEP_CARBON_UNSUPPORTED_MESSAGE,
    RunProgressState,
    _ALL_REGIONS_LABEL,
    _ProgressDisplay,
    _SIMULATION_BASE_PERCENT,
    _SIMULATION_PROGRESS_RANGE,
    _build_price_schedule,
    _bounded_percent,
    _coverage_default_display,
    canonical_region_label,
    run_policy_simulation,
)
from gui.outputs_visualization import load_emissions_data

streamlit = pytest.importorskip("streamlit")


def _baseline_config() -> dict:
    return {
        "years": [2025, 2026],
        "regions": [DEFAULT_REGION_ID],
        "allowance_market": {
            "cap": {"2025": 500_000.0, "2026": 450_000.0},
            "floor": 5.0,
            "ccr1_trigger": 10.0,
            "ccr1_qty": 0.0,
            "ccr2_trigger": 20.0,
            "ccr2_qty": 0.0,
            "cp_id": "CP1",
            "bank0": 50_000.0,
            "annual_surrender_frac": 1.0,
            "carry_pct": 1.0,
            "full_compliance_years": [2026],
            "control_period_years": 2,
            "resolution": "annual",
        },
    }


def _frames_for_years(years: list[int]) -> object:
    base = baseline_frames(year=years[0])
    load = float(base.demand()["demand_mwh"].iloc[0])
    demand = pd.DataFrame(
        [
            {"year": year, "region": DEFAULT_REGION_ID, "demand_mwh": load}
            for year in years
        ]
    )
    return base.with_frame("demand", demand)


class _SimpleOutputs:
    def __init__(
        self,
        annual: pd.DataFrame,
        emissions: pd.DataFrame,
        price: pd.DataFrame,
        flows: pd.DataFrame,
    ) -> None:
        self.annual = annual
        self.emissions_by_region = emissions
        self.price_by_region = price
        self.flows = flows
        self.emissions_total: Mapping[int, float] = {}

    def to_csv(self, target: Path) -> None:
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        self.annual.to_csv(target / "annual.csv", index=False)
        self.emissions_by_region.to_csv(target / "emissions_by_region.csv", index=False)
        self.price_by_region.to_csv(target / "price_by_region.csv", index=False)
        self.flows.to_csv(target / "flows.csv", index=False)


def _cleanup_temp_dir(result: dict) -> None:
    temp_dir = result.get("temp_dir")
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _assert_price_schedule(result: Mapping[str, Any], expected: Mapping[int, float]) -> None:
    annual = with_carbon_vector_columns(result["annual"])
    assert_aliases_match_canonical(annual)
    annual = annual.set_index("year")
    for year, price in expected.items():
        assert annual.loc[year, "cp_last"] == pytest.approx(
            price, rel=0.0, abs=1e-9
        )


def _emissions_by_year(result: Mapping[str, Any]) -> pd.Series:
    """Return the annual emissions indexed by year for convenience."""

    annual = result["annual"]
    assert isinstance(annual, pd.DataFrame)
    series = annual.set_index("year")["emissions_tons"].astype(float)
    series.index = series.index.astype(int)
    return series


def test_coverage_default_display_uses_region_labels():
    first_region = list(REGIONS)[0]
    second_region = list(REGIONS)[1]

    coverage_default = [first_region]
    coverage_choices = [
        _ALL_REGIONS_LABEL,
        canonical_region_label(first_region),
        canonical_region_label(second_region),
    ]

    display_values = _coverage_default_display(coverage_default, coverage_choices)

    assert display_values == [canonical_region_label(first_region)]

    fallback_values = _coverage_default_display(coverage_default, [_ALL_REGIONS_LABEL])
    assert fallback_values == [_ALL_REGIONS_LABEL]


def test_write_outputs_to_temp_falls_back_when_default_unwritable(monkeypatch):
    from gui import app as gui_app

    class DummyOutputs:
        def __init__(self) -> None:
            self.saved_to: Path | None = None

        def to_csv(self, target: Path) -> None:
            self.saved_to = Path(target)
            csv_path = self.saved_to / "dummy.csv"
            csv_path.write_text("value")

    fallback_base = Path.cwd() / ".graniteledger" / "tmp"

    monkeypatch.delenv("GRANITELEDGER_TMPDIR", raising=False)
    monkeypatch.setattr(gui_app.tempfile, "gettempdir", lambda: "/unwritable")

    def fake_mkdtemp(prefix: str, dir: str | None = None) -> str:
        if dir == "/unwritable":
            raise PermissionError("read-only filesystem")
        assert dir == str(fallback_base)
        target_dir = Path(dir) / "fallback"
        target_dir.mkdir(parents=True, exist_ok=False)
        return str(target_dir)

    monkeypatch.setattr(gui_app.tempfile, "mkdtemp", fake_mkdtemp)

    outputs = DummyOutputs()
    temp_dir, csv_files = gui_app._write_outputs_to_temp(outputs)

    try:
        assert outputs.saved_to == temp_dir
        assert temp_dir == fallback_base / "fallback"
        assert csv_files == {"dummy.csv": b"value"}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(fallback_base, ignore_errors=True)


def test_backend_generates_outputs(tmp_path):
    config = _baseline_config()
    frames = _frames_for_years([2025, 2026])
    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        cap_regions=[DEFAULT_REGION_ID],
        frames=frames,
    )

    assert "error" not in result
    annual = result["annual"]
    assert not annual.empty
    assert {"cp_last", "emissions_tons", "bank"}.issubset(annual.columns)

    csv_files = result["csv_files"]
    assert {
        "annual.csv",
        "emissions_by_region.csv",
        "price_by_region.csv",
        "flows.csv",
    } <= set(csv_files)
    for content in csv_files.values():
        assert isinstance(content, (bytes, bytearray))

    _cleanup_temp_dir(result)



def test_backend_policy_toggle_affects_price():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    enabled = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=[DEFAULT_REGION_ID],
        frames=frames,
        carbon_policy_enabled=True,
    )
    disabled = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=[DEFAULT_REGION_ID],
        frames=frames,
        carbon_policy_enabled=False,
    )

    assert "error" not in enabled
    assert "error" not in disabled

    enabled_annual = with_carbon_vector_columns(enabled["annual"])
    disabled_annual = with_carbon_vector_columns(disabled["annual"])
    assert_aliases_match_canonical(enabled_annual)
    assert_aliases_match_canonical(disabled_annual)

    price_enabled = float(
        enabled["annual"].loc[enabled["annual"]["year"] == 2025, "cp_last"].iloc[0]
    )
    price_disabled = float(
        disabled["annual"].loc[disabled["annual"]["year"] == 2025, "cp_last"].iloc[0]
    )


    assert price_enabled >= 0.0
    assert price_disabled == pytest.approx(0.0)
    assert price_enabled >= price_disabled

    _cleanup_temp_dir(enabled)
    _cleanup_temp_dir(disabled)


def test_backend_marks_allowance_price_output():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=[DEFAULT_REGION_ID],
        frames=frames,
        carbon_policy_enabled=True,
    )


    assert result.get('_price_output_type') == 'allowance'

    _cleanup_temp_dir(result)


def test_emissions_pipeline_yields_regional_records():
    years = list(range(2025, 2031))
    config = _baseline_config()
    config['years'] = years
    allowance = config['allowance_market']
    allowance['cap'] = {str(year): 800_000.0 - 10_000.0 * idx for idx, year in enumerate(years)}
    allowance['full_compliance_years'] = [years[-1]]
    allowance['control_period_years'] = len(years)

    frames = baseline_frames(year=years[0])
    base_load = float(frames.demand()['demand_mwh'].iloc[0])
    target_region = DEFAULT_REGION_ID
    demand = pd.DataFrame(
        [
            {"year": year, "region": target_region, "demand_mwh": base_load}
            for year in years
        ]
    )
    frames = frames.with_frame('demand', demand)
    units = frames.units()
    units['region'] = target_region
    frames = frames.with_frame('units', units)

    policy_records = []
    for idx, year in enumerate(years):
        policy_records.append(
            {
                'year': year,
                'cap_tons': allowance['cap'][str(year)],
                'floor_dollars': 0.0,
                'ccr1_trigger': 10.0,
                'ccr1_qty': 0.0,
                'ccr2_trigger': 20.0,
                'ccr2_qty': 0.0,
                'cp_id': 'CP1',
                'full_compliance': year == years[-1],
                'bank0': allowance.get('bank0', 0.0),
                'annual_surrender_frac': allowance.get('annual_surrender_frac', 1.0),
                'carry_pct': allowance.get('carry_pct', 1.0),
                'policy_enabled': True,
                'resolution': 'annual',
            }
        )
    policy = pd.DataFrame(policy_records)
    frames = frames.with_frame('policy', policy)

    result = run_policy_simulation(
        config,
        start_year=years[0],
        end_year=years[-1],
        cap_regions=[10],
        frames=frames,
    )

    emissions_result = result['emissions_by_region']
    assert isinstance(emissions_result, pd.DataFrame)
    assert not emissions_result.empty
    assert {'year', 'region', 'region_label', 'emissions_tons'}.issubset(emissions_result.columns)
    expected_label = canonical_region_label(target_region)
    assert emissions_result['region_label'].astype(str).str.contains(expected_label, regex=False).any()

    processed = load_emissions_data(result)
    assert not processed.empty
    assert processed.attrs.get('emissions_source') == 'engine'
    assert processed['region_label'].astype(str).str.contains(expected_label, regex=False).any()

    _cleanup_temp_dir(result)


def test_backend_returns_technology_frames(monkeypatch):
    config = _baseline_config()
    frames = _frames_for_years([2025])

    capacity_df = pd.DataFrame(
        [
            {"year": 2025, "technology": "wind", "capacity_mw": 10.0},
            {"year": 2025, "technology": "solar", "capacity_mw": 5.0},
        ]
    )
    generation_df = pd.DataFrame(
        [
            {"year": 2025, "technology": "wind", "generation_mwh": 25.0},
            {"year": 2025, "technology": "solar", "generation_mwh": 15.0},
        ]
    )

    class StubOutputs:
        def __init__(self) -> None:
            self.annual = pd.DataFrame(
                [
                    {
                        "year": 2025,
                        "allowance_price": 0.0,
                        "emissions_tons": 0.0,
                        "bank": 0.0,
                    }
                ]
            )
            self.emissions_by_region = pd.DataFrame(
                [{"year": 2025, "region": DEFAULT_REGION_ID, "emissions_tons": 0.0}]
            )
            self.price_by_region = pd.DataFrame(
                [{"year": 2025, "region": DEFAULT_REGION_ID, "price": 0.0}]
            )
            self.flows = pd.DataFrame([{"from": "A", "to": "B", "value": 0.0}])
            self.capacity_by_technology = capacity_df
            self.generation_by_technology = generation_df
            self.emissions_total: Mapping[int, float] = {}

        def to_csv(self, target: Path) -> None:
            self.annual.to_csv(target / "annual.csv", index=False)
            self.emissions_by_region.to_csv(
                target / "emissions_by_region.csv", index=False
            )
            self.price_by_region.to_csv(target / "price_by_region.csv", index=False)
            self.flows.to_csv(target / "flows.csv", index=False)

    def stub_runner(frames_obj, **kwargs):
        assert kwargs.get("capacity_expansion") is True
        return StubOutputs()

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: stub_runner)

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=[DEFAULT_REGION_ID],
        frames=frames,
        dispatch_capacity_expansion=True,
    )

    assert "capacity_by_technology" in result
    assert isinstance(result["capacity_by_technology"], pd.DataFrame)
    assert "generation_by_technology" in result
    assert isinstance(result["generation_by_technology"], pd.DataFrame)

    _cleanup_temp_dir(result)


def test_backend_marks_carbon_price_output():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=False,
        carbon_price_enabled=True,
        carbon_price_value=15.0,
    )

    assert result.get('_price_output_type') == 'carbon'

    _cleanup_temp_dir(result)


def test_cap_region_alias_resolution_collapses_duplicates():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=["NYISO_J", "nyiso j", "New York City"],
        frames=frames,
    )

    assert "error" not in result
    assert result.get("cap_regions") == ["NYISO_J"]
    carbon_cfg = result["config"]["modules"]["carbon_policy"]
    assert carbon_cfg.get("regions") == ["NYISO_J"]

    _cleanup_temp_dir(result)


def test_cap_region_all_selection_collapses_to_empty():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=["All"],
        frames=frames,
    )

    assert "error" not in result
    assert result.get("cap_regions") in (None, [])
    carbon_cfg = result["config"]["modules"]["carbon_policy"]
    assert carbon_cfg.get("regions") in (None, [])

    _cleanup_temp_dir(result)


def test_cap_region_unknown_label_errors():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=["Atlantis"],
        frames=frames,
    )

    assert "error" in result
    assert "Unable to resolve cap region" in result["error"]


def test_render_results_carbon_price_hides_allowance_columns(monkeypatch):
    from gui import app as gui_app

    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=False,
        carbon_price_enabled=True,
        carbon_price_value=25.0,
    )

    assert result.get('_price_output_type') == 'carbon'

    class DummyTab:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyStreamlit:
        def __init__(self) -> None:
            self.dataframes = []
            self.tab_labels = []

        def error(self, *args, **kwargs):
            return None

        def caption(self, *args, **kwargs):
            return None

        def subheader(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

        def markdown(self, *args, **kwargs):
            return None

        def line_chart(self, *args, **kwargs):
            return None

        def bar_chart(self, *args, **kwargs):
            return None

        def altair_chart(self, *args, **kwargs):
            return None

        def dataframe(self, frame, **kwargs):
            self.dataframes.append(frame)
            return None

        def tabs(self, labels):
            self.tab_labels.append(list(labels))
            return [DummyTab() for _ in labels]

        def download_button(self, *args, **kwargs):
            return None

        def multiselect(self, label, options, *, default=None, key=None):
            return list(default or [])

    dummy_st = DummyStreamlit()
    monkeypatch.setattr(gui_app, "st", dummy_st)

    gui_app._render_results(result)

    assert dummy_st.tab_labels, "Expected result tabs to be rendered"
    assert "Allowance bank" not in dummy_st.tab_labels[0]

    price_tables = [
        frame
        for frame in dummy_st.dataframes
        if isinstance(frame, pd.DataFrame) and "Carbon price ($/ton)" in frame.columns
    ]

    assert price_tables, "Expected carbon price table to be rendered"

    allowed_columns = {
        "year",
        "Carbon price ($/ton)",
        "cp_all",
        "cp_exempt",
        "cp_effective",
        "cp_last",
        "emissions_tons",
    }
    disallowed_columns = {"allowances_total", "bank"}

    for table in price_tables:
        assert set(table.columns).issubset(allowed_columns)
        assert disallowed_columns.isdisjoint(table.columns)

    _cleanup_temp_dir(result)



def test_dispatch_capacity_toggle_updates_config():
    config = _baseline_config()
    frames = _frames_for_years([2025])
    module_config = {
        "electricity_dispatch": {"enabled": True, "capacity_expansion": True}
    }

    disabled = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=[DEFAULT_REGION_ID],
        frames=frames,
        dispatch_capacity_expansion=False,
        module_config=module_config,
    )

    assert "error" not in disabled
    assert disabled["config"].get("sw_expansion") == 0
    dispatch_cfg = disabled["module_config"]["electricity_dispatch"]
    assert dispatch_cfg["capacity_expansion"] is False

    enabled = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=[DEFAULT_REGION_ID],
        frames=frames,
        dispatch_capacity_expansion=True,
        module_config=module_config,
    )

    assert "error" not in enabled
    assert enabled["config"].get("sw_expansion") == 1
    assert enabled["module_config"]["electricity_dispatch"]["capacity_expansion"] is True

    _cleanup_temp_dir(disabled)
    _cleanup_temp_dir(enabled)


def test_backend_handles_renamed_engine_outputs(monkeypatch):
    config = _baseline_config()
    frames = _frames_for_years([2025])

    annual = pd.DataFrame(
        [
            {
                "year": 2025,
                "cp_all": 12.0,
                "cp_effective": 12.0,
                "cp_exempt": 0.0,
                "cp_last": 12.0,
            }
        ]
    )
    emissions = pd.DataFrame(
        [{"year": 2025, "region": DEFAULT_REGION_ID, "emissions_tons": 1.0}]
    )
    prices = pd.DataFrame(
        [{"year": 2025, "region": DEFAULT_REGION_ID, "price": 45.0}]
    )
    flows = pd.DataFrame(
        [{"year": 2025, "from_region": "A", "to_region": "B", "flow_mwh": 10.0}]
    )


    class FakeOutputs:
        def __init__(self) -> None:
            self.annual_results = annual
            self.emissions = emissions
            self.dispatch_price_by_region = prices
            self.network_flows = flows
            self.emissions_total: Mapping[int, float] = {}

        def to_csv(self, outdir):
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            self.annual_results.to_csv(outdir / "annual.csv", index=False)
            self.emissions.to_csv(outdir / "emissions_by_region.csv", index=False)
            self.dispatch_price_by_region.to_csv(outdir / "price_by_region.csv", index=False)
            self.network_flows.to_csv(outdir / "flows.csv", index=False)

    def fake_runner(*args, **kwargs):
        return FakeOutputs()

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: fake_runner)

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
    )

    pd.testing.assert_frame_equal(result["annual"][annual.columns], annual)
    emissions_result = result["emissions_by_region"]
    assert {"region_canonical", "region_label"}.issubset(emissions_result.columns)
    pd.testing.assert_frame_equal(
        emissions_result[emissions.columns],
        emissions,
    )
    pd.testing.assert_frame_equal(result["price_by_region"], prices)
    pd.testing.assert_frame_equal(result["flows"], flows)

    _cleanup_temp_dir(result)


def test_recent_results_repository_tracks_outputs(monkeypatch):
    from gui import recent_results

    recent_results.clear_recent_result()

    config = _baseline_config()
    frames = _frames_for_years([2025])

    annual = pd.DataFrame(
        [
            {
                "year": 2025,
                "cp_all": 5.0,
                "cp_effective": 5.0,
                "cp_exempt": 0.0,
                "cp_last": 5.0,
            }
        ]
    )
    emissions = pd.DataFrame(
        [{"year": 2025, "region": DEFAULT_REGION_ID, "emissions_tons": 1.0}]
    )
    prices = pd.DataFrame(
        [{"year": 2025, "region": DEFAULT_REGION_ID, "price": 15.0}]
    )


    class CapturingOutputs:
        def __init__(self) -> None:
            self.annual = annual
            self.emissions_by_region = emissions
            self.price_by_region = prices
            self.flows = pd.DataFrame()
            self.emissions_total: Mapping[int, float] = {}

        def to_csv(self, target: Path) -> None:
            target = Path(target)
            target.mkdir(parents=True, exist_ok=True)
            self.annual.to_csv(target / "annual.csv", index=False)
            self.emissions_by_region.to_csv(
                target / "emissions_by_region.csv", index=False
            )
            self.price_by_region.to_csv(target / "price_by_region.csv", index=False)

    def fake_runner(*args, **kwargs):
        return CapturingOutputs()

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: fake_runner)

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
    )

    stored = recent_results.get_recent_result()
    assert stored is result
    pd.testing.assert_frame_equal(stored["annual"][annual.columns], annual)
    emissions_result = stored["emissions_by_region"]
    assert {"region_canonical", "region_label"}.issubset(emissions_result.columns)
    pd.testing.assert_frame_equal(
        emissions_result[emissions.columns],
        emissions,
    )

    recent_results.clear_recent_result()
    _cleanup_temp_dir(result)


def test_backend_handles_legacy_runner_without_deep_kw(monkeypatch):
    config = _baseline_config()
    frames = _frames_for_years([2025])

    annual = pd.DataFrame(
        [
            {
                "year": 2025,
                "cp_all": 12.0,
                "cp_effective": 12.0,
                "cp_exempt": 0.0,
                "cp_last": 12.0,
            }
        ]
    )
    emissions = pd.DataFrame(
        [{"year": 2025, "region": DEFAULT_REGION_ID, "emissions_tons": 1.0}]
    )
    prices = pd.DataFrame(
        [{"year": 2025, "region": DEFAULT_REGION_ID, "price": 45.0}]
    )
    flows = pd.DataFrame(
        [{"year": 2025, "from_region": "A", "to_region": "B", "flow_mwh": 10.0}]
    )



    called: dict[str, bool] = {}

    class LegacyOutputs:
        def __init__(self) -> None:
            self.annual = annual
            self.emissions_by_region = emissions
            self.price_by_region = prices
            self.flows = flows
            self.emissions_total: Mapping[int, float] = {}

        def to_csv(self, target: Path) -> None:
            target = Path(target)
            target.mkdir(parents=True, exist_ok=True)
            self.annual.to_csv(target / "annual.csv", index=False)
            self.emissions_by_region.to_csv(target / "emissions_by_region.csv", index=False)
            self.price_by_region.to_csv(target / "price_by_region.csv", index=False)
            self.flows.to_csv(target / "flows.csv", index=False)

    def legacy_runner(
        frames,
        *,
        years=None,
        price_initial=0.0,
        enable_floor=True,
        enable_ccr=True,
        use_network=False,
        carbon_price_schedule=None,
        progress_cb=None,
    ):
        called["executed"] = True
        return LegacyOutputs()

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: legacy_runner)

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        deep_carbon_pricing=False,
    )

    assert "error" not in result
    assert called.get("executed") is True
    pd.testing.assert_frame_equal(result["annual"][annual.columns], annual)
    emissions_result = result["emissions_by_region"]
    assert {"region_canonical", "region_label"}.issubset(emissions_result.columns)
    pd.testing.assert_frame_equal(
        emissions_result[emissions.columns],
        emissions,
    )
    pd.testing.assert_frame_equal(result["price_by_region"], prices)
    pd.testing.assert_frame_equal(result["flows"], flows)

    _cleanup_temp_dir(result)


def test_backend_rejects_deep_mode_when_runner_lacks_kw(monkeypatch):
    config = _baseline_config()
    frames = _frames_for_years([2025])
    called: dict[str, bool] = {}

    def legacy_runner(
        frames,
        *,
        years=None,
        price_initial=0.0,
        enable_floor=True,
        enable_ccr=True,
        use_network=False,
        carbon_price_schedule=None,
        progress_cb=None,
    ):
        called["executed"] = True
        return {}

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: legacy_runner)

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        deep_carbon_pricing=True,
        carbon_price_enabled=True,
        carbon_price_value=10.0,
    )

    assert result.get("error") == DEEP_CARBON_UNSUPPORTED_MESSAGE
    assert "executed" not in called


def test_backend_disabled_toggle_propagates_flags(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        policy = frames.policy().to_policy()
        captured["enabled"] = policy.enabled
        captured["ccr1"] = policy.ccr1_enabled
        captured["ccr2"] = policy.ccr2_enabled
        captured["control"] = policy.control_period_length
        captured["banking"] = policy.banking_enabled
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=False,
        ccr1_enabled=False,
        ccr2_enabled=False,
        allowance_banking_enabled=False,
        control_period_years=4,
    )

    assert "error" not in result
    assert captured.get("enabled") is False
    assert captured.get("ccr1") is False
    assert captured.get("ccr2") is False
    assert captured.get("control") is None
    assert captured.get("banking") is False

    _cleanup_temp_dir(result)


def test_backend_enforces_carbon_mode_exclusivity(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        policy = frames.policy().to_policy()
        captured["policy_enabled"] = policy.enabled
        captured["price_schedule"] = kwargs.get("carbon_price_schedule")
        captured["price_value"] = kwargs.get("carbon_price_value")
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025, 2026])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        frames=frames,
        carbon_policy_enabled=True,
        carbon_price_enabled=True,
        carbon_price_value=15.0,
        carbon_price_escalator_pct=0.0,
    )

    assert "error" not in result
    assert captured.get("policy_enabled") is False
    schedule = captured.get("price_schedule")
    assert isinstance(schedule, Mapping)
    assert schedule  # non-empty schedule when price enabled
    assert captured.get("price_value") == pytest.approx(15.0)

    module_config = result["module_config"]
    policy_cfg = module_config["carbon_policy"]
    price_cfg = module_config["carbon_price"]
    assert not policy_cfg.get("enabled")
    assert price_cfg.get("enabled")
    assert sum(1 for flag in (policy_cfg.get("enabled"), price_cfg.get("enabled")) if flag) == 1

    _cleanup_temp_dir(result)


def test_backend_control_period_defaults_to_config(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        policy = frames.policy().to_policy()
        captured["control"] = policy.control_period_length
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025, 2026])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        frames=frames,
        control_period_years=None,
    )

    assert "error" not in result
    assert captured.get("control") == 2
    carbon_cfg = result["module_config"]["carbon_policy"]
    assert carbon_cfg.get("control_period_years") is None

    _cleanup_temp_dir(result)


class _DummyStatus:
    def __init__(self) -> None:
        self.updates: list[tuple[str, str]] = []

    def update(self, *, label: str, state: str) -> None:
        self.updates.append((label, state))


class _DummyProgressBar:
    def __init__(self) -> None:
        self.values: list[int] = []

    def progress(self, value: int) -> None:
        self.values.append(int(value))


class _DummyLogContainer:
    def __init__(self) -> None:
        self.rendered: list[str] = []

    def markdown(self, value: str) -> None:
        self.rendered.append(value)

    def caption(self, value: str) -> None:
        self.rendered.append(f"caption:{value}")


def test_progress_display_throttles_iteration_updates():
    status = _DummyStatus()
    progress_bar = _DummyProgressBar()
    log_container = _DummyLogContainer()
    state = RunProgressState()
    display = _ProgressDisplay(status, progress_bar, log_container, state)

    display.handle_stage(
        "run_start",
        {"total_years": 2, "years": [2025, 2026], "max_iter": 60, "tolerance": 1e-3},
    )
    display.handle_stage("year_start", {"index": 0, "year": 2025})

    display.handle_iteration(
        "iteration",
        {"year": 2025, "iteration": 1, "max_iter": 60, "price": 12.5},
    )
    rendered_initial = list(log_container.rendered)
    display.handle_iteration(
        "iteration",
        {"year": 2025, "iteration": 2, "max_iter": 60, "price": 12.75},
    )
    assert log_container.rendered == rendered_initial
    assert "iteration 2/60" in state.log[-1]

    display.handle_iteration(
        "iteration",
        {"year": 2025, "iteration": 5, "max_iter": 60, "price": 13.1},
    )
    assert len(log_container.rendered) == len(rendered_initial) + 1

    display.handle_iteration(
        "iteration",
        {
            "year": 2025,
            "iteration": 12,
            "max_iter": 60,
            "price": 14.75,
            "converged": True,
        },
    )
    expected_percent = _bounded_percent(
        _SIMULATION_BASE_PERCENT
        + ((0 + 1) / 2) * _SIMULATION_PROGRESS_RANGE
    )
    assert progress_bar.values[-1] == expected_percent
    assert state.percent_complete == expected_percent
    assert "iteration 12/60" in state.log[-1]


def test_progress_display_handles_run_failed_stage():
    status = _DummyStatus()
    progress_bar = _DummyProgressBar()
    log_container = _DummyLogContainer()
    state = RunProgressState()
    display = _ProgressDisplay(status, progress_bar, log_container, state)

    display.handle_stage("compiling_assumptions", {"status": "start"})
    display.handle_stage("run_failed", {"error": "boom"})

    assert state.stage == "run_failed"
    assert state.message.startswith("Simulation failed")
    assert any("Simulation failed" in entry for entry in state.log)
    assert progress_bar.values[-1] == _bounded_percent(state.percent_complete)
    assert status.updates[-1][1] == "error"


def test_backend_carbon_price_reduces_emissions():
    config = _baseline_config()
    config["allowance_market"]["cap"] = {"2025": 1_000_000.0, "2026": 1_000_000.0}
    config["allowance_market"]["floor"] = 0.0
    frames = _frames_for_years([2025, 2026])

    baseline = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        frames=frames,
        carbon_price_enabled=False,
    )
    priced = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        frames=frames,
        carbon_price_enabled=True,
        carbon_price_value=100.0,
        carbon_price_escalator_pct=0.0,
    )

    assert "error" not in baseline
    assert "error" not in priced

    baseline_emissions = _emissions_by_year(baseline)
    priced_emissions = _emissions_by_year(priced)

    for year in baseline_emissions.index:
        assert priced_emissions.loc[year] < baseline_emissions.loc[year]

    _cleanup_temp_dir(baseline)
    _cleanup_temp_dir(priced)


def test_backend_carbon_price_schedule_lowers_future_emissions():
    config = _baseline_config()
    config["allowance_market"]["cap"] = {}
    frames = _frames_for_years([2024, 2025])

    result = run_policy_simulation(
        config,
        start_year=2024,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=False,
        carbon_price_enabled=True,
        carbon_price_schedule={2024: 0.0, 2025: 100.0},
        carbon_price_escalator_pct=0.0,
    )

    assert "error" not in result

    annual = result["annual"].set_index("year")
    first_year = int(annual.index.min())
    later_year = int(annual.index.max())

    assert later_year > first_year
    assert annual.loc[later_year, "emissions_tons"] < annual.loc[first_year, "emissions_tons"]

    _cleanup_temp_dir(result)


def test_backend_control_period_override_applies(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        policy = frames.policy().to_policy()
        captured["control"] = policy.control_period_length
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025, 2026])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        frames=frames,
        control_period_years=4,
    )

    assert "error" not in result
    assert captured.get("control") == 4
    carbon_cfg = result["module_config"]["carbon_policy"]
    assert carbon_cfg.get("control_period_years") == 4

    _cleanup_temp_dir(result)


def test_price_schedule_forward_fills_without_cap_or_banking():
    years = list(range(2025, 2031))
    schedule = {2025: 45.0, 2030: 48.15}
    expected = {
        2025: 45.0,
        2026: 45.0,
        2027: 45.0,
        2028: 45.0,
        2029: 45.0,
        2030: 48.15,
    }

    scenarios = [
        {"carbon_policy_enabled": False, "allowance_banking_enabled": True},
        {"carbon_policy_enabled": False, "allowance_banking_enabled": False},
    ]

    for options in scenarios:
        config = _baseline_config()
        frames = _frames_for_years(years)
        result = run_policy_simulation(
            config,
            start_year=2025,
            end_year=2030,
            frames=frames,
            carbon_price_enabled=True,
            carbon_price_schedule=schedule,
            **options,
        )

        assert "error" not in result
        _assert_price_schedule(result, expected)
        _cleanup_temp_dir(result)


def test_backend_errors_when_demand_years_do_not_overlap():
    config = _baseline_config()
    frames = _frames_for_years([2030, 2031])

    result = run_policy_simulation(config, frames=frames)

    assert "error" in result
    assert "Demand data covers years" in result["error"]


def test_backend_dispatch_and_carbon_modules(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        policy = frames.policy().to_policy()
        captured["carbon_enabled"] = policy.enabled
        captured["use_network"] = kwargs.get("use_network")
        captured["deep_carbon_pricing"] = kwargs.get("deep_carbon_pricing")
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025])

    module_config = {
        "carbon_policy": {"enabled": True, "allowance_banking_enabled": True},
        "electricity_dispatch": {
            "enabled": True,
            "mode": "single_node",
            "capacity_expansion": True,
            "reserve_margins": True,
        },
    }

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=[DEFAULT_REGION_ID],
        frames=frames,
        carbon_policy_enabled=True,
        dispatch_use_network=False,
        module_config=module_config,
    )

    assert "error" not in result
    assert captured.get("carbon_enabled") is True
    assert captured.get("use_network") is False
    assert captured.get("deep_carbon_pricing") is False
    assert captured.get("capacity_expansion") is True
    assert captured.get("report_by_technology") is True
    dispatch_cfg = result["module_config"]["electricity_dispatch"]
    assert dispatch_cfg["enabled"] is True
    assert dispatch_cfg["use_network"] is False
    assert dispatch_cfg.get("deep_carbon_pricing") is False
    carbon_cfg = result["module_config"]["carbon_policy"]
    assert carbon_cfg.get("regions") == [DEFAULT_REGION_ID]

    _cleanup_temp_dir(result)


def test_backend_canonicalizes_cap_region_aliases():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    alias_entries = [
        "NYISO_J",
        "New York City (NYISO_J)",
        "New York City",
        "nyiso_j",
    ]

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=alias_entries,
        frames=frames,
    )

    assert "error" not in result
    carbon_cfg = result["module_config"].get("carbon_policy", {})
    assert carbon_cfg.get("regions") == ["NYISO_J"]
    assert result.get("cap_regions") == ["NYISO_J"]

    _cleanup_temp_dir(result)


def test_backend_rejects_unknown_cap_region():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=["Atlantis"],
        frames=frames,
    )

    assert "error" in result
    assert "Atlantis" in str(result["error"])


def test_backend_mutual_exclusion_without_deep():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=True,
        cap_regions=[DEFAULT_REGION_ID],
        carbon_price_enabled=True,
        carbon_price_value=25.0,
        deep_carbon_pricing=False,
    )

    assert result.get("error") == "Cannot enable both carbon cap and carbon price simultaneously."


def test_backend_deep_carbon_combines_prices(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        captured["deep_carbon_pricing"] = kwargs.get("deep_carbon_pricing")
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=True,
        cap_regions=[DEFAULT_REGION_ID],
        carbon_price_enabled=True,
        carbon_price_value=15.0,
        deep_carbon_pricing=True,
    )

    assert "error" not in result
    assert captured.get("deep_carbon_pricing") is True

    annual = with_carbon_vector_columns(result["annual"])
    assert_aliases_match_canonical(annual)
    row = annual.loc[annual["year"] == 2025].iloc[0]
    allowance_price = float(row["cp_all"])
    exogenous_price = float(row["cp_exempt"])
    effective_price = float(row["cp_effective"])

    assert row["cp_last"] == pytest.approx(allowance_price)
    assert exogenous_price == pytest.approx(15.0)
    assert effective_price == pytest.approx(allowance_price + exogenous_price)

    dispatch_cfg = result["module_config"].get("electricity_dispatch", {})
    assert dispatch_cfg.get("deep_carbon_pricing") is True

    _cleanup_temp_dir(result)


def test_price_only_all_regions_sets_allowance_component():
    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=False,
        carbon_price_enabled=True,
        carbon_price_value=37.5,
        coverage_regions=["All"],
    )

    assert "error" not in result

    annual = with_carbon_vector_columns(result["annual"])
    row = annual.loc[annual["year"] == 2025].iloc[0]

    assert row["cp_all"] == pytest.approx(37.5)
    assert row["cp_exempt"] == pytest.approx(0.0)
    assert row["cp_effective"] == pytest.approx(37.5)

    _cleanup_temp_dir(result)


def test_backend_slider_range_covers_all_years(monkeypatch):
    config = _baseline_config()
    config["years"] = [2025, 2030]
    frames = _frames_for_years(list(range(2025, 2034)))

    captured: dict[str, list[int]] = {}

    def capturing_runner(frames_obj, **kwargs):
        years = list(kwargs.get("years", []))
        captured["years"] = years
        annual_rows = [
            {
                "year": year,
                "cp_all": 0.0,
                "cp_effective": 0.0,
                "cp_exempt": 0.0,
                "cp_last": 0.0,
                "iterations": 0,
                "emissions_tons": 0.0,
                "allowances_minted": 0.0,
                "allowances_available": 0.0,
                "bank": 0.0,
                "surrender": 0.0,
                "obligation": 0.0,
                "finalized": False,
                "shortage_flag": False,
                "ccr1_trigger": 0.0,
                "ccr1_issued": 0.0,
                "ccr2_trigger": 0.0,
                "ccr2_issued": 0.0,
            }
            for year in years
        ]
        annual = pd.DataFrame(annual_rows)
        emissions = pd.DataFrame(
            [
                {"year": year, "region": DEFAULT_REGION_ID, "emissions_tons": 0.0}
                for year in years
            ]
        )
        price = pd.DataFrame(
            [
                {"year": year, "region": DEFAULT_REGION_ID, "price": 0.0}
                for year in years
            ]
        )
        flows = pd.DataFrame(columns=["year", "from_region", "to_region", "flow_mwh"])
        return _SimpleOutputs(annual, emissions, price, flows)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2033,
        frames=frames,
    )


    expected_years = list(range(2025, 2034))
    assert captured.get("years") == expected_years
    assert result["annual"]["year"].tolist() == expected_years

    _cleanup_temp_dir(result)


def test_backend_deep_carbon_adjusts_dispatch_costs(monkeypatch):
    config = _baseline_config()
    frames = _frames_for_years([2025])

    def stub_runner(frames_obj, **kwargs):
        deep = bool(kwargs.get("deep_carbon_pricing"))
        price = 30.0 + (5.0 if deep else 0.0)
        annual = pd.DataFrame(
            [
                {
                    "year": 2025,
                    "cp_all": 25.0,
                    "cp_effective": price,
                    "cp_exempt": price - 25.0,
                    "cp_last": price,
                    "iterations": 0,
                    "emissions_tons": 0.0,
                    "allowances_minted": 0.0,
                    "allowances_available": 0.0,
                    "bank": 0.0,
                    "surrender": 0.0,
                    "obligation": 0.0,
                    "finalized": False,
                    "shortage_flag": False,
                    "ccr1_trigger": 0.0,
                    "ccr1_issued": 0.0,
                    "ccr2_trigger": 0.0,
                    "ccr2_issued": 0.0,
                }
            ]
        )
        emissions = pd.DataFrame(
            [{"year": 2025, "region": DEFAULT_REGION_ID, "emissions_tons": 0.0}]
        )
        price_df = pd.DataFrame(
            [{"year": 2025, "region": DEFAULT_REGION_ID, "price": price}]
        )
        flows = pd.DataFrame(columns=["year", "from_region", "to_region", "flow_mwh"])
        return _SimpleOutputs(annual, emissions, price_df, flows)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: stub_runner)

    result_off = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_price_enabled=True,
        carbon_price_value=15.0,
        deep_carbon_pricing=False,
    )
    result_on = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_price_enabled=True,
        carbon_price_value=15.0,
        deep_carbon_pricing=True,
    )

    assert "error" not in result_off
    assert "error" not in result_on

    off_price = float(result_off["price_by_region"]["price"].iloc[0])
    on_price = float(result_on["price_by_region"]["price"].iloc[0])
    assert on_price > off_price

    _cleanup_temp_dir(result_off)
    _cleanup_temp_dir(result_on)


def test_backend_ccr_triggers_reported():
    config = _baseline_config()
    config["allowance_market"]["ccr1_qty"] = 40_000.0
    config["allowance_market"]["ccr2_qty"] = 60_000.0
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=[DEFAULT_REGION_ID],
        frames=frames,
        enable_ccr=True,
        ccr1_enabled=True,
        ccr2_enabled=True,
    )

    assert "error" not in result
    annual = result["annual"].set_index("year")
    assert {"ccr1_trigger", "ccr2_trigger"}.issubset(annual.columns)

    first_year = annual.index[0]
    assert annual.loc[first_year, "ccr1_trigger"] == pytest.approx(
        config["allowance_market"]["ccr1_trigger"],
        rel=0.0,
        abs=1e-9,
    )
    assert annual.loc[first_year, "ccr2_trigger"] == pytest.approx(
        config["allowance_market"]["ccr2_trigger"],
        rel=0.0,
        abs=1e-9,
    )
    assert "ccr1_issued" in annual.columns
    assert "ccr2_issued" in annual.columns

    _cleanup_temp_dir(result)


def test_backend_reports_missing_deep_support(monkeypatch):
    def legacy_runner(
        frames,
        *,
        years=None,
        price_initial=0.0,
        tol=1e-3,
        max_iter=25,
        relaxation=0.5,
        enable_floor=True,
        enable_ccr=True,
        price_cap=1000.0,
        use_network=False,
        carbon_price_schedule=None,
        progress_cb=None,
    ):
        raise AssertionError("legacy runner should not be invoked when unsupported")


    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: legacy_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        carbon_policy_enabled=True,
        cap_regions=[DEFAULT_REGION_ID],
        carbon_price_enabled=True,
        carbon_price_value=20.0,
        deep_carbon_pricing=True,
    )

    assert result.get("error") == (
        "Deep carbon pricing requires an updated engine. "
        "Please upgrade engine.run_loop.run_end_to_end_from_frames."
    )

def test_backend_carbon_price_disables_cap(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        policy = frames.policy().to_policy()
        captured["carbon_enabled"] = policy.enabled
        captured["control"] = policy.control_period_length
        captured["price_schedule"] = kwargs.get("carbon_price_schedule")
        captured["price_value"] = kwargs.get("carbon_price_value")
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2026])

    module_config = {"carbon_price": {"enabled": True, "price_per_ton": 37.0}}

    result = run_policy_simulation(
        config,
        start_year=2026,
        end_year=2026,
        frames=frames,
        carbon_policy_enabled=False,
        carbon_price_enabled=True,
        carbon_price_value=37.0,
        module_config=module_config,
    )


    assert "error" not in result
    assert captured.get("carbon_enabled") is False
    assert captured.get("control") is None
    schedule = captured.get("price_schedule")
    assert isinstance(schedule, Mapping)
    assert schedule.get(2026) == pytest.approx(37.0)
    assert captured.get("price_value") == pytest.approx(37.0)

    carbon_cfg = result["module_config"].get("carbon_policy", {})
    assert carbon_cfg.get("enabled") is False
    assert carbon_cfg.get("control_period_years") is None
    price_cfg = result["module_config"].get("carbon_price", {})
    assert price_cfg.get("enabled") is True
    assert price_cfg.get("price_per_ton") == pytest.approx(37.0)

    _cleanup_temp_dir(result)


def test_backend_banking_toggle_disables_bank(tmp_path, caplog):
    config = _baseline_config()
    frames = _frames_for_years([2025, 2026])

    with caplog.at_level(logging.WARNING):
        result = run_policy_simulation(
            config,
            start_year=2025,
            end_year=2026,
            cap_regions=[DEFAULT_REGION_ID],
            frames=frames,
            allowance_banking_enabled=False,
        )

    assert "error" not in result
    annual = result["annual"]
    assert annual["bank"].eq(0.0).all()
    assert any("Allowance banking disabled" in record.message for record in caplog.records)

    _cleanup_temp_dir(result)


def test_backend_updates_allowance_market_config():
    config = _baseline_config()
    config["allowance_market"]["cap"] = {}
    frames = _frames_for_years([2025, 2026])

    schedule = {2025: 345_000.0, 2026: 320_000.0}

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        frames=frames,
        cap_regions=[DEFAULT_REGION_ID],
        carbon_cap_schedule=schedule,
        initial_bank=1234.0,
        allowance_banking_enabled=True,
        ccr1_enabled=True,
        ccr2_enabled=False,
    )

    assert "error" not in result

    allowance_module = result["module_config"]["allowance_market"]
    allowance_config = result["config"]["allowance_market"]
    expected_schedule = {year: float(value) for year, value in schedule.items()}

    assert allowance_module["enabled"] is True
    assert allowance_config["enabled"] is True
    assert allowance_module["cap"] == expected_schedule
    assert allowance_config["cap"] == expected_schedule
    assert allowance_module["bank0"] == pytest.approx(1234.0)
    assert allowance_config["bank0"] == pytest.approx(1234.0)
    assert allowance_module["ccr1_enabled"] is True
    assert allowance_module["ccr2_enabled"] is False
    assert allowance_config["ccr1_enabled"] is True
    assert allowance_config["ccr2_enabled"] is False

    _cleanup_temp_dir(result)


def test_backend_zeroes_allowance_bank_when_disabled_config():
    config = _baseline_config()
    frames = _frames_for_years([2025, 2026])

    schedule = {2025: 400_000.0, 2026: 390_000.0}

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        frames=frames,
        cap_regions=[DEFAULT_REGION_ID],
        carbon_cap_schedule=schedule,
        initial_bank=9876.0,
        allowance_banking_enabled=False,
    )

    assert "error" not in result

    allowance_module = result["module_config"]["allowance_market"]
    allowance_config = result["config"]["allowance_market"]

    assert allowance_module["bank0"] == pytest.approx(0.0)
    assert allowance_config["bank0"] == pytest.approx(0.0)
    assert allowance_module["enabled"] is True
    assert allowance_config["enabled"] is True

    _cleanup_temp_dir(result)


def test_backend_builds_price_schedule_for_run_years(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        captured["carbon_price_schedule"] = kwargs.get("carbon_price_schedule")
        captured["carbon_price_value"] = kwargs.get("carbon_price_value")
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025, 2026, 2027, 2028, 2029, 2030])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2030,
        frames=frames,
        carbon_policy_enabled=False,
        carbon_price_enabled=True,
        carbon_price_value=45.0,
        carbon_price_escalator_pct=7.0,
    )

    assert "error" not in result

    schedule = captured.get("carbon_price_schedule")
    assert isinstance(schedule, Mapping)
    expected = _build_price_schedule(2025, 2030, 45.0, 7.0)
    assert schedule == expected
    assert captured.get("carbon_price_value") == pytest.approx(45.0)

    _cleanup_temp_dir(result)


def test_backend_bank_column_tracks_allowances(tmp_path):
    config = _baseline_config()
    frames = _frames_for_years([2025, 2026])

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2026,
        cap_regions=[DEFAULT_REGION_ID],
        frames=frames,
        allowance_banking_enabled=True,
    )


    assert "error" not in result
    annual = result["annual"].copy()
    assert not annual.empty

    annual = annual.sort_values("year").reset_index(drop=True)
    bank0 = float(config["allowance_market"]["bank0"])
    expected = []
    bank_prev = bank0
    for _, row in annual.iterrows():
        allowances_total = float(row["allowances_available"])
        emissions = float(row["emissions_tons"])
        bank_prev = max(bank_prev + allowances_total - emissions, 0.0)
        expected.append(bank_prev)

    assert annual["bank"].tolist() == pytest.approx(expected)

    _cleanup_temp_dir(result)


def test_backend_returns_error_for_invalid_frames():
    config = _baseline_config()
    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=[DEFAULT_REGION_ID],
        frames={"demand": pd.DataFrame()},
    )

    assert "error" in result


def test_backend_preserves_explicit_coverage_overrides(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, pd.DataFrame] = {}

    def capturing_runner(frames, **kwargs):
        captured["coverage"] = frames.coverage()
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = _frames_for_years([2025])
    coverage = pd.DataFrame(
        [
            {"region": DEFAULT_REGION_ID, "year": 2025, "covered": False},
            {"region": SECOND_REGION_ID, "year": -1, "covered": True},
        ]
    )
    frames = frames.with_frame("coverage", coverage)

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=[DEFAULT_REGION_ID],
        frames=frames,
    )

    assert "error" not in result
    coverage_frame = captured.get("coverage")
    assert coverage_frame is not None
    default_rows = coverage_frame[
        coverage_frame["region"].astype(str) == DEFAULT_REGION_ID
    ]
    assert set(default_rows["year"].astype(int)) == {-1, 2025}
    explicit_value = default_rows.loc[default_rows["year"] == 2025, "covered"].iloc[0]
    default_value = default_rows.loc[default_rows["year"] == -1, "covered"].iloc[0]
    assert bool(explicit_value) is False
    assert bool(default_value) is True

    _cleanup_temp_dir(result)


def test_backend_builds_default_frames(tmp_path):
    config = _baseline_config()
    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        cap_regions=[DEFAULT_REGION_ID],
    )

    assert "error" not in result
    assert not result["annual"].empty

    _cleanup_temp_dir(result)


def test_backend_coverage_selection_builds_frame(monkeypatch):
    real_runner = importlib.import_module("engine.run_loop").run_end_to_end_from_frames
    captured: dict[str, object] = {}

    def capturing_runner(frames, **kwargs):
        captured["coverage_df"] = frames.coverage()
        captured["coverage_map"] = frames.coverage_for_year(2025)
        captured["capacity_expansion"] = kwargs.get("capacity_expansion")
        captured["report_by_technology"] = kwargs.get("report_by_technology")
        kwargs.pop("capacity_expansion", None)
        kwargs.pop("report_by_technology", None)
        return real_runner(frames, **kwargs)

    monkeypatch.setattr("gui.app._ensure_engine_runner", lambda: capturing_runner)

    config = _baseline_config()
    frames = baseline_frames(year=2025)
    units = frames.units()
    units.loc[units["unit_id"] == "coal-1", "region"] = THIRD_REGION_ID
    units.loc[units["unit_id"] == "gas-1", "region"] = FOURTH_REGION_ID
    units.loc[units["unit_id"] == "wind-1", "region"] = THIRD_REGION_ID
    frames = frames.with_frame("units", units)
    demand = pd.DataFrame(
        [
            {"year": 2025, "region": THIRD_REGION_ID, "demand_mwh": 250_000.0},
            {"year": 2025, "region": FOURTH_REGION_ID, "demand_mwh": 250_000.0},
        ]
    )
    frames = frames.with_frame("demand", demand)

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        coverage_regions=[THIRD_REGION_ID],
        dispatch_use_network=True,
    )

    assert "error" not in result
    coverage_df = captured.get("coverage_df")
    assert isinstance(coverage_df, pd.DataFrame)
    assert {THIRD_REGION_ID, FOURTH_REGION_ID}.issubset(set(coverage_df["region"]))
    north_flag = bool(
        coverage_df.loc[coverage_df["region"] == THIRD_REGION_ID, "covered"].iloc[0]
    )
    south_flag = bool(
        coverage_df.loc[coverage_df["region"] == FOURTH_REGION_ID, "covered"].iloc[0]
    )
    assert north_flag is True
    assert south_flag is False
    carbon_cfg = result["module_config"].get("carbon_policy", {})
    assert carbon_cfg.get("regions") == [THIRD_REGION_ID]

    _cleanup_temp_dir(result)


def test_single_region_dispatch_requires_uniform_coverage():
    config = _baseline_config()
    frames = baseline_frames(year=2025)
    units = frames.units()
    units.loc[units["unit_id"] == "coal-1", "region"] = THIRD_REGION_ID
    units.loc[units["unit_id"] == "gas-1", "region"] = FOURTH_REGION_ID
    frames = frames.with_frame("units", units)

    demand = pd.DataFrame(
        [
            {"year": 2025, "region": THIRD_REGION_ID, "demand_mwh": 200_000.0},
            {"year": 2025, "region": FOURTH_REGION_ID, "demand_mwh": 200_000.0},
        ]
    )
    frames = frames.with_frame("demand", demand)

    result = run_policy_simulation(
        config,
        start_year=2025,
        end_year=2025,
        frames=frames,
        coverage_regions=[THIRD_REGION_ID],
        dispatch_use_network=False,
    )

    assert "error" in result
    assert "uniform carbon coverage" in str(result["error"]).lower()


def test_build_policy_frame_control_override():
    from gui.app import _build_policy_frame

    config = _baseline_config()
    years = [2025, 2026, 2027]
    frame = _build_policy_frame(
        config,
        years,
        carbon_policy_enabled=True,
        control_period_years=2,
    )

    assert set(frame["year"]) == set(years)
    assert frame["policy_enabled"].all()
    assert frame["control_period_years"].dropna().unique().tolist() == [2]
    assert frame["bank_enabled"].all()


def test_build_policy_frame_disabled_defaults():
    from gui.app import _build_policy_frame

    config = _baseline_config()
    years = [2025]
    frame = _build_policy_frame(config, years, carbon_policy_enabled=False)

    assert not frame["policy_enabled"].any()
    assert frame["cap_tons"].iloc[0] > 0.0
    assert bool(frame["ccr1_enabled"].iloc[0]) is False
    assert frame["bank_enabled"].eq(False).all()


def test_load_config_data_accepts_various_sources(tmp_path):
    from gui.app import _load_config_data

    mapping = {"a": 1}
    assert _load_config_data(mapping) == mapping

    toml_text = "value = 1\n"
    assert _load_config_data(toml_text.encode("utf-8"))["value"] == 1

    temp_file = tmp_path / "config.toml"
    temp_file.write_text("value = 2\n", encoding="utf-8")
    assert _load_config_data(str(temp_file))["value"] == 2

    stream = io.StringIO("value = 3\n")
    assert _load_config_data(stream)["value"] == 3

    with pytest.raises(TypeError):
        _load_config_data(object())


def test_year_and_selection_helpers_cover_branches():
    from gui.app import _years_from_config, _select_years

    config = {"years": [{"year": 2025}, 2026]}
    years = _years_from_config(config)
    assert years == [2025, 2026]

    fallback = {"start_year": 2024, "end_year": 2022}
    fallback_years = _years_from_config(fallback)
    assert fallback_years == [2022, 2023, 2024]

    selected = _select_years(fallback_years, start_year=2023, end_year=2024)
    assert selected == [2023, 2024]

    sparse_years = [2025, 2030]
    expanded = _select_years(sparse_years, start_year=2025, end_year=2030)
    assert expanded == [2025, 2026, 2027, 2028, 2029, 2030]

    with pytest.raises(ValueError):
        _select_years(years, start_year=2026, end_year=2024)
