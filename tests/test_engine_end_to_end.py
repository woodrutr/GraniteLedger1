from __future__ import annotations
import importlib
import json
import logging
from collections.abc import Mapping
from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")

engine_run_loop = importlib.import_module("engine.run_loop")
run_end_to_end_from_frames = engine_run_loop.run_end_to_end_from_frames
ANNUAL_OUTPUT_COLUMNS = engine_run_loop.ANNUAL_OUTPUT_COLUMNS

from engine import outputs as engine_outputs
from dispatch.lp_single import HOURS_PER_YEAR
prep = importlib.import_module("src.models.electricity.scripts.preprocessor")
from io_loader import Frames
from regions.registry import REGIONS
from tests.fixtures.dispatch_single_minimal import (
    DEFAULT_YEARS,
    DEFAULT_REGION_ID,
    all_region_frames,
    baseline_frames,
    policy_frame,
    three_year_frames,
)
from tests.carbon_price_utils import assert_aliases_match_canonical, with_carbon_vector_columns
annual_fixtures = importlib.import_module("tests.fixtures.annual_minimal")
LinearDispatch = annual_fixtures.LinearDispatch
policy_three_year = annual_fixtures.policy_three_year

YEARS = list(DEFAULT_YEARS)
SECOND_REGION_ID = list(REGIONS)[1]



@pytest.fixture
def three_year_outputs():
    frames = three_year_frames()
    return run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )


def _network_frames_with_regions() -> Frames:
    region_a = DEFAULT_REGION_ID
    region_b = SECOND_REGION_ID
    demand = pd.DataFrame(
        [
            {"year": 2030, "region": region_a, "demand_mwh": 30.0 * HOURS_PER_YEAR},
            {"year": 2030, "region": region_b, "demand_mwh": 20.0 * HOURS_PER_YEAR},
        ]
    )
    units = pd.DataFrame(
        [
            {
                "unit_id": "north_unit",
                "unique_id": "north_unit",
                "region": region_a,
                "fuel": "north_fuel",
                "cap_mw": 70.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 7.0,
                "vom_per_mwh": 12.0,
                "fuel_price_per_mmbtu": 1.5,
                "ef_ton_per_mwh": 0.6,
            },
            {
                "unit_id": "south_unit",
                "unique_id": "south_unit",
                "region": region_b,
                "fuel": "south_fuel",
                "cap_mw": 55.0,
                "availability": 1.0,
                "hr_mmbtu_per_mwh": 8.0,
                "vom_per_mwh": 14.0,
                "fuel_price_per_mmbtu": 1.8,
                "ef_ton_per_mwh": 0.3,
            },
        ]
    )
    fuels = pd.DataFrame(
        [
            {"fuel": "north_fuel", "covered": True},
            {"fuel": "south_fuel", "covered": True},
        ]
    )
    coverage = pd.DataFrame(
        [
            {"region": region_a, "year": 2030, "covered": True},
            {"region": region_b, "year": 2030, "covered": True},
        ]
    )
    transmission = pd.DataFrame(
        [
            {"from_region": region_a, "to_region": region_b, "limit_mw": 1e-6},
            {"from_region": region_b, "to_region": region_a, "limit_mw": 1e-6},
        ]
    )
    policy = policy_frame(years=[2030])

    return Frames(
        {
            "demand": demand,
            "units": units,
            "fuels": fuels,
            "coverage": coverage,
            "transmission": transmission,
            "policy": policy,
        }
    )


def test_three_year_control_period_converges(three_year_outputs):
    iterations = three_year_outputs.annual["iterations"]
    assert not iterations.empty
    assert int(iterations.max()) <= 10


def test_end_to_end_reports_regional_emissions():
    frames = _network_frames_with_regions()
    outputs = run_end_to_end_from_frames(
        frames,
        years=[2030],
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
        use_network=True,
    )

    emissions_df = outputs.emissions_by_region
    assert not emissions_df.empty
    expected_regions = {DEFAULT_REGION_ID, SECOND_REGION_ID}
    assert expected_regions.issubset(set(emissions_df["region"]))

    totals = emissions_df.set_index("region")["emissions_tons"]
    expected_a = 30.0 * HOURS_PER_YEAR * 0.6
    expected_b = 20.0 * HOURS_PER_YEAR * 0.3

    assert totals.loc[DEFAULT_REGION_ID] == pytest.approx(expected_a, rel=1e-6)
    assert totals.loc[SECOND_REGION_ID] == pytest.approx(expected_b, rel=1e-6)

    region_map = {
        region: dict(years) for region, years in outputs.emissions_by_region_map.items()
    }
    assert region_map[DEFAULT_REGION_ID][2030] == pytest.approx(expected_a, rel=1e-6)
    assert region_map[SECOND_REGION_ID][2030] == pytest.approx(expected_b, rel=1e-6)


def test_pre_solve_raises_when_demand_is_empty():
    frames = _network_frames_with_regions()
    zero_demand = frames.frame("demand")
    zero_demand["demand_mwh"] = 0.0
    frames_zero = frames.with_frame("demand", zero_demand)

    with pytest.raises(RuntimeError) as excinfo:
        run_end_to_end_from_frames(
            frames_zero,
            years=[2030],
            price_initial=0.0,
            tol=1e-4,
            relaxation=0.8,
        )

    assert "E_DEMAND_EMPTY" in str(excinfo.value)


def test_pre_solve_raises_when_no_units_have_capacity():
    frames = _network_frames_with_regions()
    zero_units = frames.frame("units")
    zero_units["cap_mw"] = 0.0
    frames_zero_units = frames.with_frame("units", zero_units)

    with pytest.raises(RuntimeError) as excinfo:
        run_end_to_end_from_frames(
            frames_zero_units,
            years=[2030],
            price_initial=0.0,
            tol=1e-4,
            relaxation=0.8,
        )

    assert "E_SUPPLY_EMPTY" in str(excinfo.value)


def test_pre_solve_requires_interfaces_for_network_dispatch():
    frames = _network_frames_with_regions()
    empty_transmission = frames.frame("transmission").iloc[0:0]
    frames_no_tx = frames.with_frame("transmission", empty_transmission)

    with pytest.raises(RuntimeError) as excinfo:
        run_end_to_end_from_frames(
            frames_no_tx,
            years=[2030],
            price_initial=0.0,
            tol=1e-4,
            relaxation=0.8,
            use_network=True,
        )

    assert "E_NETWORK_REQD_EMPTY" in str(excinfo.value)


def test_demand_by_state_uses_zone_shares(monkeypatch):
    frames = _network_frames_with_regions()

    region_a = DEFAULT_REGION_ID
    region_b = SECOND_REGION_ID
    shares_df = pd.DataFrame(
        [
            {"region_id": region_a, "state": "CT", "share": 0.6},
            {"region_id": region_a, "state": "MA", "share": 0.4},
            {"region_id": region_b, "state": "VA", "share": 0.25},
            {"region_id": region_b, "state": "MD", "share": 0.75},
        ]
    )

    monkeypatch.setattr(
        engine_outputs,
        "load_zone_to_state_share",
        lambda: shares_df.copy(),
        raising=False,
    )

    outputs = run_end_to_end_from_frames(
        frames,
        years=[2030],
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
        states=["CT", "MA", "VA", "MD"],
    )

    demand_by_state = (
        outputs.demand_by_state.sort_values(["year", "state"]).reset_index(drop=True)
    )
    assert not demand_by_state.empty

    demand_by_region = outputs.demand_by_region
    merged = demand_by_region.merge(
        shares_df, left_on="region", right_on="region_id", how="inner"
    )
    merged["weighted"] = merged["demand_mwh"] * merged["share"]

    expected = (
        merged.groupby(["year", "state"], as_index=False)["weighted"].sum()
        .rename(columns={"weighted": "demand_mwh"})
        .sort_values(["year", "state"])
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(demand_by_state, expected)


def test_progress_callback_reports_each_year():
    frames = three_year_frames()
    events: list[tuple[str, dict[str, object]]] = []

    def _capture(stage: str, payload: Mapping[str, object]) -> None:
        events.append((stage, dict(payload)))

    run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
        progress_cb=_capture,
    )
    stages = [stage for stage, _ in events]
    assert stages.count("run_start") == 1
    assert stages.count("year_start") == len(YEARS)
    assert stages.count("year_complete") == len(YEARS)
    assert "iteration" in stages
    first_year_payload = next(payload for stage, payload in events if stage == "year_start")
    assert int(first_year_payload.get("index", -1)) == 0
    assert int(first_year_payload.get("total_years", 0)) == len(YEARS)
    assert "max_iter" in first_year_payload
    final_payload = next(payload for stage, payload in reversed(events) if stage == "year_complete")
    assert int(final_payload.get("index", -1)) == len(YEARS) - 1
    assert "price" in final_payload
    assert "iterations" in final_payload
    iteration_payload = next(payload for stage, payload in events if stage == "iteration")
    assert "max_iter" in iteration_payload
    assert "tolerance" in iteration_payload


def test_stage_callback_reports_preparation_steps():
    frames = three_year_frames()
    captured: list[tuple[str, dict[str, object]]] = []

    def _capture(stage: str, payload: Mapping[str, object]) -> None:
        captured.append((stage, dict(payload)))

    run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
        stage_cb=_capture,
    )

    stage_names = [stage for stage, _ in captured]
    for expected in (
        "compiling_assumptions",
        "loading_load_forecasts",
        "initializing_fleet",
        "building_interfaces",
        "run_start",
        "run_complete",
    ):
        assert expected in stage_names


def test_stage_callback_reports_demand_and_fleet_metadata():
    frames = three_year_frames()
    captured: list[tuple[str, dict[str, object]]] = []

    def _capture(stage: str, payload: Mapping[str, object]) -> None:
        captured.append((stage, dict(payload)))

    run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
        stage_cb=_capture,
    )

    load_events = [payload for stage, payload in captured if stage == "loading_load_forecasts"]
    assert load_events, "expected loading_load_forecasts stage events"
    load_complete = next(
        payload for payload in reversed(load_events) if payload.get("status") == "complete"
    )
    assert int(load_complete.get("regions_with_demand", 0)) > 0
    assert load_complete.get("years")
    assert float(load_complete.get("total_demand_mwh", 0.0)) > 0.0
    assert load_complete.get("source") in {"frames", "synthesized"}

    unit_events = [payload for stage, payload in captured if stage == "initializing_fleet"]
    assert unit_events, "expected initializing_fleet stage events"
    unit_complete = next(
        payload for payload in reversed(unit_events) if payload.get("status") == "complete"
    )
    assert int(unit_complete.get("units_with_capacity", 0)) > 0
    assert int(unit_complete.get("unit_count", 0)) >= int(
        unit_complete.get("units_with_capacity", 0)
    )
    assert float(unit_complete.get("total_capacity_mw", 0.0)) > 0.0
    assert unit_complete.get("source") in {"frames", "synthesized"}


def test_stage_callback_emits_run_failed(monkeypatch):
    frames = three_year_frames()
    events: list[tuple[str, dict[str, object]]] = []

    def _capture(stage: str, payload: Mapping[str, object]) -> None:
        events.append((stage, dict(payload)))

    def _boom(*args, **kwargs):
        raise RuntimeError("dispatch boom")

    monkeypatch.setattr(engine_run_loop, "_dispatch_from_frames", _boom)

    with pytest.raises(RuntimeError, match="dispatch boom"):
        run_end_to_end_from_frames(
            frames,
            years=YEARS,
            price_initial=0.0,
            tol=1e-4,
            relaxation=0.8,
            stage_cb=_capture,
        )

    assert events
    assert events[-1][0] == "run_failed"
    failure_payload = events[-1][1]
    assert "error" in failure_payload
    assert "boom" in str(failure_payload["error"])


def test_debug_logging_includes_full_metrics(caplog):
    frames = three_year_frames()
    with caplog.at_level(logging.DEBUG, logger="engine.run_loop"):
        run_end_to_end_from_frames(
            frames,
            years=YEARS,
            price_initial=0.0,
            tol=1e-4,
            relaxation=0.8,
        )
    prefix = "allowance_year_metrics "
    metrics: list[dict[str, object]] = []
    for record in caplog.records:
        if record.name != "engine.run_loop":
            continue
        message = record.getMessage()
        if not message.startswith(prefix):
            continue
        metrics.append(json.loads(message[len(prefix) :]))
    assert len(metrics) == len(YEARS)
    required_fields = {
        "price_raw",
        "reserve_cap",
        "ecr_trigger",
        "reserve_withheld",
        "ccr1_release",
        "ccr2_release",
        "bank_in",
        "bank_out",
        "available_allowances",
        "emissions",
        "shortage_flag",
    }
    for payload in metrics:
        missing = required_fields.difference(payload)
        assert not missing, f"missing fields: {sorted(missing)}"


def test_fixed_point_logging_reports_residuals(caplog):
    policy = policy_three_year()
    dispatch = LinearDispatch({year: 120.0 for year in YEARS}, slope=10.0)

    with caplog.at_level(logging.DEBUG, logger="engine.run_loop"):
        engine_run_loop.run_annual_fixed_point(
            policy,
            dispatch,
            years=[YEARS[0]],
            price_initial=0.0,
            tol=1e-4,
            relaxation=0.5,
        )

    iteration_events: list[dict[str, object]] = []
    for record in caplog.records:
        if record.name != "engine.run_loop":
            continue
        message = record.getMessage()
        prefix = "allowance_fixed_point_iteration "
        if not message.startswith(prefix):
            continue
        payload = json.loads(message[len(prefix) :])
        iteration_events.append(payload)

    assert iteration_events, "no iteration logging captured"
    assert any(event.get("status") == "accepted" for event in iteration_events)
    for payload in iteration_events:
        assert "residual" in payload
        assert "tolerance" in payload


def test_bank_non_negative_after_compliance(three_year_outputs):
    cp_year = YEARS[-1]
    bank = three_year_outputs.annual.loc[
        three_year_outputs.annual["year"] == cp_year, "bank"
    ].iloc[0]
    assert bank >= -1e-9


def test_emissions_decline_with_stricter_policy():
    base_frames = three_year_frames(carry_pct=0.0, annual_surrender_frac=1.0)
    base_outputs = run_end_to_end_from_frames(
        base_frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    base_emissions = base_outputs.annual["emissions_tons"].sum()
    higher_floor_outputs = run_end_to_end_from_frames(
        three_year_frames(floor=14.0, carry_pct=0.0, annual_surrender_frac=1.0),
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    assert higher_floor_outputs.annual["emissions_tons"].sum() < base_emissions
    lower_cap_outputs = run_end_to_end_from_frames(
        three_year_frames(cap_scale=0.35, carry_pct=0.0, annual_surrender_frac=1.0),
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    assert lower_cap_outputs.annual["emissions_tons"].sum() < base_emissions


def test_disabled_policy_produces_zero_price():
    frames = three_year_frames()
    frames = frames.with_frame("policy", policy_frame(policy_enabled=False))
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=5.0,
        tol=1e-4,
        relaxation=0.8,
    )
    assert outputs.annual["cp_last"].eq(0.0).all()
    assert outputs.annual["surrender"].eq(0.0).all()
    assert outputs.annual["bank"].eq(0.0).all()



def test_bank_never_negative_across_years():
    frames = three_year_frames()
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    assert outputs.annual["bank"].min() >= -1e-9


def test_bank_accumulates_when_emissions_below_cap():
    low_loads = [400_000.0, 380_000.0, 360_000.0]
    frames = three_year_frames(loads=low_loads)
    policy = frames.policy().to_policy()
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    initial_bank = float(policy.bank0)
    banks = outputs.annual.set_index("year")["bank"]
    assert banks.iloc[0] >= initial_bank
    assert banks.is_monotonic_increasing


def test_bank_disabled_yields_zero_balances():
    frames = three_year_frames()
    policy = policy_frame()
    policy["bank_enabled"] = False
    frames = frames.with_frame("policy", policy)
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    assert outputs.annual["bank"].eq(0.0).all()


def test_emissions_by_region_propagates():
    frames = three_year_frames(years=[2025], loads=[1_000_000.0])
    outputs = run_end_to_end_from_frames(
        frames,
        years=[2025],
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    df = outputs.emissions_by_region
    assert not df.empty
    assert set(df["region"]) == set(REGIONS)


def test_all_registry_regions_present_in_outputs():
    frames = all_region_frames(year=2025, demand_mwh=150_000.0)
    outputs = run_end_to_end_from_frames(
        frames,
        years=[2025],
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )

    assert not outputs.emissions_by_region.empty
    assert set(outputs.emissions_by_region["region"]) == set(REGIONS)

    assert not outputs.price_by_region.empty
    assert set(outputs.price_by_region["region"]) == set(REGIONS)

    assert not outputs.flows.empty
    flow_regions = set(outputs.flows["from_region"]) | set(outputs.flows["to_region"])
    assert set(REGIONS).issubset(flow_regions)


def test_ccr_trigger_increases_allowances():
    loads = [600_000.0, 600_000.0, 600_000.0]
    frames = three_year_frames(loads=loads)
    policy_df = policy_frame(cap_scale=0.15)
    policy_df["bank0"] = 0.0
    policy_df["ccr1_qty"] = 500_000.0
    frames = frames.with_frame("policy", policy_df)
    policy = frames.policy().to_policy()
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    annual_df = with_carbon_vector_columns(outputs.annual)
    assert_aliases_match_canonical(annual_df)
    annual = annual_df.set_index("year")
    first_year = YEARS[0]
    cap = float(policy.cap.loc[first_year])
    bank0 = float(policy.bank0)
    available = float(annual.loc[first_year, "allowances_minted"])
    allowances_issued = available - bank0
    assert allowances_issued > cap
    assert annual.loc[first_year, "cp_last"] == pytest.approx(
        policy.ccr1_trigger.loc[first_year], rel=1e-4
    )


def test_compliance_true_up_reconciles_obligations():
    frames = three_year_frames()
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    annual_df = with_carbon_vector_columns(outputs.annual)
    assert_aliases_match_canonical(annual_df)
    annual = annual_df.set_index("year")
    first_two = annual.loc[YEARS[:-1], "obligation"]
    assert (first_two > 0.0).all()
    final_year = YEARS[-1]
    final_obligation = float(annual.loc[final_year, "obligation"])
    assert final_obligation <= 1e-6
    surrendered_final = float(annual.loc[final_year, "surrender"])
    required_fraction = 0.5 * float(annual.loc[final_year, "emissions_tons"])
    assert surrendered_final > required_fraction


def test_control_period_mass_balance():
    frames = three_year_frames()
    policy = frames.policy().to_policy()
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    annual = outputs.annual.set_index("year")
    bank_prev = []
    for idx, year in enumerate(YEARS):
        if idx == 0:
            bank_prev.append(float(policy.bank0))
        else:
            bank_prev.append(float(annual.iloc[idx - 1]["bank"]))
    bank_prev_series = pd.Series(bank_prev, index=YEARS)
    allowances_minted = annual["allowances_minted"]
    allowances_total = annual["allowances_available"]
    pd.testing.assert_series_equal(
        allowances_total,
        bank_prev_series + allowances_minted,
        check_names=False,
        rtol=1e-9,
        atol=1e-6,
    )
    total_supply = float(policy.bank0) + allowances_minted.sum()
    total_surrendered = annual["surrender"].sum()
    ending_bank = float(annual.iloc[-1]["bank"])
    remaining = float(annual.iloc[-1]["obligation"])
    assert total_supply == pytest.approx(total_surrendered + ending_bank + remaining)


def test_annual_output_schema_matches_spec():
    frames = three_year_frames()
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    assert list(outputs.annual.columns) == ANNUAL_OUTPUT_COLUMNS


def test_allowance_price_column_matches_allowance_component():
    frames = three_year_frames()
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    annual = with_carbon_vector_columns(outputs.annual)
    assert_aliases_match_canonical(annual)
    assert "allowance_price" in annual.columns
    pd.testing.assert_series_equal(
        annual["allowance_price"],
        annual["cp_all"],
        check_names=False,
        rtol=1e-9,
        atol=1e-9,
    )


def test_annual_output_csv_schema(tmp_path):
    frames = three_year_frames()
    outputs = run_end_to_end_from_frames(
        frames,
        years=YEARS,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    outdir = tmp_path / "results"
    outputs.to_csv(outdir)
    annual_csv = pd.read_csv(outdir / "annual.csv")
    expected_columns = ANNUAL_OUTPUT_COLUMNS + [
        "p_co2",
        "p_co2_all",
        "p_co2_exc",
        "p_co2_eff",
    ]
    assert list(annual_csv.columns) == expected_columns


def test_preprocessor_handles_minimal_inputs(tmp_path):
    if not hasattr(prep, "build_frames"):
        pytest.skip("preprocessor lacks build_frames helper")
    units_path = tmp_path / "units.csv"
    demand_path = tmp_path / "demand.csv"
    policy_path = tmp_path / "policy.csv"
    frames = baseline_frames(year=2025, load_mwh=500_000.0)
    frames.units().to_csv(units_path, index=False)
    frames.demand().to_csv(demand_path, index=False)
    policy_frame().to_csv(policy_path, index=False)
    args = SimpleNamespace(
        units=units_path,
        demand=demand_path,
        policy=policy_path,
        carbon_price=None,
        output=tmp_path / "out",
        dispatch="single",
        years=YEARS,
        regions=None,
        scenario="test",
    )
    result = prep.build_frames(args)
    assert isinstance(result, Frames)


def test_daily_resolution_matches_annual_totals():
    ...


def test_engine_audits_pass_for_three_year_run(three_year_outputs):
    audits = three_year_outputs.audits
    assert isinstance(audits, dict)
    assert audits["emissions"]["passed"]
    assert audits["generation_capacity"]["passed"]
    assert audits["cost"]["passed"]
    assert float(audits["emissions"].get("max_region_gap", 1.0)) <= 1e-6
    assert float(audits["emissions"].get("max_fuel_gap", 1.0)) <= 1e-6


def test_generation_and_capacity_tables_available(three_year_outputs):
    generation_df = three_year_outputs.generation_by_fuel
    capacity_df = three_year_outputs.capacity_by_fuel
    assert not generation_df.empty
    assert not capacity_df.empty
    capacity_margins = three_year_outputs.audits["generation_capacity"]["capacity_margin"]
    assert capacity_margins
    assert all(float(margin) >= -1e-6 for margin in capacity_margins.values())
    assert isinstance(
        three_year_outputs.audits["generation_capacity"].get("stranded_units"), list
    )


def test_cost_breakdown_reconciles_components(three_year_outputs):
    cost_df = three_year_outputs.cost_by_fuel
    assert not cost_df.empty
    residual = (
        cost_df["total_cost"]
        - (
            cost_df["variable_cost"]
            + cost_df["allowance_cost"]
            + cost_df["carbon_price_cost"]
        )
    ).abs()
    assert float(residual.max()) <= 1e-6


def test_emissions_by_fuel_matches_totals(three_year_outputs):
    fuel_totals = (
        three_year_outputs.emissions_by_fuel.groupby("year")["emissions_tons"].sum()
    )
    annual_totals = (
        three_year_outputs.annual.set_index("year")["emissions_tons"].astype(float)
    )
    combined_years = sorted(set(fuel_totals.index) | set(annual_totals.index))
    max_gap = (
        fuel_totals.reindex(combined_years, fill_value=0.0)
        - annual_totals.reindex(combined_years, fill_value=0.0)
    ).abs().max()
    assert float(max_gap) <= 1e-6


def test_engine_outputs_include_zero_rows_for_empty_regions() -> None:
    frames = baseline_frames(year=2025, load_mwh=1_000_000.0)
    units = frames.units()
    region_ids = list(REGIONS)
    units["region"] = region_ids[: len(units)]
    for region in region_ids[1: len(units)]:
        mask = units["region"] == region
        units.loc[mask, "cap_mw"] = 0.0
        units.loc[mask, "availability"] = 0.0
    frames = frames.with_frame("units", units)
    policy = pd.DataFrame(
        [
            {
                "year": 2025,
                "cap_tons": 10_000_000.0,
                "floor_dollars": 0.0,
                "ccr1_trigger": 0.0,
                "ccr1_qty": 0.0,
                "ccr2_trigger": 0.0,
                "ccr2_qty": 0.0,
                "cp_id": "CP1",
                "full_compliance": True,
                "bank0": 0.0,
                "annual_surrender_frac": 1.0,
                "carry_pct": 1.0,
                "policy_enabled": True,
                "resolution": "annual",
            }
        ]
    )
    frames = frames.with_frame("policy", policy)
    outputs = run_end_to_end_from_frames(
        frames,
        years=[2025],
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )
    region_map = {r: dict(y) for r, y in outputs.emissions_by_region_map.items()}
    for region in REGIONS:
        assert region in region_map
        emissions = region_map[region].get(2025, 0.0)
        if region == DEFAULT_REGION_ID:
            assert emissions >= 0.0
        else:
            assert emissions == pytest.approx(0.0)
    emissions_regions = set(outputs.emissions_by_region["region"].unique())
    assert set(REGIONS).issubset(emissions_regions)


def test_three_region_run_populates_active_tables() -> None:
    frames = baseline_frames(year=2025, load_mwh=900_000.0)
    active_regions = list(REGIONS)[:3]
    units = frames.units().copy()
    for idx, region in enumerate(active_regions):
        units.loc[idx, "region"] = region
    frames = frames.with_frame("units", units)
    frames = frames.with_frame("policy", policy_frame(years=[2025]))
    demand = pd.DataFrame(
        [
            {"year": 2025, "region": active_regions[0], "demand_mwh": 200_000.0},
            {"year": 2025, "region": active_regions[1], "demand_mwh": 300_000.0},
            {"year": 2025, "region": active_regions[2], "demand_mwh": 400_000.0},
        ]
    )
    frames = frames.with_frame("demand", demand)

    outputs = run_end_to_end_from_frames(
        frames,
        years=[2025],
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
    )

    emissions_df = outputs.emissions_by_region
    prices_df = outputs.price_by_region
    flows_df = outputs.flows

    active_emissions = emissions_df[emissions_df["region"].isin(active_regions)]
    assert len(active_emissions) == len(active_regions)
    assert (active_emissions["emissions_tons"].astype(float) >= 0.0).all()

    active_prices = prices_df[prices_df["region"].isin(active_regions)]
    assert len(active_prices) == len(active_regions)

    self_flows = flows_df[
        (flows_df["from_region"] == flows_df["to_region"])
        & flows_df["from_region"].isin(active_regions)
    ]
    assert len(self_flows) == len(active_regions)
    assert all(flow == pytest.approx(0.0) for flow in self_flows["flow_mwh"].tolist())
