"""Tests for the :mod:`granite_io.frames_api` helpers."""

from __future__ import annotations

import importlib
import pytest

pd = pytest.importorskip("pandas")

run_fixed_point_from_frames = importlib.import_module(
    "engine.run_loop"
).run_fixed_point_from_frames
Frames = importlib.import_module("io_loader").Frames
ConfigError = importlib.import_module("policy.allowance_annual").ConfigError
policy_frame_three_year = importlib.import_module(
    "tests.fixtures.annual_minimal"
).policy_frame_three_year
fixtures = importlib.import_module("tests.fixtures.dispatch_single_minimal")
baseline_frames = fixtures.baseline_frames
DEFAULT_REGION_ID = fixtures.DEFAULT_REGION_ID
REGION_IDS = list(importlib.import_module("regions.registry").REGIONS)
REGION_A = REGION_IDS[0]
REGION_B = REGION_IDS[1]
REGION_C = REGION_IDS[2]


def test_legacy_io_import_resolves_frames_module() -> None:
    """The legacy ``io.frames_api`` import should point to the real module."""

    granite_module = importlib.import_module("granite_io.frames_api")
    legacy_module = importlib.import_module("io.frames_api")

    assert legacy_module is granite_module
    assert getattr(importlib.import_module("io"), "frames_api") is granite_module


def test_demand_validation_requires_unique_pairs() -> None:
    """Duplicate region-year entries should raise a clear validation error."""

    frames = Frames(
        {
            "demand": pd.DataFrame(
                [
                    {"year": 2030, "region": REGION_A, "demand_mwh": 10.0},
                    {"year": 2030, "region": REGION_A, "demand_mwh": 12.0},
                ]
            )
        }
    )

    with pytest.raises(ValueError, match="duplicate year/region"):  # demand validation
        frames.demand()


def test_peak_demand_validation_and_lookup() -> None:
    peak = pd.DataFrame(
        [
            {"year": 2030, "region": REGION_A, "peak_demand_mw": 100.0},
            {"year": 2030, "region": REGION_A, "peak_demand_mw": 105.0},
            {"year": 2031, "region": REGION_B, "peak_demand_mw": 90.0},
        ]
    )

    frames = Frames({"peak_demand": peak})

    with pytest.raises(ValueError, match="duplicate year/region"):
        frames.peak_demand()

    unique_peak = peak.drop_duplicates(subset=["year", "region"], keep="last")
    frames = Frames({"peak_demand": unique_peak})
    result = frames.peak_demand()
    assert result.shape[0] == 2
    by_year = frames.peak_demand_for_year(2031)
    assert by_year == {REGION_B: 90.0}


def test_region_coverage_defaults_and_overrides() -> None:
    """Coverage defaults should apply unless a year-specific override exists."""

    coverage = pd.DataFrame(
        [
            {"region": REGION_A, "covered": True},
            {"region": REGION_B, "covered": False},
            {"region": REGION_B, "year": 2035, "covered": True},
        ]
    )

    frames = Frames({"coverage": coverage})

    default = frames.coverage_for_year(2030)
    override = frames.coverage_for_year(2035)

    assert default[REGION_A] is True
    assert default[REGION_B] is False
    assert override[REGION_B] is True


def test_boolean_columns_accept_common_tokens() -> None:
    """Boolean validation should understand mixed types and strings."""

    fuels = pd.DataFrame(
        [
            {"fuel": "gas", "covered": "true"},
            {"fuel": "coal", "covered": "0"},
            {"fuel": "wind", "covered": 1},
        ]
    )

    coverage = pd.DataFrame(
        [
            {"region": REGION_A, "covered": "false"},
            {"region": REGION_B, "covered": 1, "year": 2030},
            {"region": REGION_C, "covered": 0, "year": 2031},
        ]
    )

    policy = pd.DataFrame(
        [
            {
                "year": 2025,
                "cap_tons": 100.0,
                "floor_dollars": 3.0,
                "ccr1_trigger": 1.0,
                "ccr1_qty": 1.0,
                "ccr2_trigger": 2.0,
                "ccr2_qty": 2.0,
                "cp_id": "A",
                "full_compliance": "yes",
                "bank0": 0.0,
                "annual_surrender_frac": 0.5,
                "carry_pct": 1.0,
            },
            {
                "year": 2026,
                "cap_tons": 110.0,
                "floor_dollars": 3.0,
                "ccr1_trigger": 1.1,
                "ccr1_qty": 1.1,
                "ccr2_trigger": 2.1,
                "ccr2_qty": 2.1,
                "cp_id": "B",
                "full_compliance": 0,
                "bank0": 0.0,
                "annual_surrender_frac": 0.5,
                "carry_pct": 1.0,
            },
        ]
    )

    frames = Frames({"fuels": fuels, "coverage": coverage, "policy": policy})

    fuels_result = frames.fuels()
    assert fuels_result["covered"].dtype == bool
    assert fuels_result.set_index("fuel")["covered"].to_dict() == {
        "gas": True,
        "coal": False,
        "wind": True,
    }

    coverage_result = frames.coverage()
    assert coverage_result["covered"].dtype == bool
    coverage_lookup = {
        (row.region, row.year): row.covered for row in coverage_result.itertuples(index=False)
    }
    assert coverage_lookup[(REGION_A, -1)] is False
    assert coverage_lookup[(REGION_B, 2030)] is True
    assert coverage_lookup[(REGION_C, 2031)] is False

    policy_spec = frames.policy()
    assert policy_spec.full_compliance_years == {2025}


def test_boolean_columns_reject_invalid_tokens() -> None:
    """Unknown boolean tokens should raise a validation error."""

    frames = Frames(
        {
            "fuels": pd.DataFrame(
                [
                    {"fuel": "gas", "covered": "maybe"},
                ]
            )
        }
    )

    with pytest.raises(ValueError, match="boolean-like values"):
        frames.fuels()


def test_build_frames_populates_demand_from_load_forecasts() -> None:
    """Load forecast selections should materialize demand rows for the engine."""

    from granite_io.frames_api import build_frames

    selection = {"load": {"isos": ["iso_ne"], "scenario": "celt_2025_baseline"}}

    frames = build_frames("input", selection)

    load_frame = frames.frame("load")
    demand_frame = frames.demand()

    assert not load_frame.empty
    assert not demand_frame.empty

    # Ensure the demand frame matches the normalized zone identifiers and magnitudes.
    ct_load = load_frame[load_frame["zone"] == "ISO_NE_CT"].iloc[0]
    ct_demand = demand_frame[demand_frame["region"] == "ISO-NE_CT"].iloc[0]
    assert ct_demand["year"] == int(ct_load["year"])
    assert ct_demand["demand_mwh"] == pytest.approx(float(ct_load["load_gwh"]) * 1_000.0)


def _unit_row(
    *,
    unit_id: str,
    region: str,
    fuel: str,
    cap_mw: float = 100.0,
    availability: float = 0.9,
    heat_rate: float = 7.5,
    vom: float = 2.0,
    fuel_price: float = 3.0,
    emission_rate: float = 0.5,
) -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "unique_id": unit_id,
        "region": region,
        "fuel": fuel,
        "cap_mw": cap_mw,
        "availability": availability,
        "hr_mmbtu_per_mwh": heat_rate,
        "vom_per_mwh": vom,
        "fuel_price_per_mmbtu": fuel_price,
        "ef_ton_per_mwh": emission_rate,
    }


def test_units_require_inventory_for_each_demand_region() -> None:
    """Every demand region must have at least one generating unit."""

    demand = pd.DataFrame(
        [
            {"year": 2025, "region": REGION_A, "demand_mwh": 1000.0},
            {"year": 2025, "region": REGION_B, "demand_mwh": 1200.0},
        ]
    )

    units = pd.DataFrame(
        [
            _unit_row(unit_id="a", region=REGION_A, fuel="gas"),
        ]
    )

    frames = Frames({"demand": demand, "units": units})

    with pytest.raises(ValueError, match="unit inventory for active regions"):
        frames.units()


def test_units_require_fossil_joined_costs_and_emissions() -> None:
    """Fossil units must provide non-zero fuel prices and emission factors."""

    demand = pd.DataFrame([
        {"year": 2025, "region": REGION_A, "demand_mwh": 1500.0},
    ])

    base_unit = _unit_row(unit_id="gas-1", region=REGION_A, fuel="gas")

    fuels = pd.DataFrame([
        {"fuel": "gas", "covered": True},
        {"fuel": "wind", "covered": False},
    ])

    frames_missing_cost = Frames(
        {
            "demand": demand,
            "units": pd.DataFrame([{**base_unit, "fuel_price_per_mmbtu": 0.0}]),
            "fuels": fuels,
        }
    )

    with pytest.raises(ValueError, match="fuel price data for fossil units"):
        frames_missing_cost.units()

    frames_missing_ef = Frames(
        {
            "demand": demand,
            "units": pd.DataFrame([{**base_unit, "ef_ton_per_mwh": 0.0}]),
            "fuels": fuels,
        }
    )

    with pytest.raises(ValueError, match="emission factors for fossil units"):
        frames_missing_ef.units()


def test_frame_helper_methods_provide_copies_and_defaults() -> None:
    """Ensure helper accessors return defensive copies and optional defaults."""

    base = pd.DataFrame({"value": [1.0, 2.0]})
    frames = Frames({"Example": base})

    assert frames.has_frame("example") is True

    retrieved = frames.frame("EXAMPLE")
    assert retrieved.equals(base)
    assert retrieved is not base

    optional_existing = frames.optional_frame("example")
    assert optional_existing.equals(base)
    assert optional_existing is not base

    assert frames.optional_frame("missing") is None

    default_df = pd.DataFrame({"value": []})
    assert frames.optional_frame("missing", default=default_df) is default_df


def test_policy_spec_round_trip() -> None:
    """The policy accessor should convert to an :class:`RGGIPolicyAnnual`."""

    frames = Frames({"policy": policy_frame_three_year()})

    spec = frames.policy()
    policy = spec.to_policy()

    assert policy.bank0 == pytest.approx(10.0)
    assert policy.annual_surrender_frac == pytest.approx(0.5)
    assert policy.carry_pct == pytest.approx(1.0)
    assert policy.full_compliance_years == {2027}
    assert list(policy.cap.index) == [2025, 2026, 2027]
    assert policy.enabled is True
    assert policy.ccr1_enabled is True
    assert policy.ccr2_enabled is True
    assert policy.control_period_length is None
    assert policy.banking_enabled is True
    assert spec.resolution == "annual"
    assert policy.resolution == "annual"


def test_policy_enabled_requires_required_columns() -> None:
    """Enabled carbon policy should fail when required columns are absent."""

    policy = pd.DataFrame(
        [
            {
                "year": 2025,
                "cap_tons": 100.0,
                "policy_enabled": True,
            }
        ]
    )

    frames = Frames({"policy": policy})

    with pytest.raises(ConfigError, match="requires columns"):
        frames.policy()


def test_missing_policy_frame_raises_config_error() -> None:
    """Carbon-enabled runs must provide a policy frame."""

    frames = Frames({}, carbon_policy_enabled=True)

    with pytest.raises(ConfigError, match="requires a 'policy' frame"):
        frames.policy()


def test_fixed_point_runs_from_frames() -> None:
    """The engine should operate directly on in-memory frame data."""

    frames = baseline_frames(year=2025, load_mwh=1_000_000.0)
    demand = pd.DataFrame(
        [
            {"year": 2025, "region": DEFAULT_REGION_ID, "demand_mwh": 1_200_000.0},
            {"year": 2026, "region": DEFAULT_REGION_ID, "demand_mwh": 1_050_000.0},
            {"year": 2027, "region": DEFAULT_REGION_ID, "demand_mwh": 900_000.0},
        ]
    )
    frames = frames.with_frame("demand", demand)
    frames = frames.with_frame("policy", policy_frame_three_year())

    results = run_fixed_point_from_frames(
        frames,
        years=[2025, 2026, 2027],
        price_initial=0.0,
        tol=1e-4,
    )

    assert set(results) == {2025, 2026, 2027}
    assert all("emissions" in year_result for year_result in results.values())
    assert results[2027]["finalize"]["finalized"]


def test_policy_spec_respects_optional_columns() -> None:
    base = policy_frame_three_year()
    base['full_compliance'] = [False, False, False]
    base['policy_enabled'] = [False, False, False]
    base['ccr1_enabled'] = [False, False, False]
    base['ccr2_enabled'] = [True, True, True]
    base['control_period_years'] = [2, 2, 2]
    base['bank_enabled'] = [False, False, False]

    frames = Frames({"policy": base})
    policy = frames.policy().to_policy()

    assert policy.enabled is False
    assert policy.ccr1_enabled is False
    assert policy.ccr2_enabled is True
    assert policy.control_period_length == 2
    assert policy.full_compliance_years == {2026}
    assert policy.banking_enabled is False


def test_policy_disabled_allows_minimal_columns() -> None:
    frames = Frames(
        {
            "policy": pd.DataFrame(
                [
                    {"year": 2025, "policy_enabled": False},
                    {"year": 2026, "policy_enabled": False},
                ]
            )
        }
    )

    spec = frames.policy()
    policy = spec.to_policy()

    assert spec.enabled is False
    assert policy.enabled is False
    assert list(policy.cap.index) == [2025, 2026]
    assert policy.cap.eq(0.0).all()
    assert policy.bank0 == pytest.approx(0.0)
    assert policy.banking_enabled is False
