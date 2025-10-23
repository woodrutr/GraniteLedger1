from __future__ import annotations

import importlib
import pytest

pd = pytest.importorskip("pandas")

policy_loader = importlib.import_module("config.policy_loader")
load_annual_policy = policy_loader.load_annual_policy
series_from_year_map = policy_loader.series_from_year_map
normalize_policy_zones = getattr(policy_loader, "_normalize_policy_zones")
ConfigError = importlib.import_module("policy.allowance_annual").ConfigError


def _sample_state_index() -> dict[str, set[str]]:
    return {
        "CT": {"ISO-NE_CT"},
        "MA": {"ISO-NE_MA"},
        "NY": {
            "NYISO_A",
            "NYISO_B",
            "NYISO_C",
            "NYISO_D",
            "NYISO_E",
            "NYISO_F",
            "NYISO_G",
            "NYISO_H",
            "NYISO_I",
            "NYISO_J",
            "NYISO_K",
        },
    }


def test_normalize_policy_zones_expands_covered_states(monkeypatch):
    monkeypatch.setattr(policy_loader, "STATE_INDEX", _sample_state_index(), raising=False)
    cfg = {"covered_states": ["ct", "MA"], "covered_zones": ["PJM_PSEG"]}

    with pytest.warns(DeprecationWarning):
        normalize_policy_zones(cfg)

    assert "covered_zones" in cfg
    assert list(cfg["covered_zones"].keys()) == ["ISO-NE_CT", "ISO-NE_MA", "PJM_PSEG"]
    assert all(cfg["covered_zones"].values())


def test_normalize_policy_zones_preserves_existing_flags(monkeypatch):
    monkeypatch.setattr(policy_loader, "STATE_INDEX", _sample_state_index(), raising=False)
    cfg = {"covered_states": ["CT"], "covered_zones": {"PJM_PSEG": False, "NYISO_J": True}}

    with pytest.warns(DeprecationWarning):
        normalize_policy_zones(cfg)

    assert cfg["covered_zones"]["PJM_PSEG"] is False
    assert cfg["covered_zones"]["NYISO_J"] is True
    assert cfg["covered_zones"]["ISO-NE_CT"] is True
    assert list(cfg["covered_zones"].keys()) == ["ISO-NE_CT", "NYISO_J", "PJM_PSEG"]


def test_normalize_policy_zones_mapping_filters_falsey_states(monkeypatch):
    monkeypatch.setattr(policy_loader, "STATE_INDEX", _sample_state_index(), raising=False)
    cfg = {"covered_states": {"NY": True, "PA": False}}

    with pytest.warns(DeprecationWarning):
        normalize_policy_zones(cfg)

    zones = cfg["covered_zones"]
    assert all(zones.values())
    assert set(zones) == {"NYISO_A", "NYISO_B", "NYISO_C", "NYISO_D", "NYISO_E", "NYISO_F", "NYISO_G", "NYISO_H", "NYISO_I", "NYISO_J", "NYISO_K"}


def test_normalize_policy_zones_unknown_state_raises(monkeypatch):
    monkeypatch.setattr(policy_loader, "STATE_INDEX", _sample_state_index(), raising=False)
    cfg = {"covered_states": ["ZZ"]}

    with pytest.raises(ValueError):
        with pytest.warns(DeprecationWarning):
            normalize_policy_zones(cfg)


def test_series_from_year_map_basic():
    cfg = {
        "years": [2027, 2026, 2025],
        "cap": {2025: 100, 2026: 90, 2027: 80},
    }

    series = series_from_year_map(cfg, "cap")

    assert isinstance(series, pd.Series)
    assert list(series.index) == [2025, 2026, 2027]
    assert series.loc[2026] == pytest.approx(90.0)
    assert not series.attrs.get("fill_forward", False)


def test_series_from_year_map_fill_forward_floor():
    cfg = {
        "years": [2025, 2026, 2027],
        "floor": {"values": {2025: 4.0, 2027: 5.0}, "fill": "forward"},
    }

    series = series_from_year_map(cfg, "floor")

    assert list(series.index) == [2025, 2026, 2027]
    assert series.loc[2026] == pytest.approx(4.0)
    assert series.loc[2027] == pytest.approx(5.0)
    assert series.attrs["fill_forward"] is True


def test_series_from_year_map_fill_forward_not_allowed_for_other_keys():
    cfg = {
        "years": [2025, 2026],
        "ccr1_trigger": {"values": {2025: 7.0}, "fill": "forward"},
    }

    with pytest.raises(ValueError) as exc:
        series_from_year_map(cfg, "ccr1_trigger")

    assert "ccr1_trigger" in str(exc.value)
    assert "Fill-forward" in str(exc.value)


def test_series_from_year_map_reports_missing_years():
    cfg = {
        "years": [2025, 2026, 2027],
        "cap": {2025: 100.0, 2027: 80.0},
    }

    with pytest.raises(ValueError) as exc:
        series_from_year_map(cfg, "cap")

    message = str(exc.value)
    assert "cap" in message
    assert "2026" in message


def test_load_annual_policy_builds_series_and_validates_alignment():
    cfg = {
        "years": [2025, 2026, 2027],
        "cap": {2025: 100.0, 2026: 90.0, 2027: 95.0},
        "floor": {"values": {2025: 4.0}, "fill_forward": True},
        "ccr1_trigger": {2025: 7.0, 2026: 7.0, 2027: 7.0},
        "ccr1_qty": {2025: 30.0, 2026: 30.0, 2027: 30.0},
        "ccr2_trigger": {2025: 13.0, 2026: 13.0, 2027: 13.0},
        "ccr2_qty": {2025: 60.0, 2026: 60.0, 2027: 60.0},
        "cp_id": {2025: "CP1", 2026: "CP1", 2027: "CP1"},
        "bank0": 15.0,
        "full_compliance_years": [2027],
        "annual_surrender_frac": 0.5,
        "carry_pct": 1.0,
    }

    policy = load_annual_policy(cfg)

    expected_years = [2025, 2026, 2027]
    for series in [
        policy.cap,
        policy.floor,
        policy.ccr1_trigger,
        policy.ccr1_qty,
        policy.ccr2_trigger,
        policy.ccr2_qty,
    ]:
        assert list(series.index) == expected_years

    assert policy.floor.loc[2026] == pytest.approx(4.0)
    assert policy.floor.loc[2027] == pytest.approx(4.0)
    assert policy.cap.loc[2025] == pytest.approx(100.0)
    assert policy.ccr2_qty.loc[2027] == pytest.approx(60.0)
    assert policy.bank0 == pytest.approx(15.0)
    assert policy.full_compliance_years == {2027}
    assert policy.enabled is True
    assert policy.ccr1_enabled is True
    assert policy.ccr2_enabled is True
    assert policy.control_period_length is None
    assert policy.resolution == "annual"


def test_load_annual_policy_flags_missing_series_years():
    cfg = {
        "years": [2025, 2026, 2027],
        "cap": {2025: 100.0, 2026: 90.0, 2027: 95.0},
        "floor": {2025: 4.0, 2026: 4.0, 2027: 4.0},
        "ccr1_trigger": {2025: 7.0, 2026: 7.0, 2027: 7.0},
        "ccr1_qty": {2025: 30.0, 2026: 30.0, 2027: 30.0},
        "ccr2_trigger": {2025: 13.0, 2026: 13.0, 2027: 13.0},
        "ccr2_qty": {2025: 60.0, 2027: 60.0},
        "cp_id": {2025: "CP1", 2026: "CP1", 2027: "CP1"},
    }

    with pytest.raises(ValueError) as exc:
        load_annual_policy(cfg)

    assert "ccr2_qty" in str(exc.value)
    assert "2026" in str(exc.value)


def test_load_annual_policy_respects_module_flags():
    cfg = {
        "years": [2025, 2026],
        "cap": {2025: 100.0, 2026: 90.0},
        "floor": {2025: 4.0, 2026: 4.0},
        "ccr1_trigger": {2025: 7.0, 2026: 7.0},
        "ccr1_qty": {2025: 30.0, 2026: 30.0},
        "ccr2_trigger": {2025: 13.0, 2026: 13.0},
        "ccr2_qty": {2025: 60.0, 2026: 60.0},
        "cp_id": {2025: "CP1", 2026: "CP1"},
        "bank0": 0.0,
        "annual_surrender_frac": 1.0,
        "carry_pct": 1.0,
        "full_compliance_years": [],
        "enabled": False,
        "ccr1_enabled": False,
        "ccr2_enabled": True,
        "control_period_years": 2,
    }

    policy = load_annual_policy(cfg)

    assert policy.enabled is False
    assert policy.ccr1_enabled is False
    assert policy.ccr2_enabled is True
    assert policy.control_period_length == 2
    assert policy.full_compliance_years == {2026}


def test_enabled_policy_requires_cap_schedule():
    cfg = {"enabled": True}

    with pytest.raises(ConfigError) as exc:
        load_annual_policy(cfg)

    assert "cap" in str(exc.value).lower()


def test_disabled_policy_allows_missing_inputs():
    policy = load_annual_policy({"enabled": False})

    assert policy.enabled is False
    assert len(policy.cap) == 0
    assert policy.ccr1_enabled is True
    assert policy.ccr2_enabled is True
    assert policy.resolution == "annual"


def test_load_annual_policy_supports_daily_resolution():
    cfg = {
        "years": [2025],
        "cap": {2025: 50.0},
        "floor": {2025: 2.0},
        "ccr1_trigger": {2025: 5.0},
        "ccr1_qty": {2025: 10.0},
        "ccr2_trigger": {2025: 9.0},
        "ccr2_qty": {2025: 20.0},
        "cp_id": {2025: "CP"},
        "bank0": 5.0,
        "annual_surrender_frac": 0.5,
        "carry_pct": 1.0,
        "resolution": "daily",
    }

    policy = load_annual_policy(cfg)

    assert policy.resolution == "daily"


def test_daily_policy_compliance_year_mapping():
    cfg = {
        "years": [2025001, 2025002],
        "cap": {2025001: 80.0, 2025002: 75.0},
        "floor": {2025001: 2.0, 2025002: 2.5},
        "ccr1_trigger": {2025001: 5.0, 2025002: 5.0},
        "ccr1_qty": {2025001: 10.0, 2025002: 10.0},
        "ccr2_trigger": {2025001: 9.0, 2025002: 9.0},
        "ccr2_qty": {2025001: 15.0, 2025002: 15.0},
        "cp_id": {2025001: "CP", 2025002: "CP"},
        "bank0": 0.0,
        "annual_surrender_frac": 1.0,
        "carry_pct": 1.0,
        "resolution": "daily",
    }

    policy = load_annual_policy(cfg)

    assert policy.compliance_year_for(2025001) == 2025
    assert policy.compliance_year_for("2025002") == 2025
    assert policy.compliance_year_for(2025) == 2025
    assert policy.compliance_year_for("2025") == 2025
