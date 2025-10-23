from __future__ import annotations

import pytest

from gui.app import (
    CarbonPriceConfig,
    _build_cap_reduction_schedule,
    _build_price_schedule,
    _build_price_escalator_schedule,
    _merge_price_schedules,
    _normalize_price_schedule,
)


def test_normalize_price_schedule_handles_malformed_entries() -> None:
    schedule = {
        '2024': '25.5',
        '2022': None,
        2023.0: '10',
        'bad': '4',
        2021: '',
        2020.5: 7.25,
    }

    normalized = _normalize_price_schedule(schedule)

    assert normalized == {2020: 7.25, 2023: 10.0, 2024: 25.5}
    assert list(normalized) == [2020, 2023, 2024]


def test_merge_price_schedules_overrides_and_sorts() -> None:
    base = {'2025': '5', '2024': '3'}
    override = {2026: 7, '2024': '4'}

    merged = _merge_price_schedules(base, override)

    assert merged == {2024: 4.0, 2025: 5.0, 2026: 7.0}
    assert list(merged) == [2024, 2025, 2026]


def test_carbon_price_config_builds_sorted_schedule_from_years() -> None:
    config = CarbonPriceConfig.from_mapping(
        {},
        enabled=True,
        value=12.5,
        schedule=None,
        years=[2023, '2021', 'invalid', 2022.0],
    )

    assert config.schedule == {2021: 12.5, 2022: 12.5, 2023: 12.5}
    assert list(config.schedule) == [2021, 2022, 2023]
    assert config.escalator_pct == 0.0


def test_carbon_price_config_applies_escalator_growth() -> None:
    config = CarbonPriceConfig.from_mapping(
        {},
        enabled=True,
        value=10.0,
        schedule=None,
        years=[2020, 2021, 2022],
        escalator_pct=10.0,
    )

    assert config.escalator_pct == 10.0
    assert config.schedule[2020] == pytest.approx(10.0)
    assert config.schedule[2021] == pytest.approx(11.0)
    assert config.schedule[2022] == pytest.approx(12.1)


def test_build_price_escalator_schedule_handles_empty_years() -> None:
    schedule = _build_price_escalator_schedule(20.0, 5.0, [])

    assert schedule == {}


def test_build_price_escalator_schedule_handles_unsorted_years() -> None:
    schedule = _build_price_escalator_schedule(10.0, 5.0, [2025, 2023, 2024])

    assert list(schedule) == [2023, 2024, 2025]
    assert schedule[2023] == pytest.approx(10.0)
    assert schedule[2025] == pytest.approx(11.025)


def test_build_price_escalator_schedule_fills_gaps() -> None:
    schedule = _build_price_escalator_schedule(15.0, 10.0, [2025, 2030])

    assert list(schedule) == [2025, 2026, 2027, 2028, 2029, 2030]
    assert schedule[2025] == pytest.approx(15.0)
    assert schedule[2030] == pytest.approx(24.15765)


def test_build_price_schedule_generates_geometric_growth() -> None:
    schedule = _build_price_schedule(2025, 2030, 45.0, 7.0)

    assert list(schedule) == [2025, 2026, 2027, 2028, 2029, 2030]
    assert schedule[2025] == pytest.approx(45.0)
    assert schedule[2026] == pytest.approx(48.15)
    assert schedule[2027] == pytest.approx(51.5205)
    assert schedule[2028] == pytest.approx(55.126935)
    assert schedule[2029] == pytest.approx(58.98582)
    assert schedule[2030] == pytest.approx(63.114828)


def test_build_price_schedule_supports_descending_ranges() -> None:
    schedule = _build_price_schedule(2030, 2025, 50.0, 5.0)

    assert list(schedule) == [2025, 2026, 2027, 2028, 2029, 2030]
    assert schedule[2030] == pytest.approx(50.0)
    assert schedule[2025] == pytest.approx(63.814078)


def test_build_cap_reduction_schedule_percent_and_fixed() -> None:
    years = [2025, 2026, 2027]
    percent_schedule = _build_cap_reduction_schedule(100.0, "percent", 10.0, years)
    fixed_schedule = _build_cap_reduction_schedule(100.0, "fixed", 5.0, years)

    assert percent_schedule == {2025: 100.0, 2026: 90.0, 2027: 80.0}
    assert fixed_schedule == {2025: 100.0, 2026: 95.0, 2027: 90.0}


def test_carbon_price_config_fills_missing_years() -> None:
    config = CarbonPriceConfig.from_mapping(
        {},
        enabled=True,
        value=20.0,
        schedule=None,
        years=[2025, 2030],
        escalator_pct=5.0,
    )

    assert list(config.schedule) == [2025, 2026, 2027, 2028, 2029, 2030]
    assert config.schedule[2025] == pytest.approx(20.0)
    assert config.schedule[2026] == pytest.approx(21.0)
    assert config.schedule[2030] == pytest.approx(25.525631)
