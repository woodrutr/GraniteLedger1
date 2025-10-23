"""Dataclass definitions for Streamlit module configuration records."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GeneralConfigResult:
    """Container for user-selected general configuration settings."""

    config_label: str
    config_source: Any
    run_config: dict[str, Any]
    candidate_years: list[int]
    start_year: int
    end_year: int
    selected_years: list[int]
    regions: list[int | str]
    preset_key: str | None = None
    preset_label: str | None = None
    lock_carbon_controls: bool = False
    max_iterations: int = 0
    fuel_price_selection: dict[str, str] = field(default_factory=dict)


@dataclass
class CarbonModuleSettings:
    """Record of carbon policy sidebar selections."""

    enabled: bool
    price_enabled: bool
    enable_floor: bool
    enable_ccr: bool
    ccr1_enabled: bool
    ccr2_enabled: bool
    ccr1_price: float | None
    ccr2_price: float | None
    ccr1_escalator_pct: float
    ccr2_escalator_pct: float
    banking_enabled: bool
    coverage_regions: list[str]
    control_period_years: int | None
    price_per_ton: float
    price_escalator_pct: float = 0.0
    initial_bank: float = 0.0
    cap_regions: list[Any] = field(default_factory=list)
    cap_start_value: float | None = None
    cap_reduction_mode: str = "percent"
    cap_reduction_value: float = 0.0
    cap_schedule: dict[int, float] = field(default_factory=dict)
    floor_value: float = 0.0
    floor_escalator_mode: str = "fixed"
    floor_escalator_value: float = 0.0
    floor_schedule: dict[int, float] = field(default_factory=dict)
    price_schedule: dict[int, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class DispatchModuleSettings:
    """Record of electricity dispatch sidebar selections."""

    enabled: bool
    mode: str
    capacity_expansion: bool
    reserve_margins: bool
    deep_carbon_pricing: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class IncentivesModuleSettings:
    """Record of incentives sidebar selections."""

    enabled: bool
    production_credits: list[dict[str, Any]]
    investment_credits: list[dict[str, Any]]
    errors: list[str] = field(default_factory=list)


@dataclass
class DemandModuleSettings:
    """Record of user-selected demand curve assignments by region."""

    enabled: bool
    curve_by_region: dict[str, str | None]
    forecast_by_region: dict[str, str | None] = field(default_factory=dict)
    load_forecasts: dict[str, str | None] = field(default_factory=dict)
    custom_load_forecasts: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class OutputsModuleSettings:
    """Record of outputs sidebar selections."""

    enabled: bool
    directory: str
    resolved_path: Path
    show_csv_downloads: bool
    errors: list[str] = field(default_factory=list)


__all__ = [
    "CarbonModuleSettings",
    "DemandModuleSettings",
    "DispatchModuleSettings",
    "GeneralConfigResult",
    "IncentivesModuleSettings",
    "OutputsModuleSettings",
]

