"""Tests for the :mod:`engine.data_loaders` re-export facade."""

from __future__ import annotations

import importlib


def _import_data_loaders():
    """Return a freshly imported ``engine.data_loaders`` module."""

    return importlib.import_module("engine.data_loaders")


def test_forecast_exports_available():
    data_loaders = _import_data_loaders()
    assert not hasattr(data_loaders, "load_demand_forecasts")
    assert hasattr(data_loaders, "load_iso_scenario_bundle")
    assert hasattr(data_loaders, "available_iso_scenarios")


def test_unit_and_transmission_exports_available():
    data_loaders = _import_data_loaders()
    assert hasattr(data_loaders, "load_unit_fleet")
    assert hasattr(data_loaders, "derive_fuels")
    assert hasattr(data_loaders, "build_transmission_topology")
