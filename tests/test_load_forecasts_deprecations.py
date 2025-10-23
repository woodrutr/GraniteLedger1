"""Guards against resurrecting deprecated demand loader entrypoints."""

from __future__ import annotations

import importlib
import inspect


def test_load_demand_forecasts_removed() -> None:
    mod = importlib.import_module("engine.data_loaders.load_forecasts")
    members = {name for name, _ in inspect.getmembers(mod)}
    assert "load_demand_forecasts" not in members
