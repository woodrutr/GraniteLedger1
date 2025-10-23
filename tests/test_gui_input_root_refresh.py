"""Tests for refreshing GUI forecast caches when the input root changes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import engine.settings as engine_settings
from gui import app as gui_app
from gui import helpers as gui_helpers


class _DummyInputRoot:
    """Simple stub mirroring the engine ``input_root`` contract."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def __call__(self) -> Path:
        return self._path

    def set_path(self, path: Path) -> None:
        self._path = Path(path)

    def cache_clear(self) -> None:  # noqa: D401 - mimic functools.lru_cache API
        """Compatibility no-op for ``functools.lru_cache.cache_clear``."""


def test_forecast_caches_refresh_with_new_input_root(monkeypatch, tmp_path):
    """Changing ``input_root`` should refresh cached forecast data."""

    gui_app._cached_forecast_frame.cache_clear()
    gui_app._cached_iso_scenario_map.cache_clear()
    gui_app._cached_input_root.cache_clear()

    initial_root = tmp_path / "initial"
    updated_root = tmp_path / "updated"
    initial_root.mkdir()
    updated_root.mkdir()

    dummy_root = _DummyInputRoot(initial_root)
    monkeypatch.setattr(engine_settings, "input_root", dummy_root)
    monkeypatch.setattr(gui_app, "input_root", dummy_root)

    load_calls: list[str | None] = []

    def _fake_load_forecasts_frame(*, base_path: str | None):
        load_calls.append(base_path)
        return pd.DataFrame({"iso": [], "scenario": []})

    monkeypatch.setattr(gui_app, "_regions_load_forecasts_frame", _fake_load_forecasts_frame)

    initial_base_path = gui_app._resolve_forecast_base_path()
    gui_app._cached_forecast_frame(initial_base_path)

    assert load_calls == [initial_base_path]

    dummy_root.set_path(updated_root)

    resolved_base_path = gui_app._resolve_forecast_base_path()
    gui_app._cached_forecast_frame(resolved_base_path)

    assert resolved_base_path == str(updated_root)
    assert load_calls == [initial_base_path, resolved_base_path]

    gui_app._cached_forecast_frame.cache_clear()
    gui_app._cached_iso_scenario_map.cache_clear()
    gui_app._cached_input_root.cache_clear()


def test_helpers_root_reloads_environment_changes(monkeypatch, tmp_path):
    """``gui.helpers._root`` should re-resolve when the env var changes."""

    engine_settings.input_root.cache_clear()

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()

    monkeypatch.setenv("GRANITELEDGER_INPUT_ROOT", str(first_root))

    resolved_first = gui_helpers._root()
    assert resolved_first == first_root.resolve()

    monkeypatch.setenv("GRANITELEDGER_INPUT_ROOT", str(second_root))

    resolved_second = gui_helpers._root()
    assert resolved_second == second_root.resolve()

    engine_settings.input_root.cache_clear()
