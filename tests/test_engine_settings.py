"""Tests for ``engine.settings`` helpers."""

from __future__ import annotations

from pathlib import Path

from engine import settings


def test_input_root_discovers_nested_directory(tmp_path, monkeypatch):
    """``input_root`` should locate data inside a nested project directory."""

    fake_settings_path = tmp_path / "package" / "engine" / "settings.py"
    fake_settings_path.parent.mkdir(parents=True)
    fake_settings_path.write_text("", encoding="utf-8")

    nested = tmp_path / "GraniteLedger-main" / "input" / "electricity" / "load_forecasts"
    nested.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GRANITELEDGER_INPUT_ROOT", raising=False)
    monkeypatch.setattr(settings, "__file__", str(fake_settings_path))
    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path / "package")

    settings.input_root.cache_clear()
    try:
        discovered = settings.input_root()
    finally:
        settings.input_root.cache_clear()

    assert discovered == nested.resolve()


def test_input_root_env_parent_directory(tmp_path, monkeypatch):
    """Environment overrides should tolerate parent directories."""

    fake_settings_path = tmp_path / "package" / "engine" / "settings.py"
    fake_settings_path.parent.mkdir(parents=True)
    fake_settings_path.write_text("", encoding="utf-8")

    env_root = tmp_path / "custom_inputs"
    load_root = env_root / "electricity" / "load_forecasts"
    (load_root / "iso" / "scenario").mkdir(parents=True)

    monkeypatch.setenv("GRANITELEDGER_INPUT_ROOT", str(env_root))
    monkeypatch.setattr(settings, "__file__", str(fake_settings_path))
    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path / "package")

    settings.input_root.cache_clear()
    try:
        discovered = settings.input_root()
    finally:
        settings.input_root.cache_clear()

    assert discovered == load_root.resolve()


def test_configure_load_forecast_path_overrides_root(tmp_path, monkeypatch):
    """``configure_load_forecast_path`` should override the discovered root."""

    csv_path = tmp_path / "custom" / "load_forecasts.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("iso,zone,scenario,year,load_gwh\n", encoding="utf-8")

    monkeypatch.delenv("GRANITELEDGER_INPUT_ROOT", raising=False)
    monkeypatch.setattr(settings, "REPO_ROOT", tmp_path / "package")

    settings.input_root.cache_clear()

    try:
        settings.configure_load_forecast_path(csv_path)
        discovered = settings.input_root()
    finally:
        settings.configure_load_forecast_path(None)
        settings.input_root.cache_clear()

    assert discovered == csv_path.parent.resolve()
