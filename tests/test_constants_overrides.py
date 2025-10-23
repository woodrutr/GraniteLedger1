"""Tests for environment and configuration overrides of engine constants."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import engine.constants as constants
import engine.constants_overrides as overrides


def _reload_constants() -> object:
    overrides.clear_cache()
    return importlib.reload(constants)


def test_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GRANITELEDGER_PRICE_TOL", raising=False)
    monkeypatch.delenv("GRANITELEDGER_RUN_CONFIG", raising=False)

    module = _reload_constants()
    assert module.PRICE_TOL == pytest.approx(1e-6)

    monkeypatch.setenv("GRANITELEDGER_PRICE_TOL", "0.25")
    module = _reload_constants()
    assert module.PRICE_TOL == pytest.approx(0.25)

    monkeypatch.delenv("GRANITELEDGER_PRICE_TOL", raising=False)
    module = _reload_constants()
    assert module.PRICE_TOL == pytest.approx(1e-6)


def test_run_config_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GRANITELEDGER_PRICE_TOL", raising=False)
    monkeypatch.delenv("GRANITELEDGER_RUN_CONFIG", raising=False)

    module = _reload_constants()
    assert module.FLOW_TOL == pytest.approx(1e-6)

    config_path = tmp_path / "run_config.toml"
    config_path.write_text("[engine.constants]\nFLOW_TOL = 0.123456\n", encoding="utf-8")

    monkeypatch.setenv("GRANITELEDGER_RUN_CONFIG", str(config_path))
    module = _reload_constants()
    assert module.FLOW_TOL == pytest.approx(0.123456)

    monkeypatch.delenv("GRANITELEDGER_RUN_CONFIG", raising=False)
    module = _reload_constants()
    assert module.FLOW_TOL == pytest.approx(1e-6)
