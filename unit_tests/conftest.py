"""Test configuration that skips optional suites when heavy dependencies are missing."""

from __future__ import annotations

import importlib.util
from typing import Iterable

import pytest

_REQUIRED_MODULES: dict[str, str] = {
    "pyomo.environ": "pyomo",
    "pandas": "pandas",
}


def _missing_modules(modules: Iterable[tuple[str, str]]) -> list[str]:
    missing: list[str] = []
    for module_name, display_name in modules:
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            missing.append(display_name)
    return missing


_missing = _missing_modules(tuple(_REQUIRED_MODULES.items()))
collect_ignore_glob = ["*"] if _missing else []


def pytest_report_header(config: pytest.Config) -> str:  # type: ignore[override]
    if _missing:
        missing = ", ".join(sorted(_missing))
        return f"unit tests skipped: missing optional dependencies: {missing}"
    return ""
