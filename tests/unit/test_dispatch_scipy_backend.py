"""Tests for the SciPy dispatch backend dependency handling."""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dispatch.solvers import scipy_backend


@pytest.fixture(autouse=True)
def _restore_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure monkeypatched dependency markers are reset after each test."""

    original_linprog = scipy_backend.linprog
    original_pulp = scipy_backend.pulp

    yield

    monkeypatch.setattr(scipy_backend, "linprog", original_linprog, raising=False)
    monkeypatch.setattr(scipy_backend, "pulp", original_pulp, raising=False)


def test_scipy_backend_instantiation_defers_dependency_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure SciPyBackend can be instantiated even if optional deps are missing."""

    monkeypatch.setattr(scipy_backend, "linprog", None, raising=False)
    monkeypatch.setattr(scipy_backend, "pulp", None, raising=False)

    backend = scipy_backend.SciPyBackend()

    with pytest.raises(ModuleNotFoundError):
        backend.solve([1.0], None, None, None, None, [(0.0, None)])


def test_dependencies_available_reflects_module_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """`dependencies_available` reports True when either optional library is present."""

    monkeypatch.setattr(scipy_backend, "linprog", object(), raising=False)
    monkeypatch.setattr(scipy_backend, "pulp", None, raising=False)
    assert scipy_backend.dependencies_available()

    monkeypatch.setattr(scipy_backend, "linprog", None, raising=False)
    monkeypatch.setattr(scipy_backend, "pulp", None, raising=False)
    assert not scipy_backend.dependencies_available()
