"""Utilities for selecting linear programming solver backends."""

from __future__ import annotations

from typing import Callable, Dict

from .base import LpBackend, LpSolveStatus
from .scipy_backend import SciPyBackend, dependencies_available

_BACKEND_FACTORIES: Dict[str, Callable[[], LpBackend]] = {
    "scipy": SciPyBackend,
}


def get_backend(name: str) -> LpBackend:
    """Return an instantiated LP backend for ``name``.

    Parameters
    ----------
    name:
        Identifier for the backend. Comparison is case-insensitive and ignores
        surrounding whitespace.

    Raises
    ------
    ValueError
        If ``name`` does not correspond to a registered backend.
    """

    normalized = name.strip().lower()
    try:
        factory = _BACKEND_FACTORIES[normalized]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown LP backend: {name!r}") from exc
    return factory()


__all__ = [
    "LpBackend",
    "LpSolveStatus",
    "SciPyBackend",
    "dependencies_available",
    "get_backend",
]
