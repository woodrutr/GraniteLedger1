"""Helpers for coordinating iteration limits across solver components."""
from __future__ import annotations

from typing import Tuple

from .constants import (
    ITER_MAX_CAP_EXPANSION,
    ITER_MAX_CAP_SOLVER,
    MAX_UNIQUE_DIR_ATTEMPTS,
)


_DEFAULT_LIMIT = max(
    int(ITER_MAX_CAP_SOLVER),
    int(ITER_MAX_CAP_EXPANSION),
    int(MAX_UNIQUE_DIR_ATTEMPTS),
)

_iteration_limit: int | None = None


def default_limit() -> int:
    """Return the default global iteration limit used by the engine."""

    return int(_DEFAULT_LIMIT)


def get_iteration_limit() -> int | None:
    """Return the user-selected iteration limit or ``None`` when unset."""

    return _iteration_limit


def set_iteration_limit(limit: int | None) -> None:
    """Set the global iteration limit used by iterative routines."""

    global _iteration_limit
    if limit is None:
        _iteration_limit = None
        return

    try:
        parsed = int(limit)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("Iteration limit must be an integer or None") from exc

    if parsed <= 0:
        raise ValueError("Iteration limit must be a positive integer")

    _iteration_limit = parsed


def get_effective_iteration_limit(fallback: int | None = None) -> int:
    """Return the active iteration limit falling back to ``fallback``."""

    if fallback is None or fallback <= 0:
        fallback = _DEFAULT_LIMIT
    if _iteration_limit is None:
        return int(fallback)
    return max(1, int(_iteration_limit))


def resolve_iteration_limit(default: int) -> int:
    """Return the limit that should be applied for a specific solver."""

    return get_effective_iteration_limit(default)


def slider_bounds() -> Tuple[int, int]:
    """Return inclusive bounds for the GUI iteration slider."""

    maximum = max(_DEFAULT_LIMIT * 2, _DEFAULT_LIMIT + 50)
    return (1, maximum)


__all__ = [
    "default_limit",
    "get_effective_iteration_limit",
    "get_iteration_limit",
    "resolve_iteration_limit",
    "set_iteration_limit",
    "slider_bounds",
]
