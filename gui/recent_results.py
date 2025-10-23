"""Lightweight repository for the most recent GUI run results."""

from __future__ import annotations

from threading import RLock
from typing import Any, Mapping, MutableMapping

__all__ = [
    "get_recent_result",
    "record_recent_result",
    "clear_recent_result",
]


_LOCK = RLock()
_RECENT_RESULT: Mapping[str, Any] | None = None


def _is_mapping(candidate: object) -> bool:
    """Return ``True`` when ``candidate`` behaves like a mapping."""

    if isinstance(candidate, Mapping):
        return True
    # ``MutableMapping`` is kept for compatibility with classes only registering
    # the mutable ABC.  ``isinstance`` above covers most cases but subclasses of
    # :class:`collections.abc.MutableMapping` without ``Mapping`` in their MRO
    # appear in certain tests.
    return isinstance(candidate, MutableMapping)


def record_recent_result(result: Mapping[str, Any] | None) -> None:
    """Persist ``result`` as the most recent GUI run output."""

    global _RECENT_RESULT
    with _LOCK:
        if _is_mapping(result):
            _RECENT_RESULT = result  # type: ignore[assignment]
        else:
            _RECENT_RESULT = None


def get_recent_result() -> Mapping[str, Any] | None:
    """Return the most recently recorded GUI run result."""

    with _LOCK:
        return _RECENT_RESULT


def clear_recent_result() -> None:
    """Reset the cached GUI run result."""

    record_recent_result(None)
