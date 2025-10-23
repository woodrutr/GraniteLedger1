"""Helpers for loading optional engine constant overrides."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar, cast

try:  # pragma: no cover - Python < 3.11 fallback
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - dependency fallback
    import tomli as tomllib  # type: ignore[import-not-found]


LOGGER = logging.getLogger(__name__)

_ENV_PREFIX = "GRANITELEDGER_"
_RUN_CONFIG_ENV = "GRANITELEDGER_RUN_CONFIG"
_CONSTANTS_SECTION_CANDIDATES: Sequence[Sequence[str]] = (
    ("engine", "constants"),
    ("engine_constants",),
    ("constants",),
)

T = TypeVar("T")


def _default_run_config_path() -> Path:
    """Return the default run_config path within the repository."""

    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "src" / "common" / "run_config.toml"


@lru_cache(maxsize=1)
def _config_overrides() -> dict[str, Any]:
    """Return overrides sourced from ``run_config.toml`` if available."""

    candidates: list[Path] = []
    env_value = os.environ.get(_RUN_CONFIG_ENV)
    if env_value:
        candidates.append(Path(env_value).expanduser())

    default_path = _default_run_config_path()
    if default_path.exists():
        candidates.append(default_path)

    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if not resolved.exists():
            continue
        try:
            with resolved.open("rb") as handle:
                data = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            LOGGER.debug("Unable to read run_config overrides from %s: %s", resolved, exc)
            continue
        section = _extract_constants_section(data)
        if section:
            return {str(key).upper(): value for key, value in section.items()}
    return {}


def _extract_constants_section(data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the first matching constants section from ``data``."""

    for path in _CONSTANTS_SECTION_CANDIDATES:
        cursor: Any = data
        for key in path:
            if not isinstance(cursor, Mapping):
                break
            cursor = cursor.get(key)
        else:
            if isinstance(cursor, Mapping):
                return cursor
    return {}


def _raw_override(name: str) -> Any | None:
    env_key = f"{_ENV_PREFIX}{name}"
    if env_key in os.environ:
        return os.environ[env_key]
    return _config_overrides().get(name.upper())


def get_constant(name: str, default: T, cast_func: Callable[[Any], T] | None = None) -> T:
    """Return ``name`` resolved against overrides falling back to ``default``."""

    raw = _raw_override(name)
    if raw is None:
        return default

    converter: Callable[[Any], T]
    if cast_func is not None:
        converter = cast_func
    else:
        converter = cast(Callable[[Any], T], type(default))

    try:
        return converter(raw)
    except (TypeError, ValueError) as exc:
        LOGGER.warning("Invalid override for %s=%r: %s", name, raw, exc)
        return default


def clear_cache() -> None:
    """Reset cached override state (primarily for use in tests)."""

    _config_overrides.cache_clear()


__all__ = ["get_constant", "clear_cache"]
