"""Shared engine configuration helpers."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from engine.constants import REPO_ROOT

_CONFIGURED_LOAD_FORECAST_ROOT: Path | None = None


def _coerce_load_forecast_root(base: Path) -> Path:
    """Return ``base`` adjusted to the canonical load forecast directory."""

    try:
        candidate = base.resolve()
    except OSError:
        candidate = base

    # Direct hits are returned immediately.
    name = candidate.name.lower()
    if name == "load_forecasts":
        return candidate

    # Common misconfigurations: pointing at ``input`` or ``electricity`` roots.
    suffixes = [
        Path("input") / "electricity" / "load_forecasts",
        Path("inputs") / "electricity" / "load_forecasts",
        Path("electricity") / "load_forecasts",
        Path("load_forecasts"),
    ]
    for suffix in suffixes:
        nested = candidate / suffix
        if nested.exists():
            try:
                return nested.resolve()
            except OSError:
                return nested

    return candidate


def _find_nested_input_root(base: Path) -> Path | None:
    """Return the first ``input/electricity/load_forecasts`` directory under ``base``.

    Some deployment environments unpack the project into an additional directory
    (for example ``GraniteLedger-main/``) and execute the GUI from the parent
    directory.  In that case ``Path.cwd()`` does not contain the expected
    ``input`` tree directly, so we scan one level deeper to locate it.
    """

    try:
        entries = sorted(base.iterdir(), key=lambda item: item.name.lower())
    except OSError:
        return None

    for entry in entries:
        if not entry.is_dir():
            continue
        for name in ("input", "inputs"):
            candidate = entry / name / "electricity" / "load_forecasts"
            if candidate.exists():
                try:
                    return candidate.resolve()
                except OSError:
                    return candidate
    return None


@lru_cache(maxsize=1)
def input_root() -> Path:
    """Return the root directory for electricity demand inputs.

    Resolution order:

    1. ``GRANITELEDGER_INPUT_ROOT`` environment variable.
    2. Repository-style relative layout when running from source.
    3. Current working directory fallback.
    """

    if _CONFIGURED_LOAD_FORECAST_ROOT is not None:
        return _CONFIGURED_LOAD_FORECAST_ROOT

    env = os.getenv("GRANITELEDGER_INPUT_ROOT")
    if env:
        candidate = Path(env).expanduser()
        return _coerce_load_forecast_root(candidate)

    for name in ("input", "inputs"):
        repo_candidate = REPO_ROOT / name / "electricity" / "load_forecasts"
        if repo_candidate.exists():
            return _coerce_load_forecast_root(repo_candidate)

    here = Path(__file__).resolve()
    for ancestor in (here, *here.parents):
        for name in ("input", "inputs"):
            candidate = ancestor / name / "electricity" / "load_forecasts"
            if candidate.exists():
                return _coerce_load_forecast_root(candidate)

    nested = _find_nested_input_root(Path.cwd())
    if nested:
        return _coerce_load_forecast_root(nested)

    for name in ("input", "inputs"):
        fallback = Path.cwd() / name / "electricity" / "load_forecasts"
        if fallback.exists():
            return _coerce_load_forecast_root(fallback)

    return _coerce_load_forecast_root(
        Path.cwd() / "input" / "electricity" / "load_forecasts"
    )


def configure_load_forecast_path(path: os.PathLike[str] | str | None) -> None:
    """Configure the consolidated load forecast CSV location.

    Parameters
    ----------
    path:
        Absolute path to ``load_forecasts.csv`` or the directory that contains
        it.  Passing ``None`` clears the override and reverts to automatic
        discovery.
    """

    global _CONFIGURED_LOAD_FORECAST_ROOT

    if path is None:
        _CONFIGURED_LOAD_FORECAST_ROOT = None
        input_root.cache_clear()
        return

    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError:
        resolved = candidate

    if resolved.name.lower().endswith(".csv"):
        root_candidate = resolved.parent
    else:
        root_candidate = resolved

    _CONFIGURED_LOAD_FORECAST_ROOT = _coerce_load_forecast_root(root_candidate)
    input_root.cache_clear()


__all__ = ["input_root", "configure_load_forecast_path"]
