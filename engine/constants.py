"""Authoritative constants shared across engine and dispatch modules."""

from __future__ import annotations

from pathlib import Path

from .constants_overrides import get_constant


_PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _PACKAGE_ROOT.parent


def _detect_input_dir() -> Path:
    """Return the repository input directory, tolerating ``inputs`` variants."""
    for name in ("input", "inputs"):
        candidate = REPO_ROOT / name
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    # Fall back to the canonical ``input`` directory.
    return REPO_ROOT / "input"


INPUT_DIR = _detect_input_dir()
OUTPUT_DIR = REPO_ROOT / "output"
LOAD_FORECASTS_STRICT_PARQUET = OUTPUT_DIR / "load_forecasts_strict.parquet"
CSV_YEAR_COL = "Year"
CSV_LOAD_COL = "Load_GWh"

HOURS_PER_YEAR: float = get_constant("HOURS_PER_YEAR", 8760.0, float)
PRICE_TOL: float = get_constant("PRICE_TOL", 1e-6, float)
FLOW_TOL: float = get_constant("FLOW_TOL", 1e-6, float)
EMISS_TOL: float = get_constant("EMISS_TOL", 1e-6, float)

ITER_MAX_CAP_EXPANSION: int = get_constant("ITER_MAX_CAP_EXPANSION", 500, int)
ITER_MAX_CAP_SOLVER: int = get_constant("ITER_MAX_CAP_SOLVER", 200, int)
MAX_UNIQUE_DIR_ATTEMPTS: int = get_constant("MAX_UNIQUE_DIR_ATTEMPTS", 1000, int)

DEFAULT_CACHE_DIR: str = get_constant("DEFAULT_CACHE_DIR", ".cache", str)


__all__ = [
    "REPO_ROOT",
    "INPUT_DIR",
    "OUTPUT_DIR",
    "LOAD_FORECASTS_STRICT_PARQUET",
    "CSV_YEAR_COL",
    "CSV_LOAD_COL",
    "HOURS_PER_YEAR",
    "PRICE_TOL",
    "FLOW_TOL",
    "EMISS_TOL",
    "ITER_MAX_CAP_EXPANSION",
    "ITER_MAX_CAP_SOLVER",
    "MAX_UNIQUE_DIR_ATTEMPTS",
    "DEFAULT_CACHE_DIR",
]

