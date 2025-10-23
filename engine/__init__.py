
"""Engine package public API."""

from __future__ import annotations

from typing import Any, Mapping

try:  # pragma: no cover - optional dependency guard
    import pandas as pd
except Exception:  # pragma: no cover - fallback when pandas is unavailable
    pd = None  # type: ignore

_UNIT_FRAME: "pd.DataFrame | None" = None


def _ensure_pandas() -> None:
    if pd is None:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "pandas is required to register unit characteristics; install it with `pip install pandas`."
        )


def set_units(df: "pd.DataFrame | Mapping[str, Any] | None") -> None:
    """Register generating unit characteristics for downstream consumers.

    Parameters
    ----------
    df:
        DataFrame containing the normalized EI unit fleet columns. The loader
        is expected to provide ``region_id``, ``heat_rate_mmbtu_per_mwh``,
        ``capacity_mw``, and ``co2_short_ton_per_mwh``.
    """

    _ensure_pandas()
    assert pd is not None  # Satisfy type-checkers

    if df is None:
        raise ValueError("Units DataFrame must be provided")

    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception as exc:  # pragma: no cover - defensive guard
            raise TypeError("Units data must be convertible to a pandas DataFrame") from exc

    required = {
        "region_id",
        "heat_rate_mmbtu_per_mwh",
        "capacity_mw",
        "co2_short_ton_per_mwh",
    }
    missing = required - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(
            "Units DataFrame is missing required columns: " + missing_list
        )

    working = df.loc[:, list(required)].copy()
    working["region_id"] = working["region_id"].astype(str).str.strip()
    working = working[working["region_id"].astype(bool)]
    working["region_id"] = working["region_id"].str.upper()

    renamed = working.rename(
        columns={
            "heat_rate_mmbtu_per_mwh": "heat_rate",
            "capacity_mw": "capacity",
            "co2_short_ton_per_mwh": "co2_rate",
        }
    )

    for column in ("heat_rate", "capacity", "co2_rate"):
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce").fillna(0.0)

    result = renamed.loc[:, ["region_id", "heat_rate", "capacity", "co2_rate"]].reset_index(
        drop=True
    )

    global _UNIT_FRAME
    _UNIT_FRAME = result


def registered_units() -> "pd.DataFrame":
    """Return the currently registered unit fleet DataFrame."""

    _ensure_pandas()
    assert pd is not None

    if _UNIT_FRAME is None:
        return pd.DataFrame(columns=["region_id", "heat_rate", "capacity", "co2_rate"])
    return _UNIT_FRAME.copy(deep=True)


def clear_units() -> None:
    """Clear the cached unit registry (primarily for tests)."""

    global _UNIT_FRAME
    _UNIT_FRAME = None


# Authoritative orchestrator entrypoint
try:  # pragma: no cover - keep package importable when orchestrator missing
    from engine.orchestrate import run_policy_simulation  # noqa: F401
except Exception:  # pragma: no cover - fallback when orchestrator not yet built
    pass


__all__ = [
    "clear_units",
    "registered_units",
    "run_policy_simulation",
    "set_units",
]
