
"""Determinism post-processing for EngineOutputs."""
from __future__ import annotations

from typing import Any

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

from engine.utils.determinism import sort_dataframe

KNOWN_FRAME_ATTRS = [
    "annual_df",
    "emissions_df",
    "generation_df",
    "capacity_df",
    "cost_df",
    "stranded_df",
    "flows_df",
    "prices_df",
    "audits_df",
]

def apply_determinism(outputs: Any) -> Any:
    """Sort known DataFrame attributes by index and columns if present.

    Returns the same object for fluent usage.
    """
    if outputs is None:
        return outputs
    for name in KNOWN_FRAME_ATTRS:
        try:
            val = getattr(outputs, name, None)
            if val is not None and pd is not None and isinstance(val, pd.DataFrame):
                setattr(outputs, name, sort_dataframe(val, by_index=True))
        except Exception:
            # Non-fatal
            pass
    return outputs
