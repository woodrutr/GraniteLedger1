from __future__ import annotations
import pandas as pd
from typing import Mapping, Any, Sequence
from engine.prices.normalize import coerce_price_mapping
from engine.prices.types import CarbonPriceVector

def _first_present(source: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key not in source:
            continue
        value = source[key]
        if value is not None and value != "":
            return value
    return None

def dataframe_to_carbon_vector(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with canonical :class:`CarbonPriceVector` columns.

    The returned frame is a shallow copy of *df* with ``cp_*`` columns populated
    from the canonical inputs when available. Legacy keys (``p_co2*``) are only
    used as a fallback for backwards compatibility; callers should provide
    ``cp_*`` columns.
    """
    if df is None or df.empty:
        return df

    rows = []
    for _, row in df.iterrows():
        raw = row.to_dict()
        price_mapping = {
            "all": _first_present(raw, ["cp_all", "allowance_price", "p_co2_all", "p_co2"]),
            "effective": _first_present(
                raw,
                ["cp_effective", "carbon_price", "effective_carbon_price", "p_co2_eff", "p_co2"],
            ),
            "exempt": _first_present(raw, ["cp_exempt", "p_co2_exc"]),
            "last": _first_present(raw, ["cp_last", "last", "price_last", "p_co2"]),
        }

        default_last_value = _first_present(raw, ["cp_last", "price_last", "p_co2"])
        try:
            default_last = float(default_last_value) if default_last_value is not None else None
        except (TypeError, ValueError):
            default_last = None

        cp = coerce_price_mapping(price_mapping, default_last=default_last, year=raw.get("year", None))

        newrow = dict(row)
        newrow["cp_all"] = cp.all
        newrow["cp_effective"] = cp.effective
        newrow["cp_exempt"] = cp.exempt
        newrow["cp_last"] = cp.last
        rows.append(newrow)

    return pd.DataFrame(rows, columns=list(df.columns) + ["cp_all", "cp_effective", "cp_exempt", "cp_last"])

