from __future__ import annotations
from typing import Mapping, Any, Optional
from .types import CarbonPriceVector

def _as_mapping(obj: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(obj, Mapping):
        return obj
    # allow simple objects with __dict__
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        return d
    return None

def coerce_price_mapping(
    m: Any,
    *,
    default_last: float | None = None,
    year: int | None = None,
) -> CarbonPriceVector:
    """Coerce various inputs into a CarbonPriceVector.

    Accepts:
      - CarbonPriceVector (returned as-is)
      - Mapping or object providing canonical keys ('all', 'effective', 'exempt', 'last')
      - Scalar number (treated as 'effective', others 0)
    """
    if isinstance(m, CarbonPriceVector):
        return m

    # scalar
    try:
        val = float(m)  # type: ignore[arg-type]
        return CarbonPriceVector(
            all=val,
            effective=val,
            exempt=0.0,
            last=default_last,
            year=year,
        )
    except Exception:
        pass

    mm = _as_mapping(m)
    if mm is None:
        raise TypeError(f"Unsupported carbon price input type: {type(m)!r}")

    canonical_keys = {"all", "effective", "exempt", "last"}
    canonical_values: dict[str, float | None] = {}

    legacy_indicators = {
        "allowance_price_last",
        "exogenous_price_last",
        "effective_price_last",
        "price_last",
        "allowance_price",
        "carbon_price",
        "effective_carbon_price",
        "p_co2_all",
        "p_co2_exc",
        "p_co2_eff",
    }
    unexpected = sorted(k for k in mm if k in legacy_indicators)
    if unexpected:
        raise ValueError(
            "Carbon price mapping uses legacy keys "
            f"{unexpected}. Provide canonical keys ('all', 'effective', 'exempt', 'last') "
            "or pre-compute a CarbonPriceVector."
        )

    for key in canonical_keys:
        value = mm.get(key)
        if value is None:
            canonical_values[key] = None
            continue
        try:
            canonical_values[key] = float(value)
        except (TypeError, ValueError):
            canonical_values[key] = None

    all_v = float(canonical_values.get("all") or 0.0)
    eff_v = float(canonical_values.get("effective") or all_v)
    exc_v = float(canonical_values.get("exempt") or 0.0)
    last_raw = canonical_values.get("last")
    last_v: float | None
    if last_raw is not None:
        last_v = float(last_raw)
    else:
        last_v = default_last

    return CarbonPriceVector(
        all=all_v,
        effective=eff_v,
        exempt=exc_v,
        last=last_v,
        year=year,
    )

