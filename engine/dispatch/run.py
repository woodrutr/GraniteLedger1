
"""Dispatch runner extraction.

Implements `run_dispatch_year` as the engine-level entry for per-year dispatch.
Behavior preserved by delegating to legacy `engine.run_loop._dispatch_from_frames`.
No GUI imports.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping, Optional, Union

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    from dispatch.interface import DispatchResult  # type: ignore
except Exception:  # pragma: no cover
    DispatchResult = object  # type: ignore

try:
    from granite_io.frames_api import Frames  # type: ignore
except Exception:  # pragma: no cover
    Frames = object  # type: ignore

FramesType = Union["Frames", Mapping[str, "pd.DataFrame"]]  # type: ignore[name-defined]

def run_dispatch_year(
    frames: FramesType,
    year: int,
    allowance_cost: float,
    *,
    carbon_price: float,
    use_network: bool,
    period_weights: Optional[Mapping[Any, float]],
    carbon_price_schedule: Optional[Mapping[int, float]],
    deep_carbon_pricing: bool,
    config: Optional[Mapping[str, Any]] = None,
) -> DispatchResult:
    """Run dispatch for a single year using the existing backend."""
    # Defer import to avoid circular refs
    from engine import run_loop as _legacy

    def _normalize_frames(obj: FramesType) -> FramesType:
        if hasattr(Frames, "coerce"):
            try:
                return Frames.coerce(obj)  # type: ignore[attr-defined]
            except Exception:
                return obj
        return obj

    normalized_frames = _normalize_frames(frames)

    def _extract_load_frame(source: FramesType) -> Any:
        optional = getattr(source, "optional_frame", None)
        if callable(optional):
            try:
                result = optional("load")
                if result is not None:
                    return result
            except Exception:
                pass

        frame_fn = getattr(source, "frame", None)
        if callable(frame_fn):
            try:
                return frame_fn("load")
            except Exception:
                pass

        if isinstance(source, Mapping):
            return source.get("load")  # type: ignore[call-arg]

        attr = getattr(source, "load", None)
        if callable(attr):
            try:
                return attr()
            except TypeError:
                return attr
        return attr

    load_frame = _extract_load_frame(normalized_frames)

    if load_frame is None:
        raise AssertionError("dispatch requires a non-empty load frame")

    if pd is None:
        raise AssertionError("pandas is required to validate load data for dispatch")

    if not isinstance(load_frame, pd.DataFrame):
        load_frame = pd.DataFrame(load_frame)

    if load_frame.empty:
        raise AssertionError("dispatch requires a non-empty load frame")

    modeled_years: Iterable[int] = ()
    if isinstance(config, Mapping):
        years_value = config.get("years")
        if isinstance(years_value, Iterable) and not isinstance(years_value, (str, bytes)):
            normalized_years = []
            for candidate in years_value:
                try:
                    normalized_years.append(int(candidate))
                except Exception:
                    continue
            modeled_years = normalized_years

    if modeled_years:
        if "year" not in load_frame.columns:
            raise AssertionError(
                "load frame must include a 'year' column to validate modeled years"
            )
        load_years = {
            int(value)
            for value in pd.to_numeric(load_frame["year"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        }
        missing_years = sorted({int(y) for y in modeled_years if int(y) not in load_years})
        if missing_years:
            missing_str = ", ".join(str(y) for y in missing_years)
            raise AssertionError(
                f"load frame is missing demand rows for modeled years: {missing_str}"
            )

    dispatch_kwargs = {
        "frames": frames,
        "use_network": use_network,
        "period_weights": period_weights,
        "carbon_price_schedule": carbon_price_schedule,
        "deep_carbon_pricing": deep_carbon_pricing,
    }

    try:
        builder = _legacy._dispatch_from_frames(  # type: ignore[attr-defined]
            **dispatch_kwargs
        )
    except TypeError:
        # Older signatures used positional arguments and fewer keywords
        try:
            builder = _legacy._dispatch_from_frames(  # type: ignore[attr-defined]
                frames,
                use_network,
                period_weights,
            )
        except TypeError:
            builder = _legacy._dispatch_from_frames(  # type: ignore[attr-defined]
                frames=frames,
                use_network=use_network,
            )

    if not callable(builder):
        raise RuntimeError("dispatch_from_frames did not return a callable dispatch solver")

    try:
        return builder(year, allowance_cost, carbon_price=carbon_price)
    except TypeError:
        try:
            return builder(year, allowance_cost, carbon_price)
        except TypeError:
            return builder(year, allowance_cost)


__all__ = ["run_dispatch_year"]
