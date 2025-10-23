"""Streamlit helpers for configuring solver iteration intensity."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - optional dependency for GUI usage
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - streamlit optional
    st = None  # type: ignore[assignment]

from engine import compute_intensity as _intensity


@dataclass(frozen=True)
class ComputeIntensitySettings:
    """Record of the iteration intensity selection from the GUI."""

    max_iterations: int


_SLIDER_KEY = "compute_intensity_max_iterations"


def _default_slider_value() -> int:
    configured = _intensity.get_iteration_limit()
    if configured is not None:
        return int(configured)
    return int(_intensity.default_limit())


def render(container: Any, *, help_text: str | None = None) -> ComputeIntensitySettings:
    """Render the iteration intensity slider and update engine configuration."""

    bounds = _intensity.slider_bounds()
    default_value = _default_slider_value()
    max_iterations = int(default_value)

    if st is None:
        _intensity.set_iteration_limit(max_iterations)
        return ComputeIntensitySettings(max_iterations=max_iterations)

    if _SLIDER_KEY not in st.session_state:
        st.session_state[_SLIDER_KEY] = max_iterations

    session_value = st.session_state.get(_SLIDER_KEY)
    if isinstance(session_value, int) and session_value > 0:
        default_value = int(session_value)

    slider_value = container.slider(
        "Iteration limit",
        min_value=int(bounds[0]),
        max_value=int(bounds[1]),
        value=int(default_value),
        step=1,
        help=help_text
        or "Sets the maximum number of iterations allowed for solvers during this run.",
        key=_SLIDER_KEY,
    )

    max_iterations = int(slider_value)
    _intensity.set_iteration_limit(max_iterations)

    return ComputeIntensitySettings(max_iterations=max_iterations)


__all__ = ["ComputeIntensitySettings", "render"]
