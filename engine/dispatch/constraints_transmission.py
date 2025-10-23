"""Transmission constraint helpers for the dispatch solver."""

from __future__ import annotations

from typing import Dict, Tuple


def add_transmission_constraints(
    model,
    flow_vars: Dict[tuple[str, str], object],
    edges_df,
    caps_dict: Dict[tuple[str, str], float],
) -> None:
    """Add capacity limits for declared transmission edges only."""

    declared = set(caps_dict.keys())
    unexpected = {key for key in flow_vars.keys() if key not in declared}
    if unexpected:
        sample = ", ".join(map(str, sorted(unexpected))[:10])
        raise RuntimeError(
            f"Flows defined for non-listed interfaces: {sample}"
        )

    for key, var in flow_vars.items():
        cap = caps_dict[key]
        if cap > 0.0:
            model.add_constraint(var <= cap)


__all__ = ["add_transmission_constraints"]
