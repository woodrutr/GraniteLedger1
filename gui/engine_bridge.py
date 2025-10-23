
"""GUI ↔ Engine bridge."""
from __future__ import annotations

from typing import Any, Mapping

from engine.orchestrate import run_policy_simulation
from gui.config_adapter import build_engine_config
from gui.demand_helpers import load_user_demand_csv


def run_from_gui(gui_config: Mapping[str, Any], frames):
    """Translate GUI config to engine config and run."""

    cfg = build_engine_config(gui_config)
    return run_policy_simulation(cfg, frames)


__all__ = ["run_policy_simulation", "run_from_gui"]
