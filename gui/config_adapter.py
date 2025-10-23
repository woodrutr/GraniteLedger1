
"""Translate GUI control dict to engine orchestrator config."""
from __future__ import annotations
from typing import Mapping, MutableMapping, Any, Dict

BOOL_KEYS = {
    "policy_enabled": True,
    "enable_floor": True,
    "enable_ccr": True,
    "enable_ccr1": True,
    "enable_ccr2": True,
    "banking_enabled": True,
    "deep_carbon_pricing": False,
    "use_network": False,
    "audits_enabled": True,
}

FLOAT_KEYS = {
    "tol": 1e-3,
    "relaxation": 0.5,
    "price_cap": 1000.0,
    "high_price": 1000.0,
    "annual_surrender_frac": 1.0,
    "carry_pct": 1.0,
    "carbon_price_value": 0.0,
}

RENAME = {
    "initial_price": "price_initial",
    "start": "start_year",
    "end": "end_year",
    "price_floor_enabled": "enable_floor",
    "ccr_on": "enable_ccr",
}

def _as_bool(x: Any, default: bool) -> bool:
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1","true","yes","y","on"}: return True
        if s in {"0","false","no","n","off"}: return False
    try:
        return bool(x)
    except Exception:
        return default

def _as_float(x: Any, default: float) -> float:
    try:
        if x is None: return float(default)
        return float(x)
    except Exception:
        return float(default)

def build_engine_config(gui_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    # Pass-through obvious fields
    for k in (
        "years",
        "start_year",
        "end_year",
        "carbon_price_schedule",
        "period_weights",
        "normalized_demand",
        "progress_cb",
        "stage_cb",
        "allowance_supply",
        "bank_start",
        "outstanding_start",
    ):
        if k in gui_cfg: cfg[k] = gui_cfg[k]
    demand_table = gui_cfg.get("demand_table")
    if demand_table is not None:
        cfg["demand_table"] = demand_table
    # Renames
    for old, new in RENAME.items():
        if old in gui_cfg and new not in cfg:
            cfg[new] = gui_cfg[old]
    # Bools
    for k, d in BOOL_KEYS.items():
        if k in gui_cfg:
            cfg[k] = _as_bool(gui_cfg[k], d)
        else:
            cfg[k] = d
    # Floats
    for k, d in FLOAT_KEYS.items():
        if k in gui_cfg:
            cfg[k] = _as_float(gui_cfg[k], d)
        else:
            cfg[k] = d
    # Ints
    if "max_iter" in gui_cfg:
        try: cfg["max_iter"] = int(gui_cfg["max_iter"])
        except Exception: cfg["max_iter"] = 25
    else:
        cfg["max_iter"] = 25
    return cfg
