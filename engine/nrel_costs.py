"""NREL ATB cost loader for capacity expansion.

Loads and processes NREL Annual Technology Baseline (ATB) cost data
to provide CAPEX, Fixed O&M, and technology parameters for capacity expansion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

LOGGER = logging.getLogger(__name__)

# Technology mapping from model fuel types to NREL ATB categories
TECH_MAPPING = {
    "solar": {
        "nrel_tech": "Utility-Scale PV",
        "nrel_detail": "Class5",  # Average resource class
        "availability": 0.25,  # ~25% capacity factor
        "lifetime_years": 30,
        "heat_rate": 0.0,
    },
    "wind": {
        "nrel_tech": "LandbasedWind",
        "nrel_detail": "Class4",  # Average resource class
        "availability": 0.35,  # ~35% capacity factor
        "lifetime_years": 30,
        "heat_rate": 0.0,
    },
    "wind_offshore": {
        "nrel_tech": "OffShoreWind",
        "nrel_detail": "Class3",
        "availability": 0.45,  # Higher for offshore
        "lifetime_years": 30,
        "heat_rate": 0.0,
    },
    "natural_gas_combined_cycle": {
        "nrel_tech": "NaturalGas_CCCCSAvgCF",
        "nrel_detail": "CCAvgCF",
        "availability": 0.87,  # High availability baseload
        "lifetime_years": 30,
        "heat_rate": 6.5,  # MMBtu/MWh - efficient NGCC
    },
    "natural_gas_combustion_turbine": {
        "nrel_tech": "NaturalGas_CTAvgCF",
        "nrel_detail": "CTAvgCF",
        "availability": 0.93,  # Peaking units
        "lifetime_years": 30,
        "heat_rate": 10.0,  # MMBtu/MWh - less efficient peaker
    },
    "battery": {
        "nrel_tech": "Utility-Scale Battery Storage",
        "nrel_detail": "4Hr Battery Storage",
        "availability": 0.90,
        "lifetime_years": 15,
        "heat_rate": 0.0,
    },
}


def _ensure_pandas() -> None:
    """Ensure pandas is available."""
    if pd is None:
        raise ImportError("pandas is required for NREL cost loading")


def load_nrel_costs(
    atb_path: str | Path,
    *,
    scenario: str = "Moderate",
    atb_year: int = 2024,
) -> Dict[str, Dict[str, float]]:
    """Load NREL ATB costs for capacity expansion candidates.
    
    Args:
        atb_path: Path to ATBe.csv file
        scenario: Cost scenario (Conservative, Moderate, Advanced)
        atb_year: ATB data year to use (2024, 2025, etc.)
    
    Returns:
        Dictionary mapping fuel types to cost parameters:
        {
            "solar": {
                "capex_per_mw": 1234567.0,
                "fixed_om_per_mw": 12345.0,
                "availability": 0.25,
                "lifetime_years": 30,
                "heat_rate": 0.0,
            },
            ...
        }
    """
    _ensure_pandas()
    
    path = Path(atb_path)
    if not path.exists():
        LOGGER.warning(f"NREL ATB file not found: {atb_path}")
        return _default_costs()
    
    try:
        # Load ATB CSV
        atb_df = pd.read_csv(path)
        
        # Filter for the specified year and scenario
        filtered = atb_df[
            (atb_df["atb_year"] == atb_year) &
            (atb_df["core_metric_case"] == scenario)
        ].copy()
        
        if filtered.empty:
            LOGGER.warning(
                f"No data found for year={atb_year}, scenario={scenario}. Using defaults."
            )
            return _default_costs()
        
        # Extract costs by technology
        costs = {}
        
        for fuel_key, tech_spec in TECH_MAPPING.items():
            nrel_tech = tech_spec["nrel_tech"]
            
            # Find CAPEX (core_metric_variable == "CAPEX" or core_metric_parameter == "CAPEX")
            capex_rows = filtered[
                (filtered["technology_alias"].str.contains(nrel_tech, case=False, na=False)) &
                (
                    (filtered["core_metric_parameter"] == "CAPEX") |
                    (filtered["core_metric_variable"] == "CAPEX")
                )
            ]
            
            # Find Fixed O&M
            fom_rows = filtered[
                (filtered["technology_alias"].str.contains(nrel_tech, case=False, na=False)) &
                (filtered["core_metric_parameter"] == "Fixed O&M")
            ]
            
            if not capex_rows.empty:
                # Use mean of matching rows (some technologies have multiple entries)
                capex_per_kw = float(capex_rows["value"].mean())
                capex_per_mw = capex_per_kw * 1000.0  # Convert $/kW to $/MW
            else:
                LOGGER.debug(f"No CAPEX found for {fuel_key}, using default")
                capex_per_mw = _default_costs()[fuel_key]["capex_per_mw"]
            
            if not fom_rows.empty:
                fom_per_kw_yr = float(fom_rows["value"].mean())
                fom_per_mw = fom_per_kw_yr * 1000.0  # Convert $/kW-yr to $/MW-yr
            else:
                LOGGER.debug(f"No Fixed O&M found for {fuel_key}, using default")
                fom_per_mw = _default_costs()[fuel_key]["fixed_om_per_mw"]
            
            costs[fuel_key] = {
                "capex_per_mw": capex_per_mw,
                "fixed_om_per_mw": fom_per_mw,
                "availability": tech_spec["availability"],
                "lifetime_years": tech_spec["lifetime_years"],
                "heat_rate": tech_spec["heat_rate"],
            }
        
        LOGGER.info(f"Loaded NREL costs for {len(costs)} technologies from {atb_path}")
        return costs
        
    except Exception as exc:
        LOGGER.error(f"Error loading NREL ATB data from {atb_path}: {exc}")
        return _default_costs()


def _default_costs() -> Dict[str, Dict[str, float]]:
    """Return default cost assumptions when NREL data is unavailable.
    
    Based on typical 2024 industry values for reference.
    """
    return {
        "solar": {
            "capex_per_mw": 1_100_000.0,  # $1.1M/MW (~$1100/kW)
            "fixed_om_per_mw": 18_000.0,  # $18k/MW-yr
            "availability": 0.25,
            "lifetime_years": 30,
            "heat_rate": 0.0,
        },
        "wind": {
            "capex_per_mw": 1_400_000.0,  # $1.4M/MW
            "fixed_om_per_mw": 43_000.0,  # $43k/MW-yr
            "availability": 0.35,
            "lifetime_years": 30,
            "heat_rate": 0.0,
        },
        "wind_offshore": {
            "capex_per_mw": 3_500_000.0,  # $3.5M/MW
            "fixed_om_per_mw": 90_000.0,  # $90k/MW-yr
            "availability": 0.45,
            "lifetime_years": 30,
            "heat_rate": 0.0,
        },
        "natural_gas_combined_cycle": {
            "capex_per_mw": 1_000_000.0,  # $1M/MW
            "fixed_om_per_mw": 13_000.0,  # $13k/MW-yr
            "availability": 0.87,
            "lifetime_years": 30,
            "heat_rate": 6.5,
        },
        "natural_gas_combustion_turbine": {
            "capex_per_mw": 700_000.0,  # $700k/MW
            "fixed_om_per_mw": 8_000.0,  # $8k/MW-yr
            "availability": 0.93,
            "lifetime_years": 30,
            "heat_rate": 10.0,
        },
        "battery": {
            "capex_per_mw": 1_500_000.0,  # $1.5M/MW for 4-hour battery
            "fixed_om_per_mw": 25_000.0,  # $25k/MW-yr
            "availability": 0.90,
            "lifetime_years": 15,
            "heat_rate": 0.0,
        },
    }


__all__ = ["load_nrel_costs", "TECH_MAPPING"]
