"""Generate capacity expansion candidates with NREL costs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

from engine.nrel_costs import load_nrel_costs

LOGGER = logging.getLogger(__name__)

# Default candidate sizes (MW) for new builds
DEFAULT_UNIT_SIZES = {
    "solar": 100.0,  # 100 MW solar farm
    "wind": 150.0,  # 150 MW wind farm
    "wind_offshore": 200.0,  # 200 MW offshore wind
    "natural_gas_combined_cycle": 500.0,  # 500 MW NGCC plant
    "natural_gas_combustion_turbine": 100.0,  # 100 MW peaker
    "battery": 100.0,  # 100 MW / 400 MWh battery
}

# Variable O&M costs ($/MWh)
DEFAULT_VOM = {
    "solar": 0.0,
    "wind": 0.0,
    "wind_offshore": 0.0,
    "natural_gas_combined_cycle": 3.5,
    "natural_gas_combustion_turbine": 7.0,
    "battery": 0.5,
}

# Emissions factors (tons CO2/MWh)
DEFAULT_EMISSIONS = {
    "solar": 0.0,
    "wind": 0.0,
    "wind_offshore": 0.0,
    "natural_gas_combined_cycle": 0.37,  # ~370 kg/MWh
    "natural_gas_combustion_turbine": 0.55,  # Higher for peakers
    "battery": 0.0,
}


def _ensure_pandas() -> None:
    """Ensure pandas is available."""
    if pd is None:
        raise ImportError("pandas is required for expansion candidates")


def create_expansion_candidates(
    regions: List[str],
    *,
    atb_path: str | Path | None = None,
    scenario: str = "Moderate",
    fuel_prices: Dict[str, float] | None = None,
    max_builds_per_tech: int = 10,
    enabled_techs: List[str] | None = None,
) -> pd.DataFrame:
    """Create capacity expansion candidate DataFrame with NREL costs.
    
    Args:
        regions: List of region IDs where capacity can be built
        atb_path: Path to NREL ATBe.csv file (optional)
        scenario: NREL cost scenario (Conservative, Moderate, Advanced)
        fuel_prices: Dictionary of fuel prices in $/MMBtu
        max_builds_per_tech: Maximum number of builds per technology per region
        enabled_techs: List of enabled technology keys (None = all technologies)
    
    Returns:
        DataFrame with expansion candidate specifications including NREL costs
    """
    _ensure_pandas()
    
    if fuel_prices is None:
        fuel_prices = {
            "natural_gas": 4.0,  # $/MMBtu
            "coal": 2.5,
            "oil": 15.0,
        }
    
    # Load NREL costs
    if atb_path is not None and Path(atb_path).exists():
        nrel_costs = load_nrel_costs(atb_path, scenario=scenario)
        LOGGER.info(f"Loaded NREL costs from {atb_path} with scenario={scenario}")
    else:
        if atb_path is not None:
            LOGGER.warning(f"ATB file not found: {atb_path}, using default costs")
        nrel_costs = load_nrel_costs("", scenario=scenario)  # Will use defaults
    
    # Determine which technologies to include
    if enabled_techs is None:
        enabled_techs = list(nrel_costs.keys())
    
    candidates = []
    
    for region in regions:
        for tech_key in enabled_techs:
            if tech_key not in nrel_costs:
                LOGGER.warning(f"Technology {tech_key} not found in NREL costs, skipping")
                continue
            
            costs = nrel_costs[tech_key]
            
            # Map technology to fuel type and fuel price
            if "natural_gas" in tech_key:
                fuel = "natural_gas"
                fuel_price = fuel_prices.get("natural_gas", 4.0)
            elif tech_key in ("solar", "wind", "wind_offshore"):
                fuel = tech_key
                fuel_price = 0.0  # No fuel cost for renewables
            elif tech_key == "battery":
                fuel = "storage"
                fuel_price = 0.0
            else:
                fuel = tech_key
                fuel_price = fuel_prices.get(tech_key, 0.0)
            
            candidate = {
                "unit_id": f"{region}_{tech_key}_new",
                "unique_id": f"{region}_{tech_key}_new",
                "region": region,
                "fuel": fuel,
                "cap_mw": DEFAULT_UNIT_SIZES.get(tech_key, 100.0),
                "availability": costs["availability"],
                "hr_mmbtu_per_mwh": costs["heat_rate"],
                "vom_per_mwh": DEFAULT_VOM.get(tech_key, 0.0),
                "fuel_price_per_mmbtu": fuel_price,
                "ef_ton_per_mwh": DEFAULT_EMISSIONS.get(tech_key, 0.0),
                "co2_short_ton_per_mwh": DEFAULT_EMISSIONS.get(tech_key, 0.0),
                "capex_per_mw": costs["capex_per_mw"],
                "fixed_om_per_mw": costs["fixed_om_per_mw"],
                "lifetime_years": costs["lifetime_years"],
                "max_builds": max_builds_per_tech,
            }
            
            candidates.append(candidate)
    
    df = pd.DataFrame(candidates)
    
    LOGGER.info(
        f"Created {len(df)} expansion candidates across {len(regions)} regions "
        f"with {len(enabled_techs)} technologies"
    )
    
    return df


__all__ = ["create_expansion_candidates"]
