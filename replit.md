# Overview

GraniteLedger is an open-source energy systems modeling platform designed to analyze carbon policy scenarios across electricity markets. It integrates a modular dispatch engine with carbon allowance market mechanics to simulate multi-year policy outcomes. The platform offers both a Streamlit-based GUI and CLI interfaces for running simulations, supporting custom demand forecasts, transmission constraints, and flexible carbon pricing mechanisms. Its core purpose is to provide an efficient tool for policy analysis, offering insights into generation mix, regional emissions, allowance prices, and transmission flows under various carbon policies.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture

**Streamlit GUI (`gui/app.py`)**: The primary interactive interface for discovering forecasts, selecting demand scenarios, configuring carbon policies, executing simulations, and visualizing results.

## Backend Architecture

**Dispatch Engine (`dispatch/`)**:
- **Single-region solver** (`lp_single.py`): Uses merit-order dispatch.
- **Network solver** (`lp_network.py`): Multi-region Linear Programming (LP) with transmission constraints.
- **Capacity expansion** (`capacity_expansion.py`): Logic for investment decisions, integrated with NREL Annual Technology Baseline (ATB) cost data for economically-driven technology builds.
- Supports pluggable solver backends.

**Allowance Market Integration** (`engine/run_loop.py`): Implements annual fixed-point iteration between dispatch and carbon pricing, including bisection-based market clearing, CCR logic, and banking balance tracking.

**Policy Framework** (`policy/`): Defines various carbon policies like annual allowance supply, banking limits, and clean energy standards.

**Data Normalization Pipeline**: Handles region ID canonicalization, ISO/zone alias resolution, and load forecast validation.

**UI/UX Decisions**:
- Interactive Stacked Bar Charts for Capacity Visualization: Uses Altair for stacked bar charts with interactive legends and enhanced tooltips.
- Interactive Stacked Area Charts for Generation Visualization: Uses Altair for stacked area charts with interactive legends and enhanced tooltips.
- Capacity Factor Analysis Chart: Visualizes actual vs. theoretical maximum capacity factors for each technology.

**Technical Implementations**:
- Fixed capacity factor issues by adding technology-based default capacity factors in `engine/data_loaders/units.py`.
- Implemented scarcity pricing in the single-region solver to correctly trigger capacity expansion.
- Added full capacity expansion functionality for networked multi-region dispatch.

**Critical Model Fixes (October 2025)**:
1. **Hard Emissions Cap Enforcement**: Implemented LP constraint (∑emission_rate×generation ≤ cap_tons) in both network and single-region solvers. Model now REFUSES to find solutions that violate the emissions cap, returning RuntimeError with diagnostics instead of just pricing violations. Cap value from policy system is threaded through run_loop.py to all dispatch solvers.

2. **Zonal Capacity Expansion**: Refactored capacity expansion logic to use locational marginal pricing (LMP) for build decisions. System now evaluates economics per zone using region_prices dict and unserved_by_region dict instead of system-wide aggregated values. Candidates are matched to zones and built only where zonal prices/shortfall justify investment, enabling proper response to zonal scarcity pricing (e.g., $1M/MWh).

3. **Zonal Dispatch with LMP**: Network solver enforces zonal constraints where generators serve only their zone's load unless transmission capacity exists between zones. Capacity expansion now respects these zonal signals, building in high-price/high-shortfall zones rather than averaging signals across the system.

4. **Curtailment/Surplus Handling**: Fixed LP solver failures under tight emissions caps by adding a curtailment variable to the load balance constraint. The revised equation is: Generation + Imports − Exports = Demand + Unserved − Curtailment. This allows zones with excess clean generation (forced by emissions caps) to shed surplus when transmission capacity is insufficient to export it. Curtailment is free (zero cost) and unbounded, preventing the solver from attempting negative unserved energy (mathematically impossible).

5. **Backwards Compatibility**: Dispatch summaries include both legacy fields (price, shortfall_mwh) and zonal fields (region_prices, unserved_by_region) to maintain compatibility with existing code while enabling zonal functionality.

## Data Storage

**Input Management**: Primarily uses CSV-based data ingestion from the `input/engine/` hierarchy for load forecasts, unit characteristics, transmission interfaces, and financial parameters. The system enforces the use of authoritative datasets, failing fast if required files are missing.

**Frame Container** (`granite_io/frames_api.Frames`): An immutable wrapper for validated pandas DataFrames, enforcing schemas and supporting scenario composition.

**Output Persistence** (`engine/outputs.py`): Serializes simulation results to a CSV directory structure, including annual summaries and transmission flow records.

# External Dependencies

**Optimization Solvers**:
- **scipy**: Default for single-region dispatch.
- **pyomo**: Optional for network LP.

**Data Processing**:
- **pandas**: Core data manipulation and frame validation.
- **numpy**: Numerical operations.

**GUI Frameworks**:
- **streamlit**: Primary web interface.
- **dash** + **plotly**: Alternative visualization dashboard.
- **altair**: Used for interactive charting in the Streamlit GUI.

**Configuration Management**:
- **tomllib/tomli**: TOML config parsing.
- **typer**: CLI argument handling.

**Scientific Libraries**:
- **sympy**: Sensitivity analysis.
- **networkx**: Graph analysis for transmission topology.
- **scipy**: Statistical functions, optimization routines.

**Database/Storage**:
- **parquet**: Supported via pandas for cached forecast data.