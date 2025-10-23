# Phase 1 — Transmission (EI Edges)

## Files
- Input CSV: `input/engine/transmission/ei_edges.csv`
- Loader: `engine/data_loaders/transmission.py` (`load_edges()`)
- Validator: `engine/validation/validators.py::assert_edges_valid`
- ETL scaffold: `scripts/build_ei_edges_from_sources.py`

## Columns
from_zone,to_zone,season,limit_mw,wheel_cost_per_mwh,contracted_reserve_mw,loss_pct,rule_tag,effective_start,effective_end,source_doc,build_timestamp

## Rules
- One row per direction and season.
- Include special cases via `rule_tag` (e.g., HQ_PhaseII).
- Keep loss_pct in [0, 0.2].
- Ensure reverse direction exists for each forward edge.
