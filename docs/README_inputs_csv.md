# CSV Engine Inputs

Place CSVs under `input/engine/` so the engine prefers them without GUI changes.

## Transmission
File: `input/engine/transmission/ei_edges.csv`

Columns:
- from_zone,to_zone,season,limit_mw,wheel_cost_per_mwh,contracted_reserve_mw,loss_pct,rule_tag,effective_start,effective_end,source_doc,build_timestamp

## Units
File: `input/engine/units/us_units.csv`

Columns:
- plant_id_eia,generator_id,unit_id_gl,unit_name,owner_name,lat,lon,zone,state,fuel,nameplate_mw,net_max_mw,heat_rate_mmbtu_per_mwh,co2_t_per_mwh,vom_per_mwh,start_cost_usd,min_up_hours,min_down_hours,ramp_mw_per_min,online_year,retire_year,covered,source_doc,build_timestamp

### Legacy EI-format fleet file

The GUI and orchestrator continue to look for the legacy EI-format CSV at
`input/ei_units.csv`. The helper that populates the dispatch tables first tries
to read that file before falling back to the engine catalog above. If the file
is missing, `_build_default_frames` fails and the sidebar shows the warning
``Dispatch requires demand and unit data, but no frames are available.`` To load
custom units, either place the EI-format file at `input/ei_units.csv` or pass a
`Frames` object that already contains a non-empty `units` DataFrame.

## Finance
Files: `input/engine/finance/atb_wacc.csv`, `input/engine/finance/atb_capex.csv`
