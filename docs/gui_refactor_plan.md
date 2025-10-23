# GUI Helper Module Refactor Plan

## Goals and Constraints
- Preserve the current Streamlit experience provided by `gui/app.py` while reducing its size and complexity.
- Consolidate related helper functions into focused modules so that domain logic (data discovery, configuration parsing, schedules, etc.) can be reused and tested independently.
- Keep the orchestration, layout definitions, and stateful Streamlit callbacks inside `gui/app.py`.
- Avoid breaking imports for downstream users; provide compatibility aliases during migration when needed.

## Current Observations
- Forecast discovery, caching, and normalization helpers (_e.g._ `_regions_available_iso_scenarios`, `_discover_iso_scenarios`, `_cached_forecast_frame`) occupy the first several hundred lines of `gui/app.py` and primarily manipulate engine I/O rather than UI concerns.【F:gui/app.py†L66-L756】
- Session tracking and staged input helpers (`_ensure_progress_state`, `_update_staged_run_inputs`, `_trigger_streamlit_rerun`, etc.) form another cluster of non-UI logic that only depends on `st.session_state` interactions.【F:gui/app.py†L1702-L1813】
- Configuration parsing and normalization helpers (`_load_config_data`, `_years_from_config`, `_regions_from_config`, `_normalize_region_labels`, etc.) convert raw TOML and JSON-like payloads into canonical structures for the widgets.【F:gui/app.py†L2930-L3240】
- Policy schedule utilities (`_normalize_price_schedule`, `_build_price_schedule`, `_build_cap_reduction_schedule`, etc.) transform numeric inputs but are not tied to Streamlit APIs.【F:gui/app.py†L2220-L2459】
- Demand/output frame builders (`_build_demand_output_frame`, downloads directory helpers) convert curated selections into pandas frames and could live in a data-centric helper module.【F:gui/app.py†L2584-L2858】

## Proposed Module Breakdown
1. **`gui/forecast_helpers.py`**
   - Responsibilities: ISO discovery, zone lookup, scenario loading, cached forecast table assembly, and forecast summary utilities.
   - Candidate moves: `_regions_available_iso_scenarios`, `_regions_available_zones`, `_discover_iso_scenarios`, `_discover_iso_zones`, `_load_iso_scenario_frame`, `_cached_input_root`, `_cached_forecast_frame`, `_scenario_frame_subset`, `_summarize_forecasts`, and related helpers that only touch engine loaders and pandas.【F:gui/app.py†L66-L756】【F:gui/app.py†L1127-L1178】
   - `gui/app.py` will import a thin facade (e.g., `from .forecast_helpers import cached_forecast_frame`) to keep call sites unchanged.

2. **`gui/session_state.py`**
   - Responsibilities: encapsulate Streamlit session keys, run token resets, staged run input management, rerun triggers, and safe coercion helpers that gate UI state.
   - Candidate moves: `_ensure_progress_state`, `_reset_progress_state`, `_staged_run_inputs_state`, `_update_staged_run_inputs`, `_trigger_streamlit_rerun`, `_bounded_percent`, `_safe_int`, `_safe_float`, and related constants like `_SESSION_RUN_TOKEN_KEY` and `_SCRIPT_ITERATION_KEY`.【F:gui/app.py†L673-L1813】
   - Provide an interface such as `session.ensure_progress_state()` so the GUI file reads clearly.

3. **`gui/config_parsers.py`**
   - Responsibilities: load configuration sources, derive year ranges, normalize regions/states/coverage, and bridge to canonical region metadata.
   - Candidate moves: `_load_config_data`, `_years_from_config`, `_select_years`, `_regions_from_config`, `_normalize_state_codes`, `_states_from_config`, `_normalize_region_labels`, `_resolve_canonical_region`, `_normalize_coverage_selection`, `_coverage_default_display`, `_normalize_cap_region_entries`, plus helper accessors for region lookup caches.【F:gui/app.py†L2930-L3378】
   - This module can expose pure functions for config ingestion with small dependency surface (pathlib, tomllib, normalization utilities).

4. **`gui/policy_schedules.py`**
   - Responsibilities: sanitize periods, merge/expand price schedules, build escalator and cap-reduction schedules, and other numeric schedule helpers.
   - Candidate moves: `_sanitize_control_period`, `_normalize_price_schedule`, `_merge_price_schedules`, `_expand_or_build_price_schedule`, `_build_price_schedule`, `_build_price_escalator_schedule`, `_build_cap_reduction_schedule`, and related helpers like `_apply_schedule_growth`/`_coerce_year_value_map` later in the file.【F:gui/app.py†L2220-L2476】【F:gui/app.py†L5670-L5796】
   - Keeps numerical policy math in one place for easier unit testing.

5. **`gui/demand_builders.py`**
   - Responsibilities: build demand frames, assemble default frames (`_build_default_frames`, `_default_units`, etc.), and ensure year coverage for run inputs.
   - Candidate moves: `_build_demand_output_frame`, `_build_default_frames`, `_ensure_years_in_demand`, `_apply_region_weights_to_frames`, `_available_regions_from_frames`, `_build_coverage_frame`, `_default_units`, `_default_fuels`, `_default_transmission` and related helpers.【F:gui/app.py†L2584-L2858】【F:gui/app.py†L5670-L6116】
   - Module would return pandas objects ready for consumption by Streamlit widgets.

6. **`gui/output_export.py`**
   - Responsibilities: temporary output directory management, writing results to disk, extracting output dataframes, and normalizing dispatch/price frames for downloads.
   - Candidate moves: `_temporary_output_directory`, `_write_outputs_to_temp`, `_extract_output_dataframe`, `_normalize_dispatch_price_frame`, `_with_legacy_carbon_price_columns`, `_read_uploaded_dataframe`, `_validate_frame_override`, and cleanup helpers like `_cleanup_session_temp_dirs` that deal with filesystem operations.【F:gui/app.py†L6116-L8697】
   - Keeps filesystem and pandas post-processing separate from UI layout.

7. **UI Section Modules (optional second phase)**
   - Once the pure helpers move out, consider splitting large renderer clusters (`_render_demand_module_section`, `_render_carbon_policy_section`, `_render_outputs_section`, etc.) into domain-specific view modules (e.g., `gui/views/demand.py`). These functions are tightly coupled to Streamlit but can still be grouped for readability.【F:gui/app.py†L3807-L8834】
   - This phase can follow after helper extraction to avoid massive churn.

## Migration Strategy
1. **Introduce Modules Incrementally**: Create new helper modules with copied functions and add thin wrappers or re-export the old names to prevent breaking call sites. Update imports in `gui/app.py` once a group is relocated.
2. **Add Unit Coverage**: For each new helper module, port existing tests or create focused unit tests (especially for schedule math and config parsing) to guard behavior.
3. **Maintain Backwards Compatibility**: Temporarily leave delegating functions in `gui/app.py` (calling into the new module) so external imports continue to work during transition; remove duplicates after dependent code is updated.
4. **Document Module APIs**: Include module-level docstrings describing responsibilities, and update developer docs to reflect the new structure.
5. **Refine Type Hints and Visibility**: Convert internal-only helpers to module-private (prefix `_`) within the helper files and expose public facades where appropriate, clarifying which utilities are safe to import elsewhere.

## Validation Plan
- Run the existing GUI smoke tests or minimal Streamlit run to ensure the app still launches.
- Execute automated unit tests (`pytest`) focusing on the migrated helpers.
- Perform targeted manual runs for forecasting, configuration upload, and policy schedule editing to ensure UI bindings still work.

## Sequencing and Ownership
1. Start with forecast/data-loading helpers (least coupling to UI) to prove the pattern.
2. Follow with configuration parsing and policy schedules to enable unit coverage.
3. Move session state helpers once confidence grows, because Streamlit state bugs can be disruptive.
4. Address demand/dataframe builders and output exporters afterward, coordinating with engine-team stakeholders who rely on these utilities.
5. Optionally split UI rendering modules as a follow-up milestone once helper migrations land and tests are stable.

