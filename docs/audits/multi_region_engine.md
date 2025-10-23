# Multi-Region Engine Audit

## Scope and methodology
- Reviewed registry definitions, helper scaffolding, and engine integration for canonical Eastern Interconnection regions.
- Inspected frame validation, dispatch solvers, annual aggregation, and output packaging to confirm regional identifiers propagate end-to-end.
- Augmented automated tests to exercise multi-region runs and zero-fill behaviour.

## Findings by component

### 1. Registry and helpers
- `regions/registry.py` enumerates all EI zones; helper classes in `regions/__init__.py` expose matching `region_id` attributes for each entry, ensuring consistent instantiation.
- `DEFAULT_REGION_ID` usage is limited to fixtures/tests (e.g., `tests/fixtures/dispatch_single_minimal.py`) and is not hard-coded inside production engine paths.
- Recommendation: none.

### 2. Frame inputs
- Added `_validate_region_labels` in `granite_io/frames_api.py` to enforce that demand, units, transmission, and coverage frames only reference canonical region IDs. The helper trims whitespace, rejects missing values, and raises on unknown labels.
- Demand and policy loaders already emit per-region records; new validation guarantees future data sources cannot regress to placeholder names.
- Transmission interfaces now share a consolidated schema with explicit `InterfaceSpec` typing so import/export directions, limits, and wheeling charges are harmonised before dispatch.
- Recommendation: extend similar validation to any future optional frames (e.g., capacity expansion) when introduced.

### 3. Dispatch engine
- `dispatch.lp_network.solve_from_frames` and `dispatch.lp_single.solve` tag generators with canonical regions and return per-region aggregates through `DispatchResult` (`generation_by_region`, `emissions_by_region`, `region_prices`, `flows`).
- No remaining fallbacks to "system" within solvers; the only guard lives in the annual aggregator when a dispatch result lacks region splits.
- Recommendation: Consider surfacing a debug warning when a solver returns empty `emissions_by_region` to avoid silently defaulting to "system" during future refactors.

### 4. DispatchResult contract
- `dispatch/interface.py` exposes region-aware attributes and is now exercised across the test suite with canonical IDs.
- Cost/capacity breakdowns remain fuel-centric; region dictionaries are provided for emissions, generation, prices, flows, and coverage.
- Recommendation: evaluate whether per-region cost/capacity summaries are needed alongside fuel-based tables (see gaps below).

### 5. Annual aggregation
- `_build_engine_outputs` in `engine/run_loop.py` merges dispatch payloads, injecting zero rows for regions missing explicit data so every REGIONS entry is represented in emissions, prices, and flows. Self-flows are zero-filled when interties are absent.
- Aggregated emissions totals reconcile against `emissions_total`; audits already enforce per-fuel reconciliation.
- Gap: aggregator still records a synthetic `"system"` bucket when dispatch supplies no regional splits. Should be phased out once all solvers guarantee canonical coverage.

### 6. EngineOutputs bundle
- `engine/outputs.py` exports regional emissions, prices, and flows. Global generation/capacity/cost tables remain fuel-oriented; no per-region tables exist today.
- Gap: to achieve symmetry, introduce additional dataframes (`generation_by_region`, `capacity_by_region`, `cost_by_region`) built from `DispatchResult.generation_by_region` and corresponding cost maps during aggregation.
- Gap: stranded unit reporting is global; consider a region column to align with registry usage.

### 7. Tests
- Updated unit and integration tests to rely exclusively on canonical region IDs.
- Added regression (`test_three_region_run_populates_active_tables`) verifying multi-region runs populate emissions, price, and flow tables for active regions with zero self-flows when no interties exist.
- Strengthened `test_engine_outputs_include_zero_rows_for_empty_regions` to assert `emissions_by_region` contains every REGIONS entry even when unused.

## Gap summary & recommended follow-ups
- **Per-region reporting**: Extend `_build_engine_outputs` and `EngineOutputs` to materialize per-region generation, capacity, cost, and stranded-unit tables sourced from existing `DispatchResult` maps.
- **System fallback**: Remove the legacy `"system"` bucket once dispatch solvers universally emit canonical region IDs; until then emit a warning to highlight missing regionalisation.
- **Audit coverage**: When capacity expansion modules arrive, ensure their frame validators reuse `_validate_region_labels` to prevent regressions.

