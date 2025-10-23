# GraniteLedger Developer Guide

Welcome to GraniteLedger! This document is a quick orientation for new
contributors who want to understand how data flows through the platform—from
input ingestion, through the dispatch and allowance engines, to the graphical
interface that visualises results.

## 1. Input Data Flow

1. **Data loading** – User supplied CSVs are converted into Pandas DataFrames and
   wrapped inside the :class:`granite_io.frames_api.Frames` container. The loader applies
   schema validation, enforces primary keys, and caches metadata such as carbon
   policy enablement and price schedules. Load forecast ingestion now always
   routes through the strict validator, removing the previous legacy fallback
   path and ensuring any malformed CSVs surface descriptive errors immediately.
2. **Scenario construction** – Helper methods (e.g. `Frames.with_frame`) allow
   tests and tools to inject modified data while preserving validation logic.
3. **Demand and units** – The dispatch engine consumes demand profiles via
   `Frames.demand_for_year`, generator characteristics from `Frames.units`, and
   interface definitions from `Frames.interfaces`. All inputs are normalised to
   canonical region identifiers before the optimisation routines run.
4. **Policy specification** – Carbon allowance configuration is provided by the
   `Frames.policy()` accessor. It produces a `PolicySpec` wrapper that converts
   frame data into an :class:`policy.allowance_annual.RGGIPolicyAnnual`
   instance with cap trajectories, banking rules, CCR triggers, and compliance
   year bookkeeping.

## 2. Dispatch and Allowance Computation

1. **Dispatch solvers** – Depending on the configuration, either the
   single-region merit-order solver (`dispatch.lp_single.solve`) or the network
   linear programme (`dispatch.lp_network.solve_from_frames`) is selected. Both
   return a :class:`dispatch.interface.DispatchResult` describing generation,
   prices, flows, emissions, and coverage metadata.
2. **Allowance market** – The engine iterates between dispatch and carbon
   policy. `_solve_allowance_market_year` performs a bisection search on the
   allowance price, invoking the chosen dispatch solver until emissions and
   allowance supply converge within tolerance.
3. **Fixed-point integration** – `engine.run_loop.run_fixed_point_from_frames`
   coordinates multi-year runs. It normalises price schedules, computes period
   weights for compliance years, and constructs dispatch callbacks that apply
   carbon price adders when required. Banking balances, CCR releases, and
   shortages are all accounted for year-over-year.
4. **Output assembly** – `_build_engine_outputs` packages the annual summaries,
   regional emissions, price trajectories, and transmission flows into an
   :class:`engine.outputs.EngineOutputs` dataclass ready for persistence.

## 3. GUI Consumption

1. **Persistence** – CLI and automation paths typically persist EngineOutputs
   to disk via `EngineOutputs.to_csv(...)`, producing the CSV files expected
   by the GUI.
2. **Interactive application** – The Streamlit GUI (`gui/app.py`) reads the
   serialized outputs, enriches them with metadata, and presents interactive
   charts and tables.
3. **User interaction loop** – Adjustments to demand, policy settings, or
   technology assumptions trigger a rebuild of the Frames bundle followed by a
   rerun of the engine. Cached frames ensure that repeated operations remain
   snappy while still honouring validation rules.

## 4. Testing Strategy

* **Unit tests** cover validation helpers, dispatch utilities, carbon policy
  primitives, and output formatting to give quick feedback on core logic.
* **Integration tests** exercise a three-year scenario end-to-end, verifying
  that each output frame is populated and that the allowance market converges.

Happy hacking! If you get stuck, browse the `tests/fixtures/` directory for
examples of constructing scenarios entirely in memory.

## 5. Region Identity and State Mapping

* **Canonical identifiers** – Region names coming from input CSVs are normalised
  via `engine.normalization.normalize_region_id` so that dispatch, allowance,
  and GUI components speak a shared language (e.g., `nyiso_z1` becomes
  `NYISO_Z1`). The ingestion layer now rejects any frame whose regions fail to
  resolve to this canonical set, preventing downstream mismatches. Always use
  the canonical ID when creating joins or lookups.
* **State membership** – The base mapping of ISO control areas to constituent
  states lives in `input/regions/iso_state_zones.yaml`. Optional project- or
  scenario-specific overrides are merged in from
  `input/regions/state_to_regions.json`, which enumerates the zones a state
  participates in after normalisation.
* **Share weights** – State apportionment uses
  `input/regions/zone_to_state_share.csv` to describe what fraction of each
  region flows to a given state. The optional `REGION_SHARES_STRICT`
  environment flag (consumed by `engine.run_loop`) will verify that the shares
  cover at least 95% of the regions listed in `state_to_regions.json` and raise
  a descriptive error if any state is under-covered. This helps surface missing
  metadata early when preparing new scenarios.
* Canonical region IDs come only from `regions/*.py`.
* Data files must reference those IDs exactly.
* No inferred shares. If a state lacks weights, fix the CSV/JSON.
