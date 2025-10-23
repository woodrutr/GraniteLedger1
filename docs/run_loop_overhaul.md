# Engine Run Loop Overhaul

## Current Run Loop Responsibilities

The legacy `engine.run_loop` module is the integration point between allowance
market mechanics and the dispatch solvers. The existing implementation is a
monolithic file that mixes pure helpers, IO validation and iterative control
logic. The major responsibilities are:

1. **Input normalisation** – build carbon price vectors, coerce pandas frames,
   resolve canonical region identifiers and validate zone/state coverage.
2. **Dispatch orchestration** – lazily construct either the network or
   single-zone linear program and scale the results when period weights are
   applied.
3. **Allowance market solving** – implement a bisection-based solver that
   clears the allowance market for a given year, accounting for CCR rules,
   banking limits and surrender fractions.
4. **Annual fixed-point iteration** – iterate between dispatch outcomes and
   allowance prices to converge on a consistent market clearing price series.
5. **Output construction** – aggregate dispatch payloads into
   `EngineOutputs`, run audits and produce annual summary tables that match the
   public CSV layout.
6. **Operational hardening** – enforce configuration guardrails (missing
   frames, price schedule coverage, network requirements) and emit progress
   callbacks for the GUI.

These responsibilities are entangled within a single 3,000+ line module, which
makes testing, reasoning about state transitions and securing invariants
challenging.

## Proposed Clean Rewrite

The rewrite focuses on decomposing the run loop into explicit collaborators
with well defined responsibilities:

- **State containers** encapsulate mutable engine state (`FixedPointRuntimeState`
  for allowance banking) so control flow is explicit and serialisable.
- **Coordinator classes** (starting with `AnnualFixedPointRunner`) own the
  high-level iteration logic while depending on the existing domain helpers for
  policy calculations, dispatch solving and auditing.
- **Validation gateways** remain as pure functions that can be exercised in
  isolation, giving us well scoped unit tests and easier reasoning about
  failure modes.
- **Progress/event emitters** become injectable callbacks so UI clients can
  observe deterministic stage transitions without relying on module level
  globals.

The same strategy will be applied incrementally to the dispatch builder and
end-to-end orchestration functions. Each extracted component keeps the existing
interfaces so downstream callers (GUI, CLI, tests) continue to operate without
adjustments.

## Rewrite Project Plan

1. **Inventory & Documentation (complete)** – capture the existing run loop
   responsibilities and interactions in this document.
2. **Introduce explicit runtime containers (complete)** – add dataclasses to
   model mutable state, ensuring that transitions are auditable and safe.
3. **Refactor annual fixed-point iteration (complete)** – move the loop into an
   `AnnualFixedPointRunner` coordinator while preserving behaviour.
4. **Modularise dispatch orchestration (next)** – extract a dispatcher factory
   that separates frame validation from LP execution strategy.
5. **Modularise end-to-end orchestration (next)** – wrap `run_end_to_end` inside
   a class that coordinates validation, dispatch, allowance solving and output
   aggregation, enabling better resource management and targeted tests.
6. **Progressive hardening (next)** – once the coordinators exist, add
   type-safe configuration objects and signature checks so invalid manifests are
   rejected early.
7. **Regression coverage (ongoing)** – keep running the existing test suite and
   extend it with unit tests that target the new coordinators to guard the
   refactor.

## Work Completed in This Change

- Added the `FixedPointRuntimeState` dataclass and the
  `AnnualFixedPointRunner` coordinator, removing the deeply nested function and
  clarifying how allowance state, bank balances and convergence criteria are
  handled.
- Preserved the public API (`run_annual_fixed_point`) while routing it through
  the new coordinator for immediate behavioural parity and easier future
  enhancements.
- Verified the refactor against the allowance regression test suite and the
  dispatch fixed-point integration test.

This change lays the foundation for extracting the remaining orchestration
pieces into composable units in subsequent iterations.
