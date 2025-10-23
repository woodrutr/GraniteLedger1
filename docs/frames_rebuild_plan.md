# Frames Architecture Rebuild Proposal

## Context

The existing frame layer grew organically around a mixture of legacy CSV ingestion, shims that
mirror the original `io.frames_api` contract, and one-off helpers inside the electricity
preprocessor. The result is a confusing split between mutable dictionaries (for example the
`FrameStore` helper in `src/models/electricity/scripts/preprocessor.py`) and immutable containers
(`granite_io.frames_api.Frames`) that are only partially validated. Downstream, the orchestrator and
engine run loop rely on duck-typed access to those frames, which makes it difficult to pass the
vectorized data objects (such as `CarbonPriceVector` in `engine/run_loop.py`) through the pipeline
without re-normalizing the same information multiple times.

A clean-sheet rebuild is needed to make the frame layer explicit, type-safe, and vector-friendly so
that the orchestrator can hand off a coherent data bundle to the run loop.

## Goals

1. **Explicit data contracts.** Every frame should declare its schema, required joins, and canonical
   identifier fields in one place.
2. **Vector-friendly payloads.** The frame layer must carry structured objects (for example price
   vectors) without flattening them into scalar columns and reconstructing them later.
3. **Deterministic transformations.** Inputs from CSV/SQL/REST need a reproducible normalization
   pipeline that records provenance and intermediate versions of each frame.
4. **Composable orchestration.** The orchestrator should assemble a `ModelInputBundle` that exposes
   typed accessors for the run loop instead of ad hoc dictionary lookups.
5. **Backward compatibility shims.** Provide a narrow compatibility surface so existing callers can
   migrate incrementally.

## Proposed Architecture

### 1. Declarative Frame Catalog

Create a `engine/frames/catalog.py` module that declares the canonical frames using typed metadata:

```python
@dataclass(frozen=True)
class FrameSpec:
    name: str
    schema: pa.Schema
    index: Sequence[str]
    vector_fields: Mapping[str, Type[VectorLike]] = field(default_factory=dict)
    validators: Sequence[Callable[[pd.DataFrame], pd.DataFrame]] = ()
```

The catalog lists each frame, including nested vector fields (for instance, the allowance price
vector) and reusable validators. This replaces the scattered validation routines in
`granite_io/frames_api.py` with declarative registrations.

### 2. Frame Pipeline Core

Introduce a `FramePipeline` service that coordinates ingestion, validation, enrichment, and
materialization:

1. **Ingestors** convert raw sources (CSV, SQL, API responses) into pandas dataframes according to
   the catalog schema.
2. **Transformers** operate on typed `FrameDraft` objects that track metadata (source, version,
   applied transforms) and support vector columns.
3. **Publishers** finalize `FrameDraft` objects into immutable `FrameArtifact` instances that back
   the `Frames` mapping exposed to the rest of the engine.

Every stage emits structured provenance records, making it easy to debug or replay transformations.

### 3. Vector-Aware Columns

Replace manual conversion to/from scalar columns by introducing a `VectorColumn` helper that wraps a
vectorized object plus its metadata (e.g., fiscal year, units). The helper knows how to serialize to
flat columns when exporting, but within the engine it keeps the vector instance intact.

For example, a `CarbonPriceVector` column would be stored as a `VectorColumn` with a backing list of
floats and methods to evaluate at a given compliance year. The run loop can consume the object
without rehydrating it from strings.

### 4. Engine Bundle

Define a `ModelInputBundle` dataclass in `engine/io/bundle.py` that captures the validated outputs of
the pipeline:

```python
@dataclass(frozen=True)
class ModelInputBundle:
    frames: Frames
    vectors: VectorRegistry
    years: tuple[int, ...]
    policy: PolicySpec | None
```

The orchestrator constructs a bundle once and passes it to the run loop, which now operates on a
single object instead of juggling multiple mappings. The bundle also records feature toggles (for
example, whether banking is enabled) so the run loop no longer needs to infer them from individual
frames.

### 5. Compatibility Layer

Expose a `frames_legacy` module that adapts the new catalog to the current `Frames` interface. The
adapter simply reads the `ModelInputBundle` and populates legacy accessors so legacy code paths keep
working while we migrate call sites. The adapter is intentionally thin: it should emit clear deprecation
warnings whenever a consumer relies on an implicit schema or mutates frames in place.

## Integration Plan

1. **Bootstrap the catalog.** Extract the existing validation logic from
   `granite_io/frames_api.py` into catalog entries. Add schema definitions using `pyarrow` (already
   available via pandas) for consistent typing.
2. **Build ingestion adapters.** Wrap the existing electricity preprocessor outputs so they emit
   `FrameDraft` objects rather than mutating dictionaries. This removes the `FrameStore`
   indirection, making transformations explicit.
3. **Implement vector columns.** Start with carbon pricing data because `engine/run_loop.py`
   already works with `CarbonPriceVector`. Replace the ad hoc `_cp_from_entry` logic with an
   evaluation method on the vector column class and ensure the orchestrator registers those vectors
   in the bundle.
4. **Bundle hand-off.** Update `engine/orchestrate.py` to request a `ModelInputBundle` from the
   pipeline. Refactor the run loop to accept the bundle, using typed accessors for demand, units,
   transmission, and policy.
5. **Legacy bridge.** Provide adapters that expose the old dictionary-style API (including
   `FrameStore`) backed by the new bundle to maintain backward compatibility for external scripts
   and tests.
6. **Progressive rollout.** Migrate modules feature-by-feature (e.g., demand → units → policy),
   deleting shims once no longer used. Each migration should include focused integration tests that
   validate the bundle against the current regression suite.

## Deliverables

- `engine/frames/` package containing the catalog, pipeline core, vector helpers, and bundle types.
- Updated orchestrator and run loop signatures that accept `ModelInputBundle` instead of raw frame
  dictionaries.
- Compatibility adapters (`io_loader.py`, `src/models/electricity/scripts/preprocessor.py`) that
  translate old patterns into the new bundle during the transition phase.
- Documentation describing the catalog schema and migration playbook for downstream consumers.

## Risks and Mitigations

- **Schema drift:** Mitigate by enforcing schema checks at pipeline boundaries and generating
  documentation from the catalog definitions.
- **Performance regressions:** Cache intermediate `FrameArtifact` objects keyed by source hash so
  repeated runs avoid rebuilding frames when inputs are unchanged.
- **Adoption friction:** Provide codemods and linters that flag direct dictionary access to frames,
  guiding developers toward the typed bundle.

## Next Steps

1. Stand up the `engine/frames` package skeleton with unit tests for the catalog and pipeline core.
2. Port carbon policy frames as the pilot use case, validating vector column support end-to-end.
3. Roll the migration across remaining frames, deleting legacy shims and updating documentation once
   the engine exclusively consumes `ModelInputBundle` instances.
