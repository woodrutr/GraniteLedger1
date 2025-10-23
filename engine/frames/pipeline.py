"""Core frame ingestion and publishing pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Dict

try:  # pragma: no cover - optional dependency guard
    import pandas as pd
except Exception:  # pragma: no cover - fallback when pandas unavailable
    pd = None  # type: ignore

from engine.frames.bundle import ModelInputBundle
from engine.frames.catalog import FrameCatalog, FrameSpec, default_catalog
from engine.frames.vector import VectorColumn, VectorRegistry

try:  # pragma: no cover - prefer refactored IO namespace
    from granite_io.frames_api import Frames, PolicySpec  # type: ignore
except Exception:  # pragma: no cover - fallback when refactor package unavailable
    Frames = object  # type: ignore
    PolicySpec = object  # type: ignore

try:  # pragma: no cover - reuse normalization helpers from the run loop
    from engine.run_loop import _expand_price_schedule, _prepare_carbon_price_schedule
except Exception:  # pragma: no cover - fallback when run loop helpers unavailable
    _expand_price_schedule = None  # type: ignore
    _prepare_carbon_price_schedule = None  # type: ignore


@dataclass(frozen=True)
class FrameDraft:
    """Mutable draft emitted by the ingestion layer."""

    name: str
    data: "pd.DataFrame"
    spec: FrameSpec
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameArtifact:
    """Immutable artifact exposed to consumers after validation."""

    name: str
    data: "pd.DataFrame"
    spec: FrameSpec
    metadata: Mapping[str, Any] = field(default_factory=dict)


class FramePipeline:
    """End-to-end orchestrator for frame ingestion, validation, and publishing."""

    def __init__(self, catalog: FrameCatalog | None = None):
        self._catalog = catalog or default_catalog()

    @property
    def catalog(self) -> FrameCatalog:
        return self._catalog

    def _ingest(self, name: str, value: object) -> FrameDraft:
        if pd is None:
            raise RuntimeError("pandas is required to ingest frame inputs")
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"frame '{name}' must be provided as a pandas DataFrame")
        spec = self._catalog.get(name) or FrameSpec(name=name)
        return FrameDraft(name=name, data=value.copy(deep=True), spec=spec)

    def _publish(self, draft: FrameDraft) -> FrameArtifact:
        data = draft.spec.apply_validators(draft.data)
        return FrameArtifact(name=draft.name, data=data, spec=draft.spec, metadata=draft.metadata)

    def _normalize_carbon_schedule(
        self,
        *,
        schedule: Mapping[Any, Any] | float | None,
        value: float | None,
        years: Iterable[int] | None,
    ) -> dict[int, float] | None:
        if schedule is None and value in (None, ""):
            return None
        if _prepare_carbon_price_schedule is None or _expand_price_schedule is None:
            if isinstance(schedule, Mapping):
                return {int(k): float(v) for k, v in schedule.items() if k is not None}
            if isinstance(value, (int, float)):
                if years:
                    return {int(year): float(value) for year in years}
                return {}
            return None
        normalized = _prepare_carbon_price_schedule(schedule, value)
        expanded = _expand_price_schedule(normalized, years or [])
        return expanded if expanded else dict(normalized)

    def build_bundle(
        self,
        inputs: Mapping[str, "pd.DataFrame"],
        *,
        years: Iterable[int] | None = None,
        carbon_policy_enabled: bool = True,
        banking_enabled: bool = True,
        carbon_price_schedule: Mapping[Any, Any] | float | None = None,
        carbon_price_value: float | None = None,
    ) -> ModelInputBundle:
        if pd is None:
            raise RuntimeError("pandas is required to build frame bundles")
        drafts: Dict[str, FrameDraft] = {}
        for name, df in inputs.items():
            drafts[name.lower()] = self._ingest(name, df)

        artifacts: Dict[str, FrameArtifact] = {}
        for name, draft in drafts.items():
            artifacts[name] = self._publish(draft)

        frames_input = {name: artifact.data for name, artifact in artifacts.items()}
        frames = Frames.coerce(
            frames_input,  # type: ignore[arg-type]
            carbon_policy_enabled=carbon_policy_enabled,
            banking_enabled=banking_enabled,
            carbon_price_schedule=carbon_price_schedule
            if isinstance(carbon_price_schedule, Mapping)
            else None,
        )

        vector_registry = VectorRegistry()
        schedule = self._normalize_carbon_schedule(
            schedule=carbon_price_schedule,
            value=carbon_price_value,
            years=years,
        )
        if schedule:
            vector_registry.register(
                VectorColumn(
                    name="carbon_price",
                    values=schedule,
                    metadata={"source": "config", "kind": "price_schedule"},
                )
            )

        normalized_years = tuple(sorted({int(year) for year in years})) if years else tuple(frames.years())

        policy_spec: PolicySpec | None = None
        if carbon_policy_enabled:
            try:
                policy_spec = frames.policy()  # type: ignore[assignment]
            except Exception:
                policy_spec = None

        bundle_meta: Dict[str, Any] = {}
        if schedule:
            bundle_meta["carbon_price_schedule"] = schedule

        return ModelInputBundle(
            frames=frames,
            vectors=vector_registry,
            years=normalized_years,
            policy=policy_spec,
            meta=bundle_meta,
        )
