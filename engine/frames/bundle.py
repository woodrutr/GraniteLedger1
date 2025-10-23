"""Typed bundle describing validated inputs for the engine run loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

try:  # pragma: no cover - optional dependency guard during type checking
    import pandas as pd
except Exception:  # pragma: no cover - fallback when pandas is unavailable
    pd = None  # type: ignore

from engine.frames.vector import VectorRegistry

try:  # pragma: no cover - prefer refactored IO namespace
    from granite_io.frames_api import Frames, PolicySpec  # type: ignore
except Exception:  # pragma: no cover - fallback when refactor package unavailable
    Frames = object  # type: ignore
    PolicySpec = object  # type: ignore


@dataclass(frozen=True)
class ModelInputBundle:
    """Immutable container wrapping validated frame artifacts."""

    frames: "Frames"
    vectors: VectorRegistry
    years: Tuple[int, ...]
    policy: "PolicySpec | None" = None
    meta: Mapping[str, object] | None = None

    def with_years(self, years: Iterable[int]) -> "ModelInputBundle":
        """Return a new bundle with ``years`` replaced."""

        normalized_years = tuple(sorted({int(year) for year in years}))
        return ModelInputBundle(
            frames=self.frames,
            vectors=self.vectors,
            years=normalized_years,
            policy=self.policy,
            meta=self.meta,
        )
