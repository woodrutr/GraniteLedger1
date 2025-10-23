"""Helpers for managing vector-aware frame columns."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

try:  # pragma: no cover - optional dependency guard
    import pandas as pd
except Exception:  # pragma: no cover - fallback
    pd = None  # type: ignore

try:  # pragma: no cover - optional import for price vectors
    from engine.prices.types import CarbonPriceVector
except Exception:  # pragma: no cover - fallback when pricing module unavailable
    CarbonPriceVector = object  # type: ignore


@dataclass(frozen=True)
class VectorColumn:
    """Representation of a structured column backed by vector data."""

    name: str
    values: Mapping[int, float] | CarbonPriceVector | Iterable[float]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_schedule(self) -> Mapping[int, float]:
        """Return the column as a year-indexed schedule when possible."""

        if isinstance(self.values, Mapping):
            return {int(year): float(value) for year, value in self.values.items()}
        if isinstance(self.values, CarbonPriceVector):  # pragma: no branch - simple predicate
            schedule = {int(year): float(self.values.at(year)) for year in self.values.years}
            return schedule
        if pd is not None:
            series = pd.Series(list(self.values), dtype=float)
            return {int(index): float(value) for index, value in enumerate(series.tolist(), start=1)}
        return {int(index): float(value) for index, value in enumerate(self.values, start=1)}


class VectorRegistry:
    """Light-weight registry for vectorized payloads referenced by frames."""

    def __init__(self):
        self._registry: dict[str, VectorColumn] = {}

    def register(self, column: VectorColumn) -> None:
        self._registry[column.name] = column

    def get(self, name: str) -> VectorColumn:
        if name not in self._registry:
            raise KeyError(f"vector '{name}' has not been registered")
        return self._registry[name]

    def maybe_get(self, name: str) -> VectorColumn | None:
        return self._registry.get(name)

    def as_price_schedule(self, name: str) -> Mapping[int, float] | None:
        column = self.maybe_get(name)
        if column is None:
            return None
        return column.as_schedule()

    def __contains__(self, name: str) -> bool:  # pragma: no cover - trivial helper
        return name in self._registry

    def __iter__(self):  # pragma: no cover - trivial helper
        return iter(self._registry.items())
