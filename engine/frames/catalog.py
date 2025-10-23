"""Declarative catalog describing frame metadata and validators."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, MutableMapping

try:  # pragma: no cover - optional dependency for schema typing
    import pyarrow as pa
except Exception:  # pragma: no cover - optional dependency guard
    pa = None  # type: ignore

try:  # pragma: no cover - optional dependency guard for runtime validation
    import pandas as pd
except Exception:  # pragma: no cover - fallback when pandas unavailable
    pd = None  # type: ignore


Validator = Callable[["pd.DataFrame"], "pd.DataFrame"]


@dataclass(frozen=True)
class FrameSpec:
    """Static description of a frame and its validation contract."""

    name: str
    schema: "pa.Schema | None" = None
    index: Sequence[str] = field(default_factory=tuple)
    vector_fields: Mapping[str, type] | None = None
    validators: Sequence[Validator] = field(default_factory=tuple)
    description: str | None = None

    def apply_validators(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Run the registered validators returning a defensively copied frame."""

        if pd is None:
            return df
        validated = df.copy(deep=True)
        for validator in self.validators:
            validated = validator(validated)
        return validated


class FrameCatalog:
    """Registry mapping frame names to :class:`FrameSpec` definitions."""

    def __init__(self, specs: Iterable[FrameSpec] | None = None):
        self._specs: MutableMapping[str, FrameSpec] = OrderedDict()
        if specs:
            for spec in specs:
                self.register(spec)

    def register(self, spec: FrameSpec) -> None:
        key = spec.name.lower()
        self._specs[key] = spec

    def get(self, name: str) -> FrameSpec | None:
        return self._specs.get(name.lower())

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._specs

    def __iter__(self):  # pragma: no cover - trivial iteration helper
        return iter(self._specs.values())


def default_catalog() -> FrameCatalog:
    """Return a catalog pre-populated with the canonical frame specs."""

    specs: list[FrameSpec] = [
        FrameSpec(
            name="demand",
            description="Annual demand in MWh by region.",
            index=("year", "region"),
        ),
        FrameSpec(
            name="peak_demand",
            description="Peak demand in MW by region.",
            index=("year", "region"),
        ),
        FrameSpec(
            name="units",
            description="Generating unit fleet characteristics.",
            index=("unit_id",),
        ),
        FrameSpec(
            name="transmission",
            description="Transmission interfaces and limits.",
            index=("interface_id",),
        ),
        FrameSpec(
            name="policy",
            description="Allowance policy configuration records.",
            index=("year",),
        ),
    ]
    return FrameCatalog(specs)
