"""Utility helpers for technology-specific electricity incentives.

This module provides a light-weight data contract used to capture technology
specific production and investment incentives.  The goal is to expose a single
interface that can be populated from configuration files, GUI inputs, or other
sources and then converted into the structured frames consumed by the
electricity pipeline.

The implementation intentionally focuses on providing a reliable placeholder
that downstream optimisation code can hook into without prescribing specific
behaviour.  The records captured here therefore only store metadata about the
incentives (credit value, eligible quantities, etc.) and defer any actual model
impacts to future work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, MutableMapping

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None

from .technology_metadata import get_technology_label, resolve_technology_key


_PANDAS_REQUIRED_MESSAGE = (
    'pandas is required for technology incentive frame construction; '
    'install it with `pip install pandas`.'
)


def _ensure_pandas():
    """Return :mod:`pandas` or raise an informative error."""

    if pd is None:  # pragma: no cover - helper exercised indirectly
        raise ImportError(_PANDAS_REQUIRED_MESSAGE)
    return pd


_CREDIT_TYPE_PRODUCTION = 'production'
_CREDIT_TYPE_INVESTMENT = 'investment'


def _coerce_credit_type(value: object) -> str | None:
    if value in (None, ''):
        return None
    normalized = str(value).strip().lower()
    if normalized in (_CREDIT_TYPE_PRODUCTION, _CREDIT_TYPE_INVESTMENT):
        return normalized
    if normalized.startswith('prod'):
        return _CREDIT_TYPE_PRODUCTION
    if normalized.startswith('inv'):
        return _CREDIT_TYPE_INVESTMENT
    return None


def _coerce_float(value: object) -> float | None:
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object) -> int | None:
    if value in (None, ''):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            coerced = float(value)
        except (TypeError, ValueError):
            return None
        if coerced.is_integer():
            return int(coerced)
        return None


def _coerce_bool_flag(value: object, *, default: bool = True) -> bool:
    if value in (None, ''):
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        raise TypeError('enabled flag must be 0 or 1 if provided as a number')
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {'true', 't', 'yes', 'y', '1', 'on'}:
            return True
        if normalized in {'false', 'f', 'no', 'n', '0', 'off'}:
            return False
        raise ValueError('enabled flag must be a boolean string token')
    raise TypeError('enabled flag must be provided as a boolean value')


@dataclass(frozen=True)
class IncentiveRecord:
    """Representation of a single technology incentive entry."""

    technology_id: int
    year: int
    credit_type: str
    credit_value: float
    limit_value: float | None = None

    @property
    def technology_label(self) -> str:
        return get_technology_label(self.technology_id)

    @property
    def limit_units(self) -> str:
        return 'MWh' if self.credit_type == _CREDIT_TYPE_PRODUCTION else 'MW'

    def to_table_row(self) -> MutableMapping[str, Any]:
        return {
            'type': 'Production' if self.credit_type == _CREDIT_TYPE_PRODUCTION else 'Investment',
            'technology': self.technology_label,
            'year': self.year,
            'credit_value': self.credit_value,
            'limit_value': self.limit_value,
            'limit_units': self.limit_units,
        }

    def to_config_entry(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            'technology': self.technology_label,
            'year': self.year,
        }
        if self.credit_type == _CREDIT_TYPE_PRODUCTION:
            base['credit_per_mwh'] = self.credit_value
            if self.limit_value is not None:
                base['limit_mwh'] = self.limit_value
        else:
            base['credit_per_mw'] = self.credit_value
            if self.limit_value is not None:
                base['limit_mw'] = self.limit_value
        return base


@dataclass
class TechnologyIncentiveModule:
    """Modular application of technology incentive frames."""

    name: str = 'technology_incentives'
    enabled: bool = True
    _inputs: Mapping[str, 'pd.DataFrame'] = field(default_factory=dict)

    @property
    def inputs(self) -> dict[str, 'pd.DataFrame']:
        return {key: value.copy(deep=True) for key, value in self._inputs.items()}

    def apply(self, state: Any) -> Any:
        """Attach the module's frames to ``state`` when enabled."""

        if not self.enabled:
            return state

        assign: Any
        target = state

        if hasattr(target, 'with_frame') and callable(getattr(target, 'with_frame')):
            def assign(name: str, df: 'pd.DataFrame') -> None:
                nonlocal target
                target = target.with_frame(name, df.copy(deep=True))
        elif hasattr(target, '__setitem__'):
            def assign(name: str, df: 'pd.DataFrame') -> None:
                target[name] = df.copy(deep=True)
        else:
            raise TypeError('state must support frame assignment via with_frame or __setitem__')

        for frame_name, frame_df in self._inputs.items():
            assign(frame_name, frame_df)

        return target


class TechnologyIncentives:
    """Container describing technology-specific incentives."""

    def __init__(
        self,
        records: Iterable[IncentiveRecord] | None = None,
        *,
        enabled: bool = True,
    ):
        self._records: list[IncentiveRecord] = []
        self.enabled = bool(enabled)
        if records is not None:
            for record in records:
                if isinstance(record, IncentiveRecord):
                    self._records.append(record)

    def __iter__(self) -> Iterator[IncentiveRecord]:  # pragma: no cover - trivial
        return iter(self._records)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._records)

    def is_empty(self) -> bool:
        return not self._records

    def to_table_rows(self) -> list[MutableMapping[str, Any]]:
        return [record.to_table_row() for record in self._records]

    def to_config(self) -> dict[str, list[dict[str, Any]]]:
        production: list[dict[str, Any]] = []
        investment: list[dict[str, Any]] = []
        for record in self._records:
            entry = record.to_config_entry()
            if record.credit_type == _CREDIT_TYPE_PRODUCTION:
                production.append(entry)
            else:
                investment.append(entry)

        config: dict[str, Any] = {}
        if not self.enabled:
            config['enabled'] = False
        if production:
            config['production'] = production
        if investment:
            config['investment'] = investment
        return config

    def to_frames(self) -> dict[str, 'pd.DataFrame']:
        pandas = _ensure_pandas()
        credit_rows: list[dict[str, Any]] = []
        limit_rows: list[dict[str, Any]] = []
        for record in self._records:
            credit_rows.append(
                {
                    'tech': record.technology_id,
                    'year': record.year,
                    'incentive_type': record.credit_type,
                    'credit_per_unit': record.credit_value,
                }
            )
            limit_rows.append(
                {
                    'tech': record.technology_id,
                    'year': record.year,
                    'incentive_type': record.credit_type,
                    'limit_value': record.limit_value,
                }
            )

        credit_df = pandas.DataFrame(credit_rows)
        if credit_df.empty:
            credit_df = pandas.DataFrame(
                columns=['tech', 'year', 'incentive_type', 'credit_per_unit']
            )
        else:
            credit_df['tech'] = credit_df['tech'].astype(int)
            credit_df['year'] = credit_df['year'].astype(int)
            credit_df['incentive_type'] = credit_df['incentive_type'].astype(str)
            credit_df['credit_per_unit'] = credit_df['credit_per_unit'].astype(float)

        limit_df = pandas.DataFrame(limit_rows)
        if limit_df.empty:
            limit_df = pandas.DataFrame(
                columns=['tech', 'year', 'incentive_type', 'limit_value']
            )
        else:
            limit_df['tech'] = limit_df['tech'].astype(int)
            limit_df['year'] = limit_df['year'].astype(int)
            limit_df['incentive_type'] = limit_df['incentive_type'].astype(str)
            limit_df['limit_value'] = limit_df['limit_value'].astype(float)

        return {
            'TechnologyIncentiveCredit': credit_df.reset_index(drop=True),
            'TechnologyIncentiveLimit': limit_df.reset_index(drop=True),
        }

    def modules(self, *, enabled: bool | None = None) -> list[TechnologyIncentiveModule]:
        frames = self.to_frames()
        module_enabled = self.enabled if enabled is None else bool(enabled)
        inputs = {name: df.copy(deep=True) for name, df in frames.items()}
        return [TechnologyIncentiveModule(enabled=module_enabled, _inputs=inputs)]

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> 'TechnologyIncentives':
        if not isinstance(config, Mapping):
            return cls()

        enabled_flag = _coerce_bool_flag(config.get('enabled'), default=True)

        records: list[IncentiveRecord] = []
        for raw in config.get('production', []):
            record = cls._parse_config_entry(raw, _CREDIT_TYPE_PRODUCTION)
            if record is not None:
                records.append(record)
        for raw in config.get('investment', []):
            record = cls._parse_config_entry(raw, _CREDIT_TYPE_INVESTMENT)
            if record is not None:
                records.append(record)
        return cls(records, enabled=enabled_flag)

    @classmethod
    def from_table_rows(
        cls, rows: Iterable[Mapping[str, Any]] | None
    ) -> 'TechnologyIncentives':
        if rows is None:
            return cls()
        records: list[IncentiveRecord] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            credit_type = _coerce_credit_type(row.get('type'))
            tech_id = resolve_technology_key(row.get('technology'))
            year = _coerce_int(row.get('year'))
            credit_value = _coerce_float(row.get('credit_value'))
            limit_value = _coerce_float(row.get('limit_value'))
            if credit_type is None or tech_id is None or year is None or credit_value is None:
                continue
            records.append(
                IncentiveRecord(
                    technology_id=tech_id,
                    year=year,
                    credit_type=credit_type,
                    credit_value=credit_value,
                    limit_value=limit_value,
                )
            )
        return cls(records)

    @staticmethod
    def _parse_config_entry(entry: Mapping[str, Any], credit_type: str) -> IncentiveRecord | None:
        if not isinstance(entry, Mapping):
            return None

        tech_id = resolve_technology_key(entry.get('technology'))
        year = _coerce_int(entry.get('year'))
        if tech_id is None or year is None:
            return None

        if credit_type == _CREDIT_TYPE_PRODUCTION:
            credit_value = _coerce_float(entry.get('credit_per_mwh'))
            limit_value = _coerce_float(entry.get('limit_mwh'))
        else:
            credit_value = _coerce_float(entry.get('credit_per_mw'))
            limit_value = _coerce_float(entry.get('limit_mw'))

        if credit_value is None:
            return None

        return IncentiveRecord(
            technology_id=tech_id,
            year=year,
            credit_type=credit_type,
            credit_value=credit_value,
            limit_value=limit_value,
        )


__all__ = ['IncentiveRecord', 'TechnologyIncentiveModule', 'TechnologyIncentives']

