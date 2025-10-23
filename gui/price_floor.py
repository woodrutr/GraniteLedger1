"""Utilities for computing and parsing price floor schedules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class PriceFloorParameters:
    """Normalized representation of user provided price floor settings."""

    base_value: float
    escalation_type: str
    escalation_value: float


def _normalize_years(years: Iterable[int | float | str]) -> list[int]:
    normalized: list[int] = []
    for entry in years:
        try:
            year = int(entry)
        except (TypeError, ValueError):
            continue
        if year not in normalized:
            normalized.append(year)
    normalized.sort()
    return normalized


def parse_currency_value(value: object, default: float = 0.0) -> float:
    """Return ``value`` as a non-negative float rounded to two decimals."""

    if value is None:
        return round(max(default, 0.0), 2)

    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return round(max(default, 0.0), 2)
        filtered = []
        decimal_seen = False
        for char in cleaned:
            if char.isdigit():
                filtered.append(char)
            elif char == "." and not decimal_seen:
                filtered.append(char)
                decimal_seen = True
            elif char in {",", "_", " "}:
                continue
        try:
            parsed = float("".join(filtered)) if filtered else float(cleaned)
        except ValueError:
            return round(max(default, 0.0), 2)
    else:
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return round(max(default, 0.0), 2)

    return round(max(parsed, 0.0), 2)


def parse_percentage_value(value: object, default: float = 0.0) -> float:
    """Return ``value`` as a float rounded to two decimals."""

    if value is None:
        return round(default, 2)

    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return round(default, 2)
        filtered = []
        decimal_seen = False
        for char in cleaned:
            if char.isdigit() or (char == "-" and not filtered):
                filtered.append(char)
            elif char == "." and not decimal_seen:
                filtered.append(char)
                decimal_seen = True
            elif char in {",", "_", " ", "%"}:
                continue
        try:
            parsed = float("".join(filtered)) if filtered else float(cleaned)
        except ValueError:
            return round(default, 2)
    else:
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return round(default, 2)

    return round(parsed, 2)


def infer_parameters(
    schedule: Mapping[int, float] | Mapping[str, float] | None,
    *,
    default_base: float = 0.0,
    default_type: str = "fixed",
    default_escalator: float = 0.0,
) -> PriceFloorParameters:
    """Infer floor parameters from an existing schedule mapping."""

    if not isinstance(schedule, Mapping) or not schedule:
        return PriceFloorParameters(default_base, default_type, default_escalator)

    normalized_schedule: dict[int, float] = {}
    for key, raw_value in schedule.items():
        try:
            year = int(key)
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        normalized_schedule[year] = value

    if not normalized_schedule:
        return PriceFloorParameters(default_base, default_type, default_escalator)

    years = sorted(normalized_schedule)
    base_value = round(max(normalized_schedule[years[0]], 0.0), 2)

    if len(years) < 2:
        return PriceFloorParameters(base_value, default_type, default_escalator)

    diffs = [
        round(normalized_schedule[years[idx + 1]] - normalized_schedule[years[idx]], 4)
        for idx in range(len(years) - 1)
    ]
    if diffs and all(abs(diff - diffs[0]) <= 1e-4 for diff in diffs):
        return PriceFloorParameters(base_value, "fixed", round(diffs[0], 2))

    ratios: list[float] = []
    for idx in range(len(years) - 1):
        first = normalized_schedule[years[idx]]
        second = normalized_schedule[years[idx + 1]]
        if first == 0:
            ratios = []
            break
        ratios.append(round(((second / first) - 1.0) * 100.0, 4))

    if ratios and all(abs(ratio - ratios[0]) <= 1e-4 for ratio in ratios):
        return PriceFloorParameters(base_value, "percent", round(ratios[0], 2))

    return PriceFloorParameters(base_value, default_type, default_escalator)


def build_schedule(
    years: Iterable[int | float | str],
    base_value: float,
    escalation_type: str,
    escalation_value: float,
) -> dict[int, float]:
    """Construct a year-indexed floor schedule from the supplied parameters."""

    year_list = _normalize_years(years)
    if not year_list:
        return {}

    schedule: dict[int, float] = {}
    mode = (escalation_type or "fixed").strip().lower()
    if mode not in {"fixed", "percent"}:
        mode = "fixed"

    value = round(max(float(base_value), 0.0), 2)
    schedule[year_list[0]] = value

    for idx in range(1, len(year_list)):
        previous = schedule[year_list[idx - 1]]
        if mode == "percent":
            rate = float(escalation_value) / 100.0
            next_value = previous * (1.0 + rate)
        else:
            next_value = previous + float(escalation_value)
        schedule[year_list[idx]] = round(max(next_value, 0.0), 2)

    return schedule


__all__ = [
    "PriceFloorParameters",
    "build_schedule",
    "infer_parameters",
    "parse_currency_value",
    "parse_percentage_value",
]
