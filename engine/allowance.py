"""Allowance market accounting helpers."""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from engine.constants import PRICE_TOL


def _coerce_numeric(series: pd.Series | float, default: float = 0.0) -> pd.Series:
    if isinstance(series, pd.Series):
        values = pd.to_numeric(series, errors="coerce").fillna(default)
        return values.astype(float)
    return pd.Series([float(series) if series is not None else float(default)])


def enforce_bank_trajectory(annual: pd.DataFrame) -> pd.DataFrame:
    """Ensure allowance bank levels honour emissions compliance each year."""

    if annual.empty:
        return annual

    required = {"year", "allowances_minted", "emissions_tons"}
    if not required.issubset(annual.columns):
        return annual

    working = annual.copy()
    working = working.sort_values("year").reset_index(drop=True)

    minted = _coerce_numeric(working["allowances_minted"])
    emissions = _coerce_numeric(working["emissions_tons"])
    allowances_available_source = None
    if "allowances_available" in working.columns:
        allowances_available_source = _coerce_numeric(
            working["allowances_available"], 0.0
        )
    if "bank_start" in working.columns:
        bank_start = _coerce_numeric(working["bank_start"])
    else:
        bank_start = pd.Series([0.0] * len(working))

    surrender_source = None
    if "surrender" in working.columns:
        surrender_source = _coerce_numeric(working["surrender"], 0.0)

    obligation_source = None
    if "obligation" in working.columns:
        obligation_source = _coerce_numeric(working["obligation"], 0.0)

    allowances_available: list[float] = []
    surrender_values: list[float] = []
    bank_balances: list[float] = []
    shortage_flags: list[bool] = []

    bank_source = None
    if "bank" in working.columns:
        bank_source = _coerce_numeric(working["bank"], 0.0)

    previous_bank = 0.0
    for idx, _row in working.iterrows():
        if idx == 0:
            if allowances_available_source is not None:
                start_bank = max(
                    float(allowances_available_source.iloc[idx])
                    - float(minted.iloc[idx]),
                    0.0,
                )
            else:
                start_bank = float(bank_start.iloc[idx]) if len(bank_start) > 0 else 0.0
        else:
            start_bank = previous_bank
        total_allowances = max(0.0, start_bank + float(minted.iloc[idx]))
        emissions_value = max(0.0, float(emissions.iloc[idx]))
        bank_after = max(total_allowances - emissions_value, 0.0)
        if bank_source is not None:
            candidate_bank = float(bank_source.iloc[idx])
            if not pd.isna(candidate_bank):
                bank_after = max(candidate_bank, 0.0)
        surrendered = min(emissions_value, total_allowances)
        if surrender_source is not None:
            candidate = float(surrender_source.iloc[idx])
            if not pd.isna(candidate):
                candidate = max(candidate, 0.0)
                surrendered = min(candidate, total_allowances)
        shortage = emissions_value > total_allowances + 1e-9

        allowances_available.append(total_allowances)
        surrender_values.append(surrendered)
        bank_balances.append(bank_after)
        shortage_flags.append(shortage)

        previous_bank = bank_after

    working["allowances_available"] = allowances_available
    working["surrender"] = surrender_values
    working["bank"] = bank_balances
    working["shortage_flag"] = shortage_flags
    if "obligation" in working.columns:
        if obligation_source is not None:
            working["obligation"] = [
                max(float(obligation_source.iloc[idx]), 0.0)
                for idx in range(len(working))
            ]
        else:
            working["obligation"] = [
                max(total - given, 0.0)
                for total, given in zip(allowances_available, surrender_values)
            ]

    floor_column = "floor"
    if floor_column in working.columns:
        floors = _coerce_numeric(working[floor_column], 0.0)
        shortage_mask = pd.Series(shortage_flags)
        if shortage_mask.any():
            enforced_floor = floors[shortage_mask] + PRICE_TOL
            price_columns: Iterable[str] = (
                "allowance_price",
                "cp_last",
                "cp_all",
                "cp_effective",
            )
            for column in price_columns:
                if column not in working.columns:
                    continue
                values = _coerce_numeric(working.loc[shortage_mask, column], 0.0)
                adjusted = values.where(values > enforced_floor, enforced_floor)
                working.loc[shortage_mask, column] = adjusted

    return working


__all__ = ["enforce_bank_trajectory"]
