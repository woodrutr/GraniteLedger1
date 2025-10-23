"""Structured containers for storing engine outputs and serialising to CSV."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, cast

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = cast(Any, None)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

from engine.emissions import summarize_emissions_by_state
from engine.regions.shares import load_zone_to_state_share


def _ensure_pandas() -> None:
    """Ensure :mod:`pandas` is available before working with outputs."""

    if pd is None:  # pragma: no cover - helper exercised indirectly
        raise ImportError(
            "pandas is required for engine.outputs; install it with `pip install pandas`.",
        )


def _coerce_states(states: Sequence[str] | Mapping[str, Any] | None) -> tuple[str, ...]:
    if states is None:
        return ()
    if isinstance(states, Mapping):
        iterable: Iterable[object] = states.keys()
    elif isinstance(states, (str, bytes)):
        iterable = [states]
    else:
        iterable = states

    normalized: list[str] = []
    seen: set[str] = set()
    for entry in iterable:
        if entry is None:
            continue
        text = str(entry).strip().upper()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return tuple(normalized)


def _apportion_zone_metric(
    df_zone_metric: "pd.DataFrame",
    value_col: str,
    states: tuple[str, ...],
) -> "pd.DataFrame":
    columns = ["year", "state", value_col]
    if not states or df_zone_metric.empty or value_col not in df_zone_metric.columns:
        return pd.DataFrame(columns=columns)

    shares = load_zone_to_state_share()
    if shares.empty:
        return pd.DataFrame(columns=columns)

    working = df_zone_metric.copy()
    if "region" in working.columns and "region_id" not in working.columns:
        working = working.rename(columns={"region": "region_id"})
    if "region_id" not in working.columns:
        return pd.DataFrame(columns=columns)

    working["year"] = pd.to_numeric(working.get("year", 0), errors="coerce")
    working = working.dropna(subset=["year"])
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["year"] = working["year"].astype(int)
    working["region_id"] = working["region_id"].astype(str)
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce").fillna(0.0)

    regional_totals = (
        working.groupby(["year", "region_id"], as_index=False)[value_col].sum()
    )
    merged = regional_totals.merge(shares, on="region_id", how="inner")
    merged = merged[merged["state"].isin(states)]
    if merged.empty:
        return pd.DataFrame(columns=columns)

    merged[value_col] = merged[value_col] * merged["share"]
    result = (
        merged.groupby(["year", "state"], as_index=False)[value_col].sum()
        .sort_values(["year", "state"])
        .reset_index(drop=True)
    )
    return result[columns]

@dataclass(frozen=True)
class EngineOutputs:
    """Container bundling the primary outputs of the annual engine."""

    annual: pd.DataFrame
    emissions_by_region: pd.DataFrame
    price_by_region: pd.DataFrame
    flows: pd.DataFrame
    limiting_factors: list[str] = field(default_factory=list)
    emissions_total: Mapping[int, float] = field(default_factory=dict)
    emissions_by_region_map: Mapping[str, Mapping[int, float]] = field(default_factory=dict)
    generation_by_fuel: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["year", "fuel", "generation_mwh"])
    )
    capacity_by_fuel: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["year", "fuel", "capacity_mwh", "capacity_mw"]
        )
    )
    cost_by_fuel: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "year",
                "fuel",
                "variable_cost",
                "allowance_cost",
                "carbon_price_cost",
                "total_cost",
            ]
        )
    )
    demand_by_region: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["year", "region", "demand_mwh"]
        )
    )
    peak_demand_by_region: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "year",
                "region",
                "peak_demand_mw",
                "shortfall_mw",
                "available_capacity_mw",
            ]
        )
    )
    generation_by_region: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["year", "region", "fuel", "generation_mwh"]
        )
    )
    capacity_by_region: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["year", "region", "fuel", "capacity_mwh", "capacity_mw"]
        )
    )
    cost_by_region: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "year",
                "region",
                "fuel",
                "variable_cost",
                "allowance_cost",
                "carbon_price_cost",
                "total_cost",
            ]
        )
    )
    emissions_by_fuel: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["year", "fuel", "emissions_tons"])
    )
    stranded_units: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["year", "unit", "capacity_mwh", "capacity_mw"]
        )
    )
    capacity_by_technology: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["year", "technology", "capacity_mw", "capacity_mwh"]
        )
    )
    generation_by_technology: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["year", "technology", "generation_mwh"]
        )
    )
    carbon_price_schedule: Mapping[int | None, float] = field(default_factory=dict)
    carbon_price_applied: Mapping[int, float] = field(default_factory=dict)
    audits: Mapping[str, Any] = field(default_factory=dict)
    demand_by_state: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["year", "state", "demand_mwh"])
    )
    emissions_by_state: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["year", "state", "emissions_tons"])
    )
    states: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _ensure_pandas()
        states = _coerce_states(self.states)
        object.__setattr__(self, "states", states)

        if states:
            if hasattr(self, "demand_by_region") and isinstance(
                self.demand_by_region, pd.DataFrame
            ) and self.demand_by_state.empty:
                demand_state = _apportion_zone_metric(
                    self.demand_by_region, "demand_mwh", states
                )
                object.__setattr__(self, "demand_by_state", demand_state)

            if hasattr(self, "emissions_by_region") and isinstance(
                self.emissions_by_region, pd.DataFrame
            ) and self.emissions_by_state.empty:
                emissions_state = summarize_emissions_by_state(self.emissions_by_region)
                if not emissions_state.empty:
                    emissions_state = emissions_state[emissions_state["state"].isin(states)]
                    emissions_state = emissions_state.sort_values(["year", "state"])
                    emissions_state = emissions_state.reset_index(drop=True)
                object.__setattr__(self, "emissions_by_state", emissions_state)

        if hasattr(self, "annual") and isinstance(self.annual, pd.DataFrame):
            annual = self.annual.copy()
            legacy_pairs = [
                ("p_co2", "cp_last"),
                ("p_co2_all", "cp_all"),
                ("p_co2_exc", "cp_exempt"),
                ("p_co2_eff", "cp_effective"),
            ]
            updated = False
            for legacy, canonical in legacy_pairs:
                if canonical not in annual.columns and legacy in annual.columns:
                    annual[canonical] = annual[legacy]
                    updated = True
            for canonical in ("cp_last", "cp_all", "cp_exempt", "cp_effective"):
                if canonical not in annual.columns:
                    annual[canonical] = 0.0
                    updated = True
            if "allowance_price" not in annual.columns and "cp_all" in annual.columns:
                annual["allowance_price"] = annual["cp_all"]
                updated = True
            if updated:
                object.__setattr__(self, "annual", annual)

    @classmethod
    def empty(cls) -> "EngineOutputs":
        """Return an :class:`EngineOutputs` instance with empty DataFrames."""

        _ensure_pandas()

        return cls(
            annual=pd.DataFrame(
                columns=[
                    "year",
                    "cp_last",
                    "allowance_price",
                    "cp_all",
                    "cp_exempt",
                    "cp_effective",
                    "iterations",
                    "emissions_tons",
                    "allowances_minted",
                    "allowances_available",
                    "bank_start",
                    "bank",
                    "surrender",
                    "obligation",
                    "finalized",
                    "shortage_flag",
                    "ccr1_trigger",
                    "ccr1_issued",
                    "ccr2_trigger",
                    "ccr2_issued",
                    "floor",
                ]
            ),
            emissions_by_region=pd.DataFrame(
                columns=["year", "region", "emissions_tons"]
            ),
            price_by_region=pd.DataFrame(columns=["year", "region", "price"]),
            flows=pd.DataFrame(columns=["year", "from_region", "to_region", "flow_mwh"]),
            demand_by_state=pd.DataFrame(columns=["year", "state", "demand_mwh"]),
            emissions_by_state=pd.DataFrame(columns=["year", "state", "emissions_tons"]),
            states=(),
        )

    def to_csv(
        self,
        outdir: str | Path,
        *,
        annual_filename: str = 'annual.csv',
        emissions_filename: str = 'emissions_by_region.csv',
        price_filename: str = 'price_by_region.csv',
        flows_filename: str = 'flows.csv',
        generation_filename: str = 'generation_by_fuel.csv',
        generation_technology_filename: str = 'generation_by_technology.csv',
        capacity_filename: str = 'capacity_by_fuel.csv',
        capacity_technology_filename: str = 'capacity_by_technology.csv',
        cost_filename: str = 'cost_by_fuel.csv',
        regional_demand_filename: str = 'demand_by_region.csv',
        regional_generation_filename: str = 'generation_by_region.csv',
        regional_capacity_filename: str = 'capacity_by_region.csv',
        regional_cost_filename: str = 'cost_by_region.csv',
        emissions_fuel_filename: str = 'emissions_by_fuel.csv',
        stranded_filename: str = 'stranded_units.csv',
    ) -> None:
        """Persist the stored DataFrames to ``outdir`` as CSV files."""

        _ensure_pandas()

        output_dir = Path(outdir)
        output_dir.mkdir(parents=True, exist_ok=True)

        alias_map = [
            ("cp_last", "p_co2"),
            ("cp_all", "p_co2_all"),
            ("cp_exempt", "p_co2_exc"),
            ("cp_effective", "p_co2_eff"),
        ]
        annual_frame = self.annual.copy()
        for canonical, legacy in alias_map:
            if canonical in annual_frame.columns and legacy not in annual_frame.columns:
                annual_frame[legacy] = annual_frame[canonical]
        if "allowance_price" not in annual_frame.columns and "cp_all" in annual_frame.columns:
            annual_frame["allowance_price"] = annual_frame["cp_all"]
        self_columns = list(annual_frame.columns)
        export_order = list(dict.fromkeys(self_columns + [legacy for _, legacy in alias_map]))
        annual_frame = annual_frame.loc[:, [col for col in export_order if col in annual_frame.columns]]

        annual_frame.to_csv(output_dir / annual_filename, index=False)
        self.emissions_by_region.to_csv(output_dir / emissions_filename, index=False)
        self.price_by_region.to_csv(output_dir / price_filename, index=False)
        self.flows.to_csv(output_dir / flows_filename, index=False)
        if not self.generation_by_fuel.empty:
            self.generation_by_fuel.to_csv(output_dir / generation_filename, index=False)
        if not self.generation_by_technology.empty:
            self.generation_by_technology.to_csv(
                output_dir / generation_technology_filename, index=False
            )
        if not self.capacity_by_fuel.empty:
            self.capacity_by_fuel.to_csv(output_dir / capacity_filename, index=False)
        if not self.capacity_by_technology.empty:
            self.capacity_by_technology.to_csv(
                output_dir / capacity_technology_filename, index=False
            )
        if not self.cost_by_fuel.empty:
            self.cost_by_fuel.to_csv(output_dir / cost_filename, index=False)
        if not self.demand_by_region.empty:
            self.demand_by_region.to_csv(
                output_dir / regional_demand_filename, index=False
            )
        if not self.generation_by_region.empty:
            self.generation_by_region.to_csv(
                output_dir / regional_generation_filename, index=False
            )
        if not self.capacity_by_region.empty:
            self.capacity_by_region.to_csv(
                output_dir / regional_capacity_filename, index=False
            )
        if not self.cost_by_region.empty:
            self.cost_by_region.to_csv(output_dir / regional_cost_filename, index=False)
        if not self.emissions_by_fuel.empty:
            self.emissions_by_fuel.to_csv(
                output_dir / emissions_fuel_filename, index=False
            )
        if not self.stranded_units.empty:
            self.stranded_units.to_csv(output_dir / stranded_filename, index=False)

    def emissions_summary_table(self) -> "pd.DataFrame":
        """Return a normalised emissions-by-region table for reporting."""

        _ensure_pandas()

        frame = self.emissions_by_region.copy()
        if frame.empty:
            return frame

        working = frame.copy()
        if "year" in working.columns:
            working["year"] = pd.to_numeric(working["year"], errors="coerce")
            working = working.dropna(subset=["year"])
            working["year"] = working["year"].astype(int)
        else:
            working["year"] = 0

        if "region" in working.columns:
            working["region"] = working["region"].astype(str)
        else:
            working["region"] = "system"

        working["emissions_tons"] = pd.to_numeric(
            working.get("emissions_tons", 0.0), errors="coerce"
        ).fillna(0.0)

        summary_columns = ["year", "region", "emissions_tons"]
        return working[summary_columns].sort_values(summary_columns[:2]).reset_index(drop=True)


__all__ = ['EngineOutputs']

