"""Centralised access to validated model input frames."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import logging

import pandas as pd

from engine.data_loaders.load_forecasts import load_iso_scenario_table
from engine.normalization import normalize_region_id
from policy.allowance_annual import ConfigError, RGGIPolicyAnnual

LOGGER = logging.getLogger(__name__)

_DEMAND_KEY = "demand"
_PEAK_DEMAND_KEY = "peak_demand"
_UNITS_KEY = "units"
_FUELS_KEY = "fuels"
_TRANSMISSION_KEY = "transmission"
_COVERAGE_KEY = "coverage"
_POLICY_KEY = "policy"
_LOAD_KEY = "load"

_REQUIRED_LOAD_COLUMNS = {"iso", "zone", "scenario", "year", "load_gwh"}
_LOAD_COLUMN_ORDER = ["iso", "zone", "scenario", "year", "load_gwh"]

_TRANSMISSION_COLUMNS = [
    "interface_id",
    "from_region",
    "to_region",
    "capacity_mw",
    "reverse_capacity_mw",
    "efficiency",
    "added_cost_per_mwh",
    "contracted_flow_mw_forward",
    "contracted_flow_mw_reverse",
    "notes",
    "profile_id",
    "in_service_year",
    "interface_type",
]


def _normalize_name(name: str) -> str:
    """Normalise frame identifiers to a consistent string key."""

    if not isinstance(name, str):  # pragma: no cover - defensive programming
        name = str(name)
    return name.lower()


def _ensure_dataframe(name: str, value: object) -> pd.DataFrame:
    """Return a defensive copy of ``value`` ensuring it is a DataFrame."""

    if not isinstance(value, pd.DataFrame):
        raise TypeError(f'frame "{name}" must be provided as a pandas DataFrame')
    return value.copy(deep=True)


def _coerce_region_label(value: object) -> str | None:
    """Return a normalized region identifier when possible."""

    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return normalize_region_id(text)
    except ValueError:
        sanitized = text.replace(" ", "_").upper()
        return sanitized or None


def _normalize_carbon_price_schedule(
    schedule: Mapping[Any, Any] | None,
) -> Dict[int | None, float]:
    """Normalise an optional carbon price schedule to use numeric keys."""

    if not schedule:
        return {}

    normalized: Dict[int | None, float] = {}
    for key, value in schedule.items():
        try:
            year = int(key) if key is not None else None
        except (TypeError, ValueError):
            continue
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        normalized[year] = price

    LOGGER.debug(
        "carbon_price_schedule_normalized keys=%s",
        sorted(key for key in normalized if key is not None),
    )

    return normalized


def _validate_columns(frame: str, df: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    """Ensure ``df`` contains the ``required`` columns, returning a copy."""

    missing = [column for column in required if column not in df.columns]
    if missing:
        columns = ", ".join(missing)
        raise ValueError(f'{frame} frame is missing required columns: {columns}')
    return df.copy(deep=True)


def _require_numeric(
    frame: str, column: str, series: pd.Series, *, allow_missing: bool = False
) -> pd.Series:
    """Return ``series`` coerced to numeric values ensuring no invalid data."""

    numeric = pd.to_numeric(series, errors="coerce")

    if allow_missing:
        original = pd.Series(series)
        missing_mask = original.isna()
        if original.dtype == object:
            normalized = original.astype(str).str.strip().str.lower()
            missing_tokens = {"", "na", "n/a", "none"}
            missing_mask |= normalized.isin(missing_tokens)

        invalid_mask = numeric.isna() & ~missing_mask
        if invalid_mask.any():
            raise ValueError(
                f"{frame} frame column '{column}' must contain numeric values or be blank"
            )
    elif numeric.isna().any():
        raise ValueError(f"{frame} frame column '{column}' must contain numeric values")

    return numeric


def _coerce_bool(series: pd.Series, frame: str, column: str) -> pd.Series:
    """Return ``series`` converted to booleans with explicit validation."""

    true_tokens = {"true", "t", "yes", "y", "on", "1"}
    false_tokens = {"false", "f", "no", "n", "off", "0"}

    def convert(value: object):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, Integral):
            if value in (0, 1):
                return bool(value)
            raise ValueError
        if isinstance(value, Real):
            if value in (0.0, 1.0):
                return bool(int(value))
            raise ValueError
        if isinstance(value, str):
            normalised = value.strip().lower()
            if normalised in true_tokens:
                return True
            if normalised in false_tokens:
                return False
        raise ValueError

    try:
        coerced = series.map(convert)
    except ValueError as exc:
        raise ValueError(
            f"{frame} frame column '{column}' must contain boolean-like values"
        ) from exc

    if getattr(coerced, "isna", None) is not None and coerced.isna().any():
        return coerced.astype("boolean")
    return coerced.astype(bool)


@dataclass(frozen=True)
class PolicySpec:
    """Data contract for allowance policy information."""

    cap: pd.Series
    floor: pd.Series
    ccr1_trigger: pd.Series
    ccr1_qty: pd.Series
    ccr2_trigger: pd.Series
    ccr2_qty: pd.Series
    cp_id: pd.Series
    bank0: float
    full_compliance_years: set[int]
    annual_surrender_frac: float
    carry_pct: float
    banking_enabled: bool = True
    enabled: bool = True
    ccr1_enabled: bool = True
    ccr2_enabled: bool = True
    control_period_years: int | None = None
    resolution: str = "annual"

    def to_policy(self) -> RGGIPolicyAnnual:
        """Instantiate :class:`RGGIPolicyAnnual` from the stored specification."""

        return RGGIPolicyAnnual(
            cap=self.cap,
            floor=self.floor,
            ccr1_trigger=self.ccr1_trigger,
            ccr1_qty=self.ccr1_qty,
            ccr2_trigger=self.ccr2_trigger,
            ccr2_qty=self.ccr2_qty,
            cp_id=self.cp_id,
            bank0=self.bank0,
            full_compliance_years=self.full_compliance_years,
            annual_surrender_frac=self.annual_surrender_frac,
            carry_pct=self.carry_pct,
            banking_enabled=self.banking_enabled,
            enabled=self.enabled,
            ccr1_enabled=self.ccr1_enabled,
            ccr2_enabled=self.ccr2_enabled,
            control_period_length=self.control_period_years,
            resolution=self.resolution,
        )


class Frames(Mapping[str, pd.DataFrame]):
    """Light-weight container offering validated access to model data frames."""

    def __init__(
        self,
        frames: Mapping[str, pd.DataFrame] | None = None,
        *,
        carbon_policy_enabled: bool | None = None,
        banking_enabled: bool | None = None,
        carbon_price_schedule: Mapping[Any, Any] | None = None,
    ):
        self._frames: Dict[str, pd.DataFrame] = {}
        self._meta: Dict[str, Any] = {}
        self._carbon_policy_enabled = True if carbon_policy_enabled is None else bool(
            carbon_policy_enabled
        )
        self._banking_enabled = True if banking_enabled is None else bool(banking_enabled)
        self._carbon_price_schedule: Dict[int | None, float] = _normalize_carbon_price_schedule(
            carbon_price_schedule
        )
        if frames:
            for name, df in frames.items():
                key = _normalize_name(name)
                self._frames[key] = _ensure_dataframe(name, df)

    # ------------------------------------------------------------------
    # Mapping interface
    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> pd.DataFrame:
        normalized = _normalize_name(key)
        if normalized not in self._frames:
            raise KeyError(f'frame {key!r} is not present')
        return self._frames[normalized].copy(deep=True)

    def __iter__(self) -> Iterator[str]:
        return iter(self._frames)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._frames)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def coerce(
        cls,
        frames: Frames | Mapping[str, pd.DataFrame] | None,
        *,
        carbon_policy_enabled: bool | None = None,
        banking_enabled: bool | None = None,
        carbon_price_schedule: Mapping[Any, Any] | None = None,
    ) -> Frames:
        """Return ``frames`` as a :class:`Frames` instance."""

        if frames is None:
            raise ValueError('frames must be supplied as a Frames instance or mapping')
        if isinstance(frames, cls):
            if carbon_policy_enabled is not None:
                frames._carbon_policy_enabled = bool(carbon_policy_enabled)
            if banking_enabled is not None:
                frames._banking_enabled = bool(banking_enabled)
            if carbon_price_schedule is not None:
                frames._carbon_price_schedule = _normalize_carbon_price_schedule(
                    carbon_price_schedule
                )
            return frames
        if isinstance(frames, Mapping):
            return cls(
                frames,
                carbon_policy_enabled=carbon_policy_enabled,
                banking_enabled=banking_enabled,
                carbon_price_schedule=carbon_price_schedule,
            )
        raise TypeError('frames must be provided as Frames or a mapping of names to DataFrames')

    def with_frame(self, name: str, df: pd.DataFrame) -> "Frames":
        """Return a new container with ``name`` replaced by ``df``."""

        updated = dict(self._frames)
        updated[_normalize_name(name)] = _ensure_dataframe(name, df)
        new_frames = Frames(
            updated,
            carbon_policy_enabled=self._carbon_policy_enabled,
            banking_enabled=self._banking_enabled,
            carbon_price_schedule=self._carbon_price_schedule,
        )
        new_frames._meta = dict(getattr(self, "_meta", {}))
        if hasattr(self, "deep_carbon_pricing_enabled"):
            setattr(
                new_frames,
                "deep_carbon_pricing_enabled",
                getattr(self, "deep_carbon_pricing_enabled"),
            )
        return new_frames

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def has_frame(self, name: str) -> bool:
        """Return ``True`` if ``name`` exists in the container."""

        return _normalize_name(name) in self._frames

    def frame(self, name: str) -> pd.DataFrame:
        """Return the stored DataFrame for ``name``."""

        return self[name]

    def optional_frame(
        self, name: str, default: pd.DataFrame | None = None
    ) -> pd.DataFrame | None:
        """Return the DataFrame for ``name`` if present, otherwise ``default``."""

        normalized = _normalize_name(name)
        if normalized not in self._frames:
            return default
        return self._frames[normalized].copy(deep=True)

    def years(self) -> list[int]:
        """Return sorted unique years referenced in demand or policy frames."""

        years: set[int] = set()

        demand_df = self.optional_frame(_DEMAND_KEY)
        if isinstance(demand_df, pd.DataFrame) and not demand_df.empty and 'year' in demand_df.columns:
            demand_years = pd.to_numeric(demand_df['year'], errors='coerce').dropna()
            years.update(int(value) for value in demand_years.astype(int).unique())

        peak_df = self.optional_frame(_PEAK_DEMAND_KEY)
        if isinstance(peak_df, pd.DataFrame) and not peak_df.empty and 'year' in peak_df.columns:
            peak_years = pd.to_numeric(peak_df['year'], errors='coerce').dropna()
            years.update(int(value) for value in peak_years.astype(int).unique())

        policy_df = self.optional_frame(_POLICY_KEY)
        if isinstance(policy_df, pd.DataFrame) and not policy_df.empty and 'year' in policy_df.columns:
            policy_years = pd.to_numeric(policy_df['year'], errors='coerce').dropna()
            years.update(int(value) for value in policy_years.astype(int).unique())

        return sorted(years)

    # ------------------------------------------------------------------
    # Metadata accessors
    # ------------------------------------------------------------------
    @property
    def carbon_policy_enabled(self) -> bool:
        """Return the cached carbon policy enabled flag."""

        return bool(self._carbon_policy_enabled)

    @property
    def banking_enabled(self) -> bool:
        """Return the cached allowance banking enabled flag."""

        return bool(self._banking_enabled)

    @property
    def carbon_price_schedule(self) -> Dict[int | None, float]:
        """Return a copy of the configured carbon price schedule."""

        schedule = dict(self._carbon_price_schedule)
        meta = getattr(self, "_meta", {})
        if isinstance(meta, Mapping):
            meta_schedule = _normalize_carbon_price_schedule(
                meta.get("carbon_price_schedule")
            )
            if meta_schedule:
                schedule.update(meta_schedule)
            elif not schedule and meta.get("carbon_price_default") is not None:
                try:
                    base = float(meta["carbon_price_default"])
                except (TypeError, ValueError):
                    base = None
                if base is not None:
                    years = self.years()
                    if years:
                        LOGGER.debug(
                            "carbon_price_schedule_defaulted base=%.6f years=%s",
                            base,
                            years,
                        )
                        return {year: base for year in years}
        return schedule

    # ------------------------------------------------------------------
    # Accessors with schema validation
    # ------------------------------------------------------------------
    def demand(self) -> pd.DataFrame:
        """Return validated demand data with columns (year, region, demand_mwh)."""

        df = self[_DEMAND_KEY]
        df = _validate_columns('demand', df, ['year', 'region', 'demand_mwh'])
        df['year'] = _require_numeric('demand', 'year', df['year']).astype(int)
        df['region'] = _validate_region_labels('demand', 'region', df['region'])
        df['demand_mwh'] = _require_numeric('demand', 'demand_mwh', df['demand_mwh']).astype(float)

        duplicates = df.duplicated(subset=['year', 'region'])
        if duplicates.any():
            dupes = df.loc[duplicates, ['year', 'region']].to_records(index=False)
            raise ValueError(
                'demand frame contains duplicate year/region pairs: '
                + ', '.join(f'({year}, {region})' for year, region in dupes)
            )

        return df.sort_values(['year', 'region']).reset_index(drop=True)

    def peak_demand(self) -> pd.DataFrame:
        """Return validated peak demand data (year, region, peak_demand_mw)."""

        df = self[_PEAK_DEMAND_KEY]
        df = _validate_columns('peak_demand', df, ['year', 'region', 'peak_demand_mw'])
        df['year'] = _require_numeric('peak_demand', 'year', df['year']).astype(int)
        df['region'] = _validate_region_labels('peak_demand', 'region', df['region'])
        df['peak_demand_mw'] = _require_numeric(
            'peak_demand', 'peak_demand_mw', df['peak_demand_mw']
        ).astype(float)

        duplicates = df.duplicated(subset=['year', 'region'])
        if duplicates.any():
            dupes = df.loc[duplicates, ['year', 'region']].to_records(index=False)
            raise ValueError(
                'peak_demand frame contains duplicate year/region pairs: '
                + ', '.join(f'({year}, {region})' for year, region in dupes)
            )

        return df.sort_values(['year', 'region']).reset_index(drop=True)

    def demand_for_year(self, year: int) -> Dict[str, float]:
        """Return demand by region for ``year`` as a mapping."""

        demand = self.demand()
        filtered = demand[demand['year'] == int(year)]
        if filtered.empty:
            raise KeyError(f'demand for year {year} is unavailable')
        grouped = filtered.groupby('region')['demand_mwh'].sum()
        return {str(region): float(value) for region, value in grouped.items()}

    def peak_demand_for_year(self, year: int) -> Dict[str, float]:
        """Return peak demand by region for ``year`` as a mapping."""

        peak = self.peak_demand()
        filtered = peak[peak['year'] == int(year)]
        if filtered.empty:
            raise KeyError(f'peak demand for year {year} is unavailable')
        grouped = filtered.groupby('region')['peak_demand_mw'].sum()
        return {str(region): float(value) for region, value in grouped.items()}

    def units(self) -> pd.DataFrame:
        """Return validated generating unit characteristics."""

        required = [
            'unit_id',
            'region',
            'fuel',
            'cap_mw',
            'availability',
            'hr_mmbtu_per_mwh',
            'vom_per_mwh',
            'fuel_price_per_mmbtu',
        ]
        df = self[_UNITS_KEY]
        df = _validate_columns('units', df, required)

        df['unit_id'] = df['unit_id'].astype(str)
        fallback_ids = df['unit_id'].str.strip()

        if 'unique_id' not in df.columns:
            df['unique_id'] = df['unit_id']

        unique_raw = df['unique_id']
        normalized_unique = unique_raw.astype(str).str.strip()
        invalid_tokens = {'', 'nan', 'none', 'null'}
        missing_mask = unique_raw.isna()
        missing_mask |= normalized_unique.str.lower().isin(invalid_tokens)
        df['unique_id'] = normalized_unique.where(~missing_mask, fallback_ids)

        if df['unique_id'].duplicated().any():
            duplicates = sorted(df.loc[df['unique_id'].duplicated(), 'unique_id'].unique())
            raise ValueError('units frame contains duplicate unique_id values: ' + ', '.join(duplicates))

        df['region'] = _validate_region_labels('units', 'region', df['region'])
        df['fuel'] = df['fuel'].astype(str)

        emission_columns = [col for col in ('co2_short_ton_per_mwh', 'ef_ton_per_mwh') if col in df.columns]
        if not emission_columns:
            raise ValueError(
                "units frame must include an emission factor column ('co2_short_ton_per_mwh' or 'ef_ton_per_mwh')"
            )

        numeric_columns = [
            'cap_mw',
            'availability',
            'hr_mmbtu_per_mwh',
            'vom_per_mwh',
            'fuel_price_per_mmbtu',
        ] + emission_columns
        if 'carbon_cost_per_mwh' in df.columns:
            numeric_columns.append('carbon_cost_per_mwh')
        for column in numeric_columns:
            df[column] = _require_numeric('units', column, df[column]).astype(float)

        df['availability'] = df['availability'].clip(lower=0.0, upper=1.0)

        if 'carbon_cost_per_mwh' not in df.columns:
            df['carbon_cost_per_mwh'] = 0.0
        else:
            df['carbon_cost_per_mwh'] = df['carbon_cost_per_mwh'].fillna(0.0)

        if 'co2_short_ton_per_mwh' in df.columns:
            df['ef_ton_per_mwh'] = df['co2_short_ton_per_mwh']

        # Ensure the generating fleet covers every active demand region.
        try:
            demand_regions = set(self.demand()['region'].unique())
        except KeyError:
            demand_regions = set()

        if demand_regions:
            unit_regions = set(df['region'].unique())
            missing_regions = sorted(region for region in demand_regions if region not in unit_regions)
            if missing_regions:
                missing_list = ", ".join(missing_regions)
                raise ValueError(
                    "units frame is missing unit inventory for active regions: " + missing_list
                )

        # Fossil fuel units require explicit fuel price and emission factor joins.
        fossil_fuels: set[str] = set()
        if self.has_frame(_FUELS_KEY):
            fuels_df = self.fuels()
            if 'co2_short_ton_per_mwh' in fuels_df.columns:
                fossil_fuels = {
                    str(row.fuel)
                    for row in fuels_df.itertuples(index=False)
                    if getattr(row, 'co2_short_ton_per_mwh', 0.0)
                    and float(row.co2_short_ton_per_mwh) > 0.0
                }
            else:
                fossil_fuels = {
                    str(row.fuel)
                    for row in fuels_df.itertuples(index=False)
                    if bool(getattr(row, 'covered', False))
                }

        if fossil_fuels:
            fossil_mask = df['fuel'].isin(fossil_fuels)
        else:
            fossil_mask = df['hr_mmbtu_per_mwh'] > 0.0

        fossil_mask &= (df['cap_mw'] > 0.0)
        fossil_mask &= (df['hr_mmbtu_per_mwh'] > 0.0)

        def _format_pairs(mask: pd.Series) -> str:
            pairs = df.loc[mask, ['region', 'fuel']].drop_duplicates()
            return ", ".join(f"{row.region}/{row.fuel}" for row in pairs.itertuples(index=False))

        missing_cost = fossil_mask & (df['fuel_price_per_mmbtu'] <= 0.0)
        if missing_cost.any():
            raise ValueError(
                "units frame is missing fuel price data for fossil units; verify fuel_price joins for: "
                + _format_pairs(missing_cost)
            )

        missing_ef = fossil_mask & (df['ef_ton_per_mwh'] <= 0.0)
        if missing_ef.any():
            raise ValueError(
                "units frame is missing emission factors for fossil units; verify emissions rate joins for: "
                + _format_pairs(missing_ef)
            )

        return df.reset_index(drop=True)

    def expansion_options(self) -> pd.DataFrame:
        """Return validated capacity expansion candidate data if available."""

        df = self.optional_frame('expansion')
        if df is None or df.empty:
            return pd.DataFrame(
                columns=[
                    'unit_id',
                    'unique_id',
                    'region',
                    'fuel',
                    'cap_mw',
                    'availability',
                    'hr_mmbtu_per_mwh',
                    'vom_per_mwh',
                    'fuel_price_per_mmbtu',
                    'ef_ton_per_mwh',
                    'capex_per_mw',
                    'fixed_om_per_mw',
                    'lifetime_years',
                    'max_builds',
                ]
            )

        required = [
            'unit_id',
            'region',
            'fuel',
            'cap_mw',
            'availability',
            'hr_mmbtu_per_mwh',
            'vom_per_mwh',
            'fuel_price_per_mmbtu',
            'ef_ton_per_mwh',
            'capex_per_mw',
            'fixed_om_per_mw',
            'lifetime_years',
        ]

        df = _validate_columns('expansion', df, required)

        df['unit_id'] = df['unit_id'].astype(str)
        if 'unique_id' not in df.columns:
            df['unique_id'] = df['unit_id']
        unique_raw = df['unique_id']
        normalized_unique = unique_raw.astype(str).str.strip()
        invalid_tokens = {'', 'nan', 'none', 'null'}
        missing_mask = unique_raw.isna()
        missing_mask |= normalized_unique.str.lower().isin(invalid_tokens)
        df['unique_id'] = normalized_unique.where(~missing_mask, df['unit_id'])
        df['region'] = df['region'].astype(str)
        df['fuel'] = df['fuel'].astype(str)

        numeric_columns = [
            'cap_mw',
            'availability',
            'hr_mmbtu_per_mwh',
            'vom_per_mwh',
            'fuel_price_per_mmbtu',
            'ef_ton_per_mwh',
            'capex_per_mw',
            'fixed_om_per_mw',
            'lifetime_years',
        ]

        for column in numeric_columns:
            df[column] = _require_numeric('expansion', column, df[column]).astype(float)

        df['availability'] = df['availability'].clip(lower=0.0, upper=1.0)
        df['cap_mw'] = df['cap_mw'].clip(lower=0.0)
        df['lifetime_years'] = df['lifetime_years'].clip(lower=1.0)

        if 'max_builds' in df.columns:
            df['max_builds'] = _require_numeric('expansion', 'max_builds', df['max_builds']).fillna(1.0)
        else:
            df['max_builds'] = 1.0

        df['max_builds'] = df['max_builds'].clip(lower=0.0)

        return df.reset_index(drop=True)

    def technology_incentives(self) -> pd.DataFrame:
        """Return merged technology incentive information if available."""

        if pd is None:  # pragma: no cover - helper exercised indirectly
            raise ImportError(
                'pandas is required to access technology incentives; '
                'install it with `pip install pandas`.'
            )

        credit = self.optional_frame('TechnologyIncentiveCredit')
        limit = self.optional_frame('TechnologyIncentiveLimit')

        base_columns = ['tech', 'year', 'incentive_type']
        credit_df = pd.DataFrame(columns=base_columns + ['credit_per_unit'])
        limit_df = pd.DataFrame(columns=base_columns + ['limit_value'])

        if credit is not None and not credit.empty:
            credit_df = credit.reset_index().rename(columns=str)
            credit_df = credit_df[base_columns + ['credit_per_unit']]
        if limit is not None and not limit.empty:
            limit_df = limit.reset_index().rename(columns=str)
            limit_df = limit_df[base_columns + ['limit_value']]

        if credit_df.empty and limit_df.empty:
            return pd.DataFrame(
                columns=base_columns
                + ['credit_per_unit', 'limit_value', 'limit_units']
            )

        merged = pd.merge(credit_df, limit_df, how='outer', on=base_columns)
        merged['credit_per_unit'] = pd.to_numeric(
            merged.get('credit_per_unit'), errors='coerce'
        )
        merged['limit_value'] = pd.to_numeric(merged.get('limit_value'), errors='coerce')
        merged['limit_units'] = merged['incentive_type'].map(
            {
                'production': 'MWh',
                'investment': 'MW',
            }
        )

        return merged.sort_values(base_columns).reset_index(drop=True)

    def fuels(self) -> pd.DataFrame:
        """Return validated fuel metadata (fuel label and coverage flag)."""

        df = self[_FUELS_KEY]
        df = _validate_columns('fuels', df, ['fuel', 'covered'])
        df['fuel'] = df['fuel'].astype(str)
        if df['fuel'].duplicated().any():
            duplicates = sorted(df.loc[df['fuel'].duplicated(), 'fuel'].unique())
            raise ValueError('fuels frame contains duplicate fuel labels: ' + ', '.join(duplicates))

        df['covered'] = _coerce_bool(df['covered'], 'fuels', 'covered')
        if 'co2_short_ton_per_mwh' in df.columns:
            df['co2_short_ton_per_mwh'] = pd.to_numeric(
                df['co2_short_ton_per_mwh'], errors='coerce'
            ).fillna(0.0)
            df['co2_short_ton_per_mwh'] = _require_numeric(
                'fuels',
                'co2_short_ton_per_mwh',
                df['co2_short_ton_per_mwh'],
                allow_missing=True,
            ).astype(float)

        return df.reset_index(drop=True)

    def transmission(self) -> pd.DataFrame:
        """Return validated transmission interfaces or an empty frame if absent."""

        if _TRANSMISSION_KEY not in self._frames:
            return pd.DataFrame(columns=_TRANSMISSION_COLUMNS)

        df = self[_TRANSMISSION_KEY].copy(deep=True)
        if df.empty:
            return pd.DataFrame(columns=_TRANSMISSION_COLUMNS)

        if 'capacity_mw' not in df.columns and 'limit_mw' in df.columns:
            df['capacity_mw'] = df['limit_mw']
        if 'reverse_capacity_mw' not in df.columns and 'reverse_limit_mw' in df.columns:
            df['reverse_capacity_mw'] = df['reverse_limit_mw']
        if 'added_cost_per_mwh' not in df.columns and 'cost_per_mwh' in df.columns:
            df['added_cost_per_mwh'] = df['cost_per_mwh']

        rename_map = {
            'contracted_flow_mw_from_to': 'contracted_flow_mw_forward',
            'contracted_flow_mw_to_from': 'contracted_flow_mw_reverse',
            'adder_cost_per_MWh': 'added_cost_per_mwh',
            'interfaceID': 'interface_id',
            'type': 'interface_type',
        }
        df = df.rename(columns={key: value for key, value in rename_map.items() if key in df.columns})

        if 'contracted_flow_mw_forward' not in df.columns and 'contracted_flow_mw' in df.columns:
            df['contracted_flow_mw_forward'] = df['contracted_flow_mw']
        if 'contracted_flow_mw_reverse' not in df.columns:
            df['contracted_flow_mw_reverse'] = df.get('uncontracted_flow_mw', 0.0)

        defaults: Dict[str, object] = {
            'interface_id': None,
            'capacity_mw': pd.NA,
            'reverse_capacity_mw': pd.NA,
            'efficiency': 1.0,
            'added_cost_per_mwh': 0.0,
            'contracted_flow_mw_forward': 0.0,
            'contracted_flow_mw_reverse': 0.0,
            'notes': None,
            'profile_id': None,
            'in_service_year': pd.NA,
            'interface_type': None,
        }
        for column, default in defaults.items():
            if column not in df.columns:
                df[column] = default

        df = _validate_columns('transmission', df, _TRANSMISSION_COLUMNS)
        df['from_region'] = _validate_region_labels(
            'transmission', 'from_region', df['from_region']
        )
        df['to_region'] = _validate_region_labels('transmission', 'to_region', df['to_region'])

        def _numeric_series(column: str, *, allow_missing: bool = True, minimum: float | None = 0.0) -> pd.Series:
            series = pd.to_numeric(df[column], errors='coerce')
            if not allow_missing and series.isna().any():
                raise ValueError(
                    f"transmission frame column '{column}' must contain numeric values"
                )
            if minimum is not None and (series.dropna() < minimum).any():
                comparator = 'non-negative' if minimum == 0.0 else f'>= {minimum}'
                raise ValueError(
                    f"transmission frame column '{column}' must contain {comparator} values"
                )
            return series.astype(float)

        df['capacity_mw'] = _numeric_series('capacity_mw')
        df['reverse_capacity_mw'] = _numeric_series('reverse_capacity_mw')

        df['efficiency'] = _numeric_series('efficiency', minimum=None)
        df['efficiency'] = df['efficiency'].fillna(1.0)
        if (df['efficiency'] <= 0.0).any():
            raise ValueError('transmission frame column "efficiency" must be positive')

        df['added_cost_per_mwh'] = _numeric_series('added_cost_per_mwh')
        df['added_cost_per_mwh'] = df['added_cost_per_mwh'].fillna(0.0)

        df['contracted_flow_mw_forward'] = _numeric_series('contracted_flow_mw_forward')
        df['contracted_flow_mw_reverse'] = _numeric_series('contracted_flow_mw_reverse')
        df['contracted_flow_mw_forward'] = df['contracted_flow_mw_forward'].fillna(0.0)
        df['contracted_flow_mw_reverse'] = df['contracted_flow_mw_reverse'].fillna(0.0)

        def _clean_string(series: pd.Series) -> pd.Series:
            def normalize(value: object) -> str | None:
                if pd.isna(value):
                    return None
                text = str(value).strip()
                return text or None

            return series.map(normalize)

        df['interface_id'] = _clean_string(df['interface_id'])
        df['notes'] = _clean_string(df['notes'])
        df['profile_id'] = _clean_string(df['profile_id'])
        if 'interface_type' in df.columns:
            df['interface_type'] = _clean_string(df['interface_type'])
        else:
            df['interface_type'] = None

        df['in_service_year'] = pd.to_numeric(df['in_service_year'], errors='coerce').astype('Int64')

        if 'limit_mw' not in df.columns:
            df['limit_mw'] = df['capacity_mw']
        else:
            df['limit_mw'] = _numeric_series('limit_mw')

        if 'reverse_limit_mw' in df.columns:
            df['reverse_limit_mw'] = _numeric_series('reverse_limit_mw')

        if 'cost_per_mwh' not in df.columns:
            df['cost_per_mwh'] = df['added_cost_per_mwh']
        else:
            df['cost_per_mwh'] = _numeric_series('cost_per_mwh')

        df['contracted_flow_mw'] = df['contracted_flow_mw_forward']

        ordered_columns = [column for column in _TRANSMISSION_COLUMNS if column in df.columns]
        extra_columns = [column for column in df.columns if column not in ordered_columns]
        return df[ordered_columns + extra_columns].reset_index(drop=True)

    def transmission_limits(self) -> Dict[Tuple[str, str], float]:
        """Return transmission limits keyed by region pairs."""

        frame = self.transmission()
        limits: Dict[Tuple[str, str], float] = {}
        for row in frame.itertuples(index=False):
            capacity = getattr(row, 'capacity_mw', None)
            if pd.isna(capacity):
                continue
            key = (str(row.from_region), str(row.to_region))
            limits[key] = float(capacity)
        return limits

    def coverage(self) -> pd.DataFrame:
        """Return validated coverage flags by region and (optionally) year."""

        if _COVERAGE_KEY not in self._frames:
            return pd.DataFrame(columns=['region', 'year', 'covered'])

        df = self[_COVERAGE_KEY]
        df = _validate_columns('coverage', df, ['region', 'covered'])

        df['region'] = _validate_region_labels('coverage', 'region', df['region'])
        df['covered'] = _coerce_bool(df['covered'], 'coverage', 'covered')

        if 'year' in df.columns:
            year_series = df['year']
            default_mask = year_series.isna()
            if default_mask.any():
                df.loc[default_mask, 'year'] = -1
            df['year'] = _require_numeric('coverage', 'year', df['year']).astype(int)
        else:
            df = df.assign(year=-1)

        duplicates = df.duplicated(subset=['region', 'year'])
        if duplicates.any():
            dupes = df.loc[duplicates, ['region', 'year']].to_records(index=False)
            raise ValueError(
                'coverage frame contains duplicate region/year combinations: '
                + ', '.join(f'({region}, {year})' for region, year in dupes)
            )

        return df[['region', 'year', 'covered']].reset_index(drop=True)

    def coverage_for_year(self, year: int) -> Dict[str, bool]:
        """Return coverage flags for ``year`` keyed by model region."""

        coverage = self.coverage()
        if coverage.empty:
            return {}

        year = int(year)
        mapping = {
            str(row.region): bool(row.covered)
            for row in coverage.itertuples(index=False)
            if int(row.year) == -1
        }

        for row in coverage.itertuples(index=False):
            if int(row.year) == year:
                mapping[str(row.region)] = bool(row.covered)

        return mapping

    def policy(self) -> PolicySpec:
        """Return the allowance policy specification."""

        try:
            df = self[_POLICY_KEY].copy(deep=True)
        except KeyError as exc:
            if self._carbon_policy_enabled:
                raise ConfigError("enabled carbon policy requires a 'policy' frame") from exc

            self._carbon_policy_enabled = False
            self._banking_enabled = False

            empty_numeric = pd.Series(dtype=float)
            empty_cp = pd.Series(dtype=object)

            return PolicySpec(
                cap=empty_numeric,
                floor=empty_numeric,
                ccr1_trigger=empty_numeric,
                ccr1_qty=empty_numeric,
                ccr2_trigger=empty_numeric,
                ccr2_qty=empty_numeric,
                cp_id=empty_cp,
                bank0=0.0,
                full_compliance_years=set(),
                annual_surrender_frac=0.0,
                carry_pct=0.0,
                banking_enabled=False,
                enabled=False,
                ccr1_enabled=False,
                ccr2_enabled=False,
                control_period_years=None,
                resolution="annual",
            )
        required = [
            'year',
            'cap_tons',
            'floor_dollars',
            'ccr1_trigger',
            'ccr1_qty',
            'ccr2_trigger',
            'ccr2_qty',
            'cp_id',
            'full_compliance',
            'bank0',
            'annual_surrender_frac',
            'carry_pct',
        ]
        missing_columns = [column for column in required if column not in df.columns]
        if 'year' in missing_columns:
            raise ValueError("policy frame is missing required columns: year")

        policy_enabled = True
        if 'policy_enabled' in df.columns:
            enabled_series = _coerce_bool(df['policy_enabled'], 'policy', 'policy_enabled')
            unique_enabled = enabled_series.dropna().unique()
            if len(unique_enabled) == 0:
                policy_enabled = bool(self._carbon_policy_enabled)
            elif len(unique_enabled) != 1:
                raise ValueError(
                    'policy frame must provide a single policy_enabled value shared across years'
                )
            else:
                policy_enabled = bool(unique_enabled[0])

        if policy_enabled and missing_columns:
            missing_list = ', '.join(sorted(missing_columns))
            raise ConfigError(
                f'enabled carbon policy requires columns: {missing_list}'
            )

        if missing_columns:
            for column in missing_columns:
                if column == 'cp_id':
                    df[column] = 'NoPolicy'
                elif column == 'full_compliance':
                    df[column] = False
                elif column == 'bank0':
                    df[column] = 0.0
                elif column == 'annual_surrender_frac':
                    df[column] = 0.0
                elif column == 'carry_pct':
                    df[column] = 1.0
                else:
                    df[column] = 0.0

        df = _validate_columns('policy', df, required)

        resolution = 'annual'
        if 'resolution' in df.columns:
            resolution_series = df['resolution'].dropna()
            unique_res = {
                str(value).strip().lower()
                for value in resolution_series.unique()
                if str(value).strip()
            }
            if len(unique_res) > 1:
                raise ValueError(
                    'policy frame must provide a single resolution value shared across years'
                )
            if unique_res:
                resolution = unique_res.pop()
            df = df.drop(columns=['resolution'])

        df['year'] = _require_numeric('policy', 'year', df['year']).astype(int)
        if df['year'].duplicated().any():
            duplicates = sorted(df.loc[df['year'].duplicated(), 'year'].unique())
            raise ValueError('policy frame contains duplicate years: ' + ', '.join(map(str, duplicates)))

        numeric_columns = [
            'cap_tons',
            'floor_dollars',
            'ccr1_trigger',
            'ccr1_qty',
            'ccr2_trigger',
            'ccr2_qty',
            'bank0',
            'annual_surrender_frac',
            'carry_pct',
        ]
        for column in numeric_columns:
            df[column] = _require_numeric('policy', column, df[column]).astype(float)

        df['cp_id'] = df['cp_id'].astype(str)
        df['full_compliance'] = _coerce_bool(df['full_compliance'], 'policy', 'full_compliance')

        bank_values = df['bank0'].unique()
        if len(bank_values) == 0:
            bank0 = 0.0
        elif len(bank_values) != 1:
            raise ValueError('policy frame must provide a single bank0 value shared across years')
        else:
            bank0 = float(bank_values[0])

        surrender_values = df['annual_surrender_frac'].unique()
        if len(surrender_values) == 0:
            annual_surrender_frac = 0.0
        elif len(surrender_values) != 1:
            raise ValueError('policy frame must provide a single annual_surrender_frac value shared across years')
        else:
            annual_surrender_frac = float(surrender_values[0])

        carry_values = df['carry_pct'].unique()
        if len(carry_values) == 0:
            carry_pct = 1.0
        elif len(carry_values) != 1:
            raise ValueError('policy frame must provide a single carry_pct value shared across years')
        else:
            carry_pct = float(carry_values[0])

        bank_enabled = True
        if 'bank_enabled' in df.columns:
            bank_series = _coerce_bool(df['bank_enabled'], 'policy', 'bank_enabled')
            unique_bank = bank_series.unique()
            if len(unique_bank) != 1:
                raise ValueError(
                    'policy frame must provide a single bank_enabled value shared across years'
                )
            bank_enabled = bool(unique_bank[0])

        ccr1_enabled = True
        if 'ccr1_enabled' in df.columns:
            ccr1_series = _coerce_bool(df['ccr1_enabled'], 'policy', 'ccr1_enabled')
            unique_ccr1 = ccr1_series.unique()
            if len(unique_ccr1) != 1:
                raise ValueError(
                    'policy frame must provide a single ccr1_enabled value shared across years'
                )
            ccr1_enabled = bool(unique_ccr1[0])

        ccr2_enabled = True
        if 'ccr2_enabled' in df.columns:
            ccr2_series = _coerce_bool(df['ccr2_enabled'], 'policy', 'ccr2_enabled')
            unique_ccr2 = ccr2_series.unique()
            if len(unique_ccr2) != 1:
                raise ValueError(
                    'policy frame must provide a single ccr2_enabled value shared across years'
                )
            ccr2_enabled = bool(unique_ccr2[0])

        control_period_years = None
        if 'control_period_years' in df.columns:
            cp_numeric = pd.to_numeric(df['control_period_years'], errors='coerce')
            cp_numeric = cp_numeric.dropna()
            if not cp_numeric.empty:
                unique_cp = cp_numeric.unique()
                if len(unique_cp) != 1:
                    raise ValueError(
                        'policy frame must provide a single control_period_years value shared across years'
                    )
                control_candidate = unique_cp[0]
                control_int = int(control_candidate)
                if control_int <= 0:
                    raise ValueError('control_period_years must be a positive integer')
                control_period_years = control_int

        if not policy_enabled:
            bank_enabled = False

        if not bank_enabled:
            bank0 = 0.0
            carry_pct = 0.0

        index = df['year']
        cap = pd.Series(df['cap_tons'].values, index=index)
        floor = pd.Series(df['floor_dollars'].values, index=index)
        ccr1_trigger = pd.Series(df['ccr1_trigger'].values, index=index)
        ccr1_qty = pd.Series(df['ccr1_qty'].values, index=index)
        ccr2_trigger = pd.Series(df['ccr2_trigger'].values, index=index)
        ccr2_qty = pd.Series(df['ccr2_qty'].values, index=index)
        cp_id = pd.Series(df['cp_id'].values, index=index)

        full_compliance_years = {int(year) for year, flag in zip(df['year'], df['full_compliance']) if flag}

        self._carbon_policy_enabled = bool(policy_enabled)
        self._banking_enabled = bool(bank_enabled)

        return PolicySpec(
            cap=cap,
            floor=floor,
            ccr1_trigger=ccr1_trigger,
            ccr1_qty=ccr1_qty,
            ccr2_trigger=ccr2_trigger,
            ccr2_qty=ccr2_qty,
            cp_id=cp_id,
            bank0=bank0,
            full_compliance_years=full_compliance_years,
            annual_surrender_frac=annual_surrender_frac,
            carry_pct=carry_pct,
            banking_enabled=bank_enabled,
            enabled=policy_enabled,
            ccr1_enabled=ccr1_enabled,
            ccr2_enabled=ccr2_enabled,
            control_period_years=control_period_years,
            resolution=resolution,
        )


__all__ = ['Frames', 'PolicySpec', 'build_frames']


def _discover_iso_codes(root: Path) -> list[str]:
    """Return ISO identifiers discovered under known forecast directories."""

    search_roots = [
        root / "electricity" / "load_forecasts",
        root / "load_forecasts",
        root,
    ]

    for candidate_root in search_roots:
        if not candidate_root.is_dir():
            continue
        iso_dirs = [entry.name for entry in candidate_root.iterdir() if entry.is_dir()]
        if iso_dirs:
            return sorted(dict.fromkeys(iso_dirs))
    return []


def build_frames(input_root: str, selection: Dict[str, Any] | None) -> Frames:
    """Construct a :class:`Frames` object for the provided ``selection``."""

    root = Path(input_root)
    if not root.exists():
        raise FileNotFoundError(f"input_root not found: {input_root}")
    if not root.is_dir():
        raise NotADirectoryError(f"input_root is not a directory: {input_root}")

    selection = selection or {}
    load_selection = selection.get(_LOAD_KEY, {})
    if isinstance(load_selection, Mapping):
        load_mapping = dict(load_selection)
    else:
        load_mapping = {}

    raw_isos = load_mapping.get("isos") or []
    if isinstance(raw_isos, str):
        raw_isos = [raw_isos]

    iso_codes = [str(value).strip() for value in raw_isos if str(value).strip()]

    scenario = load_mapping.get("scenario")
    if not scenario:
        raise ValueError("selection.load.scenario is required")

    if not iso_codes:
        iso_codes = _discover_iso_codes(root)
        if not iso_codes:
            raise ValueError("no ISOs discovered under electricity/load_forecasts")

    frames: List[pd.DataFrame] = []
    for iso in iso_codes:
        table = load_iso_scenario_table(iso=iso, scenario=scenario, input_root=str(root))
        if table.empty:
            continue
        frames.append(table)

    if frames:
        load_frame = pd.concat(frames, ignore_index=True)
    else:
        load_frame = pd.DataFrame(columns=_LOAD_COLUMN_ORDER)

    missing = _REQUIRED_LOAD_COLUMNS - set(load_frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"load frame missing columns: {missing_list}")

    if not load_frame.empty:
        ordered_columns = [column for column in _LOAD_COLUMN_ORDER if column in load_frame.columns]
        load_frame = load_frame.loc[:, ordered_columns]

    load_frame = load_frame.reset_index(drop=True)

    demand_columns = ["year", "region", "demand_mwh"]
    demand_records: list[dict[str, object]] = []
    if not load_frame.empty:
        for _, row in load_frame.iterrows():
            region_label = _coerce_region_label(row.get("zone"))
            if not region_label:
                continue
            year_value = row.get("year")
            load_value = row.get("load_gwh")
            try:
                year_int = int(year_value)
                load_float = float(load_value)
            except (TypeError, ValueError):
                continue
            demand_records.append(
                {
                    "year": year_int,
                    "region": region_label,
                    "demand_mwh": load_float * 1_000.0,
                }
            )

    if demand_records:
        demand_frame = pd.DataFrame(demand_records, columns=demand_columns)
        demand_frame = demand_frame.sort_values(["year", "region"]).drop_duplicates(
            subset=["year", "region"], keep="last"
        )
        demand_frame = demand_frame.reset_index(drop=True)
    else:
        demand_frame = pd.DataFrame(columns=demand_columns)

    return Frames({_LOAD_KEY: load_frame, _DEMAND_KEY: demand_frame})


def _validate_region_labels(frame: str, column: str, series: pd.Series) -> pd.Series:
    """Return ``series`` coerced to canonical region identifiers."""

    if series.isna().any():
        raise ValueError(f"{frame} frame column '{column}' contains missing region values")

    normalized = series.map(normalize_region_id)
    coerced = normalized.astype(str).str.strip()
    if (coerced == "").any():
        raise ValueError(f"{frame} frame column '{column}' contains empty region labels")

    return coerced
