"""This file is the main preprocessor for the electricity model.

It established the parameters and sets that will be used in the model. It contains:
 - A class that contains all sets used in the model
 - A collection of support functions to read in and setup parameter data
 - The preprocessor function, which produces an instance of the Set class and a dict of params
 - A collection of support functions to write out the inputs to the output directory

"""

from __future__ import annotations

###################################################################################################
# Setup

# Import pacakges
from pathlib import Path
import os
import copy
from collections.abc import Iterable
from typing import Any, Mapping, cast

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = cast(object, None)

# Import python modules
from main.definitions import PROJECT_ROOT
from src.common.utilities import scale_load, scale_load_with_enduses
from io_loader import Frames
from .incentives import TechnologyIncentives


def _ensure_pandas():
    """Ensure :mod:`pandas` is available before using the preprocessor."""

    if pd is None:  # pragma: no cover - helper exercised indirectly
        raise ImportError(
            "pandas is required for src.models.electricity.scripts.preprocessor; "
            "install it with `pip install pandas`."
        )
    return pd


def _coerce_years_iterable(value: object) -> list[Any]:
    """Return ``value`` as a list of years suitable for dataframe construction."""

    if value is None:
        return []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _is_float_dtype(dtype: Any) -> bool:
    """Return ``True`` if ``dtype`` is recognised as a floating dtype."""

    _ensure_pandas()
    from pandas.api.types import is_float_dtype as _is_float

    return bool(_is_float(dtype))


def _is_integer_dtype(dtype: Any) -> bool:
    """Return ``True`` if ``dtype`` is recognised as an integer dtype."""

    _ensure_pandas()
    from pandas.api.types import is_integer_dtype as _is_int

    return bool(_is_int(dtype))


class FrameStore:
    """Mutable helper that stores frames using the :class:`Frames` contract."""

    def __init__(
        self,
        frames: Frames | Mapping[str, pd.DataFrame] | None = None,
        *,
        carbon_policy_enabled: bool | None = None,
    ) -> None:
        if frames is None:
            mapping: Mapping[str, pd.DataFrame] | None = None
        elif isinstance(frames, Frames):
            mapping = {key: frames[key] for key in frames}
        else:
            mapping = dict(frames)
        self._frames = Frames(mapping, carbon_policy_enabled=carbon_policy_enabled)

    def __getitem__(self, key: str) -> pd.DataFrame:
        return self._frames[key]

    def __setitem__(self, key: str, value: pd.DataFrame) -> None:
        self._frames = self._frames.with_frame(key, value)

    def with_frame(self, key: str, value: pd.DataFrame) -> "FrameStore":
        self._frames = self._frames.with_frame(key, value)
        return self

    def get(self, key: str, default: Any = None) -> pd.DataFrame | Any:
        try:
            return self._frames[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        try:
            self._frames[key]  # type: ignore[index]
        except KeyError:
            return False
        return True

    def __iter__(self):
        return iter(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    def keys(self):
        return self._frames.keys()

    def items(self):
        for key in self._frames:
            yield key, self._frames[key]

    def to_frames(self) -> Frames:
        return self._frames


def _build_demand_frame(load_df: pd.DataFrame | None) -> pd.DataFrame:
    """Aggregate hourly load into annual regional demand."""

    _ensure_pandas()

    if load_df is None or load_df.empty:
        return pd.DataFrame(columns=['year', 'region', 'demand_mwh'])

    df = load_df.copy()
    if not isinstance(df, pd.DataFrame):  # pragma: no cover - defensive guard
        raise TypeError('load data must be provided as a pandas DataFrame')

    if not isinstance(df.index, pd.RangeIndex):
        if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
            df = df.reset_index()

    required = {'year', 'region', 'Load'}
    missing = required - set(df.columns)
    if missing:
        missing_str = ', '.join(sorted(missing))
        raise ValueError(f'Load frame is missing required columns: {missing_str}')

    df = df[['year', 'region', 'Load']].copy()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['Load'] = pd.to_numeric(df['Load'], errors='coerce')
    df = df.dropna(subset=['year', 'region', 'Load'])
    df['year'] = df['year'].astype(int)

    aggregated = (
        df.groupby(['year', 'region'], as_index=False)['Load']
        .sum()
        .rename(columns={'Load': 'demand_mwh'})
    )

    return aggregated[['year', 'region', 'demand_mwh']]


def _build_transmission_frame(tranlimit_df: pd.DataFrame | None) -> pd.DataFrame:
    """Normalise transmission limits into the Frames contract structure."""

    _ensure_pandas()

    if tranlimit_df is None or tranlimit_df.empty:
        return pd.DataFrame(columns=['from_region', 'to_region', 'limit_mw'])

    df = tranlimit_df.copy()
    if not isinstance(df.index, pd.RangeIndex):
        if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
            df = df.reset_index()

    required = {'region', 'region1', 'TranLimit'}
    missing = required - set(df.columns)
    if missing:
        missing_str = ', '.join(sorted(missing))
        raise ValueError(
            f'TranLimit frame is missing required columns: {missing_str}'
        )

    df = df[['region', 'region1', 'TranLimit']].dropna(subset=['region', 'region1'])
    df['limit_mw'] = pd.to_numeric(df['TranLimit'], errors='coerce')
    df = df.dropna(subset=['limit_mw'])
    df['limit_mw'] = df['limit_mw'].astype(float)
    if (df['limit_mw'] < 0.0).any():
        raise ValueError('TranLimit input must provide non-negative TranLimit values')

    df = df.rename(columns={'region': 'from_region', 'region1': 'to_region'})
    df['from_region'] = df['from_region'].astype(str)
    df['to_region'] = df['to_region'].astype(str)

    aggregated = (
        df.groupby(['from_region', 'to_region'], as_index=False)['limit_mw']
        .max()
        .sort_values(['from_region', 'to_region'])
        .reset_index(drop=True)
    )

    return aggregated[['from_region', 'to_region', 'limit_mw']]


def _default_coverage_frame(setin) -> pd.DataFrame:
    """Return a coverage frame marking all configured regions as uncovered."""

    _ensure_pandas()

    regions = list(getattr(setin, 'region', []) or [])
    if not regions:
        return pd.DataFrame(columns=['region', 'year', 'covered'])

    return pd.DataFrame(
        {
            'region': regions,
            'year': [-1] * len(regions),
            'covered': [False] * len(regions),
        }
    )


def _is_carbon_policy_enabled(setin) -> bool:
    """Return ``True`` when the carbon policy should be treated as enabled."""

    explicit_flag = getattr(setin, 'carbon_policy_enabled', None)
    if explicit_flag is not None:
        return bool(explicit_flag)
    return getattr(setin, 'carbon_cap', None) is not None


def _default_policy_frame(setin) -> pd.DataFrame:
    """Construct a default allowance policy table that disables the policy."""

    _ensure_pandas()

    years = list(getattr(setin, 'years', []) or [])
    if not years:
        start_year = getattr(setin, 'start_year', None)
        years = [int(start_year)] if start_year is not None else []

    columns = [
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
        'policy_enabled',
    ]

    if not years:
        return pd.DataFrame(columns=columns)

    enabled = _is_carbon_policy_enabled(setin)
    bank0 = float(getattr(setin, 'carbon_allowance_start_bank', 0.0))

    data = {
        'year': [int(year) for year in years],
        'cap_tons': [0.0] * len(years),
        'floor_dollars': [0.0] * len(years),
        'ccr1_trigger': [0.0] * len(years),
        'ccr1_qty': [0.0] * len(years),
        'ccr2_trigger': [0.0] * len(years),
        'ccr2_qty': [0.0] * len(years),
        'cp_id': ['NoPolicy'] * len(years),
        'full_compliance': [False] * len(years),
        'bank0': [bank0] + [0.0] * (len(years) - 1),
        'annual_surrender_frac': [0.0] * len(years),
        'carry_pct': [1.0] * len(years),
        'policy_enabled': [enabled] * len(years),
    }

    return pd.DataFrame(data, columns=columns)


def _attach_contract_frames(all_frames: FrameStore, setin) -> FrameStore:
    """Ensure the Frames core contract tables are present with validated schemas."""

    if 'demand' not in all_frames:
        load_df = all_frames.get('Load')
        demand_frame = _build_demand_frame(load_df)
        all_frames['demand'] = demand_frame

    if 'transmission' not in all_frames:
        transmission_source = all_frames.get('TranLimit')
        transmission_frame = _build_transmission_frame(transmission_source)
        all_frames['transmission'] = transmission_frame

    if 'fuels' not in all_frames:
        all_frames['fuels'] = pd.DataFrame(columns=['fuel', 'covered'])

    if 'coverage' not in all_frames:
        all_frames['coverage'] = _default_coverage_frame(setin)

    if 'policy' not in all_frames:
        all_frames['policy'] = _default_policy_frame(setin)

    return all_frames

# switch to load data from csvs(0) or from db(1)
# note: this is a future feature, currently not available
db_switch = 0

if db_switch == 1:
    from sqlalchemy import create_engine, MetaData, Table, select
    from sqlalchemy.ext.automap import automap_base
    from sqlalchemy.orm import sessionmaker

# Establish paths
data_root = Path(PROJECT_ROOT, 'input', 'electricity')


DEFAULT_TECH_EMISSIONS_RATE_TON_PER_GWH = {
    1: 1000.0,
    2: 800.0,
    3: 620.0,
    4: 370.0,
    5: 0.0,
    6: 0.0,
    7: 0.0,
    8: 0.0,
    9: 0.0,
    10: 0.0,
    11: 0.0,
    12: 0.0,
    13: 0.0,
    14: 0.0,
    15: 0.0,
}

DEFAULT_CAPACITY_BUILD_LIMIT = 1_000_000.0


def _filter_disabled_capacity_build_frames(all_frames, disabled_techs):
    """Align supply-curve dependent frames and remove disabled build technologies."""

    supply_curve = all_frames.get('SupplyCurve')
    if isinstance(supply_curve, pd.DataFrame):
        if 'SupplyPrice' in all_frames and isinstance(
            all_frames['SupplyPrice'], pd.DataFrame
        ):
            price_cols = [
                col
                for col in ('region', 'season', 'tech', 'step', 'year')
                if col in supply_curve.columns
                and col in all_frames['SupplyPrice'].columns
            ]
            if price_cols:
                valid_keys = supply_curve[price_cols].drop_duplicates()
                price_frame = pd.merge(
                    valid_keys,
                    all_frames['SupplyPrice'],
                    on=price_cols,
                    how='inner',
                )
                ordered_cols = price_cols + [
                    col for col in all_frames['SupplyPrice'].columns if col not in price_cols
                ]
                all_frames['SupplyPrice'] = price_frame[ordered_cols].reset_index(drop=True)

        if 'CapFactorVRE' in all_frames and isinstance(
            all_frames['CapFactorVRE'], pd.DataFrame
        ):
            vre_cols = [
                col
                for col in ('tech', 'year', 'region', 'step')
                if col in supply_curve.columns
                and col in all_frames['CapFactorVRE'].columns
            ]
            if vre_cols:
                valid_keys = supply_curve[vre_cols].drop_duplicates()
                vre_frame = pd.merge(
                    valid_keys,
                    all_frames['CapFactorVRE'],
                    on=vre_cols,
                    how='inner',
                )
                ordered_cols = vre_cols + [
                    col
                    for col in all_frames['CapFactorVRE'].columns
                    if col not in vre_cols
                ]
                all_frames['CapFactorVRE'] = vre_frame[ordered_cols].reset_index(drop=True)

        available_techs = set(supply_curve['tech'].unique())
        if 'SupplyCurveLearning' in all_frames and isinstance(
            all_frames['SupplyCurveLearning'], pd.DataFrame
        ):
            all_frames['SupplyCurveLearning'] = all_frames['SupplyCurveLearning'][
                all_frames['SupplyCurveLearning']['tech'].isin(available_techs)
            ]

    if not disabled_techs:
        return all_frames

    disabled = {int(tech) for tech in disabled_techs}

    for key in (
        'CapCost',
        'CapCostInitial',
        'CapacityBuildLimit',
    ):
        if key in all_frames:
            frame = all_frames[key]
            if isinstance(frame, pd.DataFrame):
                all_frames[key] = frame[~frame['tech'].isin(disabled)]

    if isinstance(supply_curve, pd.DataFrame):
        available_techs = set(supply_curve['tech'].unique())
        for key in ('SupplyPrice', 'CapFactorVRE', 'SupplyCurveLearning'):
            if key not in all_frames:
                continue
            frame = all_frames[key]
            if isinstance(frame, pd.DataFrame):
                all_frames[key] = frame[frame['tech'].isin(available_techs - disabled)]

    return all_frames


def apply_allowance_overrides(
    allowances_df: pd.DataFrame,
    overrides: dict,
    cap_groups: list[str],
) -> pd.DataFrame:
    """Apply run configuration overrides to the allowance procurement data."""

    _ensure_pandas()

    if not overrides or not cap_groups:
        return allowances_df

    group_lookup = {group.lower(): group for group in cap_groups}
    per_group_overrides: dict[str, dict[int, float]] = {}
    default_override: dict[int, float] = {}
    global_override: dict[int, float] = {}

    def normalize_year_map(year_value_map):
        normalized_map: dict[int, float] = {}
        for year_key, override_value in year_value_map.items():
            try:
                year_int = int(year_key)
            except (TypeError, ValueError):
                continue
            try:
                normalized_map[year_int] = float(override_value)
            except (TypeError, ValueError):
                continue
        return normalized_map

    for group_key, override_value in overrides.items():
        if isinstance(override_value, dict):
            key_str = None if group_key is None else str(group_key)
            normalized_key = key_str.lower() if key_str is not None else None
            year_map = normalize_year_map(override_value)
            if not year_map:
                continue
            if key_str in cap_groups:
                per_group_overrides.setdefault(key_str, {}).update(year_map)
            elif normalized_key in group_lookup:
                canonical = group_lookup[normalized_key]
                per_group_overrides.setdefault(canonical, {}).update(year_map)
            elif normalized_key in {'__all__', '*'}:
                global_override.update(year_map)
            elif normalized_key in {'__default__', 'default'} or key_str in {
                '__default__',
                'default',
                None,
            }:
                default_override.update(year_map)
            else:
                continue
        else:
            try:
                year_int = int(group_key)
            except (TypeError, ValueError):
                continue
            try:
                value_float = float(override_value)
            except (TypeError, ValueError):
                continue
            default_override[year_int] = value_float

    def apply_year_overrides(target_groups, year_map):
        for group in target_groups:
            for year, value in year_map.items():
                allowances_df.loc[
                    (allowances_df['cap_group'] == group)
                    & (allowances_df['year'] == year),
                    'CarbonAllowanceProcurement',
                ] = value

    overridden_groups: set[str] = set()
    if global_override:
        apply_year_overrides(cap_groups, global_override)
        overridden_groups.update(cap_groups)

    for group_name, year_map in per_group_overrides.items():
        apply_year_overrides([group_name], year_map)
        overridden_groups.add(group_name)

    if default_override:
        remaining_groups = (
            set(cap_groups) - overridden_groups if overridden_groups else set(cap_groups)
        )
        if remaining_groups:
            apply_year_overrides(sorted(remaining_groups), default_override)

    return allowances_df


###################################################################################################
class Sets:
    """Generates an initial batch of sets that are used to solve electricity model. Sets include: \n
    - Scenario descriptor and model switches \n
    - Regional sets \n
    - Temporal sets \n
    - Technology type sets \n
    - Supply curve step sets \n
    - Other

    """

    def __init__(self, settings):
        # Output root
        self.OUTPUT_ROOT = settings.OUTPUT_ROOT

        # Switches
        self.sw_trade = settings.sw_trade
        self.sw_expansion = settings.sw_expansion
        self.sw_agg_years = settings.sw_agg_years
        self.sw_rm = settings.sw_rm
        self.sw_ramp = settings.sw_ramp
        self.sw_learning = settings.sw_learning
        self.sw_reserves = settings.sw_reserves
        explicit_policy_flag = getattr(settings, 'carbon_policy_enabled', None)
        self.carbon_policy_enabled = (
            None if explicit_policy_flag is None else bool(explicit_policy_flag)
        )
        self.carbon_cap = settings.carbon_cap
        self.carbon_allowance_procurement_overrides = (
            getattr(settings, 'carbon_allowance_procurement', {})
        )
        self.carbon_allowance_start_bank = getattr(
            settings, 'carbon_allowance_start_bank', 0.0
        )
        self.carbon_allowance_bank_enabled = getattr(
            settings, 'carbon_allowance_bank_enabled', True
        )
        self.carbon_allowance_allow_borrowing = getattr(
            settings, 'carbon_allowance_allow_borrowing', False
        )
        self.carbon_ccr1_enabled = getattr(settings, 'carbon_ccr1_enabled', True)
        self.carbon_ccr2_enabled = getattr(settings, 'carbon_ccr2_enabled', True)
        self.carbon_cap_groups = getattr(settings, 'carbon_cap_groups', {}) or {}
        self.active_carbon_cap_groups = {}
        self.cap_groups = []
        self.cap_group_membership = pd.DataFrame(columns=['cap_group', 'region'])
        self.carbon_allowance_by_cap_group = pd.DataFrame()
        self.carbon_price_by_cap_group = pd.DataFrame()

        self.restypes = [
            'spinning',
            'regulation',
            'flex',
        ]  # old vals: 1=spinning, 2=regulation, 3=flex
        self.sw_builds = pd.read_csv(data_root / 'sw_builds.csv')
        self.sw_retires = pd.read_csv(data_root / 'sw_retires.csv')
        self.electricity_expansion_overrides = copy.deepcopy(
            getattr(settings, 'electricity_expansion_overrides', {}) or {}
        )
        self.disabled_expansion_techs = set(
            getattr(settings, 'disabled_expansion_techs', set())
        )
        if self.disabled_expansion_techs:
            mask = self.sw_builds['tech'].isin(self.disabled_expansion_techs)
            if mask.any():
                self.sw_builds.loc[mask, 'builds'] = 0
        self.capacity_build_limits = copy.deepcopy(
            getattr(settings, 'capacity_build_limits', {}) or {}
        )

        incentives = getattr(settings, 'electricity_incentives', TechnologyIncentives())
        if isinstance(incentives, TechnologyIncentives):
            self.technology_incentives = TechnologyIncentives(list(incentives))
        else:
            self.technology_incentives = TechnologyIncentives()

        # Load Setting
        self.load_scalar = settings.scale_load

        # Regional Sets
        self.region = settings.regions

        # Temporal Sets
        self.sw_temporal = settings.sw_temporal
        self.cw_temporal = settings.cw_temporal

        # Temporal Sets - Years
        self.years = settings.years
        self.y = settings.years
        self.start_year = settings.start_year
        self.year_map = settings.year_map
        self.WeightYear = settings.WeightYear

        # Temporal Sets - Seasons and Days
        self.season = range(1, self.cw_temporal['Map_s'].max() + 1)
        self.num_days = self.cw_temporal['Map_day'].max()
        self.day = range(1, self.num_days + 1)

        # Temporal Sets - Hours
        # number of time periods in a day
        self.num_hr_day = int(
            self.cw_temporal['Map_hour'].max() / self.cw_temporal['Map_day'].max()
        )
        self.h = range(1, self.num_hr_day + 1)
        # Number of time periods the model solves for: days x number of periods per day
        self.num_hr = self.num_hr_day * self.num_days
        self.hour = range(1, self.num_days * len(self.h) + 1)
        # First time period of the day and all time periods that are not the first hour
        self.hour1 = range(1, self.num_days * len(self.h) + 1, len(self.h))
        self.hour23 = list(set(self.hour) - set(self.hour1))

        # Technology Sets
        def load_and_assign_subsets(df, col):
            """create list based on tech subset assignment

            Parameters
            ----------
            df : pd.DataFrame
                data frame containing tech subsets
            col : str
                name of tech subset

            Returns
            -------
            list
                list of techs in subset
            """
            # set attributes for the main list
            main = list(df.columns)[0]
            df = df.set_index(df[main])

            # return subset of list based on col assignments
            subset_list = list(df[df[col].notna()].index)
            # print(col,subset_list)

            return subset_list

        # read in subset dataframe from inputs
        tech_subsets = pd.read_csv(data_root / 'tech_subsets.csv')
        self.tech_subset_names = tech_subsets.columns

        for tss in self.tech_subset_names:
            # create the technology subsets based on the tech_subsets input
            setattr(self, tss, load_and_assign_subsets(tech_subsets, tss))

        # Misc Inputs
        self.step = range(1, 5)
        self.TransLoss = 0.02  # Transmission losses %
        self.H2Heatrate = (
            13.84 / 1000000
        )  # 13.84 kwh/kg, for kwh/kg H2 -> 54.3, #conversion kwh/kg to GWh/kg


###################################################################################################
# functions to read in and setup parameter data


### Load csvs
def readin_csvs(
    frames: Frames | Mapping[str, pd.DataFrame] | None = None,
) -> Frames:
    """Read CSV input data returning a :class:`Frames` container."""

    _ensure_pandas()

    frames_obj = Frames() if frames is None else Frames.coerce(frames)
    csv_dir = Path(data_root, 'cem_inputs')
    for filename in os.listdir(csv_dir):
        f = Path(csv_dir, filename)
        if f.is_file():
            df = pd.read_csv(f)
            frames_obj = frames_obj.with_frame(filename[:-4], df)
    return frames_obj


### Load table from SQLite DB
def readin_sql(
    frames: Frames | Mapping[str, pd.DataFrame] | None = None,
) -> Frames:
    """Reads in all of the tables from a SQL databased and returns a dictionary of dataframes,
    where the key is the table name and the value is the table data.

    Parameters
    ----------
    all_frames : dictionary
        empty dictionary to be filled with dataframes

    Returns
    -------
    Frames
        completed container filled with dataframes from the input directory
    """
    _ensure_pandas()

    frames_obj = Frames() if frames is None else Frames.coerce(frames)
    db_dir = data_root / 'cem_inputs_database.db'
    engine = create_engine('sqlite:///' + db_dir)
    Session = sessionmaker(bind=engine)
    session = Session()

    Base = automap_base()
    Base.prepare(autoload_with=engine)
    metadata = MetaData()
    metadata.reflect(engine)

    for table in metadata.tables.keys():
        df = load_data(table, metadata, engine)
        df = df.drop(columns=['id'])
        frames_obj = frames_obj.with_frame(table, df)

    session.close()

    return frames_obj


def load_data(tablename, metadata, engine):
    """loads the data from the SQL database; used in readin_sql function.

    Parameters
    ----------
    tablename : string
        table name
    metadata : SQL metadata
        SQL metadata
    engine : SQL engine
        SQL engine

    Returns
    -------
    dataframe
        table from SQL db as a dataframe
    """
    _ensure_pandas()

    table = Table(tablename, metadata, autoload_with=engine)
    query = select(table.c).where()

    with engine.connect() as connection:
        result = connection.execute(query)
        df = pd.read_sql(query, connection)

    return df


def subset_dfs(all_frames, setin, i):
    """filters dataframes based on the values within the set

    Parameters
    ----------
    all_frames : dictionary
        dictionary of dataframes where the key is the file name and the value is the table data
    setin : Sets
        contains an initial batch of sets that are used to solve electricity model
    i : string
        name of the set contained within the sets class that the df will be filtered based on.

    Returns
    -------
    dictionary
        completed dictionary filled with dataframes filtered based on set inputs specified
    """
    for key in all_frames:
        if i in all_frames[key].columns:
            all_frames[key] = all_frames[key].loc[all_frames[key][i].isin(getattr(setin, i))]

    return all_frames


def _flatten_capacity_build_limit_overrides(overrides) -> pd.DataFrame:
    """Convert nested run configuration overrides into a dataframe."""

    if not overrides:
        return pd.DataFrame(
            columns=['region', 'tech', 'year', 'step', 'CapacityBuildLimit']
        )

    records: list[dict] = []
    for region, tech_map in overrides.items():
        if not isinstance(tech_map, dict):
            continue
        for tech, year_map in tech_map.items():
            if not isinstance(year_map, dict):
                continue
            for year, value in year_map.items():
                if isinstance(value, dict):
                    for step, limit in value.items():
                        records.append(
                            {
                                'region': region,
                                'tech': tech,
                                'year': year,
                                'step': step,
                                'CapacityBuildLimit': limit,
                            }
                        )
                else:
                    records.append(
                        {
                            'region': region,
                            'tech': tech,
                            'year': year,
                            'CapacityBuildLimit': value,
                        }
                    )

    if not records:
        return pd.DataFrame(
            columns=['region', 'tech', 'year', 'step', 'CapacityBuildLimit']
        )

    df = pd.DataFrame.from_records(records)
    if 'step' not in df.columns:
        df['step'] = pd.NA
    return df


def _prepare_capacity_limit_overrides(df: pd.DataFrame, base_df: pd.DataFrame):
    """Return per-year and per-step override frames coerced to base dtypes."""

    if df is None or df.empty:
        empty_year = pd.DataFrame(columns=['region', 'tech', 'year', 'CapacityBuildLimit'])
        empty_step = pd.DataFrame(
            columns=['region', 'tech', 'year', 'step', 'CapacityBuildLimit']
        )
        return empty_year, empty_step

    prepared = df.copy()
    if isinstance(prepared.index, pd.MultiIndex):
        prepared = prepared.reset_index()

    required_cols = {'region', 'tech', 'year', 'CapacityBuildLimit'}
    missing = required_cols - set(prepared.columns)
    if missing:
        missing_str = ', '.join(sorted(missing))
        raise ValueError(
            'CapacityBuildLimit input must include the following columns: '
            f'{missing_str}'
        )

    prepared['region'] = pd.to_numeric(prepared['region'], errors='coerce')
    prepared['year'] = pd.to_numeric(prepared['year'], errors='coerce')
    prepared['CapacityBuildLimit'] = pd.to_numeric(
        prepared['CapacityBuildLimit'], errors='coerce'
    )

    tech_dtype = base_df['tech'].dtype
    if _is_integer_dtype(tech_dtype) or _is_float_dtype(tech_dtype):
        prepared['tech'] = pd.to_numeric(prepared['tech'], errors='coerce')
    else:
        prepared['tech'] = prepared['tech'].astype(str)

    if 'step' in prepared.columns:
        prepared['step'] = pd.to_numeric(prepared['step'], errors='coerce')
    else:
        prepared['step'] = pd.NA

    prepared = prepared.dropna(subset=['region', 'tech', 'year', 'CapacityBuildLimit'])

    region_dtype = base_df['region'].dtype
    year_dtype = base_df['year'].dtype
    step_dtype = base_df['step'].dtype

    prepared['region'] = prepared['region'].astype(region_dtype)
    prepared['year'] = prepared['year'].astype(year_dtype)
    if _is_integer_dtype(tech_dtype) or _is_float_dtype(tech_dtype):
        prepared['tech'] = prepared['tech'].astype(tech_dtype)
    else:
        prepared['tech'] = prepared['tech'].astype(base_df['tech'].dtype)

    per_step = prepared[prepared['step'].notna()].copy()
    if not per_step.empty:
        if _is_integer_dtype(step_dtype) or _is_float_dtype(step_dtype):
            per_step['step'] = per_step['step'].astype(step_dtype)
        else:
            per_step['step'] = per_step['step'].astype(base_df['step'].dtype)
        per_step = per_step[
            ['region', 'tech', 'year', 'step', 'CapacityBuildLimit']
        ].drop_duplicates()

    per_year = prepared[prepared['step'].isna()].copy()
    per_year = per_year[['region', 'tech', 'year', 'CapacityBuildLimit']].drop_duplicates()

    return per_year, per_step


def _overlay_capacity_limits(base_df: pd.DataFrame, override_df: pd.DataFrame, match_cols):
    """Overlay override values onto the base capacity limit table."""

    if override_df is None or override_df.empty:
        return base_df

    override = override_df.copy()
    override = override[match_cols + ['CapacityBuildLimit']]
    override = override.drop_duplicates(match_cols, keep='last')
    override = override.rename(columns={'CapacityBuildLimit': 'CapacityBuildLimit_override'})

    merged = base_df.merge(override, on=match_cols, how='left')
    mask = merged['CapacityBuildLimit_override'].notna()
    if mask.any():
        merged.loc[mask, 'CapacityBuildLimit'] = merged.loc[
            mask, 'CapacityBuildLimit_override'
        ]
    merged = merged.drop(columns=['CapacityBuildLimit_override'])
    return merged


def build_capacity_build_limits(all_frames, setin):
    """Create the capacity build limit frame with CSV and override data."""

    cap_cost_df = all_frames.get('CapCost')
    if cap_cost_df is None or cap_cost_df.empty:
        return pd.DataFrame(
            columns=['region', 'tech', 'year', 'step', 'CapacityBuildLimit']
        )

    base_index = cap_cost_df[['region', 'tech', 'year', 'step']].drop_duplicates()
    base_index = base_index.reset_index(drop=True)
    base_index = base_index.astype(cap_cost_df[['region', 'tech', 'year', 'step']].dtypes.to_dict())
    base_index['CapacityBuildLimit'] = float(DEFAULT_CAPACITY_BUILD_LIMIT)

    raw_limits = all_frames.get('CapacityBuildLimit')
    if raw_limits is not None and not raw_limits.empty:
        per_year, per_step = _prepare_capacity_limit_overrides(raw_limits, base_index)
        if not per_year.empty:
            base_index = _overlay_capacity_limits(
                base_index, per_year, ['region', 'tech', 'year']
            )
        if not per_step.empty:
            base_index = _overlay_capacity_limits(
                base_index, per_step, ['region', 'tech', 'year', 'step']
            )

    override_df = _flatten_capacity_build_limit_overrides(
        getattr(setin, 'capacity_build_limits', {})
    )
    if not override_df.empty:
        per_year, per_step = _prepare_capacity_limit_overrides(override_df, base_index)
        if not per_year.empty:
            base_index = _overlay_capacity_limits(
                base_index, per_year, ['region', 'tech', 'year']
            )
        if not per_step.empty:
            base_index = _overlay_capacity_limits(
                base_index, per_step, ['region', 'tech', 'year', 'step']
            )

    disabled_techs = getattr(setin, 'disabled_expansion_techs', set())
    if disabled_techs:
        base_index.loc[base_index['tech'].isin(disabled_techs), 'CapacityBuildLimit'] = 0.0

    index_cols = ['region', 'tech', 'year', 'step']
    columns = index_cols + ['CapacityBuildLimit']
    base_index = base_index[columns]
    base_index = base_index.sort_values(index_cols).reset_index(drop=True)
    return base_index


def _expand_frame_to_cap_groups(df, groups, group_column='cap_group'):
    """Ensure a dataframe contains entries for each requested cap group."""

    if df is None or df.empty or not groups:
        columns = [group_column] + [col for col in (df.columns if df is not None else []) if col != group_column]
        return pd.DataFrame(columns=columns)

    df = df.copy()
    groups = {str(group) for group in groups}

    if group_column in df.columns:
        df[group_column] = df[group_column].astype(str)
        df = df[df[group_column].isin(groups)]
    else:
        df['_tmp'] = 1
        groups_df = pd.DataFrame({group_column: sorted(groups)})
        groups_df['_tmp'] = 1
        df = pd.merge(groups_df, df, on='_tmp', how='outer').drop(columns=['_tmp'])

    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)

    column_order = [group_column] + [col for col in df.columns if col != group_column]
    df = df[column_order]
    return df


def _apply_cap_group_value_overrides(df, group_definitions, value_col, override_keys):
    """Apply overrides defined in the configuration to cap group values."""

    if df is None or df.empty or not group_definitions:
        return df

    df = df.copy()
    if 'year' in df.columns:
        df['year'] = df['year'].astype(int)
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

    for group_name, config in group_definitions.items():
        if not isinstance(config, dict):
            continue
        overrides = None
        for key in override_keys:
            value = config.get(key)
            if isinstance(value, dict) and value:
                overrides = value
                break
        if not overrides:
            continue
        for year_key, override_value in overrides.items():
            try:
                year = int(year_key)
                override_float = float(override_value)
            except (TypeError, ValueError):
                continue
            mask = (df['cap_group'] == str(group_name)) & (df['year'] == year)
            if mask.any():
                df.loc[mask, value_col] = override_float
            else:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                'cap_group': [str(group_name)],
                                'year': [year],
                                value_col: [override_float],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

    df = df.sort_values(['cap_group', 'year']).reset_index(drop=True)
    return df


def build_cap_group_inputs(all_frames, setin):
    """Construct cap group membership and allowance/price tables."""

    _ensure_pandas()

    cap_group_map = all_frames.get('CarbonCapGroupMap')
    fallback_membership = all_frames.get('CarbonCapGroup')

    if cap_group_map is not None and not cap_group_map.empty:
        membership_source = cap_group_map.copy()
    elif fallback_membership is not None and not fallback_membership.empty:
        membership_source = fallback_membership.copy()
    else:
        membership_source = pd.DataFrame(
            {
                'cap_group': ['system'] * len(setin.region),
                'region': setin.region,
            }
        )

    if 'cap_group' not in membership_source.columns:
        raise ValueError('CarbonCapGroup input must include a cap_group column')
    if 'region' not in membership_source.columns:
        raise ValueError('CarbonCapGroup input must include a region column')

    membership = membership_source[['cap_group', 'region']].dropna().copy()
    membership.loc[:, 'cap_group'] = membership['cap_group'].astype(str)
    membership.loc[:, 'region'] = pd.to_numeric(membership['region'], errors='coerce')
    membership = membership.dropna(subset=['region'])
    if len(setin.region) > 0:
        region_dtype = pd.Series(setin.region).dtype
        membership.loc[:, 'region'] = membership['region'].astype(region_dtype)
    else:
        membership.loc[:, 'region'] = membership['region'].astype(int)
    membership = membership[membership['region'].isin(setin.region)]
    membership = membership.drop_duplicates()

    if membership.empty and len(setin.region) > 0:
        membership = pd.DataFrame(
            {
                'cap_group': ['system'] * len(setin.region),
                'region': setin.region,
            }
        )

    configured_groups = getattr(setin, 'carbon_cap_groups', {}) or {}
    configured_groups = {
        str(group_name): (config if isinstance(config, dict) else {})
        for group_name, config in configured_groups.items()
    }
    if configured_groups and not membership.empty:
        membership = membership[membership['cap_group'].isin(configured_groups.keys())]

        if not membership.empty:
            filtered_membership = []
            for group_name, group_members in membership.groupby('cap_group', sort=False):
                config = configured_groups.get(group_name) or {}
                if 'regions' in config:
                    allowed_regions = {int(region) for region in config['regions']}
                    group_members = group_members[
                        group_members['region'].isin(allowed_regions)
                    ]
                if not group_members.empty:
                    filtered_membership.append(group_members)
            if filtered_membership:
                membership = pd.concat(filtered_membership, ignore_index=True)
            else:
                membership = pd.DataFrame(columns=membership.columns)

    membership = membership.drop_duplicates().sort_values(['cap_group', 'region'])
    membership = membership.reset_index(drop=True)

    valid_regions = set(getattr(setin, 'region', []))
    if valid_regions:
        region_dtype = pd.Series(sorted(valid_regions)).dtype
    else:
        region_dtype = membership['region'].dtype if 'region' in membership.columns else float

    if membership.empty:
        configured_records: list[dict[str, object]] = []
        for group_name, config in configured_groups.items():
            if not isinstance(config, dict):
                continue
            regions = config.get('regions')
            if not regions:
                continue
            for region in regions:
                try:
                    region_value = int(region)
                except (TypeError, ValueError):
                    continue
                if valid_regions and region_value not in valid_regions:
                    continue
                configured_records.append(
                    {'cap_group': str(group_name), 'region': region_value}
                )
        if configured_records:
            configured_df = pd.DataFrame(configured_records)
            configured_df['cap_group'] = configured_df['cap_group'].astype(str)
            configured_df['region'] = pd.to_numeric(
                configured_df['region'], errors='coerce'
            )
            configured_df = configured_df.dropna(subset=['region'])
            if not configured_df.empty:
                configured_df['region'] = configured_df['region'].astype(region_dtype)
                membership = configured_df.drop_duplicates().sort_values(
                    ['cap_group', 'region']
                )
                membership = membership.reset_index(drop=True)
    else:
        current_groups = set(membership['cap_group'])
        group_region_map = {
            group: set(membership[membership['cap_group'] == group]['region'])
            for group in current_groups
        }
        additional_records: list[dict[str, object]] = []
        for group_name, config in configured_groups.items():
            if not isinstance(config, dict):
                continue
            group_key = str(group_name)
            if group_key not in group_region_map:
                continue
            regions = config.get('regions')
            if not regions:
                continue
            for region in regions:
                try:
                    region_value = int(region)
                except (TypeError, ValueError):
                    continue
                if valid_regions and region_value not in valid_regions:
                    continue
                if region_value in group_region_map[group_key]:
                    continue
                additional_records.append(
                    {'cap_group': group_key, 'region': region_value}
                )
                group_region_map[group_key].add(region_value)
        if additional_records:
            additional_df = pd.DataFrame(additional_records)
            additional_df['cap_group'] = additional_df['cap_group'].astype(str)
            additional_df['region'] = pd.to_numeric(
                additional_df['region'], errors='coerce'
            )
            additional_df = additional_df.dropna(subset=['region'])
            if not additional_df.empty:
                additional_df['region'] = additional_df['region'].astype(region_dtype)
                membership = pd.concat([membership, additional_df], ignore_index=True)
                membership = (
                    membership.drop_duplicates()
                    .sort_values(['cap_group', 'region'])
                    .reset_index(drop=True)
                )

    if membership.empty:
        membership = pd.DataFrame(
            {
                'cap_group': ['system'] * len(setin.region),
                'region': setin.region,
            }
        )

    active_groups = list(dict.fromkeys(membership['cap_group']))
    if not active_groups:
        active_groups = ['system']

    membership_index = membership.copy()
    membership_index['CarbonCapGroupMembership'] = 1.0
    membership_index = membership_index.set_index(['cap_group', 'region'])

    all_frames = all_frames.with_frame(
        'CarbonCapGroupMembership', membership_index[['CarbonCapGroupMembership']]
    )

    setin.cap_groups = active_groups
    setin.cap_group_membership = membership_index
    setin.cap_group_region_index = membership_index[['CarbonCapGroupMembership']]
    setin.active_carbon_cap_groups = {
        group: configured_groups.get(group, {}) for group in active_groups
    }

    coverage_frame = all_frames.get('coverage')
    coverage_records: dict[tuple[str, int], bool] = {}
    explicit_keys: set[tuple[str, int]] = set()
    if isinstance(coverage_frame, pd.DataFrame) and not coverage_frame.empty:
        coverage_df = coverage_frame.copy()
        if not isinstance(coverage_df.index, pd.RangeIndex):
            coverage_df = coverage_df.reset_index()
        if 'region' not in coverage_df.columns and 'region' in coverage_df.index.names:
            coverage_df = coverage_df.reset_index()
        if 'region' not in coverage_df.columns:
            coverage_df = pd.DataFrame(columns=['region', 'year', 'covered'])
        if 'year' not in coverage_df.columns:
            coverage_df = coverage_df.assign(year=-1)
        if 'covered' not in coverage_df.columns:
            coverage_df = coverage_df.assign(covered=False)
        coverage_df['year'] = pd.to_numeric(coverage_df['year'], errors='coerce').fillna(-1).astype(int)
        coverage_df['region'] = coverage_df['region'].astype(str)
        coverage_df['covered'] = coverage_df['covered'].astype(bool)
        coverage_records = {
            (str(row.region), int(row.year)): bool(row.covered)
            for row in coverage_df.itertuples(index=False)
        }
        explicit_keys = set(coverage_records)

    valid_regions = list(dict.fromkeys(getattr(setin, 'region', []) or []))
    label_to_value = {str(region): region for region in valid_regions}
    valid_labels = list(label_to_value)
    covered_labels = {
        str(region)
        for region in (membership['region'] if not membership.empty else [])
    }

    for label in valid_labels:
        default_key = (label, -1)
        if default_key not in coverage_records:
            coverage_records[default_key] = False

    for label in valid_labels:
        flag = label in covered_labels
        matching_keys = [key for key in coverage_records if key[0] == label]
        if matching_keys:
            for key in matching_keys:
                if key in explicit_keys:
                    continue
                coverage_records[key] = flag
        else:
            coverage_records[(label, -1)] = flag

    if coverage_records:
        coverage_entries = []
        for (label, year), covered in sorted(coverage_records.items(), key=lambda item: (item[0][0], item[0][1])):
            region_value = label_to_value.get(label, label)
            coverage_entries.append(
                {'region': region_value, 'year': int(year), 'covered': bool(covered)}
            )
        coverage_result = pd.DataFrame(coverage_entries, columns=['region', 'year', 'covered'])
        coverage_result['year'] = coverage_result['year'].astype(int)
        coverage_result['covered'] = coverage_result['covered'].astype(bool)
        all_frames = all_frames.with_frame('coverage', coverage_result)

    years = list(getattr(setin, 'years', []))
    cap_group_year_combos = None
    if active_groups and years:
        cap_group_year_combos = pd.MultiIndex.from_product(
            (active_groups, years), names=['cap_group', 'year']
        ).to_frame(index=False)

    def prepare_cap_group_table(source_df, value_col, missing_year_msg, missing_value_msg):
        default_group = active_groups[0] if active_groups else 'system'
        if source_df is None or source_df.empty:
            df = pd.DataFrame(columns=['cap_group', 'year', value_col])
        else:
            df = source_df.copy()
            if isinstance(df.index, pd.MultiIndex) or (
                'year' not in df.columns and getattr(df.index, 'name', None) == 'year'
            ):
                df = df.reset_index()
        if df.empty and cap_group_year_combos is not None:
            df = cap_group_year_combos.copy()
            df[value_col] = 0.0
            return df
        if df.empty:
            return df
        if 'year' not in df.columns:
            raise ValueError(missing_year_msg)
        if value_col not in df.columns:
            raise ValueError(missing_value_msg)
        df = df[['year', value_col] + ([
            'cap_group'
        ] if 'cap_group' in df.columns else [])]
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce').fillna(0.0)
        df[value_col] = df[value_col].astype(float)
        if 'cap_group' in df.columns:
            df['cap_group'] = df['cap_group'].astype(str)
        else:
            df['cap_group'] = default_group
        unknown_groups = set(df['cap_group']) - set(active_groups)
        if unknown_groups:
            unknown_str = ', '.join(sorted(map(str, unknown_groups)))
            raise ValueError(
                f"{value_col} input includes cap_group values not present in the "
                f'CarbonCapGroup mapping: {unknown_str}'
            )
        df = (
            df[['cap_group', 'year', value_col]]
            .groupby(['cap_group', 'year'], as_index=False)[value_col]
            .sum()
        )
        if cap_group_year_combos is not None:
            df = cap_group_year_combos.merge(df, on=['cap_group', 'year'], how='left')
            df[value_col] = df[value_col].fillna(0.0)
        return df

    allowances_df = prepare_cap_group_table(
        all_frames.get('CarbonAllowanceProcurement'),
        'CarbonAllowanceProcurement',
        'CarbonAllowanceProcurement input must include a year column',
        'CarbonAllowanceProcurement input must include a '
        'CarbonAllowanceProcurement column',
    )

    overrides = getattr(setin, 'carbon_allowance_procurement_overrides', {}) or {}
    if not allowances_df.empty and overrides and active_groups:
        allowances_df = apply_allowance_overrides(
            allowances_df, overrides, list(active_groups)
        )
    allowances_df['CarbonAllowanceProcurement'] = allowances_df[
        'CarbonAllowanceProcurement'
    ].astype(float)
    if {'cap_group', 'year'}.issubset(allowances_df.columns):
        allowances_indexed = allowances_df.set_index(['cap_group', 'year'])
    else:
        allowances_indexed = allowances_df
    all_frames = all_frames.with_frame('CarbonAllowanceProcurement', allowances_indexed)
    all_frames = all_frames.with_frame(
        'CarbonAllowanceProcurementByCapGroup', allowances_indexed
    )

    price_df = prepare_cap_group_table(
        all_frames.get('CarbonAllowancePrice'),
        'CarbonPrice',
        'CarbonAllowancePrice input must include a year column',
        'CarbonAllowancePrice input must include a CarbonPrice column',
    )
    price_df['CarbonPrice'] = price_df.get('CarbonPrice', 0.0)
    if {'cap_group', 'year'}.issubset(price_df.columns):
        price_df['CarbonPrice'] = price_df['CarbonPrice'].fillna(0.0).astype(float)
        price_indexed = price_df.set_index(['cap_group', 'year'])
    else:
        price_indexed = price_df
    all_frames = all_frames.with_frame('CarbonAllowancePrice', price_indexed)
    all_frames = all_frames.with_frame('CarbonAllowancePriceByCapGroup', price_indexed)
    all_frames = all_frames.with_frame('CarbonPrice', price_indexed)

    def prepare_ccr_table(source_name, value_col, missing_year_msg, missing_value_msg):
        df = prepare_cap_group_table(
            all_frames.get(source_name),
            value_col,
            missing_year_msg,
            missing_value_msg,
        )
        if df.empty:
            return df
        df[value_col] = df[value_col].fillna(0.0).astype(float)
        return df

    ccr1_trigger_df = prepare_ccr_table(
        'CarbonCCR1Trigger',
        'CarbonCCR1Trigger',
        'CarbonCCR1Trigger input must include a year column',
        'CarbonCCR1Trigger input must include a CarbonCCR1Trigger column',
    )
    ccr1_qty_df = prepare_ccr_table(
        'CarbonCCR1Quantity',
        'CarbonCCR1Quantity',
        'CarbonCCR1Quantity input must include a year column',
        'CarbonCCR1Quantity input must include a CarbonCCR1Quantity column',
    )
    ccr2_trigger_df = prepare_ccr_table(
        'CarbonCCR2Trigger',
        'CarbonCCR2Trigger',
        'CarbonCCR2Trigger input must include a year column',
        'CarbonCCR2Trigger input must include a CarbonCCR2Trigger column',
    )
    ccr2_qty_df = prepare_ccr_table(
        'CarbonCCR2Quantity',
        'CarbonCCR2Quantity',
        'CarbonCCR2Quantity input must include a year column',
        'CarbonCCR2Quantity input must include a CarbonCCR2Quantity column',
    )

    if not getattr(setin, 'carbon_ccr1_enabled', True) and not ccr1_qty_df.empty:
        ccr1_qty_df['CarbonCCR1Quantity'] = 0.0
    if not getattr(setin, 'carbon_ccr2_enabled', True) and not ccr2_qty_df.empty:
        ccr2_qty_df['CarbonCCR2Quantity'] = 0.0

    if not ccr1_trigger_df.empty:
        if {'cap_group', 'year'}.issubset(ccr1_trigger_df.columns):
            ccr1_trigger_indexed = ccr1_trigger_df.set_index(['cap_group', 'year'])
        else:
            ccr1_trigger_indexed = ccr1_trigger_df
        all_frames = all_frames.with_frame('CarbonCCR1Trigger', ccr1_trigger_indexed)
    if not ccr1_qty_df.empty:
        if {'cap_group', 'year'}.issubset(ccr1_qty_df.columns):
            ccr1_qty_indexed = ccr1_qty_df.set_index(['cap_group', 'year'])
        else:
            ccr1_qty_indexed = ccr1_qty_df
        all_frames = all_frames.with_frame('CarbonCCR1Quantity', ccr1_qty_indexed)
    if not ccr2_trigger_df.empty:
        if {'cap_group', 'year'}.issubset(ccr2_trigger_df.columns):
            ccr2_trigger_indexed = ccr2_trigger_df.set_index(['cap_group', 'year'])
        else:
            ccr2_trigger_indexed = ccr2_trigger_df
        all_frames = all_frames.with_frame('CarbonCCR2Trigger', ccr2_trigger_indexed)
    if not ccr2_qty_df.empty:
        if {'cap_group', 'year'}.issubset(ccr2_qty_df.columns):
            ccr2_qty_indexed = ccr2_qty_df.set_index(['cap_group', 'year'])
        else:
            ccr2_qty_indexed = ccr2_qty_df
        all_frames = all_frames.with_frame('CarbonCCR2Quantity', ccr2_qty_indexed)

    start_bank_df = prepare_cap_group_table(
        all_frames.get('CarbonStartBank'),
        'CarbonStartBank',
        'CarbonStartBank input must include a year column',
        'CarbonStartBank input must include a CarbonStartBank column',
    )
    if not start_bank_df.empty:
        start_bank_df['CarbonStartBank'] = (
            start_bank_df['CarbonStartBank'].fillna(0.0).astype(float)
        )
        if (
            getattr(setin, 'carbon_allowance_start_bank', 0.0) != 0.0
            and len(years) > 0
        ):
            first_year = years[0]
            start_bank_df.loc[
                start_bank_df['year'] == first_year, 'CarbonStartBank'
            ] = setin.carbon_allowance_start_bank
    if {'cap_group', 'year'}.issubset(start_bank_df.columns):
        start_bank_indexed = start_bank_df.set_index(['cap_group', 'year'])
    else:
        start_bank_indexed = start_bank_df
    all_frames = all_frames.with_frame('CarbonStartBank', start_bank_indexed)

    frames = all_frames.to_frames()
    try:
        policy_spec = frames.policy()
    except Exception:  # pragma: no cover - defensive guard
        policy_spec = None

    default_surrender_frac = 0.0
    default_carry_pct = 1.0
    if policy_spec is not None:
        try:
            default_surrender_frac = float(getattr(policy_spec, 'annual_surrender_frac', 0.0))
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            default_surrender_frac = 0.0
        try:
            default_carry_pct = float(getattr(policy_spec, 'carry_pct', 1.0))
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            default_carry_pct = 1.0

    setin.annual_surrender_frac = default_surrender_frac
    setin.carry_pct = default_carry_pct

    def _lookup_override(config: Mapping[str, Any] | Any, year: int, default: float) -> float:
        if isinstance(config, Mapping):
            for key in (year, str(year)):
                if key in config:
                    try:
                        return float(config[key])
                    except (TypeError, ValueError):
                        return float(default)
            return float(default)
        if config is None:
            return float(default)
        try:
            return float(config)
        except (TypeError, ValueError):
            return float(default)

    def _build_policy_param_frame(column: str, default_value: float, override_key: str):
        if cap_group_year_combos is None or cap_group_year_combos.empty:
            return pd.DataFrame(columns=['cap_group', 'year', column])
        records: list[dict[str, object]] = []
        for row in cap_group_year_combos.itertuples(index=False):
            cap_group = str(row.cap_group)
            year = int(row.year)
            group_config = setin.active_carbon_cap_groups.get(cap_group, {}) or {}
            override = group_config.get(override_key)
            value = _lookup_override(override, year, default_value)
            records.append({'cap_group': cap_group, 'year': year, column: value})
        df = pd.DataFrame(records, columns=['cap_group', 'year', column])
        df[column] = pd.to_numeric(df[column], errors='coerce').fillna(default_value).astype(float)
        return df

    surrender_df = _build_policy_param_frame(
        'AnnualSurrenderFrac', default_surrender_frac, 'annual_surrender_frac'
    )
    carry_df = _build_policy_param_frame('CarryPct', default_carry_pct, 'carry_pct')

    if not surrender_df.empty:
        surrender_df = surrender_df.set_index(['cap_group', 'year'])
    if not carry_df.empty:
        carry_df = carry_df.set_index(['cap_group', 'year'])

    all_frames = all_frames.with_frame('AnnualSurrenderFrac', surrender_df)
    all_frames = all_frames.with_frame('CarryPct', carry_df)

    frames = all_frames.to_frames()

    # Trigger schema validation for the Frames core contract tables.
    for accessor in (
        frames.demand,
        frames.fuels,
        frames.transmission,
        frames.coverage,
        frames.policy,
    ):
        try:
            accessor()
        except KeyError:
            continue

    return all_frames, setin


def fill_values(row, subset_list):
    """Function to fill in the subset values, is used to assign all years within the year
    solve range to each year the model will solve for.

    Parameters
    ----------
    row : int
        row number in df
    subset_list : list
        list of values to map

    Returns
    -------
    int
        value from subset_list
    """
    if row in subset_list:
        return row
    for i in range(len(subset_list) - 1):
        if subset_list[i] < row < subset_list[i + 1]:
            return subset_list[i + 1]
    return subset_list[-1]


def avg_by_group(df, set_name, map_frame):
    """takes in a dataframe and groups it by the set specified and then averages the data.

    Parameters
    ----------
    df : dataframe
        parameter data to be modified
    set_name : str
        name of the column/set to average the data by
    map_frame : dataframe
        data that maps the set name to the new grouping for that set

    Returns
    -------
    dataframe
        parameter data that is averaged by specified set mapping
    """
    map_df = map_frame.copy()
    df = df.sort_values(by=list(df[:-1]))
    # print(df.tail())

    # location of y column and list of cols needed for the groupby
    pos = df.columns.get_loc(set_name)
    map_name = 'Map_' + set_name
    groupby_cols = list(df.columns[:-1]) + [map_name]
    groupby_cols.remove(set_name)

    # group df by year map data and update y col
    df[set_name] = df[set_name].astype(int)
    df = pd.merge(df, map_df, how='left', on=[set_name])
    df = df.groupby(by=groupby_cols, as_index=False).mean()
    df[set_name] = df[map_name].astype(int)
    df = df.drop(columns=[map_name]).reset_index(drop=True)

    # move back to original position
    y_col = df.pop(set_name)
    df.insert(pos, set_name, y_col)

    # used to qa
    df = df.sort_values(by=list(df[:-1]))
    # print(df.tail())

    return df


# add seasons to data without seasons
def add_season_index(cw_temporal, df, pos):
    """adds a season index to the input dataframe

    Parameters
    ----------
    cw_temporal : dataframe
        dataframe that includes the season index
    df : dataframe
        parameter data to be modified
    pos : int
        column position for the seasonal set

    Returns
    -------
    dataframe
        modified parameter data now indexed by season
    """
    df_s = cw_temporal[['Map_s']].copy().rename(columns={'Map_s': 'season'}).drop_duplicates()
    df = pd.merge(df, df_s, how='cross')
    s_col = df.pop('season')
    df.insert(pos, 'season', s_col)

    return df


def time_map(cw_temporal, rename_cols):
    """create temporal mapping parameters

    Parameters
    ----------
    cw_temporal : pd.DataFrame
        temporal crosswalks
    rename_cols : dict
        columns to rename from/to

    Returns
    -------
    pd.DataFrame
        data frame with temporal mapping parameters
    """
    df = cw_temporal[list(rename_cols.keys())].rename(columns=rename_cols).drop_duplicates()
    return df


def capacitycredit_df(all_frames, setin):
    """builds the capacity credit dataframe

    Parameters
    ----------
    all_frames : dict of pd.DataFrame
        dictionary of dataframes where the key is the file name and the value is the table data
    setin : Sets
        an initial batch of sets that are used to solve electricity model

    Returns
    -------
    pd.DataFrame
        formatted capacity credit data frame
    """
    df = pd.merge(
        all_frames['SupplyCurve'], all_frames['MapHourSeason'], on=['season'], how='left'
    ).drop(columns=['season'])

    # capacity credit is hourly capacity factor for vre technologies
    df = pd.merge(
        df, all_frames['CapFactorVRE'], how='left', on=['tech', 'year', 'region', 'step', 'hour']
    ).rename(columns={'CapFactorVRE': 'CapacityCredit'})

    # capacity credit = 1 for dispatchable technologies
    df['CapacityCredit'] = df['CapacityCredit'].fillna(1)

    # capacity credit is seasonal limit for hydro
    df2 = pd.merge(
        all_frames['HydroCapFactor'],
        all_frames['MapHourSeason'],
        on=['season'],
        how='left',
    ).drop(columns=['season'])
    df2['tech'] = setin.T_hydro[0]
    df = pd.merge(df, df2, how='left', on=['tech', 'region', 'hour'])
    df.loc[df['tech'].isin(setin.T_hydro), 'CapacityCredit'] = df['HydroCapFactor']
    df = df.drop(columns=['SupplyCurve', 'HydroCapFactor'])
    df = df[['tech', 'year', 'region', 'step', 'hour', 'CapacityCredit']]
    return df


def create_hourly_params(all_frames, key, cols):
    """Expands params that are indexed by season to be indexed by hour

    Parameters
    ----------
    all_frames : dict of pd.DataFrame
        dictionary of dataframes where the key is the file name and the value is the table data
    key : str
        name of data frame to access
    cols : list[str]
        column names to keep in data frame

    Returns
    -------
    pd.DataFrame
        data frame with name key with new hourly index
    """
    df = pd.merge(all_frames[key], all_frames['MapHourSeason'], on=['season'], how='left').drop(
        columns=['season']
    )
    df = df[cols]
    return df


def create_subsets(df, col, subset):
    """Create subsets off of full sets

    Parameters
    ----------
    df : pd.DataFrame
        data frame of full data
    col : str
        column name
    subset : list[str]
        names of values to subset

    Returns
    -------
    pd.DataFrame
        data frame containing subset of full data
    """
    df = df[df[col].isin(subset)].dropna()
    return df


def create_hourly_sets(all_frames, df):
    """expands sets that are indexed by season to be indexed by hour

    Parameters
    ----------
    all_frames : dict of pd.DataFrame
        dictionary of dataframes where the key is the file name and the value is the table data
    df : pd.DataFrame
        data frame containing seasonal data

    Returns
    -------
    pd.DataFrame
        data frame containing updated hourly set
    """
    df = pd.merge(df, all_frames['MapHourSeason'].reset_index(), on=['season'], how='left').drop(
        columns=['season']
    )
    return df


def hourly_sc_subset(all_frames, subset):
    """Creates sets/subsets that are related to the supply curve

    Parameters
    ----------
    all_frames : dict of pd.DataFrame
        dictionary of dataframes where the key is the file name and the value is the table data
    subset : list
        list of technologies to subset

    Returns
    -------
    pd.DataFrame
        data frame containing sets/subsets related to supply curve
    """
    df = create_hourly_sets(
        all_frames, create_subsets(all_frames['SupplyCurve'].reset_index(), 'tech', subset)
    )
    return df


def hr_sub_sc_subset(all_frames, T_subset, hr_subset):
    """creates supply curve subsets by hour

    Parameters
    ----------
    all_frames : dict of pd.DataFrame
        dictionary of dataframes where the key is the file name and the value is the table data
    T_subset : list
        list of technologies to subset
    hr_subset : list
        list of hours to subset

    Returns
    -------
    pd.DataFrame
        data frame containing supply curve related hourly subset
    """
    il = ['tech', 'year', 'region', 'step', 'hour']
    df_index = create_subsets(hourly_sc_subset(all_frames, T_subset), 'hour', hr_subset).set_index(
        il
    )
    return df_index


def step_sub_sc_subset(all_frames, T_subset, step_subset):
    """creates supply curve subsets by step

    Parameters
    ----------
    all_frames : dict of pd.DataFrame
        dictionary of dataframes where the key is the file name and the value is the table data
    T_subset : list
       technologies to subset
    step_subset : list
        step numbers to subset

    Returns
    -------
    pd.DataFrame
        data frame containing supply curve subsets by step
    """
    df = create_subsets(
        create_subsets(all_frames['SupplyCurve'].reset_index(), 'tech', T_subset),
        'step',
        step_subset,
    )
    return df


def create_sc_sets(all_frames, setin):
    """creates supply curve sets

    Parameters
    ----------
    all_frames : dict of pd.DataFrame
        dictionary of dataframes where the key is the file name and the value is the table data
    setin : Sets
        an initial batch of sets that are used to solve electricity model

    Returns
    -------
    Sets
       updated Set containing all sets related to supply curve
    """
    # sets that are related to the supply curve
    index_list = ['tech', 'year', 'region', 'step', 'hour']

    setin.generation_total_index = hourly_sc_subset(all_frames, setin.T_gen).set_index(index_list)
    setin.Storage_index = hourly_sc_subset(all_frames, setin.T_stor).set_index(index_list)
    setin.H2Gen_index = hourly_sc_subset(all_frames, setin.T_h2).set_index(index_list)
    setin.generation_ramp_index = hourly_sc_subset(all_frames, setin.T_conv).set_index(index_list)
    setin.generation_dispatchable_ub_index = hourly_sc_subset(all_frames, setin.T_disp).set_index(
        index_list
    )

    setin.ramp_most_hours_balance_index = hr_sub_sc_subset(all_frames, setin.T_conv, setin.hour23)
    setin.ramp_first_hour_balance_index = hr_sub_sc_subset(all_frames, setin.T_conv, setin.hour1)
    setin.storage_most_hours_balance_index = hr_sub_sc_subset(
        all_frames, setin.T_stor, setin.hour23
    )
    setin.storage_first_hour_balance_index = hr_sub_sc_subset(all_frames, setin.T_stor, setin.hour1)

    setin.generation_hydro_ub_index = create_hourly_sets(
        all_frames, step_sub_sc_subset(all_frames, setin.T_hydro, [2])
    ).set_index(index_list)

    setin.capacity_hydro_ub_index = (
        step_sub_sc_subset(all_frames, setin.T_hydro, [1])
        .drop(columns=['step'])
        .set_index(['tech', 'year', 'region', 'season'])
    )

    setin.reserves_procurement_index = pd.merge(
        create_hourly_sets(all_frames, all_frames['SupplyCurve'].reset_index()),
        pd.DataFrame({'restypes': setin.restypes}),
        how='cross',
    ).set_index(['restypes'] + index_list)

    return setin


def create_other_sets(all_frames, setin):
    """creates other (non-supply curve) sets

    Parameters
    ----------
    all_frames : dict of pd.DataFrame
        dictionary of dataframes where the key is the file name and the value is the table data
    setin : Sets
        an initial batch of sets that are used to solve electricity model

    Returns
    -------
    Sets
        updated Sets which has non-supply curve-related sets updated
    """
    # other sets
    setin.Build_index = setin.sw_builds[setin.sw_builds['builds'] == 1].set_index(['tech', 'step'])

    setin.capacity_retirements_index = pd.merge(
        all_frames['SupplyCurve']
        .reset_index()
        .drop(columns=['season', 'SupplyCurve'])
        .drop_duplicates(),
        setin.sw_retires[setin.sw_retires['retires'] == 1],
        on=['tech', 'step'],
        how='right',
    ).set_index(['tech', 'year', 'region', 'step'])

    setin.trade_interational_index = (
        pd.merge(
            all_frames['TranLimitGenInt'].reset_index(),
            all_frames['TranLimitCapInt'].reset_index(),
            how='inner',
        )
        .drop(columns=['TranLimitGenInt'])
        .set_index(['region', 'region1', 'year', 'step', 'hour'])
    )

    setin.trade_interregional_index = create_hourly_sets(
        all_frames, all_frames['TranLimit'].reset_index()
    ).set_index(['region', 'region1', 'year', 'hour'])

    return setin


###################################################################################################
def preprocessor(setin):
    """main preprocessor function that generates the final dataframes and sets sent over to the
    electricity model. This function reads in the input data, modifies it based on the temporal
    and regional mapping specified in the inputs, and gets it into the final formatting needed.
    Also adds some additional regional sets to the set class based on parameter inputs.

    Parameters
    ----------
    setin : Sets
        an initial batch of sets that are used to solve electricity model

    Returns
    -------
    all_frames : Frames
        container of dataframes where the key is the file name and the value is the table data
    setin : Sets
        an initial batch of sets that are used to solve electricity model
    """

    # READ IN INPUT DATA

    _ensure_pandas()

    # read in raw data
    frames_container: Frames | None = None
    if db_switch == 0:
        # add csv input files to all frames
        frames_container = readin_csvs()
    elif db_switch == 1:
        # add sql db tables to all frames
        frames_container = readin_sql()
    else:
        frames_container = Frames()

    carbon_policy_enabled = _is_carbon_policy_enabled(setin)
    all_frames = FrameStore(
        frames_container, carbon_policy_enabled=carbon_policy_enabled
    )

    incentives = getattr(setin, 'technology_incentives', TechnologyIncentives())
    if not isinstance(incentives, TechnologyIncentives):
        incentives = TechnologyIncentives()
    for module in incentives.modules():
        all_frames = module.apply(all_frames)

    # Ensure emissions data are available for all generation technologies
    emissions_df = all_frames.get('EmissionsRate')
    if emissions_df is None or emissions_df.empty:
        emissions_df = pd.DataFrame(
            {
                'tech': list(DEFAULT_TECH_EMISSIONS_RATE_TON_PER_GWH.keys()),
                'EmissionsRate': list(DEFAULT_TECH_EMISSIONS_RATE_TON_PER_GWH.values()),
            }
        )
    else:
        emissions_df = emissions_df.copy()
    if 'tech' not in emissions_df.columns:
        raise ValueError('EmissionsRate input must include a tech column')
    if 'EmissionsRate' not in emissions_df.columns:
        raise ValueError('EmissionsRate input must include an EmissionsRate column')
    emissions_df['EmissionsRate'] = emissions_df['EmissionsRate'].astype(float)
    missing_techs = set(setin.T_gen) - set(emissions_df['tech'])
    if missing_techs:
        unknown_techs = [
            tech
            for tech in missing_techs
            if tech not in DEFAULT_TECH_EMISSIONS_RATE_TON_PER_GWH
        ]
        if unknown_techs:
            missing_str = ', '.join(str(tech) for tech in sorted(unknown_techs, key=str))
            raise ValueError(
                'EmissionsRate input is missing emissions rate data for technologies: '
                f'{missing_str}'
            )

        defaults_to_add = sorted(
            (
                tech
                for tech in missing_techs
                if tech in DEFAULT_TECH_EMISSIONS_RATE_TON_PER_GWH
            ),
            key=str,
        )
        if defaults_to_add:
            defaults_df = pd.DataFrame(
                {
                    'tech': defaults_to_add,
                    'EmissionsRate': [
                        DEFAULT_TECH_EMISSIONS_RATE_TON_PER_GWH[tech]
                        for tech in defaults_to_add
                    ],
                }
            )
            try:
                defaults_df = defaults_df.astype(
                    emissions_df[['tech', 'EmissionsRate']].dtypes.to_dict()
                )
            except Exception:  # pragma: no cover - defensive dtype alignment
                pass

            emissions_df = pd.concat(
                [
                    emissions_df,
                    defaults_df,
                ],
                ignore_index=True,
            )
    emissions_df = emissions_df[emissions_df['tech'].isin(setin.T_gen)]
    all_frames['EmissionsRate'] = emissions_df

    # read in load data from residential input directory
    res_dir = Path(PROJECT_ROOT, 'input', 'residential')
    if setin.load_scalar == 'annual':
        all_frames['Load'] = scale_load(res_dir).reset_index(drop=True)
    elif setin.load_scalar == 'enduse':
        all_frames['Load'] = scale_load_with_enduses(res_dir).reset_index(drop=True)
    else:
        raise ValueError('load_scalar in TOML must be set to "annual" or "enduse"')

    # REGIONALIZE DATA

    # international trade sets
    r_file = all_frames['TranLimitCapInt'][['region', 'region1']].drop_duplicates()
    r_file = r_file[r_file['region'].isin(setin.region)]
    setin.region_int_trade = list(r_file['region'].unique())
    setin.region_int = list(r_file['region1'].unique())
    setin.region1 = setin.region + setin.region_int

    # subset df by region
    all_frames = subset_dfs(all_frames, setin, 'region')
    all_frames = subset_dfs(all_frames, setin, 'region1')

    all_frames, setin = build_cap_group_inputs(all_frames, setin)

    setin.region_trade = all_frames['TranLimit']['region'].unique()

    # TEMPORALIZE DATA

    # create temporal mapping df
    cw_temporal = setin.cw_temporal

    # year weights
    all_frames['WeightYear'] = setin.WeightYear

    # last year values used
    filter_list = ['CapCost', 'CapacityBuildLimit']
    if 'CarbonAllowanceProcurement' in all_frames:
        filter_list.append('CarbonAllowanceProcurement')
    for key in filter_list:
        if key not in all_frames:
            continue
        df = all_frames[key]
        years = getattr(setin, 'years')
        if isinstance(df, pd.DataFrame):
            if 'year' in df.columns:
                all_frames[key] = df.loc[df['year'].isin(years)]
            elif isinstance(df.index, pd.MultiIndex) and 'year' in df.index.names:
                selector = df.index.get_level_values('year').isin(years)
                all_frames[key] = df.loc[selector]

    # average values in years/hours used
    for key in all_frames.keys():
        if 'year' in all_frames[key].columns:
            all_frames[key] = avg_by_group(all_frames[key], 'year', setin.year_map)
        if 'hour' in all_frames[key].columns:
            all_frames[key] = avg_by_group(
                all_frames[key], 'hour', cw_temporal[['hour', 'Map_hour']]
            )

    all_frames['MapDaySeason'] = time_map(cw_temporal, {'Map_day': 'day', 'Map_s': 'season'})
    all_frames['MapHourDay'] = time_map(cw_temporal, {'Map_hour': 'hour', 'Map_day': 'day'})
    all_frames['MapHourSeason'] = time_map(cw_temporal, {'Map_hour': 'hour', 'Map_s': 'season'})
    all_frames['WeightHour'] = time_map(
        cw_temporal, {'Map_hour': 'hour', 'WeightHour': 'WeightHour'}
    )
    all_frames['WeightDay'] = time_map(cw_temporal, {'Map_day': 'day', 'WeightDay': 'WeightDay'})

    # weights per season
    weight_season = cw_temporal[
        ['Map_s', 'Map_hour', 'WeightDay', 'WeightHour']
    ].drop_duplicates()
    weight_season = weight_season.assign(
        WeightSeason=weight_season['WeightDay'] * weight_season['WeightHour']
    )
    weight_season = (
        weight_season
        .drop(columns=['WeightDay', 'WeightHour', 'Map_hour'])
        .groupby(['Map_s'])
        .agg('sum')
        .reset_index()
        .rename(columns={'Map_s': 'season'})
    )
    all_frames['WeightSeason'] = weight_season

    if 'CarbonAllowanceProcurement' in all_frames:
        allowances_df = all_frames['CarbonAllowanceProcurement'].reset_index()
        allowances_df = pd.merge(
            allowances_df, setin.WeightYear, on='year', how='left'
        ).fillna({'WeightYear': 1})
        allowances_df['CarbonAllowanceProcurement'] = (
            allowances_df['CarbonAllowanceProcurement'] * allowances_df['WeightYear']
        )
        allowances_df = allowances_df.drop(columns=['WeightYear']).set_index(
            ['cap_group', 'year']
        )
        all_frames['CarbonAllowanceProcurement'] = allowances_df
    if 'CarbonAllowanceProcurementByCapGroup' in all_frames:
        allowances_group_df = all_frames[
            'CarbonAllowanceProcurementByCapGroup'
        ].reset_index()
        allowances_group_df = pd.merge(
            allowances_group_df, setin.WeightYear, on='year', how='left'
        ).fillna({'WeightYear': 1})
        allowances_group_df['CarbonAllowanceProcurement'] = (
            allowances_group_df['CarbonAllowanceProcurement']
            * allowances_group_df['WeightYear']
        )
        allowances_group_df = allowances_group_df[
            ['cap_group', 'year', 'CarbonAllowanceProcurement']
        ]
        all_frames['CarbonAllowanceProcurementByCapGroup'] = allowances_group_df

    # using same T_vre capacity factor for all model years and reordering columns
    all_frames['CapFactorVRE'] = pd.merge(
        all_frames['CapFactorVRE'],
        pd.DataFrame({'year': _coerce_years_iterable(getattr(setin, 'years', []))}),
        how='cross',
    )
    all_frames['CapFactorVRE'] = all_frames['CapFactorVRE'][
        ['tech', 'year', 'region', 'step', 'hour', 'CapFactorVRE']
    ]

    # Update load to be the total demand in each time segment rather than the average
    load_df = pd.merge(
        all_frames['Load'], all_frames['WeightHour'], how='left', on=['hour']
    )
    load_df['Load'] = load_df['Load'] * load_df['WeightHour']
    load_df = load_df.drop(columns=['WeightHour'])
    all_frames['Load'] = load_df

    all_frames = _attach_contract_frames(all_frames, setin)

    # add seasons to data without seasons
    all_frames['SupplyCurve'] = add_season_index(cw_temporal, all_frames['SupplyCurve'], 1)

    def price_MWh_to_GWh(dic, names: list[str]):
        """changing units of prices to all be in $/GWh so obj is $

        Parameters
        ----------
        dic : dict of pd.DataFrames
            all_frames, main dictionary containing all inputs
        names : list[str]
            names of price tables

        Returns
        -------
        dict of pd.DataFrames
            the original dict of data frames with updated price units
        """
        for name in names:
            if name not in dic:
                continue
            df = dic[name].copy()
            if name not in df.columns:
                continue
            df.loc[:, name] = df[name] * 1000
            dic[name] = df
        return dic

    all_frames = price_MWh_to_GWh(
        all_frames,
        [
            'SupplyPrice',
            'TranCost',
            'TranCostInt',
            'RegReservesCost',
            'RampUpCost',
            'RampDownCost',
            'CapCost',
            'CapCostInitial',
        ],
    )

    # Recalculate Supply Curve Learning

    # save first year of supply curve summer capacity for learning
    supply_curve_learning = all_frames['SupplyCurve'][
        (all_frames['SupplyCurve']['year'] == setin.start_year)
        & (all_frames['SupplyCurve']['season'] == 2)
    ].copy()

    # set up first year capacity for learning.
    supply_curve_learning = (
        pd.merge(supply_curve_learning, all_frames['CapCost'], how='outer')
        .drop(columns=['season', 'year', 'CapCost', 'region', 'step'])
        .rename(columns={'SupplyCurve': 'SupplyCurveLearning'})
        .groupby(['tech'])
        .agg('sum')
        .reset_index()
    )

    # if cap = 0, set to minimum unit size (0.1 for now)
    supply_curve_learning.loc[
        supply_curve_learning['SupplyCurveLearning'] == 0.0, 'SupplyCurveLearning'
    ] = 0.01

    all_frames['SupplyCurveLearning'] = supply_curve_learning

    all_frames['CapacityCredit'] = capacitycredit_df(all_frames, setin)
    all_frames['CapacityBuildLimit'] = build_capacity_build_limits(all_frames, setin)
    all_frames = _filter_disabled_capacity_build_frames(
        all_frames, getattr(setin, 'disabled_expansion_techs', set())
    )

    # expand a few parameters to be hourly
    TLCI_cols = ['region', 'region1', 'year', 'hour', 'TranLimitCapInt']
    TLGI_cols = ['region1', 'step', 'year', 'hour', 'TranLimitGenInt']
    all_frames['TranLimitCapInt'] = create_hourly_params(all_frames, 'TranLimitCapInt', TLCI_cols)
    all_frames['TranLimitGenInt'] = create_hourly_params(all_frames, 'TranLimitGenInt', TLGI_cols)

    # sets the index for all df in dict
    for key, frame in list(all_frames.items()):
        if frame.shape[1] <= 1:
            continue
        index = list(frame.columns[:-1])
        if not index:
            continue
        all_frames[key] = frame.set_index(index)

    if 'CarbonCapGroupMembership' in all_frames:
        setin.cap_group_membership = all_frames['CarbonCapGroupMembership']
        if not getattr(setin, 'cap_groups', []):
            setin.cap_groups = list(
                all_frames['CarbonCapGroupMembership']
                .index.get_level_values('cap_group')
                .unique()
            )
    if 'CarbonAllowanceProcurementByCapGroup' in all_frames:
        setin.carbon_allowance_by_cap_group = all_frames[
            'CarbonAllowanceProcurementByCapGroup'
        ]
    if 'CarbonAllowancePriceByCapGroup' in all_frames:
        setin.carbon_price_by_cap_group = all_frames[
            'CarbonAllowancePriceByCapGroup'
        ]

    # create more indices for the model
    setin = create_sc_sets(all_frames, setin)
    setin = create_other_sets(all_frames, setin)

    return all_frames, setin


###################################################################################################
# Review Inputs


def makedir(dir_out):
    """creates a folder directory based on the path provided

    Parameters
    ----------
    dir_out : str
        path of directory
    """
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)


def output_inputs(OUTPUT_ROOT):
    """function developed initial for QA purposes, writes out to csv all of the dfs and sets passed
    to the electricity model to an output directory.

    Parameters
    ----------
    OUTPUT_ROOT : str
        path of output directory

    Returns
    -------
    all_frames : dictionary
        dictionary of dataframes where the key is the file name and the value is the table data
    setin : Sets
        an initial batch of sets that are used to solve electricity model
    """
    output_path = Path(OUTPUT_ROOT, 'electricity_inputs/')
    makedir(output_path)

    years = list(pd.read_csv(PROJECT_ROOT / 'src/integrator/input/sw_year.csv').dropna()['year'])
    regions = list(pd.read_csv(PROJECT_ROOT / 'src/integrator/input/sw_reg.csv').dropna()['region'])

    # Build sets used for model
    all_frames = {}
    setA = Sets(years, regions)

    # creates the initial data
    all_frames, setB = preprocessor(setA)
    for key in all_frames:
        # print(key, list(all_frames[key].reset_index().columns))
        fname = key + '.csv'
        all_frames[key].to_csv(output_path / fname)

    return all_frames, setB


def print_sets(setin):
    """function developed initially for QA purposes, prints out all of the sets passed to the
    electricity model.

    Parameters
    ----------
    setin : Sets
        an initial batch of sets that are used to solve electricity model
    """
    set_list = dir(setin)
    set_list = sorted([x for x in set_list if '__' not in x])
    for item in set_list:
        if isinstance(getattr(setin, item), pd.DataFrame):
            print(item, ':', getattr(setin, item).reset_index().columns)
        else:
            print(item, ':', getattr(setin, item))


# all_frames, setB = output_inputs(PROJECT_ROOT)
# print_sets(setB)
