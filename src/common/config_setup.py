"""This file contains Config_settings class. It establishes the main settings used when running
the model. It takes these settings from the run_config.toml file. It has universal configurations
(e.g., configs that cut across modules and/or solve options) and module specific configs."""

###################################################################################################
# Setup

# Import packages
import pandas as pd
import numpy as np
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    try:
        import tomli as tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
        raise ModuleNotFoundError(
            'Python 3.11+ or the tomli package is required to read TOML configuration files.'
        ) from exc
from pathlib import Path
from collections import OrderedDict
from collections.abc import Mapping
import types
import argparse
import hashlib
import json
import re
import copy

# Constants
SHORT_TON_TO_METRIC_TON = 0.90718474

# Import python modules
from main.definitions import PROJECT_ROOT
from src.integrator.utilities import create_temporal_mapping
from src.common import utilities as _common_utilities
from src.common.iteration_status import IterationStatus

from engine.compute_intensity import resolve_iteration_limit
from engine.constants import MAX_UNIQUE_DIR_ATTEMPTS

make_dir = _common_utilities.make_dir

try:  # pragma: no cover - compatibility for legacy installs
    _get_downloads_directory = _common_utilities.get_downloads_directory
except AttributeError:  # pragma: no cover - fallback path
    _get_downloads_directory = None


def _resolve_downloads_directory(app_subdir: str = 'GraniteLedger') -> Path:
    """Return the configured downloads directory with a compatibility fallback."""

    if _get_downloads_directory is not None:
        try:
            return _get_downloads_directory(app_subdir=app_subdir)
        except Exception:  # pragma: no cover - defensive guard to keep config setup working
            pass

    base_path = Path.home() / 'Downloads'
    if app_subdir:
        base_path = base_path / app_subdir
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path

from src.models.electricity.scripts.technology_metadata import resolve_technology_key
from src.models.electricity.scripts.incentives import TechnologyIncentives


###################################################################################################
# Configuration Class


class Config_settings:
    """Generates the model settings that are used to solve. Settings include:  \n
    - Iterative Solve Config Settings \n
    - Spatial Config Settings \n
    - Temporal Config Settings \n
    - Electricity Config Settings \n
    - Other
    """

    _OUTPUT_OVERRIDE_KEY = 'output_name'
    _OUTPUT_NAME_SANITIZER = re.compile(r'[^A-Za-z0-9._-]+')

    def __init__(
        self,
        config_path: Path,
        args: argparse.Namespace | None = None,
        test=False,
        capacity_build_limits: Mapping | None = None,
    ):
        """Creates configuration object upon instantiation

        Parameters
        ----------
        config_path : Path
            Path to run_config.toml
        args : Namespace
            Parsed arguments fed to main.py or other parsed object
        test : bool, optional
            Used only for unit testing in unit_tests directory, by default False

        Raises
        ------
        ValueError
            No modules turned on; check run_config.toml
        ValueError
            sw_expansion: Must turn RM switch off if no expansion
        ValueError
            sw_expansion: Must turn learning switch off if no expansion

        """
        # __INIT__: Grab arguments namespace and set paths
        self.missing_checks = set()
        self.test = test
        self.args = args
        if not args:
            self.args = types.SimpleNamespace()
            self.args.op_mode = None
            self.args.debug = False
            self.args.output_name = None
            self.args.capacity_build_limits = None
        else:
            if not hasattr(self.args, 'output_name'):
                self.args.output_name = None
            if not hasattr(self.args, 'capacity_build_limits'):
                self.args.capacity_build_limits = None
        self.PROJECT_ROOT = PROJECT_ROOT
        self.config_path = Path(config_path)

        # __INIT__: Dump toml, sse args to set mode
        with open(config_path, 'rb') as src:
            config = tomllib.load(src)
        self._raw_config = copy.deepcopy(config)

        ############################################################################################
        # Universal Configs

        # __INIT__: Solve Mode
        self.default_mode = config['default_mode']
        if not self.args.op_mode:
            print('No mode arg passed, therefore...')
            self.selected_mode = self.default_mode
            print(f'using mode {self.default_mode} specified in run_config file')
        else:
            self.selected_mode = self.args.op_mode

            # __INIT__: Setting output paths
        # Setup the output directory and write out its name for other scripts to grab
        self.output_dir_status = IterationStatus(
            iterations=0,
            converged=True,
            limit=resolve_iteration_limit(MAX_UNIQUE_DIR_ATTEMPTS),
            metadata={},
        )
        if test:
            OUTPUT_ROOT = Path(PROJECT_ROOT, 'unit_tests', 'test_logs')
            self.output_dir_status = IterationStatus(
                iterations=1,
                converged=True,
                limit=resolve_iteration_limit(MAX_UNIQUE_DIR_ATTEMPTS),
                metadata={"base_name": OUTPUT_ROOT.name, "suffix": 0},
            )
        else:
            output_folder_name = self._determine_output_folder(config)
            downloads_root = _resolve_downloads_directory()
            OUTPUT_ROOT = downloads_root / output_folder_name
            OUTPUT_ROOT, self.output_dir_status = self._ensure_unique_output_dir(
                OUTPUT_ROOT
            )

        self.OUTPUT_ROOT = OUTPUT_ROOT
        self.output_folder_name = OUTPUT_ROOT.name
        make_dir(self.OUTPUT_ROOT)

        #####
        ### __INIT__: Methods and Modules Configuration
        #####

        # __INIT__: Set modules from config
        self.electricity = config['electricity']
        self.hydrogen = config['hydrogen']
        self.residential = config['residential']

        # __INIT__: Redirects and raises based on conditions
        if (not any((self.electricity, self.hydrogen, self.residential))) and (
            self.selected_mode in ('unified-combo', 'gs-combo', 'standalone')
        ):
            raise ValueError('No modules turned on; check run_config.toml')

        # __INIT__: Single module case
        if [self.electricity, self.hydrogen, self.residential].count(True) == 1:
            print('Only one module is turned on; running standalone mode')
            self.run_method = 'run_standalone'

        # __INIT__: Combinations of Modules and Mode --> run guidance
        self.method_options = {}

        match self.selected_mode:
            case 'unified-combo':
                # No elec case
                if self.hydrogen and self.residential and not self.electricity:
                    print('not an available option, running default version')
                    print(
                        'running unified-combo with electricity, hydrogen, and residential modules'
                    )
                    self.electricity = True

                # else, assign method as gs and set options
                self.run_method = 'run_unified'
                self.method_options = {}
            case 'gs-combo':
                # No elec case
                if self.hydrogen and self.residential and not self.electricity:
                    print('not an available option, running default version')
                    print('running gs-combo with electricity, hydrogen, and residential modules')
                    self.electricity = True

                # else, assign method as gs and set options
                self.run_method = 'run_gs'
                self.method_options = {
                    'update_h2_price': self.hydrogen,
                    'update_elec_price': True,
                    'update_h2_demand': self.hydrogen,
                    'update_load': self.residential,
                    'update_elec_demand': False,
                }
            case 'standalone':
                self.run_method = 'run_standalone'
            case 'elec':
                self.run_method = 'run_elec_solo'
            case 'h2':
                self.run_method = 'run_h2_solo'
            case 'residential':
                self.run_method = 'run_residential_solo'
            case _:
                raise ValueError('Mode: Nonexistant mode specified')

        # __INIT__: Iterative Solve Configs
        self.tol = config['tol']
        self.force_10 = config['force_10']
        self.max_iter = config['max_iter']

        # __INIT__: Spatial Configs
        self.regions = config['regions']

        # __INIT__:  Temporal Configs
        self.sw_temporal = config.get('sw_temporal', 'default')
        self.cw_temporal = create_temporal_mapping(self.sw_temporal)

        # __INIT__:  Temporal Configs -
        self.sw_agg_years = int(config.get('sw_agg_years', 0))
        self.years = config['years']
        if self.sw_agg_years == 1:
            self.start_year = config['start_year']
        else:
            self.start_year = self.years[0]

        ############################################################################################
        # __INIT__: Carbon Policy Configs
        self._configure_carbon_policy(config)

        ############################################################################################
        # __INIT__:  Electricity Configs
        self.sw_trade = int(config.get('sw_trade', 0))
        self.sw_rm = int(config.get('sw_rm', 0))
        self.sw_ramp = int(config.get('sw_ramp', 0))
        self.sw_reserves = int(config.get('sw_reserves', 0))
        self.sw_learning = int(config.get('sw_learning', 0))
        self.sw_expansion = int(config.get('sw_expansion', 0))

        raw_expansion_overrides = config.get('electricity_expansion_overrides')
        self.electricity_expansion_overrides = self._normalize_expansion_overrides(
            raw_expansion_overrides
        )
        self.disabled_expansion_techs = {
            tech
            for tech, allowed in self.electricity_expansion_overrides.items()
            if not allowed
        }

        self.electricity_incentives = TechnologyIncentives.from_config(
            config.get('electricity_incentives')
        )

        raw_capacity_limits = capacity_build_limits
        capacity_limits_source = 'runtime_argument'
        if raw_capacity_limits is None:
            raw_capacity_limits = getattr(self.args, 'capacity_build_limits', None)
            capacity_limits_source = 'runtime_argument'
        if raw_capacity_limits is None:
            raw_capacity_limits = config.get('capacity_build_limits')
            capacity_limits_source = 'config_file'
        if raw_capacity_limits is None:
            capacity_limits_source = 'none'
        self.capacity_build_limits = self._normalize_capacity_build_limits(
            raw_capacity_limits
        )
        self.capacity_build_limits_source = capacity_limits_source

        ############################################################################################
        # __INIT__: Residential Configs
        self.scale_load = config.get('scale_load', 'enduse')
        if test and self.scale_load == 'enduse':
            self.scale_load = 'annual'

        if not test:
            self.view_regions = config['view_regions']
            self.view_years = config['view_years']
            self.sensitivity = config['sensitivity']
            self.change_var = config['change_var']
            self.percent_change = config['percent_change']
            self.complex = config['complex']

        ############################################################################################
        # __INIT__: Hydrogen Configs
        self.h2_data_folder = self.PROJECT_ROOT / config.get(
            'h2_data_folder', 'input/hydrogen/all_regions'
        )

    def build_user_inputs_snapshot(self) -> dict:
        """Return a structured mapping of user-selected configuration inputs."""

        def _copy_mapping(data):
            if isinstance(data, Mapping):
                return {key: _copy_mapping(value) for key, value in data.items()}
            if isinstance(data, list):
                return [_copy_mapping(item) for item in data]
            if isinstance(data, tuple):
                return [_copy_mapping(item) for item in data]
            if isinstance(data, set):
                return sorted(_copy_mapping(item) for item in data)
            return data

        mode_source = 'config_file'
        if getattr(self.args, 'op_mode', None):
            mode_source = 'runtime_argument'

        run_metadata = OrderedDict(
            selected_mode=self.selected_mode,
            default_mode=self.default_mode,
            mode_source=mode_source,
            run_method=self.run_method,
            method_options=_copy_mapping(getattr(self, 'method_options', {})),
            output_directory=str(self.OUTPUT_ROOT),
            output_folder_name=self.output_folder_name,
            config_path=str(self.config_path),
            config_hash=self._config_hash(self._raw_config),
            output_name_from_args=getattr(self.args, 'output_name', None),
            output_name_from_config=self._raw_config.get(self._OUTPUT_OVERRIDE_KEY),
        )

        cli_arguments = OrderedDict(
            op_mode=getattr(self.args, 'op_mode', None),
            debug=bool(getattr(self.args, 'debug', False)),
            output_name=getattr(self.args, 'output_name', None),
            capacity_build_limits_provided=(
                getattr(self.args, 'capacity_build_limits', None) is not None
            ),
        )

        modules = OrderedDict(
            electricity=bool(self.electricity),
            hydrogen=bool(self.hydrogen),
            residential=bool(self.residential),
        )

        iterative_settings = OrderedDict(
            tol=self.tol,
            force_10=bool(self.force_10),
            max_iter=self.max_iter,
        )

        temporal_settings = OrderedDict(
            sw_temporal=self.sw_temporal,
            sw_agg_years=self.sw_agg_years,
            start_year=self.start_year,
            years=list(self.years),
        )

        spatial_settings = OrderedDict(regions=list(self.regions))

        electricity_settings = OrderedDict(
            switches=OrderedDict(
                sw_trade=self.sw_trade,
                sw_expansion=self.sw_expansion,
                sw_rm=self.sw_rm,
                sw_ramp=self.sw_ramp,
                sw_reserves=self.sw_reserves,
                sw_learning=self.sw_learning,
            ),
            expansion_overrides=_copy_mapping(
                self._raw_config.get('electricity_expansion_overrides', {})
            ),
            disabled_expansion_techs=sorted(self.disabled_expansion_techs),
        )

        residential_settings = OrderedDict()
        for key in (
            'scale_load',
            'view_regions',
            'view_years',
            'sensitivity',
            'change_var',
            'percent_change',
            'complex',
        ):
            if hasattr(self, key):
                value = getattr(self, key)
                if isinstance(value, (list, tuple, set)):
                    residential_settings[key] = list(value)
                else:
                    residential_settings[key] = value

        hydrogen_settings = OrderedDict(h2_data_folder=str(self.h2_data_folder))

        carbon_policy = OrderedDict(
            enabled=bool(self.carbon_policy_enabled),
            ccr1_enabled=bool(self.carbon_ccr1_enabled),
            ccr2_enabled=bool(self.carbon_ccr2_enabled),
            control_period_years=self.carbon_control_period_years,
            carbon_cap=self.carbon_cap,
            allowance_procurement=_copy_mapping(self.carbon_allowance_procurement),
            allowance_procurement_overrides=_copy_mapping(
                self.carbon_allowance_procurement_overrides
            ),
            allowance_start_bank=self.carbon_allowance_start_bank,
            allowance_bank_enabled=bool(self.carbon_allowance_bank_enabled),
            allowance_allow_borrowing=bool(self.carbon_allowance_allow_borrowing),
            allowance_market=_copy_mapping(self._raw_config.get('allowance_market', {})),
            default_cap_group=(
                _copy_mapping(vars(self.default_cap_group)) if self.default_cap_group else None
            ),
            cap_groups=[
                OrderedDict(
                    name=name,
                    config=_copy_mapping(group_config),
                )
                for name, group_config in self.carbon_cap_groups.items()
            ],
        )

        capacity_limits = OrderedDict(
            source=self.capacity_build_limits_source,
            values=_copy_mapping(self.get_capacity_build_limits()),
        )

        snapshot = OrderedDict(
            run_metadata=run_metadata,
            cli_arguments=cli_arguments,
            modules=modules,
            iterative_settings=iterative_settings,
            temporal_settings=temporal_settings,
            spatial_settings=spatial_settings,
            electricity_settings=electricity_settings,
            residential_settings=residential_settings,
            hydrogen_settings=hydrogen_settings,
            carbon_policy=carbon_policy,
            capacity_build_limits=capacity_limits,
            missing_validations=sorted(self.missing_checks),
            raw_configuration=_copy_mapping(self._raw_config),
        )

        return snapshot

    def _determine_output_folder(self, config: dict) -> str:
        override = self._resolve_output_override(config)
        if override is not None:
            return override
        config_hash = self._config_hash(config)
        return f"{self.selected_mode}_{config_hash}"

    def _resolve_output_override(self, config: dict) -> str | None:
        override = getattr(self.args, 'output_name', None)
        if override in (None, ''):
            override = config.get(self._OUTPUT_OVERRIDE_KEY)
        if override in (None, ''):
            return None
        return self._sanitize_output_name(override)

    @classmethod
    def _sanitize_output_name(cls, name: object) -> str:
        sanitized = cls._OUTPUT_NAME_SANITIZER.sub('_', str(name).strip())
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        if not sanitized:
            raise ValueError('Output directory override must include at least one valid character')
        return sanitized

    @classmethod
    def _config_hash(cls, config: dict) -> str:
        sanitized_config = cls._sanitize_config_for_hash(config)
        serialized = json.dumps(
            sanitized_config,
            sort_keys=True,
            default=str,
            separators=(',', ':'),
        )
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:10]

    @classmethod
    def _sanitize_config_for_hash(cls, data):
        if isinstance(data, dict):
            return {
                key: cls._sanitize_config_for_hash(value)
                for key, value in data.items()
                if key != cls._OUTPUT_OVERRIDE_KEY
            }
        if isinstance(data, list):
            return [cls._sanitize_config_for_hash(item) for item in data]
        return data

    @staticmethod
    def _ensure_unique_output_dir(candidate: Path) -> tuple[Path, IterationStatus]:
        iteration_limit = max(1, int(resolve_iteration_limit(MAX_UNIQUE_DIR_ATTEMPTS)))
        parent = candidate.parent
        base_name = candidate.name
        attempts = 0

        for suffix in range(iteration_limit):
            if suffix == 0:
                alternative = candidate
            else:
                alternative = parent / f"{base_name}_{suffix:02d}"
            attempts += 1
            if not alternative.exists():
                status = IterationStatus(
                    iterations=attempts,
                    converged=True,
                    limit=iteration_limit,
                    metadata={"base_name": base_name, "suffix": suffix},
                )
                return alternative, status

        message = (
            f"Unable to find a unique output directory for '{base_name}' after "
            f"{iteration_limit} attempts"
        )
        status = IterationStatus(
            iterations=attempts,
            converged=False,
            limit=iteration_limit,
            message=message,
            metadata={"base_name": base_name, "suffix": iteration_limit - 1},
        )
        error = RuntimeError(message)
        setattr(error, "iteration_status", status)
        raise error

    def _configure_carbon_policy(self, config: dict):
        """Parse carbon policy configuration data into normalized attributes."""

        # Establish safe defaults so downstream code can rely on attributes existing.
        self.carbon_cap_groups = OrderedDict()
        self.default_cap_group = None
        self.carbon_cap = None
        self.carbon_allowance_procurement = {}
        self.carbon_allowance_procurement_overrides = {}
        self.carbon_allowance_start_bank = 0.0
        self.carbon_allowance_bank_enabled = True
        self.carbon_allowance_allow_borrowing = False
        self.carbon_policy_enabled = True
        self.carbon_ccr1_enabled = True
        self.carbon_ccr2_enabled = True
        self.carbon_control_period_years = None

        carbon_policy_section = config.get('carbon_policy')
        if not isinstance(carbon_policy_section, dict):
            carbon_policy_section = {}

        def _get_policy_value(*keys, default=None):
            for key in keys:
                if key in carbon_policy_section:
                    value = carbon_policy_section[key]
                    if value not in (None, ''):
                        return value
            for key in keys:
                if key in config:
                    value = config[key]
                    if value not in (None, ''):
                        return value
            return default

        self.carbon_policy_enabled = self._normalize_bool(
            _get_policy_value('enabled', 'policy_enabled', 'carbon_policy_enabled'),
            default=True,
        )
        self.carbon_ccr1_enabled = self._normalize_bool(
            _get_policy_value('ccr1_enabled'),
            default=True,
        )
        self.carbon_ccr2_enabled = self._normalize_bool(
            _get_policy_value('ccr2_enabled'),
            default=True,
        )
        control_period_raw = _get_policy_value('control_period_years')
        if control_period_raw not in (None, ''):
            try:
                control_period_value = int(control_period_raw)
            except (TypeError, ValueError):
                control_period_value = None
            else:
                if control_period_value <= 0:
                    control_period_value = None
        else:
            control_period_value = None
        self.carbon_control_period_years = control_period_value

        if not self.carbon_policy_enabled:
            self.carbon_cap_groups = OrderedDict()
            self.default_cap_group = None
            return

        legacy_cap_value = self._normalize_carbon_cap(
            _get_policy_value('carbon_cap')
        )
        legacy_allowances = (
            self._normalize_year_value_schedule(
                _get_policy_value('carbon_allowance_procurement')
            )
            or {}
        )
        legacy_prices = (
            self._normalize_year_value_schedule(
                _get_policy_value('carbon_price')
            )
            or {}
        )
        legacy_start_bank = self._normalize_float(
            _get_policy_value('carbon_allowance_start_bank', 'start_bank'),
            default=0.0,
        )
        legacy_bank_enabled = self._normalize_bool(
            _get_policy_value('carbon_allowance_bank_enabled', 'bank_enabled'),
            default=True,
        )
        legacy_allow_borrowing = self._normalize_bool(
            _get_policy_value(
                'carbon_allowance_allow_borrowing', 'allow_borrowing'
            ),
            default=False,
        )
        legacy_overrides = (
            self._normalize_year_value_schedule(
                _get_policy_value('carbon_allowance_procurement_overrides')
            )
            or {}
        )

        default_regions = tuple(int(region) for region in self.regions)

        known_dict_keys = {
            'allowances',
            'allowance_procurement',
            'carbon_allowance_procurement',
            'allowance_procurement_overrides',
            'prices',
            'price',
            'carbon_price',
            'cap_schedule',
            'caps',
            'carbon_caps',
        }

        def _parse_group_entries(groups):
            """Return ordered (name, config) pairs from raw TOML structures."""

            entries: list[tuple[str, dict]] = []
            if isinstance(groups, dict):
                for group_name, group_config in groups.items():
                    entries.append((str(group_name), group_config))
            elif isinstance(groups, list):
                for idx, group_entry in enumerate(groups, start=1):
                    if isinstance(group_entry, dict):
                        entry = dict(group_entry)
                        nested_groups: list[tuple[str, dict]] = []
                        for key, value in list(entry.items()):
                            if (
                                isinstance(value, dict)
                                and key not in known_dict_keys
                            ):
                                nested_groups.append((str(key), value))
                                entry.pop(key)
                        group_name = (
                            entry.get('name')
                            or entry.get('group')
                            or entry.get('id')
                            or f'group_{idx}'
                        )
                        entry.pop('name', None)
                        entry.pop('group', None)
                        entry.pop('id', None)
                        entries.append((str(group_name), entry))
                        entries.extend(nested_groups)
                    elif group_entry is not None:
                        entries.append((str(group_entry), {}))
            elif groups is not None:
                # Handle simple string/int names.
                entries.append((str(groups), {}))
            return entries

        combined_group_configs = OrderedDict()
        for source in (
            config.get('carbon_cap_groups'),
            carbon_policy_section.get('carbon_cap_groups'),
        ):
            for group_name, group_config in _parse_group_entries(source):
                existing = combined_group_configs.get(group_name, {})
                merged = dict(existing)
                if isinstance(group_config, dict):
                    merged.update(group_config)
                combined_group_configs[group_name] = merged

        if not combined_group_configs:
            combined_group_configs['default'] = {'regions': list(self.regions)}

        built_groups = OrderedDict()
        for group_name, raw_config in combined_group_configs.items():
            normalized = dict(self._normalize_single_cap_group(raw_config))
            normalized['name'] = group_name

            cap_value = normalized.get('cap', normalized.get('carbon_cap'))
            cap_float = None
            if isinstance(cap_value, str):
                lowered = cap_value.strip().lower()
                if lowered not in {'none', 'null', ''}:
                    try:
                        cap_float = float(cap_value)
                    except (TypeError, ValueError):
                        cap_float = None
            elif cap_value is not None:
                try:
                    cap_float = float(cap_value)
                except (TypeError, ValueError):
                    cap_float = None
            if cap_float is None:
                cap_float = legacy_cap_value
            normalized['cap'] = cap_float

            regions = normalized.get('regions')
            if regions:
                regions = tuple(self._normalize_regions(regions))
            else:
                regions = default_regions
            normalized['regions'] = regions

            allowance_schedule = (
                normalized.get('allowances')
                or normalized.get('allowance_procurement')
                or normalized.get('carbon_allowance_procurement')
            )
            allowances = (
                self._normalize_year_value_schedule(allowance_schedule)
                if isinstance(allowance_schedule, (dict, list))
                else allowance_schedule
            )
            if not allowances:
                allowances = dict(legacy_allowances)
            normalized['allowances'] = dict(allowances)
            normalized['allowance_procurement'] = dict(allowances)
            normalized['carbon_allowance_procurement'] = dict(allowances)

            price_schedule = (
                normalized.get('prices')
                or normalized.get('price')
                or normalized.get('carbon_price')
            )
            prices = (
                self._normalize_year_value_schedule(price_schedule)
                if isinstance(price_schedule, (dict, list))
                else price_schedule
            )
            if not prices:
                prices = dict(legacy_prices)
            normalized['prices'] = dict(prices)
            normalized['price'] = dict(prices)
            normalized['carbon_price'] = dict(prices)

            overrides_schedule = normalized.get('allowance_procurement_overrides')
            overrides = (
                self._normalize_year_value_schedule(overrides_schedule)
                if isinstance(overrides_schedule, (dict, list))
                else overrides_schedule
            )
            if not overrides:
                overrides = dict(legacy_overrides)
            normalized['allowance_procurement_overrides'] = dict(overrides)
            normalized['carbon_allowance_procurement_overrides'] = dict(overrides)

            normalized['start_bank'] = self._normalize_float(
                normalized.get('start_bank', normalized.get('carbon_allowance_start_bank')),
                default=legacy_start_bank,
            )
            normalized['bank_enabled'] = self._normalize_bool(
                normalized.get('bank_enabled', normalized.get('carbon_allowance_bank_enabled')),
                default=legacy_bank_enabled,
            )
            normalized['allow_borrowing'] = self._normalize_bool(
                normalized.get(
                    'allow_borrowing',
                    normalized.get('carbon_allowance_allow_borrowing'),
                ),
                default=legacy_allow_borrowing,
            )

            built_groups[group_name] = normalized

        self.carbon_cap_groups = OrderedDict(built_groups.items())

        if self.carbon_cap_groups:
            default_name, default_config = next(iter(self.carbon_cap_groups.items()))
            allowance_map = dict(
                default_config.get('allowance_procurement', {})
            )
            self.carbon_cap = default_config.get('cap')
            self.carbon_allowance_procurement = allowance_map
            self.carbon_allowance_procurement_overrides = dict(
                default_config.get('allowance_procurement_overrides', {})
            )
            self.carbon_allowance_start_bank = float(
                default_config.get('start_bank', 0.0)
            )
            self.carbon_allowance_bank_enabled = bool(
                default_config.get('bank_enabled', True)
            )
            self.carbon_allowance_allow_borrowing = bool(
                default_config.get('allow_borrowing', False)
            )
            self.default_cap_group = types.SimpleNamespace(
                name=default_name,
                cap=self.carbon_cap,
                regions=tuple(default_config.get('regions', default_regions)),
                allowance_procurement=self.carbon_allowance_procurement,
                prices=dict(default_config.get('prices', {})),
                start_bank=self.carbon_allowance_start_bank,
                bank_enabled=self.carbon_allowance_bank_enabled,
                allow_borrowing=self.carbon_allowance_allow_borrowing,
                allowance_procurement_overrides=self.carbon_allowance_procurement_overrides,
            )
        else:
            self.default_cap_group = None



    ################################################################################################
    # Set Attributes Update

    # Runs configuration checks when you set attributes
    def __setattr__(self, name, value):
        """Update to generic setattr function that includes checks for appropriate attribute values

        Parameters
        ----------
        name : str
            attribute name
        value : _type_
            attribute value
        """
        super().__setattr__(name, value)

        # __SETATTR__: dictionary of check methods and setting attributes to pass thru those methods
        check_dict = {
            '_check_regions': {'region'},
            '_check_res_settings': {'view_regions', 'view_years', 'regions', 'years'},
            '_check_int': {'max_iter', 'start_year', 'sw_learning', 'percent_change'},
            '_check_elec_expansion_settings': {'sw_expansion', 'sw_rm', 'sw_learning'},
            '_additional_year_settings': {'sw_agg_years', 'years', 'start_year'},
            '_check_true_false': {
                'electricity',
                'hydrogen',
                'residential',
                'force_10',
                'sensitivity',
                'complex',
                'carbon_allowance_bank_enabled',
                'carbon_allowance_allow_borrowing',
                'carbon_policy_enabled',
                'carbon_ccr1_enabled',
                'carbon_ccr2_enabled',
            },
            '_check_zero_one': {
                'sw_trade',
                'sw_expansion',
                'sw_rm',
                'sw_ramp',
                'sw_reserves',
                'sw_agg_years',
            },
        }

        # __SETATTR__: Create empty all_check_sets
        all_check_sets = set()

        # __SETATTR__: For each check method pass the setting attributes through
        for check_method in check_dict.keys():
            # __SETATTR__: Add checks to all_check_sets
            all_check_sets.union(check_dict[check_method])

            # __SETATTR__: If set value in check dictionary, run check method
            if name in check_dict[check_method]:
                method = getattr(self, check_method)
                method(name, value)

        # __SETATTR__: Create a list of all the items not being checked
        if name not in all_check_sets:
            self.missing_checks.add(name)

    ################################################################################################
    # Check Methods

    def _has_all_attributes(self, attrs: set):
        """Determines if all attributes within the set exist or not

        Parameters
        ----------
        attrs : set
            set of setting attributes

        Returns
        -------
        bool
            True or False
        """
        return all(hasattr(self, attr) for attr in attrs)

    def _additional_year_settings(self, name, value):
        """Checks year related settings to see if values are within expected ranges and updates
        other settings linked to years if years is changed.

        Parameters
        ----------
        name : str
            attribute name
        value : _type_
            attribute value

        Raises
        ------
        TypeError
            Error
        """
        # Years settings
        if hasattr(self, 'years'):
            if not isinstance(self.years, list):
                raise TypeError('years: must be a list')
            if not all(isinstance(year, int) for year in self.years):
                raise TypeError('years: only include years (integers) in year list')
            if min(self.years) < 1900:
                raise TypeError('years: should probably only include years after 1900')
            if max(self.years) > 2200:
                raise TypeError('years: should probably only include years before 2200')

        # Years related settings
        if self._has_all_attributes({'sw_agg_years', 'years', 'start_year'}):
            if self.sw_agg_years == 1:
                all_years_list = list(range(self.start_year, max(self.years) + 1))
            else:
                all_years_list = self.years
            solve_array = np.array(self.years)
            mapped_list = [solve_array[solve_array >= year].min() for year in all_years_list]
            self.year_map = pd.DataFrame({'year': all_years_list, 'Map_year': mapped_list})
            self.WeightYear = (
                self.year_map.groupby(['Map_year'])
                .agg('count')
                .reset_index()
                .rename(columns={'Map_year': 'year', 'year': 'WeightYear'})
            )

    def _normalize_carbon_cap_groups(self, groups_config):
        """Standardize carbon cap group settings into dictionaries."""

        normalized = {}
        if isinstance(groups_config, dict):
            items = groups_config.items()
        elif isinstance(groups_config, list):
            items = [(group_name, {}) for group_name in groups_config]
        else:
            items = []

        for group_name, group_config in items:
            normalized[str(group_name)] = self._normalize_single_cap_group(
                group_config
            )
        return normalized

    def _normalize_single_cap_group(self, group_config):
        if group_config is None:
            return {}
        if not isinstance(group_config, dict):
            return {'value': group_config}

        normalized = {}

        allowance_data = self._coalesce_group_value(
            group_config,
            'allowances',
            'allowance_procurement',
            'carbon_allowance_procurement',
        )
        if allowance_data is not None:
            allowances = self._normalize_year_value_schedule(allowance_data)
            if allowances:
                normalized['allowances'] = allowances

        price_data = self._coalesce_group_value(
            group_config,
            'prices',
            'price',
            'carbon_price',
        )
        if price_data is not None:
            prices = self._normalize_year_value_schedule(price_data)
            if prices:
                normalized['prices'] = prices

        if 'regions' in group_config:
            regions = self._normalize_regions(group_config['regions'])
            if regions:
                normalized['regions'] = regions

        for key, value in group_config.items():
            if key in {
                'allowances',
                'allowance_procurement',
                'carbon_allowance_procurement',
                'prices',
                'price',
                'carbon_price',
                'regions',
            }:
                continue
            normalized[key] = value

        return normalized

    def _normalize_caps_by_group(self, caps_config):
        """Normalize nested carbon cap schedules keyed by group name."""

        normalized = {}
        if isinstance(caps_config, dict):
            for group_name, schedule in caps_config.items():
                normalized_schedule = self._normalize_year_value_schedule(schedule)
                if normalized_schedule:
                    normalized[str(group_name)] = normalized_schedule
        return normalized

    def _normalize_year_value_schedule(self, schedule):
        """Convert schedules keyed by year into int/float dictionaries."""

        normalized = {}
        if isinstance(schedule, dict):
            iterator = schedule.items()
        elif isinstance(schedule, list):
            iterator = []
            for item in schedule:
                if isinstance(item, dict):
                    year = item.get('year')
                    value = item.get('value')
                    if value is None:
                        value = item.get('amount')
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    year, value = item
                else:
                    continue
                iterator.append((year, value))
        else:
            iterator = []

        for year, value in iterator:
            try:
                year_key = int(year)
                value_float = float(value)
            except (TypeError, ValueError):
                continue
            normalized[year_key] = value_float
        return normalized

    def _normalize_carbon_cap(self, value):
        """Normalize carbon cap entries, handling disabled markers and units."""

        if value in (None, ''):
            return None
        if isinstance(value, str) and value.strip().lower() in {'none', 'null'}:
            return None
        try:
            return float(value) * SHORT_TON_TO_METRIC_TON
        except (TypeError, ValueError):
            return None

    def _normalize_float(self, value, default=0.0):
        """Coerce a value to float with a fallback default."""

        if value in (None, ''):
            return default
        if isinstance(value, str) and value.strip().lower() in {'none', 'null'}:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _normalize_bool(self, value, default=False):
        """Coerce an input to a boolean value."""

        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {'true', 't', 'yes', 'y', '1', 'on'}:
                return True
            if lowered in {'false', 'f', 'no', 'n', '0', 'off'}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    def _normalize_regions(self, regions):
        """Normalize region lists to integer identifiers."""

        if regions in (None, ''):
            return []
        normalized = []
        if isinstance(regions, str):
            candidates = [part.strip() for part in regions.split(',') if part.strip()]
        elif isinstance(regions, (list, tuple, set)):
            candidates = regions
        else:
            candidates = [regions]

        for region in candidates:
            try:
                normalized.append(int(region))
            except (TypeError, ValueError):
                continue
        return normalized

    def _coerce_int_or_none(self, value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _normalize_capacity_build_limits(self, limits_source):
        """Normalize capacity build limit overrides into a nested mapping."""

        if not isinstance(limits_source, Mapping):
            return {}

        normalized: dict[int, dict[str, dict[int, dict[int, float] | float]]] = {}

        for region_key, tech_map in limits_source.items():
            region = self._coerce_int_or_none(region_key)
            if region is None or not isinstance(tech_map, Mapping):
                continue
            region_entry = normalized.setdefault(region, {})
            for tech_key, year_map in tech_map.items():
                if tech_key in (None, '') or not isinstance(year_map, Mapping):
                    continue
                tech_str = str(tech_key)
                tech_entry = region_entry.setdefault(tech_str, {})
                for year_key, value in year_map.items():
                    year = self._coerce_int_or_none(year_key)
                    if year is None:
                        continue
                    if isinstance(value, Mapping):
                        step_map: dict[int, float] = {}
                        for step_key, limit_value in value.items():
                            step = self._coerce_int_or_none(step_key)
                            if step is None:
                                continue
                            try:
                                limit_float = float(limit_value)
                            except (TypeError, ValueError):
                                continue
                            step_map[step] = limit_float
                        if step_map:
                            tech_entry[year] = step_map
                    else:
                        try:
                            limit_float = float(value)
                        except (TypeError, ValueError):
                            continue
                        tech_entry[year] = limit_float

        return normalized

    def get_capacity_build_limits(self) -> dict[int, dict[str, dict[int, dict[int, float] | float]]]:
        """Return a deep copy of the configured capacity build limits."""

        return copy.deepcopy(self.capacity_build_limits)

    def _normalize_expansion_overrides(self, overrides_source):
        """Normalize per-technology expansion overrides."""

        if not isinstance(overrides_source, Mapping):
            return {}

        normalized: dict[int, bool] = {}

        for tech_key, allowed in overrides_source.items():
            tech_id = resolve_technology_key(tech_key)
            if tech_id is None:
                continue
            normalized[tech_id] = self._normalize_bool(allowed, default=True)

        return normalized

    def _coalesce_group_value(self, group_config, *keys):
        """Return the first value present in a group configuration for any key."""

        for key in keys:
            if key in group_config:
                return group_config[key]
        return None

    # TODO: no hard coded values! regions should be flexible, come up with a better check
    def _check_regions(self, name, value):
        """Checks to see if region is between the current default values of 1 and 25.

        Parameters
        ----------
        name : str
            attribute name
        value : _type_
            attribute value

        Raises
        ------
        TypeError
            Error
        """
        if not isinstance(value, list):
            raise TypeError('regions: must be a list')
        if min(value) < 1:
            raise ValueError('regions: Nonexistant region specified')
        if max(value) > 25:
            raise ValueError('regions: Nonexistant region specified')

    def _check_elec_expansion_settings(self, name, value):
        """Checks that switches for reserve margin and learning are on only if expansion is on.

        Parameters
        ----------
        name : str
            attribute name
        value : _type_
            attribute value

        Raises
        ------
        TypeError
            Error
        """
        # You cannot run with a reserve margin without expansion
        if self._has_all_attributes({'sw_expansion', 'sw_rm'}):
            if self.sw_expansion == 0:  # if expansion is off
                if self.sw_rm == 1:
                    raise ValueError('sw_expansion: Must turn RM switch off if no expansion')

        # You cannot run with learning without expansion
        if self._has_all_attributes({'sw_expansion', 'sw_learning'}):
            if self.sw_expansion == 0:  # if expansion is off
                if self.sw_learning == 1:
                    raise ValueError('sw_expansion: Must turn learning switch off if no expansion')

    def _check_true_false(self, name, value):
        """Checks if attribute is either true or false

        Parameters
        ----------
        name : str
            attribute name
        value : _type_
            attribute value

        Raises
        ------
        TypeError
            Error
        """
        if value is True:
            pass
        elif value is False:
            pass
        else:
            raise ValueError(f'{name}: Must be either true or false')

    def _check_int(self, name, value):
        """Checks if attribute is an integer

        Parameters
        ----------
        name : str
            attribute name
        value : _type_
            attribute value

        Raises
        ------
        TypeError
            Error
        """
        if not isinstance(value, int):
            raise ValueError(f'{name}: Must be an integer')

    def _check_zero_one(self, name, value):
        """Checks if attribute is either zero or one

        Parameters
        ----------
        name : str
            attribute name
        value : _type_
            attribute value

        Raises
        ------
        TypeError
            Error
        """
        if (value == 0) or (value == 1):
            pass
        else:
            raise ValueError(f'{name}: Must be either 0 or 1')

    def _check_res_settings(self, name, value):
        """Checks if view year or region settings are subsets of year or region

        Parameters
        ----------
        name : str
            attribute name
        value : _type_
            attribute value

        Raises
        ------
        TypeError
            Error
        """
        if self.residential:
            if hasattr(self, 'view_regions'):
                if not isinstance(self.view_regions, list):
                    raise ValueError(f'{name}: Must be a list')
                if not set(self.view_regions).issubset(self.regions):
                    raise ValueError('view_regions: Must be a subset of regions')
            if hasattr(self, 'view_years'):
                if not isinstance(self.view_years, list):
                    raise ValueError(f'{name}: Must be a list')
                if not set(self.view_years).issubset(self.years):
                    raise ValueError('view_years: Must be a subset of years')
