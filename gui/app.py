"""Streamlit interface for running BlueSky policy simulations.

The GUI assumes that core dependencies such as :mod:`pandas` are installed.
"""

from __future__ import annotations

import copy
import inspect
import itertools
import io
import importlib.util
import json
import logging
import re
import shutil
import subprocess
import sys
import os
import tempfile
import math
from datetime import date
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar
from uuid import uuid4

import pandas as pd
try:  # pragma: no cover - optional dependency wrapper
    import streamlit as st  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency wrapper
    st = None  # type: ignore[assignment]

from gui.engine_module import ensure_engine_package
from .session_state import CarbonSessionState

ensure_engine_package()

from engine.settings import input_root
from engine.data_loaders.ei_units import load_ei_units
from engine.normalization import (
    normalize_iso_name,
    normalize_region,
    normalize_region_id,
    normalize_token,
)
from engine.io.adapters import normalize_load_frame
from common.regions_schema import iter_region_records
from common.validators import validate_demand_table, DemandValidationError

try:  # pragma: no cover - optional dependency wrapper
    import yaml
except Exception:  # pragma: no cover - optional dependency wrapper
    yaml = None  # type: ignore[assignment]

from . import forecast_helpers as _forecast_helpers_module
from .demand_helpers import (
    _DEMAND_CURVE_BASE_DIR,
    _ALL_REGIONS_LABEL,
    _FrameProxy,
    _ScenarioSelection,
    _coerce_manifest_list,
    load_user_demand_csv,
    normalize_gui_demand_table,
    _build_demand_output_frame,
    _canonical_forecast_selection,
    _default_scenario_manifests,
    _demand_frame_from_manifests,
    _decode_forecast_selection,
    _discovered_region_names,
    _encode_forecast_selection,
    _iso_state_groups,
    _forecast_bundles_from_manifests,
    _forecast_bundles_from_selection,
    _manifests_from_selection,
    _normalize_state_codes,
    _normalize_state_forecast_selection,
    _states_from_config,
    _ordered_scenarios,
    _summarize_forecasts,
    _format_demand_curve_label,
    _load_demand_curve_catalog,
    select_forecast_bundles,
)

from .forecast_helpers import (
    _available_iso_scenarios_map,
    _cached_forecast_frame as _fh_cached_forecast_frame,
    _cached_input_root,
    _cached_iso_scenario_map,
    _call_loader_variants,
    _clear_forecast_cache,
    _coerce_base_path,
    _discover_bundle_records,
    discover_iso_scenarios,
    _iter_strings,
    _load_iso_scenario_frame,
    _regions_load_forecasts_frame,
    _regions_scenario_frame,
    _regions_scenario_index,
    _regions_zones_for,
    _resolve_forecast_base_path as _fh_resolve_forecast_base_path,
    _scenario_frame_subset,
    forecast_frame_error,
)


def _cached_forecast_frame(base_path: str | None) -> pd.DataFrame:
    """Return cached forecast frame, honoring patched region loaders."""

    loader = _regions_load_forecasts_frame
    original_loader = _forecast_helpers_module._regions_load_forecasts_frame
    if loader is original_loader:
        return _fh_cached_forecast_frame(base_path)

    wrapped = _fh_cached_forecast_frame.__wrapped__
    globals_dict = wrapped.__globals__
    original = globals_dict.get("_regions_load_forecasts_frame", original_loader)
    globals_dict["_regions_load_forecasts_frame"] = loader
    try:
        return wrapped(base_path)
    finally:
        globals_dict["_regions_load_forecasts_frame"] = original


_cached_forecast_frame.cache_clear = _fh_cached_forecast_frame.cache_clear  # type: ignore[attr-defined]
_cached_forecast_frame.cache_info = _fh_cached_forecast_frame.cache_info  # type: ignore[attr-defined]


def _resolve_forecast_base_path() -> str:
    """Return the current forecast base path honoring patched input roots."""

    helper_module = _forecast_helpers_module
    if input_root is helper_module.input_root:
        return _fh_resolve_forecast_base_path()

    original_input_root = helper_module.input_root
    helper_module.input_root = input_root
    try:
        return _fh_resolve_forecast_base_path()
    finally:
        helper_module.input_root = original_input_root


def available_regions() -> list[dict[str, str]]:
    """Return canonical region records for GUI consumers."""

    return list(iter_region_records())


REGION_HELPERS: dict[str, object] = {}


@contextmanager
def _sidebar_panel(container: Any, enabled: bool) -> Iterator[Any]:
    """Yield a sidebar-friendly container for grouping related controls.

    The GUI frequently renders form elements inside expanders or other
    containers and previously relied on a small wrapper that handled styling
    and conditional enablement.  Some call sites still expect that helper to be
    available, so we recreate it here to provide a consistent interface.  The
    ``enabled`` flag is preserved for compatibility with existing callers even
    though widget-level ``disabled`` attributes are used for fine-grained
    control of interactivity.
    """

    if container is None:
        # Defensive fallback for tests or unusual execution contexts where the
        # Streamlit container has not been created.  In that case, simply yield
        # the provided object so that downstream code can continue to operate
        # without raising an exception.
        yield container
        return

    panel_factory = getattr(container, "container", None)
    panel = container
    if callable(panel_factory):
        panel = panel_factory()

    try:
        yield panel
    finally:
        # No cleanup is required; the context manager primarily exists to
        # mirror the previous helper API and ensure deterministic closing of
        # any nested Streamlit containers.
        pass


_SCRIPT_ITERATION_KEY = "graniteledger_script_iteration"
_ACTIVE_RUN_ITERATION_KEY = "graniteledger_active_run_iteration"
_SESSION_RUN_TOKEN_KEY = "graniteledger_session_run_token"
_CURRENT_SESSION_RUN_TOKEN = f"graniteledger-{uuid4()}"


@lru_cache(maxsize=1)
def _dispatch_backend_available() -> bool:
    """Return whether at least one dispatch LP backend dependency is installed."""

    try:
        from dispatch.solvers import scipy_backend
    except ModuleNotFoundError:
        return False

    return scipy_backend.dependencies_available()


# -------------------------
# Optional imports / shims
# -------------------------
try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 fallback
    try:
        import tomli as tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Python 3.11+ or the tomli package is required to read TOML configuration files."
        ) from exc

try:
    from main.definitions import PROJECT_ROOT
except ModuleNotFoundError:  # fallback for packaged app execution
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from gui.module_settings import (
        CarbonModuleSettings,
        DemandModuleSettings,
        DispatchModuleSettings,
        GeneralConfigResult,
        IncentivesModuleSettings,
        OutputsModuleSettings,
    )
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    from module_settings import (  # type: ignore[import-not-found]
        CarbonModuleSettings,
        DemandModuleSettings,
        DispatchModuleSettings,
        GeneralConfigResult,
        IncentivesModuleSettings,
        OutputsModuleSettings,
    )

try:
    from gui.compute_intensity import (
        ComputeIntensitySettings,
        render as render_compute_intensity,
    )
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    from compute_intensity import (  # type: ignore[import-not-found]
        ComputeIntensitySettings,
        render as render_compute_intensity,
    )


_DEMAND_SETTINGS_FALLBACK_ERROR = (
    "Demand module returned an unexpected response; load forecasts have been disabled."
)


def _coerce_demand_module_settings(value: Any) -> DemandModuleSettings:
    """Return a :class:`DemandModuleSettings` instance for arbitrary inputs."""

    if isinstance(value, DemandModuleSettings):
        return value

    if isinstance(value, Mapping):
        try:
            curve_by_region = dict(value.get("curve_by_region", {}) or {})
            forecast_by_region = dict(value.get("forecast_by_region", {}) or {})
            load_forecasts = dict(value.get("load_forecasts", {}) or {})
            custom_load_forecasts = dict(value.get("custom_load_forecasts", {}) or {})
            errors = list(value.get("errors", []) or [])
            enabled = bool(value.get("enabled", False))
        except Exception:  # pragma: no cover - defensive conversion guard
            LOGGER.debug(
                "Unable to coerce demand module mapping into settings", exc_info=True
            )
        else:
            return DemandModuleSettings(
                enabled=enabled,
                curve_by_region=curve_by_region,
                forecast_by_region=forecast_by_region,
                load_forecasts=load_forecasts,
                custom_load_forecasts=custom_load_forecasts,
                errors=errors,
            )

    LOGGER.warning(
        "Demand module returned unexpected value type %s; defaulting to disabled load forecasts.",
        type(value).__name__,
    )

    errors = [_DEMAND_SETTINGS_FALLBACK_ERROR]
    return DemandModuleSettings(
        enabled=False,
        curve_by_region={},
        forecast_by_region={},
        load_forecasts={},
        custom_load_forecasts={},
        errors=errors,
    )

try:
    from gui import price_floor
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    import price_floor  # type: ignore[import-not-found]

try:
    from gui import helpers as gui_helpers
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    import helpers as gui_helpers  # type: ignore[import-not-found]

try:
    from gui.price_adapter import dataframe_to_carbon_vector
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    from price_adapter import dataframe_to_carbon_vector  # type: ignore[import-not-found]

# Region metadata helpers (robust to running as a script)
try:
    from gui.region_metadata import (
        DEFAULT_REGION_METADATA,
        canonical_region_label,
        canonical_region_value,
        region_alias_map,
        region_display_label,
    )
except ModuleNotFoundError:
    try:
        from region_metadata import (  # type: ignore[import-not-found]
            DEFAULT_REGION_METADATA,
            canonical_region_label,
            canonical_region_value,
            region_alias_map,
            region_display_label,
        )
    except ModuleNotFoundError:
        # Safe no-op fallbacks so the UI can still render
        DEFAULT_REGION_METADATA = {}

        def region_alias_map() -> dict[str, str]:  # type: ignore[return-type]
            return {}

        def canonical_region_label(x: object) -> str:
            return str(x)

        def canonical_region_value(x: object):
            return x

        def region_display_label(x: object) -> str:
            return str(x)


try:
    from gui.outputs_visualization import (
        filter_emissions_by_regions,
        load_emissions_data,
        region_selection_options,
        summarize_emissions_totals,
    )
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    from outputs_visualization import (  # type: ignore[import-not-found]
        filter_emissions_by_regions,
        load_emissions_data,
        region_selection_options,
        summarize_emissions_totals,
    )


try:
    from gui.recent_results import (
        get_recent_result,
        record_recent_result,
    )
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    from recent_results import (  # type: ignore[import-not-found]
        get_recent_result,
        record_recent_result,
    )


try:
    from gui.rggi import apply_rggi_defaults
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    try:
        from rggi import apply_rggi_defaults  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
        def apply_rggi_defaults(modules: dict[str, Any]) -> None:
            return None


def _set_state_if_changed(key: str, value: Any) -> None:
    """Update the Streamlit session state when values actually change."""

    if st is None:
        return

    previous = st.session_state.get(key)
    if previous != value:
        st.session_state[key] = value

try:  # pragma: no cover - optional dependency
    import altair as alt
except ImportError:  # pragma: no cover - graceful fallback when Altair missing
    alt = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from engine.run_loop import run_end_to_end_from_frames as _RUN_END_TO_END
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    try:  # pragma: no cover - optional dependency
        from engine.run_loop import run_end_to_end_from_frames as _RUN_END_TO_END
    except ModuleNotFoundError:
        _RUN_END_TO_END = None

try:  # pragma: no cover - optional dependency shim
    from engine.outputs import EngineOutputs as _EngineOutputs
except ModuleNotFoundError:  # pragma: no cover - optional dependency shim
    _EngineOutputs = None

try:
    from io_loader import Frames
    from engine.data_loaders import (
        load_edges,
        derive_fuels,
        load_unit_fleet,
        load_demand_forecasts_selection,
    )
    from engine.data_loaders.units import load_units
    from engine.manifest import build_manifest, manifest_to_markdown
    from engine.deepdoc import build_deep_doc, deep_doc_to_markdown

except (ModuleNotFoundError, ImportError):  # pragma: no cover - fallback when root not on sys.path
    sys.path.append(str(PROJECT_ROOT))
    from io_loader import Frames
    from engine.data_loaders import (
        load_edges,
        derive_fuels,
        load_unit_fleet,
        load_demand_forecasts_selection,
    )
    from engine.data_loaders.units import load_units
    from engine.manifest import build_manifest, manifest_to_markdown
    from engine.deepdoc import build_deep_doc, deep_doc_to_markdown

FramesType = Frames

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _regions_available_iso_scenarios(
    base_path: str | os.PathLike[str] | None = None,
) -> dict[str, list[str]]:
    """Return a mapping of ISO → scenarios for ``base_path``."""

    try:
        frame = _regions_load_forecasts_frame(base_path=base_path)
    except RuntimeError:
        return {}
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.debug("Unable to obtain available ISO scenarios", exc_info=True)
        return {}

    frame_obj = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    return _regions_scenario_index(frame_obj)


def _regions_available_zones(
    base_path: str | os.PathLike[str] | None,
    iso: str,
    scenario: str,
) -> list[str]:
    """Return available zones for ``iso``/``scenario`` discovered in cached data."""

    try:
        frame = _regions_load_forecasts_frame(base_path=base_path)
    except RuntimeError:
        frame = pd.DataFrame()
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.debug(
            "Unable to load cached forecast zones for %s/%s",
            iso,
            scenario,
            exc_info=True,
        )
        frame = pd.DataFrame()

    frame_obj = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)

    zones = _regions_zones_for(frame_obj, iso, scenario)
    if zones:
        return zones

    with _temporary_forecast_helper_overrides() as helper_module:
        return helper_module._discover_iso_zones(base_path, iso, scenario)


@contextmanager
def _temporary_forecast_helper_overrides():
    """Ensure helper module lookups respect GUI monkeypatches."""

    original_discover = _forecast_helpers_module._discover_bundle_records
    _forecast_helpers_module._discover_bundle_records = _discover_bundle_records
    try:
        yield _forecast_helpers_module
    finally:
        _forecast_helpers_module._discover_bundle_records = original_discover


def _set_session_state_if_changed(key: str, value: Any) -> None:
    if st is None:  # pragma: no cover - guard for non-UI usage
        return

    current = st.session_state.get(key)
    if isinstance(value, Mapping):
        changed = not isinstance(current, Mapping) or dict(current) != dict(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        changed = not isinstance(current, Sequence) or list(current) != list(value)
    else:
        changed = current != value

    if changed:
        st.session_state[key] = value
def _selected_forecast_bundles_from_state() -> Sequence[_ScenarioSelection] | None:
    if st is None:  # pragma: no cover - fallback for non-UI usage
        return None

    selection = st.session_state.get("forecast_selections")
    if not isinstance(selection, Mapping) or not selection:
        st.session_state.setdefault("selected_forecast_bundles", [])
        return None

    base_path = _resolve_forecast_base_path()
    frame = _cached_forecast_frame(base_path)

    try:
        manifests = _manifests_from_selection(
            selection,
            frame=frame,
            base_path=base_path,
        )
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.debug(
            "Unable to resolve forecast manifests from session state",
            extra={"manifest_selection": dict(selection)},
        )
        _set_session_state_if_changed("selected_forecast_bundles", [])
        return None

    if not manifests:
        _set_session_state_if_changed("selected_forecast_bundles", [])
        return None

    _set_session_state_if_changed("selected_forecast_bundles", manifests)
    return manifests



def _detect_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:  # pragma: no cover - Git metadata optional
        return None
    commit = result.stdout.strip()
    return commit or None


_GIT_COMMIT = _detect_git_commit()

try:  # pragma: no cover - optional dependency shim
    from src.common.utilities import get_downloads_directory as _get_downloads_directory
except ImportError:  # pragma: no cover - compatibility fallback
    _get_downloads_directory = None

# Tech metadata
try:
    from src.models.electricity.scripts.technology_metadata import (
        TECH_ID_TO_LABEL,
        get_technology_label,
        resolve_technology_key,
    )
except ModuleNotFoundError:
    TECH_ID_TO_LABEL = {}
    def get_technology_label(x: Any) -> str: return str(x)
    def resolve_technology_key(x: Any) -> int | None:
        try:
            return int(x)
        except Exception:
            return None

# -------------------------
# Constants
# -------------------------
STREAMLIT_REQUIRED_MESSAGE = (
    "streamlit is required to run the policy simulator UI. Install streamlit to continue."
)
ENGINE_RUNNER_REQUIRED_MESSAGE = (
    "engine.run_loop.run_end_to_end_from_frames is required to run the policy simulator UI."
)
DEEP_CARBON_UNSUPPORTED_MESSAGE = (
    "The installed simulation engine does not support the deep carbon pricing mode."
)
_SINGLE_REGION_COVERAGE_MESSAGE = (
    "Single-region dispatch requires uniform carbon coverage for the selected region. "
    "Update the carbon coverage selection so all assets share the same coverage status."
)


def _ensure_engine_runner():
    """Return the network runner callable used to solve the market model."""

    if _RUN_END_TO_END is None:
        raise ModuleNotFoundError(ENGINE_RUNNER_REQUIRED_MESSAGE)
    return _RUN_END_TO_END


def _runner_supports_keyword(runner: Any, keyword: str) -> bool:
    """Return ``True`` when ``runner`` accepts ``keyword``.

    The simulation engine's ``run_end_to_end_from_frames`` callable has evolved over
    time and may expose different keyword arguments depending on the installed
    engine version.  We defensively inspect the callable signature while being
    tolerant of objects that either hide their signature (for example, C
    extensions) or forward arbitrary keyword arguments via ``**kwargs``.
    """

    if not keyword:
        return False

    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):  # pragma: no cover - builtin or C-accelerated callables
        return True

    parameters = signature.parameters
    if keyword in parameters:
        return True

    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _ensure_streamlit() -> None:
    """Raise an informative error when the GUI stack is unavailable."""

    if st is None:
        raise ModuleNotFoundError(STREAMLIT_REQUIRED_MESSAGE)


from pathlib import Path

from engine.settings import input_root
from engine.normalization import (
    normalize_iso_name,
    normalize_region_id,
    normalize_token,
)
from regions.registry import REGIONS

DEFAULT_CONFIG_PATH = Path(PROJECT_ROOT, "src", "common", "run_config.toml")
_DEFAULT_LOAD_MWH = 1_000_000.0


def _load_config_data(source: Any | None = None) -> dict[str, Any]:
    """Return a configuration mapping loaded from ``source``.

    Parameters
    ----------
    source:
        The configuration payload to load. Supported inputs include:

        - ``None`` (falls back to :data:`DEFAULT_CONFIG_PATH`).
        - Any mapping (returned as a shallow ``dict`` copy).
        - ``bytes``/``bytearray``/``memoryview`` containing TOML text.
        - Path-like objects or strings pointing to a TOML file on disk.
        - File-like objects providing a ``read()`` method that yields ``str`` or
          ``bytes`` content.

    Returns
    -------
    dict[str, Any]
        Parsed configuration data.

    Raises
    ------
    TypeError
        If the ``source`` type is unsupported.
    FileNotFoundError
        If ``source`` is path-like but the file does not exist.
    ValueError
        If byte-oriented payloads are not UTF-8 decodable.
    """

    if source is None:
        source = DEFAULT_CONFIG_PATH

    if isinstance(source, Mapping):
        return dict(source)

    raw_bytes: bytes

    if isinstance(source, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(source)
    elif isinstance(source, (str, os.PathLike)):
        try:
            candidate = Path(source)
        except TypeError as exc:  # pragma: no cover - defensive guard
            raise TypeError(f"Unsupported configuration source type: {type(source)!r}") from exc

        if not candidate.exists():
            raise FileNotFoundError(f"Configuration file not found: {candidate}")
        if not candidate.is_file():
            raise FileNotFoundError(f"Configuration path is not a file: {candidate}")
        raw_bytes = candidate.read_bytes()
    elif hasattr(source, "read"):
        data = source.read()
        if isinstance(data, (bytes, bytearray, memoryview)):
            raw_bytes = bytes(data)
        elif isinstance(data, str):
            raw_bytes = data.encode("utf-8")
        else:  # pragma: no cover - defensive guard for exotic streams
            raw_bytes = str(data).encode("utf-8")
    else:
        raise TypeError(f"Unsupported configuration source type: {type(source)!r}")

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Configuration data must be UTF-8 encoded") from exc

    loaded = tomllib.loads(text)
    if isinstance(loaded, dict):
        return loaded
    return dict(loaded)


def _get_region_id_list() -> list[str]:
    """Return the canonical region identifiers from the registry."""

    region_ids = list(REGIONS.keys())
    if region_ids:
        return region_ids
    return ["system"]


def _get_default_region_id() -> str:
    """Return the default region identifier for the GUI."""

    return _get_region_id_list()[0]


def _build_region_lookup(region_ids: Iterable[str]) -> dict[str, str]:
    """Create a flexible lookup map for region aliases."""

    lookup: dict[str, str] = {}
    for region in region_ids:
        # direct lowercase mapping
        lookup.setdefault(str(region).lower(), str(region))

        # tokenized form
        token = normalize_token(region)
        if token:
            lookup.setdefault(token, str(region))

        # canonicalized region_id
        canonical = normalize_region_id(region)
        if canonical:
            lookup.setdefault(canonical.lower(), str(region))
            canonical_token = normalize_token(canonical)
            if canonical_token:
                lookup.setdefault(canonical_token, str(region))

            # allow zone-only shorthand (e.g., "NYUP" from "nyiso_nyup")
            if "_" in canonical:
                zone_part = canonical.split("_", 1)[-1]
                lookup.setdefault(zone_part.lower(), str(region))
                zone_token = normalize_token(zone_part)
                if zone_token:
                    lookup.setdefault(zone_token, str(region))

    return lookup


def _get_region_lookup() -> dict[str, str]:
    """Return the current region lookup mapping."""

    return _build_region_lookup(_get_region_id_list())

_LARGE_ALLOWANCE_SUPPLY = 1e12
_GENERAL_REGIONS_NORMALIZED_KEY = "general_regions_normalized_selection"
_ALL_REGIONS_LABEL = "All regions"
_T = TypeVar("_T")

_RUNNER_SIGNATURE: inspect.Signature | None = None
_RUNNER_ACCEPTS_VAR_KEYWORDS: bool | None = None
_RUNNER_SIGNATURE_CACHE_RUNNER: Callable[..., Any] | None = None
_RUNNER_KEYWORD_SUPPORT: dict[str, bool] = {}


SIDEBAR_SECTIONS: list[tuple[str, bool]] = [
    ("General config", False),
    ("Fuel cost helper", True),
    ("Demand curves", False),
    ("Carbon policy", False),
    ("Electricity dispatch", False),
    ("Incentives / credits", False),
    ("Outputs", False),
]

(
    OUTPUTS_SECTION_LABEL,
    OUTPUTS_SECTION_DEFAULT_EXPANDED,
) = SIDEBAR_SECTIONS[-1]

_GENERAL_PRESET_STATE_KEY = "general_config_active_preset"
_GENERAL_PRESET_WIDGET_KEY = "general_config_preset_option"

_GENERAL_CONFIG_PRESETS = [
    ("manual", "Manual configuration", None, False),
    ("rggi", "Eastern Interconnection – RGGI", apply_rggi_defaults, True),
]

_FUEL_PRICE_HELPER_STATE_KEY = "fuel_price_helper_selection"
_FUEL_PRICE_HELPER_ENABLED_KEY = "fuel_price_helper_enabled"


def _fuel_price_data_path(override: str | os.PathLike[str] | None = None) -> Path:
    """Return the resolved path to the fuel price catalog CSV."""

    if override is not None:
        candidate = Path(override)
        if candidate.is_dir():
            direct = candidate / "fuel_prices_annual.csv"
            nested = candidate / "fuel_prices" / "fuel_prices_annual.csv"
            if direct.exists():
                return direct
            return nested
        return candidate

    base_root = Path(_cached_input_root())
    try:
        input_root = base_root.parents[1]
    except IndexError:
        input_root = base_root.parent if base_root.parent != base_root else base_root

    catalog_root = input_root / "fuel_prices"
    direct = catalog_root / "fuel_prices_annual.csv"
    nested = catalog_root / "fuel_prices" / "fuel_prices_annual.csv"
    if direct.exists():
        return direct
    return nested


@lru_cache(maxsize=1)
def _load_fuel_price_catalog(path: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Load the fuel price catalog used for supply curve selections."""

    csv_path = _fuel_price_data_path(path)
    try:
        frame = pd.read_csv(csv_path)
    except FileNotFoundError:
        LOGGER.debug("Fuel price catalog not found at %s", csv_path)
        return pd.DataFrame(columns=["year", "region_id", "scenario_id", "fuel", "price_per_mmbtu"])
    except Exception:
        LOGGER.debug("Unable to read fuel price catalog from %s", csv_path, exc_info=True)
        return pd.DataFrame(columns=["year", "region_id", "scenario_id", "fuel", "price_per_mmbtu"])

    required = {"year", "region_id", "scenario_id", "fuel", "price_per_mmbtu"}
    missing = required - set(frame.columns)
    if missing:
        LOGGER.warning(
            "Fuel price catalog missing expected columns: %s",
            ", ".join(sorted(missing)),
        )
        return pd.DataFrame(columns=list(required))

    working = frame.copy()
    working["fuel"] = working["fuel"].astype(str).str.strip().str.upper()
    working["scenario_id"] = working["scenario_id"].astype(str).str.strip()
    working["region_id"] = working["region_id"].astype(str).str.strip()
    working["year"] = pd.to_numeric(working["year"], errors="coerce")
    working["price_per_mmbtu"] = pd.to_numeric(working["price_per_mmbtu"], errors="coerce")
    working = working.dropna(subset=["fuel", "scenario_id", "year", "price_per_mmbtu"])
    if working.empty:
        return pd.DataFrame(columns=list(required))

    working["year"] = working["year"].astype(int)
    return working


def _fuel_price_catalog_by_fuel(
    frame: pd.DataFrame, years: Iterable[int] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Return a mapping of fuel → scenario metadata for the helper UI."""

    if frame.empty:
        return {}

    if years:
        try:
            year_set = {int(year) for year in years}
        except Exception:
            year_set = set()
        if year_set:
            subset = frame[frame["year"].isin(year_set)]
            if not subset.empty:
                frame = subset

    catalog: dict[str, list[dict[str, Any]]] = {}
    for fuel, fuel_frame in frame.groupby("fuel"):
        entries: list[dict[str, Any]] = []
        for scenario_id, scenario_frame in fuel_frame.groupby("scenario_id"):
            prices = scenario_frame["price_per_mmbtu"].dropna().astype(float)
            if prices.empty:
                avg_price = math.nan
                min_price = math.nan
                max_price = math.nan
            else:
                avg_price = float(prices.mean())
                min_price = float(prices.min())
                max_price = float(prices.max())
            years_list = sorted(
                {
                    int(year)
                    for year in pd.to_numeric(scenario_frame["year"], errors="coerce")
                    if not pd.isna(year)
                }
            )
            entries.append(
                {
                    "scenario_id": str(scenario_id),
                    "average_price": avg_price,
                    "min_price": min_price,
                    "max_price": max_price,
                    "years": years_list,
                }
            )
        if entries:
            entries.sort(key=lambda item: item["scenario_id"])
            catalog[str(fuel)] = entries

    return catalog


def _format_fuel_price_option_label(option: Mapping[str, Any]) -> str:
    """Build a human-readable label for a fuel price scenario option."""

    scenario = str(option.get("scenario_id", "")).strip()
    avg_price = option.get("average_price")
    if isinstance(avg_price, (int, float)) and not math.isnan(avg_price):
        return f"{scenario} – avg ${avg_price:.2f}/MMBtu"
    return scenario or "Unnamed scenario"


def _normalize_fuel_price_selection(
    selection: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Return a normalized mapping of fuel → scenario identifiers."""

    normalized: dict[str, str] = {}
    if not isinstance(selection, Mapping):
        return normalized

    for fuel, scenario in selection.items():
        if scenario in (None, ""):
            continue
        fuel_key = str(fuel).strip().upper()
        scenario_id = str(scenario).strip()
        if fuel_key and scenario_id:
            normalized[fuel_key] = scenario_id

    return normalized


def _render_fuel_price_helper_controls(
    container: Any,
    *,
    years: Iterable[int] | None,
    existing_selection: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Render the fuel price helper UI and return the selected scenarios."""

    base_selection = _normalize_fuel_price_selection(existing_selection)
    if st is None:
        return base_selection

    container.caption(
        "Assign supply curve scenarios to fuels. These selections populate fuel prices when running the model."
    )

    catalog_frame = _load_fuel_price_catalog()
    if catalog_frame.empty:
        data_path = _fuel_price_data_path()
        container.info(
            f"Fuel price catalog not found at `{data_path.as_posix()}`. Upload a catalog to enable selections."
        )
        return base_selection

    session_selection_raw = st.session_state.get(_FUEL_PRICE_HELPER_STATE_KEY)
    session_selection = _normalize_fuel_price_selection(session_selection_raw)
    if session_selection:
        base_selection.update(session_selection)

    catalog = _fuel_price_catalog_by_fuel(catalog_frame, years=years)
    if not catalog:
        container.info("Fuel price catalog does not contain any supply curve scenarios.")
        return base_selection

    selection: dict[str, str] = dict(base_selection)
    summary_rows: list[dict[str, Any]] = []

    for fuel in sorted(catalog):
        options = catalog.get(fuel, [])
        if not options:
            continue
        option_pairs = [(_format_fuel_price_option_label(entry), entry) for entry in options]
        labels = [label for label, _ in option_pairs]
        default_id = selection.get(fuel)
        default_index = 0
        if default_id:
            for idx, entry in enumerate(options):
                if entry.get("scenario_id") == default_id:
                    default_index = idx
                    break

        selected_label = container.selectbox(
            f"{fuel.replace('_', ' ').title()} supply curve",
            options=labels,
            index=default_index if 0 <= default_index < len(labels) else 0,
            key=f"fuel_price_helper_{fuel.lower()}",
        )
        selected_entry = next(entry for label, entry in option_pairs if label == selected_label)
        selection[fuel] = str(selected_entry.get("scenario_id", ""))

        summary_rows.append(
            {
                "Fuel": fuel.replace("_", " ").title(),
                "Scenario": selected_entry.get("scenario_id"),
                "Average $/MMBtu": selected_entry.get("average_price"),
                "Min $/MMBtu": selected_entry.get("min_price"),
                "Max $/MMBtu": selected_entry.get("max_price"),
            }
        )

    st.session_state[_FUEL_PRICE_HELPER_STATE_KEY] = dict(selection)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        container.dataframe(summary_df, use_container_width=True)

    data_path = _fuel_price_data_path()
    container.caption(f"Fuel price data source: `{data_path.as_posix()}`.")

    return selection

SIDEBAR_STYLE = """
<style>
.sidebar-module {
    border: 1px solid var(--secondary-background-color);
    border-radius: 0.5rem;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.75rem;
}
.sidebar-module.disabled {
    opacity: 0.5;
}
</style>
"""

_download_directory_fallback_used = False


def _fallback_downloads_directory(app_subdir: str = "GraniteLedger") -> Path:
    """Return a reasonable downloads location when utilities helper is unavailable."""

    base_path = Path.home() / "Downloads"
    if app_subdir:
        base_path = base_path / app_subdir
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def get_downloads_directory(app_subdir: str = "GraniteLedger") -> Path:
    """Resolve the downloads directory, falling back to the user's home folder."""

    global _download_directory_fallback_used

    if _get_downloads_directory is not None:
        try:
            return _get_downloads_directory(app_subdir=app_subdir)
        except Exception:  # pragma: no cover - defensive: ensure GUI still loads
            LOGGER.warning(
                "Falling back to home Downloads directory; helper raised an error."
            )
    if not _download_directory_fallback_used:
        LOGGER.warning(
            "get_downloads_directory is unavailable; using ~/Downloads for model outputs."
        )
        _download_directory_fallback_used = True
    return _fallback_downloads_directory(app_subdir)

if "GeneralConfigResult" not in globals():

    @dataclass
    class GeneralConfigResult:
        """Container for user-selected general configuration settings."""

        config_label: str
        config_source: Any
        run_config: dict[str, Any]
        candidate_years: list[int]
        start_year: int
        end_year: int
        selected_years: list[int]
        regions: list[int | str]
        preset_key: str | None = None
        preset_label: str | None = None
        lock_carbon_controls: bool = False
        max_iterations: int = 0
        fuel_price_selection: dict[str, str] = field(default_factory=dict)


if "CarbonModuleSettings" not in globals():

    @dataclass
    class CarbonModuleSettings:
        """Record of carbon policy sidebar selections."""

        enabled: bool
        price_enabled: bool
        enable_floor: bool
        enable_ccr: bool
        ccr1_enabled: bool
        ccr2_enabled: bool
        ccr1_price: float | None
        ccr2_price: float | None
        ccr1_escalator_pct: float
        ccr2_escalator_pct: float
        banking_enabled: bool
        coverage_regions: list[str]
        control_period_years: int | None
        price_per_ton: float
        price_escalator_pct: float = 0.0
        initial_bank: float = 0.0
        cap_regions: list[Any] = field(default_factory=list)
        cap_start_value: float | None = None
        cap_reduction_mode: str = "percent"
        cap_reduction_value: float = 0.0
        cap_schedule: dict[int, float] = field(default_factory=dict)
        floor_value: float = 0.0
        floor_escalator_mode: str = "fixed"
        floor_escalator_value: float = 0.0
        floor_schedule: dict[int, float] = field(default_factory=dict)
        price_schedule: dict[int, float] = field(default_factory=dict)
        errors: list[str] = field(default_factory=list)


@dataclass
class CarbonPolicyConfig:
    """Normalized carbon allowance policy configuration for engine runs."""

    enabled: bool = True
    enable_floor: bool = True
    enable_ccr: bool = True
    ccr1_enabled: bool = True
    ccr2_enabled: bool = True
    ccr1_price: float | None = None
    ccr2_price: float | None = None
    ccr1_escalator_pct: float = 0.0
    ccr2_escalator_pct: float = 0.0
    allowance_banking_enabled: bool = True
    control_period_years: int | None = None
    floor_value: float = 0.0
    floor_escalator_mode: str = "fixed"
    floor_escalator_value: float = 0.0

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None,
        *,
        enabled: bool | None = None,
        enable_floor: bool | None = None,
        enable_ccr: bool | None = None,
        ccr1_enabled: bool | None = None,
        ccr2_enabled: bool | None = None,
        allowance_banking_enabled: bool | None = None,
        control_period_years: int | None = None,
        floor_value: float | None = None,
        floor_escalator_mode: str | None = None,
        floor_escalator_value: float | None = None,
        ccr1_price: float | None = None,
        ccr2_price: float | None = None,
        ccr1_escalator_pct: float | None = None,
        ccr2_escalator_pct: float | None = None,
    ) -> "CarbonPolicyConfig":
        record = dict(mapping) if isinstance(mapping, Mapping) else {}

        def _coerce_bool(value: Any, default: bool) -> bool:
            return bool(value) if value is not None else default

        enabled_val = _coerce_bool(enabled, bool(record.get('enabled', True)))
        enable_floor_val = _coerce_bool(enable_floor, bool(record.get('enable_floor', True)))
        enable_ccr_val = _coerce_bool(enable_ccr, bool(record.get('enable_ccr', True)))
        ccr1_val = _coerce_bool(ccr1_enabled, bool(record.get('ccr1_enabled', True)))
        ccr2_val = _coerce_bool(ccr2_enabled, bool(record.get('ccr2_enabled', True)))
        banking_val = _coerce_bool(
            allowance_banking_enabled,
            bool(record.get('allowance_banking_enabled', True)),
        )

        control_period_val = _sanitize_control_period(
            control_period_years
            if control_period_years is not None
            else record.get('control_period_years')
        )

        floor_value_raw = floor_value if floor_value is not None else record.get("floor_value")
        floor_value_val = price_floor.parse_currency_value(floor_value_raw, 0.0)
        floor_mode_raw = (
            floor_escalator_mode if floor_escalator_mode is not None else record.get("floor_escalator_mode")
        )
        floor_mode_val = str(floor_mode_raw or "fixed").strip().lower()
        if floor_mode_val not in {"fixed", "percent"}:
            floor_mode_val = "fixed"
        floor_escalator_raw = (
            floor_escalator_value if floor_escalator_value is not None else record.get("floor_escalator_value")
        )
        if floor_mode_val == "percent":
            floor_escalator_val = price_floor.parse_percentage_value(floor_escalator_raw, 0.0)
        else:
            floor_escalator_val = price_floor.parse_currency_value(floor_escalator_raw, 0.0)
        def _coerce_optional(value: Any, fallback: float | None) -> float | None:
            if value in (None, ""):
                return fallback
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback

        ccr1_price_val = _coerce_optional(
            ccr1_price if ccr1_price is not None else record.get('ccr1_price'),
            None,
        )
        ccr2_price_val = _coerce_optional(
            ccr2_price if ccr2_price is not None else record.get('ccr2_price'),
            None,
        )
        ccr1_escalator_val = _coerce_optional(
            ccr1_escalator_pct
            if ccr1_escalator_pct is not None
            else record.get('ccr1_escalator_pct'),
            0.0,
        ) or 0.0
        ccr2_escalator_val = _coerce_optional(
            ccr2_escalator_pct
            if ccr2_escalator_pct is not None
            else record.get('ccr2_escalator_pct'),
            0.0,
        ) or 0.0

        config = cls(
            enabled=enabled_val,
            enable_floor=enable_floor_val,
            enable_ccr=enable_ccr_val,
            ccr1_enabled=ccr1_val,
            ccr2_enabled=ccr2_val,
            ccr1_price=ccr1_price_val,
            ccr2_price=ccr2_price_val,
            ccr1_escalator_pct=float(ccr1_escalator_val),
            ccr2_escalator_pct=float(ccr2_escalator_val),
            allowance_banking_enabled=banking_val,
            control_period_years=control_period_val,
            floor_value=float(floor_value_val),
            floor_escalator_mode=str(floor_mode_val),
            floor_escalator_value=float(floor_escalator_val),
        )

        if not config.enabled:
            config.disable_cap()
        elif not config.enable_ccr:
            config.ccr1_enabled = False
            config.ccr2_enabled = False

        return config

    def disable_cap(self) -> None:
        """Disable allowance trading mechanics when the cap is inactive."""

        self.enabled = False
        self.enable_floor = False
        self.enable_ccr = False
        self.ccr1_enabled = False
        self.ccr2_enabled = False
        self.ccr1_price = None
        self.ccr2_price = None
        self.ccr1_escalator_pct = 0.0
        self.ccr2_escalator_pct = 0.0
        self.allowance_banking_enabled = False
        self.control_period_years = None

    def disable_for_price(self) -> None:
        """Disable the cap when an exogenous carbon price is active."""

        self.disable_cap()

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary representation."""

        return {
            'enabled': bool(self.enabled),
            'enable_floor': bool(self.enable_floor),
            'enable_ccr': bool(self.enable_ccr),
            'ccr1_enabled': bool(self.ccr1_enabled),
            'ccr2_enabled': bool(self.ccr2_enabled),
            'ccr1_price': self.ccr1_price,
            'ccr2_price': self.ccr2_price,
            'ccr1_escalator_pct': float(self.ccr1_escalator_pct),
            'ccr2_escalator_pct': float(self.ccr2_escalator_pct),
            'allowance_banking_enabled': bool(self.allowance_banking_enabled),
            'control_period_years': self.control_period_years,
        }


@dataclass
class RunProgressState:
    """State container for tracking and rendering run progress."""

    stage: str = "idle"
    message: str = ""
    percent_complete: int = 0
    total_years: int = 1
    years: list[Any] = field(default_factory=list)
    current_index: int = -1
    current_year: Any | None = None
    iterations_expected: int = 0
    iteration_targets: dict[Any, int] = field(default_factory=dict)
    iteration_tolerance: float | None = None
    log: list[str] = field(default_factory=list)

    def reset(self) -> None:
        """Return the tracker to an initial idle state."""

        self.stage = "idle"
        self.message = ""
        self.percent_complete = 0
        self.total_years = 1
        self.years.clear()
        self.current_index = -1
        self.current_year = None
        self.iterations_expected = 0
        self.iteration_targets.clear()
        self.iteration_tolerance = None
        self.log.clear()


def _ensure_progress_state() -> RunProgressState:
    """Return the progress tracker stored in the current Streamlit session."""

    if st is None:
        raise ModuleNotFoundError(STREAMLIT_REQUIRED_MESSAGE)

    state = st.session_state.get("_run_progress_state")
    if isinstance(state, RunProgressState):
        return state

    tracker = RunProgressState()
    st.session_state["_run_progress_state"] = tracker
    return tracker


def _reset_progress_state() -> RunProgressState:
    """Reset the session progress tracker and return it."""

    tracker = _ensure_progress_state()
    tracker.reset()
    return tracker


def _staged_run_inputs_state() -> dict[str, Any]:
    """Return the mutable mapping used to stage run inputs in session state."""

    if st is None:
        return {}

    staged = st.session_state.get("_staged_run_inputs")
    if isinstance(staged, dict):
        return staged

    staged = {}
    st.session_state["_staged_run_inputs"] = staged
    return staged


def _update_staged_run_inputs(**kwargs: Any) -> None:
    """Persist the provided run inputs in Streamlit session state."""

    staged = _staged_run_inputs_state()
    staged.update({key: value for key, value in kwargs.items()})


def _apply_region_weights_to_frames(
    frames_obj: FramesType | None, region_weights: Mapping[str, float] | None
) -> FramesType | None:
    """Attach ``region_weights`` metadata to ``frames_obj`` when available."""

    if frames_obj is None or not region_weights:
        return frames_obj

    try:
        meta = getattr(frames_obj, "_meta", {})
    except AttributeError:
        return frames_obj

    sanitized: dict[str, float] = {}
    for region, weight in region_weights.items():
        if region in (None, ""):
            continue
        try:
            numeric = float(weight)
        except (TypeError, ValueError):
            continue
        region_key = str(region).strip()
        if not region_key:
            continue
        sanitized[region_key] = numeric

    if not sanitized:
        return frames_obj

    updated_meta = dict(meta) if isinstance(meta, Mapping) else {}
    updated_meta["region_weights"] = sanitized
    setattr(frames_obj, "_meta", updated_meta)
    return frames_obj


def _trigger_streamlit_rerun() -> bool:
    """Request that Streamlit immediately rerun the script."""

    if st is None:
        return False

    for attr in ("rerun", "experimental_rerun"):
        rerun_fn = getattr(st, attr, None)
        if callable(rerun_fn):
            rerun_fn()
            return True

    return False

def _bounded_percent(value: float | int) -> int:
    """Clamp a numeric percent to the inclusive range [0, 100]."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, int(round(numeric))))


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


try:  # pragma: no cover - import best-effort
    from engine import compute_intensity as _compute_intensity
    from engine.cap_mode import PRICE_TOL as _CAP_PRICE_TOL
except Exception:  # pragma: no cover - fallback when engine unavailable
    _compute_intensity = None
    _CAP_PRICE_TOL = 1e-3


@dataclass(frozen=True)
class _StageInfo:
    label: str
    target_percent: int


_PROGRESS_PREP_STAGES: dict[str, _StageInfo] = {
    "compiling_assumptions": _StageInfo("Compiling assumptions…", 10),
    "loading_load_forecasts": _StageInfo("Loading load forecasts…", 20),
    "initializing_fleet": _StageInfo("Initializing fleet…", 35),
    "building_interfaces": _StageInfo("Building interfaces…", 45),
}

_SIMULATION_BASE_PERCENT = max(info.target_percent for info in _PROGRESS_PREP_STAGES.values())
_SIMULATION_PROGRESS_RANGE = max(0, 100 - _SIMULATION_BASE_PERCENT)
_PROGRESS_LOG_MAX_ENTRIES = 12
_ITERATION_RENDER_STRIDE = 5
_DEFAULT_MAX_ITERATIONS = 60
if _compute_intensity is not None:  # pragma: no branch - simple guard
    try:
        _DEFAULT_MAX_ITERATIONS = int(
            _compute_intensity.get_effective_iteration_limit()
        )
    except Exception:
        _DEFAULT_MAX_ITERATIONS = 60
_DEFAULT_TOLERANCE = float(_CAP_PRICE_TOL or 1e-3)


@dataclass
class _ProgressDisplay:
    """Helper that routes engine progress updates to Streamlit widgets."""

    status: Any
    progress_bar: Any
    log_container: Any
    state: RunProgressState
    _last_log_key: str | None = field(default=None, init=False, repr=False)
    _last_rendered_iteration: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )

    def _update_progress(
        self,
        label: str,
        *,
        percent: float | int | None = None,
        stage: str | None = None,
        status_state: str = "running",
    ) -> None:
        if stage is not None:
            self.state.stage = stage
        self.state.message = label
        if percent is not None:
            bounded = _bounded_percent(percent)
            self.state.percent_complete = bounded
            self.progress_bar.progress(bounded)
        else:
            self.progress_bar.progress(_bounded_percent(self.state.percent_complete))
        self.status.update(label=label, state=status_state)

    def _append_log(
        self,
        message: str,
        *,
        key: str | None = None,
        coalesce: bool = False,
        render: bool = True,
    ) -> None:
        if not message:
            return
        if coalesce and key and self.state.log and self._last_log_key == key:
            self.state.log[-1] = message
        else:
            self.state.log.append(message)
            if len(self.state.log) > _PROGRESS_LOG_MAX_ENTRIES:
                del self.state.log[: -_PROGRESS_LOG_MAX_ENTRIES]
        self._last_log_key = key if coalesce else None
        if not render:
            return
        if self.log_container is not None:
            if not self.state.log:
                self.log_container.caption("Awaiting engine updates…")
            else:
                lines = "\n".join(
                    f"* {entry}" for entry in self.state.log[-_PROGRESS_LOG_MAX_ENTRIES :]
                )
                self.log_container.markdown(lines)

    def handle_stage(self, stage: str, payload: Mapping[str, object]) -> None:
        try:
            self._handle_stage(stage, payload)
        except Exception:  # pragma: no cover - defensive
            LOGGER.exception("Unable to interpret stage update for %s", stage)

    def _handle_stage(self, stage: str, payload: Mapping[str, object]) -> None:
        stage = str(stage or "").strip()
        if not stage:
            return

        if stage in _PROGRESS_PREP_STAGES:
            info = _PROGRESS_PREP_STAGES[stage]
            percent = max(self.state.percent_complete, info.target_percent)
            self._append_log(info.label, key=stage, coalesce=True)
            self._update_progress(info.label, percent=percent, stage=stage)
            return

        if stage == "run_start":
            total = _safe_int(payload.get("total_years"), 0)
            years_payload = payload.get("years")
            if isinstance(years_payload, Iterable) and not isinstance(years_payload, (str, bytes)):
                years_list = [entry for entry in years_payload]
                if years_list:
                    self.state.years = list(years_list)
                    total = max(total, len(years_list))
            if total <= 0:
                total = 1
            self.state.total_years = total
            self.state.current_index = -1
            self.state.current_year = None
            max_iter = _safe_int(payload.get("max_iter") or payload.get("max_iterations"), 0)
            if max_iter > 0:
                self.state.iterations_expected = max_iter
            elif self.state.iterations_expected <= 0:
                self.state.iterations_expected = _DEFAULT_MAX_ITERATIONS
            tolerance = _safe_float(payload.get("tolerance"))
            if tolerance is not None:
                self.state.iteration_tolerance = tolerance
            elif self.state.iteration_tolerance is None:
                self.state.iteration_tolerance = _DEFAULT_TOLERANCE
            label = f"Preparing simulation for {total} year(s)…"
            percent = max(self.state.percent_complete, _SIMULATION_BASE_PERCENT)
            self._append_log(label, key=stage, coalesce=True)
            self._update_progress(label, percent=percent, stage=stage)
            return

        if stage == "year_start":
            index = _safe_int(payload.get("index"), self.state.current_index)
            total_years = max(self.state.total_years, 1)
            self.state.current_index = max(index, 0)
            year_val = payload.get("year")
            if year_val is not None:
                self.state.current_year = year_val
            base_fraction = max(0.0, min(1.0, self.state.current_index / total_years))
            percent = _SIMULATION_BASE_PERCENT + base_fraction * _SIMULATION_PROGRESS_RANGE
            percent = max(self.state.percent_complete, percent)
            year_display = str(self.state.current_year if self.state.current_year is not None else index + 1)
            label = f"Simulating year {year_display} ({self.state.current_index + 1}/{total_years})"
            key = f"{stage}:{self.state.current_index}:{year_display}"
            self._append_log(label, key=key, coalesce=True)
            self._update_progress(label, percent=percent, stage=stage)
            return

        if stage == "year_complete":
            index = _safe_int(payload.get("index"), self.state.current_index)
            total_years = max(self.state.total_years, 1)
            self.state.current_index = max(index, 0)
            year_val = payload.get("year", self.state.current_year)
            if year_val is not None:
                self.state.current_year = year_val
            iteration_count = _safe_int(payload.get("iterations"), 0)
            if self.state.current_year is not None and iteration_count > 0:
                self.state.iteration_targets[self.state.current_year] = iteration_count
            price_val = _safe_float(payload.get("price"))
            fraction = max(0.0, min(1.0, (self.state.current_index + 1) / total_years))
            percent = _SIMULATION_BASE_PERCENT + fraction * _SIMULATION_PROGRESS_RANGE
            year_display = str(self.state.current_year if self.state.current_year is not None else index + 1)
            message_parts = [f"Completed year {year_display} of {total_years}"]
            if price_val is not None:
                message_parts.append(f"price {price_val:,.2f}")
            if iteration_count > 0:
                message_parts.append(f"{iteration_count} iteration(s)")
            label = message_parts[0]
            if len(message_parts) > 1:
                label += " (" + ", ".join(message_parts[1:]) + ")"
            key = f"{stage}:{self.state.current_index}:{year_display}"
            self._append_log(label, key=key, coalesce=True)
            self._update_progress(label, percent=percent, stage=stage)
            return

        if stage == "run_failed":
            error_message = str(
                payload.get("message")
                or payload.get("error")
                or "Simulation failed"
            ).strip()
            if error_message and not error_message.lower().startswith("simulation failed"):
                label = f"Simulation failed: {error_message}"
            elif error_message:
                label = error_message
            else:
                label = "Simulation failed"
            bounded = _bounded_percent(self.state.percent_complete or 0)
            self._append_log(label, key=stage, coalesce=True)
            self._update_progress(
                label,
                percent=bounded,
                stage="run_failed",
                status_state="error",
            )
            return

        if stage == "run_complete":
            label = "Simulation complete. Outputs updated below."
            self._append_log(label, key=stage, coalesce=True)
            self._update_progress(label, percent=100, stage="complete", status_state="complete")
            return

        # Fallback for unrecognized stages
        label = f"{stage.replace('_', ' ').title()}…"
        self._append_log(label, key=stage, coalesce=True)
        self._update_progress(label, stage=stage)

    def handle_iteration(self, stage: str, payload: Mapping[str, object]) -> None:
        if str(stage).strip().lower() != "iteration":
            self.handle_stage(stage, payload)
            return

        try:
            iteration = max(1, _safe_int(payload.get("iteration"), 0))
            year_val = payload.get("year", self.state.current_year)
            if year_val is not None:
                self.state.current_year = year_val
            year_display = str(self.state.current_year if self.state.current_year is not None else "?")
            max_iter = _safe_int(payload.get("max_iter"), 0)
            if max_iter <= 0 and self.state.current_year in self.state.iteration_targets:
                max_iter = self.state.iteration_targets.get(self.state.current_year, 0)
            if max_iter <= 0:
                max_iter = self.state.iterations_expected or _DEFAULT_MAX_ITERATIONS
            self.state.iterations_expected = max(self.state.iterations_expected, max_iter)
            if self.state.current_year is not None:
                self.state.iteration_targets[self.state.current_year] = max_iter
            tolerance = _safe_float(payload.get("tolerance"))
            if tolerance is not None:
                self.state.iteration_tolerance = tolerance
            elif self.state.iteration_tolerance is None:
                self.state.iteration_tolerance = _DEFAULT_TOLERANCE
            price_val = _safe_float(payload.get("price"))
            converged_flag = bool(payload.get("converged"))

            if max_iter > 0:
                iter_text = f"{iteration}/{max_iter}"
                if converged_flag:
                    iteration_fraction = 1.0
                else:
                    iteration_fraction = min(max(iteration / max_iter, 0.0), 1.0)
            else:
                iter_text = f"{iteration} (unknown max)"
                iteration_fraction = 1.0 if converged_flag else 0.0

            total_years = max(self.state.total_years, 1)
            base_index = max(self.state.current_index, 0)
            percent = _SIMULATION_BASE_PERCENT + (
                (base_index + iteration_fraction) / total_years
            ) * _SIMULATION_PROGRESS_RANGE
            percent = max(self.state.percent_complete, percent)

            detail_parts: list[str] = []
            tol_value = self.state.iteration_tolerance
            if tol_value is not None:
                detail_parts.append(f"tol ≤ {tol_value:,.3g}")
            if price_val is not None:
                detail_parts.append(f"price ≈ {price_val:,.2f}")
            detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
            label = f"Year {year_display}: iteration {iter_text}{detail}"

            iteration_key = f"iteration:{year_display}"
            last_rendered = self._last_rendered_iteration.get(iteration_key)
            stride = max(1, int(_ITERATION_RENDER_STRIDE))
            should_render = (
                converged_flag
                or iteration == 1
                or (max_iter > 0 and iteration >= max_iter)
                or (stride > 0 and iteration % stride == 0)
            )
            if should_render and last_rendered == iteration:
                should_render = False
            self._append_log(
                label,
                key=iteration_key,
                coalesce=True,
                render=should_render,
            )
            if should_render:
                self._last_rendered_iteration[iteration_key] = iteration

            self._update_progress(label, percent=percent, stage="iteration")
        except Exception:  # pragma: no cover - defensive
            LOGGER.exception("Unable to interpret iteration update")

    def complete(self) -> None:
        label = "Simulation complete. Outputs updated below."
        self._append_log(label, key="run_complete", coalesce=True)
        self._update_progress(label, percent=100, stage="complete", status_state="complete")

    def fail(self, message: str) -> None:
        safe_message = message or "Simulation failed"
        bounded = _bounded_percent(self.state.percent_complete or 0)
        self._append_log(safe_message, key="error", coalesce=True)
        self._update_progress(safe_message, percent=bounded, stage="error", status_state="error")


@dataclass
class CarbonPriceConfig:
    """Normalized carbon price configuration for engine runs."""

    enabled: bool = False
    price_per_ton: float = 0.0
    escalator_pct: float = 0.0
    schedule: dict[int, float] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        """Return ``True`` when the price should override the cap."""

        return bool(self.enabled and self.schedule)

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None,
        *,
        enabled: bool | None = None,
        value: float | None = None,
        schedule: Mapping[int, float] | Mapping[str, Any] | None = None,
        years: Iterable[int] | None = None,
        escalator_pct: float | None = None,
    ) -> "CarbonPriceConfig":
        record = dict(mapping) if isinstance(mapping, Mapping) else {}

        enabled_val = bool(enabled) if enabled is not None else bool(record.get('enabled', False))
        price_raw = value if value is not None else record.get('price_per_ton', record.get('price', 0.0))
        price_value = _coerce_float(price_raw, default=0.0)
        escalator_raw = (
            escalator_pct
            if escalator_pct is not None
            else record.get('price_escalator_pct', record.get('escalator_pct', 0.0))
        )
        escalator_value = _coerce_float(escalator_raw, default=0.0)

        schedule_map = _merge_price_schedules(
            record.get('price_schedule'),
            schedule,
        )

        if not schedule_map and years:
            normalized_years: list[int] = []
            for year in years:
                try:
                    normalized_years.append(int(year))
                except (TypeError, ValueError):
                    continue
            if normalized_years:
                generated_schedule = _build_price_escalator_schedule(
                    price_value,
                    escalator_value,
                    sorted(set(normalized_years)),
                )
                if generated_schedule:
                    schedule_map = generated_schedule

        if schedule_map:
            schedule_map = dict(sorted(schedule_map.items()))

        config = cls(
            enabled=bool(enabled_val),
            price_per_ton=float(price_value),
            escalator_pct=float(escalator_value),
            schedule=schedule_map,
        )

        if not config.active:
            config.schedule = {}
            config.price_per_ton = 0.0
            config.escalator_pct = 0.0

        return config

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable dictionary representation."""

        payload = {
            'enabled': bool(self.enabled),
            'price_per_ton': float(self.price_per_ton),
            'price_escalator_pct': float(self.escalator_pct),
        }
        if self.schedule:
            payload['price_schedule'] = dict(self.schedule)
        return payload


def _sanitize_control_period(value: Any) -> int | None:
    """Return ``value`` coerced to a positive integer when possible."""

    if value in (None, ''):
        return None
    try:
        period = int(value)
    except (TypeError, ValueError):
        return None
    return period if period > 0 else None


def _normalize_price_schedule(value: Any) -> dict[int, float]:
    """Return a normalized mapping of year to carbon price."""

    if not isinstance(value, Mapping):
        return {}

    entries: list[tuple[int, float]] = []
    for key, raw in value.items():
        if raw in (None, ""):
            continue
        try:
            year = int(key)
        except (TypeError, ValueError):
            continue
        try:
            price = float(raw)
        except (TypeError, ValueError):
            continue
        entries.append((year, price))

    if not entries:
        return {}

    entries.sort(key=lambda item: item[0])
    return {year: price for year, price in entries}


def _merge_price_schedules(
    *values: Mapping[int, float] | Mapping[str, Any] | None,
) -> dict[int, float]:
    """Combine candidate schedules, returning a sorted ``{year: price}`` mapping."""

    merged: dict[int, float] = {}
    for candidate in values:
        if not isinstance(candidate, Mapping):
            continue
        merged.update(_normalize_price_schedule(candidate))

    if not merged:
        return {}

    return dict(sorted(merged.items()))


def _expand_or_build_price_schedule(
    schedule: Mapping[int, float] | None,
    years: Iterable[int] | None = None,
    *,
    start: int | None = None,
    end: int | None = None,
    base: float | None = None,
    esc_pct: float | None = None,
) -> dict[int, float]:
    """
    Expand an explicit schedule to all years, or build a schedule with escalator logic.

    - If `schedule` is provided, expand it across the requested `years` with no gaps.
    - If `schedule` is empty, but start/end/base/esc_pct are given, build a schedule
      growing by esc_pct% per year.
    """

    # Case 1: Explicit schedule provided
    if schedule:
        normalized_years: list[int] = []
        if years is not None:
            for entry in years:
                try:
                    normalized_years.append(int(entry))
                except (TypeError, ValueError):
                    continue

        schedule_items = [(int(year), float(price)) for year, price in schedule.items()]
        if not schedule_items:
            return {}

        schedule_items.sort(key=lambda item: item[0])
        if not normalized_years:
            return dict(schedule_items)

        expanded: dict[int, float] = {}
        sorted_years = sorted(dict.fromkeys(normalized_years))
        current_price = schedule_items[0][1]
        index = 0
        total_schedule = len(schedule_items)

        for year in sorted_years:
            while index < total_schedule and schedule_items[index][0] <= year:
                current_price = schedule_items[index][1]
                index += 1
            expanded[year] = float(current_price)
        return expanded

    # Case 2: Build schedule from base + escalator
    try:
        start_year = int(start) if start is not None else None
        end_year = int(end) if end is not None else None
    except (TypeError, ValueError):
        return {}

    if start_year is None or end_year is None:
        return {}

    try:
        base_value = float(base) if base is not None else 0.0
    except (TypeError, ValueError):
        base_value = 0.0

    try:
        escalator_value = float(esc_pct) if esc_pct is not None else 0.0
    except (TypeError, ValueError):
        escalator_value = 0.0

    return _build_price_schedule(start_year, end_year, base_value, escalator_value)



def _build_price_schedule(
    start_year: int,
    end_year: int,
    base_value: float,
    escalator_pct: float,
) -> dict[int, float]:
    """Return a price schedule that grows geometrically each year."""

    try:
        start = int(start_year)
    except (TypeError, ValueError):
        return {}
    try:
        end = int(end_year)
    except (TypeError, ValueError):
        return {}

    try:
        base = float(base_value)
    except (TypeError, ValueError):
        base = 0.0
    try:
        escalator = float(escalator_pct)
    except (TypeError, ValueError):
        escalator = 0.0

    step = 1 if end >= start else -1
    ratio = 1.0 + (escalator or 0.0) / 100.0

    schedule_items: list[tuple[int, float]] = []
    for exponent, year in enumerate(range(start, end + step, step)):
        try:
            factor = ratio ** exponent
        except OverflowError:
            factor = float("inf")
        schedule_items.append((year, round(base * factor, 6)))

    if not schedule_items:
        return {}

    return dict(sorted(schedule_items))


def _build_price_escalator_schedule(
    base_price: float,
    escalator_pct: float,
    years: Iterable[int] | None,
) -> dict[int, float]:
    """Return a price schedule grown annually by ``escalator_pct``."""

    if years is None:
        return {}

    try:
        base_value = float(base_price)
    except (TypeError, ValueError):
        base_value = 0.0
    try:
        escalator_value = float(escalator_pct)
    except (TypeError, ValueError):
        escalator_value = 0.0

    year_list: list[int] = []
    for entry in years:
        try:
            year_list.append(int(entry))
        except (TypeError, ValueError):
            continue

    if not year_list:
        return {}

    ordered_years = sorted(dict.fromkeys(year_list))
    base_year = ordered_years[0]
    full_schedule = _build_price_schedule(
        base_year,
        ordered_years[-1],
        base_value,
        escalator_value,
    )
    return {year: float(price) for year, price in full_schedule.items()}


def _build_cap_reduction_schedule(
    start_value: float,
    reduction_mode: str,
    reduction_value: float,
    years: Iterable[int] | None,
) -> dict[int, float]:
    """Return a cap schedule reduced each year by the specified rule."""

    if years is None:
        return {}

    try:
        start_amount = float(start_value)
    except (TypeError, ValueError):
        start_amount = 0.0
    try:
        reduction_amount = float(reduction_value)
    except (TypeError, ValueError):
        reduction_amount = 0.0

    year_list: list[int] = []
    for entry in years:
        try:
            year_list.append(int(entry))
        except (TypeError, ValueError):
            continue

    if not year_list:
        return {}

    normalized_mode = (reduction_mode or "").strip().lower()
    if normalized_mode not in {"percent", "fixed"}:
        normalized_mode = "percent"

    schedule: dict[int, float] = {}
    for idx, year in enumerate(sorted(dict.fromkeys(year_list))):
        if normalized_mode == "percent":
            decrement = start_amount * (max(reduction_amount, 0.0) / 100.0) * idx
        else:
            decrement = max(reduction_amount, 0.0) * idx
        value = max(start_amount - decrement, 0.0)
        schedule[year] = float(value)
    return schedule


def _merge_module_dicts(*sections: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Combine multiple module configuration sections into a copy."""

    merged: dict[str, dict[str, Any]] = {}
    for section in sections:
        if not isinstance(section, Mapping):
            continue
        for name, settings in section.items():
            key = str(name)
            if isinstance(settings, Mapping):
                existing = merged.get(key, {})
                combined = dict(existing)
                combined.update(settings)
                merged[key] = combined
            else:
                merged[key] = {'value': settings}
    return merged


def _build_price_escalator_schedule(
    base_price: float,
    escalator_pct: float,
    years: Iterable[int] | None,
) -> dict[int, float]:
    """Return a price schedule grown annually by ``escalator_pct``."""

    if years is None:
        return {}

    try:
        base_value = float(base_price)
    except (TypeError, ValueError):
        base_value = 0.0
    try:
        escalator_value = float(escalator_pct)
    except (TypeError, ValueError):
        escalator_value = 0.0

    year_list: list[int] = []
    for entry in years:
        try:
            year_list.append(int(entry))
        except (TypeError, ValueError):
            continue

    if not year_list:
        return {}

    ordered_years = sorted(dict.fromkeys(year_list))
    base_year = ordered_years[0]
    full_schedule = _build_price_schedule(
        base_year,
        ordered_years[-1],
        base_value,
        escalator_value,
    )
    return {year: float(price) for year, price in full_schedule.items()}


def _build_cap_reduction_schedule(
    start_value: float,
    reduction_mode: str,
    reduction_value: float,
    years: Iterable[int] | None,
) -> dict[int, float]:
    """Return a cap schedule reduced each year by the specified rule."""

    if years is None:
        return {}

    try:
        start_amount = float(start_value)
    except (TypeError, ValueError):
        start_amount = 0.0
    try:
        reduction_amount = float(reduction_value)
    except (TypeError, ValueError):
        reduction_amount = 0.0

    year_list: list[int] = []
    for entry in years:
        try:
            year_list.append(int(entry))
        except (TypeError, ValueError):
            continue

    if not year_list:
        return {}

    normalized_mode = (reduction_mode or "").strip().lower()
    if normalized_mode not in {"percent", "fixed"}:
        normalized_mode = "percent"

    schedule: dict[int, float] = {}
    for idx, year in enumerate(sorted(dict.fromkeys(year_list))):
        if normalized_mode == "percent":
            decrement = start_amount * (max(reduction_amount, 0.0) / 100.0) * idx
        else:
            decrement = max(reduction_amount, 0.0) * idx
        value = max(start_amount - decrement, 0.0)
        schedule[year] = float(value)
    return schedule


def _merge_module_dicts(*sections: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Combine multiple module configuration sections into a copy."""

    merged: dict[str, dict[str, Any]] = {}
    for section in sections:
        if not isinstance(section, Mapping):
            continue
        for name, settings in section.items():
            key = str(name)
            if isinstance(settings, Mapping):
                existing = merged.get(key, {})
                combined = dict(existing)
                combined.update(settings)
                merged[key] = combined
            else:
                merged[key] = {'value': settings}
    return merged


def _build_price_escalator_schedule(
    base_price: float,
    escalator_pct: float,
    years: Iterable[int] | None,
) -> dict[int, float]:
    """Return a price schedule grown annually by ``escalator_pct``."""

    if years is None:
        return {}

    try:
        base_value = float(base_price)
    except (TypeError, ValueError):
        base_value = 0.0
    try:
        escalator_value = float(escalator_pct)
    except (TypeError, ValueError):
        escalator_value = 0.0

    year_list: list[int] = []
    for entry in years:
        try:
            year_list.append(int(entry))
        except (TypeError, ValueError):
            continue

    if not year_list:
        return {}

    ordered_years = sorted(dict.fromkeys(year_list))
    base_year = ordered_years[0]
    full_schedule = _build_price_schedule(
        base_year,
        ordered_years[-1],
        base_value,
        escalator_value,
    )
    return {year: float(price) for year, price in full_schedule.items()}


def _build_cap_reduction_schedule(
    start_value: float,
    reduction_mode: str,
    reduction_value: float,
    years: Iterable[int] | None,
) -> dict[int, float]:
    """Return a cap schedule reduced each year by the specified rule."""

    if years is None:
        return {}

    try:
        start_amount = float(start_value)
    except (TypeError, ValueError):
        start_amount = 0.0
    try:
        reduction_amount = float(reduction_value)
    except (TypeError, ValueError):
        reduction_amount = 0.0

    year_list: list[int] = []
    for entry in years:
        try:
            year_list.append(int(entry))
        except (TypeError, ValueError):
            continue

    if not year_list:
        return {}

    normalized_mode = (reduction_mode or "").strip().lower()
    if normalized_mode not in {"percent", "fixed"}:
        normalized_mode = "percent"

    schedule: dict[int, float] = {}
    for idx, year in enumerate(sorted(dict.fromkeys(year_list))):
        if normalized_mode == "percent":
            decrement = start_amount * (max(reduction_amount, 0.0) / 100.0) * idx
        else:
            decrement = max(reduction_amount, 0.0) * idx
        value = max(start_amount - decrement, 0.0)
        schedule[year] = float(value)
    return schedule


def _merge_module_dicts(*sections: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Combine multiple module configuration sections into a copy."""

    merged: dict[str, dict[str, Any]] = {}
    for section in sections:
        if not isinstance(section, Mapping):
            continue
        for name, settings in section.items():
            key = str(name)
            if isinstance(settings, Mapping):
                existing = merged.get(key, {})
                combined = dict(existing)
                combined.update(settings)
                merged[key] = combined
            else:
                merged[key] = {'value': settings}
    return merged


def _build_price_escalator_schedule(
    base_price: float,
    escalator_pct: float,
    years: Iterable[int] | None,
) -> dict[int, float]:
    """Return a price schedule grown annually by ``escalator_pct``."""

    if years is None:
        return {}

    try:
        base_value = float(base_price)
    except (TypeError, ValueError):
        base_value = 0.0
    try:
        escalator_value = float(escalator_pct)
    except (TypeError, ValueError):
        escalator_value = 0.0

    year_list: list[int] = []
    for entry in years:
        try:
            year_list.append(int(entry))
        except (TypeError, ValueError):
            continue

    if not year_list:
        return {}

    ordered_years = sorted(dict.fromkeys(year_list))
    base_year = ordered_years[0]
    full_schedule = _build_price_schedule(
        base_year,
        ordered_years[-1],
        base_value,
        escalator_value,
    )
    return {year: float(price) for year, price in full_schedule.items()}


def _build_cap_reduction_schedule(
    start_value: float,
    reduction_mode: str,
    reduction_value: float,
    years: Iterable[int] | None,
) -> dict[int, float]:
    """Return a cap schedule reduced each year by the specified rule."""

    if years is None:
        return {}

    try:
        start_amount = float(start_value)
    except (TypeError, ValueError):
        start_amount = 0.0
    try:
        reduction_amount = float(reduction_value)
    except (TypeError, ValueError):
        reduction_amount = 0.0

    year_list: list[int] = []
    for entry in years:
        try:
            year_list.append(int(entry))
        except (TypeError, ValueError):
            continue

    if not year_list:
        return {}

    normalized_mode = (reduction_mode or "").strip().lower()
    if normalized_mode not in {"percent", "fixed"}:
        normalized_mode = "percent"

    schedule: dict[int, float] = {}
    for idx, year in enumerate(sorted(dict.fromkeys(year_list))):
        if normalized_mode == "percent":
            decrement = start_amount * (max(reduction_amount, 0.0) / 100.0) * idx
        else:
            decrement = max(reduction_amount, 0.0) * idx
        value = max(start_amount - decrement, 0.0)
        schedule[year] = float(value)
    return schedule


def _merge_module_dicts(*sections: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Combine multiple module configuration sections into a copy."""

    merged: dict[str, dict[str, Any]] = {}
    for section in sections:
        if not isinstance(section, Mapping):
            continue
        for name, settings in section.items():
            key = str(name)
            if isinstance(settings, Mapping):
                existing = merged.get(key, {})
                combined = dict(existing)
                combined.update(settings)
                merged[key] = combined
            else:
                merged[key] = {'value': settings}
    return merged


def _normalize_region_labels(
    selected_labels: Iterable[str],
    previous_clean_selection: Iterable[str] | None,
) -> list[str]:
    normalized = [str(entry) for entry in selected_labels]
    if "All" in normalized and len(normalized) > 1:
        non_all = [e for e in normalized if e != "All"]
        prev = tuple(str(e) for e in (previous_clean_selection or ()))
        return non_all if prev == ("All",) and non_all else ["All"]
    return normalized


def _resolve_canonical_region(value: Any) -> str | None:
    """Map arbitrary region selectors to canonical registry identifiers."""

    if value in (None, ""):
        return None

    text = str(value).strip()
    if not text:
        return None

    region_id_list = _get_region_id_list()
    region_lookup = _build_region_lookup(region_id_list)
    lowered = text.lower()
    if lowered in {"all", "all regions", _ALL_REGIONS_LABEL.lower()}:
        return "All"

    lookup = region_lookup.get(lowered)
    if lookup is not None:
        return lookup

    try:
        index = int(text)
    except ValueError:
        index = None
    if index is not None and 1 <= index <= len(region_id_list):
        return region_id_list[index - 1]

    resolved = canonical_region_value(value)
    if isinstance(resolved, str):
        cleaned = resolved.strip()
        mapped = region_lookup.get(cleaned.lower())
        if mapped is not None:
            return mapped
        try:
            canonical = normalize_region_id(cleaned)
        except ValueError:
            canonical = ""
        if canonical:
            mapped = region_lookup.get(canonical.lower()) or region_lookup.get(
                normalize_token(canonical)
            )
            if mapped is not None:
                return mapped
            if canonical in region_id_list:
                return canonical
    elif isinstance(resolved, int) and 1 <= resolved <= len(region_id_list):
        return region_id_list[int(resolved) - 1]

    return None


def _regions_from_config(config: Mapping[str, Any]) -> list[str]:
    """Extract canonical region identifiers from ``config`` selections."""

    if not isinstance(config, Mapping):
        return []

    raw_regions = config.get("regions")
    candidates: list[Any] = []

    def _value_is_selected(value: Any) -> bool:
        if value in (None, "", False):
            return False
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return abs(float(value)) > 1e-12
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return False
            try:
                return abs(float(text)) > 1e-12
            except ValueError:
                return text.lower() not in {"false", "no", "off", "0"}
        if isinstance(value, Mapping):
            return any(_value_is_selected(entry) for entry in value.values())
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            return any(_value_is_selected(entry) for entry in value)
        return True

    def _register(entry: Any) -> None:
        if entry in (None, ""):
            return
        if isinstance(entry, Mapping):
            explicit_keys = [
                entry.get("id"),
                entry.get("region"),
                entry.get("value"),
                entry.get("name"),
                entry.get("label"),
            ]
            for candidate in explicit_keys:
                if candidate not in (None, ""):
                    _register(candidate)
            if not any(candidate not in (None, "") for candidate in explicit_keys):
                for key in entry.keys():
                    _register(key)
            return
        if isinstance(entry, Iterable) and not isinstance(entry, (str, bytes, bytearray)):
            for candidate in entry:
                _register(candidate)
            return
        candidates.append(entry)

    if isinstance(raw_regions, Mapping):
        for key, value in raw_regions.items():
            if _value_is_selected(value):
                _register(key)
            else:
                _register(value)
    else:
        _register(raw_regions)

    normalized: list[str] = []
    seen: set[str] = set()
    for entry in candidates:
        region_id = _resolve_canonical_region(entry)
        if region_id is None or region_id == "All":
            continue
        if region_id not in seen:
            seen.add(region_id)
            normalized.append(region_id)

    return normalized


def _normalize_coverage_selection(selection: Any) -> list[str]:
    """Return a normalised list of coverage region labels."""

    if isinstance(selection, Mapping):
        iterable: Iterable[Any] = selection.values()
    elif isinstance(selection, (str, bytes)) or not isinstance(selection, Iterable):
        iterable = [selection]
    else:
        iterable = selection

    normalized: list[str] = []
    for entry in iterable:
        region_token = _resolve_canonical_region(entry)
        if region_token is None:
            continue
        if region_token == "All":
            return ["All"]
        if region_token not in normalized:
            normalized.append(region_token)

    if not normalized:
        return ["All"]
    return normalized


def _coverage_default_display(
    coverage_default: Iterable[str], options: Iterable[str]
) -> list[str]:
    """Convert canonical coverage selections to UI display labels."""

    default_sequence = list(coverage_default)
    if default_sequence == ["All"]:
        return [_ALL_REGIONS_LABEL]

    choice_lookup = {str(option) for option in options}
    display_values: list[str] = []
    for token in default_sequence:
        if token == "All":
            candidate = _ALL_REGIONS_LABEL
        elif token in choice_lookup:
            candidate = str(token)
        else:
            candidate = canonical_region_label(token)

        if candidate in choice_lookup and candidate not in display_values:
            display_values.append(candidate)

    if not display_values:
        return [_ALL_REGIONS_LABEL]

    return display_values


def _normalize_cap_region_entries(
    selection: Iterable[Any] | Mapping[str, Any] | None,
) -> tuple[list[str], dict[str, str]]:
    """Return canonical cap region values and an alias map for lookup."""

    normalized: list[str] = []
    seen: set[str] = set()
    alias_source = {key.lower(): value for key, value in region_alias_map().items()}
    alias_map: dict[str, str] = {}
    default_region_id = _get_default_region_id()

    def _register_alias(key: Any, value: str) -> None:
        if key is None:
            return
        text = str(key).strip()
        if not text:
            return
        alias_map.setdefault(text, value)
        alias_map.setdefault(text.lower(), value)

    for region_id, meta in DEFAULT_REGION_METADATA.items():
        _register_alias(region_id, region_id)
        _register_alias(str(region_id), region_id)
        _register_alias(meta.code, region_id)
        _register_alias(meta.code.lower(), region_id)
        _register_alias(meta.label, region_id)
        _register_alias(meta.label.lower(), region_id)
        _register_alias(meta.area, region_id)
        _register_alias(meta.area.lower(), region_id)
        display_label = region_display_label(region_id)
        _register_alias(display_label, region_id)
        _register_alias(display_label.lower(), region_id)
        for alias in meta.aliases:
            _register_alias(alias, region_id)
            _register_alias(alias.lower(), region_id)

    encountered_all = False
    unresolved: list[str] = []

    if selection is None:
        return normalized, alias_map

    if isinstance(selection, Mapping):
        iterable: Iterable[Any] = selection.values()
    elif isinstance(selection, (str, bytes)):
        iterable = [selection]
    else:
        iterable = selection

    for entry in iterable:
        if entry in (None, ""):
            continue

        label = canonical_region_label(entry).strip()
        lowered_label = label.lower()
        if lowered_label in {"all", "all regions", _ALL_REGIONS_LABEL.lower()}:
            encountered_all = True
            continue

        resolved = canonical_region_value(entry)
        if isinstance(resolved, str):
            text = resolved.strip()
        elif isinstance(resolved, bool):
            text = str(int(resolved))
        elif isinstance(resolved, (int, float)):
            text = str(int(resolved))
        else:
            text = str(entry).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in {"all", "all regions", _ALL_REGIONS_LABEL.lower()}:
            encountered_all = True
            continue
        if lowered == "default":
            canonical_value = default_region_id
        else:
            match = alias_source.get(lowered)
            if match is None:
                match = alias_map.get(lowered)
            if match is None:
                unresolved.append(label or text or str(entry))
                continue
            canonical_value = str(match).strip()

        _register_alias(entry, canonical_value)
        _register_alias(label, canonical_value)
        _register_alias(canonical_value, canonical_value)

        if canonical_value not in seen:
            seen.add(canonical_value)
            normalized.append(canonical_value)

    if unresolved:
        unique_unresolved = list(dict.fromkeys(unresolved))
        if len(unique_unresolved) == 1:
            raise ValueError(f"Unable to resolve cap region '{unique_unresolved[0]}'")
        unresolved_list = ", ".join(f"'{entry}'" for entry in unique_unresolved)
        raise ValueError(f"Unable to resolve cap regions: {unresolved_list}")

    if encountered_all and not normalized:
        return [], alias_map

    return normalized, alias_map


# General Config UI
# -------------------------
def _render_general_config_section(
    container: Any,
    *,
    default_source: Any,
    default_label: str,
    default_config: Mapping[str, Any],
) -> GeneralConfigResult:
    try:
        base_config = copy.deepcopy(dict(default_config))
    except Exception:
        base_config = dict(default_config)

    preset_key = "manual"
    preset_label: str | None = None
    preset_apply: Callable[[dict[str, Any]], None] | None = None
    preset_locks_carbon = False

    if st is not None:
        default_preset_key = st.session_state.get(
            _GENERAL_PRESET_STATE_KEY,
            _GENERAL_CONFIG_PRESETS[0][0],
        )
        preset_labels = [entry[1] for entry in _GENERAL_CONFIG_PRESETS]
        try:
            default_index = next(
                idx
                for idx, entry in enumerate(_GENERAL_CONFIG_PRESETS)
                if entry[0] == default_preset_key
            )
        except StopIteration:
            default_index = 0
        selected_label = container.radio(
            "Configuration preset",
            options=preset_labels,
            index=default_index,
            key=_GENERAL_PRESET_WIDGET_KEY,
            help=(
                "Select a pre-configured scenario or edit the default configuration manually."
            ),
        )
        for key, label, apply_fn, lock_flag in _GENERAL_CONFIG_PRESETS:
            if label == selected_label:
                preset_key = key
                preset_label = label if key != "manual" else None
                preset_apply = apply_fn
                preset_locks_carbon = bool(lock_flag)
                st.session_state[_GENERAL_PRESET_STATE_KEY] = key
                break
    else:
        preset_key, _, preset_apply, lock_flag = _GENERAL_CONFIG_PRESETS[0]
        preset_label = None
        preset_locks_carbon = bool(lock_flag)

    config_label = default_label

    if preset_key == "manual":
        uploaded = container.file_uploader(
            "Run configuration (TOML)",
            type="toml",
            key="general_config_upload",
        )
        if uploaded is not None:
            config_label = uploaded.name or "uploaded_config.toml"
            try:
                base_config = _load_config_data(uploaded.getvalue())
            except Exception as exc:
                container.error(f"Failed to read configuration: {exc}")
                base_config = copy.deepcopy(dict(default_config))
                config_label = default_label
    else:
        config_label = preset_label or default_label

    container.caption(f"Using configuration: {config_label}")
    if preset_key != "manual":
        container.info(
            "Preset values are loaded automatically. Carbon policy settings are locked while this preset is active."
        )

    candidate_years = _years_from_config(base_config)
    current_year = date.today().year
    if candidate_years:
        year_min = min(candidate_years)
        year_max = max(candidate_years)
    else:
        try:
            year_min = int(base_config.get("start_year", current_year) or current_year)
        except (TypeError, ValueError):
            year_min = int(current_year)
        try:
            fallback_end = base_config.get("end_year", year_min + 1)
            year_max = int(fallback_end) if fallback_end not in (None, "") else year_min + 1
        except (TypeError, ValueError):
            year_max = year_min + 1
        if year_max <= year_min:
            year_max = year_min + 1
    if year_min > year_max:
        year_min, year_max = year_max, year_min

    def _coerce_year(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(fallback)

    start_default = max(
        year_min,
        min(year_max, _coerce_year(base_config.get("start_year", year_min), year_min)),
    )
    end_default = max(
        year_min,
        min(year_max, _coerce_year(base_config.get("end_year", year_max), year_max)),
    )
    if end_default <= start_default:
        end_default = start_default + 1

    # Hard bounds
    slider_min_dynamic = 2025
    slider_max_dynamic = 2050

    slider_min_value, slider_max_value = container.slider(
        "Simulation Years",
        min_value=slider_min_dynamic,
        max_value=slider_max_dynamic,
        value=(
            max(slider_min_dynamic, min(start_default, slider_max_dynamic - 1)),
            max(
                max(slider_min_dynamic + 1, min(end_default, slider_max_dynamic)),
                slider_min_dynamic + 1,
            ),
        ),
        step=1,
        key="general_year_slider",
    )


    intensity_settings = render_compute_intensity(
        container,
        help_text=(
            "Sets a unified maximum iteration count for capacity expansion, cap solver, "
            "and output directory generation."
        ),
    )

    start_year = int(slider_min_value)
    end_year = int(slider_max_value)

    if st is not None:
        st.session_state["start_year_slider"] = start_year
        st.session_state["end_year_slider"] = end_year

    invalid_year_range = start_year >= end_year
    if invalid_year_range:
        container.error("End year must be greater than start year.")

    region_options = _regions_from_config(base_config)
    default_region_values = [entry["id"] for entry in available_regions()]
    alias_to_value = region_alias_map()
    available_region_values: list[str] = []
    value_to_label: dict[str, str] = {}
    label_to_value: dict[str, str] = {}
    seen_values: set[str] = set()

    def _register_region_value(value: int | str) -> None:
        resolved = canonical_region_value(value)
        if isinstance(resolved, str):
            normalized = resolved.strip()
        else:
            normalized = str(resolved).strip()
        if not normalized or normalized == "All" or normalized in seen_values:
            return
        seen_values.add(normalized)
        label = region_display_label(normalized)
        available_region_values.append(normalized)
        value_to_label[normalized] = label
        label_to_value[label] = normalized
        alias_to_value.setdefault(label.lower(), normalized)
        alias_to_value.setdefault(normalized.lower(), normalized)
        alias_to_value.setdefault(str(value).strip().lower(), normalized)

    for candidate in (*default_region_values, *region_options):
        _register_region_value(candidate)

    region_labels = ["All"] + [value_to_label[v] for v in available_region_values]
    default_selection = ["All"]

    def _canonical_region_label_entry(entry: Any) -> str:
        text = str(entry).strip()
        if not text:
            return text
        if text == "All":
            return "All"
        if text in label_to_value:
            return text
        resolved = canonical_region_value(text)
        if isinstance(resolved, str):
            return value_to_label.get(resolved, region_display_label(resolved))
        return text

    def _canonicalize_selection(entries: Iterable[Any]) -> list[str]:
        canonical: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            label = _canonical_region_label_entry(entry)
            if label and label not in seen:
                canonical.append(label)
                seen.add(label)
        if not canonical:
            canonical = list(default_selection)
        return canonical

    if st is not None:
        st.session_state.setdefault(
            _GENERAL_REGIONS_NORMALIZED_KEY, list(default_selection)
        )
        prev_raw = st.session_state.get(_GENERAL_REGIONS_NORMALIZED_KEY, [])
        if isinstance(prev_raw, (list, tuple)):
            previous_clean_selection = _canonicalize_selection(prev_raw)
        elif isinstance(prev_raw, str):
            previous_clean_selection = _canonicalize_selection([prev_raw])
        else:
            previous_clean_selection = list(default_selection)

        existing_widget_value = st.session_state.get("general_regions")
        if isinstance(existing_widget_value, str):
            existing_entries: Iterable[Any] = [existing_widget_value]
        elif isinstance(existing_widget_value, (list, tuple, set)):
            existing_entries = existing_widget_value
        else:
            existing_entries = []

        if existing_entries:
            canonical_existing = _canonicalize_selection(existing_entries)
        else:
            canonical_existing = previous_clean_selection

        previous_clean_selection = canonical_existing
    else:
        previous_clean_selection = list(default_selection)

    selected_regions_raw = list(
        container.multiselect(
            "Regions",
            options=region_labels,
            default=previous_clean_selection,
            key="general_regions",
        )
    )
    normalized_selection = _normalize_region_labels(
        selected_regions_raw, previous_clean_selection
    )
    canonical_selection: list[str] = []
    seen_labels: set[str] = set()
    for entry in normalized_selection:
        label = _canonical_region_label_entry(entry)
        if label and label not in seen_labels:
            canonical_selection.append(label)
            seen_labels.add(label)
    selected_regions_raw = canonical_selection
    if st is not None:
        st.session_state[_GENERAL_REGIONS_NORMALIZED_KEY] = list(selected_regions_raw)

    all_selected = "All" in selected_regions_raw
    if all_selected or not selected_regions_raw:
        selected_regions = list(available_region_values)
    else:
        selected_regions = []
        for entry in selected_regions_raw:
            if entry == "All":
                continue
            value = label_to_value.get(entry)
            if value is None:
                resolved = canonical_region_value(entry)
                if isinstance(resolved, str):
                    value = resolved.strip()
            if not value:
                continue
            if value in available_region_values and value not in selected_regions:
                selected_regions.append(value)
    if not selected_regions:
        selected_regions = list(available_region_values)

    selected_states = _states_from_config(base_config)
    if st is not None:
        raw_state_selection = st.session_state.get("general_states")
        if isinstance(raw_state_selection, (list, tuple, set)):
            normalized_states = _normalize_state_codes(raw_state_selection)
            if normalized_states:
                selected_states = normalized_states
        elif isinstance(raw_state_selection, str):
            normalized_states = _normalize_state_codes([raw_state_selection])
            if normalized_states:
                selected_states = normalized_states
        st.session_state["general_states"] = list(selected_states)

    resolve_states = getattr(gui_helpers, "resolve_state_selection", None)
    state_weight_map: dict[str, float] = {}
    if callable(resolve_states) and selected_states:
        try:
            state_weight_map = dict(resolve_states(list(selected_states)))
        except Exception as exc:
            LOGGER.exception("Failed to resolve state selection: %s", exc)
            if st is not None:
                container.warning(f"Failed to resolve state selections: {exc}")
            state_weight_map = {}
    if state_weight_map:
        state_weight_map = dict(sorted(state_weight_map.items()))
    elif not selected_states:
        state_weight_map = {region: 1.0 for region in selected_regions}
    elif st is not None:
        container.warning(
            "No state-region shares were found for the selected states. Please review the"
            " registry or adjust your selections."
        )

    run_config = copy.deepcopy(base_config)
    run_config["start_year"] = start_year
    run_config["end_year"] = end_year
    run_config["regions"] = state_weight_map
    run_config["states"] = list(selected_states)
    run_config.setdefault("modules", {})
    run_config["max_iter"] = int(intensity_settings.max_iterations)

    active_preset_key: str | None = None
    active_preset_label: str | None = None
    lock_carbon_controls = False
    if preset_key != "manual" and preset_apply is not None:
        try:
            preset_apply(run_config["modules"])
        except Exception as exc:
            container.error(f"Failed to apply preset defaults: {exc}")
        else:
            active_preset_key = preset_key
            active_preset_label = preset_label
            lock_carbon_controls = bool(preset_locks_carbon)

    try:
        selected_years = _select_years(candidate_years, start_year, end_year)
    except Exception:
        selected_years = []
    if selected_years:
        try:
            selected_min = min(int(year) for year in selected_years)
            selected_max = max(int(year) for year in selected_years)
        except ValueError:
            selected_years = []
        else:
            selected_years = list(range(selected_min, selected_max + 1))
    elif not invalid_year_range:
        selected_years = list(range(start_year, end_year + 1))
    else:
        selected_years = []

    if selected_years:
        years_for_run = list(selected_years)
    elif not invalid_year_range:
        years_for_run = list(range(start_year, end_year + 1))
    else:
        years_for_run = []

    if years_for_run:
        run_config["years"] = years_for_run

    return GeneralConfigResult(
        config_label=config_label,
        config_source=run_config,
        run_config=run_config,
        candidate_years=candidate_years,
        start_year=start_year,
        end_year=end_year,
        selected_years=selected_years,
        regions=selected_regions,
        preset_key=active_preset_key,
        preset_label=active_preset_label,
        lock_carbon_controls=lock_carbon_controls,
        max_iterations=intensity_settings.max_iterations,
        fuel_price_selection={},
    )


def _render_fuel_cost_helper_section(
    container: Any,
    *,
    run_config: dict[str, Any],
    years: Iterable[int] | None,
) -> dict[str, str]:
    """Render the fuel cost helper controls in their own sidebar section."""

    existing_selection: Mapping[str, Any] | None = None
    fuel_prices_config = run_config.get("fuel_prices")
    if isinstance(fuel_prices_config, Mapping):
        existing_selection = fuel_prices_config.get("scenario_by_fuel")  # type: ignore[assignment]

    # Default fuel price scenarios matching REF prices from screenshot
    default_fuel_scenarios = {
        "COAL": "REF",
        "DISTILLATE": "REF",
        "GAS": "REF",
        "RESIDUAL": "REF",
        "URANIUM": "REF",
    }
    
    normalized_existing = _normalize_fuel_price_selection(existing_selection)
    
    # Apply defaults if no existing selection
    if not normalized_existing:
        normalized_existing = dict(default_fuel_scenarios)

    session_selection: dict[str, str] = {}
    if st is not None:
        session_selection = _normalize_fuel_price_selection(
            st.session_state.get(_FUEL_PRICE_HELPER_STATE_KEY)
        )
        if session_selection:
            normalized_existing = session_selection

    # Enable by default
    helper_enabled_default = True
    if st is not None:
        helper_enabled = bool(
            container.checkbox(
                "Use fuel cost helper",
                value=helper_enabled_default,
                key=_FUEL_PRICE_HELPER_ENABLED_KEY,
            )
        )
    else:
        helper_enabled = helper_enabled_default

    if not helper_enabled:
        if st is not None and not session_selection:
            st.session_state.pop(_FUEL_PRICE_HELPER_STATE_KEY, None)
        return dict(normalized_existing)

    selection = _render_fuel_price_helper_controls(
        container,
        years=years,
        existing_selection=normalized_existing,
    )

    if not selection and normalized_existing:
        selection = normalized_existing

    fuel_prices_section = dict(run_config.get("fuel_prices", {}))
    if selection:
        fuel_prices_section["scenario_by_fuel"] = dict(selection)
        run_config["fuel_prices"] = fuel_prices_section
    else:
        if "scenario_by_fuel" in fuel_prices_section:
            fuel_prices_section.pop("scenario_by_fuel", None)
        if fuel_prices_section:
            run_config["fuel_prices"] = fuel_prices_section
        elif "fuel_prices" in run_config:
            run_config.pop("fuel_prices")

    return dict(selection)


def _documentation_file_ready(path: str | os.PathLike | None) -> bool:
    if not path:
        return False
    try:
        candidate = Path(path)
    except TypeError:
        return False
    if not candidate.exists() or not candidate.is_file():
        return False
    try:
        return candidate.stat().st_size > 0
    except OSError:
        return False


def _sanitize_documentation_mapping(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}

    sanitized: dict[str, Any] = {key: value for key, value in payload.items()}

    manifest_paths: dict[str, str] = {}
    raw_manifest_paths = sanitized.get("manifest_paths")
    if isinstance(raw_manifest_paths, Mapping):
        json_path = raw_manifest_paths.get("json")
        if _documentation_file_ready(json_path):
            manifest_paths["json"] = str(Path(json_path))
        md_path = raw_manifest_paths.get("md")
        if _documentation_file_ready(md_path):
            manifest_paths["md"] = str(Path(md_path))
    if manifest_paths:
        sanitized["manifest_paths"] = manifest_paths
    else:
        sanitized.pop("manifest_paths", None)

    deep_doc_path = sanitized.get("deep_doc_path")
    if _documentation_file_ready(deep_doc_path):
        sanitized["deep_doc_path"] = str(Path(deep_doc_path))
    else:
        sanitized.pop("deep_doc_path", None)

    return sanitized


def _normalize_custom_record_list(
    records: Iterable[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return sanitized region-year demand records from arbitrary mappings."""

    if records is None:
        return []

    try:
        frame = pd.DataFrame(list(records))
    except Exception:
        return []

    normalized = normalize_gui_demand_table(frame)
    if normalized.empty:
        return []

    normalized = normalized.dropna(subset=["region"])
    if normalized.empty:
        return []

    aggregated = (
        normalized.groupby(["region", "year"], as_index=False)["demand_mwh"].sum()
    )

    sanitized: list[dict[str, Any]] = []
    for _, row in aggregated.iterrows():
        region_value = row.get("region")
        region_text = str(region_value).strip() if region_value not in (None, "") else ""
        if not region_text:
            continue

        canonical_region = canonical_region_value(region_text)
        if isinstance(canonical_region, bool):
            region_key = str(int(canonical_region))
        elif isinstance(canonical_region, (int, float)):
            region_key = str(int(canonical_region))
        elif isinstance(canonical_region, str) and canonical_region.strip():
            region_key = canonical_region.strip()
        else:
            region_key = region_text

        sanitized.append(
            {
                "region": region_key,
                "year": int(row["year"]),
                "demand_mwh": float(row["demand_mwh"]),
            }
        )

    return sorted(sanitized, key=lambda rec: (rec["region"], rec["year"]))


def _coerce_custom_upload_mapping(
    payload: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Normalize persisted custom load forecast entries."""

    if not isinstance(payload, Mapping):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for key, raw_entry in payload.items():
        state_code = str(key).strip().upper()
        if not state_code:
            continue

        source_label: str | None = None
        records_source: Any = raw_entry
        if isinstance(raw_entry, Mapping):
            source_candidate = raw_entry.get("source") or raw_entry.get("filename")
            if source_candidate not in (None, ""):
                source_label = str(source_candidate)
            records_source = raw_entry.get("records")
            if records_source is None:
                for alt_key in ("data", "rows"):
                    if raw_entry.get(alt_key) is not None:
                        records_source = raw_entry.get(alt_key)
                        break

        if isinstance(records_source, pd.DataFrame):
            record_iter: Iterable[Mapping[str, Any]] = records_source.to_dict("records")
        elif isinstance(records_source, Mapping):
            try:
                record_iter = list(records_source.values())
            except Exception:
                record_iter = []
        elif isinstance(records_source, Sequence) and not isinstance(
            records_source, (str, bytes, bytearray)
        ):
            record_iter = list(records_source)
        else:
            record_iter = []

        normalized_records = _normalize_custom_record_list(record_iter)
        if not normalized_records:
            continue

        entry: dict[str, Any] = {
            "state": state_code,
            "records": normalized_records,
            "regions": sorted({rec["region"] for rec in normalized_records}),
            "years": sorted({rec["year"] for rec in normalized_records}),
        }
        if source_label:
            entry["source"] = source_label
        normalized[state_code] = entry

    return normalized


def _normalize_custom_forecast_upload(
    state_code: str, frame: pd.DataFrame
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize an uploaded custom demand curve for ``state_code``."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Uploaded forecast is empty.")

    normalized_frame = normalize_gui_demand_table(frame)
    if normalized_frame.empty:
        raise ValueError("Uploaded forecast did not contain any usable demand values.")

    has_region_data = normalized_frame["region"].notna().any()
    if has_region_data:
        normalized_records = _normalize_custom_record_list(normalized_frame.to_dict("records"))
    else:
        aggregated_by_year = (
            normalized_frame.groupby("year", as_index=False)["demand_mwh"].sum()
        )
        if aggregated_by_year.empty:
            raise ValueError("Uploaded forecast did not contain any usable demand values.")

        weights = gui_helpers.resolve_state_selection([state_code])
        if not weights:
            raise ValueError(f"Unable to resolve model regions for state {state_code}.")

        weight_items: list[tuple[str, float]] = []
        for region_id, weight in weights.items():
            try:
                numeric_weight = float(weight)
            except (TypeError, ValueError):
                continue
            if abs(numeric_weight) < 1e-12:
                continue
            weight_items.append((region_id, numeric_weight))

        if not weight_items:
            raise ValueError(f"State {state_code} does not map to any model regions.")

        weighted_records: list[dict[str, Any]] = []
        for _, row in aggregated_by_year.iterrows():
            year_key = int(row["year"])
            demand_value = float(row["demand_mwh"])
            for region_id, numeric_weight in weight_items:
                weighted_records.append(
                    {
                        "region": region_id,
                        "year": year_key,
                        "demand_mwh": demand_value * float(numeric_weight),
                    }
                )

        normalized_records = _normalize_custom_record_list(weighted_records)

    if not normalized_records:
        raise ValueError("Uploaded forecast did not contain any usable region data.")

    regions = sorted({rec["region"] for rec in normalized_records})
    years = sorted({rec["year"] for rec in normalized_records})
    total_mwh = sum(rec["demand_mwh"] for rec in normalized_records)

    summary = {
        "state": state_code,
        "regions": regions,
        "years": years,
        "row_count": len(normalized_records),
        "total_mwh": float(total_mwh),
    }

    return normalized_records, summary


def _serialize_custom_forecast_entry(
    state_code: str,
    records: Sequence[Mapping[str, Any]],
    *,
    source: str | None = None,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a persisted mapping for a custom forecast upload."""

    normalized_records = _normalize_custom_record_list(records)
    if not normalized_records:
        return {}

    entry: dict[str, Any] = {
        "state": state_code,
        "records": normalized_records,
        "regions": sorted({rec["region"] for rec in normalized_records}),
        "years": sorted({rec["year"] for rec in normalized_records}),
        "row_count": len(normalized_records),
        "total_mwh": float(sum(rec["demand_mwh"] for rec in normalized_records)),
    }

    if source not in (None, ""):
        entry["source"] = str(source)

    if isinstance(summary, Mapping):
        if summary.get("regions"):
            try:
                entry["regions"] = sorted({str(value) for value in summary["regions"]})
            except Exception:
                pass
        if summary.get("years"):
            try:
                entry["years"] = sorted({int(value) for value in summary["years"]})
            except Exception:
                pass
        if summary.get("row_count") not in (None, ""):
            try:
                entry["row_count"] = int(summary["row_count"])
            except (TypeError, ValueError):
                pass
        if summary.get("total_mwh") not in (None, ""):
            try:
                entry["total_mwh"] = float(summary["total_mwh"])
            except (TypeError, ValueError):
                pass

    return entry


def _describe_custom_forecast(entry: Mapping[str, Any]) -> str:
    """Return a concise description for a custom load forecast entry."""

    if not isinstance(entry, Mapping):
        return "Uploaded forecast in use."

    source = entry.get("source")
    regions = entry.get("regions")
    years = entry.get("years")
    total_mwh = entry.get("total_mwh")
    row_count = entry.get("row_count")

    parts: list[str] = []
    if source:
        parts.append(f"{source}")

    if isinstance(years, (Sequence, set)) and not isinstance(years, (str, bytes, bytearray)):
        try:
            year_values = sorted({int(year) for year in years})
        except (TypeError, ValueError):
            year_values = []
        if year_values:
            if len(year_values) == 1:
                parts.append(f"year {year_values[0]}")
            else:
                parts.append(f"years {year_values[0]}–{year_values[-1]}")

    if isinstance(regions, (Sequence, set)) and not isinstance(regions, (str, bytes, bytearray)):
        try:
            region_count = len({str(region) for region in regions if region not in (None, "")})
        except Exception:
            region_count = 0
        if region_count:
            parts.append(f"{region_count} region(s)")

    if total_mwh not in (None, ""):
        try:
            parts.append(f"{float(total_mwh):,.0f} MWh total")
        except (TypeError, ValueError):
            pass

    if row_count not in (None, ""):
        try:
            rows_int = int(row_count)
        except (TypeError, ValueError):
            rows_int = 0
        if rows_int:
            parts.append(f"{rows_int} row(s)")

    if not parts:
        return "Uploaded forecast in use."

    return ", ".join(parts)


def _render_demand_module_section(
    container: Any,
    run_config: dict[str, Any],
    *,
    regions: Sequence[Any],
    years: Sequence[int] | None = None,
) -> DemandModuleSettings:
    modules = run_config.setdefault("modules", {})
    demand_defaults = modules.get("demand")
    default_selections = _canonical_forecast_selection(
        demand_defaults.get("load_forecasts") if isinstance(demand_defaults, Mapping) else None
    )

    custom_uploads: dict[str, dict[str, Any]] = {}
    if isinstance(demand_defaults, Mapping):
        custom_uploads.update(
            _coerce_custom_upload_mapping(demand_defaults.get("custom_load_forecasts"))
        )

    session_selections: dict[str, str] = {}
    if st is not None:
        raw_session_selection = st.session_state.get("forecast_selections")
        if isinstance(raw_session_selection, Mapping):
            for state_key, scenario_label in raw_session_selection.items():
                if scenario_label:
                    state_code = str(state_key).strip().upper()
                    scenario_text = str(scenario_label).strip()
                    if state_code and scenario_text:
                        session_selections[state_code] = scenario_text

        session_custom_uploads = _coerce_custom_upload_mapping(
            st.session_state.get("custom_load_forecasts")
        )
        if session_custom_uploads:
            custom_uploads.update(session_custom_uploads)

    is_ui = st is not None
    base_path = _resolve_forecast_base_path()

    try:
        cached_frame = _cached_forecast_frame(base_path)
    except Exception:
        cached_frame = pd.DataFrame()

    if isinstance(cached_frame, pd.DataFrame):
        selection_frame = cached_frame
    else:
        try:
            selection_frame = pd.DataFrame(cached_frame)
        except Exception:
            selection_frame = pd.DataFrame()

    try:
        frame = normalize_load_frame(selection_frame.copy())
    except Exception:
        frame = selection_frame.copy()

    forecast_error = forecast_frame_error(base_path)

    display_selections: dict[str, str] = {}
    iso_display_labels: dict[str, str] = {}
    errors: list[str] = []

    if isinstance(regions, Mapping):
        region_filter_active = bool(regions)
    else:
        region_filter_active = bool(list(regions) if regions is not None else [])

    if is_ui and hasattr(container, "subheader"):
        container.subheader("Demand curves")

    iso_series = frame["iso"].astype(str)
    scenario_series = frame["scenario"].astype(str)
    state_series = frame.get("state")
    iso_token_series = iso_series.str.strip().str.lower()

    state_iso_scenario_map: dict[tuple[str, str], list[str]] = {}
    if state_series is not None:
        normalized_states = state_series.astype(str).str.strip().str.upper()
        working = frame.copy()
        working["_iso_token"] = iso_token_series
        working["_state_token"] = normalized_states
        for (state_code, iso_token_value), subset in working.groupby(
            ["_state_token", "_iso_token"]
        ):
            if not state_code or not iso_token_value:
                continue
            scenario_values = subset["scenario"].astype(str).str.strip()
            ordered = _ordered_scenarios(
                [value for value in dict.fromkeys(scenario_values) if value]
            )
            if ordered:
                state_iso_scenario_map[(state_code, iso_token_value)] = ordered

    iso_values = sorted(dict.fromkeys(iso_series.tolist()))
    if not iso_values:
        if forecast_error:
            errors.append(forecast_error)
            if is_ui:
                container.error(f"Load forecasts could not be loaded: {forecast_error}")
        elif custom_uploads and is_ui:
            container.info(
                "No built-in load forecast scenarios were available; using uploaded demand curves."
            )
        elif is_ui:
            container.info("No load forecast scenarios were available.")

        if custom_uploads:
            modules["demand"] = {
                "enabled": True,
                "curve_by_region": {},
                "forecast_by_region": {},
                "load_forecasts": {},
                "custom_load_forecasts": dict(custom_uploads),
                "errors": list(errors),
            }
            if st is not None:
                _set_state_if_changed("selected_forecast_bundles", [])
                _set_state_if_changed("forecast_selections", {})
                _set_state_if_changed("forecast_iso_labels", {})
                _set_state_if_changed("custom_load_forecasts", dict(custom_uploads))
            return DemandModuleSettings(
                enabled=True,
                curve_by_region={},
                forecast_by_region={},
                load_forecasts={},
                custom_load_forecasts=dict(custom_uploads),
                errors=list(errors),
            )

        modules["demand"] = {
            "enabled": False,
            "curve_by_region": {},
            "forecast_by_region": {},
            "load_forecasts": {},
            "custom_load_forecasts": {},
            "errors": list(errors),
        }
        if st is not None:
            _set_state_if_changed("selected_forecast_bundles", [])
            _set_state_if_changed("forecast_selections", {})
            _set_state_if_changed("forecast_iso_labels", {})
            _set_state_if_changed("custom_load_forecasts", {})
        return DemandModuleSettings(
            enabled=False,
            curve_by_region={},
            forecast_by_region={},
            load_forecasts={},
            custom_load_forecasts={},
            errors=list(errors),
        )

    iso_groups = _iso_state_groups()
    iso_display_lookup: dict[str, str] = {}
    for iso_value in iso_values:
        iso_display = str(iso_value).strip()
        iso_token = normalize_iso_name(iso_display) or normalize_token(iso_display)
        if iso_token:
            iso_display_lookup.setdefault(iso_token, iso_display)
        iso_display_lookup.setdefault(iso_display, iso_display)
        iso_display_lookup.setdefault(iso_display.upper(), iso_display)

    seen_entries: set[int] = set()
    iso_entries: list[tuple[str, str | None, Mapping[str, Any]]] = []
    for entry in iso_groups.values():
        if not isinstance(entry, Mapping):
            continue
        entry_id = id(entry)
        if entry_id in seen_entries:
            continue
        seen_entries.add(entry_id)
        entry_label = str(entry.get("label") or "").strip()
        entry_token = normalize_iso_name(entry_label) or normalize_token(entry_label)
        iso_display = None
        if entry_token and entry_token in iso_display_lookup:
            iso_display = iso_display_lookup[entry_token]
        elif entry_label and entry_label in iso_display_lookup:
            iso_display = iso_display_lookup[entry_label]
        if iso_display is None:
            continue
        iso_entries.append((iso_display, entry_token, entry))

    if not iso_entries:
        order_counter = 0
        for iso_value in iso_values:
            iso_display = iso_display_lookup.get(str(iso_value)) or str(iso_value).strip()
            if not iso_display:
                continue
            token = normalize_iso_name(iso_display) or normalize_token(iso_display)
            iso_entries.append(
                (
                    iso_display,
                    token,
                    {
                        "label": iso_display,
                        "states": [(iso_display, True)],
                        "order": order_counter,
                    },
                )
            )
            order_counter += 1

    iso_entries.sort(key=lambda item: item[2].get("order", 0))

    if not iso_entries:
        for position, iso_display in enumerate(iso_values):
            iso_label = str(iso_display).strip()
            if not iso_label:
                continue
            fallback_entry = {
                "label": iso_label,
                "states": [iso_label],
                "order": position,
            }
            iso_entries.append((iso_label, normalize_token(iso_label), fallback_entry))

    available_tokens: set[str] = set()
    for iso_display, iso_token, _ in iso_entries:
        token = iso_token or normalize_token(iso_display)
        if token:
            available_tokens.add(token)

    preferred_iso_by_state: dict[str, str] = {}
    for source in (default_selections, session_selections):
        for state_key, selection_value in source.items():
            state_code = str(state_key).strip().upper()
            if not state_code:
                continue
            iso_label, _ = _decode_forecast_selection(selection_value)
            if iso_label:
                preferred_iso_by_state[state_code] = iso_label

    processed_states: set[str] = set()
    fallback_iso_only = not region_filter_active

    # Build dropdowns grouped by ISO and state
    for iso_display, iso_token, group_entry in iso_entries:
        states = group_entry.get("states")
        if not isinstance(states, list) or not states:
            continue

        display_label = str(group_entry.get("label") or iso_display or iso_token)

        iso_key_lower = iso_display.strip().lower()
        mask = iso_token_series == iso_key_lower

        discovered_options: list[str] = []
        for lookup_iso in (iso_display, iso_token):
            if not lookup_iso:
                continue
            discovered = discover_iso_scenarios(base_path, str(lookup_iso))
            if discovered:
                discovered_options.extend(discovered)

        scenario_candidates: list[str] = []
        seen_options: set[str] = set()
        for option in discovered_options:
            text = str(option).strip()
            if text and text not in seen_options:
                scenario_candidates.append(text)
                seen_options.add(text)

        frame_options = frame.loc[mask, "scenario"].astype(str).dropna().tolist()
        for option in frame_options:
            text = str(option).strip()
            if text and text not in seen_options:
                scenario_candidates.append(text)
                seen_options.add(text)

        ordered_options = _ordered_scenarios(scenario_candidates)
        if not ordered_options:
            continue

        if is_ui and hasattr(container, "markdown"):
            container.markdown(f"**{display_label}**")

        for state_code in states:
            fallback_state = False
            if isinstance(state_code, tuple):
                state_tuple = tuple(state_code)
                state_text = str(state_tuple[0]).strip() if state_tuple else ""
                if len(state_tuple) > 1:
                    try:
                        fallback_state = bool(state_tuple[1])
                    except Exception:  # pragma: no cover - defensive guard
                        fallback_state = False
            else:
                state_text = str(state_code).strip()
            state_label = state_text.upper()
            if not state_label:
                continue

            state_lookup_key = state_text.upper()
            use_iso_label = fallback_iso_only or fallback_state
            state_label = state_text if use_iso_label else state_lookup_key

            preferred_iso = preferred_iso_by_state.get(state_lookup_key)
            preferred_token: str | None = None
            if preferred_iso:
                preferred_token = normalize_iso_name(preferred_iso) or normalize_token(preferred_iso)
                if preferred_token and preferred_token not in available_tokens:
                    preferred_token = None
            if preferred_token and iso_token and preferred_token != iso_token:
                continue
            if preferred_token is None and state_label in processed_states:
                continue

            state_custom_entry = custom_uploads.get(state_label)
            if state_custom_entry is not None:
                processed_states.add(state_label)
                if is_ui:
                    status_cols = container.columns([3, 1])
                    description = _describe_custom_forecast(state_custom_entry)
                    status_cols[0].success(f"Using uploaded forecast: {description}")
                    if status_cols[1].button(
                        "Clear upload",
                        key=f"clear_forecast_upload_{iso_token or normalize_token(display_label)}_{state_label}",
                        help="Remove the uploaded forecast and return to the built-in scenarios.",
                    ):
                        custom_uploads.pop(state_label, None)
                        state_custom_entry = None
                        processed_states.discard(state_label)

            if state_custom_entry is None:
                stored_value = session_selections.get(state_label) or default_selections.get(
                    state_label
                )
                stored_iso, stored_scenario = _decode_forecast_selection(stored_value)
                if stored_iso:
                    stored_iso_token = normalize_iso_name(stored_iso) or normalize_token(stored_iso)
                    if stored_iso_token and iso_token and stored_iso_token != iso_token:
                        stored_scenario = None

                state_options: list[str] | None = None
                if state_iso_scenario_map:
                    lookup_pairs = [(state_lookup_key, iso_key_lower)]
                    if iso_token:
                        lookup_pairs.append((state_lookup_key, iso_token.lower()))
                    for lookup in lookup_pairs:
                        if lookup in state_iso_scenario_map:
                            state_options = state_iso_scenario_map[lookup]
                            break

                options_for_state = state_options or ordered_options

                if not stored_scenario or stored_scenario not in options_for_state:
                    stored_scenario = options_for_state[0]

                iso_display_labels[state_label] = display_label

                if is_ui:
                    select_key = (
                        f"forecast_selection_{iso_token or normalize_token(display_label)}_{state_label}"
                    )
                    try:
                        default_index = options_for_state.index(stored_scenario)
                    except ValueError:
                        default_index = 0
                    selection = container.selectbox(
                        f"{state_label} load forecast scenario",
                        options=options_for_state,
                        index=default_index,
                        key=select_key,
                        help="Select the load forecast scenario for this region.",
                    )
                else:
                    selection = stored_scenario

                display_selections[state_label] = _encode_forecast_selection(
                    display_label, selection
                )
                processed_states.add(state_label)

    selection_payload: dict[str, str] = {}
    use_state_keys = len(display_selections) > 1
    for state_label, encoded_value in display_selections.items():
        value_text = str(encoded_value).strip()
        iso_label, scenario_label = _decode_forecast_selection(value_text)
        if not value_text:
            continue
        if use_state_keys:
            if iso_label and state_label.strip().lower() == iso_label.strip().lower():
                selection_payload[state_label] = scenario_label or value_text
            else:
                selection_payload[state_label] = value_text
            continue

        payload_value = value_text
        target_key = state_label
        if iso_label:
            payload_value = scenario_label or value_text
            target_key = normalize_iso_name(iso_label) or iso_label.strip().lower()
        if payload_value:
            selection_payload[target_key] = str(payload_value)

    if not selection_payload and not custom_uploads:
        if is_ui:
            container.info("No load forecast scenarios were selected; demand forecasts are disabled.")
        modules["demand"] = {
            "enabled": False,
            "curve_by_region": {},
            "forecast_by_region": {},
            "load_forecasts": {},
            "custom_load_forecasts": {},
            "errors": [],
        }
        if st is not None:
            _set_state_if_changed("selected_forecast_bundles", [])
            _set_state_if_changed("forecast_selections", {})
            _set_state_if_changed("forecast_iso_labels", {})
            _set_state_if_changed("custom_load_forecasts", {})
        return DemandModuleSettings(
            enabled=False,
            curve_by_region={},
            forecast_by_region={},
            load_forecasts={},
            custom_load_forecasts={},
            errors=[],
        )

    # Normal case
    modules["demand"] = {
        "enabled": bool(selection_payload or custom_uploads),
        "curve_by_region": {},
        "forecast_by_region": {},
        "load_forecasts": dict(selection_payload),
        "custom_load_forecasts": dict(custom_uploads),
        "errors": [],
    }

    new_selection = dict(display_selections)
    selected_manifests: Sequence[_ScenarioSelection] = []

    previous_selection: Mapping[str, Any] | None = None
    cached_manifests: Sequence[_ScenarioSelection] | None = None
    if st is not None:
        existing_selection = st.session_state.get("forecast_selections")
        if isinstance(existing_selection, Mapping):
            previous_selection = existing_selection
        existing_bundles = st.session_state.get("selected_forecast_bundles")
        if isinstance(existing_bundles, Sequence) and not isinstance(
            existing_bundles, (str, bytes, bytearray)
        ):
            cached_manifests = existing_bundles

    selection_changed = True
    if previous_selection is not None:
        selection_changed = dict(previous_selection) != new_selection

    if st is None or selection_changed:
        selected_manifests = _manifests_from_selection(
            display_selections,
            frame=frame,
            base_path=base_path,
        )
    elif cached_manifests is not None:
        selected_manifests = cached_manifests

    try:
        bundles = select_forecast_bundles(
            selection_payload,
            base_path=Path(base_path),
            frame=(
                _FrameProxy(selection_frame)
                if isinstance(selection_frame, pd.DataFrame)
                else selection_frame
            ),
        )
    except RuntimeError:
        bundles = selected_manifests
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.debug("Forecast bundle selection helper failed", exc_info=True)
        bundles = selected_manifests

    if bundles is not None:
        selected_manifests = list(bundles)

    if st is not None:
        _set_state_if_changed("forecast_selections", dict(selection_payload))
        _set_state_if_changed("forecast_iso_labels", dict(iso_display_labels))
        if selection_changed:
            _set_state_if_changed("selected_forecast_bundles", list(selected_manifests))
        _set_state_if_changed("custom_load_forecasts", dict(custom_uploads))

    return DemandModuleSettings(
        enabled=bool(selection_payload or custom_uploads),
        curve_by_region={},
        forecast_by_region={},
        load_forecasts=dict(selection_payload),
        custom_load_forecasts=dict(custom_uploads),
        errors=[],
    )



def _render_carbon_policy_section(
    container: Any,
    run_config: dict[str, Any],
    *,
    years: Iterable[int] | None = None,
    region_options: Iterable[Any] | None = None,
    lock_inputs: bool = False,
) -> CarbonModuleSettings:
    modules = run_config.setdefault("modules", {})
    defaults = modules.get("carbon_policy", {}) or {}
    price_defaults = modules.get("carbon_price", {}) or {}
    dispatch_defaults = modules.get("electricity_dispatch", {}) or {}

    raw_mode_default = str(dispatch_defaults.get("mode", "single")).strip().lower()
    single_region_dispatch = raw_mode_default != "network"
    if st is not None:
        dispatch_mode_state = st.session_state.get("dispatch_mode")
        if isinstance(dispatch_mode_state, str):
            single_region_dispatch = dispatch_mode_state.strip().lower() != "networked"

    # -------------------------
    # Defaults
    # -------------------------
    enabled_default = bool(defaults.get("enabled", True))
    enable_floor_default = bool(defaults.get("enable_floor", True))
    enable_ccr_default = bool(defaults.get("enable_ccr", True))
    ccr1_default = bool(defaults.get("ccr1_enabled", True))
    ccr2_default = bool(defaults.get("ccr2_enabled", True))
    ccr1_price_default = _coerce_optional_float(defaults.get("ccr1_price"))
    ccr2_price_default = _coerce_optional_float(defaults.get("ccr2_price"))
    ccr1_escalator_default = _coerce_float(defaults.get("ccr1_escalator_pct"), 0.0)
    ccr2_escalator_default = _coerce_float(defaults.get("ccr2_escalator_pct"), 0.0)
    banking_default = bool(defaults.get("allowance_banking_enabled", True))
    bank_default = _coerce_float(defaults.get("bank0", 0.0), default=0.0)

    coverage_default = _normalize_coverage_selection(
        defaults.get("coverage_regions", ["All"])
    )

    control_default_raw = defaults.get("control_period_years")
    try:
        control_default = int(control_default_raw)
    except (TypeError, ValueError):
        control_default = 3
    control_override_default = control_default_raw is not None

    # -------------------------
    # Price defaults
    # -------------------------
    price_enabled_default = bool(price_defaults.get("enabled", False))
    price_value_raw = price_defaults.get("price_per_ton", price_defaults.get("price", 0.0))
    price_default = _coerce_float(price_value_raw, default=0.0)
    price_schedule_default = _normalize_price_schedule(price_defaults.get("price_schedule"))
    price_escalator_default = _coerce_float(price_defaults.get("price_escalator_pct", 0.0), 0.0)

    allowance_defaults = modules.get("allowance_market", {}) or {}
    cap_schedule_default = _normalize_price_schedule(defaults.get("cap_schedule"))
    if not cap_schedule_default and isinstance(allowance_defaults, Mapping):
        cap_schedule_default = _normalize_price_schedule(allowance_defaults.get("cap"))

    cap_start_default = _coerce_float(defaults.get("cap_start_value"), 0.0)
    if cap_start_default <= 0.0 and cap_schedule_default:
        try:
            first_year = next(iter(cap_schedule_default))
        except StopIteration:
            cap_start_default = 0.0
        else:
            cap_start_default = float(cap_schedule_default[first_year])
    
    # If no cap is explicitly set, use a very high default (100M tons) to allow fossil dispatch
    # This represents an "effectively unbounded" scenario for typical regions
    if cap_start_default <= 0.0:
        cap_start_default = 100_000_000.0  # 100 million tons
    cap_reduction_mode_default = str(defaults.get("cap_reduction_mode", "percent")).strip().lower()
    if cap_reduction_mode_default not in {"percent", "fixed"}:
        cap_reduction_mode_default = "percent"
    cap_reduction_value_default = _coerce_float(defaults.get("cap_reduction_value"), 0.0)

    floor_schedule_default = _normalize_price_schedule(
        defaults.get("floor_schedule")
    )
    if not floor_schedule_default and isinstance(allowance_defaults, Mapping):
        floor_schedule_default = _normalize_price_schedule(
            allowance_defaults.get("floor")
        )

    floor_value_default = price_floor.parse_currency_value(
        defaults.get("floor_value"), 0.0
    )
    floor_mode_default = str(
        defaults.get("floor_escalator_mode", "fixed")
    ).strip().lower()
    if floor_mode_default not in {"fixed", "percent"}:
        floor_mode_default = "fixed"

    raw_floor_escalator_default = defaults.get("floor_escalator_value")
    if floor_mode_default == "percent":
        floor_escalator_default = price_floor.parse_percentage_value(
            raw_floor_escalator_default, 0.0
        )
    else:
        floor_escalator_default = price_floor.parse_currency_value(
            raw_floor_escalator_default, 0.0
        )

    inferred_floor = price_floor.infer_parameters(
        floor_schedule_default,
        default_base=floor_value_default,
        default_type=floor_mode_default,
        default_escalator=floor_escalator_default,
    )
    floor_value_default = inferred_floor.base_value
    floor_mode_default = inferred_floor.escalation_type
    if floor_mode_default == "percent":
        floor_escalator_default = price_floor.parse_percentage_value(
            raw_floor_escalator_default, inferred_floor.escalation_value
        )
    else:
        floor_escalator_default = price_floor.parse_currency_value(
            raw_floor_escalator_default, inferred_floor.escalation_value
        )

    def _extend_years(source: Iterable[Any] | None) -> None:
        if isinstance(source, Iterable) and not isinstance(source, (str, bytes, Mapping)):
            for entry in source:
                try:
                    year_candidates.append(int(entry))
                except (TypeError, ValueError):
                    continue

    year_candidates: list[int] = []
    _extend_years(years)
    if not year_candidates:
        raw_years = run_config.get("years")
        _extend_years(raw_years if isinstance(raw_years, Iterable) else None)
    if not year_candidates:
        start_raw = run_config.get("start_year")
        end_raw = run_config.get("end_year")
        start_val: int | None = None
        end_val: int | None = None
        try:
            start_val = int(start_raw)
        except (TypeError, ValueError):
            start_val = None
        try:
            end_val = int(end_raw) if end_raw is not None else None
        except (TypeError, ValueError):
            end_val = None
        if start_val is not None:
            if end_val is None:
                end_val = start_val
            step = 1 if end_val >= start_val else -1
            year_candidates.extend(range(start_val, end_val + step, step))
        elif end_val is not None:
            year_candidates.append(end_val)

    active_years = sorted(dict.fromkeys(year_candidates))

    cap_start_default_int = int(round(cap_start_default)) if cap_start_default > 0 else 0
    cap_percent_default = (
        float(cap_reduction_value_default) if cap_reduction_mode_default == "percent" else 0.0
    )
    cap_fixed_default = (
        float(cap_reduction_value_default) if cap_reduction_mode_default == "fixed" else 0.0
    )

    locked = bool(lock_inputs)

    # -------------------------
    # Coverage value map
    # -------------------------
    discovered_regions = _discovered_region_names()

    coverage_value_map: dict[str, Any] = {
        _ALL_REGIONS_LABEL: "All",
        "All": "All",
    }
    default_region_id = _get_default_region_id()
    for label in coverage_default:
        coverage_value_map.setdefault(label, canonical_region_value(label))
    if region_options is not None:
        for entry in region_options:
            label = canonical_region_label(entry)
            coverage_value_map.setdefault(label, canonical_region_value(entry))
    try:
        alias_lookup = region_alias_map()
    except Exception:
        alias_lookup = {}
    for alias, region_id in alias_lookup.items():
        coverage_value_map.setdefault(alias, region_id)
    for region_id in sorted(
        DEFAULT_REGION_METADATA,
        key=lambda value: canonical_region_label(value).lower(),
    ):
        label = region_display_label(region_id)
        coverage_value_map.setdefault(label, region_id)
        coverage_value_map.setdefault(str(region_id), region_id)
    for region_id in discovered_regions:
        coverage_value_map.setdefault(region_id, region_id)
        coverage_value_map.setdefault(region_id.lower(), region_id)

    # -------------------------
    # Coverage / Regions
    # -------------------------
    region_labels: list[str] = []
    if region_options is not None:
        for entry in region_options:
            label = canonical_region_label(entry).strip() or canonical_region_label(
                default_region_id
            )
            if label not in region_labels:
                region_labels.append(label)
    for label in coverage_default:
        if label != _ALL_REGIONS_LABEL and label not in region_labels:
            region_labels.append(label)
    for region_id in sorted(
        DEFAULT_REGION_METADATA,
        key=lambda value: canonical_region_label(value).lower(),
    ):
        label = region_display_label(region_id)
        if label not in region_labels:
            region_labels.append(label)
    for region_id in discovered_regions:
        if region_id not in region_labels:
            region_labels.append(region_id)
    if not region_labels:
        region_labels = [canonical_region_label(default_region_id)]

    coverage_choices = [_ALL_REGIONS_LABEL] + sorted(region_labels, key=str)
    coverage_default_display = _coverage_default_display(coverage_default, coverage_choices)

    coverage_regions = list(coverage_default)

    default_ccr1_price_value = float(ccr1_price_default) if ccr1_price_default is not None else 0.0
    default_ccr2_price_value = float(ccr2_price_default) if ccr2_price_default is not None else 0.0
    default_ccr1_escalator_value = float(ccr1_escalator_default)
    default_ccr2_escalator_value = float(ccr2_escalator_default)
    default_control_year_value = int(control_default if control_default > 0 else 3)
    default_price_value = float(price_default if price_default >= 0.0 else 0.0)

    carbon_session_defaults = CarbonSessionState(
        enabled=bool(enabled_default),
        price_enabled=bool(price_enabled_default),
        enable_floor=bool(enable_floor_default),
        enable_ccr=bool(enable_ccr_default),
        ccr1_enabled=bool(ccr1_default),
        ccr2_enabled=bool(ccr2_default),
        banking_enabled=bool(banking_default),
        bank0=float(bank_default),
        control_override=bool(control_override_default),
        control_years=int(default_control_year_value),
        ccr1_price=float(default_ccr1_price_value),
        ccr1_escalator=float(default_ccr1_escalator_value),
        ccr2_price=float(default_ccr2_price_value),
        ccr2_escalator=float(default_ccr2_escalator_value),
        coverage_regions=list(coverage_default_display),
        price_value=float(default_price_value),
        price_escalator=float(price_escalator_default),
        cap_start=int(cap_start_default_int),
        cap_reduction_mode=str(cap_reduction_mode_default),
        cap_reduction_percent=float(cap_percent_default),
        cap_reduction_fixed=float(cap_fixed_default),
        floor_value_input=f"{floor_value_default:.2f}",
        floor_mode=str(floor_mode_default),
        floor_escalator_input=f"{floor_escalator_default:.2f}",
    )

    if st is not None:
        carbon_session_defaults.apply_defaults(st.session_state)
        if locked:
            carbon_session_defaults.override_for_lock(st.session_state)

    # -------------------------
    # Session defaults and change tracking
    # -------------------------
    bank_value_default = float(bank_default)
    if st is not None:
        price_escalator_default = float(
            st.session_state.get("carbon_price_escalator", price_escalator_default)
        )
        cap_start_default_int = int(
            st.session_state.get("carbon_cap_start", cap_start_default_int)
        )
        cap_reduction_mode_default = str(
            st.session_state.get("carbon_cap_reduction_mode", cap_reduction_mode_default)
        ).strip().lower()
        if cap_reduction_mode_default not in {"percent", "fixed"}:
            cap_reduction_mode_default = "percent"
            st.session_state["carbon_cap_reduction_mode"] = cap_reduction_mode_default
        cap_percent_default = float(
            st.session_state.get("carbon_cap_reduction_percent", cap_percent_default)
        )
        cap_fixed_default = float(
            st.session_state.get("carbon_cap_reduction_fixed", cap_fixed_default)
        )
        floor_value_text_default = str(
            st.session_state.get("carbon_floor_value_input", f"{floor_value_default:.2f}")
        )
        floor_value_default = price_floor.parse_currency_value(
            floor_value_text_default, floor_value_default
        )
        floor_mode_default = str(
            st.session_state.get("carbon_floor_mode", floor_mode_default)
        ).strip().lower()
        if floor_mode_default not in {"fixed", "percent"}:
            floor_mode_default = "fixed"
            st.session_state["carbon_floor_mode"] = floor_mode_default
        floor_escalator_text_default = str(
            st.session_state.get(
                "carbon_floor_escalator_input", f"{floor_escalator_default:.2f}"
            )
        )
        if floor_mode_default == "percent":
            floor_escalator_default = price_floor.parse_percentage_value(
                floor_escalator_text_default, floor_escalator_default
            )
        else:
            floor_escalator_default = price_floor.parse_currency_value(
                floor_escalator_text_default, floor_escalator_default
            )
        enabled_default = bool(st.session_state.get("carbon_enable", enabled_default))
        price_enabled_default = bool(
            st.session_state.get("carbon_price_enable", price_enabled_default)
        )
        enable_floor_default = bool(st.session_state.get("carbon_floor", enable_floor_default))
        enable_ccr_default = bool(st.session_state.get("carbon_ccr", enable_ccr_default))
        ccr1_default = bool(st.session_state.get("carbon_ccr1", ccr1_default))
        ccr2_default = bool(st.session_state.get("carbon_ccr2", ccr2_default))
        banking_default = bool(st.session_state.get("carbon_banking", banking_default))
        bank_value_default = float(st.session_state.get("carbon_bank0", bank_default))
        control_override_default = bool(
            st.session_state.get("carbon_control_toggle", control_override_default)
        )
        default_control_year_value = int(
            st.session_state.get("carbon_control_years", default_control_year_value)
        )
        default_ccr1_price_value = float(
            st.session_state.get("carbon_ccr1_price", default_ccr1_price_value)
        )
        default_ccr1_escalator_value = float(
            st.session_state.get("carbon_ccr1_escalator", default_ccr1_escalator_value)
        )
        default_ccr2_price_value = float(
            st.session_state.get("carbon_ccr2_price", default_ccr2_price_value)
        )
        default_ccr2_escalator_value = float(
            st.session_state.get("carbon_ccr2_escalator", default_ccr2_escalator_value)
        )
        default_price_value = float(
            st.session_state.get("carbon_price_value", default_price_value)
        )
        coverage_default_display = list(
            st.session_state.get("carbon_coverage_regions", coverage_default_display)
        )
        coverage_default_display = _coverage_default_display(
            _normalize_coverage_selection(coverage_default_display),
            coverage_choices,
        )

    def _mark_last_changed(key: str) -> None:
        if st is None:
            return
        st.session_state["carbon_module_last_changed"] = key

    deep_pricing_allowed = bool(dispatch_defaults.get("deep_carbon_pricing", False))
    if st is not None:
        deep_pricing_allowed = bool(
            st.session_state.get("dispatch_deep_carbon", deep_pricing_allowed)
        )

    session_enabled_default = enabled_default
    session_price_default = price_enabled_default
    last_changed = None
    if st is not None:
        session_enabled_default = bool(
            st.session_state.get("carbon_enable", enabled_default)
        )
        session_price_default = bool(
            st.session_state.get("carbon_price_enable", price_enabled_default)
        )
        last_changed = st.session_state.get("carbon_module_last_changed")
        if session_enabled_default and session_price_default:
            if last_changed == "cap":
                session_price_default = False
            else:
                session_enabled_default = False
            st.session_state["carbon_enable"] = session_enabled_default
            st.session_state["carbon_price_enable"] = session_price_default


    # -------------------------
    # Cap vs Price toggles (mutually exclusive)
    # -------------------------
    cap_toggle_disabled = locked or session_price_default
    enabled = container.toggle(
        "Enable carbon cap",
        value=session_enabled_default,
        key="carbon_enable",
        on_change=lambda: _mark_last_changed("cap"),
        disabled=cap_toggle_disabled,
    )
    price_toggle_disabled = locked or bool(enabled)
    price_enabled = container.toggle(
        "Enable carbon price",
        value=session_price_default,
        key="carbon_price_enable",
        on_change=lambda: _mark_last_changed("price"),
        disabled=price_toggle_disabled,
    )

    if price_enabled and enabled:
        enabled = False
    elif enabled and not price_enabled:
        price_enabled = False

    if locked:
        price_enabled = bool(price_enabled_default)
        enabled = bool(enabled_default and not price_enabled)

    cap_schedule: dict[int, float] = dict(cap_schedule_default)
    cap_start_value = float(cap_start_default_int)
    cap_reduction_mode = cap_reduction_mode_default
    cap_reduction_value = (
        cap_percent_default if cap_reduction_mode_default == "percent" else cap_fixed_default
    )
    price_schedule: dict[int, float] = dict(price_schedule_default)
    price_escalator_value = float(price_escalator_default)

    coverage_panel_enabled = (enabled or price_enabled) and not locked
    with _sidebar_panel(container, coverage_panel_enabled) as coverage_panel:
        coverage_selection = coverage_panel.multiselect(
            "Regions covered by carbon policies",
            options=coverage_choices,
            default=coverage_default_display,
            disabled=((not (enabled or price_enabled)) or locked or single_region_dispatch),
            key="carbon_coverage_regions",
            help=(
                "Select the regions subject to the carbon cap or carbon price. "
                "Choose “All regions” to apply the policy across every region."
            ),
        )
        if single_region_dispatch and (enabled or price_enabled):
            coverage_panel.info(
                "Single-region dispatch enforces a uniform carbon coverage flag across the selected region."
            )
        coverage_regions = _normalize_coverage_selection(
            coverage_selection or coverage_default_display
        )

    # -------------------------
    # Carbon Cap Panel
    # -------------------------
    floor_value = float(floor_value_default)
    floor_mode = str(floor_mode_default)
    floor_escalator_value = float(floor_escalator_default)
    floor_schedule: dict[int, float] = dict(floor_schedule_default)

    with _sidebar_panel(container, enabled and not locked) as cap_panel:
        enable_floor = cap_panel.toggle(
            "Enable price floor",
            value=enable_floor_default,
            key="carbon_floor",
            disabled=(not enabled) or locked,
        )
        if enable_floor:
            floor_value_text = cap_panel.text_input(
                "Price floor ($/ton)",
                value=(st.session_state.get("carbon_floor_value_input") if st is not None else f"{floor_value_default:.2f}"),
                key="carbon_floor_value_input",
                disabled=(not enabled) or locked,
                help="Specify the minimum auction clearing price. Values are rounded to two decimals.",
            )
            floor_value = price_floor.parse_currency_value(floor_value_text, floor_value_default)
            floor_mode = cap_panel.radio(
                "Floor escalates by",
                options=("fixed", "percent"),
                index=0 if floor_mode_default == "fixed" else 1,
                format_func=lambda option: (
                    "Fixed amount ($/ton per year)" if option == "fixed" else "Percent (% per year)"
                ),
                key="carbon_floor_mode",
                disabled=(not enabled) or locked,
            )
            escalator_label = (
                "Annual increase ($/ton)" if floor_mode == "fixed" else "Annual increase (%)"
            )
            escalator_value_default = (
                f"{floor_escalator_default:.2f}"
                if st is None
                else st.session_state.get("carbon_floor_escalator_input", f"{floor_escalator_default:.2f}")
            )
            floor_escalator_text = cap_panel.text_input(
                escalator_label,
                value=escalator_value_default,
                key="carbon_floor_escalator_input",
                disabled=(not enabled) or locked,
                help="Set to 0 for a constant floor across all modeled years.",
            )
            if floor_mode == "percent":
                floor_escalator_value = price_floor.parse_percentage_value(
                    floor_escalator_text, floor_escalator_default
                )
            else:
                floor_escalator_value = price_floor.parse_currency_value(
                    floor_escalator_text, floor_escalator_default
                )
            schedule_years = list(active_years)
            if not schedule_years:
                schedule_years = sorted(floor_schedule_default) if floor_schedule_default else []
            if not schedule_years:
                start_year_raw = run_config.get("start_year")
                try:
                    schedule_years = [int(start_year_raw)] if start_year_raw is not None else []
                except (TypeError, ValueError):
                    schedule_years = []
            floor_schedule = price_floor.build_schedule(
                schedule_years,
                floor_value,
                floor_mode,
                floor_escalator_value,
            )
        else:
            floor_schedule = {}
        enable_ccr = cap_panel.toggle(
            "Enable CCR",
            value=enable_ccr_default,
            key="carbon_ccr",
            disabled=(not enabled) or locked,
        )
        ccr1_enabled = cap_panel.toggle(
            "Enable CCR Tier 1",
            value=ccr1_default,
            key="carbon_ccr1",
            disabled=(not (enabled and enable_ccr)) or locked,
        )
        ccr2_enabled = cap_panel.toggle(
            "Enable CCR Tier 2",
            value=ccr2_default,
            key="carbon_ccr2",
            disabled=(not (enabled and enable_ccr)) or locked,
        )

        cap_start_input = cap_panel.number_input(
            "Starting carbon cap (tons)",
            min_value=0,
            value=int(cap_start_default_int),
            step=1000,
            format="%d",
            key="carbon_cap_start",
            disabled=(not enabled) or locked,
        )
        cap_start_value = float(cap_start_input)

        reduction_options = ("percent", "fixed")
        try:
            reduction_index = reduction_options.index(cap_reduction_mode_default)
        except ValueError:
            reduction_index = 0
        cap_reduction_mode = cap_panel.radio(
            "Annual cap adjustment",
            options=reduction_options,
            index=reduction_index,
            format_func=lambda option: (
                "Decrease by % of starting value" if option == "percent" else "Decrease by fixed amount"
            ),
            key="carbon_cap_reduction_mode",
            disabled=(not enabled) or locked,
        )

        if cap_reduction_mode == "percent":
            cap_reduction_percent = float(
                cap_panel.number_input(
                    "Annual reduction (% of starting cap)",
                    min_value=0.0,
                    value=float(cap_percent_default),
                    step=0.1,
                    format="%0.2f",
                    key="carbon_cap_reduction_percent",
                    disabled=(not enabled) or locked,
                )
            )
            cap_reduction_value = cap_reduction_percent
        else:
            cap_reduction_fixed = float(
                cap_panel.number_input(
                    "Annual reduction (tons)",
                    min_value=0.0,
                    value=float(cap_fixed_default),
                    step=1000.0,
                    format="%0.0f",
                    key="carbon_cap_reduction_fixed",
                    disabled=(not enabled) or locked,
                )
            )
            cap_reduction_value = cap_reduction_fixed

        schedule_years = active_years or list(cap_schedule_default.keys())
        if enabled and schedule_years:
            cap_schedule = _build_cap_reduction_schedule(
                cap_start_value,
                cap_reduction_mode,
                cap_reduction_value,
                schedule_years,
            )
        elif enabled:
            cap_schedule = dict(cap_schedule_default)
        else:
            cap_schedule = dict(cap_schedule_default)
        floor_value = float(floor_value_default)
        floor_mode = str(floor_mode_default)
        floor_escalator_value = float(floor_escalator_default)
        floor_schedule = dict(floor_schedule_default) if enable_floor else {}

        if enabled and enable_ccr and ccr1_enabled:
            default_price1 = float(ccr1_price_default) if ccr1_price_default is not None else 0.0
            ccr1_price_value = float(
                cap_panel.number_input(
                    "CCR Tier 1 trigger price ($/ton)",
                    min_value=0.0,
                    value=default_price1,
                    step=1.0,
                    format="%0.2f",
                    key="carbon_ccr1_price",
                    disabled=(not (enabled and enable_ccr and ccr1_enabled)) or locked,
                )
            )
            ccr1_escalator_value = float(
                cap_panel.number_input(
                    "CCR Tier 1 annual escalator (%)",
                    min_value=0.0,
                    value=float(ccr1_escalator_default),
                    step=0.1,
                    format="%0.2f",
                    key="carbon_ccr1_escalator",
                    disabled=(not (enabled and enable_ccr and ccr1_enabled)) or locked,
                )
            )
        else:
            ccr1_price_value = ccr1_price_default if ccr1_price_default is not None else None
            ccr1_escalator_value = float(ccr1_escalator_default)

        if enabled and enable_ccr and ccr2_enabled:
            default_price2 = float(ccr2_price_default) if ccr2_price_default is not None else 0.0
            ccr2_price_value = float(
                cap_panel.number_input(
                    "CCR Tier 2 trigger price ($/ton)",
                    min_value=0.0,
                    value=default_price2,
                    step=1.0,
                    format="%0.2f",
                    key="carbon_ccr2_price",
                    disabled=(not (enabled and enable_ccr and ccr2_enabled)) or locked,
                )
            )
            ccr2_escalator_value = float(
                cap_panel.number_input(
                    "CCR Tier 2 annual escalator (%)",
                    min_value=0.0,
                    value=float(ccr2_escalator_default),
                    step=0.1,
                    format="%0.2f",
                    key="carbon_ccr2_escalator",
                    disabled=(not (enabled and enable_ccr and ccr2_enabled)) or locked,
                )
            )
        else:
            ccr2_price_value = ccr2_price_default if ccr2_price_default is not None else None
            ccr2_escalator_value = float(ccr2_escalator_default)

        banking_enabled = cap_panel.toggle(
            "Enable allowance banking",
            value=banking_default,
            key="carbon_banking",
            disabled=(not enabled) or locked,
        )

        if banking_enabled:
            initial_bank = float(
                cap_panel.number_input(
                    "Initial allowance bank (tons)",
                    min_value=0.0,
                    value=float(bank_value_default if bank_value_default >= 0.0 else 0.0),
                    step=1000.0,
                    format="%f",
                    key="carbon_bank0",
                    disabled=(not (enabled and banking_enabled)) or locked,
                )
            )
        else:
            initial_bank = 0.0

        control_override = cap_panel.toggle(
            "Override control period",
            value=control_override_default,
            key="carbon_control_toggle",
            disabled=(not enabled) or locked,
        )
        control_period_value = cap_panel.number_input(
            "Control period length (years)",
            min_value=1,
            value=int(control_default if control_default > 0 else 3),
            step=1,
            format="%d",
            key="carbon_control_years",
            disabled=(not (enabled and control_override)) or locked,
        )
        control_period_years = (
            _sanitize_control_period(control_period_value)
            if enabled and control_override
            else None
        )

    # -------------------------
    # Carbon Price Panel
    # -------------------------
    with _sidebar_panel(container, price_enabled and not locked) as price_panel:
        price_per_ton = price_panel.number_input(
            "Carbon price ($/ton)",
            min_value=0.0,
            value=float(price_default if price_default >= 0.0 else 0.0),
            step=1.0,
            format="%0.2f",
            key="carbon_price_value",
            disabled=(not price_enabled) or locked,
        )
        price_escalator_value = float(
            price_panel.number_input(
                "Carbon price escalator (% per year)",
                min_value=0.0,
                value=float(price_escalator_default if price_escalator_default >= 0.0 else 0.0),
                step=0.1,
                format="%0.2f",
                key="carbon_price_escalator",
                disabled=(not price_enabled) or locked,
            )
        )

        schedule_years = active_years or list(price_schedule_default.keys())
        if price_enabled and schedule_years:
            price_schedule = _build_price_escalator_schedule(
                price_per_ton,
                price_escalator_value,
                schedule_years,
            )
        elif price_enabled:
            price_schedule = price_schedule_default.copy()
        else:
            price_schedule = {}

    if locked:
        enabled = bool(enabled_default)
        price_enabled = bool(price_enabled_default)
        enable_floor = bool(enable_floor_default)
        enable_ccr = bool(enable_ccr_default)
        ccr1_enabled = bool(ccr1_default)
        ccr2_enabled = bool(ccr2_default)
        banking_enabled = bool(banking_default)
        if banking_enabled:
            initial_bank = float(bank_value_default if bank_value_default >= 0.0 else 0.0)
        else:
            initial_bank = 0.0
        control_override = bool(control_override_default)
        if enabled and control_override:
            control_period_years = _sanitize_control_period(default_control_year_value)
        else:
            control_period_years = None
        coverage_regions = list(coverage_default)
        price_per_ton = default_price_value
        price_escalator_value = float(price_escalator_default)
        price_schedule = price_schedule_default.copy() if price_enabled else {}
        cap_start_value = float(cap_start_default_int)
        cap_reduction_mode = cap_reduction_mode_default
        cap_reduction_value = (
            cap_percent_default if cap_reduction_mode_default == "percent" else cap_fixed_default
        )
        cap_schedule = dict(cap_schedule_default)

    # -------------------------
    # Errors and Return
    # -------------------------
    if not enabled:
        cap_schedule = {}
        floor_schedule = {}

    errors: list[str] = []
    deep_enabled = bool(
        modules.get("electricity_dispatch", {}).get("deep_carbon_pricing", False)
    )
    if st is not None:
        deep_enabled = bool(st.session_state.get("dispatch_deep_carbon", deep_enabled))
    if enabled and price_enabled and not deep_enabled:
        errors.append("Cannot enable both carbon cap and carbon price simultaneously.")

    policy_region_values: list[str] = []
    if coverage_regions != ["All"]:
        for label in coverage_regions:
            resolved = coverage_value_map.get(label, canonical_region_value(label))
            if isinstance(resolved, str) and resolved.lower() in {"all", "all regions"}:
                policy_region_values = []
                break
            if isinstance(resolved, str):
                normalized = resolved.strip()
            elif isinstance(resolved, bool):
                normalized = str(int(resolved))
            elif isinstance(resolved, (int, float)):
                normalized = str(int(resolved))
            else:
                normalized = str(resolved).strip()
            if not normalized:
                continue
            policy_region_values.append(normalized)

    carbon_module = modules.setdefault("carbon_policy", {})
    carbon_module.update(
        {
            "enabled": bool(enabled),
            "enable_floor": bool(enabled and enable_floor),
            "enable_ccr": bool(enabled and enable_ccr),
            "ccr1_enabled": bool(enabled and enable_ccr and ccr1_enabled),
            "ccr2_enabled": bool(enabled and enable_ccr and ccr2_enabled),
            "allowance_banking_enabled": bool(enabled and banking_enabled),
            "coverage_regions": list(coverage_regions),
            "cap_start_value": float(cap_start_value),
            "cap_reduction_mode": str(cap_reduction_mode),
            "cap_reduction_value": float(cap_reduction_value),
            "floor_value": float(floor_value),
            "floor_escalator_mode": str(floor_mode),
            "floor_escalator_value": float(floor_escalator_value),
        }
    )

    if control_period_years is None or not enabled:
        carbon_module["control_period_years"] = None
    else:
        carbon_module["control_period_years"] = int(control_period_years)

    if enabled and banking_enabled:
        carbon_module["bank0"] = float(initial_bank)
    else:
        carbon_module["bank0"] = 0.0

    if enabled and cap_schedule:
        carbon_module["cap_schedule"] = dict(cap_schedule)
    else:
        carbon_module.pop("cap_schedule", None)

    if enabled and enable_floor and floor_schedule:
        carbon_module["floor_schedule"] = dict(floor_schedule)
    else:
        carbon_module.pop("floor_schedule", None)
    if policy_region_values:
        carbon_module["regions"] = list(policy_region_values)
    else:
        carbon_module.pop("regions", None)

    allowance_market_module = modules.setdefault("allowance_market", {})
    if enabled and cap_schedule:
        allowance_market_module["cap"] = {
            int(year): float(value) for year, value in cap_schedule.items()
        }
    elif not enabled:
        allowance_market_module.pop("cap", None)

    if enabled and enable_floor and floor_schedule:
        allowance_market_module["floor"] = {
            int(year): float(value) for year, value in floor_schedule.items()
        }
    elif not enabled or not enable_floor:
        allowance_market_module.pop("floor", None)
    price_module = modules.setdefault("carbon_price", {})
    price_module["enabled"] = bool(price_enabled)
    if price_enabled:
        price_module["price_per_ton"] = float(price_per_ton)
        price_module["price_escalator_pct"] = float(price_escalator_value)
        if price_schedule:
            price_module["price_schedule"] = dict(price_schedule)
        else:
            price_module.pop("price_schedule", None)
        price_module["coverage_regions"] = list(coverage_regions)
        if policy_region_values:
            price_module["regions"] = list(policy_region_values)
        else:
            price_module.pop("regions", None)
    else:
        price_module["price_escalator_pct"] = float(price_escalator_value)
        price_module.pop("price_schedule", None)
        price_module.pop("price", None)
        if "price_per_ton" in price_module:
            price_module["price_per_ton"] = float(price_per_ton)
        price_module.pop("coverage_regions", None)
        price_module.pop("regions", None)

    return CarbonModuleSettings(
        enabled=enabled,
        price_enabled=price_enabled,
        enable_floor=enable_floor,
        enable_ccr=enable_ccr,
        ccr1_enabled=ccr1_enabled,
        ccr2_enabled=ccr2_enabled,
        ccr1_price=ccr1_price_value if 'ccr1_price_value' in locals() else ccr1_price_default,
        ccr2_price=ccr2_price_value if 'ccr2_price_value' in locals() else ccr2_price_default,
        ccr1_escalator_pct=ccr1_escalator_value if 'ccr1_escalator_value' in locals() else float(ccr1_escalator_default),
        ccr2_escalator_pct=ccr2_escalator_value if 'ccr2_escalator_value' in locals() else float(ccr2_escalator_default),
        banking_enabled=banking_enabled,
        coverage_regions=coverage_regions,
        control_period_years=control_period_years,
        price_per_ton=float(price_per_ton),
        price_escalator_pct=float(price_escalator_value),
        initial_bank=initial_bank,
        cap_regions=policy_region_values,
        cap_start_value=float(cap_start_value),
        cap_reduction_mode=str(cap_reduction_mode),
        cap_reduction_value=float(cap_reduction_value),
        cap_schedule=cap_schedule,
        floor_value=float(floor_value),
        floor_escalator_mode=str(floor_mode),
        floor_escalator_value=float(floor_escalator_value),
        floor_schedule=floor_schedule,
        price_schedule=price_schedule,
        errors=errors,
    )


# -------------------------
# Dispatch UI
# -------------------------
def _render_dispatch_section(
    container: Any,
    run_config: dict[str, Any],
    frames: FramesType | None,
) -> DispatchModuleSettings:
    modules = run_config.setdefault("modules", {})
    defaults = modules.get("electricity_dispatch", {})
    enabled_default = bool(defaults.get("enabled", False))
    mode_default = str(defaults.get("mode", "single")).lower()
    if mode_default not in {"single", "network"}:
        mode_default = "single"
    capacity_default = bool(defaults.get("capacity_expansion", True))
    reserve_default = bool(defaults.get("reserve_margins", True))
    deep_default = bool(defaults.get("deep_carbon_pricing", False))

    enabled = container.toggle(
        "Enable electricity dispatch",
        value=enabled_default,
        key="dispatch_enable",
    )

    mode_value = mode_default
    capacity_expansion = capacity_default
    reserve_margins = reserve_default
    deep_carbon_pricing = deep_default
    errors: list[str] = []

    mode_options = {"single": "Single region", "network": "Networked"}

    with _sidebar_panel(container, enabled) as panel:
        mode_label = mode_options.get(mode_default, mode_options["single"])
        mode_selection = panel.selectbox(
            "Dispatch topology",
            options=list(mode_options.values()),
            index=list(mode_options.values()).index(mode_label),
            disabled=not enabled,
            key="dispatch_mode",
        )
        mode_value = "network" if mode_selection == mode_options["network"] else "single"
        capacity_expansion = panel.checkbox(
            "Enable capacity expansion",
            value=capacity_default,
            disabled=not enabled,
            key="dispatch_capacity",
        )
        reserve_margins = panel.checkbox(
            "Enforce reserve margins",
            value=reserve_default,
            disabled=not enabled,
            key="dispatch_reserve",
        )
        default_deep_value = deep_default
        if st is not None:
            default_deep_value = bool(
                st.session_state.get("dispatch_deep_carbon", deep_default)
            )
        deep_carbon_pricing = panel.toggle(
            "Enable deep carbon pricing",
            value=default_deep_value,
            disabled=not enabled,
            key="dispatch_deep_carbon",
            help=(
                "Allows simultaneous use of allowance clearing prices and exogenous "
                "carbon prices when solving dispatch."
            ),
        )

        if enabled:
            if not _dispatch_backend_available():
                panel.warning(
                    "Electricity dispatch solver dependencies are missing. "
                    "Install SciPy (`pip install scipy`) or PuLP (`pip install pulp`) before running a scenario."
                )

            if frames is None:
                message = "Dispatch requires demand and unit data, but no frames are available."
                panel.error(message)
                errors.append(message)
            else:
                try:
                    demand_df = frames.demand()
                    units_df = frames.units()
                except Exception as exc:
                    message = f"Dispatch data unavailable: {exc}"
                    panel.error(message)
                    errors.append(message)
                else:
                    if demand_df.empty or units_df.empty:
                        message = "Dispatch requires non-empty demand and unit tables."
                        panel.error(message)
                        errors.append(message)
        else:
            mode_value = mode_default
            capacity_expansion = False
            reserve_margins = False
            deep_carbon_pricing = False

    if not enabled:
        mode_value = mode_value or "single"
        deep_carbon_pricing = False

    modules["electricity_dispatch"] = {
        "enabled": bool(enabled),
        "mode": mode_value or "single",
        "capacity_expansion": bool(capacity_expansion),
        "reserve_margins": bool(reserve_margins),
        "deep_carbon_pricing": bool(deep_carbon_pricing),
    }

    return DispatchModuleSettings(
        enabled=bool(enabled),
        mode=mode_value or "single",
        capacity_expansion=bool(capacity_expansion),
        reserve_margins=bool(reserve_margins),
        deep_carbon_pricing=bool(deep_carbon_pricing),
        errors=errors,
    )


# -------------------------
# Incentives UI
# -------------------------
def _coerce_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        value = text
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool_flag(value: Any, default: bool = True) -> bool:
    if value in (None, ""):
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1", "on"}:
            return True
        if normalized in {"false", "f", "no", "n", "0", "off"}:
            return False
    return bool(default)


def _parse_years_field(
    value: Any,
    *,
    valid_years: set[int] | None = None,
) -> tuple[list[int], list[str], list[int]]:
    if value in (None, ""):
        return [], [], []

    text = str(value).strip()
    if not text:
        return [], [], []

    normalized = text.translate({ord(char): None for char in "[]{}()"})
    tokens = [token for token in re.split(r"[;,\s]+", normalized) if token]

    parsed_years: list[int] = []
    invalid_tokens: list[str] = []
    out_of_range: list[int] = []

    valid_set = {int(year) for year in valid_years} if valid_years else set()

    for token in tokens:
        token_str = token.strip()
        if not token_str:
            continue

        if "-" in token_str:
            start_text, end_text = token_str.split("-", 1)
            try:
                start_year = int(start_text.strip())
                end_year = int(end_text.strip())
            except (TypeError, ValueError):
                invalid_tokens.append(token_str)
                continue

            step = 1 if end_year >= start_year else -1
            for year in range(start_year, end_year + step, step):
                if valid_set and year not in valid_set:
                    out_of_range.append(year)
                else:
                    parsed_years.append(year)
            continue

        try:
            year_int = int(token_str)
        except (TypeError, ValueError):
            invalid_tokens.append(token_str)
            continue

        if valid_set and year_int not in valid_set:
            out_of_range.append(year_int)
        else:
            parsed_years.append(year_int)

    parsed_years = sorted({int(year) for year in parsed_years})
    out_of_range = sorted({int(year) for year in out_of_range if year not in parsed_years})

    return parsed_years, invalid_tokens, out_of_range


def _coerce_year_values(value: Any) -> list[int]:
    """Return integers discovered within ``value``.

    The configuration files that feed the GUI frequently represent year
    selections using a mix of primitives, strings, and nested mappings. This
    helper walks the supported shapes and extracts any integers that resemble
    simulation years. The traversal is intentionally shallow – it only inspects
    common keys used by our configuration payloads – in order to avoid
    accidentally iterating over very large dictionaries while still tolerating
    the variations seen in existing fixtures.
    """

    if value in (None, ""):
        return []

    if isinstance(value, bool):
        return []

    if isinstance(value, (int, float)):
        try:
            year = int(value)
        except (TypeError, ValueError, OverflowError):
            return []
        return [year]

    if isinstance(value, str):
        parsed, _, _ = _parse_years_field(value)
        return parsed

    if isinstance(value, Mapping):
        years: list[int] = []
        for key in ("year", "start", "end", "first", "last"):
            if key in value:
                years.extend(_coerce_year_values(value.get(key)))
        for collection_key in ("years", "values", "items", "records", "data"):
            if collection_key in value:
                years.extend(_coerce_year_values(value.get(collection_key)))
        if not years:
            for key in value.keys():
                years.extend(_coerce_year_values(key))
        return years

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        years: list[int] = []
        for entry in value:
            years.extend(_coerce_year_values(entry))
        return years

    return []


def _coerce_year(value: Any) -> int | None:
    """Return ``value`` coerced to an integer year when possible."""

    if value in (None, ""):
        return None

    if isinstance(value, bool):
        return None

    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        if isinstance(value, str):
            parsed, _, _ = _parse_years_field(value)
            return parsed[0] if parsed else None
        return None


def _years_from_config(config: Mapping[str, Any]) -> list[int]:
    """Extract the distinct simulation years present in ``config``."""

    if not isinstance(config, Mapping):
        return []

    years_raw = config.get("years")
    years = sorted({int(year) for year in _coerce_year_values(years_raw)})

    if years:
        return years

    start_year = _coerce_year(config.get("start_year"))
    end_year = _coerce_year(config.get("end_year"))

    if start_year is None and end_year is None:
        return []

    if start_year is None:
        start_year = end_year
    if end_year is None:
        end_year = start_year

    if start_year is None or end_year is None:
        return []

    step = 1 if end_year >= start_year else -1
    return sorted({int(year) for year in range(start_year, end_year + step, step)})


def _select_years(
    available_years: Iterable[Any],
    start_year: int | None,
    end_year: int | None,
) -> list[int]:
    """Return a normalized list of simulation years for the requested span."""

    normalized = sorted({int(year) for year in _coerce_year_values(available_years)})

    if start_year is None and end_year is None:
        return normalized

    start = start_year if start_year is not None else (normalized[0] if normalized else None)
    end = end_year if end_year is not None else (normalized[-1] if normalized else None)

    start = _coerce_year(start)
    end = _coerce_year(end)

    if start is None or end is None:
        return normalized

    if end < start:
        raise ValueError("end_year must be greater than or equal to start_year")

    return list(range(start, end + 1))


def _data_editor_records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []

    if hasattr(value, "to_dict"):
        try:
            records = value.to_dict("records")  # type: ignore[call-arg]
        except Exception:
            records = None
        if isinstance(records, list):
            return [dict(entry) for entry in records if isinstance(entry, Mapping)]

    if isinstance(value, Mapping):
        return [dict(value)]

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        records: list[dict[str, Any]] = []
        for entry in value:
            if isinstance(entry, Mapping):
                records.append(dict(entry))
        if records:
            return records

    return []


def _simulation_years_from_config(config: Mapping[str, Any]) -> list[int]:
    try:
        base_years = _years_from_config(config)
    except Exception:
        base_years = []

    start_raw = config.get("start_year")
    end_raw = config.get("end_year")

    def _to_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    start_year = _to_int(start_raw)
    end_year = _to_int(end_raw)

    years: list[int] = []

    if base_years:
        try:
            years = _select_years(base_years, start_year, end_year)
        except Exception:
            years = [int(year) for year in base_years]
    else:
        if start_year is not None and end_year is not None:
            step = 1 if end_year >= start_year else -1
            years = list(range(start_year, end_year + step, step))
        elif start_year is not None:
            years = [start_year]
        elif end_year is not None:
            years = [end_year]

    return sorted({int(year) for year in years})


def _render_incentives_section(
    container: Any,
    run_config: dict[str, Any],
    frames: FramesType | None,
) -> IncentivesModuleSettings:
    modules = run_config.setdefault("modules", {})
    defaults = modules.get("incentives", {})
    enabled_default = bool(defaults.get("enabled", False))

    def _coerce_editor_dimension(value: Any) -> int | None:
        """Return ``value`` coerced to a positive integer when possible."""

        if value is None:
            return None

        if isinstance(value, bool):  # Guard against ``True``/``False`` being coerced to 1/0.
            return None

        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            value = candidate

        try:
            dimension = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

        return dimension if dimension > 0 else None

    def _coerce_editor_width(value: Any) -> str | int | None:
        """Return a sanitized width value for ``st.data_editor``."""

        if value is None:
            return None

        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            lowered = candidate.lower()
            if lowered in {"stretch", "auto"}:
                return "stretch" if lowered == "stretch" else "auto"
            value = candidate

        return _coerce_editor_dimension(value)

    def _coerce_editor_num_rows(value: Any) -> str | int:
        """Return a sanitized ``num_rows`` value for ``st.data_editor``."""

        if value is None:
            return "dynamic"

        if isinstance(value, str):
            candidate = value.strip()
            if not candidate or candidate.lower() == "dynamic":
                return "dynamic"
            value = candidate

        if isinstance(value, bool):
            return "dynamic"

        try:
            rows = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "dynamic"

        return rows if rows > 0 else "dynamic"

    def _editor_setting(key: str) -> Any:
        for source in (run_config.get("electricity_incentives"), defaults):
            if isinstance(source, Mapping) and key in source:
                return source[key]
        return None

    incentives_cfg = run_config.get("electricity_incentives")
    production_source: Any | None = None
    investment_source: Any | None = None
    if isinstance(incentives_cfg, Mapping):
        enabled_default = bool(incentives_cfg.get("enabled", enabled_default))
        production_source = incentives_cfg.get("production")
        investment_source = incentives_cfg.get("investment")
    if production_source is None and isinstance(defaults, Mapping):
        production_source = defaults.get("production")
    if investment_source is None and isinstance(defaults, Mapping):
        investment_source = defaults.get("investment")

    def _normalise_config_entries(
        source: Any, *, credit_key: str, limit_key: str
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        if isinstance(source, Mapping):
            iterable: Iterable[Any] = [source]
        elif isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
            iterable = source
        else:
            iterable = []
        for entry in iterable:
            if not isinstance(entry, Mapping):
                continue
            tech_id = resolve_technology_key(entry.get("technology"))
            if tech_id is None:
                continue
            try:
                year_int = int(entry.get("year"))
            except (TypeError, ValueError):
                continue
            credit_val = _coerce_optional_float(entry.get(credit_key))
            if credit_val is None:
                continue
            limit_val = _coerce_optional_float(entry.get(limit_key))
            record: dict[str, Any] = {
                "technology": get_technology_label(tech_id),
                "year": year_int,
                credit_key: float(credit_val),
            }
            if limit_val is not None:
                record[limit_key] = float(limit_val)
            entries.append(record)
        entries.sort(key=lambda item: (str(item["technology"]).lower(), int(item["year"])))
        return entries

    existing_production_entries = _normalise_config_entries(
        production_source, credit_key="credit_per_mwh", limit_key="limit_mwh"
    )
    existing_investment_entries = _normalise_config_entries(
        investment_source, credit_key="credit_per_mw", limit_key="limit_mw"
    )

    technology_options: set[str] = {
        get_technology_label(tech_id) for tech_id in sorted(TECH_ID_TO_LABEL or {})
    }
    for entry in (*existing_production_entries, *existing_investment_entries):
        label = str(entry.get("technology", "")).strip()
        if label:
            technology_options.add(label)
    if not technology_options:
        technology_options = {"Coal", "Gas", "Wind", "Solar"}
    technology_labels = sorted(technology_options)

    production_credit_col = "Credit ($/MWh)"
    production_limit_col = "Limit (MWh)"
    investment_credit_col = "Credit ($/MW)"
    investment_limit_col = "Limit (MW)"
    selection_column = "Apply Credit"

    def _build_editor_rows(
        entries: list[dict[str, Any]],
        *,
        credit_key: str,
        limit_key: str,
        credit_label: str,
        limit_label: str,
        selection_label: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for entry in entries:
            rows.append(
                {
                    selection_label: True,
                    "Technology": entry["technology"],
                    "Years": str(entry["year"]),
                    credit_label: entry.get(credit_key),
                    limit_label: entry.get(limit_key),
                }
            )
        seen = {str(row.get("Technology")) for row in rows if row.get("Technology")}
        for label in technology_labels:
            if label not in seen:
                rows.append(
                    {
                        selection_label: False,
                        "Technology": label,
                        "Years": "",
                        credit_label: None,
                        limit_label: None,
                    }
                )
        rows.sort(
            key=lambda row: (
                str(row.get("Technology", "")).lower(),
                str(row.get("Years", "")).lower(),
            )
        )
        return rows

    production_rows_default: list[dict[str, Any]] = _build_editor_rows(
        existing_production_entries,
        credit_key="credit_per_mwh",
        limit_key="limit_mwh",
        credit_label=production_credit_col,
        limit_label=production_limit_col,
        selection_label=selection_column,
    )
    investment_rows_default: list[dict[str, Any]] = _build_editor_rows(
        existing_investment_entries,
        credit_key="credit_per_mw",
        limit_key="limit_mw",
        credit_label=investment_credit_col,
        limit_label=investment_limit_col,
        selection_label=selection_column,
    )

    production_column_order: list[str] = [
        selection_column,
        "Technology",
        "Years",
        production_credit_col,
        production_limit_col,
    ]
    investment_column_order: list[str] = [
        selection_column,
        "Technology",
        "Years",
        investment_credit_col,
        investment_limit_col,
    ]

    available_years = _simulation_years_from_config(run_config)
    valid_years_set = {int(year) for year in available_years}

    def _rows_to_config_entries(
        rows: list[Mapping[str, Any]],
        *,
        credit_column: str,
        limit_column: str,
        credit_config_key: str,
        limit_config_key: str,
        context_label: str,
        valid_years: set[int],
        selection_column: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        results: dict[tuple[int, int], dict[str, Any]] = {}
        messages: list[str] = []
        for index, row in enumerate(rows, start=1):
            if selection_column and selection_column in row:
                include_row = _coerce_bool_flag(row.get(selection_column), default=False)
                if not include_row:
                    continue
            technology_value = row.get("Technology")
            technology_label = (
                str(technology_value).strip() if technology_value not in (None, "") else ""
            )
            if not technology_label:
                continue
            tech_id = resolve_technology_key(technology_label)
            if tech_id is None:
                messages.append(f'{context_label} row {index}: Unknown technology "{technology_label}".')
                continue
            years_value = row.get("Years")
            years, invalid_tokens, out_of_range = _parse_years_field(
                years_value, valid_years=valid_years
            )
            if invalid_tokens:
                tokens_display = ", ".join(
                    sorted({token.strip() for token in invalid_tokens if token.strip()})
                )
                if tokens_display:
                    messages.append(
                        f"{context_label} row {index}: Unable to parse year value(s): {tokens_display}."
                    )
            if out_of_range:
                years_display = ", ".join(str(year) for year in out_of_range)
                messages.append(
                    f"{context_label} row {index}: Year(s) {years_display} fall outside the selected simulation years."
                )
            if not years:
                years_text = str(years_value).strip() if isinstance(years_value, str) else ""
                credit_candidate = _coerce_optional_float(row.get(credit_column))
                if years_text or credit_candidate is not None:
                    messages.append(f"{context_label} row {index}: Specify one or more valid years.")
                continue
            credit_value = _coerce_optional_float(row.get(credit_column))
            if credit_value is None:
                messages.append(f"{context_label} row {index}: Provide a credit value.")
                continue
            limit_value = _coerce_optional_float(row.get(limit_column))
            label = get_technology_label(tech_id)
            for year in years:
                entry = {
                    "technology": label,
                    "year": int(year),
                    credit_config_key: float(credit_value),
                }
                if limit_value is not None:
                    entry[limit_config_key] = float(limit_value)
                results[(tech_id, int(year))] = entry
        ordered = sorted(
            results.values(),
            key=lambda item: (str(item["technology"]).lower(), int(item["year"])),
        )
        return ordered, messages

    enabled = container.toggle(
        "Enable incentives and credits",
        value=enabled_default,
        key="incentives_enable",
    )

    errors: list[str] = []
    production_entries = existing_production_entries
    investment_entries = existing_investment_entries

    production_editor_height = _coerce_editor_dimension(
        _editor_setting("production_editor_height")
    )
    production_editor_width = _coerce_editor_width(
        _editor_setting("production_editor_width")
    ) or "stretch"
    production_editor_num_rows = _coerce_editor_num_rows(
        _editor_setting("production_editor_num_rows")
    )

    investment_editor_height = _coerce_editor_dimension(
        _editor_setting("investment_editor_height")
    )
    investment_editor_width = _coerce_editor_width(
        _editor_setting("investment_editor_width")
    ) or "stretch"
    investment_editor_num_rows = _coerce_editor_num_rows(
        _editor_setting("investment_editor_num_rows")
    )

    with _sidebar_panel(container, enabled) as panel:
        panel.caption(
            "Specify technology-specific tax credits that feed the electricity capacity and generation modules."
        )
        if available_years:
            years_display = ", ".join(str(year) for year in available_years)
            panel.caption(f"Simulation years: {years_display}")
        panel.caption(
            "Enter comma-separated years or ranges (e.g., 2025, 2030-2032). "
            "Leave blank to exclude a technology."
        )

        panel.markdown("**Production tax credits ($/MWh)**")
        production_editor_kwargs: dict[str, Any] = {
            "disabled": not enabled,
            "hide_index": True,
            "num_rows": production_editor_num_rows,
            "width": production_editor_width,
            "key": "incentives_production_editor",
            "column_order": production_column_order,
            "column_config": {
                selection_column: st.column_config.CheckboxColumn(
                    "Apply Credit",
                    help=(
                        "Select to apply production tax credits for this technology. "
                        "Unchecked technologies default to $0 incentives across all years."
                    ),
                    default=False,
                ),
                "Technology": st.column_config.SelectboxColumn(
                    "Technology", options=technology_labels
                ),
                "Years": st.column_config.TextColumn(
                    "Applicable years",
                    help="Comma-separated years or ranges (e.g., 2025, 2030-2032).",
                ),
                production_credit_col: st.column_config.NumberColumn(
                    production_credit_col,
                    format="$%.2f",
                    min_value=0.0,
                    help="Credit value applied per megawatt-hour.",
                ),
                production_limit_col: st.column_config.NumberColumn(
                    production_limit_col,
                    min_value=0.0,
                    help="Optional annual limit on eligible production (MWh).",
                ),
            },
        }
        if production_editor_height is not None:
            production_editor_kwargs["height"] = production_editor_height

        # --- sanitize editor kwargs ---
        width_value = production_editor_kwargs.pop("width", None)
        if isinstance(width_value, str):
            lowered = width_value.strip().lower()
            if lowered == "stretch":
                production_editor_kwargs["use_container_width"] = True
            elif lowered != "auto":
                try:
                    production_editor_kwargs["width"] = int(width_value)
                except (TypeError, ValueError):
                    production_editor_kwargs["width"] = None
        elif width_value is not None:
            try:
                production_editor_kwargs["width"] = int(width_value)
            except (TypeError, ValueError):
                production_editor_kwargs["width"] = None

        if "height" in production_editor_kwargs:
            try:
                production_editor_kwargs["height"] = int(
                    production_editor_kwargs["height"]
                )
            except (TypeError, ValueError):
                production_editor_kwargs["height"] = None

        production_editor_value = panel.data_editor(
            production_rows_default,
            **production_editor_kwargs,
        )

        panel.markdown("**Investment tax credits ($/MW)**")
        investment_editor_kwargs: dict[str, Any] = {
            "disabled": not enabled,
            "hide_index": True,
            "num_rows": investment_editor_num_rows,
            "width": investment_editor_width,
            "key": "incentives_investment_editor",
            "column_order": investment_column_order,
            "column_config": {
                selection_column: st.column_config.CheckboxColumn(
                    "Apply Credit",
                    help=(
                        "Select to apply investment tax credits for this technology. "
                        "Unchecked technologies default to $0 incentives across all years."
                    ),
                    default=False,
                ),
                "Technology": st.column_config.SelectboxColumn(
                    "Technology", options=technology_labels
                ),
                "Years": st.column_config.TextColumn(
                    "Applicable years",
                    help="Comma-separated years or ranges (e.g., 2025, 2030-2032).",
                ),
                investment_credit_col: st.column_config.NumberColumn(
                    investment_credit_col,
                    format="$%.2f",
                    min_value=0.0,
                    help="Credit value applied per megawatt of installed capacity.",
                ),
                investment_limit_col: st.column_config.NumberColumn(
                    investment_limit_col,
                    min_value=0.0,
                    help="Optional annual limit on eligible capacity additions (MW).",
                ),
            },
        }
        if investment_editor_height is not None:
            investment_editor_kwargs["height"] = investment_editor_height

        # --- sanitize editor kwargs ---
        width_value = investment_editor_kwargs.pop("width", None)
        if isinstance(width_value, str):
            lowered = width_value.strip().lower()
            if lowered == "stretch":
                investment_editor_kwargs["use_container_width"] = True
            elif lowered != "auto":
                try:
                    investment_editor_kwargs["width"] = int(width_value)
                except (TypeError, ValueError):
                    investment_editor_kwargs["width"] = None
        elif width_value is not None:
            try:
                investment_editor_kwargs["width"] = int(width_value)
            except (TypeError, ValueError):
                investment_editor_kwargs["width"] = None

        if "height" in investment_editor_kwargs:
            try:
                investment_editor_kwargs["height"] = int(
                    investment_editor_kwargs["height"]
                )
            except (TypeError, ValueError):
                investment_editor_kwargs["height"] = None

        investment_editor_value = panel.data_editor(
            investment_rows_default,
            **investment_editor_kwargs,
        )


        validation_messages: list[str] = []
        if enabled:
            production_entries, production_messages = _rows_to_config_entries(
                _data_editor_records(production_editor_value),
                credit_column=production_credit_col,
                limit_column=production_limit_col,
                credit_config_key="credit_per_mwh",
                limit_config_key="limit_mwh",
                context_label="Production tax credit",
                valid_years=valid_years_set,
                selection_column=selection_column,
            )
            investment_entries, investment_messages = _rows_to_config_entries(
                _data_editor_records(investment_editor_value),
                credit_column=investment_credit_col,
                limit_column=investment_limit_col,
                credit_config_key="credit_per_mw",
                limit_config_key="limit_mw",
                context_label="Investment tax credit",
                valid_years=valid_years_set,
                selection_column=selection_column,
            )
            validation_messages.extend(production_messages)
            validation_messages.extend(investment_messages)

        for message in validation_messages:
            panel.error(message)
        errors.extend(validation_messages)

        if enabled:
            if frames is None:
                message = "Incentives require generating unit data."
                panel.error(message)
                errors.append(message)
            else:
                try:
                    units_df = frames.units()
                except Exception as exc:
                    message = f"Unable to access unit data: {exc}"
                    panel.error(message)
                    errors.append(message)
                else:
                    if units_df.empty:
                        message = "Incentives require at least one generating unit."
                        panel.error(message)
                        errors.append(message)

    incentives_record: dict[str, Any] = {"enabled": bool(enabled)}
    if production_entries:
        incentives_record["production"] = copy.deepcopy(production_entries)
    if investment_entries:
        incentives_record["investment"] = copy.deepcopy(investment_entries)

    run_config["electricity_incentives"] = copy.deepcopy(incentives_record)
    modules["incentives"] = copy.deepcopy(incentives_record)

    return IncentivesModuleSettings(
        enabled=bool(enabled),
        production_credits=copy.deepcopy(production_entries),
        investment_credits=copy.deepcopy(investment_entries),
        errors=errors,
    )


# -------------------------
# Outputs UI
# -------------------------
def _render_outputs_section(
    container: Any,
    run_config: dict[str, Any],
    last_result: Mapping[str, Any] | None,
) -> OutputsModuleSettings:
    modules = run_config.setdefault("modules", {})
    defaults = modules.get("outputs", {})
    enabled_default = bool(defaults.get("enabled", True))
    directory_default = str(defaults.get("directory") or run_config.get("output_name") or "outputs")
    show_csv_default = bool(defaults.get("show_csv_downloads", True))

    downloads_root = get_downloads_directory()

    enabled = container.toggle(
        "Enable output management",
        value=enabled_default,
        key="outputs_enable",
    )

    directory_value = directory_default
    show_csv_downloads = show_csv_default
    errors: list[str] = []

    with _sidebar_panel(container, enabled) as panel:
        directory_value = panel.text_input(
            "Output directory name",
            value=directory_default,
            disabled=not enabled,
            key="outputs_directory",
        ).strip()
        show_csv_downloads = panel.checkbox(
            "Show CSV downloads from last run",
            value=show_csv_default,
            disabled=not enabled,
            key="outputs_csv",
        )

        resolved_directory = downloads_root if not directory_value else downloads_root / directory_value
        panel.caption(f"Outputs will be saved to {resolved_directory}")

        if enabled and not directory_value:
            message = "Specify an output directory when the outputs module is enabled."
            panel.error(message)
            errors.append(message)

        csv_files: Mapping[str, Any] | None = None
        if enabled and show_csv_downloads:
            if isinstance(last_result, Mapping):
                csv_files = last_result.get("csv_files")  # type: ignore[assignment]
            if csv_files:
                panel.caption("Download CSV outputs from the most recent run.")
                for filename, content in sorted(csv_files.items()):
                    panel.download_button(
                        label=f"Download {filename}",
                        data=content,
                        file_name=filename,
                        mime="text/csv",
                        key=f"outputs_download_{filename}",
                    )
            else:
                panel.info("No CSV outputs are available yet.")
        elif enabled:
            panel.caption("CSV downloads will be available after the next run.")

    if not directory_value:
        directory_value = directory_default or "outputs"
    if not enabled:
        show_csv_downloads = False

    run_config["output_name"] = directory_value
    resolved_directory = downloads_root if not directory_value else downloads_root / directory_value
    modules["outputs"] = {
        "enabled": bool(enabled),
        "directory": directory_value,
        "show_csv_downloads": bool(show_csv_downloads),
        "resolved_path": str(resolved_directory),  # config serialization
    }

    return OutputsModuleSettings(
        enabled=bool(enabled),
        directory=directory_value,
        resolved_path=resolved_directory,  # keep as Path in memory
        show_csv_downloads=bool(show_csv_downloads),
        errors=errors,
    )


# -------------------------
# Frames + runner helpers
# -------------------------
def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, str):
            return float(value.strip())
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_str(value: Any, default: str | None = None) -> str:
    if default is None:
        default = _get_default_region_id()
    if value in (None, ""):
        return default
    return str(value)


def _coerce_year_set(value: Any, fallback: Iterable[int]) -> set[int]:
    years: set[int] = set()
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        for entry in value:
            try:
                years.add(int(entry))
            except (TypeError, ValueError):
                continue
    elif value not in (None, ""):
        try:
            years.add(int(value))
        except (TypeError, ValueError):
            pass
    if not years:
        years = {int(year) for year in fallback}
    return years


def _apply_schedule_growth(
    value: float,
    growth: float | None,
    *,
    mode: str = "percent",
    steps: int = 1,
    forward: bool = True,
) -> float:
    """Return ``value`` adjusted ``steps`` times according to ``growth`` settings."""

    if growth in (None, 0.0) or steps <= 0:
        return float(value)

    if mode == "percent":
        factor = 1.0 + float(growth) / 100.0
        if factor == 0.0:
            return 0.0 if forward else float(value)
        power = factor**steps
        return float(value) * power if forward else float(value) / power

    delta = float(growth) * steps
    return float(value) + delta if forward else float(value) - delta


def _coerce_year_value_map(
    entry: Any,
    years: Iterable[int],
    *,
    cast: Callable[[Any], _T],
    default: _T,
    growth: float | None = None,
    growth_mode: str = "percent",
) -> dict[int, _T]:
    values: dict[int, _T] = {}

    if isinstance(entry, Mapping):
        iterator = entry.items()
    elif isinstance(entry, Iterable) and not isinstance(entry, (str, bytes)):
        iterator = []
        for item in entry:
            if isinstance(item, Mapping) and "year" in item:
                iterator.append((item.get("year"), item.get("value", item.get("amount"))))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                iterator.append((item[0], item[1]))
    elif entry is not None:
        try:
            coerced = cast(entry)
        except (TypeError, ValueError):
            coerced = cast(default)
        normalized_years = sorted(int(year) for year in years)
        if not normalized_years:
            return {}
        start = normalized_years[0]
        end = normalized_years[-1]
        return {year: coerced for year in range(start, end + 1)}
    else:
        iterator = []

    for year, raw_value in iterator:
        try:
            year_int = int(year)
        except (TypeError, ValueError):
            continue
        try:
            values[year_int] = cast(raw_value)
        except (TypeError, ValueError):
            continue

    normalized_years = sorted(int(year) for year in years)
    if not normalized_years:
        return {}

    run_years = list(range(normalized_years[0], normalized_years[-1] + 1))

    if not values:
        default_value = cast(default)
        return {year: default_value for year in run_years}

    known_years = sorted(values.keys())
    first_year = known_years[0]
    first_value = values[first_year]
    result: dict[int, _T] = {}
    numeric_growth = growth not in (None, 0.0)
    last_numeric: float | None = None
    last_value: _T | None = None

    for year in run_years:
        if year in values:
            last_value = values[year]
            if numeric_growth:
                try:
                    last_numeric = float(last_value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    last_numeric = float(cast(default))
        else:
            if numeric_growth:
                if last_numeric is None:
                    try:
                        base_numeric = float(first_value)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        base_numeric = float(cast(default))
                    steps = first_year - year
                    last_numeric = _apply_schedule_growth(
                        base_numeric,
                        growth,
                        mode=growth_mode,
                        steps=steps,
                        forward=False,
                    )
                else:
                    last_numeric = _apply_schedule_growth(
                        last_numeric,
                        growth,
                        mode=growth_mode,
                        steps=1,
                        forward=True,
                    )
                last_value = cast(last_numeric)
            else:
                if last_value is None:
                    last_value = values[first_year]
        result[year] = cast(last_value if last_value is not None else cast(default))

    return result


def _build_policy_frame(
    config: Mapping[str, Any],
    years: Iterable[int],
    carbon_policy_enabled: bool,
    *,
    ccr1_enabled: bool | None = None,
    ccr2_enabled: bool | None = None,
    control_period_years: int | None = None,
    banking_enabled: bool = True,
    floor_escalator_mode: str | None = None,
    floor_escalator_value: float | None = None,
    ccr1_escalator_pct: float | None = None,
    ccr2_escalator_pct: float | None = None,
) -> pd.DataFrame:
    years_list = sorted(int(year) for year in years)
    if not years_list:
        raise ValueError("No years supplied for policy frame")

    market_cfg = config.get("allowance_market")
    if not isinstance(market_cfg, Mapping):
        market_cfg = {}

    bank_flag = bool(carbon_policy_enabled and banking_enabled)

    resolution_raw = market_cfg.get("resolution", "annual")
    if isinstance(resolution_raw, str):
        resolution = resolution_raw.strip().lower() or "annual"
    else:
        resolution = str(resolution_raw).strip().lower() or "annual"
    if resolution not in {"annual", "daily"}:
        resolution = "annual"

    if carbon_policy_enabled:
        ccr1_flag = _coerce_bool_flag(market_cfg.get("ccr1_enabled"), default=True)
        ccr2_flag = _coerce_bool_flag(market_cfg.get("ccr2_enabled"), default=True)
        if ccr1_enabled is not None:
            ccr1_flag = bool(ccr1_enabled)
        if ccr2_enabled is not None:
            ccr2_flag = bool(ccr2_enabled)

        control_period = control_period_years
        if control_period is None:
            raw_control = market_cfg.get("control_period_years")
            if raw_control not in (None, ""):
                try:
                    control_period = int(raw_control)
                except (TypeError, ValueError):
                    control_period = None
        if control_period is not None and control_period <= 0:
            control_period = None

        cap_map = _coerce_year_value_map(
            market_cfg.get("cap"),
            years_list,
            cast=float,
            default=0.0,
        )
        floor_growth_mode = str(floor_escalator_mode or market_cfg.get("floor_escalator_mode", "fixed")).strip().lower()
        if floor_growth_mode not in {"percent", "fixed"}:
            floor_growth_mode = "fixed"
        floor_growth_value = (
            floor_escalator_value
            if floor_escalator_value is not None
            else _coerce_float(market_cfg.get("floor_escalator_value"), 0.0)
        )
        floor_map = _coerce_year_value_map(
            market_cfg.get("floor"),
            years_list,
            cast=float,
            default=0.0,
            growth=floor_growth_value,
            growth_mode=floor_growth_mode,
        )
        ccr1_trigger_map = _coerce_year_value_map(
            market_cfg.get("ccr1_trigger"),
            years_list,
            cast=float,
            default=0.0,
            growth=ccr1_escalator_pct,
            growth_mode="percent",
        )
        ccr1_qty_map = _coerce_year_value_map(
            market_cfg.get("ccr1_qty"), years_list, cast=float, default=0.0
        )
        ccr2_trigger_map = _coerce_year_value_map(
            market_cfg.get("ccr2_trigger"),
            years_list,
            cast=float,
            default=0.0,
            growth=ccr2_escalator_pct,
            growth_mode="percent",
        )
        ccr2_qty_map = _coerce_year_value_map(
            market_cfg.get("ccr2_qty"), years_list, cast=float, default=0.0
        )
        cp_id_map = _coerce_year_value_map(
            market_cfg.get("cp_id"), years_list, cast=lambda v: _coerce_str(v, "CP1"), default="CP1"
        )
        bank0 = _coerce_float(market_cfg.get("bank0"), default=0.0)
        surrender_frac = _coerce_float(market_cfg.get("annual_surrender_frac"), default=1.0)
        carry_pct = _coerce_float(market_cfg.get("carry_pct"), default=1.0)
        if not bank_flag:
            bank0 = 0.0
            carry_pct = 0.0
        full_compliance_years = _coerce_year_set(
            market_cfg.get("full_compliance_years"), fallback=[]
        )
        if not full_compliance_years:
            if control_period:
                full_compliance_years = {
                    year
                    for idx, year in enumerate(years_list, start=1)
                    if idx % control_period == 0
                }
            if not full_compliance_years:
                full_compliance_years = {years_list[-1]}
    else:
        cap_map = {year: float(_LARGE_ALLOWANCE_SUPPLY) for year in years_list}
        floor_map = {year: 0.0 for year in years_list}
        ccr1_trigger_map = {year: 0.0 for year in years_list}
        ccr1_qty_map = {year: 0.0 for year in years_list}
        ccr2_trigger_map = {year: 0.0 for year in years_list}
        ccr2_qty_map = {year: 0.0 for year in years_list}
        cp_id_map = {year: "NoPolicy" for year in years_list}
        bank0 = _LARGE_ALLOWANCE_SUPPLY
        surrender_frac = 0.0
        carry_pct = 1.0
        full_compliance_years = set()
        ccr1_flag = False
        ccr2_flag = False
        control_period = None
        bank_flag = False

    records: list[dict[str, Any]] = []
    for year in years_list:
        records.append(
            {
                "year": year,
                "cap_tons": float(cap_map[year]),
                "floor_dollars": float(floor_map[year]),
                "ccr1_trigger": float(ccr1_trigger_map[year]),
                "ccr1_qty": float(ccr1_qty_map[year]),
                "ccr2_trigger": float(ccr2_trigger_map[year]),
                "ccr2_qty": float(ccr2_qty_map[year]),
                "cp_id": str(cp_id_map[year]),
                "full_compliance": year in full_compliance_years,
                "bank0": float(bank0),
                "annual_surrender_frac": float(surrender_frac),
                "carry_pct": float(carry_pct),
                "policy_enabled": bool(carbon_policy_enabled),
                "ccr1_enabled": bool(ccr1_flag),
                "ccr2_enabled": bool(ccr2_flag),
                "control_period_years": control_period,
                "bank_enabled": bool(bank_flag),
                "resolution": "annual" if resolution not in {"annual", "daily"} else resolution,
            }
        )

    return pd.DataFrame(records)


def _available_regions_from_frames(frames: FramesType) -> list[str]:
    """Return an ordered list of region labels present in ``frames``."""

    regions: list[str] = []

    try:
        demand = frames.demand()
        if not demand.empty and 'region' in demand.columns:
            for value in demand['region']:
                label = str(value)
                if label not in regions:
                    regions.append(label)
    except Exception:  # pragma: no cover - defensive guard
        pass

    try:
        units = frames.units()
        if not units.empty and 'region' in units.columns:
            for value in units['region']:
                label = str(value)
                if label not in regions:
                    regions.append(label)
    except Exception:  # pragma: no cover - defensive guard
        pass

    if not regions:
        regions = [_get_default_region_id()]

    return regions


def _build_coverage_frame(
    frames: FramesType,
    coverage_regions: Iterable[str] | None,
) -> pd.DataFrame | None:
    """Construct a coverage table aligning regions with enabled status."""

    if coverage_regions is None:
        return None

    normalized = _normalize_coverage_selection(coverage_regions)
    cover_all = normalized == ["All"]

    regions = _available_regions_from_frames(frames)
    ordered = list(dict.fromkeys(regions))
    for label in normalized:
        if label != "All" and label not in ordered:
            ordered.append(label)

    records = [
        {
            'region': region,
            'covered': True if cover_all else region in normalized,
        }
        for region in ordered
    ]

    return pd.DataFrame(records)


def _default_units(regions: Iterable[str] | None = None) -> pd.DataFrame:
    region_list = list(dict.fromkeys(regions or []))

    try:
        return load_units(active_regions=region_list)
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.warning(
            "Falling back to legacy unit fleet for regions: %s", region_list,
            exc_info=True,
        )

    legacy = load_unit_fleet(region_list)
    if legacy.empty:
        return pd.DataFrame(
            columns=[
                "unit_id",
                "region",
                "fuel",
                "cap_mw",
                "availability",
                "hr_mmbtu_per_mwh",
                "vom_per_mwh",
                "fuel_price_per_mmbtu",
                "ef_ton_per_mwh",
            ]
        )

    # Process columns using assign for proper DataFrame handling
    legacy = legacy.copy()
    
    # Build column transformations
    if "region_id" in legacy.columns:
        legacy = legacy.assign(region=legacy["region_id"].astype(str))
    elif "region" not in legacy.columns:
        legacy = legacy.assign(region="")
    else:
        legacy = legacy.assign(region=legacy["region"].astype(str))
    
    if "fuel" not in legacy.columns:
        legacy = legacy.assign(fuel="")
    else:
        legacy = legacy.assign(fuel=legacy["fuel"].astype(str))
    
    if "cap_mw" in legacy.columns:
        legacy = legacy.assign(cap_mw=pd.to_numeric(legacy["cap_mw"], errors="coerce").fillna(0.0))
    else:
        legacy = legacy.assign(cap_mw=0.0)
    
    legacy = legacy.assign(availability=1.0)
    
    if "hr_mmbtu_per_mwh" in legacy.columns:
        legacy = legacy.assign(hr_mmbtu_per_mwh=pd.to_numeric(legacy["hr_mmbtu_per_mwh"], errors="coerce").fillna(0.0))
    elif "heat_rate_mmbtu_per_mwh" in legacy.columns:
        legacy = legacy.assign(hr_mmbtu_per_mwh=pd.to_numeric(legacy["heat_rate_mmbtu_per_mwh"], errors="coerce").fillna(0.0))
    else:
        legacy = legacy.assign(hr_mmbtu_per_mwh=0.0)
    
    # Set default fuel prices based on fuel type to satisfy validation
    # These are placeholder values - production systems should load actual fuel price data
    fuel_price_defaults = {
        'Coal': 2.5,          # $/MMBtu
        'Gas': 4.0,           # $/MMBtu
        'GasCombinedCycle': 4.0,
        'GasTurbine': 4.5,
        'GasSteam': 4.0,
        'Oil': 15.0,          # $/MMBtu
        'Biomass': 3.0,       # $/MMBtu
        'Other': 5.0,         # $/MMBtu default for unknown fossil fuels
    }
    
    # Apply fuel price defaults based on fuel type
    fuel_prices = legacy['fuel'].map(fuel_price_defaults).fillna(0.0)
    
    legacy = legacy.assign(
        heat_rate_mmbtu_per_mwh=legacy["hr_mmbtu_per_mwh"],
        vom_per_mwh=0.0,
        fuel_price_per_mmbtu=fuel_prices
    )
    
    if "ef_ton_per_mwh" in legacy.columns:
        legacy = legacy.assign(ef_ton_per_mwh=pd.to_numeric(legacy["ef_ton_per_mwh"], errors="coerce").fillna(0.0))
    elif "co2_short_ton_per_mwh" in legacy.columns:
        legacy = legacy.assign(ef_ton_per_mwh=pd.to_numeric(legacy["co2_short_ton_per_mwh"], errors="coerce").fillna(0.0))
    elif "co2_ton_per_mwh" in legacy.columns:
        legacy = legacy.assign(ef_ton_per_mwh=pd.to_numeric(legacy["co2_ton_per_mwh"], errors="coerce").fillna(0.0))
    else:
        legacy = legacy.assign(ef_ton_per_mwh=0.0)
    
    legacy = legacy.assign(co2_ton_per_mwh=legacy["ef_ton_per_mwh"])

    # Remove any duplicate columns that may have been created
    if legacy.columns.duplicated().any():
        LOGGER.warning(f"Removing duplicate columns: {legacy.columns[legacy.columns.duplicated()].tolist()}")
        legacy = legacy.loc[:, ~legacy.columns.duplicated(keep='last')]

    # Return with expected columns
    return_cols = [
        "unit_id",
        "region",
        "fuel",
        "cap_mw",
        "availability",
        "hr_mmbtu_per_mwh",
        "heat_rate_mmbtu_per_mwh",
        "vom_per_mwh",
        "fuel_price_per_mmbtu",
        "ef_ton_per_mwh",
        "co2_ton_per_mwh",
    ]
    return legacy[return_cols].copy()


def _default_fuels(units: pd.DataFrame | None = None) -> pd.DataFrame:
    units_frame = units if units is not None else _default_units()
    return derive_fuels(units_frame)


def _default_transmission(regions: Iterable[str] | None = None) -> pd.DataFrame:
    """Load transmission edges from ei_edges.csv - no fallback topology."""
    columns = ["from_region", "to_region", "limit_mw", "cost_per_mwh"]

    try:
        topology = load_edges()
    except (FileNotFoundError, ImportError) as exc:
        LOGGER.error(
            "Unable to load transmission edges from ei_edges.csv: %s. "
            "No fallback topology is generated.", exc
        )
        return pd.DataFrame(columns=columns)
    except Exception:  # pragma: no cover - defensive guard
        LOGGER.error(
            "Unexpected error loading transmission edges",
            exc_info=True,
        )
        return pd.DataFrame(columns=columns)

    if not isinstance(topology, pd.DataFrame):
        topology = pd.DataFrame(topology)

    if topology.empty:
        return pd.DataFrame(columns=columns)

    rename_map = {}
    if "from_region" not in topology.columns:
        if "from_zone" in topology.columns:
            rename_map["from_zone"] = "from_region"
        elif "from" in topology.columns:
            rename_map["from"] = "from_region"
    if "to_region" not in topology.columns:
        if "to_zone" in topology.columns:
            rename_map["to_zone"] = "to_region"
        elif "to" in topology.columns:
            rename_map["to"] = "to_region"
    if "cost_per_mwh" not in topology.columns and "wheel_cost_per_mwh" in topology.columns:
        rename_map["wheel_cost_per_mwh"] = "cost_per_mwh"

    if rename_map:
        topology = topology.rename(columns=rename_map)

    for column in columns:
        if column not in topology.columns:
            if column in {"limit_mw", "cost_per_mwh"}:
                topology[column] = 0.0
            else:
                topology[column] = ""

    topology["from_region"] = topology["from_region"].astype(str)
    topology["to_region"] = topology["to_region"].astype(str)
    topology["limit_mw"] = pd.to_numeric(topology["limit_mw"], errors="coerce").fillna(0.0)
    topology["cost_per_mwh"] = pd.to_numeric(
        topology["cost_per_mwh"], errors="coerce"
    ).fillna(0.0)

    if regions:
        region_list = list(regions)
        mask = topology["from_region"].isin(region_list) & topology["to_region"].isin(
            region_list
        )
        topology = topology.loc[mask]

    return topology.loc[:, columns].reset_index(drop=True)


def _build_default_frames(
    years: Iterable[int],
    *,
    carbon_policy_enabled: bool = True,
    banking_enabled: bool = True,
    carbon_price_schedule: Mapping[int, float] | Mapping[str, Any] | None = None,
    forecast_bundles: Sequence[_ScenarioSelection] | None = None,
    custom_forecasts: Mapping[str, Any] | None = None,
) -> FramesType:
    frames_cls = FramesType
    manifests: Sequence[_ScenarioSelection] | None = forecast_bundles
    base_path = _resolve_forecast_base_path()
    frame = _cached_forecast_frame(base_path)

    if manifests is None:
        manifests = _selected_forecast_bundles_from_state()

    if manifests is None:
        manifests = _default_scenario_manifests(frame, base_path=base_path)

    normalized_years: tuple[int, ...] | None = None
    if years is not None:
        normalized_years = tuple(int(year) for year in years)

    demand_result = _demand_frame_from_manifests(
        list(manifests or []),
        years=normalized_years,
        base_path=base_path,
        collect_load_frames=True,
    )

    if isinstance(demand_result, tuple):
        demand_df, load_df = demand_result
    else:
        demand_df = demand_result
        load_df = pd.DataFrame(columns=["iso", "zone", "scenario", "year", "load_gwh"])

    if custom_forecasts:
        custom_entries = _coerce_custom_upload_mapping(custom_forecasts)
        if custom_entries:
            custom_frames: list[pd.DataFrame] = []
            for entry in custom_entries.values():
                records = entry.get("records")
                if isinstance(records, pd.DataFrame):
                    working_df = records.copy()
                elif isinstance(records, Sequence) and not isinstance(
                    records, (str, bytes, bytearray)
                ):
                    try:
                        working_df = pd.DataFrame.from_records(records)
                    except Exception:
                        continue
                else:
                    continue

                if working_df.empty:
                    continue

                if not {"region", "year", "demand_mwh"}.issubset(working_df.columns):
                    continue

                candidate = working_df.loc[:, ["region", "year", "demand_mwh"]].copy()
                candidate["region"] = candidate["region"].astype(str)
                candidate["year"] = pd.to_numeric(candidate["year"], errors="coerce")
                candidate["demand_mwh"] = pd.to_numeric(
                    candidate["demand_mwh"], errors="coerce"
                )
                candidate = candidate.dropna(subset=["region", "year", "demand_mwh"])
                if candidate.empty:
                    continue
                candidate["region"] = candidate["region"].astype(str)
                candidate["year"] = candidate["year"].astype(int)
                candidate["demand_mwh"] = candidate["demand_mwh"].astype(float)
                custom_frames.append(candidate)

            if custom_frames:
                custom_df = pd.concat(custom_frames, ignore_index=True)
                if normalized_years:
                    year_filter = {int(year) for year in normalized_years}
                    if year_filter:
                        custom_df = custom_df[custom_df["year"].isin(year_filter)]
                if not custom_df.empty:
                    if demand_df.empty:
                        demand_df = custom_df
                    else:
                        demand_df = pd.concat([demand_df, custom_df], ignore_index=True)
                    demand_df = (
                        demand_df.groupby(["region", "year"], as_index=False)["demand_mwh"].sum()
                    )

    default_region_id = _get_default_region_id()
    year_sequence = normalized_years or ()

    def _fallback_demand_frame(region: str) -> pd.DataFrame:
        target_years = year_sequence or (int(date.today().year),)
        records = [
            {
                "year": int(year),
                "region": region,
                "demand_mwh": float(_DEFAULT_LOAD_MWH),
            }
            for year in target_years
        ]
        return pd.DataFrame(records)

    if load_df.empty and not demand_df.empty:
        fallback_load = demand_df.copy()
        fallback_load["zone"] = fallback_load["region"].astype(str)
        fallback_load["iso"] = fallback_load["zone"].str.split("_", n=1).str[0].str.upper()
        fallback_load["scenario"] = "Baseline"
        fallback_load["load_gwh"] = pd.to_numeric(
            fallback_load["demand_mwh"], errors="coerce"
        ) / 1000.0
        fallback_load = fallback_load.dropna(subset=["zone", "year", "load_gwh"], how="any")
        if not fallback_load.empty:
            load_df = fallback_load.loc[:, ["iso", "zone", "scenario", "year", "load_gwh"]]
            load_df["year"] = load_df["year"].astype(int)

    if demand_df.empty:
        demand_df = _fallback_demand_frame(default_region_id)

    if not demand_df.empty:
        demand_df = (
            demand_df.sort_values(["year", "region"]).reset_index(drop=True)
        )

    demand_regions = (
        sorted(demand_df["region"].astype(str).unique())
        if not demand_df.empty
        else [default_region_id]
    )
    units_df = _default_units(demand_regions)
    unit_regions = sorted(
        {
            str(region)
            for region in units_df["region"].dropna().astype(str).unique()
        }
    ) if not units_df.empty else []

    if unit_regions:
        missing_regions = [
            region for region in demand_regions if region not in unit_regions
        ]
        if missing_regions:
            LOGGER.warning(
                "Default unit catalog missing coverage for demand regions: %s",
                ", ".join(missing_regions),
            )
            filtered_demand = demand_df[demand_df["region"].isin(unit_regions)]
            if filtered_demand.empty:
                fallback_region = unit_regions[0]
                demand_df = _fallback_demand_frame(fallback_region)
            else:
                demand_df = (
                    filtered_demand.sort_values(["year", "region"]).reset_index(
                        drop=True
                    )
                )

            demand_regions = (
                sorted(demand_df["region"].astype(str).unique())
                if not demand_df.empty
                else [unit_regions[0]]
            )
            units_df = units_df[units_df["region"].isin(demand_regions)]
            if units_df.empty:
                units_df = _default_units(demand_regions)
    else:
        LOGGER.warning(
            "Default unit catalog did not provide any units for regions: %s. "
            "Falling back to single-region assumptions.",
            ", ".join(demand_regions) or default_region_id,
        )
        fallback_units = _default_units([])
        if fallback_units.empty:
            fallback_region = default_region_id
            demand_df = _fallback_demand_frame(fallback_region)
            demand_regions = (
                sorted(demand_df["region"].astype(str).unique())
                if not demand_df.empty
                else [fallback_region]
            )
            units_df = fallback_units
        else:
            fallback_regions = sorted(
                fallback_units["region"].dropna().astype(str).unique()
            )
            if fallback_regions:
                demand_frames = [
                    _fallback_demand_frame(region) for region in fallback_regions
                ]
                demand_df = pd.concat(demand_frames, ignore_index=True)
            else:
                demand_df = _fallback_demand_frame(default_region_id)

            if not demand_df.empty:
                demand_df = (
                    demand_df.sort_values(["year", "region"]).reset_index(drop=True)
                )

            demand_regions = (
                sorted(demand_df["region"].astype(str).unique())
                if not demand_df.empty
                else fallback_regions or [default_region_id]
            )
            units_df = fallback_units[fallback_units["region"].isin(demand_regions)]
    fuels_df = _default_fuels(units_df)
    transmission_df = _default_transmission(demand_regions)
    base_frames = {
        "load": load_df,
        "units": units_df,
        "demand": demand_df,
        "fuels": fuels_df,
        "transmission": transmission_df,
    }
    return frames_cls(
        base_frames,
        carbon_policy_enabled=carbon_policy_enabled,
        banking_enabled=banking_enabled,
        carbon_price_schedule=carbon_price_schedule,
    )


def _ensure_years_in_demand(frames: FramesType, years: Iterable[int]) -> FramesType:
    if not years:
        return frames

    demand = frames.demand()
    if demand.empty:
        raise ValueError("Demand frame is empty; cannot infer loads for requested years")

    existing_years = {int(year) for year in demand["year"].unique()}
    target_years = {int(year) for year in years}
    missing = sorted(target_years - existing_years)
    if not missing:
        return frames

    averages = demand.groupby("region")["demand_mwh"].mean()
    new_rows: list[dict[str, Any]] = []
    for year in missing:
        for region, value in averages.items():
            new_rows.append({"year": year, "region": region, "demand_mwh": float(value)})

    demand_updated = pd.concat([demand, pd.DataFrame(new_rows)], ignore_index=True)
    demand_updated = demand_updated.sort_values(["year", "region"]).reset_index(drop=True)
    return frames.with_frame("demand", demand_updated)


def _temporary_output_directory(prefix: str = "bluesky_gui_") -> Path:
    """Create a writable temporary directory for engine CSV outputs.

    Some execution environments (notably restricted containers) provide a
    read-only ``/tmp``.  ``tempfile.mkdtemp`` raises :class:`PermissionError`
    in those cases which previously caused CSV exports to silently fail.  To
    keep the download buttons working we attempt a small set of candidate
    locations and fall back to a project specific directory under the current
    working directory or the user's home directory.
    """

    candidates: list[Path] = []

    override = os.environ.get("GRANITELEDGER_TMPDIR")
    if override:
        candidates.append(Path(override).expanduser())

    candidates.append(Path(tempfile.gettempdir()))
    candidates.append(Path.cwd() / ".graniteledger" / "tmp")

    home = Path.home()
    if home:
        candidates.append(home / ".graniteledger" / "tmp")

    tried: list[tuple[Path, Exception]] = []
    seen: set[Path] = set()
    for base_dir in candidates:
        if base_dir in seen:
            continue
        seen.add(base_dir)

        try:
            base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            tried.append((base_dir, exc))
            continue

        try:
            return Path(tempfile.mkdtemp(prefix=prefix, dir=str(base_dir)))
        except OSError as exc:
            tried.append((base_dir, exc))
            continue

    error_detail = "; ".join(f"{path}: {exc}" for path, exc in tried) or "no candidates available"
    raise RuntimeError(f"Unable to create temporary output directory ({error_detail}).")


def _write_outputs_to_temp(outputs) -> tuple[Path, dict[str, bytes]]:
    temp_dir = _temporary_output_directory()
    # Expect outputs to expose to_csv(target_dir)
    if hasattr(outputs, "to_csv"):
        try:
            outputs.to_csv(temp_dir)
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
    else:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise TypeError("Runner outputs object does not implement to_csv(Path).")
    csv_files: dict[str, bytes] = {}
    for csv_path in temp_dir.glob("*.csv"):
        csv_files[csv_path.name] = csv_path.read_bytes()
    return temp_dir, csv_files


def _extract_output_dataframe(outputs: Any, names: Sequence[str]) -> pd.DataFrame:
    """Return a DataFrame from ``outputs`` matching one of ``names``.

    The engine historically exposed results as :class:`EngineOutputs` with
    attributes named ``annual``, ``emissions_by_region`` and so on.  Some
    development branches temporarily renamed these attributes which broke the
    GUI.  This helper provides a resilient lookup that supports both the
    canonical names and any temporary aliases.  When a name cannot be resolved
    an empty DataFrame is returned so the UI can still render informative
    placeholders instead of failing outright.
    """

    for name in names:
        candidate: Any | None = None
        if hasattr(outputs, name):
            candidate = getattr(outputs, name)
        elif isinstance(outputs, Mapping):
            candidate = outputs.get(name)

        if isinstance(candidate, pd.DataFrame):
            return candidate
        if candidate is None:
            continue

        if isinstance(candidate, pd.Series):
            return candidate.to_frame().reset_index(drop=False)

        if isinstance(candidate, Mapping):
            # ``pd.DataFrame`` cannot coerce dictionaries of scalars directly – a
            # frequent pattern for single-region dispatch results.  Attempt an
            # index-oriented conversion before falling back to the generic
            # constructor so we can still surface the data in the UI.
            try:
                coerced = pd.DataFrame(candidate)
            except Exception:
                try:
                    coerced = pd.DataFrame.from_dict(candidate, orient="index")
                except Exception:  # pragma: no cover - defensive guard
                    LOGGER.warning(
                        "Unable to coerce mapping output field '%s' to a DataFrame.",
                        name,
                    )
                    continue
                else:
                    return coerced.reset_index(drop=False)
            else:
                return coerced

        try:
            coerced = pd.DataFrame(candidate)
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.warning(
                "Unable to coerce engine output field '%s' to a DataFrame.", name
            )
            continue
        else:
            return coerced

    LOGGER.warning(
        "Engine runner outputs missing expected field(s): %s", ", ".join(names)
    )
    return pd.DataFrame()


def _normalize_dispatch_price_frame(
    price_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Return a price DataFrame with best-effort column normalisation.

    Engine refactors occasionally rename the dispatch price fields or return
    mappings that are awkward to coerce into :class:`pandas.DataFrame`
    instances.  The GUI previously assumed the canonical ``['year', 'region',
    'price']`` schema which caused otherwise valid single-region outputs to be
    treated as empty.  This helper performs a defensive normalisation step so
    the UI can render whatever data is available while signalling missing
    columns to the caller.
    """

    if not isinstance(price_df, pd.DataFrame) or price_df.empty:
        return pd.DataFrame(), {"year": False, "region": False, "price": False}

    df = price_df.copy()

    # Promote index labels to columns when possible.  Many historical outputs
    # stored the region name in the index rather than an explicit column.
    if df.index.name or (getattr(df.index, "names", None) and any(df.index.names)):
        df = df.reset_index(drop=False)

    alias_map: dict[str, tuple[str, ...]] = {
        "year": ("year", "period", "calendar_year"),
        "region": ("region", "regions", "zone", "market", "node", "index"),
        "price": (
            "price",
            "value",
            "cost",
            "marginal_cost",
            "dispatch_price",
            "dispatch_cost",
        ),
    }

    rename_map: dict[str, str] = {}
    lower_lookup = {col.lower(): col for col in df.columns}
    for canonical, aliases in alias_map.items():
        for alias in aliases:
            column = lower_lookup.get(alias.lower())
            if column is not None:
                rename_map[column] = canonical
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    # When the price column is missing but only a single numeric column is
    # available, assume it represents the dispatch price.
    if "price" not in df.columns:
        numeric_columns = [
            col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
        ]
        if len(numeric_columns) == 1:
            df = df.rename(columns={numeric_columns[0]: "price"})

    # If region data is absent but the DataFrame now contains a generic
    # ``index`` column from reset_index(), interpret it as the region label.
    if "region" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "region"})

    field_flags = {key: (key in df.columns) for key in ("year", "region", "price")}
    return df, field_flags


def _with_legacy_carbon_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with legacy ``p_co2*`` columns populated when possible."""

    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    legacy_columns: dict[str, pd.Series] = {}

    if "cp_effective" in df.columns:
        series = df["cp_effective"]
        if "p_co2" not in df.columns:
            legacy_columns["p_co2"] = series
        if "p_co2_eff" not in df.columns:
            legacy_columns["p_co2_eff"] = series

    if "cp_all" in df.columns and "p_co2_all" not in df.columns:
        legacy_columns["p_co2_all"] = df["cp_all"]

    if "cp_exempt" in df.columns and "p_co2_exc" not in df.columns:
        legacy_columns["p_co2_exc"] = df["cp_exempt"]

    if not legacy_columns:
        return df

    result = df.copy()
    for column, series in legacy_columns.items():
        result[column] = series
    return result


def _read_uploaded_dataframe(uploaded_file: Any | None) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None

    try:
        if hasattr(uploaded_file, "getvalue"):
            raw = uploaded_file.getvalue()
        elif hasattr(uploaded_file, "read"):
            raw = uploaded_file.read()
        else:
            raw = uploaded_file
    except Exception as exc:
        _ensure_streamlit()
        st.error(f"Unable to read CSV: {exc}")
        return None

    if isinstance(raw, bytes):
        raw_bytes = raw
    elif isinstance(raw, str):
        raw_bytes = raw.encode("utf-8")
    else:
        raw_bytes = str(raw).encode("utf-8")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as handle:
            handle.write(raw_bytes)
            temp_path = Path(handle.name)

        df = load_user_demand_csv(temp_path)
    except ValueError as exc:
        _ensure_streamlit()
        st.error(str(exc))
        return None
    except Exception as exc:
        _ensure_streamlit()
        st.error(f"Unable to read CSV: {exc}")
        return None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass

    try:
        ei_df = load_ei_units()
    except Exception:
        ei_df = None

    _ensure_streamlit()
    try:
        selected_years = st.session_state.get("years") if "years" in st.session_state else None
        df, warns = validate_demand_table(df, ei_df, required_years=selected_years)
        df = df.rename(columns={"region_id": "region", "mwh": "demand_mwh"})
        for msg in warns:
            st.warning(msg)
    except DemandValidationError as e:
        st.error(e.message)
        if e.details:
            st.caption(str(e.details))
        return None

    if df.empty:
        st.warning("Uploaded CSV produced no demand rows.")

    return df


def _validate_frame_override(
    frames_obj: FramesType,
    frame_name: str,
    df: pd.DataFrame,
) -> tuple[FramesType | None, str | None]:
    validator_name = frame_name.lower()
    try:
        candidate = frames_obj.with_frame(frame_name, df)
        validator = getattr(candidate, validator_name, None)
        if callable(validator):
            validator()
        else:
            candidate.frame(frame_name)
        return candidate, None
    except Exception as exc:  # pragma: no cover
        return None, str(exc)


# -------------------------
# Assumptions editor tabs
# -------------------------
def _render_demand_controls(
    frames_obj: FramesType,
    years: Iterable[int],
) -> tuple[FramesType, list[str], list[str]]:  # pragma: no cover - UI helper
    _ensure_streamlit()

    notes: list[str] = []
    errors: list[str] = []
    frames_out = frames_obj

    demand_default = frames_obj.demand()
    if not demand_default.empty:
        st.caption("Current demand assumptions")
        st.dataframe(demand_default, use_container_width=True)
    else:
        st.info("No default demand data found. Provide values via the controls or upload a CSV.")

    manual_df: pd.DataFrame | None = None
    manual_note: str | None = None

    target_years = sorted({int(year) for year in years}) if years else []
    if not target_years and not demand_default.empty:
        target_years = sorted({int(year) for year in demand_default["year"].unique()})
    if not target_years:
        target_years = [int(date.today().year)]

    use_manual = st.checkbox("Create demand profile with controls", value=False, key="demand_manual_toggle")
    if use_manual:
        st.caption("Set a baseline load, per-region multipliers, and annual growth to construct demand.")
        if not demand_default.empty:
            first_year = target_years[0]
            base_year_data = demand_default[demand_default["year"] == first_year]
            default_base = float(base_year_data["demand_mwh"].mean()) if not base_year_data.empty else float(_DEFAULT_LOAD_MWH)
        else:
            default_base = float(_DEFAULT_LOAD_MWH)

        base_value = float(
            st.number_input(
                "Baseline demand for the first year (MWh)",
                min_value=0.0,
                value=max(0.0, default_base),
                step=10_000.0,
                format="%0.0f",
            )
        )
        growth_pct = float(
            st.slider(
                "Annual growth rate (%)",
                min_value=-20.0,
                max_value=20.0,
                value=0.0,
                step=0.25,
                key="demand_growth",
            )
        )

        if not demand_default.empty:
            region_labels = sorted({str(region) for region in demand_default["region"].unique()})
            region_defaults = (
                demand_default[demand_default["year"] == target_years[0]]
                .set_index("region")["demand_mwh"]
                .to_dict()
            )
        else:
            region_labels = [canonical_region_label(_get_default_region_id())]
            region_defaults = {}

        manual_records: list[dict[str, Any]] = []
        for region in region_labels:
            default_region_value = float(region_defaults.get(region, base_value or _DEFAULT_LOAD_MWH))
            multiplier_default = 1.0
            if base_value > 0.0:
                multiplier_default = default_region_value / base_value
            multiplier_default = float(max(0.1, min(3.0, multiplier_default)))

            multiplier = float(
                st.slider(
                    f"{region} demand multiplier",
                    min_value=0.1,
                    max_value=3.0,
                    value=multiplier_default,
                    step=0.05,
                    key=f"demand_scale_{region}",
                )
            )

            for index, year in enumerate(target_years):
                growth_factor = (1.0 + growth_pct / 100.0) ** index
                demand_val = base_value * multiplier * growth_factor
                manual_records.append(
                    {
                        "year": int(year),
                        "region": region,
                        "demand_mwh": float(demand_val),
                    }
                )

        manual_df = pd.DataFrame(manual_records)
        manual_note = (
            f"Demand constructed from GUI controls with baseline {base_value:,.0f} MWh, "
            f"growth {growth_pct:0.2f}% across {len(region_labels)} region(s) "
            f"and {len(target_years)} year(s)."
        )

    uploaded = st.file_uploader("Upload demand CSV", type="csv", key="demand_csv")
    if uploaded is not None:
        upload_df = _read_uploaded_dataframe(uploaded)
        if upload_df is not None:
            if manual_df is not None:
                st.info("Uploaded demand CSV overrides manual adjustments.")
                manual_df = None
                manual_note = None
            candidate, error = _validate_frame_override(frames_out, "demand", upload_df)
            if candidate is None:
                message = f"Demand CSV invalid: {error}"
                st.error(message)
                errors.append(message)
            else:
                frames_out = candidate
                notes.append(f"Demand table loaded from {uploaded.name} ({len(upload_df)} row(s)).")

    if manual_df is not None:
        candidate, error = _validate_frame_override(frames_out, "demand", manual_df)
        if candidate is None:
            message = f"Demand override invalid: {error}"
            st.error(message)
            errors.append(message)
        else:
            frames_out = candidate
            if manual_note:
                notes.append(manual_note)

    return frames_out, notes, errors


def _render_units_controls(frames_obj: FramesType) -> tuple[FramesType, list[str], list[str]]:  # pragma: no cover - UI helper
    _ensure_streamlit()

    notes: list[str] = []
    errors: list[str] = []
    frames_out = frames_obj

    try:
        units_default = frames_obj.units()
    except ValueError as exc:
        message = f"Units table invalid: {exc}"
        st.error(message)
        errors.append(message)
        units_default = pd.DataFrame()
    if not units_default.empty:
        st.caption("Current generating units")
        st.dataframe(units_default, use_container_width=True)
    else:
        st.info("No generating units are defined. Upload a CSV to provide unit characteristics.")

    manual_df: pd.DataFrame | None = None
    manual_note: str | None = None
    edit_inline = st.checkbox("Edit units inline", value=False, key="units_manual_toggle")
    if edit_inline and not units_default.empty:
        st.caption("Adjust unit properties with the controls below.")
        manual_records: list[dict[str, Any]] = []
        for index, row in units_default.iterrows():
            unit_label = str(row["unit_id"])
            st.markdown(f"**{unit_label}**")
            col_meta = st.columns(3)
            with col_meta[0]:
                unit_id = st.text_input(
                    "Unit ID",
                    value=unit_label,
                    key=f"units_unit_id_{index}",
                ).strip() or unit_label
            with col_meta[1]:
                region = st.text_input(
                    "Region",
                    value=str(row["region"]),
                    key=f"units_region_{index}",
                ).strip() or str(row["region"])
            with col_meta[2]:
                fuel = st.text_input(
                    "Fuel",
                    value=str(row["fuel"]),
                    key=f"units_fuel_{index}",
                ).strip() or str(row["fuel"])

            col_perf = st.columns(3)
            with col_perf[0]:
                cap_mw = st.number_input(
                    "Capacity (MW)",
                    min_value=0.0,
                    value=float(row["cap_mw"]),
                    step=1.0,
                    key=f"units_cap_{index}",
                )
            with col_perf[1]:
                availability = st.slider(
                    "Availability",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(row["availability"]),
                    step=0.01,
                    key=f"units_availability_{index}",
                )
            with col_perf[2]:
                heat_rate = st.number_input(
                    "Heat rate (MMBtu/MWh)",
                    min_value=0.0,
                    value=float(row["hr_mmbtu_per_mwh"]),
                    step=0.1,
                    key=f"units_heat_rate_{index}",
                )

            col_cost = st.columns(3)
            with col_cost[0]:
                vom = st.number_input(
                    "VOM ($/MWh)",
                    min_value=0.0,
                    value=float(row["vom_per_mwh"]),
                    step=0.1,
                    key=f"units_vom_{index}",
                )
            with col_cost[1]:
                fuel_price = st.number_input(
                    "Fuel price ($/MMBtu)",
                    min_value=0.0,
                    value=float(row["fuel_price_per_mmbtu"]),
                    step=0.1,
                    key=f"units_fuel_price_{index}",
                )
            with col_cost[2]:
                emission_factor = st.number_input(
                    "Emission factor (ton/MWh)",
                    min_value=0.0,
                    value=float(row["ef_ton_per_mwh"]),
                    step=0.01,
                    key=f"units_ef_{index}",
                )

            manual_records.append(
                {
                    "unit_id": unit_id,
                    "region": region,
                    "fuel": fuel,
                    "cap_mw": float(cap_mw),
                    "availability": float(availability),
                    "hr_mmbtu_per_mwh": float(heat_rate),
                    "vom_per_mwh": float(vom),
                    "fuel_price_per_mmbtu": float(fuel_price),
                    "ef_ton_per_mwh": float(emission_factor),
                }
            )

        manual_df = pd.DataFrame(manual_records)
        manual_note = f"Units modified via GUI controls ({len(manual_records)} unit(s))."
    elif edit_inline:
        st.info("Upload a units CSV to edit inline.")

    uploaded = st.file_uploader("Upload units CSV", type="csv", key="units_csv")
    if uploaded is not None:
        upload_df = _read_uploaded_dataframe(uploaded)
        if upload_df is not None:
            if manual_df is not None:
                st.info("Uploaded units CSV overrides inline edits.")
                manual_df = None
                manual_note = None
            candidate, error = _validate_frame_override(frames_out, "units", upload_df)
            if candidate is None:
                message = f"Units CSV invalid: {error}"
                st.error(message)
                errors.append(message)
            else:
                frames_out = candidate
                notes.append(f"Units loaded from {uploaded.name} ({len(upload_df)} row(s)).")

    if manual_df is not None:
        candidate, error = _validate_frame_override(frames_out, "units", manual_df)
        if candidate is None:
            message = f"Units override invalid: {error}"
            st.error(message)
            errors.append(message)
        else:
            frames_out = candidate
            if manual_note:
                notes.append(manual_note)

    return frames_out, notes, errors


def _render_fuels_controls(frames_obj: FramesType) -> tuple[FramesType, list[str], list[str]]:  # pragma: no cover - UI helper
    _ensure_streamlit()

    notes: list[str] = []
    errors: list[str] = []
    frames_out = frames_obj

    fuels_default = frames_obj.fuels()
    if not fuels_default.empty:
        st.caption("Current fuel coverage")
        st.dataframe(fuels_default, use_container_width=True)
    else:
        st.info("No fuel data available. Upload a CSV to specify fuel coverage.")

    manual_df: pd.DataFrame | None = None
    manual_note: str | None = None
    edit_inline = st.checkbox("Edit fuel coverage inline", value=False, key="fuels_manual_toggle")
    if edit_inline and not fuels_default.empty:
        st.caption("Toggle coverage and update emission factors as needed.")
        manual_records: list[dict[str, Any]] = []
        has_emission_column = "co2_short_ton_per_mwh" in fuels_default.columns
        for index, row in fuels_default.iterrows():
            fuel_label = str(row["fuel"])
            col_line = st.columns(3 if has_emission_column else 2)
            with col_line[0]:
                fuel_name = st.text_input(
                    "Fuel",
                    value=fuel_label,
                    key=f"fuels_name_{index}",
                ).strip() or fuel_label
            with col_line[1]:
                covered = st.checkbox(
                    "Covered",
                    value=bool(row["covered"]),
                    key=f"fuels_covered_{index}",
                )
            emission_value: float | None = None
            if has_emission_column:
                with col_line[2]:
                    emission_value = float(
                        st.number_input(
                            "CO₂ tons/MMBtu",
                            min_value=0.0,
                            value=float(row.get("co2_short_ton_per_mwh", 0.0)),
                            step=0.01,
                            key=f"fuels_emission_{index}",
                        )
                    )

            record: dict[str, Any] = {"fuel": fuel_name, "covered": bool(covered)}
            if has_emission_column:
                record["co2_short_ton_per_mwh"] = float(emission_value or 0.0)
            manual_records.append(record)

        manual_df = pd.DataFrame(manual_records)
        manual_note = f"Fuel coverage edited inline ({len(manual_records)} fuel(s))."
    elif edit_inline:
        st.info("Upload a fuels CSV to edit inline.")

    uploaded = st.file_uploader("Upload fuels CSV", type="csv", key="fuels_csv")
    if uploaded is not None:
        upload_df = _read_uploaded_dataframe(uploaded)
        if upload_df is not None:
            if manual_df is not None:
                st.info("Uploaded fuels CSV overrides inline edits.")
                manual_df = None
                manual_note = None
            candidate, error = _validate_frame_override(frames_out, "fuels", upload_df)
            if candidate is None:
                message = f"Fuels CSV invalid: {error}"
                st.error(message)
                errors.append(message)
            else:
                frames_out = candidate
                notes.append(f"Fuels loaded from {uploaded.name} ({len(upload_df)} row(s)).")

    if manual_df is not None:
        candidate, error = _validate_frame_override(frames_out, "fuels", manual_df)
        if candidate is None:
            message = f"Fuels override invalid: {error}"
            st.error(message)
            errors.append(message)
        else:
            frames_out = candidate
            if manual_note:
                notes.append(manual_note)

    return frames_out, notes, errors


def _render_transmission_controls(
    frames_obj: FramesType,
) -> tuple[FramesType, list[str], list[str]]:  # pragma: no cover - UI helper
    _ensure_streamlit()

    notes: list[str] = []
    errors: list[str] = []
    frames_out = frames_obj

    transmission_default = frames_obj.transmission()
    if not transmission_default.empty:
        st.caption("Current transmission limits")
        st.dataframe(transmission_default, use_container_width=True)
    else:
        st.info("No transmission limits specified. Add entries below or upload a CSV.")

    manual_df: pd.DataFrame | None = None
    manual_note: str | None = None
    edit_inline = st.checkbox("Edit transmission limits inline", value=False, key="transmission_manual_toggle")
    if edit_inline:
        editable = transmission_default.copy()
        if editable.empty:
            editable = pd.DataFrame(columns=["from_region", "to_region", "limit_mw"])
        st.caption("Use the table to add or modify directional flow limits (MW).")
        edited = st.data_editor(
            editable,
            num_rows="dynamic",
            use_container_width=True,
            key="transmission_editor",
        )
        manual_df = (edited.copy() if isinstance(edited, pd.DataFrame) else pd.DataFrame(edited)).dropna(how="all")
        manual_df = manual_df.reindex(columns=["from_region", "to_region", "limit_mw"])
        manual_note = f"Transmission table edited inline ({len(manual_df)} record(s))."

    uploaded = st.file_uploader("Upload transmission CSV", type="csv", key="transmission_csv")
    if uploaded is not None:
        upload_df = _read_uploaded_dataframe(uploaded)
        if upload_df is not None:
            if manual_df is not None:
                st.info("Uploaded transmission CSV overrides inline edits.")
                manual_df = None
                manual_note = None
            candidate, error = _validate_frame_override(frames_out, "transmission", upload_df)
            if candidate is None:
                message = f"Transmission CSV invalid: {error}"
                st.error(message)
                errors.append(message)
            else:
                frames_out = candidate
                notes.append(
                    f"Transmission limits loaded from {uploaded.name} ({len(upload_df)} row(s))."
                )

    if manual_df is not None:
        candidate, error = _validate_frame_override(frames_out, "transmission", manual_df)
        if candidate is None:
            message = f"Transmission override invalid: {error}"
            st.error(message)
            errors.append(message)
        else:
            frames_out = candidate
            if manual_note:
                notes.append(manual_note)

    return frames_out, notes, errors


# -------------------------
# Runner
# -------------------------
def _build_run_summary(
    params: Mapping[str, Any] | None,
    *,
    config_label: str | None = None,
) -> list[tuple[str, str]]:
    summary: list[tuple[str, str]] = []

    if config_label:
        summary.append(("Configuration", config_label))

    if not isinstance(params, Mapping):
        return summary

    def _coerce_int(value: object) -> int | None:
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    def _enabled_label(flag: object, *, true: str = "Enabled", false: str = "Disabled") -> str:
        return true if bool(flag) else false

    start_year = _coerce_int(params.get("start_year"))
    end_year = _coerce_int(params.get("end_year"))

    if start_year is not None and end_year is not None:
        if start_year == end_year:
            summary.append(("Simulation years", str(start_year)))
        else:
            total_years = max(0, end_year - start_year + 1)
            years_label = f"{start_year}–{end_year}"
            if total_years > 0:
                years_label = f"{years_label} ({total_years} year(s))"
            summary.append(("Simulation years", years_label))
    elif start_year is not None:
        summary.append(("Simulation start year", str(start_year)))
    elif end_year is not None:
        summary.append(("Simulation end year", str(end_year)))

    carbon_enabled = params.get("carbon_policy_enabled")
    summary.append(("Carbon policy", _enabled_label(carbon_enabled)))

    if carbon_enabled:
        summary.append(("Price floor", _enabled_label(params.get("enable_floor"))))
        ccr_enabled = params.get("enable_ccr")
        summary.append(("Cost containment reserve", _enabled_label(ccr_enabled)))
        if ccr_enabled:
            ccr_triggers: list[str] = []
            if params.get("ccr1_enabled"):
                ccr_triggers.append("CCR1")
            if params.get("ccr2_enabled"):
                ccr_triggers.append("CCR2")
            if ccr_triggers:
                summary.append(("CCR triggers", ", ".join(ccr_triggers)))
        summary.append(
            (
                "Allowance banking",
                _enabled_label(params.get("allowance_banking_enabled"), true="Allowed", false="Not allowed"),
            )
        )
        control_period = _coerce_int(params.get("control_period_years"))
        if control_period:
            summary.append(("Control period", f"{control_period} year(s)"))

    dispatch_network = params.get("dispatch_use_network")
    if dispatch_network is not None:
        summary.append(
            (
                "Electricity dispatch",
                "Network" if bool(dispatch_network) else "Zonal",
            )
        )

    capacity_toggle = params.get("dispatch_capacity_expansion")
    if capacity_toggle is not None:
        summary.append(
            (
                "Capacity expansion",
                _enabled_label(capacity_toggle, true="Enabled", false="Disabled"),
            )
        )

    module_config = params.get("module_config")
    if isinstance(module_config, Mapping):
        enabled_modules: list[str] = []
        disabled_modules: list[str] = []
        for raw_name, settings in module_config.items():
            name = str(raw_name)
            if isinstance(settings, Mapping):
                enabled = settings.get("enabled", True)
            else:
                enabled = bool(settings)
            label = name.replace("_", " ").strip().title() or name
            if bool(enabled):
                enabled_modules.append(label)
            else:
                disabled_modules.append(label)
        if enabled_modules:
            summary.append(("Modules enabled", ", ".join(sorted(enabled_modules))))
        if disabled_modules:
            summary.append(("Modules disabled", ", ".join(sorted(disabled_modules))))

    return summary

def _validate_loaded_demand(
    frames_obj: FramesType,
    manifests: Sequence[_ScenarioSelection],
    *,
    base_path: str | os.PathLike[str] | None,
) -> None:
    """Ensure the demand frame reflects the selected load forecast manifests."""

    if not manifests:
        return

    demand_df: pd.DataFrame | None = None
    try:
        if hasattr(frames_obj, "optional_frame"):
            demand_df = frames_obj.optional_frame("demand")  # type: ignore[attr-defined]
    except Exception:
        demand_df = None

    if demand_df is None:
        try:
            demand_df = frames_obj.demand()
        except Exception:
            demand_df = None

    if demand_df is None or demand_df.empty:
        root = _coerce_base_path(base_path)
        for manifest in manifests:
            iso_label = getattr(manifest, "iso", "")
            scenario_label = getattr(manifest, "scenario", "")
            if not iso_label or not scenario_label:
                continue
            try:
                scenario_df = _load_iso_scenario_frame(root, iso_label, scenario_label)
            except Exception:
                LOGGER.exception(
                    "Failed to load demand forecast for %s/%s from %s",
                    iso_label,
                    scenario_label,
                    root,
                )
                continue
            if scenario_df.empty:
                LOGGER.error(
                    "Demand forecast frame empty for %s/%s under %s",
                    iso_label,
                    scenario_label,
                    root,
                )
        raise AssertionError(
            "Demand frame is empty after applying selected load forecasts; "
            "check the load forecast inputs."
        )

    demand_years = {
        int(year)
        for year in pd.to_numeric(demand_df.get("year"), errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    }
    expected_years = {
        int(year)
        for manifest in manifests
        for year in getattr(manifest, "years", []) or []
    }
    missing_years = sorted(expected_years - demand_years)
    if missing_years:
        LOGGER.warning(
            "Demand frame missing years from load forecasts: %s",
            ", ".join(str(year) for year in missing_years),
        )

    demand_regions = {
        str(region).strip()
        for region in demand_df.get("region", pd.Series(dtype="string"))
        .dropna()
        .astype(str)
        .tolist()
        if str(region).strip()
    }
    expected_zones = {
        str(zone).strip()
        for manifest in manifests
        for zone in getattr(manifest, "zones", []) or []
        if str(zone).strip()
    }
    missing_zones = sorted(zone for zone in expected_zones if zone not in demand_regions)
    if missing_zones:
        LOGGER.warning(
            "Demand frame missing zones from load forecasts: %s",
            ", ".join(missing_zones),
        )


def run_policy_simulation(
    config_source: Any | None,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    carbon_policy_enabled: bool = True,
    enable_floor: bool = True,
    enable_ccr: bool = True,
    ccr1_enabled: bool = True,
    ccr2_enabled: bool = True,
    ccr1_price: float | None = None,
    ccr2_price: float | None = None,
    ccr1_escalator_pct: float | None = None,
    ccr2_escalator_pct: float | None = None,
    allowance_banking_enabled: bool = True,
    initial_bank: float = 0.0,
    coverage_regions: Iterable[str] | None = None,
    control_period_years: int | None = None,
    cap_regions: Sequence[Any] | None = None,
    carbon_price_enabled: bool | None = None,
    carbon_price_value: float | None = None,
    carbon_price_schedule: Mapping[int, float] | Mapping[str, Any] | None = None,
    carbon_price_escalator_pct: float | None = None,
    carbon_cap_start_value: float | None = None,
    carbon_cap_reduction_mode: str | None = None,
    carbon_cap_reduction_value: float | None = None,
    carbon_cap_schedule: Mapping[int, float] | Mapping[str, Any] | None = None,
    price_floor_value: float | None = None,
    price_floor_escalator_mode: str | None = None,
    price_floor_escalator_value: float | None = None,
    price_floor_schedule: Mapping[int, float] | Mapping[str, Any] | None = None,
    dispatch_use_network: bool = False,
    dispatch_capacity_expansion: bool | None = None,
    deep_carbon_pricing: bool = False,
    module_config: Mapping[str, Any] | None = None,
    frames: FramesType | Mapping[str, pd.DataFrame] | None = None,
    assumption_notes: Iterable[str] | None = None,
    forecast_bundles: Sequence[_ScenarioSelection] | None = None,
    run_id: str | None = None,
    progress_cb: Callable[[str, Mapping[str, object]], None] | None = None,
    stage_cb: Callable[[str, Mapping[str, object]], None] | None = None,
) -> dict[str, Any]:




    policy_override = bool(carbon_policy_enabled)
    if carbon_price_enabled is None:
        price_override: bool | None = None
    else:
        price_override = bool(carbon_price_enabled)

    try:
        config = _load_config_data(config_source)
    except Exception as exc:  # pragma: no cover
        return {"error": f"Unable to load configuration: {exc}"}

    config.setdefault("modules", {})


    try:
        base_years = _years_from_config(config)
        years = _select_years(base_years, start_year, end_year)
    except Exception as exc:
        return {"error": f"Invalid year selection: {exc}"}

    merged_modules = _merge_module_dicts(config.get("modules"), module_config)
    demand_module_cfg = merged_modules.get("demand", {}) or {}
    custom_forecast_map: Mapping[str, Any] | None = None
    if isinstance(demand_module_cfg, Mapping):
        custom_candidate = demand_module_cfg.get("custom_load_forecasts")
        if isinstance(custom_candidate, Mapping):
            custom_forecast_map = custom_candidate
    selection_mapping = _canonical_forecast_selection(
        demand_module_cfg.get("load_forecasts") if isinstance(demand_module_cfg, Mapping) else None
    )
    
    
    forecast_base_path = _resolve_forecast_base_path()
    scenario_selections: list[_ScenarioSelection] = []
    forecast_manifest_records: list[dict[str, Any]] = []
    if forecast_bundles is not None:
        scenario_selections = _coerce_manifest_list(forecast_bundles)
        if scenario_selections:
            forecast_manifest_records = _forecast_bundles_from_selection(
                scenario_selections,
                base_path=forecast_base_path,
            )
    elif selection_mapping:
        frame = _cached_forecast_frame(forecast_base_path)
        scenario_selections = _manifests_from_selection(
            selection_mapping,
            frame=frame,
            base_path=forecast_base_path,
        )
        if scenario_selections:
            forecast_manifest_records = _forecast_bundles_from_selection(
                scenario_selections,
                base_path=forecast_base_path,
            )


    dispatch_defaults = merged_modules.get("electricity_dispatch", {})
    if deep_carbon_pricing is None:
        deep_carbon_flag = bool(dispatch_defaults.get("deep_carbon_pricing", False))
    else:
        deep_carbon_flag = bool(deep_carbon_pricing)

    carbon_policy_cfg = CarbonPolicyConfig.from_mapping(
        merged_modules.get("carbon_policy"),
        enabled=policy_override,
        enable_floor=enable_floor,
        enable_ccr=enable_ccr,
        ccr1_enabled=ccr1_enabled,
        ccr2_enabled=ccr2_enabled,
        ccr1_price=ccr1_price,
        ccr2_price=ccr2_price,
        ccr1_escalator_pct=ccr1_escalator_pct,
        ccr2_escalator_pct=ccr2_escalator_pct,
        allowance_banking_enabled=allowance_banking_enabled,
        control_period_years=control_period_years,
    )

    price_cfg = CarbonPriceConfig.from_mapping(
        merged_modules.get("carbon_price"),
        enabled=price_override,
        value=carbon_price_value,
        schedule=carbon_price_schedule,
        years=years,
        escalator_pct=carbon_price_escalator_pct,
    )

    # Expand any provided carbon price schedule across all modeled years
    if price_cfg.schedule:
        expanded_schedule = _expand_or_build_price_schedule(
            price_cfg.schedule,
            years,
        )
        price_cfg.schedule = expanded_schedule if expanded_schedule else {}

    # Track whether cap constraints were explicitly requested
    explicit_cap_request = (coverage_regions is not None) or (cap_regions is not None)

    if price_cfg.active:
        if (
            carbon_policy_cfg.enabled
            and policy_override
            and explicit_cap_request
            and not deep_carbon_pricing
        ):
            return {
                "error": "Cannot enable both carbon cap and carbon price simultaneously."
            }
        if not deep_carbon_pricing:
            carbon_policy_cfg.disable_for_price()

    normalized_coverage = _normalize_coverage_selection(
        coverage_regions
        if coverage_regions is not None
        else merged_modules.get("carbon_policy", {}).get("coverage_regions", ["All"])
    )



    policy_enabled = bool(carbon_policy_cfg.enabled)
    floor_flag = bool(policy_enabled and carbon_policy_cfg.enable_floor)
    ccr_flag = bool(
        policy_enabled
        and carbon_policy_cfg.enable_ccr
        and (carbon_policy_cfg.ccr1_enabled or carbon_policy_cfg.ccr2_enabled)
    )
    banking_flag = bool(policy_enabled and carbon_policy_cfg.allowance_banking_enabled)

    raw_initial_bank = _coerce_float(initial_bank, default=0.0)

    market_cfg = config.get("allowance_market")
    bank0_from_config: float | None = None
    if isinstance(market_cfg, Mapping):
        bank0_from_config = _coerce_float(market_cfg.get("bank0"), default=0.0)
        if not banking_flag and bank0_from_config > 0.0:
            LOGGER.warning(
                "Allowance banking disabled; ignoring initial bank of %.3f tons.",
                bank0_from_config,
            )
            try:
                market_cfg["bank0"] = 0.0  # type: ignore[index]
            except Exception:  # pragma: no cover - best effort for immutable mappings
                pass

    if not banking_flag and raw_initial_bank > 0.0:
        LOGGER.warning(
            "Allowance banking disabled; ignoring requested initial bank of %.3f tons.",
            raw_initial_bank,
        )
        raw_initial_bank = 0.0

    carbon_record = merged_modules.setdefault("carbon_policy", {})
    initial_bank_value = raw_initial_bank if banking_flag else 0.0

    cap_start_value_norm: float | None
    if carbon_cap_start_value is not None:
        try:
            cap_start_value_norm = float(carbon_cap_start_value)
        except (TypeError, ValueError):
            cap_start_value_norm = None
    else:
        existing_cap_start = carbon_record.get("cap_start_value")
        try:
            cap_start_value_norm = (
                float(existing_cap_start) if existing_cap_start is not None else None
            )
        except (TypeError, ValueError):
            cap_start_value_norm = None

    raw_mode = (
        carbon_cap_reduction_mode
        if carbon_cap_reduction_mode is not None
        else carbon_record.get("cap_reduction_mode")
    )
    cap_reduction_mode_norm = str(raw_mode or "percent").strip().lower()
    if cap_reduction_mode_norm not in {"percent", "fixed"}:
        cap_reduction_mode_norm = "percent"

    if carbon_cap_reduction_value is not None:
        try:
            cap_reduction_value_norm = float(carbon_cap_reduction_value)
        except (TypeError, ValueError):
            cap_reduction_value_norm = 0.0
    else:
        existing_reduction = carbon_record.get("cap_reduction_value", 0.0)
        try:
            cap_reduction_value_norm = float(existing_reduction)
        except (TypeError, ValueError):
            cap_reduction_value_norm = 0.0

    floor_base_value = price_floor.parse_currency_value(
        price_floor_value if price_floor_value is not None else carbon_policy_cfg.floor_value,
        carbon_policy_cfg.floor_value,
    )
    floor_mode_norm = str(
        price_floor_escalator_mode
        if price_floor_escalator_mode is not None
        else carbon_policy_cfg.floor_escalator_mode
    ).strip().lower()
    if floor_mode_norm not in {"fixed", "percent"}:
        floor_mode_norm = "fixed"
    escalator_input = (
        price_floor_escalator_value
        if price_floor_escalator_value is not None
        else carbon_policy_cfg.floor_escalator_value
    )
    if floor_mode_norm == "percent":
        floor_escalator_norm = price_floor.parse_percentage_value(
            escalator_input, carbon_policy_cfg.floor_escalator_value
        )
    else:
        floor_escalator_norm = price_floor.parse_currency_value(
            escalator_input, carbon_policy_cfg.floor_escalator_value
        )
    provided_floor_schedule = price_floor_schedule
    if not isinstance(provided_floor_schedule, Mapping) or not provided_floor_schedule:
        provided_floor_schedule = carbon_record.get("floor_schedule")
    if (
        (not isinstance(provided_floor_schedule, Mapping) or not provided_floor_schedule)
        and isinstance(market_cfg, Mapping)
    ):
        provided_floor_schedule = market_cfg.get("floor")
    provided_schedule_map: dict[int, float] = {}
    if isinstance(provided_floor_schedule, Mapping):
        for year_key, value in provided_floor_schedule.items():
            try:
                year_int = int(year_key)
                provided_schedule_map[year_int] = float(value)
            except (TypeError, ValueError):
                continue
    baseline_floor_schedule: dict[int, float] = {}
    if policy_enabled and floor_flag:
        baseline_floor_schedule = price_floor.build_schedule(
            years,
            floor_base_value,
            floor_mode_norm,
            floor_escalator_norm,
        )
    if provided_schedule_map:
        baseline_floor_schedule.update(provided_schedule_map)
    floor_schedule_map = dict(sorted(baseline_floor_schedule.items())) if baseline_floor_schedule else {}
    carbon_record.update(
        {
            "enabled": policy_enabled,
            "enable_floor": floor_flag,
            "enable_ccr": ccr_flag,
            "ccr1_enabled": bool(carbon_policy_cfg.ccr1_enabled) if ccr_flag else False,
            "ccr2_enabled": bool(carbon_policy_cfg.ccr2_enabled) if ccr_flag else False,
            "allowance_banking_enabled": banking_flag,
            "coverage_regions": normalized_coverage,
            "control_period_years": (
                carbon_policy_cfg.control_period_years if policy_enabled else None
            ),
            "bank0": initial_bank_value,
            "cap_start_value": cap_start_value_norm,
            "cap_reduction_mode": cap_reduction_mode_norm,
            "cap_reduction_value": float(cap_reduction_value_norm),
            "floor_value": float(floor_base_value),
            "floor_escalator_mode": str(floor_mode_norm),
            "floor_escalator_value": float(floor_escalator_norm),
        }
    )
    # Carbon price schedule will be populated after years are finalised below
    if floor_schedule_map:
        carbon_record["floor_schedule"] = dict(sorted(floor_schedule_map.items()))
    else:
        carbon_record.pop("floor_schedule", None)

    merged_modules["carbon_price"] = price_cfg.as_dict()

    raw_run_years = []
    try:
        raw_run_years = [int(year) for year in years]
    except Exception:
        raw_run_years = []
    run_years_sorted = sorted(dict.fromkeys(raw_run_years))

    provided_price_schedule: dict[int, float] = {}
    if isinstance(price_cfg.schedule, Mapping):
        for year, value in price_cfg.schedule.items():
            try:
                provided_price_schedule[int(year)] = float(value)
            except (TypeError, ValueError):
                continue

    price_schedule_map: dict[int, float] = {}
    if price_cfg.active:
        if run_years_sorted:
            start_year = run_years_sorted[0]
            end_year = run_years_sorted[-1]
        elif provided_price_schedule:
            start_year = min(provided_price_schedule)
            end_year = max(provided_price_schedule)
        else:
            start_year = end_year = None

        if start_year is not None and end_year is not None:
            base_year = min(provided_price_schedule) if provided_price_schedule else start_year
            base_price = provided_price_schedule.get(base_year, price_cfg.price_per_ton)
            try:
                base_price_float = float(base_price)
            except (TypeError, ValueError):
                base_price_float = 0.0

            growth_pct = float(price_cfg.escalator_pct)

            # Back-adjust if provided base year is later than start_year
            if base_year is not None and base_year != start_year:
                ratio = 1.0 + (growth_pct or 0.0) / 100.0
                try:
                    steps = base_year - start_year
                    if ratio != 0.0:
                        base_price_float = base_price_float / (ratio ** steps)
                except OverflowError:
                    base_price_float = 0.0

            generated_schedule = _build_price_schedule(
                start_year,
                end_year,
                base_price_float,
                growth_pct,
            )

            # Lock schedule to run_years if explicitly provided
            if run_years_sorted:
                generated_schedule = {
                    year: generated_schedule.get(year, base_price_float)
                    for year in run_years_sorted
                }

            # Overlay provided overrides
            for year, value in provided_price_schedule.items():
                generated_schedule[int(year)] = float(value)

            price_schedule_map = dict(sorted(generated_schedule.items()))

    price_active = bool(price_cfg.active and price_schedule_map)


    cap_schedule_map: dict[int, float] = {}
    if isinstance(carbon_cap_schedule, Mapping):
        for year_key, value in carbon_cap_schedule.items():
            try:
                year_int = int(year_key)
                cap_schedule_map[year_int] = float(value)
            except (TypeError, ValueError):
                continue
    if policy_enabled and not cap_schedule_map and cap_start_value_norm is not None:
        cap_schedule_map = _build_cap_reduction_schedule(
            cap_start_value_norm,
            cap_reduction_mode_norm,
            cap_reduction_value_norm,
            years,
        )
    if cap_schedule_map:
        carbon_record["cap_schedule"] = dict(sorted(cap_schedule_map.items()))
    else:
        carbon_record.pop("cap_schedule", None)
    allowance_market_record = merged_modules.setdefault("allowance_market", {})
    allowance_market_record["enabled"] = policy_enabled
    allowance_market_record["bank0"] = float(initial_bank_value)
    allowance_market_record["ccr1_enabled"] = bool(
        carbon_policy_cfg.ccr1_enabled if ccr_flag else False
    )
    allowance_market_record["ccr2_enabled"] = bool(
        carbon_policy_cfg.ccr2_enabled if ccr_flag else False
    )
    if policy_enabled and cap_schedule_map:
        allowance_market_record["cap"] = dict(sorted(cap_schedule_map.items()))
    elif not policy_enabled:
        allowance_market_record.pop("cap", None)

    if policy_enabled and floor_schedule_map:
        allowance_market_record["floor"] = dict(sorted(floor_schedule_map.items()))
    elif not policy_enabled or not floor_flag:
        allowance_market_record.pop("floor", None)
    existing_allowance_cfg = config.get("allowance_market")
    if isinstance(existing_allowance_cfg, Mapping):
        allowance_config = dict(existing_allowance_cfg)
    else:
        allowance_config = {}
    allowance_config["enabled"] = policy_enabled
    allowance_config["bank0"] = float(initial_bank_value)
    allowance_config["ccr1_enabled"] = bool(
        carbon_policy_cfg.ccr1_enabled if ccr_flag else False
    )
    allowance_config["ccr2_enabled"] = bool(
        carbon_policy_cfg.ccr2_enabled if ccr_flag else False
    )
    if policy_enabled and cap_schedule_map:
        allowance_config["cap"] = dict(sorted(cap_schedule_map.items()))
    elif not policy_enabled:
        allowance_config.pop("cap", None)
    if policy_enabled and floor_schedule_map:
        allowance_config["floor"] = dict(sorted(floor_schedule_map.items()))
    elif not policy_enabled or not floor_flag:
        allowance_config.pop("floor", None)
    config["allowance_market"] = allowance_config
    try:
        normalized_regions, cap_region_aliases = _normalize_cap_region_entries(cap_regions)
    except ValueError as exc:
        return {"error": str(exc)}

    normalized_cap_regions: list[Any] = []
    cap_regions_all_selected = any(
        isinstance(entry, str) and entry.lower() == "all" for entry in (cap_regions or [])
    )
    if normalized_regions:
        carbon_record["regions"] = list(normalized_regions)
        normalized_cap_regions = list(normalized_regions)

    if not normalized_regions and normalized_coverage and normalized_coverage != ["All"]:
        normalized_regions = list(normalized_coverage)
        if normalized_regions:
            carbon_record["regions"] = list(normalized_regions)
            normalized_cap_regions = list(normalized_regions)
    elif normalized_coverage == ["All"]:
        cap_regions_all_selected = True

    cap_region_selection = list(normalized_regions)

    config["modules"] = merged_modules

    dispatch_record = merged_modules.setdefault("electricity_dispatch", {})
    capacity_setting = dispatch_record.get("capacity_expansion")
    if dispatch_capacity_expansion is not None:
        capacity_flag = bool(dispatch_capacity_expansion)
    elif capacity_setting is not None:
        capacity_flag = bool(capacity_setting)
    else:
        capacity_flag = True
    dispatch_record["capacity_expansion"] = capacity_flag
    dispatch_record["use_network"] = bool(dispatch_use_network)
    dispatch_record["deep_carbon_pricing"] = bool(deep_carbon_pricing)

    if capacity_flag:
        config["sw_expansion"] = 1
    else:
        config["sw_expansion"] = 0
        if config.get("sw_rm") not in (None, 0, False):
            config["sw_rm"] = 0


    def _coerce_year_range(start: int | None, end: int | None) -> list[int]:
        if start is None and end is None:
            return []
        if start is None:
            start = end
        if end is None:
            end = start
        assert start is not None and end is not None
        step = 1 if end >= start else -1
        return list(range(int(start), int(end) + step, step))

    years = _coerce_year_range(start_year, end_year)
    if not years:
        years = _years_from_config(config)
    if not years:
        fallback_year = start_year or end_year
        if fallback_year is not None:
            years = [int(fallback_year)]
        else:
            years = [int(date.today().year)]

    normalized_years = sorted({int(year) for year in years})
    if normalized_years:
        year_start = normalized_years[0]
        year_end = normalized_years[-1]
        years = list(range(year_start, year_end + 1))
    else:
        current_year = int(date.today().year)
        years = [current_year]

    config["years"] = list(years)
    config["start_year"] = int(years[0])
    config["end_year"] = int(years[-1])

    provided_price_schedule: dict[int, float] = {}
    if isinstance(price_cfg.schedule, Mapping):
        for year, value in price_cfg.schedule.items():
            try:
                provided_price_schedule[int(year)] = float(value)
            except (TypeError, ValueError):
                continue

    price_schedule_map: dict[int, float] = {}
    if price_cfg.enabled and years:
        run_years_sorted = sorted(dict.fromkeys(int(year) for year in years))
        if run_years_sorted:
            start_year_schedule = run_years_sorted[0]
            end_year_schedule = run_years_sorted[-1]
            base_year = (
                min(provided_price_schedule)
                if provided_price_schedule
                else start_year_schedule
            )
            base_price = provided_price_schedule.get(base_year, price_cfg.price_per_ton)
            try:
                base_price_float = float(base_price)
            except (TypeError, ValueError):
                base_price_float = 0.0

            growth_pct = float(price_cfg.escalator_pct)
            if base_year is not None and base_year != start_year_schedule:
                ratio = 1.0 + (growth_pct or 0.0) / 100.0
                try:
                    steps = base_year - start_year_schedule
                    if ratio != 0.0:
                        base_price_float = base_price_float / (ratio ** steps)
                except OverflowError:
                    base_price_float = 0.0

            generated_schedule = _build_price_schedule(
                start_year_schedule,
                end_year_schedule,
                base_price_float,
                growth_pct,
            )

            combined_schedule = dict(sorted(generated_schedule.items()))
            for year, value in provided_price_schedule.items():
                try:
                    combined_schedule[int(year)] = float(value)
                except (TypeError, ValueError):
                    continue

            combined_items = sorted(combined_schedule.items())
            if combined_items:
                first_price = float(combined_items[0][1])
            else:
                first_price = float(base_price_float)
            filled_schedule: dict[int, float] = {}
            last_price: float | None = None
            index = 0
            total_items = len(combined_items)

            for year in run_years_sorted:
                while index < total_items and combined_items[index][0] <= year:
                    last_price = float(combined_items[index][1])
                    index += 1
                if last_price is None:
                    last_price = first_price
                filled_schedule[year] = float(last_price)

            price_schedule_map = filled_schedule
            if price_schedule_map:
                LOGGER.debug("Using carbon price schedule: %s", price_schedule_map)

    price_active = bool(price_cfg.enabled and price_schedule_map)
    if price_active:
        price_cfg.schedule = dict(price_schedule_map)
    else:
        price_cfg.schedule = {}

    merged_modules["carbon_price"] = price_cfg.as_dict()

    carbon_price_for_frames: Mapping[int, float] | None = (
        price_schedule_map if price_active else None
    )

    bundles_for_frames: Sequence[_ScenarioSelection] | None = (
        scenario_selections or _selected_forecast_bundles_from_state()
    )

    if frames is None:
        frames_obj = _build_default_frames(
            years,
            carbon_policy_enabled=bool(policy_enabled),
            banking_enabled=bool(allowance_banking_enabled),
            carbon_price_schedule=carbon_price_for_frames,
            forecast_bundles=bundles_for_frames,
            custom_forecasts=custom_forecast_map,
        )
        demand_years: set[int] = set(years)
    else:
        frames_obj = Frames.coerce(
            frames,
            carbon_policy_enabled=bool(policy_enabled),
            banking_enabled=bool(allowance_banking_enabled),
            carbon_price_schedule=carbon_price_for_frames,
        )
        try:
            demand_years = {int(year) for year in frames_obj.demand()["year"].unique()}
        except Exception as exc:
            LOGGER.exception("Unable to read demand data from supplied frames")
            return {"error": f"Invalid demand data: {exc}"}

    requested_years = {int(year) for year in years}
    if frames is not None and demand_years and requested_years:
        if not demand_years.intersection(requested_years):
            sorted_requested = ", ".join(str(year) for year in sorted(requested_years))
            sorted_available = ", ".join(str(year) for year in sorted(demand_years))
            return {
                "error": (
                    "No demand data is available for the requested simulation years. "
                    f"Demand data covers years [{sorted_available}], but the run requested "
                    f"[{sorted_requested}]. Update the configuration or provide start_year/"
                    "end_year values that match the demand data."
                )
            }

    try:
        frames_obj = _ensure_years_in_demand(frames_obj, years)
    except Exception as exc:
        LOGGER.exception("Unable to normalise demand frame for requested years")
        return {"error": str(exc)}

    transmission_table: pd.DataFrame | None
    try:
        transmission_table = frames_obj.transmission()
    except Exception:
        transmission_table = None

    needs_transmission = False
    if dispatch_use_network:
        if not isinstance(transmission_table, pd.DataFrame) or transmission_table.empty:
            needs_transmission = True
        elif "limit_mw" in transmission_table.columns:
            try:
                active_interfaces = (transmission_table["limit_mw"].astype(float) > 0.0).sum()
            except Exception:  # pragma: no cover - defensive guard
                active_interfaces = 0
            if int(active_interfaces) == 0:
                needs_transmission = True

    if needs_transmission:
        network_regions = _available_regions_from_frames(frames_obj)
        if len(network_regions) > 1:
            fallback_transmission = _default_transmission(network_regions)
            if not fallback_transmission.empty:
                frames_obj = frames_obj.with_frame("transmission", fallback_transmission)
                needs_transmission = False
        if needs_transmission:
            LOGGER.warning(
                "Dispatch network requested but no transmission interfaces were available; "
                "falling back to zonal dispatch."
            )
            dispatch_use_network = False
    
    # Add expansion candidates if capacity expansion is enabled
    if capacity_flag:
        try:
            from engine.expansion_candidates import create_expansion_candidates
            from pathlib import Path
            
            # Get units DataFrame for fallback regions and fuel prices
            units_df_temp = frames_obj.units() if frames_obj.has_frame("units") else pd.DataFrame()
            
            # Get regions from demand or units
            expansion_regions = []
            demand_df_temp = frames_obj.demand() if frames_obj.has_frame("demand") else pd.DataFrame()
            if not demand_df_temp.empty and "region" in demand_df_temp.columns:
                expansion_regions = sorted(demand_df_temp["region"].dropna().astype(str).unique().tolist())
            if not expansion_regions and not units_df_temp.empty:
                expansion_regions = units_df_temp["region"].dropna().astype(str).unique().tolist()
            
            if expansion_regions:
                # Look for ATBe.csv in attached_assets
                atb_path = Path("attached_assets/ATBe_1760726232478.csv")
                
                # Get fuel prices from existing units if available
                fuel_prices = {}
                if not units_df_temp.empty and "fuel_price_per_mmbtu" in units_df_temp.columns:
                    fuel_price_map = units_df_temp.groupby("fuel")["fuel_price_per_mmbtu"].mean()
                    fuel_prices = fuel_price_map.to_dict()
                
                expansion_df = create_expansion_candidates(
                    regions=expansion_regions,
                    atb_path=atb_path if atb_path.exists() else None,
                    scenario="Moderate",
                    fuel_prices=fuel_prices if fuel_prices else None,
                    max_builds_per_tech=5,  # Allow up to 5 builds per technology per region
                )
                
                if not expansion_df.empty:
                    frames_obj = frames_obj.with_frame("expansion", expansion_df)
                    LOGGER.info(
                        f"Added {len(expansion_df)} capacity expansion candidates "
                        f"across {len(expansion_regions)} regions with NREL costs"
                    )
        except Exception as exc:
            LOGGER.warning(
                "Failed to generate capacity expansion candidates with NREL costs: %s",
                exc,
                exc_info=True,
            )

    region_label_map: dict[str, Any] = {}

    def _register_region_alias(alias: Any, value: Any) -> None:
        if alias is None:
            return
        text = str(alias).strip()
        if not text:
            return
        region_label_map.setdefault(text, value)
        region_label_map.setdefault(text.lower(), value)

    for alias, canonical_value in cap_region_aliases.items():
        _register_region_alias(alias, canonical_value)

    for region in normalized_regions:
        _register_region_alias(region, region)
        _register_region_alias(canonical_region_label(region), region)
        if not isinstance(region, str):
            _register_region_alias(str(region), region)

    def _ingest_region_values(values: Sequence[Any] | pd.Series | None) -> None:
        if values is None:
            return
        if isinstance(values, pd.Series):
            iterable = values.dropna().unique()
        else:
            iterable = values
        for value in iterable:
            if value is None:
                continue
            if pd.isna(value):
                continue
            resolved_value = canonical_region_value(value)
            if isinstance(resolved_value, str):
                resolved_text = resolved_value.strip() or str(value)
                _register_region_alias(value, resolved_text)
                _register_region_alias(resolved_text, resolved_text)
            else:
                resolved_int = int(resolved_value)
                _register_region_alias(value, resolved_int)
                _register_region_alias(canonical_region_label(resolved_int), resolved_int)
                _register_region_alias(str(resolved_int), resolved_int)

    demand_region_labels: set[str] = set()
    try:
        demand_df = frames_obj.demand()
    except Exception:
        demand_df = None
    if demand_df is not None and not demand_df.empty:
        _ingest_region_values(demand_df["region"])
        demand_region_labels = {str(region) for region in demand_df["region"].unique()}

    default_region_id = _get_default_region_id()
    existing_coverage_df: pd.DataFrame | None = None
    for frame_name in ("units", "coverage"):
        try:
            frame_candidate = frames_obj.optional_frame(frame_name)
        except Exception:
            frame_candidate = None
        if frame_candidate is not None and not frame_candidate.empty and "region" in frame_candidate.columns:
            _ingest_region_values(frame_candidate["region"])
            if frame_name == "coverage":
                existing_coverage_df = frame_candidate.copy()

    coverage_selection = list(normalized_coverage or [])
    cover_all = coverage_selection == ["All"]
    coverage_labels = (
        {str(label) for label in coverage_selection if str(label) and str(label) != "All"}
        if not cover_all
        else set()
    )
    for label in coverage_labels:
        resolved_value = canonical_region_value(label)
        if isinstance(resolved_value, str):
            resolved_text = resolved_value.strip() or label
            _register_region_alias(label, resolved_text)
            _register_region_alias(resolved_text, resolved_text)
        else:
            resolved_int = int(resolved_value)
            _register_region_alias(label, resolved_int)
            _register_region_alias(canonical_region_label(resolved_int), resolved_int)
            _register_region_alias(str(resolved_int), resolved_int)

    coverage_region_values: set[str] = set()
    for label in coverage_labels:
        token = _resolve_canonical_region(label)
        if token and token != "All":
            coverage_region_values.add(token)

    if not demand_region_labels:
        demand_region_labels = set(region_label_map) or set(coverage_labels)

    demand_region_values: set[str] = set()
    for label in demand_region_labels:
        token = _resolve_canonical_region(label)
        if token and token != "All":
            demand_region_values.add(token)

    normalized_existing: pd.DataFrame | None = None
    existing_keys: set[tuple[str, int]] = set()
    if existing_coverage_df is not None and not existing_coverage_df.empty:
        normalized_existing = existing_coverage_df.copy()
        if not isinstance(normalized_existing.index, pd.RangeIndex):
            normalized_existing = normalized_existing.reset_index(drop=True)
        index_names = getattr(normalized_existing.index, "names", None) or []
        if "region" not in normalized_existing.columns and "region" in index_names:
            normalized_existing = normalized_existing.reset_index()
        if "region" not in normalized_existing.columns:
            normalized_existing = normalized_existing.assign(region=pd.Series(dtype=object))
        if "year" not in normalized_existing.columns:
            normalized_existing = normalized_existing.assign(year=-1)
        if "covered" not in normalized_existing.columns:
            normalized_existing = normalized_existing.assign(covered=False)
        normalized_existing = normalized_existing.loc[:, ["region", "year", "covered"]]
        normalized_existing["year"] = pd.to_numeric(
            normalized_existing["year"], errors="coerce"
        ).fillna(-1).astype(int)
        normalized_existing["covered"] = normalized_existing["covered"].astype(bool)
        normalized_existing["region"] = [
            _resolve_canonical_region(value) or default_region_id
            for value in normalized_existing["region"]
        ]
        existing_keys = {
            (str(region), int(year))
            for region, year in zip(
                normalized_existing["region"], normalized_existing["year"]
            )
        }

    available_region_values: set[str] = {
        str(_resolve_canonical_region(value) or default_region_id)
        for value in region_label_map.values()
    }
    available_region_values |= coverage_region_values
    available_region_values |= demand_region_values
    if not available_region_values:
        available_region_values = {region for region, _ in existing_keys}

    coverage_records: list[dict[str, Any]] = []
    seen_new_keys: set[tuple[str, int]] = set()
    sorted_regions = sorted(
        available_region_values,
        key=lambda value: canonical_region_label(value).lower(),
    )
    for region_value in sorted_regions:
        canonical_region = str(_resolve_canonical_region(region_value) or region_value)
        key = (canonical_region, -1)
        if key in existing_keys or key in seen_new_keys:
            continue
        coverage_records.append(
            {
                "region": canonical_region,
                "year": -1,
                "covered": True if cover_all else canonical_region in coverage_region_values,
            }
        )
        seen_new_keys.add(key)

    if coverage_records:
        coverage_df = pd.DataFrame(coverage_records, columns=["region", "year", "covered"])
    else:
        coverage_df = pd.DataFrame(columns=["region", "year", "covered"])
    if normalized_existing is not None:
        coverage_df = pd.concat([normalized_existing, coverage_df], ignore_index=True)
    coverage_df = coverage_df.sort_values(["region", "year"]).reset_index(drop=True)
    frames_obj = frames_obj.with_frame("coverage", coverage_df)

    if not dispatch_use_network:
        check_year_value: int | None = None
        if years:
            try:
                check_year_value = int(next(iter(years)))
            except Exception:
                check_year_value = None
        if check_year_value is None:
            check_year_value = int(date.today().year)

        try:
            coverage_map = frames_obj.coverage_for_year(int(check_year_value))
        except Exception:
            coverage_map = {}
        relevant_regions = {
            str(region)
            for region in _available_regions_from_frames(frames_obj)
            if str(region)
        }
        coverage_flags = {
            region: bool(coverage_map.get(region, True))
            for region in relevant_regions
        }
        if len({flag for flag in coverage_flags.values()}) > 1:
            return {"error": _SINGLE_REGION_COVERAGE_MESSAGE}

    if normalized_regions:
        config_regions = list(dict.fromkeys(list(config.get("regions", [])) + normalized_regions))
        config["regions"] = config_regions

        cap_group_cfg = config.get('carbon_cap_groups')
        if isinstance(cap_group_cfg, list):
            if cap_group_cfg:
                first_entry = dict(cap_group_cfg[0])
                first_entry.setdefault('name', first_entry.get('name', 'default'))
                first_entry['regions'] = list(normalized_regions)
                cap_group_cfg[0] = first_entry
            else:
                cap_group_cfg.append({'name': 'default', 'regions': list(normalized_regions), 'cap': 'none'})
        elif isinstance(cap_group_cfg, Mapping):
            updated_groups = {}
            applied = False
            for key, value in cap_group_cfg.items():
                entry = dict(value) if isinstance(value, Mapping) else {}
                if not applied:
                    entry['regions'] = list(normalized_regions)
                    applied = True
                updated_groups[str(key)] = entry
            if not applied:
                updated_groups['default'] = {'regions': list(normalized_regions), 'cap': 'none'}
            config['carbon_cap_groups'] = updated_groups
        else:
            config['carbon_cap_groups'] = [{'name': 'default', 'regions': list(normalized_regions), 'cap': 'none'}]

    policy_frame = _build_policy_frame(
        config,
        years,
        bool(policy_enabled),
        ccr1_enabled=bool(ccr1_enabled),
        ccr2_enabled=bool(ccr2_enabled),
        control_period_years=control_period_years,
        banking_enabled=bool(allowance_banking_enabled),
        floor_escalator_mode=carbon_policy_cfg.floor_escalator_mode,
        floor_escalator_value=carbon_policy_cfg.floor_escalator_value,
        ccr1_escalator_pct=carbon_policy_cfg.ccr1_escalator_pct,
        ccr2_escalator_pct=carbon_policy_cfg.ccr2_escalator_pct,
    )
    frames_obj = frames_obj.with_frame('policy', policy_frame)

    policy_bank0 = 0.0
    if isinstance(policy_frame, pd.DataFrame) and not policy_frame.empty and "bank0" in policy_frame.columns:
        try:
            bank_series = pd.to_numeric(policy_frame["bank0"], errors="coerce")
        except Exception:  # pragma: no cover - defensive
            bank_series = None
        if bank_series is not None and not bank_series.empty:
            first_bank = bank_series.iloc[0]
            if pd.notna(first_bank):
                policy_bank0 = float(first_bank)

    # Apply config override if needed
    if bank0_from_config is not None:
        try:
            bank0_val = float(bank0_from_config)
        except Exception:  # pragma: no cover - defensive
            bank0_val = 0.0

        if banking_flag and bank0_val > 0.0 and policy_bank0 <= 0.0:
            policy_bank0 = bank0_val
        elif not policy_bank0:
            policy_bank0 = bank0_val
    def _load_frame_from_frames(frames_candidate: Any) -> pd.DataFrame | None:
        """Return the load forecast frame from ``frames_candidate`` when available."""

        optional_frame = getattr(frames_candidate, "optional_frame", None)
        if callable(optional_frame):
            try:
                return optional_frame("load")
            except Exception:
                return None

        if isinstance(frames_candidate, Mapping):
            candidate = frames_candidate.get("load")
            if candidate is None:
                candidate = frames_candidate.get("load_forecasts")
            if isinstance(candidate, pd.DataFrame):
                return candidate.copy(deep=True)
        return None


    def _manifest_sources_for_logging() -> str:
        """Return a human-readable summary of manifest sources."""

        descriptors: list[str] = []
        for record in forecast_manifest_records:
            if not isinstance(record, Mapping):
                continue
            iso = str(record.get("iso") or "").strip()
            scenario = str(record.get("scenario") or "").strip()
            descriptor = f"{iso}::{scenario}" if iso and scenario else iso or scenario
            path = str(record.get("path") or "").strip()
            if path:
                descriptor = f"{descriptor} [{path}]" if descriptor else path
            if descriptor:
                descriptors.append(descriptor)

        if not descriptors:
            descriptors.extend(
                f"{entry.iso}::{entry.scenario}"
                for entry in scenario_selections
                if getattr(entry, "iso", None) and getattr(entry, "scenario", None)
            )

        if not descriptors and isinstance(selection_mapping, Mapping):
            for key, value in selection_mapping.items():
                descriptors.append(f"{key}={value}")

        if not descriptors:
            return ""

        return "; ".join(dict.fromkeys(descriptors))


    load_frame = _load_frame_from_frames(frames_obj)
    load_frame_error: str | None = None

    if not isinstance(load_frame, pd.DataFrame) or load_frame.empty:
        load_frame_error = "Load forecast table is empty; demand forecasts were not loaded."
    else:
        required_columns = {"zone", "year"}
        missing_columns = sorted(required_columns - set(load_frame.columns))
        if missing_columns:
            load_frame_error = (
                "Load forecast table is missing required columns: "
                + ", ".join(missing_columns)
            )
        else:
            load_years = {
                int(value)
                for value in pd.to_numeric(load_frame["year"], errors="coerce").dropna().astype(int)
            }
            requested_years_set = {int(year) for year in years} if years else set()
            missing_years = sorted(requested_years_set - load_years)
            if requested_years_set and missing_years:
                load_frame_error = (
                    "Load forecast table is missing data for years: "
                    + ", ".join(str(year) for year in missing_years)
                )

            if load_frame_error is None and "zone" in load_frame.columns:
                zone_entries = load_frame["zone"].astype("string").str.strip().dropna()
                if zone_entries.empty:
                    load_frame_error = "Load forecast table does not contain any zone identifiers."

    if load_frame_error:
        context = _manifest_sources_for_logging()
        if context:
            load_frame_error = f"{load_frame_error} Sources attempted: {context}"
        LOGGER.error(load_frame_error)
        return {"error": load_frame_error}


    runner = _ensure_engine_runner()
    supports_deep = True
    legacy_signature = False
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):  # pragma: no cover - builtin or C-accelerated callables
        supports_deep = True
    else:
        params = signature.parameters
        if "deep_carbon_pricing" in params:
            supports_deep = True
        else:
            has_var_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in params.values()
            )
            if has_var_kwargs:
                supports_deep = True
            else:
                supports_deep = False
                modern_keywords = {"tol", "max_iter", "relaxation", "price_cap"}
                legacy_signature = modern_keywords.issubset(params.keys())

    if legacy_signature and deep_carbon_pricing:
        if not _runner_supports_keyword(runner, "deep_carbon_pricing"):
            return {
                "error": (
                    "Deep carbon pricing requires an updated engine. "
                    "Please upgrade engine.run_loop.run_end_to_end_from_frames."
                )
            }
        supports_deep = True

    if not supports_deep:
        supports_deep = _runner_supports_keyword(runner, "deep_carbon_pricing")

    if not supports_deep and deep_carbon_pricing:
        return {"error": DEEP_CARBON_UNSUPPORTED_MESSAGE}

    enable_floor_flag = bool(policy_enabled and carbon_policy_cfg.enable_floor)
    enable_ccr_flag = bool(
        policy_enabled
        and carbon_policy_cfg.enable_ccr
        and (carbon_policy_cfg.ccr1_enabled or carbon_policy_cfg.ccr2_enabled)
    )
    if price_active:
        price_value_raw = (
            carbon_price_value
            if carbon_price_value is not None
            else price_cfg.price_per_ton
        )
        try:
            runner_price_value = float(price_value_raw)
        except (TypeError, ValueError):
            runner_price_value = float(price_cfg.price_per_ton)
    else:
        runner_price_value = 0.0

    if _compute_intensity is not None:
        cap_max_iter = max(
            int(_compute_intensity.get_effective_iteration_limit(_DEFAULT_MAX_ITERATIONS)),
            1,
        )
    else:
        cap_max_iter = max(int(_DEFAULT_MAX_ITERATIONS), 1)
    cap_tol = float(_CAP_PRICE_TOL) if _CAP_PRICE_TOL else 1e-3

    runner_kwargs: dict[str, Any] = {
        "years": years,
        "price_initial": 0.0,
        "enable_floor": enable_floor_flag,
        "enable_ccr": enable_ccr_flag,
        "use_network": bool(dispatch_use_network),
        "capacity_expansion": bool(capacity_flag),
        "carbon_price_schedule": price_schedule_map if price_active else None,
        "carbon_price_value": runner_price_value,
        "deep_carbon_pricing": bool(deep_carbon_pricing),
        "progress_cb": progress_cb,
    }

    if isinstance(config, Mapping):
        states_cfg = config.get("states")
        if states_cfg:
            runner_kwargs["states"] = states_cfg

    if stage_cb is not None:
        runner_kwargs["stage_cb"] = stage_cb

    if _runner_supports_keyword(runner, "report_by_technology"):
        runner_kwargs["report_by_technology"] = True

    if not _runner_supports_keyword(runner, "capacity_expansion"):
        runner_kwargs.pop("capacity_expansion", None)

    if not _runner_supports_keyword(runner, "carbon_price_value"):
        runner_kwargs.pop("carbon_price_value", None)

    if not _runner_supports_keyword(runner, "deep_carbon_pricing"):
        if deep_carbon_pricing:
            return {
                "error": (
                    "Deep carbon pricing requires an updated engine. "
                    "Please upgrade engine.run_loop.run_end_to_end_from_frames."
                )
            }
        runner_kwargs.pop("deep_carbon_pricing", None)

    if _runner_supports_keyword(runner, "max_iter"):
        runner_kwargs["max_iter"] = cap_max_iter
    if _runner_supports_keyword(runner, "tol"):
        runner_kwargs["tol"] = cap_tol
    if not _runner_supports_keyword(runner, "stage_cb"):
        runner_kwargs.pop("stage_cb", None)

    try:
        outputs = runner(frames_obj, **runner_kwargs)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception("Policy simulation failed")
        empty_outputs = None
        if _EngineOutputs is not None:
            try:
                empty_outputs = _EngineOutputs.empty()
            except Exception:  # pragma: no cover - defensive guard
                empty_outputs = None
        payload = {"error": str(exc)}
        if empty_outputs is not None:
            payload["outputs"] = empty_outputs
        return payload

    limiting_factors: list[str] = []
    raw_factors: Iterable[str] | None = None
    if isinstance(outputs, Mapping):
        raw_factors = outputs.get("limiting_factors", [])
    else:
        raw_factors = getattr(outputs, "limiting_factors", None)
    if isinstance(raw_factors, Iterable) and not isinstance(raw_factors, (str, bytes)):
        limiting_factors = [str(entry) for entry in raw_factors]

    capacity_df = _extract_output_dataframe(outputs, ["capacity_by_technology"])
    if capacity_df is None:
        LOGGER.warning(
            "Runner outputs missing capacity_by_technology frame; capacity charts will not be displayed."
        )
        capacity_df = pd.DataFrame(
            columns=["year", "technology", "capacity_mw", "capacity_mwh"]
        )

    generation_df = _extract_output_dataframe(outputs, ["generation_by_technology"])
    if generation_df is None:
        LOGGER.warning(
            "Runner outputs missing generation_by_technology frame; generation charts will not be displayed."
        )
        generation_df = pd.DataFrame(
            columns=["year", "technology", "generation_mwh"]
        )


    temp_dir, csv_files = _write_outputs_to_temp(outputs)

    result_payload: dict[str, Any] = {}
    if isinstance(outputs, Mapping):
        result_payload.update(outputs)
    else:
        for key in (
            "demand_by_region",
            "generation_by_region",
            "capacity_by_region",
            "cost_by_region",
        ):
            value = getattr(outputs, key, None)
            if value is not None:
                result_payload[key] = value
    if isinstance(csv_files, Mapping):
        result_payload.setdefault("csv_files", csv_files)

    documentation = {
        "assumption_overrides": list(assumption_notes or []),
    }


    annual_df = _extract_output_dataframe(
        outputs, ["annual", "annual_results", "annual_output", "annual_outputs"]
    )

    emissions_field_present = False
    for key in ['emissions_by_region', 'emissions', 'emissions_region']:
        if hasattr(outputs, key) or (
            isinstance(outputs, Mapping) and key in outputs
        ):
            emissions_field_present = True
            break

    emissions_df = _extract_output_dataframe(
        outputs, ["emissions_by_region", "emissions", "emissions_region"]
    )
    raw_price_df = _extract_output_dataframe(
        outputs, ["price_by_region", "dispatch_price_by_region", "region_prices"]
    )
    price_df, price_flags = _normalize_dispatch_price_frame(raw_price_df)
    flows_df = _extract_output_dataframe(
        outputs, ["flows", "network_flows", "flows_by_region"]
    )

    # Direct extraction from outputs
    demand_region_output_df = _extract_output_dataframe(
        outputs, ["demand_by_region", "regional_demand"]
    )
    generation_region_output_df = _extract_output_dataframe(
        outputs, ["generation_by_region"]
    )
    capacity_region_output_df = _extract_output_dataframe(
        outputs, ["capacity_by_region"]
    )
    cost_region_output_df = _extract_output_dataframe(
        outputs, ["cost_by_region"]
    )

    # Backward-compatible extraction from result with aliases
    demand_region_df = _extract_result_frame(
        result_payload,
        "demand_by_region",
        aliases=("demand_by_region_output",),
    )
    generation_region_df = _extract_result_frame(
        result_payload,
        "generation_by_region",
        aliases=("generation_by_region_output",),
    )
    capacity_region_df = _extract_result_frame(
        result_payload,
        "capacity_by_region",
        aliases=("capacity_by_region_output",),
    )
    cost_region_df = _extract_result_frame(
        result_payload,
        "cost_by_region",
        aliases=("cost_by_region_output",),
    )




    if isinstance(annual_df, pd.DataFrame):
        annual_df = dataframe_to_carbon_vector(annual_df).copy()
        if not banking_flag:
            annual_df['bank'] = 0.0
        elif not annual_df.empty and {'allowances_available', 'emissions_tons'}.issubset(annual_df.columns):
            try:
                year_order = pd.to_numeric(annual_df['year'], errors='coerce')
            except Exception:  # pragma: no cover - defensive
                year_order = None
            if year_order is not None:
                ordered_index = year_order.sort_values(kind='mergesort').index
            else:
                ordered_index = annual_df.index
            bank_running = policy_bank0 if banking_flag else 0.0
            bank_values: dict[Any, float] = {}
            for idx in ordered_index:
                try:
                    allowances_total = float(annual_df.at[idx, 'allowances_available'])
                except Exception:
                    allowances_total = 0.0
                try:
                    emissions_value = float(annual_df.at[idx, 'emissions_tons'])
                except Exception:
                    emissions_value = 0.0
                bank_running = max(bank_running + allowances_total - emissions_value, 0.0)
                bank_values[idx] = bank_running
            annual_df['bank'] = annual_df.index.map(lambda idx: bank_values.get(idx, 0.0))

        def _configure_ccr_trigger(column: str, config_key: str) -> None:
            config_value = allowance_config.get(config_key)
            try:
                trigger_value = float(config_value)
            except (TypeError, ValueError):
                trigger_value = None
            if trigger_value is None:
                return
            if column in annual_df.columns:
                column_series = annual_df[column]
                if column_series.isna().all() or (column_series == 0.0).all():
                    annual_df[column] = trigger_value
            else:
                annual_df[column] = trigger_value

        _configure_ccr_trigger('ccr1_trigger', 'ccr1_trigger')
        _configure_ccr_trigger('ccr2_trigger', 'ccr2_trigger')

    if isinstance(csv_files, Mapping) and isinstance(annual_df, pd.DataFrame):
        try:
            csv_files = dict(csv_files)
            export_frame = _with_legacy_carbon_price_columns(annual_df)
            csv_files['annual.csv'] = export_frame.to_csv(index=False).encode('utf-8')
        except Exception:  # pragma: no cover - defensive
            pass

    result: dict[str, Any] = {
        'annual': annual_df,
        'emissions_by_region': emissions_df,
        'price_by_region': price_df,
        'flows': flows_df,
        'demand_by_region': demand_region_df,
        'generation_by_region': generation_region_df,
        'capacity_by_region': capacity_region_df,
        'cost_by_region': cost_region_df,
        'module_config': merged_modules,
        'config': config,
        'csv_files': csv_files,
        'temp_dir': temp_dir,
        'documentation': documentation,
    }
    if not scenario_selections and selection_mapping:
        frame = _cached_forecast_frame(forecast_base_path)
        scenario_selections = _manifests_from_selection(
            selection_mapping,
            frame=frame,
            base_path=forecast_base_path,
        )

    if not forecast_manifest_records and scenario_selections:
        forecast_manifest_records = _forecast_bundles_from_selection(
            scenario_selections,
            base_path=forecast_base_path,
        )

    manifest_data = build_manifest(
        config,
        frames_obj,
        outputs,
        forecast_manifests=forecast_manifest_records,
        git_commit=_GIT_COMMIT,
    )
    manifest_markdown = manifest_to_markdown(manifest_data)
    deep_doc = build_deep_doc(manifest_data, frames_obj, outputs)
    deep_doc_markdown = deep_doc_to_markdown(deep_doc)

    run_identifier = str(run_id) if run_id else uuid4().hex
    output_dir = PROJECT_ROOT / "output" / run_identifier
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_json_path = output_dir / "run_manifest.json"
        manifest_md_path = output_dir / "run_manifest.md"
        deep_doc_path = output_dir / "model_documentation.md"
        manifest_json_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        manifest_md_path.write_text(manifest_markdown, encoding="utf-8")
        deep_doc_path.write_text(deep_doc_markdown, encoding="utf-8")
    except Exception:  # pragma: no cover - documentation export is best-effort
        manifest_json_path = output_dir / "run_manifest.json"
        manifest_md_path = output_dir / "run_manifest.md"
        deep_doc_path = output_dir / "model_documentation.md"

    doc_section = result.setdefault("documentation", {})
    if scenario_selections:
        selection_labels: dict[str, str] = {}
        for manifest in scenario_selections:
            iso_value = getattr(manifest, "iso", None)
            scenario_value = getattr(manifest, "scenario", None)
            if iso_value is None and isinstance(manifest, Mapping):
                iso_value = manifest.get("iso")
                scenario_value = scenario_value or manifest.get("scenario")
            if not iso_value:
                continue
            iso_text = str(iso_value)
            if scenario_value:
                selection_labels[iso_text] = f"{iso_text} – {scenario_value}"
            else:
                selection_labels[iso_text] = iso_text
        if selection_labels:
            doc_section["forecast_selection"] = selection_labels
    else:
        forecasts = forecast_manifest_records or manifest_data.get("load_forecasts", [])
        if isinstance(forecasts, Sequence):
            doc_section["forecast_selection"] = {
                str(entry.get("iso")): entry.get("manifest")
                for entry in forecasts
                if isinstance(entry, Mapping) and entry.get("iso")
            }

    doc_section['manifest'] = manifest_data
    doc_section['manifest_markdown'] = manifest_markdown
    doc_section['manifest_paths'] = {
        'json': str(manifest_json_path),
        'md': str(manifest_md_path),
    }
    doc_section['deep_doc'] = deep_doc
    doc_section['deep_doc_markdown'] = deep_doc_markdown
    doc_section['deep_doc_path'] = str(deep_doc_path)
    doc_section['run_id'] = run_identifier

    run_years_info = manifest_data.get('run', {}).get('years', {})
    if isinstance(run_years_info, Mapping):
        year_min = run_years_info.get('min')
        year_max = run_years_info.get('max')
        year_span = None
        if year_min is not None and year_max is not None:
            year_span = f"{year_min}–{year_max}"
        elif run_years_info.get('all'):
            year_span = run_years_info.get('all')
    else:
        year_span = None

    price_floor_info = manifest_data.get('policy', {}).get('price_floor')
    if isinstance(price_floor_info, Mapping):
        floor_display = ", ".join(
            f"{year}: {value}" for year, value in sorted(price_floor_info.items())
        )
    else:
        floor_display = price_floor_info

    def _format_manifest_mapping(value: Any) -> Any:
        """Render manifest mapping values as display-friendly strings."""

        if isinstance(value, Mapping):
            items = list(value.items())
        elif hasattr(value, "items"):
            try:
                items = list(value.items())  # type: ignore[arg-type]
            except Exception:  # pragma: no cover - defensive guard
                return value
        else:
            return value

        try:
            items.sort(key=lambda item: (item[0] is None, str(item[0])))
        except Exception:  # pragma: no cover - defensive guard
            pass

        return ", ".join(f"{item_key}: {item_value}" for item_key, item_value in items)

    assumption_summary = {
        'years': year_span,
        'banking_enabled': manifest_data.get('run', {}).get('banking_enabled'),
        'price_floor': floor_display,
        'ccr1_trigger': _format_manifest_mapping(
            manifest_data.get('policy', {}).get('ccr1_trigger')
        ),
        'ccr2_trigger': _format_manifest_mapping(
            manifest_data.get('policy', {}).get('ccr2_trigger')
        ),
        'region_count': len(manifest_data.get('electricity', {}).get('regions', [])),
        'interfaces': manifest_data.get('electricity', {}).get('transmission', {}).get('interfaces'),
    }
    doc_section['assumption_summary'] = assumption_summary

    doc_section = _sanitize_documentation_mapping(doc_section)
    result['documentation'] = doc_section

    emissions_total_map = getattr(outputs, "emissions_total", None)
    if isinstance(emissions_total_map, Mapping):
        try:
            result["emissions_total"] = {
                int(year): float(value) for year, value in emissions_total_map.items()
            }
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.exception("Unable to normalise aggregate emissions totals")

    emissions_region_map = getattr(outputs, "emissions_by_region_map", None)
    normalized_emissions_map: dict[str, dict[int, float]] = {}
    if isinstance(emissions_region_map, Mapping):
        for region, data in emissions_region_map.items():
            try:
                normalized_emissions_map[str(region)] = {
                    int(year): float(value) for year, value in data.items()
                }
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.exception(
                    "Unable to normalise emissions mapping for region %s", region
                )
    if normalized_emissions_map:
        result["emissions_by_region_map"] = normalized_emissions_map


    if hasattr(outputs, "emissions_summary_table"):
        try:
            summary_table = outputs.emissions_summary_table()
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.exception("Unable to compute regional emissions summary table")
        else:
            if isinstance(summary_table, pd.DataFrame):
                result["emissions_summary"] = summary_table

    if limiting_factors:
        result["limiting_factors"] = limiting_factors

    result["_price_output_type"] = "allowance" if policy_enabled else "carbon"
    result["_price_field_flags"] = price_flags

    if normalized_cap_regions:
        result["cap_regions"] = normalized_cap_regions
    elif cap_regions_all_selected:
        result["cap_regions"] = []
    elif "emissions_by_region_map" in result:
        result["cap_regions"] = list(result["emissions_by_region_map"])

    demand_output = _build_demand_output_frame(
        years,
        config.get("regions"),
        merged_modules.get("demand"),
        fallback_regions=list(region_label_map.values()) if region_label_map else None,
    )
    if isinstance(demand_output, pd.DataFrame):
        result["demand_by_region"] = demand_output
        csv_mapping = result.get("csv_files")
        if isinstance(csv_mapping, Mapping):
            try:
                updated_csv = dict(csv_mapping)
                updated_csv["demand_by_region.csv"] = demand_output.to_csv(
                    index=False
                ).encode("utf-8")
            except Exception:  # pragma: no cover - defensive guard
                pass
            else:
                result["csv_files"] = updated_csv

    result["capacity_by_technology"] = capacity_df
    result["generation_by_technology"] = generation_df


    optional_frames = {
        "demand_by_region_output": demand_region_output_df,
        "generation_by_region_output": generation_region_output_df,
        "capacity_by_region_output": capacity_region_output_df,
        "cost_by_region_output": cost_region_output_df,
        "capacity_by_technology": ["capacity_by_technology"],
        "generation_by_technology": ["generation_by_technology"],
        "demand_by_region": ["demand_by_region", "demand_by_region_output"],
        "generation_by_region": ["generation_by_region", "generation_by_region_output"],
        "capacity_by_region": ["capacity_by_region", "capacity_by_region_output"],
        "cost_by_region": ["cost_by_region", "cost_by_region_output"],
    }
    for key, frame in optional_frames.items():
        if isinstance(frame, pd.DataFrame):
            result[key] = frame

    demand_region_df = (
        demand_region_output_df
        if isinstance(demand_region_output_df, pd.DataFrame)
        else _extract_result_frame(result, "demand_by_region_output")
    )
    if demand_region_df is None:
        demand_region_df = _extract_result_frame(result, "demand_by_region")

    generation_region_df = (
        generation_region_output_df
        if isinstance(generation_region_output_df, pd.DataFrame)
        else _extract_result_frame(result, "generation_by_region_output")
    )

    capacity_region_df = (
        capacity_region_output_df
        if isinstance(capacity_region_output_df, pd.DataFrame)
        else _extract_result_frame(result, "capacity_by_region_output")
    )

    cost_region_df = (
        cost_region_output_df
        if isinstance(cost_region_output_df, pd.DataFrame)
        else _extract_result_frame(result, "cost_by_region_output")
    )

    legacy_frame_aliases = {
        "demand_by_region_output": "demand_by_region",
        "generation_by_region_output": "generation_by_region",
        "capacity_by_region_output": "capacity_by_region",
        "cost_by_region_output": "cost_by_region",
    }
    for legacy_key, canonical_key in legacy_frame_aliases.items():
        if canonical_key in result and legacy_key not in result:
            result[legacy_key] = result[canonical_key]


    processed_emissions = load_emissions_data(result)
    if isinstance(processed_emissions, pd.DataFrame):
        result['_emissions_source'] = processed_emissions.attrs.get(
            'emissions_source', 'engine'
        )
        if processed_emissions.empty:
            source_marker = processed_emissions.attrs.get('emissions_source')
            if source_marker == 'missing' or (source_marker is None and not emissions_field_present):
                result.setdefault('_emissions_messages', []).append(
                    'Emissions data missing from engine outputs.'
                )
        else:
            result['emissions_by_region'] = processed_emissions
        csv_files = result.get('csv_files')
        if isinstance(csv_files, Mapping):
            try:
                updated_csv = dict(csv_files)
                updated_csv['emissions_by_region.csv'] = processed_emissions.to_csv(
                    index=False
                ).encode('utf-8')
            except Exception:  # pragma: no cover - defensive guard
                pass
            else:
                result['csv_files'] = updated_csv

    record_recent_result(result)

    return result

    # Carbon price config

      
def _extract_result_frame(
    result: Mapping[str, Any],
    key: str,
    *,
    csv_name: str | None = None,
    aliases: Sequence[str] | None = None,
) -> pd.DataFrame | None:
    """Return a DataFrame from `result` or load it from cached CSV bytes."""

    candidates: list[str] = []
    for candidate in [key, *(aliases or ())]:
        if not candidate:
            continue
        if candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        frame = result.get(candidate)
        if isinstance(frame, pd.DataFrame):
            return frame

    csv_files = result.get('csv_files')
    if isinstance(csv_files, Mapping):
        csv_candidates: list[str] = []
        if csv_name:
            csv_candidates.append(csv_name)
        for candidate in candidates:
            csv_candidates.append(f'{candidate}.csv')
        seen_files: set[str] = set()
        for filename in csv_candidates:
            if not filename:
                continue
            if filename in seen_files:
                continue
            seen_files.add(filename)
            raw = csv_files.get(filename)
            if isinstance(raw, (bytes, bytearray)):
                try:
                    return pd.read_csv(io.BytesIO(raw))
                except Exception:  # pragma: no cover - defensive guard
                    continue
    return None


def _render_capacity_factor_section(
    generation_df: pd.DataFrame | None,
    capacity_df: pd.DataFrame | None,
) -> None:
    """Render capacity factor chart showing actual utilization vs theoretical maximum."""
    _ensure_streamlit()
    st.subheader('Capacity factors by technology')
    
    if generation_df is None or generation_df.empty or capacity_df is None or capacity_df.empty:
        st.caption('Capacity factor analysis requires both generation and capacity data.')
        return
    
    # NREL maximum capacity factors by technology type
    NREL_MAX_CF = {
        'solar': 0.25,
        'wind': 0.35,
        'wind_offshore': 0.45,
        'natural_gas_combined_cycle': 0.87,
        'natural_gas_combustion_turbine': 0.93,
        'battery': 0.90,
        'coal': 0.85,
        'nuclear': 0.90,
        'hydro': 0.52,
        'biomass': 0.80,
        'geothermal': 0.90,
        'oil': 0.08,
        # Common fuel name variations
        'gascombinedcycle': 0.87,
        'gassteam': 0.50,
        'gasturbine': 0.93,
        'other': 0.50,
    }
    
    # Extract generation data
    gen_cols = ['year', 'technology', 'generation_mwh', 'generation', 'value']
    gen_value_col = None
    for col in gen_cols:
        if col in generation_df.columns and col != 'year' and col != 'technology':
            gen_value_col = col
            break
    
    if gen_value_col is None or 'year' not in generation_df.columns or 'technology' not in generation_df.columns:
        st.caption('Generation data missing required columns (year, technology, generation value).')
        return
    
    # Extract capacity data
    cap_cols = ['year', 'technology', 'capacity_mw', 'capacity', 'value']
    cap_value_col = None
    for col in cap_cols:
        if col in capacity_df.columns and col != 'year' and col != 'technology':
            cap_value_col = col
            break
    
    if cap_value_col is None or 'year' not in capacity_df.columns or 'technology' not in capacity_df.columns:
        st.caption('Capacity data missing required columns (year, technology, capacity value).')
        return
    
    # Prepare data
    gen_data = generation_df[['year', 'technology', gen_value_col]].copy()
    gen_data = gen_data.rename(columns={gen_value_col: 'generation_mwh'})
    gen_data['year'] = pd.to_numeric(gen_data['year'], errors='coerce')
    gen_data = gen_data.dropna(subset=['year'])
    
    cap_data = capacity_df[['year', 'technology', cap_value_col]].copy()
    cap_data = cap_data.rename(columns={cap_value_col: 'capacity_mw'})
    cap_data['year'] = pd.to_numeric(cap_data['year'], errors='coerce')
    cap_data = cap_data.dropna(subset=['year'])
    
    # Merge generation and capacity
    merged = gen_data.merge(cap_data, on=['year', 'technology'], how='inner')
    
    if merged.empty:
        st.caption('No matching year/technology pairs found between generation and capacity data.')
        return
    
    # Map technology names to NREL max capacity factors
    def get_nrel_max_cf(tech_name: str) -> float:
        tech_lower = str(tech_name).lower().strip().replace(' ', '').replace('_', '')
        return NREL_MAX_CF.get(tech_lower, 0.50)  # Default to 50% if unknown
    
    merged['nrel_max_cf'] = merged['technology'].apply(get_nrel_max_cf)
    
    # Calculate actual capacity factor: generation / (8760 * capacity * max_cf)
    merged['theoretical_max_mwh'] = 8760 * merged['capacity_mw'] * merged['nrel_max_cf']
    merged['actual_cf'] = merged['generation_mwh'] / merged['theoretical_max_mwh']
    
    # Convert to percentage
    merged['cf_percent'] = merged['actual_cf'] * 100
    
    # Handle edge cases (divide by zero, unrealistic values)
    merged['cf_percent'] = merged['cf_percent'].fillna(0)
    merged['cf_percent'] = merged['cf_percent'].clip(0, 200)  # Cap at 200% to show overutilization
    
    # Create interactive bar chart
    try:
        import altair as alt
        
        # Prepare chart data
        chart_data = merged[['year', 'technology', 'cf_percent']].copy()
        
        # Create selection for toggling
        technology_selection = alt.selection_point(
            fields=['technology'],
            bind='legend'
        )
        
        # Create grouped bar chart (not stacked)
        # Add NREL max CF reference line data
        nrel_ref_data = pd.DataFrame([
            {'cf_label': 'NREL Max (100%)', 'cf_value': 100}
        ])
        
        # Bar chart for actual capacity factors
        bars = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('technology:N', title='Technology', sort='-y'),
            y=alt.Y('cf_percent:Q', title='Capacity Factor (%)', scale=alt.Scale(domain=[0, max(250, chart_data['cf_percent'].max())])),
            color=alt.Color(
                'technology:N',
                title='Technology',
                scale=alt.Scale(scheme='category20')
            ),
            column=alt.Column('year:O', title='Year'),
            tooltip=[
                alt.Tooltip('year:O', title='Year'),
                alt.Tooltip('technology:N', title='Technology'),
                alt.Tooltip('cf_percent:Q', title='Capacity Factor (%)', format='.1f')
            ]
        ).transform_filter(
            technology_selection
        ).add_params(
            technology_selection
        ).properties(
            height=400,
            width=100,
            title='Actual Capacity Factors by Technology (vs 100% = NREL Max)'
        )
        
        st.altair_chart(bars, use_container_width=True)
        st.caption('💡 Click on technologies in the legend to show/hide them. Values >100% mean units are generating more than physically realistic!')
        
        # Warning if any technology exceeds 100%
        overutilized = chart_data[chart_data['cf_percent'] > 100]
        if not overutilized.empty:
            techs = overutilized['technology'].unique()
            st.warning(f"⚠️ OVERUTILIZATION DETECTED: {', '.join(techs)} running above physical limits! This prevents capacity expansion from working correctly.")
        
    except ImportError:
        # Fallback to simple display
        st.dataframe(merged[['year', 'technology', 'cf_percent']].sort_values(['year', 'technology']), use_container_width=True)
    
    # Show detailed data table
    display_cols = ['year', 'technology', 'generation_mwh', 'capacity_mw', 'nrel_max_cf', 'cf_percent']
    display_data = merged[display_cols].copy()
    display_data = display_data.rename(columns={
        'generation_mwh': 'Generation (MWh)',
        'capacity_mw': 'Capacity (MW)',
        'nrel_max_cf': 'NREL Max CF',
        'cf_percent': 'Actual CF (%)'
    })
    st.dataframe(display_data.sort_values(['year', 'technology']), use_container_width=True)


def _render_technology_section(
    frame: pd.DataFrame | None,
    *,
    section_title: str,
    candidate_columns: list[tuple[str, str]],
) -> None:
    """Render charts summarising technology-level output data."""
    _ensure_streamlit()
    st.subheader(section_title)

    if frame is None or frame.empty:
        st.caption(f"Engine did not return {section_title.lower()} data for this run.")
        return

    if 'technology' not in frame.columns:
        st.caption('Technology detail unavailable; displaying raw data instead.')
        st.dataframe(frame, use_container_width=True)
        return

    value_col: str | None = None
    value_label = ''
    for column, label in candidate_columns:
        if column in frame.columns:
            value_col = column
            value_label = label
            break

    if value_col is None:
        numeric_cols = frame.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            value_col = numeric_cols[0]
            value_label = numeric_cols[0]
        else:
            st.caption('No numeric values available to chart; showing raw data.')
            st.dataframe(frame, use_container_width=True)
            return

    display_frame = frame.copy()
    if 'year' in display_frame.columns:
        display_frame['year'] = pd.to_numeric(display_frame['year'], errors='coerce')
        display_frame = display_frame.dropna(subset=['year'])

    if display_frame.empty:
        st.caption('No valid year entries available; showing raw data.')
        st.dataframe(frame, use_container_width=True)
        return

    display_frame = display_frame.sort_values(['year', 'technology'])
    pivot = display_frame.pivot_table(
        index='year',
        columns='technology',
        values=value_col,
        aggfunc='sum',
    )

    if pivot.empty:
        st.caption('No data available to chart; showing raw data.')
        st.dataframe(frame, use_container_width=True)
        return

    # Create interactive stacked area chart with toggleable technologies
    try:
        import altair as alt
        
        # Prepare data for Altair
        chart_data = display_frame[['year', 'technology', value_col]].copy()
        chart_data = chart_data.rename(columns={value_col: 'value'})
        
        # Get unique technologies for color scheme
        technologies = sorted(chart_data['technology'].unique())
        
        # Create selection for toggling technologies - multi selection to allow multiple toggles
        technology_selection = alt.selection_point(
            fields=['technology'],
            bind='legend'
        )
        
        # Create the stacked area chart
        area_chart = alt.Chart(chart_data).mark_area().encode(
            x=alt.X('year:Q', title='Year'),
            y=alt.Y('value:Q', title=value_label, stack='zero'),
            color=alt.Color(
                'technology:N',
                title='Technology',
                scale=alt.Scale(scheme='category20')
            ),
            tooltip=[
                alt.Tooltip('year:Q', title='Year'),
                alt.Tooltip('technology:N', title='Technology'),
                alt.Tooltip('value:Q', title=value_label, format=',.0f')
            ]
        ).transform_filter(
            technology_selection
        ).add_params(
            technology_selection
        ).properties(
            height=400,
            title=f'{section_title} Over Time'
        ).interactive()
        
        st.altair_chart(area_chart, use_container_width=True)
        st.caption('💡 Click on technologies in the legend to show/hide them')
        
    except ImportError:
        # Fallback to simple line chart if Altair is not available
        st.line_chart(pivot)
    
    st.dataframe(display_frame, use_container_width=True)


def _filter_region_frame(frame: pd.DataFrame | None, region_value: int | str) -> pd.DataFrame:
    """Return ``frame`` filtered to ``region_value`` using canonical matching."""

    if frame is None or frame.empty or "region" not in frame.columns:
        return pd.DataFrame()

    working = frame.copy()
    working["region_canonical"] = working["region"].apply(canonical_region_value)
    filtered = working[working["region_canonical"] == region_value]
    return filtered.drop(columns=["region_canonical"], errors="ignore")


def _render_regional_dispatch_section(
    demand_df: pd.DataFrame | None,
    generation_df: pd.DataFrame | None,
    capacity_df: pd.DataFrame | None,
    cost_df: pd.DataFrame | None,
) -> None:
    """Display regional demand, generation, capacity, and cost summaries."""

    frames = [demand_df, generation_df, capacity_df, cost_df]
    if all(frame is None or frame.empty for frame in frames):
        st.caption('Engine did not return regional dispatch detail for this run.')
        return

    region_candidates: set[int | str] = set()
    for frame in frames:
        if frame is None or frame.empty or "region" not in frame.columns:
            continue
        for value in frame["region"].dropna():
            canonical = canonical_region_value(value)
            if isinstance(canonical, str) and canonical.strip().lower() == "default":
                continue
            region_candidates.add(canonical)

    if not region_candidates:
        st.caption('Regional dispatch detail unavailable for the current outputs.')
        return

    region_index = pd.DataFrame({"region": list(region_candidates)})
    region_options = region_selection_options(region_index)
    if not region_options:
        st.caption('Regional dispatch detail unavailable for the current outputs.')
        return

    option_labels = [label for label, _ in region_options]
    label_to_value = {label: value for label, value in region_options}
    selectbox = getattr(st, 'selectbox', None)

    if callable(selectbox):
        selected_label = selectbox(
            'Select a region to view detailed dispatch metrics',
            option_labels,
            key='dispatch_region_selection',
        )
    else:
        fallback_selector = getattr(st, 'multiselect', None)
        if fallback_selector is None:
            raise AttributeError('Streamlit interface missing selectbox and multiselect widgets')
        selected_options = fallback_selector(
            'Select a region to view detailed dispatch metrics',
            option_labels,
            default=[option_labels[0]],
            key='dispatch_region_selection_fallback',
        )
        selected_label = selected_options[0] if selected_options else option_labels[0]

    selected_value = label_to_value.get(selected_label, region_options[0][1])

    demand_filtered = _filter_region_frame(demand_df, selected_value)
    generation_filtered = _filter_region_frame(generation_df, selected_value)
    capacity_filtered = _filter_region_frame(capacity_df, selected_value)
    cost_filtered = _filter_region_frame(cost_df, selected_value)


    if demand_filtered is not None and not demand_filtered.empty:
        st.markdown('**Regional demand (MWh)**')
        demand_display = demand_filtered.copy()
        if 'year' in demand_display.columns:
            demand_display['year'] = pd.to_numeric(demand_display['year'], errors='coerce')
            demand_display = demand_display.dropna(subset=['year'])
            demand_display = demand_display.sort_values('year')
            if not demand_display.empty:
                st.bar_chart(demand_display.set_index('year')[['demand_mwh']])
        st.dataframe(demand_display, use_container_width=True)

    if generation_filtered is not None and not generation_filtered.empty:
        st.markdown('**Generation by technology (MWh)**')
        gen_display = generation_filtered.copy()
        if 'year' in gen_display.columns:
            gen_display['year'] = pd.to_numeric(gen_display['year'], errors='coerce')
            gen_display = gen_display.dropna(subset=['year'])
        if not gen_display.empty and {'year', 'fuel', 'generation_mwh'}.issubset(gen_display.columns):
            pivot = gen_display.pivot_table(
                index='year', columns='fuel', values='generation_mwh', aggfunc='sum'
            ).sort_index()
            if not pivot.empty:
                area_chart = getattr(st, 'area_chart', None)
                if area_chart is not None:
                    area_chart(pivot)
                else:
                    st.line_chart(pivot)
        st.dataframe(gen_display, use_container_width=True)

    if capacity_filtered is not None and not capacity_filtered.empty:
        st.markdown('**Capacity by technology**')
        cap_display = capacity_filtered.copy()
        if 'year' in cap_display.columns:
            cap_display['year'] = pd.to_numeric(cap_display['year'], errors='coerce')
            cap_display = cap_display.dropna(subset=['year'])
        if not cap_display.empty and {'year', 'fuel', 'capacity_mw'}.issubset(cap_display.columns):
            # Create interactive stacked bar chart with toggleable technologies
            try:
                import altair as alt
                
                # Prepare data
                chart_data = cap_display[['year', 'fuel', 'capacity_mw']].copy()
                chart_data = chart_data.rename(columns={'fuel': 'technology', 'capacity_mw': 'value'})
                
                # Create selection for toggling
                technology_selection = alt.selection_point(
                    fields=['technology'],
                    bind='legend'
                )
                
                # Create stacked bar chart
                bar_chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('year:O', title='Year'),
                    y=alt.Y('value:Q', title='Capacity (MW)', stack='zero'),
                    color=alt.Color(
                        'technology:N',
                        title='Technology',
                        scale=alt.Scale(scheme='category20')
                    ),
                    tooltip=[
                        alt.Tooltip('year:O', title='Year'),
                        alt.Tooltip('technology:N', title='Technology'),
                        alt.Tooltip('value:Q', title='Capacity (MW)', format=',.0f')
                    ]
                ).transform_filter(
                    technology_selection
                ).add_params(
                    technology_selection
                ).properties(
                    height=400,
                    title='Capacity by Technology Over Time'
                ).interactive()
                
                st.altair_chart(bar_chart, use_container_width=True)
                st.caption('💡 Click on technologies in the legend to show/hide them')
                
            except ImportError:
                # Fallback to line chart if Altair unavailable
                pivot = cap_display.pivot_table(
                    index='year', columns='fuel', values='capacity_mw', aggfunc='max'
                ).sort_index()
                if not pivot.empty:
                    st.line_chart(pivot)
        st.dataframe(cap_display, use_container_width=True)

    if cost_filtered is not None and not cost_filtered.empty:
        st.markdown('**Dispatch costs by technology ($)**')
        cost_display = cost_filtered.copy()
        st.dataframe(cost_display, use_container_width=True)


def _cleanup_session_temp_dirs() -> None:
    _ensure_streamlit()
    temp_dirs = st.session_state.get('temp_dirs', [])
    for path_str in temp_dirs:
        try:
            shutil.rmtree(path_str, ignore_errors=True)
        except Exception:  # pragma: no cover - best effort cleanup
            continue
    st.session_state['temp_dirs'] = []


def _reset_run_state_on_reload() -> None:
    try:
        _ensure_streamlit()
    except ModuleNotFoundError:  # pragma: no cover - GUI dependency missing
        return

    previous_token = st.session_state.get(_SESSION_RUN_TOKEN_KEY)
    if previous_token != _CURRENT_SESSION_RUN_TOKEN:
        st.session_state[_SESSION_RUN_TOKEN_KEY] = _CURRENT_SESSION_RUN_TOKEN
        st.session_state['run_in_progress'] = False
        st.session_state.pop('pending_run', None)


def _advance_script_iteration() -> int:
    """Increment and return the current Streamlit rerun iteration counter."""

    try:
        _ensure_streamlit()
    except ModuleNotFoundError:  # pragma: no cover - GUI dependency missing
        return 0

    current = int(st.session_state.get(_SCRIPT_ITERATION_KEY, 0)) + 1
    st.session_state[_SCRIPT_ITERATION_KEY] = current
    return current


def _recover_stuck_run_state(current_iteration: int) -> None:
    """Clear stale run state flags left behind by interrupted executions."""

    try:
        _ensure_streamlit()
    except ModuleNotFoundError:  # pragma: no cover - GUI dependency missing
        return

    if not st.session_state.get('run_in_progress'):
        st.session_state.pop(_ACTIVE_RUN_ITERATION_KEY, None)
        return

    active_iteration = st.session_state.get(_ACTIVE_RUN_ITERATION_KEY)
    stale_state = not isinstance(active_iteration, int) or active_iteration < current_iteration
    if stale_state:
        LOGGER.warning('Detected stale run_in_progress flag; resetting run state')
        st.session_state['run_in_progress'] = False
        st.session_state.pop('pending_run', None)
        st.session_state.pop(_ACTIVE_RUN_ITERATION_KEY, None)


def _build_run_summary(settings: Mapping[str, Any], *, config_label: str) -> list[tuple[str, str]]:
    """Return human-readable configuration details for confirmation dialogs."""

    def _as_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _as_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _bool_label(value: bool) -> str:
        return "Yes" if value else "No"

    start_year = _as_int(settings.get("start_year"))
    end_year = _as_int(settings.get("end_year"))

    if start_year is None and end_year is None:
        year_display = "Not specified"
    else:
        if start_year is None:
            start_year = end_year
        if end_year is None:
            end_year = start_year
        if start_year == end_year:
            year_display = f"{start_year}"
        else:
            year_display = f"{start_year} – {end_year}"

    carbon_enabled = bool(settings.get("carbon_policy_enabled", True))
    enable_floor = bool(settings.get("enable_floor", False)) if carbon_enabled else False
    enable_ccr = bool(settings.get("enable_ccr", False)) if carbon_enabled else False
    ccr1_enabled = bool(settings.get("ccr1_enabled", False)) if enable_ccr else False
    ccr2_enabled = bool(settings.get("ccr2_enabled", False)) if enable_ccr else False
    banking_enabled = (
        bool(settings.get("allowance_banking_enabled", False)) if carbon_enabled else False
    )

    control_period = settings.get("control_period_years") if carbon_enabled else None
    if not carbon_enabled:
        control_display = "Not applicable"
    elif control_period is None:
        control_display = "Automatic"
    else:
        control_display = str(control_period)

    price_enabled = bool(settings.get("carbon_price_enabled", False)) if carbon_enabled else False
    price_value = _as_float(settings.get("carbon_price_value")) if price_enabled else None
    price_schedule_raw = settings.get("carbon_price_schedule") if price_enabled else None
    schedule_entries: list[tuple[int, float]] = []
    if isinstance(price_schedule_raw, Mapping):
        for year_key, value in price_schedule_raw.items():
            year_val = _as_int(year_key)
            price_val = _as_float(value)
            if year_val is None or price_val is None:
                continue
            schedule_entries.append((year_val, price_val))
    schedule_entries.sort(key=lambda item: item[0])

    if not price_enabled:
        price_display = "Disabled"
    elif schedule_entries:
        first_year, first_price = schedule_entries[0]
        if len(schedule_entries) == 1:
            price_display = f"Schedule: {first_year} → ${first_price:,.2f}/ton"
        else:
            last_year, last_price = schedule_entries[-1]
            price_display = (
                f"Schedule ({len(schedule_entries)} entries): "
                f"{first_year} → ${first_price:,.2f}/ton, "
                f"{last_year} → ${last_price:,.2f}/ton"
            )
    elif price_value is not None:
        price_display = f"Flat ${price_value:,.2f}/ton"
    else:
        price_display = "Enabled (no price specified)"

    dispatch_network = bool(settings.get("dispatch_use_network", False))

    return [
        ("Configuration", config_label),
        ("Simulation years", year_display),
        ("Carbon cap enabled", _bool_label(carbon_enabled)),
        ("Minimum reserve price", _bool_label(enable_floor)),
        ("CCR enabled", _bool_label(enable_ccr)),
        ("CCR tranche 1", _bool_label(ccr1_enabled)),
        ("CCR tranche 2", _bool_label(ccr2_enabled)),
        ("Allowance banking enabled", _bool_label(banking_enabled)),
        ("Control period length", control_display),
        ("Carbon price", price_display),
        ("Dispatch uses network", _bool_label(dispatch_network)),
    ]



def _deduplicate_dataframe_columns(
    frame: pd.DataFrame, *, context: str | None = None
) -> pd.DataFrame:
    """Return ``frame`` without duplicate-named columns.

    Streamlit serialises DataFrames via PyArrow, which requires column labels to
    be unique. Engine outputs occasionally contain duplicate column names; this
    helper retains the first instance of each name while logging the duplicates
    for traceability so rendering can proceed without raising ``ValueError``.
    """

    if not isinstance(frame, pd.DataFrame):
        return frame

    if not frame.columns.duplicated().any():
        return frame

    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    if context:
        LOGGER.warning(
            "Dropping duplicate columns from %s: %s", context, duplicates
        )
    else:
        LOGGER.warning("Dropping duplicate columns: %s", duplicates)

    return frame.loc[:, ~frame.columns.duplicated()].copy()


def _render_results(result: Mapping[str, Any]) -> None:
    """Render charts and tables summarising the latest run results."""
    _ensure_streamlit()

    if 'error' in result:
        st.error(result['error'])
        return

    annual = result.get('annual')
    if not isinstance(annual, pd.DataFrame):
        annual = pd.DataFrame()
    else:
        annual = dataframe_to_carbon_vector(annual)

    demand_region_df = result.get('demand_by_region')
    if not isinstance(demand_region_df, pd.DataFrame):
        demand_region_df = pd.DataFrame()

    generation_region_df = result.get('generation_by_region')
    if not isinstance(generation_region_df, pd.DataFrame):
        generation_region_df = pd.DataFrame()

    capacity_region_df = result.get('capacity_by_region')
    if not isinstance(capacity_region_df, pd.DataFrame):
        capacity_region_df = pd.DataFrame()

    cost_region_df = result.get('cost_by_region')
    if not isinstance(cost_region_df, pd.DataFrame):
        cost_region_df = pd.DataFrame()

    display_annual = annual.copy()
    chart_data = pd.DataFrame()
    if not display_annual.empty and 'year' in display_annual.columns:
        display_annual['year'] = pd.to_numeric(display_annual['year'], errors='coerce')
        display_annual = display_annual.dropna(subset=['year'])
        display_annual = display_annual.sort_values('year')
        chart_data = display_annual.set_index('year')
    elif not display_annual.empty:
        chart_data = display_annual.copy()

    price_output_type = str(result.get('_price_output_type') or 'allowance')

    if price_output_type == 'carbon':
        carbon_field_aliases = {
            'allowance_price': 'cp_all',
            'allowance_price_exogenous_component': 'cp_exempt',
            'allowance_price_effective': 'cp_effective',
        }

        if not display_annual.empty:
            if 'allowance_price' in display_annual.columns and 'cp_all' in display_annual.columns:
                display_annual = display_annual.drop(columns=['allowance_price'])
            display_annual = display_annual.rename(columns=carbon_field_aliases)
            if 'cp_last' not in display_annual.columns and 'cp_all' in display_annual.columns:
                display_annual['cp_last'] = display_annual['cp_all']

        if not chart_data.empty:
            if 'allowance_price' in chart_data.columns and 'cp_all' in chart_data.columns:
                chart_data = chart_data.drop(columns=['allowance_price'])
            chart_data = chart_data.rename(columns=carbon_field_aliases)
            if 'cp_last' not in chart_data.columns and 'cp_all' in chart_data.columns:
                chart_data['cp_last'] = chart_data['cp_all']

        price_tab_label = 'Carbon price'
        price_section_title = 'Carbon price results'
        price_series_label = 'Carbon price ($/ton)'
        price_missing_caption = 'Engine did not return carbon price data for this run.'
    else:
        price_tab_label = 'Allowance price'
        price_section_title = 'Allowance market results'
        price_series_label = 'Allowance clearing price ($/ton)'
        price_missing_caption = 'Engine did not return allowance clearing price data for this run.'

    price_chart_column: str | None = None
    if not chart_data.empty:
        if price_output_type == 'carbon':
            chart_source = None
            for candidate in ('cp_effective', 'cp_all', 'cp_last'):
                if candidate in chart_data.columns:
                    chart_source = candidate
                    break
            if chart_source is not None:
                chart_data = chart_data.rename(columns={chart_source: price_series_label})
                price_chart_column = price_series_label
        elif 'p_co2' in chart_data.columns:
            chart_data = chart_data.rename(columns={'p_co2': price_series_label})
            price_chart_column = price_series_label
        elif 'cp_last' in chart_data.columns:
            chart_data = chart_data.rename(columns={'cp_last': price_series_label})
            price_chart_column = price_series_label

    display_price_table = display_annual.copy()
    display_price_table = _deduplicate_dataframe_columns(
        display_price_table, context="annual price table (initial)"
    )
    LOGGER.debug(
        "gui_price_render price_output_type=%s columns=%s",
        price_output_type,
        list(display_price_table.columns),
    )

    if price_output_type == 'carbon':
        legacy_columns = [
            'allowance_price',
            'allowance_price_exogenous_component',
            'allowance_price_effective',
            'p_co2',
            'p_co2_all',
            'p_co2_exc',
            'p_co2_eff',
        ]
        existing_legacy = [col for col in legacy_columns if col in display_price_table.columns]
        if existing_legacy:
            display_price_table = display_price_table.drop(columns=existing_legacy)

        if 'cp_effective' in display_price_table.columns and price_series_label not in display_price_table.columns:
            cp_effective_values = display_price_table.loc[:, 'cp_effective']
            if isinstance(cp_effective_values, pd.DataFrame):
                cp_effective_values = cp_effective_values.iloc[:, 0]
            display_price_table[price_series_label] = cp_effective_values

        display_price_table = _deduplicate_dataframe_columns(
            display_price_table, context="annual price table (carbon)"
        )

        allowed_price_columns = [
            'year',
            price_series_label,
            'cp_effective',
            'cp_all',
            'cp_exempt',
            'cp_last',
            'emissions_tons',
        ]
        unique_columns: list[str] = []
        for column in allowed_price_columns:
            if column in display_price_table.columns and column not in unique_columns:
                unique_columns.append(column)
        if unique_columns:
            display_price_table = display_price_table.loc[:, unique_columns]
        LOGGER.debug("gui_price_render_selected columns=%s", unique_columns)
    else:
        # Allowance output path
        if 'cp_last' in display_price_table.columns:
            display_price_table = display_price_table.rename(
                columns={'cp_last': price_series_label}
            )

        display_price_table = _deduplicate_dataframe_columns(
            display_price_table, context="annual price table (allowance)"
        )



    emissions_df = load_emissions_data(result)

    price_df = result.get('price_by_region')
    if not isinstance(price_df, pd.DataFrame):
        price_df = pd.DataFrame()
        price_flags = {'year': False, 'region': False, 'price': False}
    else:
        price_df = price_df.copy()
        price_flags = result.get(
            '_price_field_flags', {'year': True, 'region': True, 'price': True}
        )

    flows_df = result.get('flows')
    if not isinstance(flows_df, pd.DataFrame):
        flows_df = pd.DataFrame()

    demand_region_df = _extract_result_frame(
        result,
        'demand_by_region',
        aliases=('demand_by_region_output',),
    )
    generation_region_df = _extract_result_frame(
        result,
        'generation_by_region',
        aliases=('generation_by_region_output',),
    )
    capacity_region_df = _extract_result_frame(
        result,
        'capacity_by_region',
        aliases=('capacity_by_region_output',),
    )
    cost_region_df = _extract_result_frame(
        result,
        'cost_by_region',
        aliases=('cost_by_region_output',),
    )
    if not isinstance(demand_region_df, pd.DataFrame):
        demand_region_df = pd.DataFrame()
    if not isinstance(generation_region_df, pd.DataFrame):
        generation_region_df = pd.DataFrame()
    if not isinstance(capacity_region_df, pd.DataFrame):
        capacity_region_df = pd.DataFrame()
    if not isinstance(cost_region_df, pd.DataFrame):
        cost_region_df = pd.DataFrame()

    st.caption('Visualisations reflect the most recent model run.')

    show_bank_tab = price_output_type != 'carbon'
    tab_labels = [price_tab_label, 'Emissions']
    if show_bank_tab:
        tab_labels.append('Allowance bank')
    tab_labels.append('Dispatch costs')

    tabs = st.tabs(tab_labels)
    tab_iter = iter(tabs)
    price_tab = next(tab_iter)
    emissions_tab = next(tab_iter)
    bank_tab = next(tab_iter) if show_bank_tab else None
    dispatch_tab = next(tab_iter)

    with price_tab:
        st.subheader(price_section_title)
        if display_annual.empty:
            st.info('Engine did not return annual results for this run.')
        else:
            if price_chart_column and price_chart_column in chart_data.columns:
                st.markdown(f'**{price_series_label}**')
                st.line_chart(chart_data[[price_chart_column]])
            else:
                st.caption(price_missing_caption)

            st.markdown('---')
            st.dataframe(display_price_table, use_container_width=True)

    with emissions_tab:
        st.subheader('Emissions overview')
        emissions_source = str(result.get('_emissions_source') or '')
        for message in result.get('_emissions_messages', []):
            st.info(message)

        if display_annual.empty and emissions_df.empty:
            if not result.get('_emissions_messages') and emissions_source == 'missing':
                st.info('Emissions data missing from engine outputs.')
            elif not result.get('_emissions_messages'):
                st.info('No emissions data available for this run.')
            else:
                st.info('Engine did not return emissions data for this run.')
        else:
            if not chart_data.empty and 'emissions_tons' in chart_data.columns:
                st.markdown('**Total emissions (tons)**')
                st.bar_chart(chart_data[['emissions_tons']])
            elif not display_annual.empty:
                st.caption('Engine did not return total emissions data for this run.')

            if emissions_df.empty:
                if not display_annual.empty and emissions_source != 'missing':
                    st.caption('No regional emissions data available for this run.')
                elif emissions_source == 'missing' and not result.get('_emissions_messages'):
                    st.info('Emissions data missing from engine outputs.')
            else:
                display_emissions = emissions_df.copy()
                if 'year' in display_emissions.columns:
                    display_emissions['year'] = pd.to_numeric(
                        display_emissions['year'], errors='coerce'
                    )
                    display_emissions = display_emissions.dropna(subset=['year'])
                    display_emissions = display_emissions.sort_values(
                        ['year', 'region_label']
                    )

                region_options = region_selection_options(display_emissions)
                if region_options:
                    option_labels = [label for label, _ in region_options]
                    label_to_value = {label: value for label, value in region_options}
                    selected_labels = st.multiselect(
                        'Select regions to include',
                        option_labels,
                        default=option_labels,
                        key='emissions_region_selection',
                    )
                    selected_values = [
                        label_to_value[label]
                        for label in selected_labels
                        if label in label_to_value
                    ]
                    filtered_emissions = filter_emissions_by_regions(
                        display_emissions, selected_values
                    )
                    if 'year' in filtered_emissions.columns:
                        filtered_emissions = filtered_emissions.sort_values(
                            ['year', 'region_label']
                        )
                    summary_df = summarize_emissions_totals(filtered_emissions)
                    if summary_df.empty:
                        st.caption(
                            'Engine did not return emissions data for the selected regions.'
                        )
                    else:
                        st.markdown('**Emissions by region (tons)**')
                        if alt is not None:
                            chart = (
                                alt.Chart(summary_df)
                                .mark_bar()
                                .encode(
                                    x=alt.X('year:O', title='Year'),
                                    y=alt.Y(
                                        'emissions_tons:Q',
                                        title='Emissions (tons)',
                                        stack='zero',
                                    ),
                                    color=alt.Color('region_label:N', title='Region'),
                                    tooltip=[
                                        alt.Tooltip('year:O', title='Year'),
                                        alt.Tooltip('region_label:N', title='Region'),
                                        alt.Tooltip(
                                            'emissions_tons:Q',
                                            title='Emissions (tons)',
                                            format=',.0f',
                                        ),
                                    ],
                                )
                                .properties(width='container')
                            )
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            pivot = (
                                summary_df.pivot_table(
                                    index='year',
                                    columns='region_label',
                                    values='emissions_tons',
                                    aggfunc='sum',
                                )
                                .fillna(0.0)
                                .sort_index()
                            )
                            st.bar_chart(pivot)

                    st.markdown('---')
                    st.dataframe(
                        filtered_emissions.drop(
                            columns=['region_canonical'], errors='ignore'
                        ),
                        use_container_width=True,
                    )
                else:
                    st.caption(
                        'Engine did not return regional emissions data for this run; showing raw table below.'
                    )
                    st.dataframe(
                        display_emissions.drop(
                            columns=['region_canonical'], errors='ignore'
                        ),
                        use_container_width=True,
                    )


    if bank_tab is not None:
        with bank_tab:
            st.subheader('Allowance bank balance')
            if display_annual.empty:
                st.info('Engine did not return annual results for this run.')
            elif 'bank' in chart_data.columns:
                st.markdown('**Bank balance (tons)**')
                st.line_chart(chart_data[['bank']])
                st.bar_chart(chart_data[['bank']])
            else:
                st.caption('Engine did not return allowance bank data for this run.')

    with dispatch_tab:
        st.subheader('Dispatch costs and network results')
        if price_df.empty and flows_df.empty:
            st.info('Engine did not return dispatch outputs for this run.')
        else:
            if not price_df.empty:
                if all(price_flags.get(key, False) for key in ('year', 'region', 'price')):
                    display_price = price_df.copy()
                    display_price['year'] = pd.to_numeric(
                        display_price['year'], errors='coerce'
                    )
                    display_price = display_price.dropna(subset=['year'])

                    if 'region' in display_price.columns:
                        price_pivot = display_price.pivot_table(
                            index='year',
                            columns='region',
                            values='price',
                            aggfunc='mean',
                        ).sort_index()
                        st.markdown('**Dispatch costs by region ($/MWh)**')
                        st.line_chart(price_pivot)
                    else:
                        st.caption(
                            'Engine did not return regional dispatch cost data for this run; showing raw table below.'
                        )
                        st.dataframe(display_price, use_container_width=True)
                else:
                    missing = [key for key, present in price_flags.items() if not present]
                    if missing:
                        missing_display = ', '.join(sorted(missing))
                        st.caption(
                            'Dispatch price data missing expected column(s): '
                            f"{missing_display}. Showing available data below."
                        )
                    st.dataframe(price_df, use_container_width=True)

            if not flows_df.empty:
                st.markdown('---')
                st.markdown('**Interregional energy flows (MWh)**')
                st.dataframe(flows_df, use_container_width=True)
            elif price_df.empty:
                st.caption('Engine did not return dispatch network data for this run.')

            if any(
                isinstance(frame, pd.DataFrame) and not frame.empty
                for frame in (
                    demand_region_df,
                    generation_region_df,
                    capacity_region_df,
                    cost_region_df,
                )
            ):
                st.markdown('---')
                _render_regional_dispatch_section(
                    demand_region_df,
                    generation_region_df,
                    capacity_region_df,
                    cost_region_df,
                )

    # --- Technology sections ---
    capacity_df = _extract_result_frame(result, 'capacity_by_technology')
    _render_technology_section(
        capacity_df,
        section_title='Capacity by technology',
        candidate_columns=[
            ('capacity_mw', 'Capacity (MW)'),
            ('capacity', 'Capacity'),
            ('value', 'Capacity'),
        ],
    )

    generation_df = _extract_result_frame(result, 'generation_by_technology')
    _render_technology_section(
        generation_df,
        section_title='Generation by technology',
        candidate_columns=[
            ('generation_mwh', 'Generation (MWh)'),
            ('generation', 'Generation'),
            ('value', 'Generation'),
        ],
    )

    # --- Capacity factor section ---
    _render_capacity_factor_section(generation_df, capacity_df)

    # --- Assumption overrides ---
    documentation = result.get('documentation')
    overrides: list[str] = []
    if isinstance(documentation, Mapping):
        overrides = [str(entry) for entry in documentation.get('assumption_overrides', [])]

    st.subheader('Assumption overrides')
    if overrides:
        for note in overrides:
            st.markdown(f'- {note}')
    else:
        st.caption('No assumption overrides were applied in this run.')

    # --- Downloads ---
    st.subheader('Download outputs')
    csv_files = result.get('csv_files')
    if isinstance(csv_files, Mapping) and csv_files:
        for filename, content in sorted(csv_files.items()):
            st.download_button(
                label=f'Download {filename}',
                data=content,
                file_name=filename,
                mime='text/csv',
            )
    else:
        st.caption('No CSV outputs are available for download.')

    temp_dir = result.get('temp_dir')
    if temp_dir:
        st.caption(f'Temporary files saved to {temp_dir}')


def _render_outputs_panel(last_result: Mapping[str, Any] | None) -> None:
    """Render the main outputs panel with charts for the latest run."""
    _ensure_streamlit()
    if not isinstance(last_result, Mapping):
        last_result = get_recent_result()
    if not isinstance(last_result, Mapping) or not last_result:
        st.caption('Run the model to populate this panel with results.')
        return
    _render_results(last_result)


def _page_config_from_mapping(
    config: Mapping[str, Any] | None,
) -> tuple[str, str]:
    """Return Streamlit page configuration derived from ``config``."""

    default_title = 'Policy Simulator'
    default_layout = 'wide'

    if not isinstance(config, Mapping):
        return default_title, default_layout

    gui_section = config.get('gui')
    if not isinstance(gui_section, Mapping):
        return default_title, default_layout

    page_title = default_title
    title_value = gui_section.get('title')
    if title_value is not None:
        title_text = str(title_value).strip()
        if title_text:
            page_title = title_text

    layout_value = gui_section.get('layout')
    page_layout = default_layout
    if isinstance(layout_value, str):
        candidate = layout_value.strip().lower()
        if candidate in {'wide', 'centered'}:
            page_layout = candidate

    return page_title, page_layout


def main() -> None:
    """Streamlit entry point."""
    _ensure_streamlit()

    default_config_data: dict[str, Any] | None
    config_load_error: Exception | None = None
    try:
        default_config_data = _load_config_data(DEFAULT_CONFIG_PATH)
    except Exception as exc:  # pragma: no cover - defensive UI path
        default_config_data = None
        config_load_error = exc

    page_title, page_layout = _page_config_from_mapping(default_config_data)
    st.set_page_config(page_title=page_title, layout=page_layout)
    st.session_state.setdefault('last_result', None)
    st.session_state.setdefault('temp_dirs', [])
    st.session_state.setdefault('run_in_progress', False)
    record_recent_result(st.session_state.get('last_result'))
    current_iteration = _advance_script_iteration()
    _reset_run_state_on_reload()
    _recover_stuck_run_state(current_iteration)

    module_errors: list[str] = []
    assumption_notes: list[str] = []
    assumption_errors: list[str] = []

    if config_load_error is not None:
        st.warning(f'Unable to load default configuration: {config_load_error}')

    run_config: dict[str, Any] = (
        copy.deepcopy(default_config_data) if default_config_data else {}
    )
    config_label = DEFAULT_CONFIG_PATH.name
    selected_years: list[int] = []
    candidate_years: list[int] = []
    frames_for_run: FramesType | None = None
    current_year = date.today().year
    start_year_val = int(run_config.get('start_year', current_year)) if run_config else int(current_year)
    default_end_year = start_year_val + 1
    end_year_val = int(run_config.get('end_year', default_end_year)) if run_config else int(default_end_year)
    if end_year_val <= start_year_val:
        end_year_val = start_year_val + 1

    carbon_settings = CarbonModuleSettings(
        enabled=False,
        price_enabled=False,
        enable_floor=False,
        enable_ccr=False,
        ccr1_enabled=False,
        ccr2_enabled=False,
        ccr1_price=None,
        ccr2_price=None,
        ccr1_escalator_pct=0.0,
        ccr2_escalator_pct=0.0,
        banking_enabled=False,
        coverage_regions=["All"],
        control_period_years=None,
        price_per_ton=0.0,
        price_escalator_pct=0.0,
        initial_bank=0.0,
        cap_regions=[],
        cap_start_value=None,
        cap_reduction_mode="percent",
        cap_reduction_value=0.0,
        cap_schedule={},
        floor_value=0.0,
        floor_escalator_mode="fixed",
        floor_escalator_value=0.0,
        floor_schedule={},
        price_schedule={},
        errors=[],
    )


    demand_settings = DemandModuleSettings(
        enabled=False,
        curve_by_region={},
        forecast_by_region={},
        load_forecasts={},
        errors=[],
    )


    dispatch_settings = DispatchModuleSettings(
        enabled=False,
        mode='single',
        capacity_expansion=False,
        reserve_margins=False,
        deep_carbon_pricing=False,
    )
    incentives_settings = IncentivesModuleSettings(
        enabled=False,
        production_credits=[],
        investment_credits=[],
    )
    output_directory_raw = run_config.get('output_name') if run_config else None
    output_directory = str(output_directory_raw) if output_directory_raw else 'outputs'
    downloads_root = get_downloads_directory()
    resolved_output_path = downloads_root if not output_directory else downloads_root / output_directory
    outputs_settings = OutputsModuleSettings(
        enabled=False,
        directory=output_directory,
        resolved_path=resolved_output_path,
        show_csv_downloads=False,
    )
    run_in_progress = bool(st.session_state.get("run_in_progress"))
    run_clicked = False

    header_container = st.container()
    with header_container:
        st.markdown(
            """
            <style>
            div[data-testid="column"] h1 {
                margin-top: 0 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        control_col, title_col = st.columns([0.3, 1], gap="large")
        with control_col:
            run_clicked = st.button(
                "Run Model",
                type="primary",
                use_container_width=True,
                disabled=run_in_progress,
            )
            run_status_placeholder = st.empty()
        with title_col:
            st.title('Policy Simulator')

    if run_in_progress:
        run_status_placeholder.info("Simulation in progress…")
    else:
        blocking_errors = st.session_state.get("run_blocking_errors")
        if blocking_errors:
            if isinstance(blocking_errors, Mapping):
                blocking_iterable = blocking_errors.values()
            elif isinstance(blocking_errors, (str, bytes)):
                blocking_iterable = [blocking_errors]
            elif isinstance(blocking_errors, Iterable):
                blocking_iterable = blocking_errors
            else:
                blocking_iterable = [blocking_errors]

            formatted_errors = "\n".join(
                f"- {str(message).strip()}"
                for message in blocking_iterable
                if str(message).strip()
            )
            guidance = "Resolve the configuration issues highlighted in the inputs panel before running."
            if formatted_errors:
                guidance = f"{guidance}\n{formatted_errors}"
            run_status_placeholder.error(guidance)

    with st.sidebar:
        st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)

        last_result_mapping = st.session_state.get("last_result")
        if not isinstance(last_result_mapping, Mapping):
            last_result_mapping = None
        if last_result_mapping is None:
            last_result_mapping = get_recent_result()

        model_tab, regional_tab, documentation_tab, output_tab = st.tabs(
            ["Model-Wide Inputs", "Regional Inputs", "Documentation", "Output Management"]
        )

        with model_tab:
            # -------- General --------
            general_label, general_expanded = SIDEBAR_SECTIONS[0]
            general_expander = st.expander(general_label, expanded=general_expanded)
            general_result = _render_general_config_section(
                general_expander,
                default_source=DEFAULT_CONFIG_PATH,
                default_label=DEFAULT_CONFIG_PATH.name,
                default_config=default_config_data,
            )
            run_config = general_result.run_config
            config_label = general_result.config_label
            candidate_years = general_result.candidate_years
            start_year_val = general_result.start_year
            end_year_val = general_result.end_year
            selected_years = general_result.selected_years

            helper_years: Iterable[int] | None
            if selected_years:
                helper_years = selected_years
            elif end_year_val >= start_year_val:
                helper_years = list(range(start_year_val, end_year_val + 1))
            else:
                helper_years = None

            fuel_label, fuel_expanded = SIDEBAR_SECTIONS[1]
            fuel_expander = st.expander(fuel_label, expanded=fuel_expanded)
            fuel_price_selection = _render_fuel_cost_helper_section(
                fuel_expander,
                run_config=run_config,
                years=helper_years,
            )
            general_result.fuel_price_selection = fuel_price_selection

            helper_enabled = bool(fuel_price_selection)
            if st is not None:
                helper_enabled = bool(
                    st.session_state.get(
                        _FUEL_PRICE_HELPER_ENABLED_KEY,
                        helper_enabled,
                    )
                )

            if helper_enabled and fuel_price_selection:
                selection_description = ", ".join(
                    f"{fuel.replace('_', ' ').title()}: {scenario}"
                    for fuel, scenario in sorted(fuel_price_selection.items())
                )
                assumption_notes.append(
                    f"Fuel supply curve scenarios selected – {selection_description}."
                )

            if start_year_val >= end_year_val:
                year_error = "Simulation end year must be greater than start year."
                if year_error not in module_errors:
                    module_errors.append(year_error)

            # -------- Carbon --------
            carbon_label, carbon_expanded = SIDEBAR_SECTIONS[3]
            carbon_expander = st.expander(carbon_label, expanded=carbon_expanded)
            carbon_settings = _render_carbon_policy_section(
                carbon_expander,
                run_config,
                region_options=general_result.regions,
                lock_inputs=general_result.lock_carbon_controls,
            )
            module_errors.extend(carbon_settings.errors)

            # Prepare default frames (defensive)
            custom_for_default_frames: Mapping[str, Any] | None = None
            modules_cfg = run_config.get("modules") if isinstance(run_config, Mapping) else None
            if isinstance(modules_cfg, Mapping):
                demand_cfg = modules_cfg.get("demand")
                if isinstance(demand_cfg, Mapping):
                    custom_candidate = demand_cfg.get("custom_load_forecasts")
                    if isinstance(custom_candidate, Mapping):
                        custom_for_default_frames = custom_candidate

            try:
                frames_for_run = _build_default_frames(
                    selected_years
                    or list(range(int(start_year_val), int(end_year_val) + 1)),
                    carbon_policy_enabled=bool(
                        carbon_settings.enabled and not carbon_settings.price_enabled
                    ),
                    banking_enabled=bool(carbon_settings.banking_enabled),
                    carbon_price_schedule=(
                        carbon_settings.price_schedule if carbon_settings.price_enabled else None
                    ),
                    forecast_bundles=_selected_forecast_bundles_from_state(),
                    custom_forecasts=custom_for_default_frames,
                )
                frames_for_run = _apply_region_weights_to_frames(
                    frames_for_run,
                    run_config.get("regions")
                    if isinstance(run_config.get("regions"), Mapping)
                    else None,
                )
            except Exception as exc:  # pragma: no cover
                frames_for_run = None
                st.warning(f"Unable to prepare default assumption tables: {exc}")

            # -------- Dispatch --------
            dispatch_label, dispatch_expanded = SIDEBAR_SECTIONS[4]
            dispatch_expander = st.expander(dispatch_label, expanded=dispatch_expanded)
            dispatch_settings = _render_dispatch_section(
                dispatch_expander, run_config, frames_for_run
            )
            module_errors.extend(dispatch_settings.errors)

            # -------- Incentives --------
            incentives_label, incentives_expanded = SIDEBAR_SECTIONS[5]
            incentives_expander = st.expander(incentives_label, expanded=incentives_expanded)
            incentives_settings = _render_incentives_section(
                incentives_expander,
                run_config,
                frames_for_run,
            )
            module_errors.extend(incentives_settings.errors)

        with regional_tab:
            st.caption(
                "Configure regional inputs, such as demand curve assignments, for the simulation."
            )

            # -------- Demand --------
            demand_label, demand_expanded = SIDEBAR_SECTIONS[2]
            demand_expander = st.expander(demand_label, expanded=demand_expanded)
            demand_settings_raw = _render_demand_module_section(
                demand_expander,
                run_config,
                regions=general_result.regions,
                years=general_result.selected_years or selected_years,
            )
            demand_settings = _coerce_demand_module_settings(demand_settings_raw)
            module_errors.extend(demand_settings.errors)

            # -------- Assumptions --------
            st.divider()
            inputs_header = st.container()
            inputs_header.subheader("Assumption overrides")
            inputs_header.caption(
                "Adjust core assumption tables or upload CSV files to override the defaults."
            )
            if frames_for_run is not None:
                demand_tab, units_tab, fuels_tab, transmission_tab = st.tabs(
                    ["Demand", "Units", "Fuels", "Transmission"]
                )
                with demand_tab:
                    frames_for_run, notes, errors = _render_demand_controls(
                        frames_for_run, selected_years
                    )
                    assumption_notes.extend(notes)
                    assumption_errors.extend(errors)
                with units_tab:
                    frames_for_run, notes, errors = _render_units_controls(frames_for_run)
                    assumption_notes.extend(notes)
                    assumption_errors.extend(errors)
                with fuels_tab:
                    frames_for_run, notes, errors = _render_fuels_controls(frames_for_run)
                    assumption_notes.extend(notes)
                    assumption_errors.extend(errors)
                with transmission_tab:
                    frames_for_run, notes, errors = _render_transmission_controls(frames_for_run)
                    assumption_notes.extend(notes)
                    assumption_errors.extend(errors)

                if assumption_errors:
                    st.warning(
                        "Resolve the highlighted assumption issues before running the simulation."
                    )
            else:
                st.info(
                    "Default assumption tables are unavailable due to a previous error. "
                    "Resolve the issue above to edit inputs through the GUI."
                )

        with documentation_tab:
            st.caption("Review the selected load forecasts and generated documentation.")

            selection_source: Mapping[str, str] | None = None
            demand_selection = demand_settings.load_forecasts
            if demand_selection:
                selection_source = demand_selection
            elif st is not None:
                state_selection = st.session_state.get("forecast_selections")
                if isinstance(state_selection, Mapping) and state_selection:
                    selection_source = dict(state_selection)

            custom_selection: dict[str, Mapping[str, Any]] = {}
            if isinstance(demand_settings.custom_load_forecasts, Mapping):
                for key, entry in demand_settings.custom_load_forecasts.items():
                    if not entry:
                        continue
                    state_key = str(key).strip().upper()
                    if state_key:
                        if isinstance(entry, Mapping):
                            custom_selection[state_key] = entry
                        else:
                            custom_selection[state_key] = {"state": state_key, "source": "Uploaded"}

            if selection_source or custom_selection:
                state_iso_labels: dict[str, str] = {}
                if st is not None:
                    label_state = st.session_state.get("forecast_iso_labels")
                    if isinstance(label_state, Mapping):
                        for key, value in label_state.items():
                            state_code = str(key).strip().upper()
                            if state_code and value:
                                state_iso_labels[state_code] = str(value)

                rows: list[dict[str, Any]] = []
                seen_states: set[str] = set()
                if selection_source:
                    for state_key, stored_value in selection_source.items():
                        state_code = str(state_key).strip().upper()
                        iso_label, scenario_label = _decode_forecast_selection(stored_value)
                        display_iso = state_iso_labels.get(state_code) or iso_label or ""
                        display_state = state_code or str(state_key)
                        display_scenario = scenario_label or str(stored_value)
                        rows.append(
                            {
                                "State": display_state,
                                "ISO": display_iso,
                                "Scenario": display_scenario,
                            }
                        )
                        if state_code:
                            seen_states.add(state_code)

                for state_code, entry in custom_selection.items():
                    if state_code in seen_states:
                        continue
                    display_state = state_code
                    iso_label = "Custom"
                    scenario_label = "Uploaded"
                    if isinstance(entry, Mapping):
                        source_label = entry.get("source")
                        if entry.get("iso"):
                            iso_label = str(entry.get("iso"))
                        years_value = entry.get("years")
                        if source_label:
                            scenario_label = f"Uploaded ({source_label})"
                        if isinstance(years_value, Sequence) and years_value:
                            try:
                                year_values = sorted({int(year) for year in years_value})
                            except (TypeError, ValueError):
                                year_values = []
                        else:
                            year_values = []
                        if year_values:
                            if len(year_values) == 1:
                                year_text = str(year_values[0])
                            else:
                                year_text = f"{year_values[0]}–{year_values[-1]}"
                            scenario_label = f"{scenario_label} [{year_text}]"

                    rows.append(
                        {
                            "State": display_state,
                            "ISO": iso_label,
                            "Scenario": scenario_label,
                        }
                    )
                    seen_states.add(state_code)

                if rows:
                    rows = sorted(
                        rows,
                        key=lambda item: (
                            str(item.get("ISO", "")),
                            str(item.get("State", "")),
                        ),
                    )
                st.subheader("Current load forecast selections")
                st.table(pd.DataFrame(rows))
            else:
                st.info("No load forecast selections available; using synthetic fallback data.")

            documentation_info = None
            if isinstance(last_result_mapping, Mapping):
                documentation_info = last_result_mapping.get("documentation")
            if isinstance(documentation_info, Mapping) and documentation_info:
                manifest_markdown = documentation_info.get("manifest_markdown")
                manifest_data = documentation_info.get("manifest")
                manifest_paths = documentation_info.get("manifest_paths")
                if isinstance(manifest_paths, Mapping):
                    manifest_json_path = manifest_paths.get("json")
                    manifest_md_path = manifest_paths.get("md")
                else:
                    manifest_json_path = manifest_md_path = None
                deep_doc_path = documentation_info.get("deep_doc_path")
                st.subheader("Run manifest")
                if manifest_markdown:
                    st.markdown(manifest_markdown)
                if manifest_data and _documentation_file_ready(manifest_json_path):
                    st.download_button(
                        "Download manifest (JSON)",
                        data=json.dumps(manifest_data, indent=2).encode("utf-8"),
                        file_name="run_manifest.json",
                        mime="application/json",
                    )
                if manifest_markdown and _documentation_file_ready(manifest_md_path):
                    st.download_button(
                        "Download manifest (Markdown)",
                        data=str(manifest_markdown).encode("utf-8"),
                        file_name="run_manifest.md",
                        mime="text/markdown",
                    )

                assumption_summary = documentation_info.get("assumption_summary")
                if isinstance(assumption_summary, Mapping) and assumption_summary:
                    st.subheader("Assumption summary")
                    st.table(pd.DataFrame([assumption_summary]))

                deep_doc_markdown = documentation_info.get("deep_doc_markdown")
                if deep_doc_markdown:
                    with st.expander("Model documentation preview", expanded=False):
                        st.markdown(deep_doc_markdown)
                    if _documentation_file_ready(deep_doc_path):
                        st.download_button(
                            "Download model documentation",
                            data=str(deep_doc_markdown).encode("utf-8"),
                            file_name="model_documentation.md",
                            mime="text/markdown",
                        )
            else:
                st.info("Run the model to generate manifest and documentation outputs.")

        with output_tab:
            # -------- Output management --------
            outputs_expander = st.expander(
                OUTPUTS_SECTION_LABEL, expanded=OUTPUTS_SECTION_DEFAULT_EXPANDED
            )
            outputs_settings = _render_outputs_section(
                outputs_expander,
                run_config,
                last_result_mapping,
            )
            module_errors.extend(outputs_settings.errors)

    # Finalize selected years defensively
    if st is not None:
        try:
            start_year_state = int(st.session_state.get("start_year_slider", start_year_val))
        except (TypeError, ValueError):
            start_year_state = int(start_year_val)
        try:
            end_year_state = int(st.session_state.get("end_year_slider", end_year_val))
        except (TypeError, ValueError):
            end_year_state = int(end_year_val)
        start_year_val = start_year_state
        end_year_val = end_year_state

    default_years: list[int] = []
    if start_year_val < end_year_val:
        default_years = list(range(int(start_year_val), int(end_year_val) + 1))
    elif start_year_val == end_year_val:
        default_years = [int(start_year_val)]

    try:
        selected_years = _select_years(candidate_years, start_year_val, end_year_val)
    except Exception:
        selected_years = []

    if selected_years:
        try:
            selected_min = min(int(year) for year in selected_years)
            selected_max = max(int(year) for year in selected_years)
        except ValueError:
            selected_years = []
        else:
            selected_years = list(range(selected_min, selected_max + 1))
    else:
        selected_years = list(default_years)

    # Ensure frames if earlier failed
    if frames_for_run is None:
        custom_for_default_frames: Mapping[str, Any] | None = None
        modules_cfg = run_config.get("modules") if isinstance(run_config, Mapping) else None
        if isinstance(modules_cfg, Mapping):
            demand_cfg = modules_cfg.get("demand")
            if isinstance(demand_cfg, Mapping):
                custom_candidate = demand_cfg.get("custom_load_forecasts")
                if isinstance(custom_candidate, Mapping):
                    custom_for_default_frames = custom_candidate

        try:
            frames_for_run = _build_default_frames(
                selected_years
                or default_years
                or [int(start_year_val)],
                carbon_policy_enabled=bool(
                    carbon_settings.enabled and not carbon_settings.price_enabled
                ),
                banking_enabled=bool(carbon_settings.banking_enabled),
                carbon_price_schedule=(
                    carbon_settings.price_schedule if carbon_settings.price_enabled else None
                ),
                forecast_bundles=_selected_forecast_bundles_from_state(),
                custom_forecasts=custom_for_default_frames,
            )
            frames_for_run = _apply_region_weights_to_frames(
                frames_for_run,
                run_config.get("regions")
                if isinstance(run_config.get("regions"), Mapping)
                else None,
            )
        except Exception as exc:  # pragma: no cover
            frames_for_run = None
            st.warning(f"Unable to prepare default assumption tables: {exc}")

    if module_errors:
        st.warning(
            "Resolve the module configuration issues highlighted in the sidebar before running the simulation."
        )

    # ---- Run orchestration state ----
    execute_run = False
    run_inputs: dict[str, Any] | None = None

    run_in_progress = bool(st.session_state.get("run_in_progress"))

    def _collect_run_blocking_errors() -> list[str]:
        blocking: list[str] = []
        for message in itertools.chain(assumption_errors, module_errors):
            if not message:
                continue
            text = str(message).strip()
            if text and text not in blocking:
                blocking.append(text)
        return blocking

    # Build the payload that actually drives the engine
    dispatch_use_network = bool(
        dispatch_settings.enabled and dispatch_settings.mode == "network"
    )

    if selected_years:
        staged_years = list(selected_years)
    elif start_year_val < end_year_val:
        staged_years = list(range(int(start_year_val), int(end_year_val) + 1))
    else:
        staged_years = []

    _update_staged_run_inputs(
        config_source=run_config,
        start_year=int(start_year_val),
        end_year=int(end_year_val),
        years=staged_years,
        carbon_policy_enabled=bool(carbon_settings.enabled),
        enable_floor=bool(carbon_settings.enable_floor),
        price_floor_value=carbon_settings.floor_value,
        price_floor_escalator_mode=carbon_settings.floor_escalator_mode,
        price_floor_escalator_value=carbon_settings.floor_escalator_value,
        price_floor_schedule=dict(carbon_settings.floor_schedule),
        enable_ccr=bool(carbon_settings.enable_ccr),
        ccr1_enabled=bool(carbon_settings.ccr1_enabled),
        ccr2_enabled=bool(carbon_settings.ccr2_enabled),
        ccr1_price=carbon_settings.ccr1_price,
        ccr2_price=carbon_settings.ccr2_price,
        ccr1_escalator_pct=carbon_settings.ccr1_escalator_pct,
        ccr2_escalator_pct=carbon_settings.ccr2_escalator_pct,
        allowance_banking_enabled=bool(carbon_settings.banking_enabled),
        coverage_regions=list(carbon_settings.coverage_regions),
        cap_regions=list(getattr(carbon_settings, "cap_regions", [])),
        initial_bank=carbon_settings.initial_bank,
        control_period_years=carbon_settings.control_period_years,
        carbon_price_enabled=bool(carbon_settings.price_enabled),
        carbon_price_value=carbon_settings.price_per_ton
        if carbon_settings.price_enabled
        else 0.0,
        carbon_price_escalator_pct=carbon_settings.price_escalator_pct,
        carbon_price_schedule=(
            dict(carbon_settings.price_schedule) if carbon_settings.price_enabled else {}
        ),
        carbon_cap_start_value=carbon_settings.cap_start_value,
        carbon_cap_reduction_mode=str(carbon_settings.cap_reduction_mode),
        carbon_cap_reduction_value=float(carbon_settings.cap_reduction_value),
        carbon_cap_schedule=dict(carbon_settings.cap_schedule),
        dispatch_use_network=dispatch_use_network,
        dispatch_capacity_expansion=bool(
            getattr(dispatch_settings, "capacity_expansion", True)
        ),
        dispatch_deep_carbon=bool(getattr(dispatch_settings, "deep_carbon_pricing", False)),
        module_config=run_config.get("modules", {}),
        frames=frames_for_run,
        assumption_notes=list(assumption_notes),
        forecast_bundles=_selected_forecast_bundles_from_state(),
        regions=run_config.get("regions"),
        states=run_config.get("states"),
    )

    def _clone_run_payload(source: Mapping[str, Any]) -> dict[str, Any]:
        base = {
            k: v
            for k, v in source.items()
            if k not in {"frames", "assumption_notes", "forecast_bundles"}
        }
        try:
            cloned = copy.deepcopy(base)
        except Exception:  # pragma: no cover
            cloned = dict(base)
        cloned["frames"] = source.get("frames")
        notes_value = source.get("assumption_notes")
        if isinstance(notes_value, Iterable) and not isinstance(
            notes_value, (str, bytes, Mapping)
        ):
            cloned["assumption_notes"] = [str(note) for note in notes_value]
        elif notes_value not in (None, ""):
            cloned["assumption_notes"] = [str(notes_value)]
        else:
            cloned["assumption_notes"] = []
        bundles_value = source.get("forecast_bundles")
        if isinstance(bundles_value, Iterable) and not isinstance(bundles_value, (str, bytes)):
            cloned["forecast_bundles"] = list(bundles_value)
        elif bundles_value:
            cloned["forecast_bundles"] = [bundles_value]
        return cloned

    staged_inputs = _staged_run_inputs_state()

    # Handle Run button -> validate and stage execution
    if run_clicked:
        if run_in_progress:
            st.info(
                "A simulation is already in progress. Wait for it to finish before starting another run."
            )
        else:
            blocking = _collect_run_blocking_errors()
            if blocking:
                st.error("Resolve the configuration issues above before running the simulation.")
                st.session_state["run_blocking_errors"] = blocking
                st.session_state["run_in_progress"] = False
            else:
                staged_payload = _clone_run_payload(staged_inputs)
                if not staged_payload.get("frames"):
                    staged_payload["frames"] = frames_for_run
                if not staged_payload.get("forecast_bundles"):
                    staged_payload["forecast_bundles"] = _selected_forecast_bundles_from_state()
                run_inputs = staged_payload
                execute_run = True
                st.session_state["confirmed_run_inputs"] = staged_payload
                st.session_state.pop("run_blocking_errors", None)
                st.session_state["run_in_progress"] = True
                st.session_state[_ACTIVE_RUN_ITERATION_KEY] = st.session_state.get(
                    _ACTIVE_RUN_ITERATION_KEY, 0
                )

    result = st.session_state.get("last_result")

    # Outputs/progress scaffolding
    progress_state = _ensure_progress_state()
    progress_section = st.container()
    with progress_section:
        st.subheader("Run progress")
        progress_summary_placeholder = st.empty()
        progress_status_placeholder = st.empty()

    def _render_progress_summary() -> None:
        progress_summary_placeholder.empty()
        if st.session_state.get("run_in_progress"):
            progress_summary_placeholder.info("Simulation in progress…")
            return
        message = (progress_state.message or "").strip()
        if progress_state.stage == "complete" and message:
            progress_summary_placeholder.success(message)
        elif progress_state.stage == "error" and message:
            progress_summary_placeholder.error(message)
        elif message:
            progress_summary_placeholder.info(message)
        else:
            progress_summary_placeholder.caption(
                "Press **Run Model** to stage a simulation run."
            )

    _render_progress_summary()

    inputs_for_run: Mapping[str, Any] = (
        run_inputs or st.session_state.get("confirmed_run_inputs") or {}
    )
    run_result: Mapping[str, Any] | None = None

    # --- Execution branch ---
    if execute_run:
        frames_for_execution = inputs_for_run.get("frames") or frames_for_run
        bundles_for_run = inputs_for_run.get("forecast_bundles") or _selected_forecast_bundles_from_state()

        region_weights_for_run: Mapping[str, float] | None = None
        raw_weights = inputs_for_run.get("regions")
        if isinstance(raw_weights, Mapping):
            region_weights_for_run = raw_weights
        else:
            config_source = inputs_for_run.get("config_source")
            if isinstance(config_source, Mapping):
                config_regions = config_source.get("regions")
                if isinstance(config_regions, Mapping):
                    region_weights_for_run = config_regions
        if region_weights_for_run is None:
            config_regions = run_config.get("regions")
            if isinstance(config_regions, Mapping):
                region_weights_for_run = config_regions

        frames_for_execution = _apply_region_weights_to_frames(
            frames_for_execution, region_weights_for_run
        )

        assumption_notes_value = inputs_for_run.get("assumption_notes", [])
        if isinstance(assumption_notes_value, Iterable) and not isinstance(
            assumption_notes_value, (str, bytes, Mapping)
        ):
            assumption_notes_for_run = [str(note) for note in assumption_notes_value]
        elif assumption_notes_value not in (None, ""):
            assumption_notes_for_run = [str(assumption_notes_value)]
        else:
            assumption_notes_for_run = []

        progress_display: _ProgressDisplay | None = None

        scenario_log_entries: dict[str, str] = {}
        if isinstance(demand_settings.load_forecasts, Mapping) and demand_settings.load_forecasts:
            scenario_log_entries.update({str(k).strip().upper(): str(v) for k, v in demand_settings.load_forecasts.items() if str(k)})
        elif isinstance(st.session_state.get("forecast_selections"), Mapping):
            scenario_log_entries.update(
                {
                    str(k).strip().upper(): str(v)
                    for k, v in st.session_state.get("forecast_selections", {}).items()
                    if str(v)
                }
            )

        if isinstance(demand_settings.custom_load_forecasts, Mapping):
            for state_key, entry in demand_settings.custom_load_forecasts.items():
                state_code = str(state_key).strip().upper()
                if not state_code or state_code in scenario_log_entries:
                    continue
                label = "Uploaded"
                if isinstance(entry, Mapping):
                    source_label = entry.get("source")
                    if source_label:
                        label = f"Uploaded ({source_label})"
                    years_value = entry.get("years")
                    if isinstance(years_value, Sequence) and years_value:
                        try:
                            year_values = sorted({int(year) for year in years_value})
                        except (TypeError, ValueError):
                            year_values = []
                    else:
                        year_values = []
                    if year_values:
                        if len(year_values) == 1:
                            year_text = str(year_values[0])
                        else:
                            year_text = f"{year_values[0]}–{year_values[-1]}"
                        label = f"{label} [{year_text}]"
                scenario_log_entries[state_code] = label

        LOGGER.info(
            "Starting simulation run with load forecast scenarios: %s",
            scenario_log_entries,
        )

        try:
            st.session_state["run_in_progress"] = True
            st.session_state[_ACTIVE_RUN_ITERATION_KEY] = st.session_state.get(
                _ACTIVE_RUN_ITERATION_KEY, 0
            )
            _cleanup_session_temp_dirs()

            progress_state = _reset_progress_state()
            progress_state.stage = "initializing"
            progress_state.message = "Preparing simulation inputs…"
            progress_state.percent_complete = 0
            progress_summary_placeholder.empty()

            with progress_status_placeholder.container():
                with st.status("Run progress", expanded=True) as status_block:
                    progress_bar = st.progress(0)
                    log_container = st.empty()
                    progress_display = _ProgressDisplay(
                        status_block,
                        progress_bar,
                        log_container,
                        progress_state,
                    )

                    def _handle_stage(stage: str, payload: Mapping[str, object]) -> None:
                        if progress_display is not None:
                            progress_display.handle_stage(stage, payload)

                    def _handle_progress(stage: str, payload: Mapping[str, object]) -> None:
                        if progress_display is None:
                            return
                        progress_display.handle_iteration(stage, payload)

                    try:
                        run_result = run_policy_simulation(
                            inputs_for_run.get("config_source", run_config),
                            start_year=inputs_for_run.get("start_year", start_year_val),
                            end_year=inputs_for_run.get("end_year", end_year_val),
                            carbon_policy_enabled=bool(
                                inputs_for_run.get("carbon_policy_enabled", carbon_settings.enabled)
                            ),
                            enable_floor=bool(
                                inputs_for_run.get("enable_floor", carbon_settings.enable_floor)
                            ),
                            enable_ccr=bool(
                                inputs_for_run.get("enable_ccr", carbon_settings.enable_ccr)
                            ),
                            ccr1_enabled=bool(
                                inputs_for_run.get("ccr1_enabled", carbon_settings.ccr1_enabled)
                            ),
                            ccr2_enabled=bool(
                                inputs_for_run.get("ccr2_enabled", carbon_settings.ccr2_enabled)
                            ),
                            ccr1_price=inputs_for_run.get(
                                "ccr1_price", carbon_settings.ccr1_price
                            ),
                            ccr2_price=inputs_for_run.get(
                                "ccr2_price", carbon_settings.ccr2_price
                            ),
                            ccr1_escalator_pct=inputs_for_run.get(
                                "ccr1_escalator_pct", carbon_settings.ccr1_escalator_pct
                            ),
                            ccr2_escalator_pct=inputs_for_run.get(
                                "ccr2_escalator_pct", carbon_settings.ccr2_escalator_pct
                            ),
                            allowance_banking_enabled=bool(
                                inputs_for_run.get(
                                    "allowance_banking_enabled", carbon_settings.banking_enabled
                                )
                            ),
                            initial_bank=float(
                                inputs_for_run.get("initial_bank", carbon_settings.initial_bank)
                            ),
                            coverage_regions=inputs_for_run.get(
                                "coverage_regions", carbon_settings.coverage_regions
                            ),
                            control_period_years=inputs_for_run.get(
                                "control_period_years", carbon_settings.control_period_years
                            ),
                            price_floor_value=inputs_for_run.get(
                                "price_floor_value", carbon_settings.floor_value
                            ),
                            price_floor_escalator_mode=inputs_for_run.get(
                                "price_floor_escalator_mode", carbon_settings.floor_escalator_mode
                            ),
                            price_floor_escalator_value=inputs_for_run.get(
                                "price_floor_escalator_value", carbon_settings.floor_escalator_value
                            ),
                            price_floor_schedule=inputs_for_run.get(
                                "price_floor_schedule", carbon_settings.floor_schedule
                            ),
                            cap_regions=inputs_for_run.get(
                                "cap_regions", getattr(carbon_settings, "cap_regions", [])
                            ),
                            carbon_price_enabled=inputs_for_run.get(
                                "carbon_price_enabled", carbon_settings.price_enabled
                            ),
                            carbon_price_value=inputs_for_run.get(
                                "carbon_price_value", carbon_settings.price_per_ton
                            ),
                            carbon_price_schedule=inputs_for_run.get(
                                "carbon_price_schedule", carbon_settings.price_schedule
                            ),
                            dispatch_use_network=bool(
                                inputs_for_run.get("dispatch_use_network", dispatch_use_network)
                            ),
                            dispatch_capacity_expansion=inputs_for_run.get(
                                "dispatch_capacity_expansion",
                                getattr(dispatch_settings, "capacity_expansion", True),
                            ),
                            deep_carbon_pricing=bool(
                                inputs_for_run.get(
                                    "dispatch_deep_carbon",
                                    getattr(dispatch_settings, "deep_carbon_pricing", False),
                                )
                            ),
                            module_config=inputs_for_run.get(
                                "module_config", run_config.get("modules", {})
                            ),
                            frames=frames_for_execution,
                            forecast_bundles=bundles_for_run,
                            run_id=str(inputs_for_run.get("run_id"))
                            if inputs_for_run.get("run_id")
                            else None,
                            assumption_notes=assumption_notes_for_run,
                            progress_cb=_handle_progress,
                            stage_cb=_handle_stage,
                        )
                    except Exception as exc:  # defensive guard
                        LOGGER.exception("Policy simulation failed during execution")
                        run_result = {"error": str(exc)}
                        if progress_display is not None:
                            progress_display.fail(f"Simulation failed: {exc}")
                    else:
                        if isinstance(run_result, Mapping) and "error" in run_result:
                            if progress_display is not None:
                                progress_display.fail(
                                    f"Simulation failed: {run_result['error']}"
                                )
                        elif isinstance(run_result, Mapping):
                            if progress_display is not None:
                                progress_display.complete()
                        else:
                            if progress_display is not None:
                                progress_display.fail(
                                    "Simulation ended before producing results."
                                )

        except Exception as exc:  # defensive guard
            LOGGER.exception("Policy simulation failed before execution could complete")
            run_result = {"error": str(exc)}
            if progress_display is not None:
                progress_display.fail(f"Simulation failed: {exc}")

        finally:
            st.session_state["run_in_progress"] = False
            st.session_state.pop(_ACTIVE_RUN_ITERATION_KEY, None)

            if isinstance(run_result, Mapping):
                sanitized_result = dict(run_result)
                sanitized_result["documentation"] = _sanitize_documentation_mapping(
                    run_result.get("documentation")
                )
                run_result = sanitized_result
                if "error" in run_result:
                    progress_state.stage = "error"
                    progress_state.message = f"Simulation failed: {run_result['error']}"
                else:
                    progress_state.stage = "complete"
                    progress_state.percent_complete = 100
                    progress_state.message = "Simulation complete. Outputs updated below."
                st.session_state["last_result"] = run_result
            else:
                progress_state.stage = "error"
                progress_state.message = "Simulation ended before producing results."

            _render_progress_summary()

    # --- Outputs panel ---
    outputs_container = st.container()
    with outputs_container:
        st.subheader("Model outputs")
        if st.session_state.get("run_in_progress"):
            st.info("Simulation in progress... progress updates appear above.")
        else:
            _render_outputs_panel(st.session_state.get("last_result"))

    # --- Final guidance to user ---
    if isinstance(st.session_state.get("last_result"), Mapping):
        if "error" in st.session_state["last_result"]:
            st.error(st.session_state["last_result"]["error"])
        else:
            st.info(
                "Review the outputs above to explore charts and downloads from the most recent run."
            )
    else:
        st.info("Use the inputs panel to configure and run the simulation.")


def streamlit_app() -> None:
    """Expose a callable entry point for external runners."""

    main()


app = streamlit_app

if __name__ == "__main__":  # pragma: no cover
    main()

