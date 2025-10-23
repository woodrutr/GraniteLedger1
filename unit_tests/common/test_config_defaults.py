from __future__ import annotations

import importlib
import textwrap
import re
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    try:
        import tomli as tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
        raise ModuleNotFoundError(
            'Python 3.11+ or the tomli package is required to read TOML configuration files.'
        ) from exc
import pytest

pytest.importorskip("pandas")

Config_settings = importlib.import_module("src.common.config_setup").Config_settings
elec_preprocessor = importlib.import_module(
    "src.models.electricity.scripts.preprocessor"
)


def test_missing_electric_switches_use_defaults(tmp_path):
    """Missing electricity switches should fall back to safe defaults."""

    config_text = textwrap.dedent(
        """
        default_mode = "elec"
        electricity = true
        hydrogen = false
        residential = false
        tol = 0.1
        force_10 = false
        max_iter = 1
        years = [2025]
        regions = [7]
        scale_load = "annual"
        h2_data_folder = "input/hydrogen/all_regions"
        """
    ).strip()

    config_path = tmp_path / "trimmed_config.toml"
    config_path.write_text(config_text)

    settings = Config_settings(config_path, test=True)

    assert settings.sw_temporal == "default"
    assert settings.sw_agg_years == 0
    assert settings.sw_trade == 0
    assert settings.sw_expansion == 0
    assert settings.sw_rm == 0
    assert settings.sw_ramp == 0
    assert settings.sw_reserves == 0
    assert settings.sw_learning == 0

    sets = elec_preprocessor.Sets(settings)

    assert sets.sw_trade == 0
    assert sets.sw_expansion == 0
    assert sets.sw_rm == 0
    assert sets.sw_ramp == 0
    assert sets.sw_reserves == 0
    assert sets.sw_learning == 0
    assert sets.sw_agg_years == 0


def test_output_folder_name_depends_on_config(tmp_path):
    """Output directory names combine mode and a hash of the configuration."""

    base_config = textwrap.dedent(
        """
        default_mode = "elec"
        electricity = true
        hydrogen = false
        residential = false
        tol = 0.1
        force_10 = false
        max_iter = 1
        years = [2025]
        regions = [7]
        scale_load = "annual"
        h2_data_folder = "input/hydrogen/all_regions"
        """
    ).strip()

    config_path = tmp_path / 'run_config_1.toml'
    config_path.write_text(base_config)

    settings = Config_settings(config_path, test=True)
    name = settings._determine_output_folder(tomllib.loads(base_config))

    assert name.startswith(f"{settings.selected_mode}_")
    suffix = name.split('_', 1)[1]
    assert re.fullmatch(r"[0-9a-f]{10}", suffix)

    alt_config = base_config.replace('regions = [7]', 'regions = [8]')
    alt_path = tmp_path / 'run_config_2.toml'
    alt_path.write_text(alt_config)

    alt_settings = Config_settings(alt_path, test=True)
    alt_name = alt_settings._determine_output_folder(tomllib.loads(alt_config))

    assert alt_name.startswith(f"{alt_settings.selected_mode}_")
    assert alt_name != name


def test_sanitize_output_name():
    """Custom output directory overrides are normalised to safe names."""

    assert Config_settings._sanitize_output_name(' Run: Example! ') == 'Run_Example'

    with pytest.raises(ValueError):
        Config_settings._sanitize_output_name('   ___   ')


def test_ensure_unique_output_dir(tmp_path):
    """Existing directories trigger incremental suffixes to avoid overwrites."""

    base_dir = tmp_path / 'output'
    candidate = base_dir / 'elec_1234567890'
    candidate.mkdir(parents=True)

    unique, status = Config_settings._ensure_unique_output_dir(candidate)
    assert unique.name == 'elec_1234567890_01'
    assert status.converged is True
    assert status.iterations == 2
    assert status.metadata.get('suffix') == 1

    unique.mkdir()

    next_candidate, next_status = Config_settings._ensure_unique_output_dir(candidate)
    assert next_candidate.name == 'elec_1234567890_02'
    assert next_status.converged is True
    assert next_status.iterations == 3
    assert next_status.metadata.get('suffix') == 2
