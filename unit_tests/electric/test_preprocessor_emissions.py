"""Tests for emissions rate validation in the electricity preprocessor."""

from __future__ import annotations

from pathlib import Path
import importlib

import pytest

from main.definitions import PROJECT_ROOT

pytest.importorskip("pandas")

config_setup = importlib.import_module("src.common.config_setup")
prep = importlib.import_module("src.models.electricity.scripts.preprocessor")


def test_emissions_rate_requires_values_for_unknown_tech():
    """Unknown technologies must provide explicit emissions data."""

    config_path = Path(PROJECT_ROOT, "src/common", "run_config.toml")
    settings = config_setup.Config_settings(config_path, test=True)

    setin = prep.Sets(settings)

    unknown_tech = "unknown-tech"
    assert unknown_tech not in setin.T_gen
    setin.T_gen = list(setin.T_gen) + [unknown_tech]

    with pytest.raises(
        ValueError,
        match=(
            r"EmissionsRate input is missing emissions rate data for technologies: "
            r"unknown-tech"
        ),
    ):
        prep.preprocessor(setin)
