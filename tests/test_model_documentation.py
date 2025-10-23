"""Tests for model setup documentation utilities."""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

from main.definitions import PROJECT_ROOT
from src.common.config_setup import Config_settings
from src.common.documentation import create_model_setup_summary, write_model_setup_documentation


def test_model_setup_documentation_includes_runtime_overrides(tmp_path):
    config_source = Path(PROJECT_ROOT, 'src/common', 'run_config.toml')
    config_copy = tmp_path / 'run_config.toml'
    config_copy.write_bytes(config_source.read_bytes())

    args = SimpleNamespace(
        op_mode='h2',
        debug=True,
        output_name='doc-test',
        capacity_build_limits={'7': {'Solar': {'2030': 123.0}}},
    )

    settings = Config_settings(config_copy, args=args, test=True)

    summary = create_model_setup_summary(settings)
    assert summary['run_metadata']['selected_mode'] == 'h2'
    assert summary['run_metadata']['mode_source'] == 'runtime_argument'

    capacity_values = summary['capacity_build_limits']['values']
    assert '7' in capacity_values
    assert 'Solar' in capacity_values['7']
    assert capacity_values['7']['Solar']['2030'] == 123.0

    output_dir = tmp_path / 'docs'
    paths = write_model_setup_documentation(settings, directory=output_dir)

    json_path = paths['json']
    markdown_path = paths['markdown']
    config_snapshot = paths['config_snapshot']

    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data['run_metadata']['selected_mode'] == 'h2'
    assert data['run_metadata']['config_snapshot_file'] == config_snapshot.name

    assert config_snapshot and config_snapshot.exists()
    assert markdown_path.exists()
    markdown_text = markdown_path.read_text()
    assert 'Model Setup Summary' in markdown_text
    assert 'h2' in markdown_text
    assert 'capacity build limits' in markdown_text.lower()
