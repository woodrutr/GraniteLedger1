"""Utilities for recording model setup documentation."""

from __future__ import annotations

import json
import shutil
import types
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency
    import numpy as _np  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    _np = None


def create_model_setup_summary(settings) -> dict[str, Any]:
    """Return a sanitized snapshot of user-selected inputs for ``settings``."""

    snapshot = settings.build_user_inputs_snapshot()
    return _sanitize(snapshot)


def write_model_setup_documentation(settings, directory: str | Path | None = None) -> dict[str, Path | None]:
    """Persist model setup documentation files for the provided ``settings`` instance."""

    target_dir = Path(directory) if directory is not None else Path(settings.OUTPUT_ROOT)
    target_dir.mkdir(parents=True, exist_ok=True)

    data = create_model_setup_summary(settings)

    config_copy_path: Path | None = None
    config_path = getattr(settings, 'config_path', None)
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            config_copy_path = target_dir / 'model_run_config_snapshot.toml'
            shutil.copy2(config_path, config_copy_path)
            data.setdefault('run_metadata', {})['config_snapshot_file'] = config_copy_path.name

    json_path = target_dir / 'model_setup.json'
    with json_path.open('w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=2, sort_keys=False)
        json_file.write('\n')

    markdown_path = target_dir / 'model_setup.md'
    markdown_path.write_text(_build_markdown_summary(data), encoding='utf-8')

    return {'json': json_path, 'markdown': markdown_path, 'config_snapshot': config_copy_path}


def _sanitize(value: Any) -> Any:
    """Convert ``value`` into JSON-serializable types."""

    if isinstance(value, dict):
        return {str(key): _sanitize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    if isinstance(value, set):
        return [_sanitize(item) for item in sorted(value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, types.SimpleNamespace):
        return _sanitize(vars(value))
    if _np is not None and isinstance(value, _np.generic):  # pragma: no cover - defensive guard
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _build_markdown_summary(data: dict[str, Any]) -> str:
    """Convert sanitized documentation data into a human-readable Markdown report."""

    sections = [
        ('Run metadata', data.get('run_metadata')),
        ('CLI arguments', data.get('cli_arguments')),
        ('Modules', data.get('modules')),
        ('Iterative settings', data.get('iterative_settings')),
        ('Temporal settings', data.get('temporal_settings')),
        ('Spatial settings', data.get('spatial_settings')),
        ('Electricity settings', data.get('electricity_settings')),
        ('Residential settings', data.get('residential_settings')),
        ('Hydrogen settings', data.get('hydrogen_settings')),
        ('Carbon policy', data.get('carbon_policy')),
        ('Capacity build limits', data.get('capacity_build_limits')),
        ('Missing validations', data.get('missing_validations')),
    ]

    lines: list[str] = ['# Model Setup Summary', '']

    for title, content in sections:
        if content:
            lines.append(f'## {title}')
            lines.extend(_render_mapping_items(content))
            lines.append('')

    raw_config = data.get('raw_configuration')
    if raw_config is not None:
        lines.append('## Raw configuration')
        lines.extend(_json_code_block(raw_config, 0))
        lines.append('')

    return '\n'.join(lines).rstrip() + '\n'


def _render_mapping_items(mapping: Any, indent_level: int = 0) -> list[str]:
    if isinstance(mapping, dict):
        items = mapping.items()
    elif isinstance(mapping, list):
        items = enumerate(mapping, start=1)
    else:
        return [_format_simple(mapping)]

    lines: list[str] = []
    for key, value in items:
        lines.extend(_render_item(str(key), value, indent_level))
    return lines


def _render_item(key: str, value: Any, indent_level: int) -> list[str]:
    label = key.replace('_', ' ')
    indent = '  ' * indent_level

    if isinstance(value, dict):
        lines = [f'{indent}- **{label}:**']
        lines.extend(_json_code_block(value, indent_level + 1))
        return lines

    if isinstance(value, list):
        if not value:
            return [f'{indent}- **{label}:** []']
        if all(_is_simple(item) for item in value):
            formatted = ', '.join(_format_simple(item) for item in value)
            return [f'{indent}- **{label}:** {formatted}']
        lines = [f'{indent}- **{label}:**']
        lines.extend(_json_code_block(value, indent_level + 1))
        return lines

    return [f'{indent}- **{label}:** {_format_simple(value)}']


def _json_code_block(data: Any, indent_level: int) -> list[str]:
    indent = '  ' * indent_level
    inner_indent = '  ' * (indent_level + 1)
    block = json.dumps(data, indent=2, sort_keys=True)
    lines = [f'{indent}```json']
    lines.extend(f'{inner_indent}{line}' for line in block.splitlines())
    lines.append(f'{indent}```')
    return lines


def _is_simple(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _format_simple(value: Any) -> str:
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if value is None:
        return 'None'
    return str(value)
