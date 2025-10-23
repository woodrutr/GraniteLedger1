
import json
import subprocess
import sys
from pathlib import Path

from main.definitions import PROJECT_ROOT


def _tiny_config() -> str:
    return """
years = [2025, 2026]

[allowance_market]
enabled = true
ccr1_enabled = false
ccr2_enabled = false
bank0 = 100000.0
annual_surrender_frac = 1.0
carry_pct = 1.0

[allowance_market.cap]
"2025" = 500000.0
"2026" = 480000.0

[allowance_market.floor]
"2025" = 5.0
"2026" = 7.0
""".strip()


def test_cli_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / 'config.toml'
    config_path.write_text(_tiny_config())

    output_dir = tmp_path / 'outputs'
    cmd = [
        sys.executable,
        '-m',
        'cli.run',
        '--config',
        str(config_path),
        '--years',
        '2025-2026',
        '--out',
        str(output_dir),
    ]

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    expected_files = ['allowance.csv', 'emissions.csv', 'prices.csv', 'flows.csv']
    for name in expected_files:
        csv_path = output_dir / name
        assert csv_path.exists(), (
            f"Expected {name} to be generated.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        assert csv_path.stat().st_size > 0, f'{name} should not be empty'

    # Documentation outputs are written to a run-specific directory under output/
    manifest_dirs = list((output_dir).glob('*'))
    run_dirs = [path for path in manifest_dirs if path.is_dir()]
    assert run_dirs, 'Expected a run documentation directory to be created'

    manifest_json = None
    manifest_md = None
    model_doc = None
    for run_dir in run_dirs:
        candidate_json = run_dir / 'run_manifest.json'
        candidate_md = run_dir / 'run_manifest.md'
        candidate_model_doc = run_dir / 'model_documentation.md'
        if candidate_json.exists() and candidate_md.exists() and candidate_model_doc.exists():
            manifest_json = candidate_json
            manifest_md = candidate_md
            model_doc = candidate_model_doc
            break

    assert manifest_json is not None, 'run_manifest.json not found'
    assert manifest_md is not None, 'run_manifest.md not found'
    assert model_doc is not None, 'model_documentation.md not found'

    manifest_data = json.loads(manifest_json.read_text())
    assert 'load_forecasts' in manifest_data
