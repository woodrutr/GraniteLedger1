"""Pytest fixtures for electricity unit tests."""

from pathlib import Path

import pytest


@pytest.fixture
def minimal_carbon_policy_inputs(monkeypatch):
    """Provide in-memory defaults for carbon policy inputs used in tests."""

    pd = pytest.importorskip('pandas')
    from src.models.electricity.scripts import preprocessor as prep

    original_read_csv = prep.pd.read_csv

    def _mock_read_csv(filepath, *args, **kwargs):
        filename = Path(filepath).name if isinstance(filepath, (str, Path)) else ''
        if filename == 'CarbonCapGroupMap.csv':
            return pd.DataFrame(
                {
                    'cap_group': ['national', 'national'],
                    'region': [7, 8],
                }
            )
        if filename == 'SupplyCurve.csv':
            records: list[dict] = []
            for region in (7, 8):
                for year in (2025, 2030):
                    for step in (1, 2):
                        records.append(
                            {
                                'region': region,
                                'tech': 1,
                                'step': step,
                                'year': year,
                                'SupplyCurve': 1.0,
                            }
                        )
            return pd.DataFrame(records)
        return original_read_csv(filepath, *args, **kwargs)

    monkeypatch.setattr(prep.pd, 'read_csv', _mock_read_csv)
    yield
    # monkeypatch restores the original read_csv on teardown
