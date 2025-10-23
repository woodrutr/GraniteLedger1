import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.carbon_price_utils import (
    CarbonPriceFixture,
    assert_aliases_match_canonical,
    with_carbon_vector_columns,
)

pd = pytest.importorskip("pandas")


def test_gui_price_column_selection_smoke(monkeypatch):
    gui = importlib.import_module("gui.app")
    table = pd.DataFrame(
        [
            CarbonPriceFixture(
                year=2025,
                all=1.0,
                effective=2.0,
                exempt=2.0,
                last=0.0,
            ).as_row()
            | {"emissions_tons": 10.0}
        ]
    )
    table = with_carbon_vector_columns(table)
    assert_aliases_match_canonical(table)

    result = {
        "annual": table,
        "_price_output_type": "carbon",
        "_price_field_flags": {"year": True, "region": False, "price": True},
    }

    monkeypatch.setattr(gui, "load_emissions_data", lambda _result: pd.DataFrame())

    class DummyStreamlit:
        class _Tab:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        def tabs(self, labels):
            return [self._Tab() for _ in labels]

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(gui, "st", DummyStreamlit())

    gui._render_results(result)
