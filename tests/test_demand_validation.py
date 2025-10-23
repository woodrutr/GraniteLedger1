from __future__ import annotations

import pandas as pd
import pytest

from common.validators import DemandValidationError, validate_demand_table


def test_validate_gui_duplicates_warns_and_sums() -> None:
    df = pd.DataFrame(
        {
            "Region": ["zone_a", "zone_a"],
            "Year": [2024, 2024],
            "Demand_MWh": [10, 5],
        }
    )

    normalized, warnings = validate_demand_table(df, ei_units_df=None)

    assert warnings and any("Duplicate" in message for message in warnings)
    assert len(normalized) == 1
    assert normalized.loc[0, "mwh"] == pytest.approx(15.0)
    assert normalized.loc[0, "region_id"] == "ZONE_A"
    assert normalized.loc[0, "year"] == 2024


def test_validate_negative_demand_raises() -> None:
    df = pd.DataFrame(
        {
            "region_id": ["zone_a"],
            "year": [2024],
            "mwh": [-1.0],
        }
    )

    with pytest.raises(DemandValidationError) as excinfo:
        validate_demand_table(df, ei_units_df=None)

    assert "Non-positive" in str(excinfo.value)


def test_validate_region_missing_from_units() -> None:
    df = pd.DataFrame(
        {
            "region": ["ZONE_A"],
            "year": [2024],
            "demand_mwh": [1.0],
        }
    )
    ei_units = pd.DataFrame({"region_id": ["ZONE_B"]})

    with pytest.raises(DemandValidationError) as excinfo:
        validate_demand_table(df, ei_units)

    assert excinfo.value.details == {"missing_in_units": ["ZONE_A"]}


def test_validate_missing_years_warns() -> None:
    df = pd.DataFrame(
        {
            "region_id": ["ZONE_A"],
            "year": [2024],
            "mwh": [10.0],
        }
    )

    normalized, warnings = validate_demand_table(
        df,
        ei_units_df=None,
        required_years=[2023, 2024],
    )

    assert normalized.shape == (1, 3)
    assert warnings and any("Missing years" in message for message in warnings)
