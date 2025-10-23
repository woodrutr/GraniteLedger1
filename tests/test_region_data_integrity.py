from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from gui.region_metadata import DEFAULT_REGION_METADATA

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _unique_int_values(df, *columns: str) -> set[int]:
    values: set[int] = set()
    for column in columns:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        values.update(int(value) for value in series.astype(int).tolist())
    return values


def test_region_metadata_is_unique_and_complete() -> None:
    metadata = DEFAULT_REGION_METADATA
    assert len(metadata) == 25

    ids_from_keys = set(metadata.keys())
    ids_from_values = {meta.id for meta in metadata.values()}
    assert ids_from_keys == ids_from_values

    codes = [meta.code for meta in metadata.values()]
    assert len(codes) == len(set(codes))

    labels = [meta.label for meta in metadata.values()]
    assert len(labels) == len(set(labels))


def test_region_ids_align_with_input_data() -> None:
    metadata_ids = set(DEFAULT_REGION_METADATA.keys())

    hydrogen_regions = pd.read_csv(
        PROJECT_ROOT / "input" / "hydrogen" / "all_regions" / "regions.csv"
    )
    hydrogen_hubs = pd.read_csv(
        PROJECT_ROOT / "input" / "hydrogen" / "all_regions" / "hubs.csv"
    )

    assert metadata_ids == _unique_int_values(hydrogen_regions, "Region")
    assert metadata_ids == _unique_int_values(hydrogen_hubs, "region")

    cem_inputs = PROJECT_ROOT / "input" / "electricity" / "cem_inputs"
    electricity_checks = {
        "CarbonCapGroupMap.csv": ("region",),
        "SupplyCurve.csv": ("region",),
        "SupplyPrice.csv": ("region",),
        "ReserveMargin.csv": ("region",),
        "H2Price.csv": ("region",),
    }

    for filename, columns in electricity_checks.items():
        df = pd.read_csv(cem_inputs / filename)
        values = _unique_int_values(df, *columns)
        assert metadata_ids == values, f"{filename} regions mismatch"


def test_cap_regions_match_supply_regions() -> None:
    cem_inputs = PROJECT_ROOT / "input" / "electricity" / "cem_inputs"
    cap_map = pd.read_csv(cem_inputs / "CarbonCapGroupMap.csv")
    supply_curve = pd.read_csv(cem_inputs / "SupplyCurve.csv")
    supply_price = pd.read_csv(cem_inputs / "SupplyPrice.csv")

    cap_ids = _unique_int_values(cap_map, "region")
    supply_curve_ids = _unique_int_values(supply_curve, "region")
    supply_price_ids = _unique_int_values(supply_price, "region")

    assert cap_ids == supply_curve_ids == supply_price_ids
