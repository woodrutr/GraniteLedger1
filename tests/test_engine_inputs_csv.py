import pandas as pd
from engine.data_paths import data_root
from engine.validation.validators import assert_edges_valid, assert_units_valid

def canonical_zones():
    root = data_root()
    zones = set()
    p_units = root / "units" / "us_units.csv"
    p_edges = root / "transmission" / "ei_edges.csv"
    if p_units.exists():
        zones |= set(pd.read_csv(p_units)["zone"].dropna().unique())
    if p_edges.exists():
        e = pd.read_csv(p_edges)
        if "from_region" in e.columns:
            zones |= set(e["from_region"].dropna().unique())
        if "to_region" in e.columns:
            zones |= set(e["to_region"].dropna().unique())
    return zones

def test_edges_csv_ok():
    root = data_root()
    df = pd.read_csv(root / "transmission" / "ei_edges.csv")
    assert_edges_valid(df, canonical_zones())

def test_units_csv_ok():
    root = data_root()
    df = pd.read_csv(root / "units" / "us_units.csv")
    assert_units_valid(df, canonical_zones())
