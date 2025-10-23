"""Tests for default frame construction used by the GUI."""

def test_default_frames_trim_missing_unit_regions():
    # Import lazily to avoid importing Streamlit when unavailable.
    from gui.app import _build_default_frames

    frames = _build_default_frames(years=[2025])

    demand = frames.demand()
    units = frames.units()

    assert not demand.empty, "Default demand frame should not be empty"
    assert not units.empty, "Default units frame should not be empty"

    demand_regions = set(demand["region"].astype(str).unique())
    unit_regions = set(units["region"].astype(str).unique())

    # All demand regions must have corresponding unit coverage after trimming.
    assert demand_regions.issubset(unit_regions)

