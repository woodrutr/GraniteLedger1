import pytest

pytest.importorskip("yaml")

from engine.regions.shares import expand_state_regions


def test_expand_state_regions_filters_nj() -> None:
    regions = expand_state_regions("NJ")
    assert "PJM_PSEG" in regions
    assert all(not region.startswith("NYISO_") for region in regions)
