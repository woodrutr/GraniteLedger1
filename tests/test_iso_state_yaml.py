"""Tests for iso_state_zones.yaml loading via regions.registry."""

from importlib import reload

import regions.registry as registry


def test_registry_imports_without_error():
    # Importing the module should not raise errors and should populate STATE_INDEX.
    reloaded = reload(registry)
    assert hasattr(reloaded, "STATE_INDEX")
    assert isinstance(reloaded.STATE_INDEX, dict)


def test_state_index_contains_known_pjm_and_iso_ne_regions():
    # Ensure a PJM state entry includes one of its configured regions.
    md_regions = registry.STATE_INDEX.get("MD")
    assert md_regions is not None, "Expected Maryland PJM regions to be indexed"
    assert "PJM_BGE" in md_regions

    # Ensure an ISO-NE state entry includes one of its configured regions.
    ma_regions = registry.STATE_INDEX.get("MA")
    assert ma_regions is not None, "Expected Massachusetts ISO-NE regions to be indexed"
    assert "ISO-NE_SEMA" in ma_regions
