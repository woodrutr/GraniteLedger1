"""Tests for region selection normalization behavior in the GUI."""

from gui.app import (
    available_regions,
    _normalize_coverage_selection,
    _normalize_region_labels,
    _regions_from_config,
)
from gui.region_metadata import canonical_region_value, region_display_label
from regions.registry import REGIONS


def test_normalize_removes_individuals_when_all_selected_after_individuals():
    previous = ('Northeast',)
    selection = ['Northeast', 'All']

    assert _normalize_region_labels(selection, previous) == ['All']


def test_normalize_removes_all_when_switching_from_all_to_individual():
    previous = ('All',)
    selection = ['All', 'Northeast']

    assert _normalize_region_labels(selection, previous) == ['Northeast']


def test_normalize_preserves_multiple_individuals_when_all_was_selected():
    previous = ('All',)
    selection = ['All', 'Northeast', 'Southwest']

    assert _normalize_region_labels(selection, previous) == ['Northeast', 'Southwest']


def test_normalize_handles_missing_previous_state():
    selection = ['All', 'Northeast']

    assert _normalize_region_labels(selection, None) == ['All']


def test_normalize_leaves_all_only_selection_unchanged():
    previous = ('All',)
    selection = ['All']

    assert _normalize_region_labels(selection, previous) == ['All']


def test_canonical_region_value_accepts_alias():
    assert canonical_region_value('Region 1') == 'ISO-NE_CT'
    assert canonical_region_value(1) == 'ISO-NE_CT'
    assert canonical_region_value('Region 2 – legacy FRCC label') == 'FRCC_SYS'


def test_canonical_region_value_handles_zone_tokens():
    assert canonical_region_value('ct') == 'ISO-NE_CT'
    assert canonical_region_value('ISO_NE_CT') == 'ISO-NE_CT'
    assert canonical_region_value('isone_ct') == 'ISO-NE_CT'


def test_region_display_label_contains_code_and_iso():
    label = region_display_label('NYISO_J')
    assert 'New York City' in label
    assert 'NYISO_J' in label


def test_available_regions_uses_helper_display_names():
    options = list(available_regions())
    names = {entry['name'] for entry in options}
    ids = {entry['id'] for entry in options}
    assert 'Region 1' not in names
    assert 'Connecticut' in names
    assert ids == set(REGIONS)


def test_normalize_coverage_selection_skips_invalid_entries():
    assert _normalize_coverage_selection(['Unknown']) == ['All']
    assert _normalize_coverage_selection(['Unknown', 'PJM_DOM']) == ['PJM_DOM']


def test_regions_from_config_handles_sequences_and_aliases():
    config = {"regions": [1, "PJM_DOM", "ISO-NE_CT", "PJM_DOM"]}

    assert _regions_from_config(config) == [
        'ISO-NE_CT',
        'PJM_DOM',
    ]


def test_regions_from_config_handles_mapping_weights():
    config = {
        "regions": {
            "ISO-NE_CT": 0,
            "PJM_DOM": 0.5,
            "NYISO_J": True,
            "PJM_ATSI": "0",
        }
    }

    assert _regions_from_config(config) == ['PJM_DOM', 'NYISO_J']


def test_regions_from_config_handles_nested_records():
    config = {
        "regions": [
            {"id": "PJM_AEP"},
            {"region": "NYISO_J"},
            {"label": "Connecticut"},
        ]
    }

    assert _regions_from_config(config) == ['PJM_AEP', 'NYISO_J', 'ISO-NE_CT']
