"""Tests for engine.constants helpers."""

from engine import constants


def test_discover_input_root_prefers_inputs_when_only_variant(tmp_path):
    """Helper should locate an ``inputs`` directory when ``input`` is absent."""

    candidate = tmp_path / "inputs"
    candidate.mkdir()

    result = constants._discover_input_root(tmp_path)

    assert result == candidate


def test_discover_input_root_prefers_input_when_both_exist(tmp_path):
    """``input`` takes precedence when both ``input`` and ``inputs`` exist."""

    input_dir = tmp_path / "input"
    inputs_dir = tmp_path / "inputs"
    input_dir.mkdir()
    inputs_dir.mkdir()

    result = constants._discover_input_root(tmp_path)

    assert result == input_dir


def test_discover_input_root_falls_back_to_input_path(tmp_path):
    """Missing directories default to the canonical ``input`` path."""

    expected = tmp_path / "input"

    result = constants._discover_input_root(tmp_path)

    assert result == expected
