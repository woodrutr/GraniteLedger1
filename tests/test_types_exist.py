from pathlib import Path

from main.definitions import PROJECT_ROOT


def test_project_root_is_path() -> None:
    assert isinstance(PROJECT_ROOT, Path)


def test_project_root_points_to_repository() -> None:
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()
