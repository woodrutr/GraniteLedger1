from pathlib import Path


def test_only_root_requirements_file() -> None:
    """Ensure the repository keeps a single pip requirements manifest."""
    repo_root = Path(__file__).resolve().parents[1]
    requirements = {
        path.resolve()
        for path in repo_root.rglob('requirements*.txt')
        if path.name.lower().startswith('requirements')
    }
    expected = {repo_root / 'requirements.txt'}

    assert requirements == expected, (
        'Unexpected requirements manifests detected: '
        f"{sorted(str(path.relative_to(repo_root)) for path in requirements - expected)}"
    )
