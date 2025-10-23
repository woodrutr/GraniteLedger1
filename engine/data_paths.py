from pathlib import Path
from functools import lru_cache

@lru_cache
def data_root() -> Path:
    """Return <repo>/input/engine if it exists, else <cwd>/input/engine."""
    here = Path(__file__).resolve()
    for anc in (here, *here.parents):
        cand = anc / "input" / "engine"
        if cand.exists():
            return cand
    return (Path.cwd() / "input" / "engine").resolve()
