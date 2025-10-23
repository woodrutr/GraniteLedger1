"""Utility helpers for caching tabular data with file manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

ManifestEntry = dict[str, int | str]
MANIFEST_VERSION = "load_forecasts_strict.v1"
PARQUET_ENGINE = "pyarrow"


def _manifest_path(parquet_path: Path) -> Path:
    suffix = parquet_path.suffix or ".parquet"
    return parquet_path.with_suffix(f"{suffix}.manifest.json")


def build_manifest(paths: Iterable[Path]) -> list[ManifestEntry]:
    """Return a manifest describing the ``paths`` used to build a table."""

    manifest: list[ManifestEntry] = [
        {"version": MANIFEST_VERSION, "engine": PARQUET_ENGINE}
    ]
    for path in sorted({Path(p).resolve() for p in paths}, key=str):
        stat = path.stat()
        manifest.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return manifest


def load(parquet_path: Path, manifest: Sequence[Mapping[str, object]]) -> pd.DataFrame | None:
    """Load ``parquet_path`` when the on-disk manifest matches ``manifest``."""

    manifest_path = _manifest_path(parquet_path)
    if not parquet_path.exists() or not manifest_path.exists():
        return None

    try:
        disk_manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None

    canonical_expected = json.loads(json.dumps(list(manifest), sort_keys=True))
    if canonical_expected != disk_manifest:
        return None

    return pd.read_parquet(parquet_path, engine=PARQUET_ENGINE)


def write(parquet_path: Path, frame: pd.DataFrame, manifest: Sequence[Mapping[str, object]]) -> None:
    """Persist ``frame`` and ``manifest`` to disk."""

    parquet_path = parquet_path.resolve()
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False, engine=PARQUET_ENGINE)

    manifest_path = _manifest_path(parquet_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(list(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")


__all__ = ["MANIFEST_VERSION", "PARQUET_ENGINE", "build_manifest", "load", "write"]
