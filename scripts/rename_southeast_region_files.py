#!/usr/bin/env python3
"""Rename Southeast load forecast CSV files to new canonical region identifiers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "input" / "electricity" / "load_forecasts" / "southeast"

RENAMES: dict[str, str] = {
    "southeast_frcc.csv": "FRCC_SYS.csv",
    "southeast_santee.csv": "SANTEE_COOPER_SYS.csv",
}


def iter_matches(root: Path) -> Iterable[tuple[Path, Path]]:
    """Yield ``(source, target)`` pairs for files that should be renamed."""

    for source_name, target_name in RENAMES.items():
        for source in sorted(root.rglob(source_name)):
            target = source.with_name(target_name)
            yield source, target


def rename_files(pairs: Iterable[tuple[Path, Path]], *, dry_run: bool) -> None:
    for source, target in pairs:
        if source == target:
            continue
        if target.exists():
            print(f"SKIP: {target} already exists; not renaming {source}")
            continue
        if dry_run:
            print(f"DRY RUN: {source} -> {target}")
        else:
            print(f"RENAME: {source} -> {target}")
            source.rename(target)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=DEFAULT_ROOT,
        type=Path,
        help="Root directory containing Southeast load forecast data",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview rename operations without modifying files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"ERROR: {root} is not a directory")
        return 1

    pairs = list(iter_matches(root))
    if not pairs:
        print("No files matched the rename patterns.")
        return 0

    rename_files(pairs, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
