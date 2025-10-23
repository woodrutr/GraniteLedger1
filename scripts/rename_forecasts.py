"""Utility to normalize load forecast folder and file names for GraniteLedger.

Usage:
    python scripts/rename_forecasts.py

This walks the directory returned by :func:`engine.settings.input_root` and
renames folders/files to match the conventions expected by
``engine.data_loaders.load_forecasts``:

  - Folder names: ``<source>_<vintage>_<scenario>`` (lowercase/underscores)
  - File names:  ``<iso>_<zone>_<scenario>.csv`` (lowercase/underscores)

Set the ``GRANITELEDGER_INPUT_ROOT`` environment variable to override the
default discovery behaviour.
"""

from __future__ import annotations

import re
from pathlib import Path

from engine.normalization import normalize_iso_name, normalize_token
from engine.settings import input_root


def rename_forecast_tree(root: Path) -> None:
    if not root.exists():
        print(f"Forecast root {root} missing")
        return

    for iso_dir in sorted([entry for entry in root.iterdir() if entry.is_dir()]):
        if not iso_dir.is_dir():
            continue
        canonical_iso = normalize_iso_name(iso_dir.name) or normalize_token(iso_dir.name)
        target_iso_dir = iso_dir.parent / canonical_iso
        working_iso_dir = iso_dir
        if target_iso_dir != iso_dir:
            if target_iso_dir.exists():
                print(f"Merging {iso_dir} into existing {target_iso_dir}")
                for child in list(iso_dir.iterdir()):
                    destination = target_iso_dir / child.name
                    if destination.exists():
                        print(f"  Skipping {child} -> {destination} (already exists)")
                        continue
                    print(f"  Moving {child} -> {destination}")
                    child.rename(destination)
                try:
                    iso_dir.rmdir()
                except OSError:
                    print(f"  Unable to remove {iso_dir}; directory not empty")
                working_iso_dir = target_iso_dir
            else:
                print(f"Renaming ISO folder {iso_dir} -> {target_iso_dir}")
                iso_dir.rename(target_iso_dir)
                working_iso_dir = target_iso_dir
        iso = canonical_iso

        for bundle_dir in list(working_iso_dir.iterdir()):
            if not bundle_dir.is_dir():
                continue

            # Normalize folder name like "CELT_2025_Baseline"
            tokens = re.split(r"[_\-\s]+", bundle_dir.name)
            tokens = [normalize_token(t) for t in tokens if t]
            folder_name = "_".join(tokens)
            new_bundle_dir = bundle_dir.parent / folder_name
            if new_bundle_dir != bundle_dir:
                print(f"Renaming folder {bundle_dir} -> {new_bundle_dir}")
                bundle_dir.rename(new_bundle_dir)
                bundle_dir = new_bundle_dir

            # Determine scenario (last token)
            scenario = tokens[-1] if tokens else "baseline"
            scenario_parts = [part for part in scenario.split("_") if part]

            for csv_path in list(bundle_dir.glob("*.csv")):
                stem = normalize_token(csv_path.stem)
                parts = [part for part in stem.split("_") if part]
                if scenario_parts and len(parts) >= len(scenario_parts):
                    if parts[-len(scenario_parts) :] == scenario_parts:
                        parts = parts[: -len(scenario_parts)]
                while parts and normalize_iso_name(parts[0]) == iso:
                    parts.pop(0)
                zone = "_".join(parts).strip("_") or "system"
                new_stem = f"{iso}_{zone}_{scenario}"
                new_name = csv_path.with_name(new_stem + ".csv")
                if new_name != csv_path:
                    print(f"Renaming {csv_path.name} -> {new_name.name}")
                    csv_path.rename(new_name)


if __name__ == "__main__":
    root = input_root()
    print(f"Renaming forecast tree at {root}")
    rename_forecast_tree(root)
    print("Rename complete.")
