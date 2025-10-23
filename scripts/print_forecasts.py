"""Utility script to print discovered load forecast scenarios."""

from __future__ import annotations

from engine.data_loaders.load_forecasts import available_iso_scenarios
from engine.settings import input_root


def main() -> None:
    root = input_root()
    manifests = available_iso_scenarios(base_path=root)
    print(f"Found {len(manifests)} scenario manifests")
    for manifest in manifests:
        print(manifest.iso, "|", manifest.scenario, "| zone:", manifest.zone)


if __name__ == "__main__":
    main()
