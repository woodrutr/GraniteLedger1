import logging
import sys
from pathlib import Path

try:
    from engine.data_loaders.load_forecasts import (
        available_iso_scenarios,
        load_demand_forecasts_selection,
    )
    from engine.settings import input_root
except ModuleNotFoundError:  # pragma: no cover - CLI convenience
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from engine.data_loaders.load_forecasts import (  # type: ignore[import]
        available_iso_scenarios,
        load_demand_forecasts_selection,
    )
    from engine.settings import input_root  # type: ignore[import]


logging.basicConfig(level=logging.INFO)


def main() -> None:
    root = input_root()
    print("INPUT_ROOT:", root)

    for iso in ["iso_ne", "nyiso", "pjm", "miso", "spp", "southeast", "canada"]:
        manifests = available_iso_scenarios(iso, base_path=root)
        counts: dict[str, int] = {}
        for manifest in manifests:
            counts[manifest.scenario] = counts.get(manifest.scenario, 0) + 1
        print(
            "SCENARIOS",
            iso,
            "->",
            {"total": len(manifests), "scenarios": counts},
        )

    print("SAMPLE LOAD SHAPE ROWS:")
    sample = load_demand_forecasts_selection(years=[2025, 2026], base_path=root)
    print(sample.head())


if __name__ == "__main__":
    main()
