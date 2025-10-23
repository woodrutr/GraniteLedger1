#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import logging
from pathlib import Path
import sys

import pandas as pd


logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger("audit.carbon")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=str, default="2025,2026,2027")
    parser.add_argument("--carbon-price", type=float, default=45.0)
    parser.add_argument("--escalator_pct", type=float, default=4.0)
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()

    years = [int(x) for x in args.years.split(",") if x]
    frames_mod = importlib.import_module("tests.fixtures.dispatch_single_minimal")
    run_loop = importlib.import_module("engine.run_loop")

    frames = frames_mod.three_year_frames(years=years)

    schedule: dict[int, float] = {}
    price = float(args.carbon_price)
    for year in years:
        schedule[year] = price
        price = price * (1.0 + float(args.escalator_pct) / 100.0)

    if hasattr(frames, "_meta"):
        frames._meta.setdefault("carbon_price_schedule", {str(k): v for k, v in schedule.items()})
        frames._meta["carbon_price_default"] = float(args.carbon_price)
    if hasattr(frames, "deep_carbon_pricing_enabled"):
        frames.deep_carbon_pricing_enabled = bool(args.deep)
    else:
        setattr(frames, "deep_carbon_pricing_enabled", bool(args.deep))

    outputs = run_loop.run_end_to_end_from_frames(
        frames,
        years=years,
        price_initial=0.0,
        tol=1e-4,
        relaxation=0.8,
        carbon_price_schedule=schedule,
        deep_carbon_pricing=bool(args.deep),
    )

    annual = outputs.annual.copy()
    cols = [
        column
        for column in [
            "year",
            "cp_last",
            "allowance_price",
            "cp_all",
            "cp_exempt",
            "cp_effective",
            "emissions_tons",
        ]
        if column in annual.columns
    ]
    LOGGER.info("ANNUAL:\n%s", annual[cols])

    df = annual.set_index("year")
    for year in years:
        exogenous = schedule[year]
        if "cp_exempt" in df.columns:
            assert abs(df.loc[year, "cp_exempt"] - exogenous) < 1e-3, f"cp_exempt mismatch y={year}"
        if {"cp_all", "cp_effective"}.issubset(df.columns):
            allowance_value = df.loc[year, "cp_all"]
            effective_value = df.loc[year, "cp_effective"]
            if args.deep:
                assert abs(effective_value - (allowance_value + exogenous)) < 1e-3, (
                    f"deep effective mismatch y={year}"
                )
            else:
                assert abs(effective_value - max(allowance_value, exogenous)) < 1e-3, (
                    f"shallow effective mismatch y={year}"
                )

    LOGGER.info("OK: invariants checked for years %s", years)


if __name__ == "__main__":
    main()
