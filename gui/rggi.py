from typing import Any

try:  # pragma: no cover - optional import for GUI metadata
    from gui.region_metadata import canonical_region_label
except ModuleNotFoundError:  # pragma: no cover - fallback for packaging
    try:
        from region_metadata import canonical_region_label  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - minimal fallback
        def canonical_region_label(value: Any) -> str:  # type: ignore[no-redef]
            return str(value)

# -------------------------
# RGGI Defaults (2025 Model Rule Forward)
# -------------------------

RGGI_BUDGETS = {
    2024: 69_401_609,
    2025: 66_586_609,
    2026: 78_532_784,
    2027: 69_806_919,
    2028: 61_081_054,
    2029: 52_355_189,
    2030: 43_629_324,
    2031: 34_903_459,
    2032: 26_177_594,
    2033: 17_451_729,
    2034: 14_958_625,
    2035: 12_465_521,
    2036: 9_972_417,
    2037: 7_479_313,
    2038: 7_479_313,
    2039: 7_479_313,
    2040: 7_479_313,
    2041: 7_479_313,
    2042: 7_479_313,
    2043: 7_479_313,
    2044: 7_479_313,
    2045: 7_479_313,
    2046: 7_479_313,
    2047: 7_479_313,
    2048: 7_479_313,
    2049: 7_479_313,
    2050: 7_479_313,
}

RGGI_MIN_RESERVE_PRICE = {
    2024: 2.56,
    2025: 2.62,
    2026: 2.69,
    2027: 9.00,
    2028: 9.63,
    2029: 10.30,
    2030: 11.02,
    2031: 11.79,
    2032: 12.62,
    2033: 13.50,
    2034: 14.45,
    2035: 15.46,
    2036: 16.54,
    2037: 17.70,
    2038: 18.94,
    2039: 20.27,
    2040: 21.69,
    2041: 23.21,
    2042: 24.83,
    2043: 26.57,
    2044: 28.43,
    2045: 30.42,
    2046: 32.55,
    2047: 34.83,
    2048: 37.27,
    2049: 39.88,
    2050: 42.67,
}

RGGI_ECR_TRIGGER = {2024: 7.35, 2025: 7.86, 2026: 8.41}
RGGI_ECR_WITHHOLD = {2024: 7_807_031, 2025: 7_545_597, 2026: 7_284_162}

RGGI_CCR1 = {
    "amount": {
        2024: 6_940_161,
        2025: 6_658_661,
        2026: 7_853_278,
        **{year: 11_746_358 for year in range(2027, 2051)},
    },
    "trigger": {
        2024: 15.92,
        2025: 17.03,
        2026: 18.22,
        2027: 19.50,
        2028: 20.87,
        2029: 22.33,
        2030: 23.89,
        2031: 25.56,
        2032: 27.35,
        2033: 29.26,
        2034: 31.31,
        2035: 33.50,
        2036: 35.85,
        2037: 38.36,
        2038: 41.05,
        2039: 43.92,
        2040: 46.99,
        2041: 50.28,
        2042: 53.80,
        2043: 57.57,
        2044: 61.60,
        2045: 65.91,
        2046: 70.52,
        2047: 75.46,
        2048: 80.74,
        2049: 86.39,
        2050: 92.44,
    },
}

RGGI_CCR2 = {
    "amount": {year: 11_746_358 for year in range(2027, 2051)},
    "trigger": {
        2027: 29.25,
        2028: 31.30,
        2029: 33.49,
        2030: 35.83,
        2031: 38.34,
        2032: 41.02,
        2033: 43.89,
        2034: 46.96,
        2035: 50.25,
        2036: 53.77,
        2037: 57.53,
        2038: 61.56,
        2039: 65.87,
        2040: 70.48,
        2041: 75.41,
        2042: 80.69,
        2043: 86.34,
        2044: 92.38,
        2045: 98.85,
        2046: 105.77,
        2047: 113.17,
        2048: 121.09,
        2049: 129.57,
        2050: 138.64,
    },
}

RGGI_INITIAL_BANK = 76_000_000
RGGI_REGIONS = [
    "Connecticut", "Delaware", "Maine", "Maryland", "Massachusetts",
    "New Hampshire", "New Jersey", "New York", "Rhode Island", "Vermont",
]

RGGI_REGION_IDS = (7, 8, 9, 10, 13)


def apply_rggi_defaults(modules: dict[str, Any]) -> None:
    """Apply RGGI 2025 Model Rule defaults to module configuration."""
    carbon_module = modules.setdefault("carbon_policy", {})
    coverage_labels = [canonical_region_label(region_id) for region_id in RGGI_REGION_IDS]

    carbon_module.update(
        {
            "enabled": True,
            "enable_floor": True,
            "enable_ccr": True,
            "ccr1_enabled": True,
            "ccr2_enabled": True,
            "allowance_banking_enabled": True,
            "coverage_regions": coverage_labels,
            "control_period_years": 3,
            "bank0": float(RGGI_INITIAL_BANK),
            "rggi_budgets": dict(RGGI_BUDGETS),
            "reserve_price": dict(RGGI_MIN_RESERVE_PRICE),
            "ecr_trigger": dict(RGGI_ECR_TRIGGER),
            "ecr_withhold": dict(RGGI_ECR_WITHHOLD),
            "ccr1_amount": dict(RGGI_CCR1["amount"]),
            "ccr1_trigger": dict(RGGI_CCR1["trigger"]),
            "ccr2_amount": dict(RGGI_CCR2["amount"]),
            "ccr2_trigger": dict(RGGI_CCR2["trigger"]),
        }
    )
    # Explicitly disable carbon price module
    price_module = modules.setdefault("carbon_price", {})
    price_module.update({"enabled": False})
