"""Metadata helpers for electricity technologies."""

from __future__ import annotations

TECH_ID_TO_LABEL: dict[int, str] = {
    1: 'Coal Steam',
    2: 'Oil Steam',
    3: 'Natural Gas Combustion Turbine',
    4: 'Natural Gas Combined-Cycle',
    5: 'Hydrogen Turbine',
    6: 'Nuclear',
    7: 'Biomass',
    8: 'Geothermal',
    9: 'Municipal Solid Waste',
    10: 'Hydroelectric Generation',
    11: 'Pumped Hydroelectric Storage',
    12: 'Battery Energy Storage',
    13: 'Wind Offshore',
    14: 'Wind Onshore',
    15: 'Solar',
}

_LABEL_TO_TECH_ID: dict[str, int] = {
    label.lower(): tech_id for tech_id, label in TECH_ID_TO_LABEL.items()
}


def get_technology_label(tech_id: int | str) -> str:
    """Return the human-readable label for a technology identifier."""

    try:
        tech_int = int(tech_id)
    except (TypeError, ValueError):
        tech_int = None

    if tech_int is None:
        return f'Tech {tech_id}'

    return TECH_ID_TO_LABEL.get(tech_int, f'Tech {tech_int}')


def resolve_technology_key(key: object) -> int | None:
    """Resolve a configuration key to a known technology identifier."""

    if key in (None, ''):
        return None

    try:
        return int(key)
    except (TypeError, ValueError):
        normalized = str(key).strip().lower()
        if not normalized:
            return None
        return _LABEL_TO_TECH_ID.get(normalized)
