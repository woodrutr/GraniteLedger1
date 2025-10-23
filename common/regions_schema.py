"""Canonical region definitions shared between engine and UI packages."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Mapping

# Ordered records provide deterministic iteration for both engine and GUI layers.
REGION_RECORDS: tuple[dict[str, str], ...] = (
    {"id": "ISO-NE_CT", "name": "Connecticut"},
    {"id": "ISO-NE_ME", "name": "Maine"},
    {"id": "ISO-NE_NH", "name": "New Hampshire"},
    {"id": "ISO-NE_RI", "name": "Rhode Island"},
    {"id": "ISO-NE_VT", "name": "Vermont"},
    {"id": "ISO-NE_SEMA", "name": "SEMA/RI"},
    {"id": "ISO-NE_WCMA", "name": "WCMA"},
    {"id": "ISO-NE_NEMA", "name": "NEMA/Boston"},
    {"id": "NYISO_A", "name": "West"},
    {"id": "NYISO_B", "name": "Genesee"},
    {"id": "NYISO_C", "name": "Central"},
    {"id": "NYISO_D", "name": "North"},
    {"id": "NYISO_E", "name": "Mohawk Valley"},
    {"id": "NYISO_F", "name": "Capital"},
    {"id": "NYISO_G", "name": "Hudson Valley"},
    {"id": "NYISO_H", "name": "Millwood"},
    {"id": "NYISO_I", "name": "Dunwoodie"},
    {"id": "NYISO_J", "name": "New York City"},
    {"id": "NYISO_K", "name": "Long Island"},
    {"id": "PJM_AECO", "name": "Atlantic City Electric"},
    {"id": "PJM_APS", "name": "Allegheny Power"},
    {"id": "PJM_AEP", "name": "AEP"},
    {"id": "PJM_ATSI", "name": "FirstEnergy"},
    {"id": "PJM_BGE", "name": "Baltimore Gas & Electric"},
    {"id": "PJM_COMED", "name": "ComEd"},
    {"id": "PJM_DAYTON", "name": "Dayton"},
    {"id": "PJM_DEOK", "name": "Duke Energy Ohio/Kentucky"},
    {"id": "PJM_DOM", "name": "Dominion"},
    {"id": "PJM_DPL", "name": "Delmarva"},
    {"id": "PJM_DUQ", "name": "Duquesne"},
    {"id": "PJM_JCPL", "name": "Jersey Central"},
    {"id": "PJM_METED", "name": "Metropolitan Edison"},
    {"id": "PJM_PECO", "name": "PECO"},
    {"id": "PJM_PENELEC", "name": "Penelec"},
    {"id": "PJM_PEPCO", "name": "Pepco"},
    {"id": "PJM_PPL", "name": "PPL"},
    {"id": "PJM_PSEG", "name": "PSEG"},
    {"id": "PJM_RECO", "name": "Rockland"},
    {"id": "MISO_NORTH", "name": "MISO North/Central"},
    {"id": "MISO_SOUTH", "name": "MISO South"},
    {"id": "MISO_LRZ1", "name": "MISO Local Resource Zone 1"},
    {"id": "MISO_LRZ2", "name": "MISO Local Resource Zone 2"},
    {"id": "MISO_LRZ3", "name": "MISO Local Resource Zone 3"},
    {"id": "MISO_LRZ4", "name": "MISO Local Resource Zone 4"},
    {"id": "MISO_LRZ5", "name": "MISO Local Resource Zone 5"},
    {"id": "MISO_LRZ6", "name": "MISO Local Resource Zone 6"},
    {"id": "MISO_LRZ7", "name": "MISO Local Resource Zone 7"},
    {"id": "MISO_LRZ8", "name": "MISO Local Resource Zone 8"},
    {"id": "MISO_LRZ9", "name": "MISO Local Resource Zone 9"},
    {"id": "MISO_LRZ10", "name": "MISO Local Resource Zone 10"},
    {"id": "SOCO_SYS", "name": "Southern Company"},
    {"id": "TVA_SYS", "name": "Tennessee Valley Authority"},
    {"id": "DUK_SYS", "name": "Duke Energy Carolinas/Progress"},
    {"id": "SANTEE_COOPER_SYS", "name": "Santee Cooper"},
    {"id": "SCEG_SYS", "name": "Dominion Energy South Carolina (SCE&G)"},
    {"id": "ENTERGY_SYS", "name": "Entergy"},
    {"id": "FPL_SYS", "name": "Florida Power & Light"},
    {"id": "FPC_SYS", "name": "Duke Energy Florida (FPC)"},
    {"id": "TECO_SYS", "name": "Tampa Electric"},
    {"id": "JEA_SYS", "name": "JEA"},
    {"id": "FMPA_SYS", "name": "Florida Municipal Power Agency"},
    {"id": "FRCC_SYS", "name": "Florida Reliability Coordinating Council"},
    {"id": "SPP_MO", "name": "SPP Missouri"},
    {"id": "SPP_AR", "name": "SPP Arkansas"},
    {"id": "SPP_LA", "name": "SPP Louisiana"},
    {"id": "SPP_OK", "name": "SPP Oklahoma"},
    {"id": "ONTARIO", "name": "Ontario IESO"},
    {"id": "QUEBEC", "name": "Hydro-Québec"},
    {"id": "MARITIMES", "name": "Maritimes"},
    {"id": "CANADA_MARITIMES", "name": "Canada Maritimes"},
)


REGION_MAP: "OrderedDict[str, str]" = OrderedDict(
    (entry["id"], entry["name"]) for entry in REGION_RECORDS
)

# Ensure newly added Southeast system regions are exposed through the canonical
# mapping even if upstream ordering changes.
REGION_MAP.update(
    {
        # --- ADD OR VERIFY THESE LINES EXIST ---
        "FRCC_SYS": "Florida Reliability Coordinating Council",
        "SANTEE_COOPER_SYS": "Santee Cooper",
        # -------------------------------------
    }
)


# Display names used when normalizing ISO identifiers in discovery utilities.
ISO_DISPLAY_NAMES: Mapping[str, str] = {
    "iso_ne": "ISO-NE",
    "nyiso": "NYISO",
    "pjm": "PJM",
    "miso": "MISO",
    "spp": "SPP",
    "southeast": "Southeast",
    "canada": "Canada",
}


# Legacy ordering preference retained for compatibility with legacy GUIs.
LEGACY_REGION_PRIORITY: tuple[str, ...] = ("ISO-NE_CT", "FRCC")


def iter_region_records() -> Iterable[dict[str, str]]:
    """Yield canonical region records in configured order."""

    return (dict(entry) for entry in REGION_RECORDS)


__all__ = [
    "REGION_RECORDS",
    "REGION_MAP",
    "ISO_DISPLAY_NAMES",
    "LEGACY_REGION_PRIORITY",
    "iter_region_records",
]
