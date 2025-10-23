"""Metadata and helpers for mapping region identifiers to human-friendly labels."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping

from common.regions_schema import LEGACY_REGION_PRIORITY, REGION_MAP, REGION_RECORDS
from regions.registry import REGIONS
from engine.normalization import normalize_iso_name, normalize_region_id, normalize_token

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegionMetadata:
    """Static information describing one of the canonical model regions."""

    code: str
    label: str
    area: str = ""
    aliases: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        """Return the canonical identifier for compatibility with legacy callers."""
        return self.code


def _build_legacy_aliases(limit: int = 25) -> dict[str, str]:
    """Return the legacy numeric alias mapping used by historic GUIs."""

    ordered_ids: list[str] = []

    def _canonical_id(candidate: str) -> str | None:
        if candidate in REGION_MAP:
            return candidate
        sys_variant = f"{candidate}_SYS"
        if sys_variant in REGION_MAP:
            return sys_variant
        return None

    for priority_id in LEGACY_REGION_PRIORITY:
        canonical = _canonical_id(priority_id) or priority_id
        if canonical not in ordered_ids:
            ordered_ids.append(canonical)

    for record in REGION_RECORDS:
        region_id = record["id"]
        if region_id not in ordered_ids:
            ordered_ids.append(region_id)

    aliases: dict[str, str] = {}
    for index, region_id in enumerate(ordered_ids, start=1):
        if index > limit:
            break
        label = f"Region {index}"
        aliases[label] = region_id
        aliases[str(index)] = region_id
    return aliases


LEGACY_ALIASES: Mapping[str, str] = _build_legacy_aliases()


def _build_default_metadata() -> Mapping[str, RegionMetadata]:
    """Construct metadata for all canonical regions, including legacy aliases."""

    metadata: dict[str, RegionMetadata] = {}
    alias_buckets: dict[str, list[str]] = {}

    # Bucket aliases by region code
    for alias, region_id in LEGACY_ALIASES.items():
        alias_buckets.setdefault(region_id, []).append(alias)

    # Build metadata from the registry
    for code, label in REGIONS.items():
        metadata[code] = RegionMetadata(
            code=code,
            label=label,
            aliases=tuple(alias_buckets.get(code, ())),
        )

    return metadata


DEFAULT_REGION_METADATA: Mapping[str, RegionMetadata] = _build_default_metadata()


def _normalize_alias_tokens(value: str) -> set[str]:
    """Normalize freeform alias text into canonical tokens for lookup."""
    tokens: set[str] = set()
    text = value.strip()
    if not text:
        return tokens

    normalized_text = text.replace("–", "-").replace("—", "-")
    raw_candidates = {text, normalized_text}
    candidates: set[str] = set()
    for candidate in raw_candidates:
        candidates.update(
            {
                candidate,
                candidate.replace("_", " "),
                candidate.replace("-", " "),
                candidate.replace("_", "-"),
                candidate.replace("-", "_"),
                candidate.replace("_", ""),
                candidate.replace("-", ""),
                candidate.replace("-", " ").replace("_", " "),
            }
        )

    for candidate in candidates:
        lowered = candidate.strip().lower()
        if lowered:
            tokens.add(lowered)
    return tokens


def _build_alias_map() -> dict[str, str]:
    """Build lookup from normalized tokens to canonical region codes."""
    alias_map: dict[str, str] = {}

    for code, metadata in DEFAULT_REGION_METADATA.items():
        # Map the code directly
        alias_map.setdefault(code.lower(), code)
        alias_map.setdefault(normalize_token(code), code)

        normalized_code = normalize_region_id(code)
        if normalized_code and normalized_code != code:
            alias_map.setdefault(normalize_token(normalized_code), code)
        if normalized_code and "_" in normalized_code:
            iso_token, zone_token = normalized_code.split("_", 1)

            alias_map.setdefault(zone_token.lower(), code)
            alias_map.setdefault(normalize_token(zone_token), code)

            iso_variants = {normalize_token(iso_token)}
            iso_variants.update(_normalize_alias_tokens(iso_token))
            zone_variants = {normalize_token(zone_token)}
            zone_variants.update(_normalize_alias_tokens(zone_token))

            for iso_variant in iso_variants:
                if not iso_variant:
                    continue
                for zone_variant in zone_variants:
                    if not zone_variant:
                        continue
                    alias_map.setdefault(f"{iso_variant}_{zone_variant}", code)
                    alias_map.setdefault(f"{iso_variant}{zone_variant}", code)

        # Add label aliases
        for token in _normalize_alias_tokens(metadata.label):
            alias_map.setdefault(token, code)
            alias_map.setdefault(normalize_token(token), code)

        # Add area tokens
        if metadata.area:
            for token in _normalize_alias_tokens(metadata.area):
                alias_map.setdefault(token, code)

        # Add extra aliases
        for alias in metadata.aliases:
            for token in _normalize_alias_tokens(alias):
                alias_map.setdefault(token, code)

        # Split code into parts and build composite aliases
        code_parts = [part for part in re.split(r"[_-]", metadata.code) if part]
        if code_parts:
            alias_map.setdefault(" ".join(code_parts).lower(), code)
            if metadata.label:
                alias_map.setdefault(
                    f"{code_parts[0]} {metadata.label}".lower(),
                    code,
                )

    return alias_map



_REGION_ALIAS_MAP = _build_alias_map()


def region_alias_map() -> dict[str, int]:
    """Return a copy of the alias map linking strings to canonical region IDs."""

    return dict(_REGION_ALIAS_MAP)


def canonical_region_value(value: Any) -> int | str:
    """Resolve ``value`` into a canonical region identifier when possible."""

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        candidate = int(value)
        if candidate in DEFAULT_REGION_METADATA:
            return candidate
        alias = LEGACY_ALIASES.get(f"Region {candidate}") or LEGACY_ALIASES.get(str(candidate))
        if alias is not None:
            return alias
        return candidate

    text = str(value).strip()
    if not text:
        return text

    lowered = text.lower()
    if lowered in {"all", "all regions"}:
        return "All"

    legacy_match = re.match(r"region\s*(\d+)", lowered)
    if legacy_match:
        alias = LEGACY_ALIASES.get(f"Region {legacy_match.group(1)}")
        if alias is not None:
            return alias

    for candidate in (lowered, normalize_token(text)):
        if not candidate:
            continue
        match = _REGION_ALIAS_MAP.get(candidate)
        if match is not None:
            return match

    sanitized_text = lowered.replace("-", "_").replace(" ", "_")
    if "_" in sanitized_text:
        iso_part, zone_part = sanitized_text.split("_", 1)
        iso_token = normalize_iso_name(iso_part)
        zone_token = zone_part.strip("_")
        if iso_token and zone_token:
            zone_candidates = {zone_token, normalize_token(zone_token)}
            for zone_candidate in zone_candidates:
                if not zone_candidate:
                    continue
                for composite in (f"{iso_token}_{zone_candidate}", f"{iso_token}{zone_candidate}"):
                    match = _REGION_ALIAS_MAP.get(composite)
                    if match is not None:
                        return match

    try:
        canonical_region = normalize_region_id(text)
    except ValueError:
        canonical_region = text
    if canonical_region in DEFAULT_REGION_METADATA:
        return canonical_region
    normalized_canonical = normalize_token(canonical_region)
    if normalized_canonical:
        match = _REGION_ALIAS_MAP.get(normalized_canonical)
        if match is not None:
            return match
    if canonical_region and canonical_region != text:
        return canonical_region

    return text


def region_metadata(region_id: int | str) -> RegionMetadata | None:
    """Return the metadata entry for ``region_id`` if it exists."""

    if isinstance(region_id, int) and region_id in DEFAULT_REGION_METADATA:
        return DEFAULT_REGION_METADATA[region_id]

    resolved = canonical_region_value(region_id)
    if isinstance(resolved, int):
        return DEFAULT_REGION_METADATA.get(resolved)
    if isinstance(resolved, str):
        try:
            numeric = int(resolved)
        except (TypeError, ValueError):
            return None
        return DEFAULT_REGION_METADATA.get(numeric)
    return None


def region_display_label(value: int | str) -> str:
    """Return a human-readable label for ``value`` suitable for GUI display."""

    resolved = canonical_region_value(value)

    # Case 1: legacy integer-based region ids
    if isinstance(resolved, int):
        metadata = DEFAULT_REGION_METADATA.get(resolved)
        if metadata is not None:
            return metadata.label
        return str(resolved)

    # Case 2: string-based region codes
    if isinstance(resolved, str):
        meta = DEFAULT_REGION_METADATA.get(resolved)
        if meta is not None:
            label = meta.label or resolved
            return f"{label} ({meta.code})"

        if resolved.lower() == "all":
            return "All regions"

        try:
            numeric = int(resolved)
        except (TypeError, ValueError):
            return resolved

        metadata = DEFAULT_REGION_METADATA.get(numeric)
        if metadata is not None:
            return metadata.label
        return resolved

    # Fallback
    return str(resolved)



def canonical_region_label(value: Any) -> str:
    """Return the preferred label for ``value`` after canonicalization."""

    resolved = canonical_region_value(value)
    return region_display_label(resolved)


__all__ = [
    "DEFAULT_REGION_METADATA",
    "LEGACY_ALIASES",
    "RegionMetadata",
    "canonical_region_label",
    "canonical_region_value",
    "region_alias_map",
    "region_display_label",
    "region_metadata",
]
