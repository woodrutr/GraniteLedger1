import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.regions_schema import REGION_RECORDS


@dataclass(frozen=True)
class MatchResult:
    path: Path
    region_id: str
    target: Path


def _normalized_tokens_for_region(region_id: str, label: str) -> set[str]:
    """Return a collection of tokens associated with ``region_id``."""

    base_tokens = {region_id}
    if "_" in region_id:
        base_tokens.add(region_id.split("_", 1)[1])
    base_tokens.update(part for part in region_id.split("_") if part)

    label_variants = {label, label.replace("&", "and")}
    for variant in label_variants:
        sanitized = re.sub(r"[^a-z0-9]+", " ", variant.lower()).strip()
        if sanitized:
            base_tokens.add(sanitized)
            base_tokens.add(sanitized.replace(" ", ""))
            base_tokens.update(sanitized.split())

    tokens: set[str] = set()
    for token in base_tokens:
        normalized = re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_")
        if not normalized:
            continue
        tokens.add(normalized)
        compact = normalized.replace("_", "")
        if compact:
            tokens.add(compact)
    return tokens


def build_zone_map() -> Mapping[str, str]:
    """Return a mapping from normalized zone tokens to region identifiers."""

    zone_to_region: dict[str, str] = {}
    for record in REGION_RECORDS:
        region_id = str(record["id"])
        label = str(record.get("name", ""))
        for token in _normalized_tokens_for_region(region_id, label):
            if token in zone_to_region and zone_to_region[token] != region_id:
                raise ValueError(
                    f"Conflicting region IDs for token '{token}': "
                    f"{zone_to_region[token]} vs {region_id}"
                )
            zone_to_region.setdefault(token, region_id)
    return zone_to_region


def candidate_tokens(zone: str, region_id: str) -> set[str]:
    tokens = {zone}
    rid_lower = region_id.lower()
    tokens.add(rid_lower)
    zone_compact = zone.replace("_", "")
    rid_compact = rid_lower.replace("_", "")
    tokens.add(zone_compact)
    tokens.add(rid_compact)
    if zone.endswith("k"):
        tokens.add(f"{zone}e")
    return {token for token in tokens if token}


def find_region_match(filename: str, zone_map: Mapping[str, str]) -> set[str]:
    lowercase = filename.lower()
    name_without_ext = lowercase.rsplit(".", 1)[0]
    word_tokens = set(re.split(r"[^a-z0-9]+", name_without_ext))
    word_tokens.discard("")

    def _token_matches(token: str) -> bool:
        if "_" in token:
            return token in name_without_ext
        if len(token) <= 1:
            return False
        if token in word_tokens:
            return True
        if len(token) >= 3:
            for part in word_tokens:
                if part.startswith(token) and len(part) - len(token) <= 2:
                    return True
        return False

    matches: set[str] = set()
    for zone, region_id in zone_map.items():
        tokens = candidate_tokens(zone, region_id)
        if any(_token_matches(token) for token in tokens):
            matches.add(region_id)
    return matches


def collect_renames(root: Path, zone_map: Mapping[str, str]) -> list[MatchResult]:
    matches: list[MatchResult] = []
    for path in sorted(root.rglob("*.csv")):
        if not path.is_file():
            continue
        region_matches = find_region_match(path.name, zone_map)
        if not region_matches:
            print(f"WARNING: No region match for {path}")
            continue
        if len(region_matches) > 1:
            matches_str = ", ".join(sorted(region_matches))
            print(f"WARNING: Multiple region matches for {path}: {matches_str}")
            continue
        (region_id,) = tuple(region_matches)
        target = path.with_name(f"{region_id}.csv")
        if target == path:
            continue
        if target.exists():
            print(f"WARNING: Target already exists for {path} -> {target}")
            continue
        matches.append(MatchResult(path=path, region_id=region_id, target=target))
    return matches


def rename_files(matches: Iterable[MatchResult], *, dry_run: bool) -> None:
    for match in matches:
        if dry_run:
            print(f"DRY RUN: {match.path} -> {match.target}")
        else:
            print(f"RENAME: {match.path} -> {match.target}")
            match.path.rename(match.target)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        help="Path to the root load_forecasts directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview renames without making filesystem changes",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    root = args.directory.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 1

    zone_map = build_zone_map()
    matches = collect_renames(root, zone_map)
    if not matches:
        print("No files to rename.")
        return 0

    rename_files(matches, dry_run=args.dry_run)
    return 0


+if __name__ == "__main__":
+    raise SystemExit(main())
