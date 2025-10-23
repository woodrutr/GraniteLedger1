"""Build authoritative zone-to-state share tables.

Implements the command line interface described in the task specification. The
script supports three operational modes:

* Inversion mode – normalise an uploaded zone/state share CSV.
* EIA mode – derive shares from EIA-861 sales/customers and utility mapping
  files.
* Placeholder mode – fall back to equal state splits using ISO configuration
  with optional overrides.

Outputs include a canonical CSV of shares, a JSON index mapping states to
regions, and optional QC reporting artifacts.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, getcontext
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import pandas as pd
try:  # pragma: no cover - optional dependency
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]


getcontext().prec = 28


class BuilderError(RuntimeError):
    """Base class for fatal builder failures."""

    strict_sensitive: bool = False


class StrictSensitiveError(BuilderError):
    """Error type that should honour ``--strict`` exit codes."""

    strict_sensitive = True


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build zone/state share tables")
    parser.add_argument("--iso-yaml", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--states-filter")
    parser.add_argument("--regions-filter")
    parser.add_argument("--in-csv", type=Path)
    parser.add_argument("--out-qc", type=Path)
    parser.add_argument("--summary", type=int, default=0)
    parser.add_argument("--eia861-retail", type=Path)
    parser.add_argument("--eia861-customers", type=Path)
    parser.add_argument("--eia861-year", type=int)
    parser.add_argument("--utility-map", type=Path)
    parser.add_argument("--utility-zone-override", type=Path)
    return parser.parse_args(argv)


@dataclass
class IsoUniverse:
    region_states: Dict[str, List[str]]
    state_regions: Dict[str, List[str]]
    isos: Dict[str, Mapping[str, Sequence[str]]]


def _parse_inline_list(value: str) -> List[str]:
    inner = value.strip()[1:-1].strip()
    if not inner:
        return []
    items: List[str] = []
    for segment in inner.split(";"):
        for piece in segment.split(","):
            candidate = piece.strip()
            if candidate:
                items.append(candidate)
    return items


def _parse_scalar(value: str):
    if value.startswith("[") and value.endswith("]"):
        return _parse_inline_list(value)
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def simple_yaml_load(text: str) -> Mapping[str, object]:
    root: MutableMapping[str, object] = {}
    stack: List[Tuple[int, MutableMapping[str, object]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key_part, _, value_part = line.strip().partition(":")
        key = key_part.strip()
        value_part = value_part.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value_part == "":
            container: MutableMapping[str, object] = {}
            parent[key] = container
            stack.append((indent, container))
        else:
            scalar = _parse_scalar(value_part)
            parent[key] = scalar
    return root


def load_yaml_data(path: Path) -> Mapping[str, object]:
    text = path.read_text()
    if yaml is not None:
        return yaml.safe_load(text)
    return simple_yaml_load(text)


def load_iso_universe(path: Path) -> IsoUniverse:
    try:
        data = load_yaml_data(path)
    except FileNotFoundError as exc:  # pragma: no cover - argparse ensures path
        raise BuilderError(f"iso yaml not found: {path}") from exc

    if data is None:
        data = {}

    if not isinstance(data, Mapping) or "isos" not in data:
        raise BuilderError("iso yaml missing 'isos' mapping")

    isos = data["isos"]
    region_states: Dict[str, Set[str]] = defaultdict(set)
    state_regions: Dict[str, Set[str]] = defaultdict(set)

    for iso_name, iso_data in sorted(isos.items()):
        if not isinstance(iso_data, Mapping) or "states" not in iso_data:
            raise BuilderError(f"iso entry {iso_name!r} missing states mapping")
        states = iso_data["states"]
        if not isinstance(states, Mapping):
            raise BuilderError(f"states for {iso_name!r} must be a mapping")
        for state, regions in sorted(states.items()):
            if not isinstance(regions, Sequence):
                raise BuilderError(f"regions for {state!r} must be a list")
            for region in regions:
                region_states[region].add(state)
                state_regions[state].add(region)

    region_states_sorted = {k: sorted(v) for k, v in region_states.items()}
    state_regions_sorted = {k: sorted(v) for k, v in state_regions.items()}
    return IsoUniverse(region_states_sorted, state_regions_sorted, isos)


def parse_filter_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None or value == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def apply_filters(
    universe: IsoUniverse,
    states_filter: Optional[Sequence[str]],
    regions_filter: Optional[Sequence[str]],
) -> Tuple[Dict[str, List[str]], Set[str]]:
    region_states = universe.region_states
    allowed_states: Set[str]
    if states_filter:
        allowed_states = {s for s in states_filter if s in universe.state_regions}
    else:
        allowed_states = set(universe.state_regions.keys())

    patterns = list(regions_filter or [])

    target_regions: Dict[str, List[str]] = {}
    for region, states in sorted(region_states.items()):
        if states_filter and not set(states).issubset(allowed_states):
            continue
        if patterns and not any(fnmatch.fnmatch(region, pat) for pat in patterns):
            continue
        filtered_states = [s for s in states if s in allowed_states]
        if not filtered_states:
            continue
        target_regions[region] = filtered_states

    return target_regions, allowed_states


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def ensure_directory(path: Path) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def decimalize(value: float | int | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def finalize_shares(
    raw_values: Mapping[str, Decimal],
    precision: int,
) -> OrderedDict[str, Decimal]:
    if not raw_values:
        return OrderedDict()

    total = sum(raw_values.values(), Decimal("0"))
    states = list(raw_values.keys())
    if total <= 0:
        equal = Decimal("1") / Decimal(len(states))
        normalized = {state: equal for state in states}
    else:
        normalized = {state: (value / total) for state, value in raw_values.items()}

    tiny = Decimal("1").scaleb(-precision)
    clipped = {}
    for state, value in normalized.items():
        if value < 0 and abs(value) <= tiny:
            value = Decimal("0")
        if value < 0:
            raise BuilderError(f"negative share for {state}: {value}")
        if value < tiny:
            value = Decimal("0")
        clipped[state] = value

    clipped_total = sum(clipped.values(), Decimal("0"))
    if clipped_total <= 0:
        equal = Decimal("1") / Decimal(len(states))
        clipped = {state: equal for state in states}
        clipped_total = Decimal("1")

    renormalized = {state: value / clipped_total for state, value in clipped.items()}
    quant = Decimal("1").scaleb(-precision)
    rounded = OrderedDict()
    for state in states:
        rounded[state] = renormalized[state].quantize(quant, rounding=ROUND_HALF_UP)

    diff = Decimal("1") - sum(rounded.values(), Decimal("0"))
    if diff != 0:
        adjust_state = max(states, key=lambda s: (renormalized[s], s))
        rounded[adjust_state] = (rounded[adjust_state] + diff).quantize(
            quant, rounding=ROUND_HALF_UP
        )
    return rounded


def equal_shares(states: Sequence[str], precision: int) -> OrderedDict[str, Decimal]:
    n = len(states)
    if n == 0:
        return OrderedDict()
    value = Decimal("1") / Decimal(n)
    raw = {state: value for state in states}
    return finalize_shares(raw, precision)


def invert_to_state_index(
    region_shares: Mapping[str, OrderedDict[str, Decimal]],
    threshold: float,
) -> Dict[str, Dict[str, Mapping[str, float] | List[str]]]:
    threshold_dec = decimalize(threshold)
    state_entries: Dict[str, List[Tuple[str, Decimal]]] = defaultdict(list)
    for region, shares in region_shares.items():
        for state, share in shares.items():
            if share >= threshold_dec:
                state_entries[state].append((region, share))

    result: Dict[str, Dict[str, Mapping[str, float] | List[str]]] = {}
    for state, entries in state_entries.items():
        entries.sort(key=lambda item: (-item[1], item[0]))
        regions = [region for region, _ in entries]
        weights = OrderedDict((region, float(share)) for region, share in entries)
        result[state] = {"regions": regions, "weights": weights}
    return result


def write_csv(path: Path, region_shares: Mapping[str, OrderedDict[str, Decimal]]) -> None:
    ensure_directory(path)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["region_id", "state", "share"])
        for region in sorted(region_shares.keys()):
            for state, share in region_shares[region].items():
                writer.writerow([region, state, f"{share:.6f}"])


def write_json(path: Path, state_index: Mapping[str, Mapping[str, object]]) -> None:
    ensure_directory(path)
    serialisable = {}
    for state in sorted(state_index.keys()):
        entry = state_index[state]
        serialisable[state] = {
            "regions": entry["regions"],
            "weights": {k: round(v, 6) for k, v in entry["weights"].items()},
        }
    path.write_text(json.dumps(serialisable, indent=2, sort_keys=True))


def build_qc_table(
    region_shares: Mapping[str, OrderedDict[str, Decimal]]
) -> pd.DataFrame:
    rows = []
    for region, shares in region_shares.items():
        decimals = list(shares.values())
        total = sum(decimals, Decimal("0"))
        rows.append(
            {
                "region_id": region,
                "n_states": len(decimals),
                "max_share": float(max(decimals) if decimals else Decimal("0")),
                "sum_share": float(total),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["region_id", "n_states", "max_share", "sum_share"])
    df = pd.DataFrame(rows)
    return df.sort_values("region_id").reset_index(drop=True)


def write_qc(path: Path, qc_df: pd.DataFrame) -> None:
    ensure_directory(path)
    qc_df.to_csv(path, index=False)


def print_summary(qc_df: pd.DataFrame, summary: int) -> None:
    if summary <= 0 or qc_df.empty:
        return
    display = qc_df.sort_values(["max_share", "region_id"], ascending=[False, True]).head(summary)
    print(display.to_string(index=False))


def validate_region_tokens(
    tokens: Iterable[str],
    target_regions: Mapping[str, Sequence[str]],
) -> None:
    valid = set(target_regions.keys())
    invalid = sorted(set(tokens) - valid)
    if not invalid:
        return
    suggestions = {}
    try:
        import difflib

        for token in invalid:
            matches = difflib.get_close_matches(token, valid, n=3, cutoff=0.5)
            if matches:
                suggestions[token] = matches
    except Exception:  # pragma: no cover - defensive
        suggestions = {}

    details = []
    for token in invalid:
        if token in suggestions:
            details.append(f"{token} (did you mean {', '.join(suggestions[token])})")
        else:
            details.append(token)
    raise BuilderError("invalid region_id tokens: " + ", ".join(details))


def load_uploaded_csv(
    path: Path,
    target_regions: Mapping[str, Sequence[str]],
    precision: int,
) -> Dict[str, OrderedDict[str, Decimal]]:
    df = pd.read_csv(path)
    required_cols = {"region_id", "state", "share"}
    if not required_cols.issubset(df.columns):
        raise BuilderError("uploaded CSV must contain region_id,state,share columns")

    df = df[df["region_id"].isin(target_regions.keys())]
    result: Dict[str, OrderedDict[str, Decimal]] = {}
    for region, states in target_regions.items():
        subset = df[df["region_id"] == region]
        raw: Dict[str, Decimal] = {}
        for state in states:
            match = subset[subset["state"] == state]
            if not match.empty:
                value = decimalize(match.iloc[0]["share"])
                raw[state] = value
        if raw:
            result[region] = finalize_shares(raw, precision)
    return result


def prepare_utility_weights(
    mapping_df: pd.DataFrame,
    override_df: Optional[pd.DataFrame],
    target_regions: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    if override_df is not None and not override_df.empty:
        df = override_df.copy()
        if "weight" not in df.columns:
            raise BuilderError("utility_zone_override missing weight column")
    else:
        df = mapping_df.copy()
        if "region_id" not in df.columns:
            raise BuilderError("utility map missing region_id column")
        if "utility_id_eia" not in df.columns:
            raise BuilderError("utility map missing utility_id_eia column")
        if "weight_hint" in df.columns:
            df["weight_hint"] = df["weight_hint"].fillna(0)
        else:
            df["weight_hint"] = 0.0
        weights = []
        for utility, util_df in df.groupby("utility_id_eia"):
            util_df = util_df.copy()
            hints = util_df["weight_hint"].astype(float)
            if len(util_df) == 1:
                weight = 1.0
                weights.append((utility, util_df.iloc[0]["region_id"], weight))
                continue
            hint_sum = hints.sum()
            if hint_sum > 0:
                for _, row in util_df.iterrows():
                    weights.append(
                        (utility, row["region_id"], float(row["weight_hint"]) / hint_sum)
                    )
            else:
                equal = 1.0 / len(util_df)
                for _, row in util_df.iterrows():
                    weights.append((utility, row["region_id"], equal))
        df = pd.DataFrame(weights, columns=["utility_id_eia", "region_id", "weight"])

    if "weight" not in df.columns:
        raise BuilderError("utility weights missing weight column")

    df["weight"] = df["weight"].astype(float)
    df = df[df["region_id"].isin(target_regions.keys())]
    if df.empty:
        raise BuilderError("no utility mappings remain after filtering")

    validate_region_tokens(df["region_id"].unique(), target_regions)

    grouped = df.groupby("utility_id_eia")
    normalised_rows = []
    for utility, util_df in grouped:
        total = util_df["weight"].sum()
        if not math.isfinite(total) or total <= 0:
            raise BuilderError(f"utility {utility} has invalid weights")
        util_df = util_df.copy()
        util_df["weight"] = util_df["weight"] / total
        for _, row in util_df.iterrows():
            normalised_rows.append(
                {
                    "utility_id_eia": int(row["utility_id_eia"]),
                    "region_id": row["region_id"],
                    "weight": float(row["weight"]),
                }
            )

    return pd.DataFrame(normalised_rows)


def aggregate_metric(
    df: Optional[pd.DataFrame],
    metric_col: str,
    year: int,
    allowed_states: Set[str],
) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    required = {"utility_id_eia", "state", "data_year", metric_col}
    if not required.issubset(df.columns):
        raise BuilderError(f"EIA file missing columns: {required}")
    filtered = df[df["data_year"] == year]
    if filtered.empty:
        return pd.DataFrame(columns=["utility_id_eia", "state", metric_col])
    filtered = filtered[filtered["state"].isin(allowed_states)]
    if filtered.empty:
        return pd.DataFrame(columns=["utility_id_eia", "state", metric_col])
    grouped = (
        filtered.groupby(["utility_id_eia", "state"], as_index=False)[metric_col].sum()
    )
    return grouped


def compute_region_state_metric(
    metric_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    metric_col: str,
) -> pd.DataFrame:
    merged = metric_df.merge(weights_df, on="utility_id_eia", how="left")
    if merged["region_id"].isna().any():
        utilities = merged[merged["region_id"].isna()]["utility_id_eia"].unique()
        raise BuilderError(
            "utilities missing region mapping: " + ", ".join(str(u) for u in utilities)
        )
    merged[metric_col] = merged[metric_col] * merged["weight"]
    grouped = (
        merged.groupby(["region_id", "state"], as_index=False)[metric_col].sum()
    )
    return grouped


def build_region_shares_from_metrics(
    target_regions: Mapping[str, Sequence[str]],
    precision: int,
    sales_df: Optional[pd.DataFrame],
    customers_df: Optional[pd.DataFrame],
) -> Dict[str, OrderedDict[str, Decimal]]:
    sales_lookup: Dict[Tuple[str, str], float] = {}
    if sales_df is not None:
        sales_lookup = {
            (row["region_id"], row["state"]): float(row["sales_mwh"])
            for _, row in sales_df.iterrows()
        }

    customers_lookup: Dict[Tuple[str, str], float] = {}
    if customers_df is not None:
        customers_lookup = {
            (row["region_id"], row["state"]): float(row["customers"])
            for _, row in customers_df.iterrows()
        }

    region_shares: Dict[str, OrderedDict[str, Decimal]] = {}
    for region, states in target_regions.items():
        region_states = list(states)
        values: Dict[str, Decimal] = {}
        has_sales = False
        invalid = False
        for state in region_states:
            key = (region, state)
            if key in sales_lookup:
                has_sales = True
                value = decimalize(sales_lookup[key])
                if value.is_nan():
                    invalid = True
                else:
                    values[state] = value
        if not has_sales or invalid or sum(values.values(), Decimal("0")) <= 0:
            # try customers fallback
            values = {}
            has_customers = False
            invalid = False
            for state in region_states:
                key = (region, state)
                if key in customers_lookup:
                    has_customers = True
                    value = decimalize(customers_lookup[key])
                    if value.is_nan():
                        invalid = True
                    else:
                        values[state] = value
            if not has_customers or invalid or sum(values.values(), Decimal("0")) <= 0:
                shares = equal_shares(region_states, precision)
                region_shares[region] = shares
                continue
        shares = finalize_shares(values, precision)
        region_shares[region] = shares

    return region_shares


def apply_overrides(
    region_shares: Dict[str, OrderedDict[str, Decimal]],
    overrides: Mapping[str, Mapping[str, float]],
    precision: int,
) -> Dict[str, OrderedDict[str, Decimal]]:
    updated = dict(region_shares)
    for region, states in overrides.items():
        if region not in updated:
            continue
        raw = {state: decimalize(value) for state, value in states.items() if state in updated[region]}
        if not raw:
            continue
        updated[region] = finalize_shares(raw, precision)
    return updated


def placeholder_shares(
    target_regions: Mapping[str, Sequence[str]],
    precision: int,
    overrides_path: Path,
) -> Dict[str, OrderedDict[str, Decimal]]:
    region_shares = {
        region: equal_shares(states, precision)
        for region, states in target_regions.items()
    }
    if overrides_path.exists():
        overrides_data = load_yaml_data(overrides_path)
        if isinstance(overrides_data, Mapping):
            region_shares = apply_overrides(region_shares, overrides_data, precision)
    return region_shares


def ensure_complete_regions(
    region_shares: Mapping[str, OrderedDict[str, Decimal]],
    target_regions: Mapping[str, Sequence[str]],
    strict: bool,
) -> None:
    missing = sorted(set(target_regions.keys()) - set(region_shares.keys()))
    if missing:
        message = "missing regions: " + ", ".join(missing)
        if strict:
            raise StrictSensitiveError(message)
        raise BuilderError(message)


def enforce_tolerance(
    qc_df: pd.DataFrame,
    tolerance: float,
    strict: bool,
) -> None:
    if qc_df.empty:
        return
    deviations = (qc_df["sum_share"] - 1.0).abs()
    worst = deviations.max()
    if math.isnan(worst):
        worst = float("inf")
    if worst > tolerance:
        message = f"per-region share sums exceed tolerance (max deviation {worst:.6g})"
        if strict:
            raise StrictSensitiveError(message)
        raise BuilderError(message)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    states_filter = parse_filter_list(args.states_filter)
    regions_filter = parse_filter_list(args.regions_filter)

    universe = load_iso_universe(args.iso_yaml)
    target_regions, allowed_states = apply_filters(universe, states_filter, regions_filter)

    try:
        if args.in_csv:
            region_shares = load_uploaded_csv(args.in_csv, target_regions, args.precision)
            ensure_complete_regions(region_shares, target_regions, args.strict)
        else:
            eia_inputs_ready = (
                args.utility_map
                and args.eia861_year
                and (args.eia861_retail or args.eia861_customers)
            )
            if eia_inputs_ready:
                mapping_df = load_table(args.utility_map)
                override_df = load_table(args.utility_zone_override) if args.utility_zone_override else None
                weights_df = prepare_utility_weights(mapping_df, override_df, target_regions)

                sales_df_raw = load_table(args.eia861_retail) if args.eia861_retail else None
                customers_df_raw = load_table(args.eia861_customers) if args.eia861_customers else None

                sales_metric = aggregate_metric(
                    sales_df_raw, "sales_mwh", args.eia861_year, allowed_states
                )
                customers_metric = aggregate_metric(
                    customers_df_raw, "customers", args.eia861_year, allowed_states
                )

                sales_by_region = None
                if sales_metric is not None and not sales_metric.empty:
                    sales_by_region = compute_region_state_metric(
                        sales_metric, weights_df, "sales_mwh"
                    )
                customers_by_region = None
                if customers_metric is not None and not customers_metric.empty:
                    customers_by_region = compute_region_state_metric(
                        customers_metric, weights_df, "customers"
                    )

                region_shares = build_region_shares_from_metrics(
                    target_regions, args.precision, sales_by_region, customers_by_region
                )
            else:
                overrides_path = args.iso_yaml.parent / "zone_share_overrides.yaml"
                region_shares = placeholder_shares(target_regions, args.precision, overrides_path)

            ensure_complete_regions(region_shares, target_regions, args.strict)

        qc_df = build_qc_table(region_shares)
        enforce_tolerance(qc_df, args.tolerance, args.strict)

        write_csv(args.out_csv, region_shares)
        state_index = invert_to_state_index(region_shares, args.threshold)
        write_json(args.out_json, state_index)

        if args.out_qc:
            write_qc(args.out_qc, qc_df)
        if args.summary:
            print_summary(qc_df, args.summary)
        return 0
    except StrictSensitiveError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2 if args.strict else 1
    except BuilderError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - manual invocation
    sys.exit(main())
