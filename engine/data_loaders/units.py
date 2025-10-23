from __future__ import annotations

from collections.abc import Iterable
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from engine.normalization import normalize_region_id


LOGGER = logging.getLogger(__name__)


_CANON_RENAME = {
    # capacity
    "capacity (mw)": "cap_mw",
    "capacity_mw": "cap_mw",
    "cap_mw": "cap_mw",
    # heat rate
    "heat rate (mmbtu/mwh)": "hr_mmbtu_per_mwh",
    "heat_rate (mmbtu/mwh)": "hr_mmbtu_per_mwh",
    "heat_rate_mmbtu_mwh": "hr_mmbtu_per_mwh",
    "heat_rate_mmbtu_per_mwh": "hr_mmbtu_per_mwh",
    "hr_mmbtu_per_mwh": "hr_mmbtu_per_mwh",
    # emission rate
    "emission rate (short ton/mwh)": "ef_ton_per_mwh",
    "emission_rate (short ton/mwh)": "ef_ton_per_mwh",
    "emission_rate_ton_mwh": "ef_ton_per_mwh",
    "co2_short_ton_per_mwh": "ef_ton_per_mwh",
    "ef_ton_per_mwh": "ef_ton_per_mwh",
}

_NUMERIC = [
    "cap_mw",
    "hr_mmbtu_per_mwh",
    "ef_ton_per_mwh",
    "vom_per_mwh",
    "availability",
    "fuel_price_per_mmbtu",
]


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with canonical column names and numeric coercions."""

    work = df.copy()
    work.columns = work.columns.str.strip().str.lower()
    work = work.rename(columns=_CANON_RENAME)

    # Backfill when only legacy names are provided
    if "cap_mw" not in work and "capacity_mw" in work:
        work = work.rename(columns={"capacity_mw": "cap_mw"})
    if "hr_mmbtu_per_mwh" not in work and "heat_rate_mmbtu_mwh" in work:
        work = work.rename(columns={"heat_rate_mmbtu_mwh": "hr_mmbtu_per_mwh"})
    if "ef_ton_per_mwh" not in work and "emission_rate_ton_mwh" in work:
        work = work.rename(columns={"emission_rate_ton_mwh": "ef_ton_per_mwh"})

    for column in _NUMERIC:
        if column in work.columns:
            # Access the column
            col_data = work[column]
            
            # Handle DataFrame case (duplicate columns)
            if isinstance(col_data, pd.DataFrame):
                LOGGER.warning(f"Column '{column}' is a DataFrame (duplicate columns), taking first occurrence")
                col_data = col_data.iloc[:, 0]
                # Remove duplicate columns
                col_positions = [i for i, c in enumerate(work.columns) if c == column]
                if len(col_positions) > 1:
                    # Keep only first occurrence
                    keep_indices = [i for i in range(len(work.columns)) if work.columns[i] != column or i == col_positions[0]]
                    work = work.iloc[:, keep_indices]
            
            try:
                work[column] = pd.to_numeric(col_data, errors="coerce")
            except (TypeError, ValueError) as e:
                LOGGER.error(f"Column '{column}' failed pd.to_numeric: {e}")
                # Last resort: convert to list and recreate
                try:
                    values = col_data.tolist() if hasattr(col_data, 'tolist') else [col_data] * len(work)
                    work[column] = pd.to_numeric(pd.Series(values, index=work.index), errors="coerce")
                except Exception as e2:
                    LOGGER.error(f"Cannot convert column '{column}': {e2}. Setting to 0.")
                    work[column] = 0.0

    return work


def _as_string_series(series: pd.Series | None, index: pd.Index) -> pd.Series:
    """Return ``series`` coerced to the pandas ``string`` dtype on ``index``."""

    if series is None:
        return pd.Series(pd.NA, index=index, dtype="string")

    working = pd.Series(series, copy=True)
    if not working.index.equals(index):
        working = working.reindex(index)

    try:
        stringified = working.astype("string")
    except TypeError:  # pragma: no cover - fallback for non-standard types
        stringified = working.astype(str).astype("string")

    return stringified.str.strip()


def _coerce_identifier(value: object) -> str | None:
    """Return a normalised identifier string for ``value`` when possible."""

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:  # pragma: no cover - non-scalar objects
        pass

    text = str(value).strip()
    return text or None


def _normalize_unique_identifiers(units: pd.DataFrame) -> pd.Series:
    """Return a normalised ``unique_id`` series for ``units``."""

    index = units.index
    unique_raw = _as_string_series(units.get("unique_id"), index)
    fallback_raw = _as_string_series(units.get("unit_id"), index)

    invalid_tokens = {"", "nan", "none", "null"}
    mask = unique_raw.isna() | unique_raw.str.lower().isin(invalid_tokens)
    normalized = unique_raw.where(~mask, fallback_raw)

    missing_mask = normalized.isna() | normalized.eq("")
    if missing_mask.any():
        generated: dict[int | str, str] = {}
        for idx, row in units.loc[missing_mask].iterrows():
            candidate = _coerce_identifier(getattr(row, "unit_id", None))
            if candidate is None:
                plant = _coerce_identifier(
                    getattr(row, "plant_id_eia", getattr(row, "plant_id", None))
                )
                generator = _coerce_identifier(getattr(row, "generator_id", None))
                if plant and generator:
                    candidate = f"{plant}_{generator}"
                elif plant:
                    candidate = f"{plant}_{idx}"
                else:
                    candidate = f"unit_{idx}"
            generated[idx] = candidate

        generated_series = pd.Series(generated, dtype="string")
        normalized.loc[generated_series.index] = generated_series

    final_mask = normalized.isna() | normalized.eq("")
    if final_mask.any():
        fallback_ids = {
            idx: f"unit_{idx}" for idx in normalized.index[final_mask]
        }
        normalized.loc[list(fallback_ids)] = pd.Series(
            fallback_ids, dtype="string"
        )

    return normalized.astype("string").str.strip()


def load_unit_fleet(
    active_regions: Iterable[str] | None = None,
    path: str | Path | Iterable[str | Path] = "input/ei_units.csv",
) -> pd.DataFrame:
    """Load the legacy EI unit fleet CSV preserving dispatch attributes."""

    if isinstance(path, Iterable) and not isinstance(path, (str, Path)):
        frames = [
            load_unit_fleet(active_regions=active_regions, path=p)  # recursive call
            for p in path
        ]
        return pd.concat(frames, ignore_index=True)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Unit fleet file not found: {path}")

    df = pd.read_csv(
        path,
        dtype={
            "unique_id": "string",
            "unit_id": "string",
            "region_id": "string",
            "generator_id": "string",
            "unit_code": "string",
            "eia_plant_id": "Int64",
        },
        keep_default_na=False,
        na_values=[""],
    )
    df = df.dropna(how="all")
    df.columns = df.columns.str.strip().str.lower()

    if (
        "co2_kg_per_mmbtu" in df.columns
        and "co2_short_ton_per_mwh" not in df.columns
        and "ef_ton_per_mwh" not in df.columns
    ):
        df["co2_short_ton_per_mwh"] = (
            pd.to_numeric(df["co2_kg_per_mmbtu"], errors="coerce") / 1000
        )

    df = _normalize_schema(df)

    rename_map = {
        "unit id": "unit_id",
        "unique id": "unique_id",
        "region id": "region_id",
        "technology": "fuel",
        "is_covered": "covered",
    }

    df = df.rename(columns=rename_map)
    normalized_unique = _normalize_unique_identifiers(df)
    df["unique_id"] = normalized_unique
    df["unit_id"] = normalized_unique

    out = df

    required = [
        "unit_id",
        "hr_mmbtu_per_mwh",
        "cap_mw",
        "region_id",
        "ef_ton_per_mwh",
    ]
    for col in required:
        if col not in out.columns:
            raise ValueError(f"Missing required column: {col}")

    columns = [
        "unit_id",
        "unique_id",
        "region_id",
        "cap_mw",
        "hr_mmbtu_per_mwh",
        "ef_ton_per_mwh",
    ]
    if "fuel" in out.columns:
        columns.insert(2, "fuel")
    if "covered" in out.columns and "covered" not in columns:
        columns.append("covered")

    out = out.reindex(columns=columns)
    out["cap_mw"] = pd.to_numeric(out["cap_mw"], errors="coerce")
    out["hr_mmbtu_per_mwh"] = pd.to_numeric(out["hr_mmbtu_per_mwh"], errors="coerce")
    out["ef_ton_per_mwh"] = pd.to_numeric(out["ef_ton_per_mwh"], errors="coerce")

    # normalize and filter regions
    def _normalize_region(value: Any) -> str:
        text = str(value).strip()
        if not text:
            return ""
        try:
            return normalize_region_id(text)
        except ValueError:
            return text.upper()

    out["region_id"] = out["region_id"].map(_normalize_region)
    bad = out["region_id"].eq("")
    if bad.any():
        LOGGER.warning("Dropping %d rows with unknown region_id", int(bad.sum()))
    out = out.loc[~bad].copy()
    if active_regions:
        target = {normalize_region_id(region) for region in active_regions}
        out = out[out["region_id"].isin(target)].copy()

    # keys: unique_id authoritative
    out["unique_id"] = out["unique_id"].astype("string").str.strip()
    out["unit_id"] = out["unique_id"]
    if out["unique_id"].isna().any() or (out["unique_id"] == "").any():
        raise ValueError("unique_id contains null/empty after normalization")
    if out["unique_id"].duplicated().any():
        raise ValueError("duplicate unique_id in supplied file")

    if "fuel" in out.columns:
        out["fuel"] = out["fuel"].astype(str).str.strip()
    if "covered" in out.columns:
        out["covered"] = (
            out["covered"].astype(str).str.strip().str.lower().isin(
                {"1", "true", "t", "yes", "y"}
            )
        )

    return out


def _read_units_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_schema(df)
    rename_map: dict[str, str] = {
        "unit_id_gl": "unit_id",
        "unit_id": "unit_id",
        "zone": "region_id",
        "region": "region_id",
        "fuel": "fuel",
        "net_max_mw": "cap_mw",
        "nameplate_mw": "cap_mw_nameplate",
        "heat_rate_mmbtu_per_mwh": "hr_mmbtu_per_mwh",
        "co2_t_per_mwh": "ef_ton_per_mwh",
        "vom_per_mwh": "vom_per_mwh",
        "covered": "covered",
    }
    for source, target in rename_map.items():
        if source in df.columns and target not in df.columns:
            df[target] = df.pop(source)

    if "cap_mw" not in df.columns and "cap_mw_nameplate" in df.columns:
        df["cap_mw"] = df["cap_mw_nameplate"]

    df["cap_mw"] = pd.to_numeric(df.get("cap_mw"), errors="coerce")
    df["hr_mmbtu_per_mwh"] = pd.to_numeric(df.get("hr_mmbtu_per_mwh"), errors="coerce")

    vom_source = df["vom_per_mwh"] if "vom_per_mwh" in df.columns else pd.Series(0.0, index=df.index)
    df["vom_per_mwh"] = pd.to_numeric(vom_source, errors="coerce").fillna(0.0)

    ef_source = df.get("ef_ton_per_mwh")
    df["ef_ton_per_mwh"] = pd.to_numeric(ef_source, errors="coerce")

    if "availability" in df.columns:
        availability_source = df["availability"]
    else:
        # Set realistic technology-based capacity factors when availability column is missing
        fuel_upper = df.get("fuel", pd.Series("", index=df.index)).astype(str).str.upper().str.strip()
        availability_source = pd.Series(1.0, index=df.index)
        
        # Fossil fuel thermal plants
        availability_source.loc[fuel_upper.str.contains("COAL", na=False)] = 0.60
        availability_source.loc[fuel_upper.str.contains("GAS.*COMBINED|NGCC|COMBINEDCYCLE", regex=True, na=False)] = 0.75
        availability_source.loc[fuel_upper.str.contains("GAS.*STEAM|GASSTEAM", regex=True, na=False)] = 0.50
        availability_source.loc[fuel_upper.str.contains("GAS.*TURBINE|NGCT|CT|COMBUSTION", regex=True, na=False)] = 0.20
        availability_source.loc[fuel_upper.str.contains("OIL", na=False)] = 0.30
        
        # Renewables
        availability_source.loc[fuel_upper.isin({"SOLAR", "PV", "UTILITY-SCALE PV"})] = 0.25
        availability_source.loc[fuel_upper.str.contains("WIND", na=False)] = 0.35
        availability_source.loc[fuel_upper.isin({"HYDRO", "WATER"})] = 0.50
        availability_source.loc[fuel_upper.isin({"GEOTHERMAL"})] = 0.85
        
        # Baseload plants
        availability_source.loc[fuel_upper.isin({"NUCLEAR"})] = 0.90
        availability_source.loc[fuel_upper.isin({"BIOMASS"})] = 0.70
    
    df["availability"] = pd.to_numeric(availability_source, errors="coerce").fillna(1.0)

    df["region_id"] = df.get("region_id", "").astype(str)
    df["fuel"] = df.get("fuel", "").astype(str)
    df["covered"] = df.get("covered", False)

    return df


def _attach_fuel_prices(
    units: pd.DataFrame,
    *,
    fuel_price_path: Path,
    default_scenario: str = "REF",
) -> pd.Series:
    if not fuel_price_path.exists():
        return pd.Series(0.0, index=units.index)

    prices = pd.read_csv(fuel_price_path)
    if prices.empty:
        return pd.Series(0.0, index=units.index)

    prices["scenario_id"] = prices.get("scenario_id", "").astype(str)
    target_prices = prices
    if "scenario_id" in prices.columns and default_scenario:
        mask = prices["scenario_id"].str.upper() == default_scenario.upper()
        selected = prices.loc[mask]
        if not selected.empty:
            target_prices = selected

    target_prices = target_prices.copy()
    target_prices["region_id"] = target_prices.get("region_id", "").astype(str).str.upper()
    target_prices["fuel"] = target_prices.get("fuel", "").astype(str).str.upper()
    target_prices["price_per_mmbtu"] = pd.to_numeric(
        target_prices.get("price_per_mmbtu"), errors="coerce"
    )

    averages = (
        target_prices.dropna(subset=["price_per_mmbtu"])
        .groupby("fuel")["price_per_mmbtu"]
        .mean()
    )

    merged = units.copy()
    merged["region_norm"] = merged["region_id"].astype(str).str.upper()
    merged["fuel_norm"] = merged["fuel"].astype(str).str.upper()
    merged = merged.merge(
        target_prices[["region_id", "fuel", "price_per_mmbtu"]],
        left_on=["region_norm", "fuel_norm"],
        right_on=["region_id", "fuel"],
        how="left",
    )

    prices_series = merged["price_per_mmbtu"].copy()
    missing_mask = prices_series.isna()
    if missing_mask.any():
        fallback = merged.loc[missing_mask, "fuel_norm"].map(averages)
        prices_series.loc[missing_mask] = fallback
    prices_series = prices_series.fillna(0.0)

    return prices_series.astype(float)


def load_units(
    *,
    active_regions: Iterable[str] | None = None,
    unit_catalog: str | Path | None = None,
    fuel_price_catalog: str | Path | None = "input/fuel_prices/fuel_prices_annual.csv",
    default_price_scenario: str = "REF",
) -> pd.DataFrame:
    """Return units formatted for dispatch, merging defaults as needed."""

    # ei_units.csv is the only authoritative source
    if unit_catalog is None:
        units = load_unit_fleet(active_regions=active_regions)
    else:
        catalog_path = Path(unit_catalog)
        if catalog_path.exists():
            units = _read_units_catalog(catalog_path)
        else:
            raise FileNotFoundError(f"Unit catalog not found: {catalog_path}")

    if units.empty:
        raise ValueError("Unit catalog produced an empty frame")

    units = _normalize_schema(units.copy())
    normalized_unique = _normalize_unique_identifiers(units)
    units["unique_id"] = normalized_unique
    units["unit_id"] = normalized_unique

    units["region_id"] = units["region_id"].astype(str)
    units["fuel"] = units["fuel"].astype(str)

    def _normalize_optional(value: Any) -> str:
        try:
            return normalize_region_id(value)
        except ValueError:
            return ""

    # Normalize regions and drop unknowns up front
    region_norm = units["region_id"].map(_normalize_optional)
    drop_mask = region_norm.eq("")
    if drop_mask.any():
        LOGGER.warning("Dropping %d unit rows with unknown region_id", int(drop_mask.sum()))
    units = units.loc[~drop_mask].copy()
    region_norm = region_norm.loc[units.index]
    units["region_id"] = region_norm

    if active_regions:
        def _norm_active(regions: Iterable[str]) -> set[str]:
            ok: set[str] = set()
            bad: list[str] = []
            for region in regions:
                try:
                    ok.add(normalize_region_id(region))
                except ValueError:
                    bad.append(str(region))
            if bad:
                LOGGER.warning("Ignoring unknown active regions: %s", bad)
            return ok

        normalized = _norm_active(active_regions)
        if not normalized:
            raise ValueError("No valid active regions after normalization")

        filtered = units[units["region_id"].isin(normalized)].copy()

        if filtered.empty:
            raise ValueError(
                f"Unit catalog {catalog_path} has no rows for regions: {sorted(normalized)}"
            )

        units = filtered
        region_norm = region_norm.loc[units.index]
    else:
        region_norm = region_norm.loc[units.index]

    if units.empty:
        return units

    units["cap_mw"] = pd.to_numeric(
        units.get("cap_mw", pd.Series(pd.NA, index=units.index)), errors="coerce"
    )
    missing_capacity = units["cap_mw"].isna()
    if missing_capacity.any():
        LOGGER.info(
            "Substituting zero capacity for %s units missing data from catalog %s.",
            int(missing_capacity.sum()),
            str(catalog_path),
        )
        units.loc[missing_capacity, "cap_mw"] = 0.0

    units["hr_mmbtu_per_mwh"] = pd.to_numeric(
        units.get("hr_mmbtu_per_mwh", pd.Series(pd.NA, index=units.index)),
        errors="coerce",
    )

    if "ef_ton_per_mwh" not in units.columns:
        emissions = units.get("ef_ton_per_mwh")
        units["ef_ton_per_mwh"] = pd.to_numeric(
            emissions if emissions is not None else pd.Series(pd.NA, index=units.index),
            errors="coerce",
        )
    else:
        units["ef_ton_per_mwh"] = pd.to_numeric(
            units["ef_ton_per_mwh"], errors="coerce"
        )

    # Non-emitting fuels default to zero; fossil fuels must provide an EF.
    zero_ef_fuels = {
        "NUCLEAR",
        "HYDRO",
        "WIND",
        "SOLAR",
        "GEOTHERMAL",
        "WATER",
        "PV",
        "WIND ONSHORE",
        "WIND OFFSHORE",
    }
    fuel_upper = units["fuel"].astype(str).str.upper().str.strip()
    units.loc[fuel_upper.isin(zero_ef_fuels), "ef_ton_per_mwh"] = 0.0

    missing_ef = units["ef_ton_per_mwh"].isna()
    fossil_mask = ~fuel_upper.isin(zero_ef_fuels)
    if (missing_ef & fossil_mask).any():
        examples = units.loc[missing_ef & fossil_mask, ["region_id", "fuel"]].drop_duplicates()
        raise ValueError(
            "Units table missing emission factors for fossil units; examples: "
            + ", ".join(f"{region}/{fuel}" for region, fuel in examples.head(8).itertuples(index=False))
        )

    # Any remaining NAs correspond to non-emitting fuels.
    units["ef_ton_per_mwh"] = units["ef_ton_per_mwh"].fillna(0.0)

    if "vom_per_mwh" in units.columns:
        vom_source = units["vom_per_mwh"]
    else:
        vom_source = pd.Series(0.0, index=units.index)
    units["vom_per_mwh"] = pd.to_numeric(vom_source, errors="coerce").fillna(0.0)

    if "availability" in units.columns:
        availability_source = units["availability"]
    else:
        # Set realistic technology-based capacity factors when availability column is missing
        fuel_upper = units["fuel"].astype(str).str.upper().str.strip()
        availability_source = pd.Series(1.0, index=units.index)
        
        # Fossil fuel thermal plants (limited by maintenance, cycling, economics)
        availability_source.loc[fuel_upper.str.contains("COAL", na=False)] = 0.60
        availability_source.loc[fuel_upper.str.contains("GAS.*COMBINED|NGCC|COMBINEDCYCLE", regex=True, na=False)] = 0.75
        availability_source.loc[fuel_upper.str.contains("GAS.*STEAM|GASSTEAM", regex=True, na=False)] = 0.50
        availability_source.loc[fuel_upper.str.contains("GAS.*TURBINE|NGCT|CT|COMBUSTION", regex=True, na=False)] = 0.20
        availability_source.loc[fuel_upper.str.contains("OIL", na=False)] = 0.30
        
        # Renewables (resource-limited)
        availability_source.loc[fuel_upper.isin({"SOLAR", "PV", "UTILITY-SCALE PV"})] = 0.25
        availability_source.loc[fuel_upper.str.contains("WIND", na=False)] = 0.35
        availability_source.loc[fuel_upper.isin({"HYDRO", "WATER"})] = 0.50
        availability_source.loc[fuel_upper.isin({"GEOTHERMAL"})] = 0.85
        
        # Baseload plants
        availability_source.loc[fuel_upper.isin({"NUCLEAR"})] = 0.90
        availability_source.loc[fuel_upper.isin({"BIOMASS"})] = 0.70
        
        LOGGER.info(
            "Availability column missing from units data - applied technology-based capacity factors: "
            "Coal=0.60, GasCC=0.75, GasCT=0.20, Nuclear=0.90, Solar=0.25, Wind=0.35"
        )
    
    units["availability"] = pd.to_numeric(
        availability_source, errors="coerce"
    ).fillna(1.0)

    fuel_price_path = Path(fuel_price_catalog) if fuel_price_catalog else Path("")
    units["fuel_price_per_mmbtu"] = _attach_fuel_prices(
        units,
        fuel_price_path=fuel_price_path,
        default_scenario=default_price_scenario,
    )
    if (units["fuel_price_per_mmbtu"] <= 0.0).any():
        positive_mask = units["fuel_price_per_mmbtu"] > 0.0
        if positive_mask.any():
            fallback_price = units.loc[positive_mask, "fuel_price_per_mmbtu"].mean()
        else:
            fallback_price = 1.0
        units.loc[~positive_mask, "fuel_price_per_mmbtu"] = float(fallback_price or 1.0)

    if region_norm is None or len(region_norm) != len(units):
        region_norm = units["region_id"].map(_normalize_optional)
    units["region"] = region_norm.astype(str).replace({"": pd.NA})
    units = units.dropna(subset=["region"]).copy()

    identifier_series = units["unique_id"].astype("string").str.strip()
    columns: list[tuple[str, Any]] = [
        ("unique_id", identifier_series),
        ("unit_id", identifier_series),
        ("region", units["region"]),
        ("fuel", units["fuel"].astype(str).str.strip()),
        ("cap_mw", units["cap_mw"].astype(float)),
        ("availability", units["availability"].clip(lower=0.0, upper=1.0)),
        ("hr_mmbtu_per_mwh", units["hr_mmbtu_per_mwh"].astype(float)),
        ("vom_per_mwh", units["vom_per_mwh"].astype(float)),
        ("fuel_price_per_mmbtu", units["fuel_price_per_mmbtu"].astype(float)),
        ("ef_ton_per_mwh", units["ef_ton_per_mwh"].astype(float)),
    ]

    output = pd.DataFrame({name: series for name, series in columns})
    for column in ("unique_id", "unit_id"):
        if output[column].isna().any():
            output[column] = output[column].ffill().bfill()
        output.dropna(subset=[column], inplace=True)
        output[column] = output[column].astype(str).str.strip()

    dup = output["unique_id"].duplicated(keep=False)
    if dup.any():
        sample = output.loc[dup, ["unique_id", "region", "fuel", "cap_mw"]].head(30)
        raise ValueError(
            "duplicate unique_id after normalization:\n" + sample.to_string(index=False)
        )

    output["unit_id"] = output["unique_id"]

    return output.reset_index(drop=True)


def _normalize_boolean(value: Any, *, default: bool) -> bool:
    """Return a best-effort boolean interpretation of ``value``."""

    if value is pd.NA or value is None:
        return bool(default)

    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)

    text = str(value).strip().lower()
    if not text:
        return bool(default)
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return bool(default)


def derive_fuels(
    units: pd.DataFrame,
    *,
    default_coverage: bool = True,
) -> pd.DataFrame:
    """Aggregate per-fuel attributes from a unit fleet table."""

    columns = ["fuel", "covered", "co2_short_ton_per_mwh"]
    if not isinstance(units, pd.DataFrame) or units.empty:
        return pd.DataFrame(columns=columns)

    working = units.copy()
    
    # Remove duplicate columns immediately after copy
    # Check manually since .duplicated() might not detect all cases
    col_list = working.columns.tolist()
    if len(col_list) != len(set(col_list)):
        LOGGER.warning(f"Input DataFrame has {len(col_list) - len(set(col_list))} duplicate columns. Removing...")
        # Keep only unique columns (first occurrence)
        seen = set()
        unique_cols = []
        for col in col_list:
            if col not in seen:
                seen.add(col)
                unique_cols.append(col)
        working = working[unique_cols]

    rename_map = {
        "Fuel": "fuel",
        "fuel_type": "fuel",
        "Technology": "fuel",
        "Heat Rate (mmBtu/MWh)": "hr_mmbtu_per_mwh",
        "heat_rate": "hr_mmbtu_per_mwh",
        "Heat Rate": "hr_mmbtu_per_mwh",
        "Emission Rate (short ton/MWh)": "ef_ton_per_mwh",
        "Emission Rate (short ton per MWh)": "ef_ton_per_mwh",
        "emission_rate": "ef_ton_per_mwh",
        "Emission Rate (short ton/MWh) ": "ef_ton_per_mwh",
        "Covered": "covered",
        "COVERED": "covered",
        "is_covered": "covered",
    }
    # Build a safe rename dict that only includes existing columns  
    safe_rename = {source: target for source, target in rename_map.items() 
                   if source in working.columns and target not in working.columns}
    
    # Use pandas rename instead of pop to avoid column corruption
    if safe_rename:
        working = working.rename(columns=safe_rename)
    
    # Remove duplicate columns if any exist (this can happen after rename)
    if working.columns.duplicated().any():
        LOGGER.warning(f"Duplicate columns found: {working.columns[working.columns.duplicated()].tolist()}")
        working = working.loc[:, ~working.columns.duplicated(keep='last')]
    
    working = _normalize_schema(working)

    if "fuel" not in working.columns:
        return pd.DataFrame(columns=columns)

    working["fuel"] = working["fuel"].astype(str).str.strip()
    working = working[working["fuel"].astype(bool)]
    if working.empty:
        return pd.DataFrame(columns=columns)

    # Handle missing columns properly - use column if exists, otherwise fill with NaN
    if "hr_mmbtu_per_mwh" in working.columns:
        working["hr_mmbtu_per_mwh"] = pd.to_numeric(
            working["hr_mmbtu_per_mwh"], errors="coerce"
        )
    else:
        working["hr_mmbtu_per_mwh"] = pd.Series([float("nan")] * len(working), index=working.index)
    
    if "ef_ton_per_mwh" in working.columns:
        working["ef_ton_per_mwh"] = pd.to_numeric(
            working["ef_ton_per_mwh"], errors="coerce"
        )
    else:
        working["ef_ton_per_mwh"] = pd.Series([float("nan")] * len(working), index=working.index)

    coverage_column = working.get("covered")
    if coverage_column is not None:
        working["covered"] = [
            _normalize_boolean(value, default=default_coverage)
            for value in coverage_column
        ]
    else:
        working["covered"] = pd.Series([bool(default_coverage)] * len(working), index=working.index)

    ratio = working["ef_ton_per_mwh"] / working["hr_mmbtu_per_mwh"]
    ratio = ratio.replace([pd.NA, pd.NaT], float("nan"))
    ratio = ratio.astype(float)
    invalid_mask = (working["hr_mmbtu_per_mwh"] <= 0) | ratio.isna()
    ratio.loc[invalid_mask] = float("nan")
    working["co2_short_ton_per_mwh"] = ratio

    grouped = working.groupby("fuel", sort=True)
    coverage = grouped["covered"].any()
    emissions = grouped["co2_short_ton_per_mwh"].mean()

    result = pd.DataFrame(
        {
            "fuel": coverage.index.astype(str),
            "covered": coverage.astype(bool).values,
            "co2_short_ton_per_mwh": emissions.reindex(coverage.index).astype(float).values,
        }
    )
    return result.reset_index(drop=True)


__all__ = ["load_unit_fleet", "load_units", "derive_fuels"]
