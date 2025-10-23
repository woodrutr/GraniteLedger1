from __future__ import annotations
import pytest
pd = pytest.importorskip("pandas")
from granite_io.frames_api import (
    Frames,
    _coerce_bool,
    _ensure_dataframe,
    _normalize_carbon_price_schedule,
    _normalize_name,
    _require_numeric,
    _validate_columns,
)
from regions.registry import REGIONS

REGION_IDS = list(REGIONS)
REGION_A = REGION_IDS[0]
REGION_B = REGION_IDS[1]


@pytest.fixture
def demand_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"year": 2025, "region": REGION_A, "demand_mwh": 100.0},
            {"year": 2025, "region": REGION_B, "demand_mwh": 150.0},
            {"year": 2026, "region": REGION_A, "demand_mwh": 125.0},
        ]
    )


def test_normalize_name_casts_inputs_to_lowercase() -> None:
    assert _normalize_name("Demand") == "demand"
    assert _normalize_name(123) == "123"


def test_ensure_dataframe_validates_type(demand_frame: pd.DataFrame) -> None:
    result = _ensure_dataframe("demand", demand_frame)
    assert result is not demand_frame
    with pytest.raises(TypeError):
        _ensure_dataframe("demand", [1, 2, 3])


def test_normalize_carbon_price_schedule_filters_invalid_entries() -> None:
    schedule = {"2025": "15", "bad": "x", None: 12.0}
    normalized = _normalize_carbon_price_schedule(schedule)
    assert normalized == {2025: 15.0, None: 12.0}


def test_validate_columns_and_require_numeric(demand_frame: pd.DataFrame) -> None:
    validated = _validate_columns("demand", demand_frame, ["year", "demand_mwh"])
    years = _require_numeric("demand", "year", validated["year"])
    assert years.dtype.kind in {"i", "f"}
    with pytest.raises(ValueError):
        _validate_columns("demand", demand_frame.drop(columns=["year"]), ["year"])
    with pytest.raises(ValueError):
        _require_numeric("demand", "demand_mwh", pd.Series(["x", "y"]))


def test_require_numeric_allows_missing_values_when_requested() -> None:
    series = pd.Series(["1.5", None, " "])
    numeric = _require_numeric(
        "fuels",
        "co2_short_ton_per_mwh",
        series,
        allow_missing=True,
    )
    assert numeric.iloc[0] == pytest.approx(1.5)
    assert pd.isna(numeric.iloc[1])
    assert pd.isna(numeric.iloc[2])
    with pytest.raises(ValueError):
        _require_numeric(
            "fuels",
            "co2_short_ton_per_mwh",
            pd.Series(["abc"]),
            allow_missing=True,
        )


def test_coerce_bool_converts_tokens_to_boolean() -> None:
    series = pd.Series(["true", "False", 1, 0, None])
    coerced = _coerce_bool(series, "coverage", "covered")
    assert list(coerced.astype("boolean")) == [True, False, True, False, pd.NA]


def test_frames_demand_accessors_validate_and_group(demand_frame: pd.DataFrame) -> None:
    frames = Frames(
        {
            "demand": demand_frame,
            "units": pd.DataFrame(
                [
                    {
                        "unit_id": "u1",
                        "unique_id": "u1",
                        "region": REGION_A,
                        "fuel": "wind",
                        "cap_mw": 10.0,
                        "availability": 1.0,
                        "hr_mmbtu_per_mwh": 0.0,
                        "vom_per_mwh": 0.0,
                        "fuel_price_per_mmbtu": 0.0,
                        "ef_ton_per_mwh": 0.0,
                    }
                ]
            ),
            "fuels": pd.DataFrame({"fuel": ["wind"], "covered": [True]}),
            "transmission": pd.DataFrame(columns=["from_region", "to_region", "limit_mw"]),
        }
    )
    validated = frames.demand()
    assert list(validated.columns) == ["year", "region", "demand_mwh"]
    demand_map = frames.demand_for_year(2025)
    assert demand_map == {REGION_A: 100.0, REGION_B: 150.0}
    with pytest.raises(KeyError):
        frames.demand_for_year(2030)
    duplicate = demand_frame.copy()
    duplicate.loc[len(duplicate)] = {"year": 2025, "region": REGION_A, "demand_mwh": 5.0}
    frames_with_dupes = Frames(
        {
            "demand": duplicate,
            "units": frames.units(),
            "fuels": frames.fuels(),
            "transmission": frames.transmission(),
        }
    )
    with pytest.raises(ValueError):
        frames_with_dupes.demand()
