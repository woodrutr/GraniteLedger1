import types

import pandas as pd


def test_coerce_years_iterable_handles_scalar():
    from src.models.electricity.scripts import preprocessor

    helper = preprocessor._coerce_years_iterable

    assert helper(2030) == [2030]
    assert helper(None) == []
    assert helper([2025, 2030]) == [2025, 2030]
    assert helper(range(2)) == [0, 1]
    assert helper("2030") == ["2030"]


def test_capfactor_cross_join_accepts_scalar_year():
    from src.models.electricity.scripts import preprocessor

    capfactor = pd.DataFrame(
        {
            "tech": ["wind"],
            "region": [1],
            "step": [1],
            "hour": [1],
            "CapFactorVRE": [0.5],
        }
    )

    setin = types.SimpleNamespace(years=2030)
    all_frames = {"CapFactorVRE": capfactor}

    merged = pd.merge(
        all_frames["CapFactorVRE"],
        pd.DataFrame({"year": preprocessor._coerce_years_iterable(setin.years)}),
        how="cross",
    )

    assert merged["year"].tolist() == [2030]

