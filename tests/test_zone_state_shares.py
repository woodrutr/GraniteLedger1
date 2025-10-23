from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from engine.regions import shares as zone_shares


def test_zone_state_share_rows_sum_to_unity() -> None:
    share_df = zone_shares.load_zone_to_state_share()
    grouped = share_df.groupby("region_id")["share"].sum()
    for _, total in grouped.items():
        assert total == pytest.approx(1.0, abs=1e-6)

    for region_id in ("PJM_PEP", "PJM_DOM"):
        region_rows = share_df[share_df["region_id"] == region_id]
        if region_rows.empty:
            continue
        assert len(region_rows["state"].unique()) == 2
        assert ((region_rows["share"] > 0.0) & (region_rows["share"] < 1.0)).all()
