import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "geo" / "build_zone_state_shares.py"


def run_script(tmp_path: Path, args: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT)] + args
    return subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=tmp_path)


def write_iso(tmp_path: Path) -> Path:
    iso_yaml = tmp_path / "iso_state_zones.yaml"
    iso_yaml.write_text(
        """
schema_version: 1
isos:
  TEST:
    states:
      MD: [TEST_A, TEST_B]
      PA: [TEST_A]
      VA: [TEST_B]
  ALT:
    states:
      NJ: [ALT_A]
      PA: [ALT_A]
"""
    )
    return iso_yaml


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def test_invert_uploaded_csv(tmp_path):
    iso_yaml = write_iso(tmp_path)
    input_csv = tmp_path / "uploaded.csv"
    input_csv.write_text(
        "region_id,state,share\n"
        "TEST_A,MD,0.7\n"
        "TEST_A,PA,0.3\n"
        "TEST_B,MD,0.4\n"
        "TEST_B,VA,0.6\n"
        "ALT_A,NJ,0.5\n"
        "ALT_A,PA,0.5\n"
    )

    out_csv = tmp_path / "zone_to_state_share.csv"
    out_json = tmp_path / "state_to_regions.json"
    out_qc = tmp_path / "qc.csv"

    run_script(
        tmp_path,
        [
            "--iso-yaml",
            str(iso_yaml),
            "--in-csv",
            str(input_csv),
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
            "--out-qc",
            str(out_qc),
            "--summary",
            "5",
        ],
    )

    df = read_csv(out_csv)
    sums = df.groupby("region_id")["share"].sum().round(6)
    assert (sums == 1.0).all()

    state_index = json.loads(out_json.read_text())
    assert set(state_index["MD"]["regions"]) == {"TEST_A", "TEST_B"}
    assert "ALT_A" in state_index["NJ"]["regions"]


def test_equal_split_fallback(tmp_path):
    iso_yaml = write_iso(tmp_path)
    out_csv = tmp_path / "shares.csv"
    out_json = tmp_path / "state_index.json"

    run_script(
        tmp_path,
        [
            "--iso-yaml",
            str(iso_yaml),
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
        ],
    )

    df = read_csv(out_csv)
    region = df[df["region_id"] == "TEST_A"].set_index("state")["share"].to_dict()
    assert region == {"MD": 0.5, "PA": 0.5}


def test_eia_path(tmp_path):
    iso_yaml = write_iso(tmp_path)
    utility_map = tmp_path / "utility_map.csv"
    utility_map.write_text("utility_id_eia,region_id\n1,TEST_A\n1,TEST_B\n2,ALT_A\n")

    utility_override = tmp_path / "override.csv"
    utility_override.write_text(
        "utility_id_eia,region_id,weight\n1,TEST_A,0.25\n1,TEST_B,0.75\n2,ALT_A,1.0\n"
    )

    retail = tmp_path / "retail.csv"
    retail.write_text(
        "data_year,utility_id_eia,state,sales_mwh\n"
        "2023,1,MD,100\n"
        "2023,1,VA,300\n"
        "2023,2,NJ,50\n"
        "2023,2,PA,50\n"
    )

    customers = tmp_path / "customers.csv"
    customers.write_text(
        "data_year,utility_id_eia,state,customers\n"
        "2023,1,MD,10\n"
        "2023,1,VA,30\n"
        "2023,2,NJ,5\n"
        "2023,2,PA,5\n"
    )

    out_csv = tmp_path / "shares.csv"
    out_json = tmp_path / "state_index.json"

    run_script(
        tmp_path,
        [
            "--iso-yaml",
            str(iso_yaml),
            "--utility-map",
            str(utility_map),
            "--utility-zone-override",
            str(utility_override),
            "--eia861-retail",
            str(retail),
            "--eia861-customers",
            str(customers),
            "--eia861-year",
            "2023",
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
        ],
    )

    df = read_csv(out_csv)
    md_share = df[(df["region_id"] == "TEST_B") & (df["state"] == "VA")]["share"].iloc[0]
    assert md_share == 0.75


def test_filters(tmp_path):
    iso_yaml = write_iso(tmp_path)
    out_csv = tmp_path / "shares.csv"
    out_json = tmp_path / "state_index.json"

    run_script(
        tmp_path,
        [
            "--iso-yaml",
            str(iso_yaml),
            "--states-filter",
            "MD,PA",
            "--regions-filter",
            "TEST_*",
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
        ],
    )

    df = read_csv(out_csv)
    assert set(df["region_id"].unique()) == {"TEST_A"}
    assert set(df["state"].unique()) <= {"MD", "PA"}
