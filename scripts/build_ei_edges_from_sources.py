"""Phase 1 ETL: Build EI edges CSV from raw sources.
Expected raw drops in data_raw/:
- ferc715/ (EHV topology, >=100kV)
- nerc_ttc/seasonal_ttc.csv
- iso_docs/{pjm,nyiso,iso_ne,miso,spp,tva}/*.csv
- eia930/interchange.csv (hourly)

Output: input/engine/transmission/ei_edges.csv
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

def _read_optional(path: Path, **kwargs):
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()

def main(project_root: str = "."):
    root = Path(project_root)
    data_raw = root / "data_raw"
    out = root / "input" / "engine" / "transmission"
    out.mkdir(parents=True, exist_ok=True)

    # TODO: Replace placeholders with real parsers and joins
    frames = []
    for iso in ["pjm","nyiso","iso_ne","miso","spp","tva"]:
        p = data_raw / "iso_docs" / iso / "interfaces.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    base = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[
        "from_zone","to_zone","season","limit_mw","wheel_cost_per_mwh",
        "contracted_reserve_mw","loss_pct","rule_tag",
        "effective_start","effective_end","source_doc","build_timestamp"
    ])

    base["season"] = base.get("season", "annual")
    base["wheel_cost_per_mwh"] = base.get("wheel_cost_per_mwh", 0.0)
    base["contracted_reserve_mw"] = base.get("contracted_reserve_mw", 0.0)
    base["loss_pct"] = base.get("loss_pct", 0.0)
    base["build_timestamp"] = datetime.utcnow().isoformat()

    # Ensure reverse rows
    rev = base.rename(columns={"from_zone":"to_zone","to_zone":"from_zone"})
    all_edges = pd.concat([base, rev], ignore_index=True).drop_duplicates(subset=["from_zone","to_zone","season"])
    all_edges.to_csv(out / "ei_edges.csv", index=False)
    print("wrote", out / "ei_edges.csv")

if __name__ == "__main__":
    main()
