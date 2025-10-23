import pandas as pd
from engine.data_paths import data_root

def load_wacc_csv() -> pd.DataFrame:
    p = data_root() / "finance" / "atb_wacc.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def load_capex_csv() -> pd.DataFrame:
    p = data_root() / "finance" / "atb_capex.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()
