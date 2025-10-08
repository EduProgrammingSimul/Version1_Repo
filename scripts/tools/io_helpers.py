import os, hashlib
from typing import Dict, List, Tuple
import pandas as pd


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_columns(df: pd.DataFrame, cols: List[str], name=""):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {name or 'DataFrame'}: {missing}"
        )


def ensure_path(p: str):
    os.makedirs(p, exist_ok=True)


def write_csv(df: pd.DataFrame, path: str):
    ensure_path(os.path.dirname(path))
    df.to_csv(path, index=False)


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def list_timeseries_paths(
    results_root: str, controllers: List[str], scenarios: List[str]
) -> Dict[Tuple[str, str], str]:
    out = {}
    for c in controllers:
        for s in scenarios:
            p = os.path.join(results_root, "validation", c, s, "timeseries.csv")
            if os.path.exists(p):
                out[(c, s)] = p
    return out
