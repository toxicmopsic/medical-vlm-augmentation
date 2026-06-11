from __future__ import annotations

from pathlib import Path

import pandas as pd



def read_cache(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)



def write_cache(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False, engine="pyarrow", compression="snappy")
