from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd



def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p



def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))



def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())



def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)



def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False, engine="pyarrow", compression="snappy")



def write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text)
