from __future__ import annotations

from pathlib import Path

from thesis_rrg.utils.io import ensure_dir, write_json



def export_predictions(df, path: str | Path):
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False)



def export_metrics(metrics: dict, path: str | Path):
    write_json(path, metrics)
