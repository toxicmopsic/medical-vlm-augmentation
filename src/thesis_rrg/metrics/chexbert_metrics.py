from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd



def maybe_run_chexbert_labeler(
    df_in: pd.DataFrame,
    text_col: str,
    out_dir: Path,
    tag: str,
    repo_dir: Path,
    cache_dir: Path,
) -> Optional[pd.DataFrame]:
    import subprocess

    label_py = repo_dir / "src" / "label.py"
    if not label_py.exists():
        return None

    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(
        hf_hub_download(
            repo_id="StanfordAIMI/RRG_scorers",
            filename="chexbert.pth",
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
    )

    in_csv = out_dir / f"chexbert_input_{tag}.csv"
    out_csv = out_dir / f"chexbert_output_{tag}.csv"
    pd.DataFrame({"Report Impression": df_in[text_col].fillna("").astype(str).tolist()}).to_csv(in_csv, index=False)

    cmd = ["python", str(label_py), "-c", str(ckpt_path), "-d", str(in_csv), "-o", str(out_csv)]
    subprocess.run(cmd, check=True)
    return pd.read_csv(out_csv)



def compute_chexbert_extra_positives(pred_labels: pd.DataFrame, ref_labels: pd.DataFrame) -> dict[str, Any]:
    drop_cols = {c for c in pred_labels.columns if c.lower().strip() in {"report impression", "impression", "report"}}
    label_cols = [c for c in pred_labels.columns if c not in drop_cols]

    def pos_set(row: pd.Series) -> set:
        s = set()
        for c in label_cols:
            try:
                v = int(row[c])
            except Exception:
                continue
            if v == 1:
                s.add(c)
        return s

    n = min(len(pred_labels), len(ref_labels))
    extras = []
    per_label = {c: 0 for c in label_cols}

    for i in range(n):
        sp = pos_set(pred_labels.iloc[i])
        sr = pos_set(ref_labels.iloc[i])
        extra = sp - sr
        extras.append(len(extra))
        for c in extra:
            per_label[c] += 1

    h_cxp = float(np.mean(extras)) if extras else 0.0
    per_label_rate = {c: per_label[c] / max(n, 1) for c in label_cols}

    return {"n": n, "H_cxp": h_cxp, "extra_positive_rate": per_label_rate}
