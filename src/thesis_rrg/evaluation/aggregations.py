from __future__ import annotations

import pandas as pd



def build_predictions_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    if "ref_full" not in df.columns:
        df["ref_full"] = (df["ref_findings"].fillna("") + " " + df["ref_impression"].fillna("")).str.strip()
    if "pred_full" not in df.columns and "raw_pred" in df.columns:
        df["pred_full"] = df["raw_pred"].fillna("").astype(str).str.strip()

    return df
