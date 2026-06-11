from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import pandas as pd
from tqdm.auto import tqdm

from thesis_rrg.data.readers.section_extractors import extract_section
from thesis_rrg.utils.io import write_parquet



def study_report_path(files_root: Path, subject_id: int, study_id: int) -> Path:
    sid = str(subject_id)
    prefix = f"p{sid[:2]}"
    return files_root / prefix / f"p{sid}" / f"s{study_id}.txt"



def read_report_sections(files_root: Path, subject_id: int, study_id: int) -> Dict[str, str]:
    p = study_report_path(files_root, subject_id, study_id)
    if not p.exists():
        return {"indication": "", "findings": "", "impression": ""}

    txt = p.read_text(errors="ignore")
    indication = extract_section(txt, ["INDICATION", "HISTORY", "CLINICAL HISTORY", "REASON FOR EXAM", "EXAMINATION"])
    findings = extract_section(txt, ["FINDINGS"])
    impression = extract_section(txt, ["IMPRESSION"])

    return {"indication": indication, "findings": findings, "impression": impression}



def build_or_load_mimic_cache(mimic_root: Path, cache_dir: Path, cache_tag: str) -> pd.DataFrame:
    physionet_root = mimic_root / "physionet.org" / "files" / "mimic-cxr" / "2.0.0"
    files_root = physionet_root / "files"

    record_list_csv = physionet_root / "cxr-record-list.csv"
    split_csv = mimic_root / "mimic-cxr-2.0.0-split.csv"
    meta_csv = mimic_root / "mimic-cxr-2.0.0-metadata.csv"

    cache_dir.mkdir(parents=True, exist_ok=True)

    prepared_cache = cache_dir / f"mimic_prepared_frontal1perstudy_sections_{cache_tag}.parquet"
    clean_cache = cache_dir / f"mimic_prepared_frontal1perstudy_sections_{cache_tag}_clean.parquet"

    if clean_cache.exists():
        return pd.read_parquet(clean_cache)

    if prepared_cache.exists():
        df = pd.read_parquet(prepared_cache)
    else:
        df_rec = pd.read_csv(record_list_csv)
        df_split = pd.read_csv(split_csv)
        df_meta = pd.read_csv(meta_csv)

        df = df_rec.merge(df_split[["dicom_id", "split"]], on="dicom_id", how="inner")
        df = df.merge(df_meta[["dicom_id", "ViewPosition"]], on="dicom_id", how="left")

        df = df[df["ViewPosition"].isin(["AP", "PA"])].copy()
        df["dcm_path"] = df["path"].apply(lambda p: str(physionet_root / p))
        df = (
            df.sort_values(["subject_id", "study_id", "dicom_id"])
            .drop_duplicates("study_id", keep="first")
            .reset_index(drop=True)
        )

        sections = []
        for row in tqdm(df[["subject_id", "study_id"]].to_dict("records"), desc="Read reports"):
            sec = read_report_sections(files_root, int(row["subject_id"]), int(row["study_id"]))
            sections.append(sec)

        df["indication"] = [s["indication"] for s in sections]
        df["ref_findings"] = [s["findings"] for s in sections]
        df["ref_impression"] = [s["impression"] for s in sections]

        write_parquet(df, prepared_cache)

    if "dcm_path" not in df.columns or (len(df) > 0 and not Path(str(df.iloc[0]["dcm_path"])).exists()):
        df["dcm_path"] = df["path"].apply(lambda p: str(physionet_root / p))

    dcm_path = df["dcm_path"].astype(str)
    exists = dcm_path.map(os.path.exists)
    df = df.loc[exists].reset_index(drop=True)

    write_parquet(df, clean_cache)
    return df
