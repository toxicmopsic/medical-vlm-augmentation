from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from tqdm.auto import tqdm

from thesis_rrg.utils.config_utils import resolve_experiment_data_value
from thesis_rrg.utils.io import ensure_dir, write_json
from thesis_rrg.utils.torch_utils import resolve_torch_dtype


_WS_RE = re.compile(r"\s+")
_DEID_SEX_RE = re.compile(r"(?<![A-Za-z0-9])_+\s*(?P<sex>[MF])\b", re.IGNORECASE)
_FEMALE_WORD_RE = re.compile(r"\b(?:female|woman|women|lady|girl)\b", re.IGNORECASE)
_MALE_WORD_RE = re.compile(r"\b(?:male|man|men|gentleman|boy)\b", re.IGNORECASE)


def _cfg(cfg: DictConfig, key: str, default: Any = None, sentinels: Iterable[Any] = ("", "none", None)) -> Any:
    return resolve_experiment_data_value(cfg, key, default=default, sentinels=sentinels)


def _as_int(cfg: DictConfig, key: str, default: int) -> int:
    return int(_cfg(cfg, key, default))


def _as_float(cfg: DictConfig, key: str, default: float) -> float:
    return float(_cfg(cfg, key, default))


def _as_bool(cfg: DictConfig, key: str, default: bool) -> bool:
    raw = _cfg(cfg, key, default)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw)


def _id_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "ok", "on"}
    return bool(value)


def _write_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False)


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def clean_generation_text(text: Any) -> str:
    return _WS_RE.sub(" ", "" if text is None else str(text).replace("\r", " ").replace("\n", " ")).strip()


def view_phrase(view_position: Any) -> str:
    view = str(view_position).strip().upper()
    if view == "AP":
        return "AP frontal chest radiograph."
    if view == "PA":
        return "PA frontal chest radiograph."
    raise ValueError(f"Unsupported ViewPosition for H4 generation: {view_position!r}")


def patient_sex_phrase(patient_sex: Any) -> str:
    sex = str(patient_sex).strip().upper()
    if sex in {"F", "FEMALE", "WOMAN"}:
        return "Adult female."
    if sex in {"M", "MALE", "MAN"}:
        return "Adult male."
    return "Adult patient."


def build_roentgen_prompt(impression: Any, view_position: Any, patient_sex: Any = None) -> str:
    cleaned = clean_generation_text(impression)
    return f"{patient_sex_phrase(patient_sex)} {view_phrase(view_position)} {cleaned}".strip()


def _field_names_from_cfg(cfg: DictConfig, key: str, default: list[str]) -> list[str]:
    raw = _cfg(cfg, key, default)
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return [str(x).strip() for x in raw if str(x).strip()]


def infer_patient_sex_from_texts(texts: list[tuple[str, Any]]) -> dict[str, str]:
    matches: list[dict[str, str]] = []
    for source, raw_text in texts:
        text = clean_generation_text(raw_text)
        if not text:
            continue

        for match in _DEID_SEX_RE.finditer(text):
            sex = match.group("sex").upper()
            matches.append({"patient_sex": sex, "source": source, "evidence": match.group(0)})
        for match in _FEMALE_WORD_RE.finditer(text):
            matches.append({"patient_sex": "F", "source": source, "evidence": match.group(0)})
        for match in _MALE_WORD_RE.finditer(text):
            matches.append({"patient_sex": "M", "source": source, "evidence": match.group(0)})

    unique = {m["patient_sex"] for m in matches}
    if len(unique) == 1:
        first = matches[0]
        sex = first["patient_sex"]
        return {
            "patient_sex": sex,
            "patient_sex_label": "female" if sex == "F" else "male",
            "patient_sex_source": first["source"],
            "patient_sex_evidence": first["evidence"],
            "patient_sex_status": "ok",
            "roentgen_sex_phrase": patient_sex_phrase(sex),
        }
    if len(unique) > 1:
        evidence = "; ".join(f"{m['source']}={m['evidence']}" for m in matches[:8])
        return {
            "patient_sex": "unknown",
            "patient_sex_label": "unknown",
            "patient_sex_source": "conflict",
            "patient_sex_evidence": evidence,
            "patient_sex_status": "conflict",
            "roentgen_sex_phrase": patient_sex_phrase(None),
        }
    return {
        "patient_sex": "unknown",
        "patient_sex_label": "unknown",
        "patient_sex_source": "",
        "patient_sex_evidence": "",
        "patient_sex_status": "not_found",
        "roentgen_sex_phrase": patient_sex_phrase(None),
    }


def _prepare_fixed_source_manifest(cfg: DictConfig, fixed_df: pd.DataFrame) -> pd.DataFrame:
    work = fixed_df.copy()
    for col in ["ref_findings", "ref_impression", "indication"]:
        if col in work:
            work[col] = work[col].fillna("").astype(str)
        else:
            work[col] = ""

    if "study_id" not in work.columns:
        raise ValueError("Fixed H4 source manifest must contain study_id.")
    if "ViewPosition" not in work.columns:
        raise ValueError("Fixed H4 source manifest must contain ViewPosition.")
    if "ref_impression" not in work.columns or work["ref_impression"].fillna("").astype(str).str.strip().eq("").all():
        if "raw_impression" in work.columns:
            work["ref_impression"] = work["raw_impression"].fillna("").astype(str)
    if "raw_impression" not in work.columns:
        work["raw_impression"] = work["ref_impression"].map(clean_generation_text)

    sex_text_fields = _field_names_from_cfg(cfg, "h4_sex_text_fields", ["indication", "ref_findings", "ref_impression"])
    if "patient_sex" not in work.columns:
        sex_rows = []
        for row in work.to_dict("records"):
            sex_rows.append(infer_patient_sex_from_texts([(field, row.get(field, "")) for field in sex_text_fields]))
        work = pd.concat([work, pd.DataFrame(sex_rows, index=work.index)], axis=1)

    for col in ["patient_sex_label", "patient_sex_status", "patient_sex_source", "patient_sex_evidence"]:
        if col not in work.columns:
            inferred = []
            for row in work.to_dict("records"):
                inferred.append(infer_patient_sex_from_texts([(field, row.get(field, "")) for field in sex_text_fields]))
            inferred_df = pd.DataFrame(inferred, index=work.index)
            for inferred_col in ["patient_sex", "patient_sex_label", "patient_sex_status", "patient_sex_source", "patient_sex_evidence"]:
                if inferred_col not in work.columns and inferred_col in inferred_df.columns:
                    work[inferred_col] = inferred_df[inferred_col]
            break

    require_sex = _as_bool(cfg, "h4_require_patient_sex", True)
    if require_sex:
        work = work[work["patient_sex"].astype(str).str.upper().isin(["M", "F"])].copy()

    use_sex_in_prompt = _as_bool(cfg, "h4_use_patient_sex_in_prompt", True)
    work["patient_sex"] = work["patient_sex"].astype(str).str.upper()
    work["h4_use_patient_sex_in_prompt"] = bool(use_sex_in_prompt)
    work["roentgen_sex_phrase"] = [
        patient_sex_phrase(sex) if use_sex_in_prompt else patient_sex_phrase(None)
        for sex in work["patient_sex"]
    ]
    work["raw_impression"] = work["raw_impression"].map(clean_generation_text)
    work["cleaned_impression"] = work["raw_impression"]
    work["roentgen_prompt"] = [
        build_roentgen_prompt(impression, view, sex if use_sex_in_prompt else None)
        for impression, view, sex in zip(work["cleaned_impression"], work["ViewPosition"], work["patient_sex"])
    ]
    work["source_key"] = work["study_id"].map(_id_str)
    work["h4_source_order"] = np.arange(len(work), dtype=np.int64)
    return work.reset_index(drop=True)


def token_count(tokenizer: Any, text: str) -> int:
    enc = tokenizer(text, add_special_tokens=True, truncation=False)
    ids = enc["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return int(len(ids))


def load_roentgen_tokenizer(cfg: DictConfig):
    from transformers import AutoTokenizer

    roentgen_model_id = str(_cfg(cfg, "h4_roentgen_model_id", "stanfordmimi/RoentGen-v2"))
    text_encoder_id = str(_cfg(cfg, "h4_roentgen_text_encoder_id", roentgen_model_id))
    cache_dir = _cfg(cfg, "h4_hf_cache_dir", None)
    kwargs = {"cache_dir": str(cache_dir)} if cache_dir not in (None, "") else {}

    model_ids = []
    for model_id in [roentgen_model_id, text_encoder_id]:
        if model_id and model_id not in model_ids:
            model_ids.append(model_id)

    errors = []
    for model_id in model_ids:
        for subfolder in ["tokenizer", None]:
            try:
                location = f"{model_id}/{subfolder}" if subfolder is not None else model_id
                print(f"[H4] Loading RoentGen tokenizer from {location}", flush=True)
                if subfolder is None:
                    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=subfolder, **kwargs)
                print(f"[H4] Loaded RoentGen tokenizer from {location}", flush=True)
                return tokenizer
            except Exception as exc:
                errors.append(f"{model_id} subfolder={subfolder}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Could not load RoentGen-v2 tokenizer for H4 source token filtering. "
        "Tried:\n" + "\n".join(errors)
    )


def load_medsiglip_tokenizer(cfg: DictConfig):
    from transformers import AutoProcessor, AutoTokenizer

    model_id = str(_cfg(cfg, "h4_medsiglip_model_id", "google/medsiglip-448"))
    cache_dir = _cfg(cfg, "h4_hf_cache_dir", None)
    kwargs = {"cache_dir": str(cache_dir)} if cache_dir not in (None, "") else {}
    try:
        print(f"[H4] Loading MedSigLIP tokenizer via processor from {model_id}", flush=True)
        processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            print(f"[H4] Loaded MedSigLIP tokenizer from {model_id}", flush=True)
            return tokenizer
    except Exception:
        pass
    print(f"[H4] Loading MedSigLIP tokenizer directly from {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    print(f"[H4] Loaded MedSigLIP tokenizer from {model_id}", flush=True)
    return tokenizer


def h4_manifest_paths(cfg: DictConfig) -> dict[str, Path]:
    root_dir = Path.cwd()
    if "paths" in cfg and "root_dir" in cfg.paths:
        root_dir = Path(str(cfg.paths.root_dir))
    out_dir = Path(str(_cfg(cfg, "h4_out_dir", root_dir / "H4_synthetic_roentgen_v2"))).expanduser()
    artifact_name = str(_cfg(cfg, "h4_artifact_name", "", sentinels=("none", None))).strip()
    if artifact_name:
        safe_artifact_name = re.sub(r"[^A-Za-z0-9_.=-]+", "_", artifact_name).strip("._")
        if not safe_artifact_name:
            raise ValueError(f"Invalid H4 artifact name: {artifact_name!r}")
        artifact_dir_cfg = _cfg(cfg, "h4_artifact_dir", None)
        if artifact_dir_cfg in (None, ""):
            artifacts_dir = Path(str(_cfg(cfg, "h4_artifacts_dir", out_dir / "artifacts"))).expanduser()
            artifact_dir = artifacts_dir / safe_artifact_name
        else:
            artifact_dir = Path(str(artifact_dir_cfg)).expanduser()
        manifests_dir = artifact_dir / "manifests"
        images_raw_dir = artifact_dir / "images_raw"
    else:
        safe_artifact_name = ""
        artifact_dir = out_dir
        manifests_dir = Path(str(_cfg(cfg, "h4_manifests_dir", out_dir / "manifests"))).expanduser()
        images_raw_dir = Path(str(_cfg(cfg, "h4_images_raw_dir", out_dir / "images_raw"))).expanduser()
    return {
        "out_dir": out_dir,
        "artifact_name": safe_artifact_name,
        "artifact_dir": artifact_dir,
        "manifests_dir": manifests_dir,
        "images_raw_dir": images_raw_dir,
        "eligible_pool": manifests_dir / "eligible_train_pool.csv",
        "generation_prompts": manifests_dir / "generation_prompts.csv",
        "candidates": manifests_dir / "synthetic_candidates_manifest.csv",
        "qc": manifests_dir / "synthetic_qc_manifest.csv",
        "selected": manifests_dir / "synthetic_selected_manifest.csv",
        "source_status": manifests_dir / "source_study_status.csv",
        "train_real_baseline": manifests_dir / "train_real_baseline_manifest.csv",
        "train_synthetic_pretrain": manifests_dir / "train_synthetic_pretrain_manifest.csv",
        "train_real_finetune": manifests_dir / "train_real_finetune_manifest.csv",
        "summary": manifests_dir / "h4_prepare_summary.json",
    }


def build_h4_source_manifest(cfg: DictConfig, df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    paths = h4_manifest_paths(cfg)
    fixed_source_path = _cfg(cfg, "h4_fixed_source_manifest_path", "", sentinels=("none", None))
    if paths["eligible_pool"].exists() and not force:
        print(f"[H4] Loading existing eligible source manifest: {paths['eligible_pool']}", flush=True)
        out = pd.read_csv(paths["eligible_pool"])
        require_sex = _as_bool(cfg, "h4_require_patient_sex", True)
        use_sex_in_prompt = _as_bool(cfg, "h4_use_patient_sex_in_prompt", True)
        sex_cols = {"patient_sex", "patient_sex_status", "roentgen_sex_phrase"}
        has_sex_cols = sex_cols.issubset(set(out.columns))
        if "h4_use_patient_sex_in_prompt" in out.columns:
            prompt_flags = {bool(_truthy(x)) for x in out["h4_use_patient_sex_in_prompt"].dropna().unique()}
            prompt_schema_matches = prompt_flags == {bool(use_sex_in_prompt)}
        else:
            prompt_schema_matches = not use_sex_in_prompt
        if (not require_sex or has_sex_cols) and prompt_schema_matches:
            print(f"[H4] Loaded eligible source rows={len(out)}", flush=True)
            return out
        print("[H4] Existing source manifest does not match current patient-sex prompt schema; rebuilding it", flush=True)

    print("[H4] Building eligible source manifest", flush=True)
    roentgen_tokenizer = load_roentgen_tokenizer(cfg)
    medsiglip_tokenizer = load_medsiglip_tokenizer(cfg)

    if fixed_source_path not in (None, ""):
        fixed_path = Path(str(fixed_source_path)).expanduser()
        if not fixed_path.exists():
            raise FileNotFoundError(f"H4 fixed source manifest not found: {fixed_path}")
        print(f"[H4] Loading fixed source manifest: {fixed_path}", flush=True)
        work = _prepare_fixed_source_manifest(cfg, pd.read_csv(fixed_path))
        print(f"[H4] Fixed source rows before token filters: {len(work)}", flush=True)
        work["roentgen_prompt_tokens"] = [
            token_count(roentgen_tokenizer, text)
            for text in tqdm(work["roentgen_prompt"].tolist(), desc="H4 fixed RoentGen prompt token counts")
        ]
        work["medsiglip_impression_tokens"] = [
            token_count(medsiglip_tokenizer, text)
            for text in tqdm(work["raw_impression"].tolist(), desc="H4 fixed MedSigLIP impression token counts")
        ]
        max_rg_tokens = _as_int(cfg, "h4_roentgen_max_tokens", 77)
        max_msl_tokens = _as_int(cfg, "h4_medsiglip_max_tokens", 64)
        before_tokens = len(work)
        work = work[
            (work["roentgen_prompt_tokens"] <= max_rg_tokens)
            & (work["medsiglip_impression_tokens"] <= max_msl_tokens)
        ].copy()
        print(f"[H4] Fixed source rows after token filters: {len(work)} / {before_tokens}", flush=True)

        prompt_cols = [
            "h4_source_order",
            "source_key",
            "subject_id",
            "study_id",
            "dicom_id",
            "ViewPosition",
            "patient_sex",
            "patient_sex_label",
            "patient_sex_status",
            "patient_sex_source",
            "patient_sex_evidence",
            "roentgen_sex_phrase",
            "h4_use_patient_sex_in_prompt",
            "roentgen_prompt",
            "roentgen_prompt_tokens",
            "medsiglip_impression_tokens",
            "raw_impression",
        ]
        _write_csv(work, paths["eligible_pool"])
        _write_csv(work[[c for c in prompt_cols if c in work.columns]], paths["generation_prompts"])
        print(f"[H4] Wrote fixed eligible source manifest: {paths['eligible_pool']}", flush=True)
        print(f"[H4] Wrote fixed generation prompt manifest: {paths['generation_prompts']}", flush=True)
        return work

    source_split = str(_cfg(cfg, "h4_source_split", "train")).strip().lower()
    if source_split == "valid":
        source_split = "validate"

    work = df.copy()
    print(f"[H4] Source rows before filters: {len(work)}", flush=True)
    for col in ["ref_findings", "ref_impression", "indication"]:
        if col in work:
            work[col] = work[col].fillna("").astype(str)

    work = work[work["split"].astype(str).str.lower() == source_split].copy()
    print(f"[H4] After split={source_split}: {len(work)}", flush=True)
    work = work[work["ViewPosition"].astype(str).str.upper().isin(["AP", "PA"])].copy()
    print(f"[H4] After AP/PA filter: {len(work)}", flush=True)
    # work = work[work["dcm_path"].astype(str).map(os.path.exists)].copy()
    print(f"[H4] After DICOM exists filter: {len(work)}", flush=True)
    work = work[(work["ref_findings"].str.strip() != "") & (work["ref_impression"].str.strip() != "")].copy()
    print(f"[H4] After non-empty FINDINGS/IMPRESSION filter: {len(work)}", flush=True)

    work = (
        work.sort_values(["subject_id", "study_id", "dicom_id"])
        .drop_duplicates("study_id", keep="first")
        .reset_index(drop=True)
    )
    print(f"[H4] After one-image-per-study filter: {len(work)}", flush=True)

    sex_text_fields = _field_names_from_cfg(cfg, "h4_sex_text_fields", ["indication", "ref_findings", "ref_impression"])
    sex_rows = []
    for row in work.to_dict("records"):
        sex_rows.append(infer_patient_sex_from_texts([(field, row.get(field, "")) for field in sex_text_fields]))
    sex_df = pd.DataFrame(sex_rows, index=work.index)
    work = pd.concat([work, sex_df], axis=1)
    require_sex = _as_bool(cfg, "h4_require_patient_sex", True)
    if require_sex:
        work = work[work["patient_sex"].isin(["M", "F"])].copy()
        print(f"[H4] After patient sex extraction filter: {len(work)}", flush=True)
    else:
        print(
            "[H4] Patient sex extraction counts: "
            f"{work['patient_sex_status'].value_counts(dropna=False).to_dict()}",
            flush=True,
        )

    use_sex_in_prompt = _as_bool(cfg, "h4_use_patient_sex_in_prompt", True)
    work["h4_use_patient_sex_in_prompt"] = bool(use_sex_in_prompt)
    work["roentgen_sex_phrase"] = [
        patient_sex_phrase(sex) if use_sex_in_prompt else patient_sex_phrase(None)
        for sex in work["patient_sex"]
    ]
    work["raw_impression"] = work["ref_impression"].map(clean_generation_text)
    work["cleaned_impression"] = work["raw_impression"]
    work["roentgen_prompt"] = [
        build_roentgen_prompt(impression, view, sex if use_sex_in_prompt else None)
        for impression, view, sex in zip(work["cleaned_impression"], work["ViewPosition"], work["patient_sex"])
    ]

    work["roentgen_prompt_tokens"] = [
        token_count(roentgen_tokenizer, text)
        for text in tqdm(work["roentgen_prompt"].tolist(), desc="H4 RoentGen prompt token counts")
    ]
    work["medsiglip_impression_tokens"] = [
        token_count(medsiglip_tokenizer, text)
        for text in tqdm(work["raw_impression"].tolist(), desc="H4 MedSigLIP impression token counts")
    ]

    max_rg_tokens = _as_int(cfg, "h4_roentgen_max_tokens", 77)
    max_msl_tokens = _as_int(cfg, "h4_medsiglip_max_tokens", 64)
    work = work[
        (work["roentgen_prompt_tokens"] <= max_rg_tokens)
        & (work["medsiglip_impression_tokens"] <= max_msl_tokens)
    ].copy()
    print(f"[H4] After token filters: {len(work)}", flush=True)

    order = str(_cfg(cfg, "h4_case_order", "random")).strip().lower()
    if order == "random":
        seed_default = int(cfg.get("seed", 0))
        work = work.sample(frac=1.0, random_state=int(_cfg(cfg, "h4_source_seed", seed_default))).reset_index(drop=True)
    elif order in {"sorted", "sequential", "as_is"}:
        work = work.reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported h4_case_order={order!r}. Use random or sorted.")

    work["h4_source_order"] = np.arange(len(work), dtype=np.int64)
    work["source_key"] = work["study_id"].map(_id_str)

    prompt_cols = [
        "h4_source_order",
        "source_key",
        "subject_id",
        "study_id",
        "dicom_id",
        "ViewPosition",
        "patient_sex",
        "patient_sex_label",
        "patient_sex_status",
        "patient_sex_source",
        "patient_sex_evidence",
        "roentgen_sex_phrase",
        "h4_use_patient_sex_in_prompt",
        "roentgen_prompt",
        "roentgen_prompt_tokens",
        "medsiglip_impression_tokens",
        "raw_impression",
    ]
    _write_csv(work, paths["eligible_pool"])
    _write_csv(work[[c for c in prompt_cols if c in work.columns]], paths["generation_prompts"])
    print(f"[H4] Wrote eligible source manifest: {paths['eligible_pool']}", flush=True)
    print(f"[H4] Wrote generation prompt manifest: {paths['generation_prompts']}", flush=True)
    return work


def candidate_relative_path(row: pd.Series | dict, candidate_idx: int, seed: int) -> Path:
    subject_id = _id_str(row["subject_id"])
    study_id = _id_str(row["study_id"])
    prefix = f"p{subject_id[:2]}"
    return Path(prefix) / f"p{subject_id}" / f"s{study_id}" / f"cand_{candidate_idx:02d}_seed{seed}.png"


def _candidate_seed(cfg: DictConfig, source_order: int, candidate_idx: int) -> int:
    base = _as_int(cfg, "h4_generation_seed_base", 133700)
    k = _as_int(cfg, "h4_candidates_per_study", 2)
    return int(base + source_order * k + candidate_idx)


def technical_qc_image(
    image_path: str | Path,
    expected_size: int = 512,
    min_std: float = 0.01,
    min_p_range: float = 0.05,
    max_near_black_frac: float = 0.995,
    max_near_white_frac: float = 0.995,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "image_opens": False,
        "expected_size_ok": False,
        "rgb_ok": False,
        "technical_ok": False,
        "technical_reason": "not_evaluated",
        "width": None,
        "height": None,
        "mean_intensity": None,
        "std_intensity": None,
        "p01_intensity": None,
        "p99_intensity": None,
        "p99_p01_range": None,
        "near_black_frac": None,
        "near_white_frac": None,
    }

    try:
        img = Image.open(Path(image_path))
        out["image_opens"] = True
        img = img.convert("RGB")
        out["rgb_ok"] = True
    except Exception as exc:
        out["technical_reason"] = f"open_failed:{type(exc).__name__}"
        return out

    width, height = img.size
    out["width"] = int(width)
    out["height"] = int(height)
    out["expected_size_ok"] = bool(expected_size <= 0 or (width == expected_size and height == expected_size))
    if not out["expected_size_ok"]:
        out["technical_reason"] = "unexpected_size"
        return out

    arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    mean = float(arr.mean())
    std = float(arr.std())
    p01 = float(np.percentile(arr, 1))
    p99 = float(np.percentile(arr, 99))
    p_range = float(p99 - p01)
    near_black = float((arr <= 0.02).mean())
    near_white = float((arr >= 0.98).mean())

    out.update(
        {
            "mean_intensity": mean,
            "std_intensity": std,
            "p01_intensity": p01,
            "p99_intensity": p99,
            "p99_p01_range": p_range,
            "near_black_frac": near_black,
            "near_white_frac": near_white,
        }
    )

    if near_black >= max_near_black_frac:
        out["technical_reason"] = "almost_all_black"
    elif near_white >= max_near_white_frac:
        out["technical_reason"] = "almost_all_white"
    elif std < min_std:
        out["technical_reason"] = "low_contrast_std"
    elif p_range < min_p_range:
        out["technical_reason"] = "low_percentile_range"
    else:
        out["technical_ok"] = True
        out["technical_reason"] = "ok"
    return out


class MedSigLIPScorer:
    def __init__(self, cfg: DictConfig):
        from transformers import AutoProcessor

        self.model_id = str(_cfg(cfg, "h4_medsiglip_model_id", "google/medsiglip-448"))
        self.device = str(_cfg(cfg, "h4_medsiglip_device", _cfg(cfg, "h4_device", "cuda:0" if torch.cuda.is_available() else "cpu")))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        self.max_tokens = _as_int(cfg, "h4_medsiglip_max_tokens", 64)
        cache_dir = _cfg(cfg, "h4_hf_cache_dir", None)
        kwargs = {"cache_dir": str(cache_dir)} if cache_dir not in (None, "") else {}
        dtype = resolve_torch_dtype(str(_cfg(cfg, "h4_medsiglip_dtype", "auto")))
        if dtype is not None and self.device.startswith("cuda"):
            kwargs["torch_dtype"] = dtype

        print(f"[H4] Loading MedSigLIP QC model {self.model_id} on {self.device}", flush=True)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            **({"cache_dir": kwargs["cache_dir"]} if "cache_dir" in kwargs else {}),
        )
        try:
            from transformers import AutoModelForZeroShotImageClassification

            model = AutoModelForZeroShotImageClassification.from_pretrained(self.model_id, **kwargs)
        except Exception:
            from transformers import SiglipModel

            model = SiglipModel.from_pretrained(self.model_id, **kwargs)
        self.model = model.eval().to(self.device)
        print(f"[H4] Loaded MedSigLIP QC model {self.model_id} on {self.device}", flush=True)

    @torch.inference_mode()
    def score_pairs(self, image_paths: list[str], texts: list[str]) -> list[dict[str, float]]:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(
            text=[clean_generation_text(t) for t in texts],
            images=images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
        ).to(self.device)
        outputs = self.model(**inputs)

        logits = outputs.logits_per_image
        if logits.ndim == 2:
            pair_logits = torch.diagonal(logits, offset=0)
        else:
            pair_logits = logits.reshape(-1)
        scores = torch.sigmoid(pair_logits)

        image_embeds = getattr(outputs, "image_embeds", None)
        text_embeds = getattr(outputs, "text_embeds", None)
        if image_embeds is not None and text_embeds is not None:
            image_embeds = F.normalize(image_embeds, dim=-1)
            text_embeds = F.normalize(text_embeds, dim=-1)
            cosine = (image_embeds * text_embeds).sum(dim=-1)
        else:
            cosine = torch.full_like(scores, float("nan"))

        return [
            {
                "score": float(score),
                "logit": float(logit),
                "cosine": float(cos),
            }
            for score, logit, cos in zip(
                scores.detach().cpu().tolist(),
                pair_logits.detach().cpu().tolist(),
                cosine.detach().cpu().tolist(),
            )
        ]

    @torch.inference_mode()
    def score(self, image_paths: list[str], texts: list[str]) -> list[float]:
        return [x["score"] for x in self.score_pairs(image_paths, texts)]


def select_medsiglip_score(score_info: dict[str, float], score_kind: str) -> float:
    kind = score_kind.strip().lower()
    if kind in {"sigmoid", "sigmoid_logit", "prob", "probability", "pair_probability"}:
        return float(score_info["score"])
    if kind in {"logit", "logits", "pair_logit"}:
        return float(score_info["logit"])
    if kind in {"cosine", "embedding_cosine", "raw_cosine"}:
        return float(score_info["cosine"])
    raise ValueError(
        "Unsupported h4_medsiglip_score_kind="
        f"{score_kind!r}. Use one of: sigmoid_logit, logit, cosine."
    )


def _load_roentgen_pipeline(cfg: DictConfig):
    from diffusers import DiffusionPipeline

    model_id = str(_cfg(cfg, "h4_roentgen_model_id", "stanfordmimi/RoentGen-v2"))
    cache_dir = _cfg(cfg, "h4_hf_cache_dir", None)
    dtype = resolve_torch_dtype(str(_cfg(cfg, "h4_roentgen_dtype", "bf16")))
    kwargs: dict[str, Any] = {}
    if cache_dir not in (None, ""):
        kwargs["cache_dir"] = str(cache_dir)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    print(f"[H4] Loading RoentGen-v2 pipeline {model_id}", flush=True)
    pipe = DiffusionPipeline.from_pretrained(model_id, **kwargs)
    device = str(_cfg(cfg, "h4_device", "cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    print(f"[H4] Moving RoentGen-v2 pipeline to {device}", flush=True)
    pipe = pipe.to(device)
    if _as_bool(cfg, "h4_enable_xformers", True):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    print(f"[H4] RoentGen-v2 pipeline is ready on {device}", flush=True)
    return pipe, device


def _generate_candidate_images(cfg: DictConfig, pipe: Any, device: str, row: pd.Series, missing: list[tuple[int, int, Path]]) -> list[dict[str, Any]]:
    if not missing:
        return []

    prompts = [str(row["roentgen_prompt"])] * len(missing)
    generator_device = device if device.startswith("cuda") else "cpu"
    generators = [torch.Generator(device=generator_device).manual_seed(seed) for _, seed, _ in missing]
    kwargs = {
        "num_inference_steps": _as_int(cfg, "h4_roentgen_num_inference_steps", 75),
        "guidance_scale": _as_float(cfg, "h4_roentgen_guidance_scale", 3.0),
        "height": _as_int(cfg, "h4_roentgen_resolution", 512),
        "width": _as_int(cfg, "h4_roentgen_resolution", 512),
        "generator": generators,
    }
    images = pipe(prompts, **kwargs).images

    records = []
    for image, (candidate_idx, seed, out_path) in zip(images, missing):
        ensure_dir(out_path.parent)
        image.save(out_path)
        records.append(
            {
                "source_key": _id_str(row["study_id"]),
                "subject_id": _id_str(row["subject_id"]),
                "study_id": _id_str(row["study_id"]),
                "dicom_id": _id_str(row["dicom_id"]),
                "ViewPosition": str(row["ViewPosition"]),
                "patient_sex": str(row.get("patient_sex", "")),
                "patient_sex_label": str(row.get("patient_sex_label", "")),
                "patient_sex_status": str(row.get("patient_sex_status", "")),
                "patient_sex_source": str(row.get("patient_sex_source", "")),
                "patient_sex_evidence": str(row.get("patient_sex_evidence", "")),
                "roentgen_sex_phrase": str(row.get("roentgen_sex_phrase", "")),
                "h4_use_patient_sex_in_prompt": bool(row.get("h4_use_patient_sex_in_prompt", False)),
                "h4_source_order": int(row["h4_source_order"]),
                "candidate_idx": int(candidate_idx),
                "candidate_seed": int(seed),
                "image_path": str(out_path),
                "roentgen_prompt": str(row["roentgen_prompt"]),
                "raw_impression": str(row["raw_impression"]),
                "generation_status": "generated",
                "roentgen_model_id": str(_cfg(cfg, "h4_roentgen_model_id", "stanfordmimi/RoentGen-v2")),
                "guidance_scale": _as_float(cfg, "h4_roentgen_guidance_scale", 3.0),
                "num_inference_steps": _as_int(cfg, "h4_roentgen_num_inference_steps", 75),
                "resolution": _as_int(cfg, "h4_roentgen_resolution", 512),
            }
        )
    return records


def _qc_candidates_for_study(
    cfg: DictConfig,
    row: pd.Series,
    candidate_rows: list[dict[str, Any]],
    scorer: MedSigLIPScorer | None,
    existing_qc: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    use_technical = _as_bool(cfg, "h4_use_technical_qc", True)
    use_medsiglip = _as_bool(cfg, "h4_use_medsiglip_qc", True)
    min_sim = _as_float(cfg, "h4_medsiglip_min_sim", 0.00027)
    score_kind = str(_cfg(cfg, "h4_medsiglip_score_kind", "sigmoid_logit"))
    expected_size = _as_int(cfg, "h4_expected_raw_size", _as_int(cfg, "h4_roentgen_resolution", 512))

    out = []
    missing_msl: list[dict[str, Any]] = []
    for cand in candidate_rows:
        key = (_id_str(cand["study_id"]), int(cand["candidate_idx"]))
        cached = existing_qc.get(key)
        if cached is not None:
            out.append(cached)
            continue

        record = dict(cand)
        if use_technical:
            record.update(
                technical_qc_image(
                    cand["image_path"],
                    expected_size=expected_size,
                    min_std=_as_float(cfg, "h4_technical_min_std", 0.01),
                    min_p_range=_as_float(cfg, "h4_technical_min_p_range", 0.05),
                    max_near_black_frac=_as_float(cfg, "h4_technical_max_near_black_frac", 0.995),
                    max_near_white_frac=_as_float(cfg, "h4_technical_max_near_white_frac", 0.995),
                )
            )
        else:
            record.update({"technical_ok": True, "technical_reason": "disabled"})

        record["medsiglip_text"] = str(row["raw_impression"])
        record["medsiglip_sim"] = None
        record["medsiglip_score_kind"] = score_kind
        record["medsiglip_sigmoid"] = None
        record["medsiglip_logit"] = None
        record["medsiglip_cosine"] = None
        record["medsiglip_ok"] = True if not use_medsiglip else False
        record["medsiglip_reason"] = "disabled" if not use_medsiglip else "not_evaluated"
        if use_medsiglip and bool(record["technical_ok"]):
            missing_msl.append(record)
        out.append(record)

    if use_medsiglip and missing_msl:
        if scorer is None:
            raise RuntimeError("H4 MedSigLIP QC is enabled but scorer is not initialized.")
        score_rows = scorer.score_pairs(
            [str(x["image_path"]) for x in missing_msl],
            [str(x["medsiglip_text"]) for x in missing_msl],
        )
        by_key = {}
        for record, score_info in zip(missing_msl, score_rows):
            sim = select_medsiglip_score(score_info, score_kind)
            record["medsiglip_sim"] = float(sim)
            record["medsiglip_score_kind"] = score_kind
            record["medsiglip_sigmoid"] = float(score_info["score"])
            record["medsiglip_logit"] = float(score_info["logit"])
            record["medsiglip_cosine"] = float(score_info["cosine"])
            record["medsiglip_ok"] = bool(sim >= min_sim)
            record["medsiglip_reason"] = "ok" if sim >= min_sim else "below_threshold"
            by_key[(_id_str(record["study_id"]), int(record["candidate_idx"]))] = record
        out = [by_key.get((_id_str(x["study_id"]), int(x["candidate_idx"])), x) for x in out]

    for record in out:
        record["all_filters_pass"] = bool(record.get("technical_ok", True)) and bool(record.get("medsiglip_ok", True))
        if record["all_filters_pass"]:
            record["reject_reason"] = ""
        elif not bool(record.get("technical_ok", True)):
            record["reject_reason"] = str(record.get("technical_reason", "technical_failed"))
        else:
            record["reject_reason"] = str(record.get("medsiglip_reason", "medsiglip_failed"))
    return out


def _select_best(qc_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    passed = [x for x in qc_rows if _truthy(x.get("all_filters_pass", False))]
    if not passed:
        return None

    def _score(row: dict[str, Any]) -> tuple[float, int]:
        sim = row.get("medsiglip_sim")
        sim_value = -1.0 if pd.isna(sim) or sim is None else float(sim)
        return sim_value, -int(row.get("candidate_idx", 0))

    return max(passed, key=_score)


def _refresh_qc_thresholds_from_scores(cfg: DictConfig, qc_df: pd.DataFrame) -> pd.DataFrame:
    if qc_df.empty:
        return qc_df

    out = qc_df.copy()
    use_technical = _as_bool(cfg, "h4_use_technical_qc", True)
    use_medsiglip = _as_bool(cfg, "h4_use_medsiglip_qc", True)
    min_sim = _as_float(cfg, "h4_medsiglip_min_sim", 0.00027)
    score_kind = str(_cfg(cfg, "h4_medsiglip_score_kind", "sigmoid_logit"))

    if use_technical and "technical_ok" in out.columns:
        technical_ok = out["technical_ok"].map(_truthy)
    else:
        technical_ok = pd.Series(True, index=out.index)
        out["technical_ok"] = True
        if "technical_reason" not in out.columns:
            out["technical_reason"] = "disabled"

    if not use_medsiglip:
        medsiglip_ok = pd.Series(True, index=out.index)
        out["medsiglip_ok"] = True
        out["medsiglip_reason"] = "disabled"
    elif "medsiglip_sim" in out.columns:
        scores = pd.to_numeric(out["medsiglip_sim"], errors="coerce")
        medsiglip_ok = scores.ge(min_sim).fillna(False)
        out["medsiglip_ok"] = medsiglip_ok
        out["medsiglip_score_kind"] = score_kind
        out["medsiglip_reason"] = np.where(medsiglip_ok, "ok", "below_threshold")
    elif "medsiglip_ok" in out.columns:
        medsiglip_ok = out["medsiglip_ok"].map(_truthy)
    else:
        medsiglip_ok = pd.Series(False, index=out.index)
        out["medsiglip_ok"] = False
        out["medsiglip_reason"] = "missing_score"

    all_filters_pass = technical_ok & medsiglip_ok
    out["technical_ok"] = technical_ok
    out["all_filters_pass"] = all_filters_pass

    technical_reason = out.get("technical_reason", pd.Series("technical_failed", index=out.index)).fillna("technical_failed")
    medsiglip_reason = out.get("medsiglip_reason", pd.Series("medsiglip_failed", index=out.index)).fillna("medsiglip_failed")
    out["reject_reason"] = np.where(
        all_filters_pass,
        "",
        np.where(~technical_ok, technical_reason.astype(str), medsiglip_reason.astype(str)),
    )
    return out


def _candidate_rows_for_study(candidates: list[dict[str, Any]], study_id: str) -> list[dict[str, Any]]:
    return [x for x in candidates if _id_str(x.get("study_id")) == study_id]


def _checkpoint_manifests(
    cfg: DictConfig,
    source_df: pd.DataFrame,
    candidates: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> None:
    paths = h4_manifest_paths(cfg)
    candidates_df = pd.DataFrame(candidates)
    qc_df = pd.DataFrame(qc_rows)
    selected_df = pd.DataFrame(selected_rows)
    _write_csv(candidates_df, paths["candidates"])
    _write_csv(qc_df, paths["qc"])
    _write_csv(selected_df, paths["selected"])
    write_h4_training_manifests(cfg, source_df, selected_df, qc_df)


def _candidate_cache_matches_source(candidates_df: pd.DataFrame, source_df: pd.DataFrame) -> bool:
    if candidates_df.empty:
        return True
    required_candidate_cols = {"study_id", "roentgen_prompt"}
    required_source_cols = {"study_id", "roentgen_prompt"}
    if not required_candidate_cols.issubset(candidates_df.columns) or not required_source_cols.issubset(source_df.columns):
        return False
    if "patient_sex" in source_df.columns and "patient_sex" not in candidates_df.columns:
        return False

    cand = candidates_df.copy()
    src = source_df.copy()
    cand["study_id"] = cand["study_id"].map(_id_str)
    src["study_id"] = src["study_id"].map(_id_str)
    src_cols = ["study_id", "roentgen_prompt"] + (["patient_sex"] if "patient_sex" in src.columns else [])
    check = cand.merge(src[src_cols].drop_duplicates("study_id"), on="study_id", how="inner", suffixes=("_cand", "_src"))
    if check.empty:
        return False
    if (check["roentgen_prompt_cand"].astype(str) != check["roentgen_prompt_src"].astype(str)).any():
        return False
    if "patient_sex_cand" in check.columns and "patient_sex_src" in check.columns:
        if (check["patient_sex_cand"].astype(str) != check["patient_sex_src"].astype(str)).any():
            return False
    return True


def _real_source_subset_for_training(cfg: DictConfig, source_df: pd.DataFrame, selected_ids: list[str]) -> pd.DataFrame:
    source = source_df.copy()
    source["study_id"] = source["study_id"].map(_id_str)

    policy = str(_cfg(cfg, "h4_real_subset_policy", "selected")).strip().lower()
    if policy in {"selected", "accepted", "accepted_synthetic"}:
        if not selected_ids:
            return source.iloc[0:0].copy()
        return source.set_index("study_id", drop=False).loc[selected_ids].reset_index(drop=True)

    if policy in {"source_window", "fixed_source", "fixed", "eligible_window"}:
        subset_size = _as_int(cfg, "h4_real_subset_size", -1)
        if subset_size <= 0:
            max_source = _as_int(cfg, "h4_max_source_studies", -1)
            target_accepted = _as_int(cfg, "h4_target_accepted_studies", -1)
            if max_source > 0:
                subset_size = max_source
            elif target_accepted > 0:
                subset_size = target_accepted
            else:
                subset_size = len(source)
        return source.iloc[: min(int(subset_size), len(source))].reset_index(drop=True)

    raise ValueError(
        f"Unsupported h4_real_subset_policy={policy!r}. "
        "Use selected or source_window."
    )


def write_h4_training_manifests(
    cfg: DictConfig,
    source_df: pd.DataFrame | None = None,
    selected_df: pd.DataFrame | None = None,
    qc_df: pd.DataFrame | None = None,
) -> dict[str, int]:
    paths = h4_manifest_paths(cfg)
    if source_df is None:
        source_df = _read_csv(paths["eligible_pool"])
    if selected_df is None:
        selected_df = _read_csv(paths["selected"])
    if qc_df is None:
        qc_df = _read_csv(paths["qc"])

    if source_df.empty:
        _write_csv(pd.DataFrame(), paths["train_real_baseline"])
        _write_csv(pd.DataFrame(), paths["train_synthetic_pretrain"])
        _write_csv(pd.DataFrame(), paths["train_real_finetune"])
        return {"n_real": 0, "n_synthetic": 0}

    source = source_df.copy()
    source["study_id"] = source["study_id"].map(_id_str)
    if selected_df.empty:
        selected_ids = []
        selected_df = pd.DataFrame()
    else:
        selected_df = selected_df.drop_duplicates("study_id", keep="first").copy()
        selected_ids = selected_df["study_id"].map(_id_str).tolist()

    real_source = _real_source_subset_for_training(cfg, source, selected_ids)

    shared_cols = [
        "subject_id",
        "study_id",
        "dicom_id",
        "split",
        "ViewPosition",
        "patient_sex",
        "patient_sex_label",
        "patient_sex_status",
        "patient_sex_source",
        "patient_sex_evidence",
        "roentgen_sex_phrase",
        "h4_use_patient_sex_in_prompt",
        "dcm_path",
        "indication",
        "ref_findings",
        "ref_impression",
        "raw_impression",
        "roentgen_prompt",
        "roentgen_prompt_tokens",
        "medsiglip_impression_tokens",
        "h4_source_order",
    ]
    real_df = real_source[[c for c in shared_cols if c in real_source.columns]].copy()
    real_df["image_source"] = "real_dicom"

    if selected_df.empty:
        _write_csv(real_df, paths["train_real_baseline"])
        _write_csv(pd.DataFrame(), paths["train_synthetic_pretrain"])
        _write_csv(real_df.copy(), paths["train_real_finetune"])
        return {"n_real": int(len(real_df)), "n_synthetic": 0}

    selected_source = source.set_index("study_id", drop=False).loc[selected_ids].reset_index(drop=True)
    selected_source_df = selected_source[[c for c in shared_cols if c in selected_source.columns]].copy()
    selected_meta = selected_df[
        [
            c
            for c in [
                "study_id",
                "candidate_idx",
                "candidate_seed",
                "image_path",
                "medsiglip_sim",
                "medsiglip_score_kind",
                "medsiglip_sigmoid",
                "medsiglip_logit",
                "medsiglip_cosine",
                "technical_ok",
                "medsiglip_ok",
            ]
            if c in selected_df.columns
        ]
    ].copy()
    selected_meta["study_id"] = selected_meta["study_id"].map(_id_str)
    synth_df = selected_source_df.merge(selected_meta, on="study_id", how="inner")
    synth_df["image_source"] = "synthetic_png"

    _write_csv(real_df, paths["train_real_baseline"])
    _write_csv(synth_df, paths["train_synthetic_pretrain"])
    _write_csv(real_df.copy(), paths["train_real_finetune"])

    return {"n_real": int(len(real_df)), "n_synthetic": int(len(synth_df))}


def write_h4_source_status(
    cfg: DictConfig,
    source_df: pd.DataFrame,
    candidates: list[dict[str, Any]],
    qc_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    paths = h4_manifest_paths(cfg)
    candidate_df = pd.DataFrame(candidates)
    qc_df = pd.DataFrame(qc_rows)
    selected_df = pd.DataFrame(selected_rows)

    selected_by_study = {}
    if not selected_df.empty:
        for row in selected_df.to_dict("records"):
            selected_by_study[_id_str(row["study_id"])] = row

    cand_counts = {}
    if not candidate_df.empty:
        cand_counts = candidate_df.groupby(candidate_df["study_id"].map(_id_str)).size().to_dict()

    pass_counts = {}
    best_sims = {}
    if not qc_df.empty:
        tmp = qc_df.copy()
        tmp["study_id"] = tmp["study_id"].map(_id_str)
        pass_counts = tmp.groupby("study_id")["all_filters_pass"].sum().to_dict()
        if "medsiglip_sim" in tmp:
            best_sims = tmp.groupby("study_id")["medsiglip_sim"].max().to_dict()

    rows = []
    for row in source_df.to_dict("records"):
        study_id = _id_str(row["study_id"])
        selected = selected_by_study.get(study_id)
        candidate_count = int(cand_counts.get(study_id, 0))
        passed_count = int(pass_counts.get(study_id, 0))
        if selected is not None:
            status = "accepted"
        elif candidate_count > 0:
            status = "rejected_no_candidate_passed"
        else:
            status = "not_attempted"
        rows.append(
            {
                "study_id": study_id,
                "subject_id": _id_str(row.get("subject_id")),
                "dicom_id": _id_str(row.get("dicom_id")),
                "ViewPosition": row.get("ViewPosition", ""),
                "patient_sex": row.get("patient_sex", ""),
                "patient_sex_label": row.get("patient_sex_label", ""),
                "patient_sex_status": row.get("patient_sex_status", ""),
                "patient_sex_source": row.get("patient_sex_source", ""),
                "h4_source_order": row.get("h4_source_order", None),
                "status": status,
                "candidate_count": candidate_count,
                "passed_candidate_count": passed_count,
                "best_medsiglip_sim": best_sims.get(study_id, None),
                "selected_candidate_idx": None if selected is None else selected.get("candidate_idx"),
                "selected_image_path": "" if selected is None else selected.get("image_path", ""),
            }
        )
    status_df = pd.DataFrame(rows)
    _write_csv(status_df, paths["source_status"])
    return status_df


def prepare_h4_synthetic_dataset(cfg: DictConfig, df: pd.DataFrame, force: bool = False) -> dict[str, Any]:
    paths = h4_manifest_paths(cfg)
    for path in [paths["out_dir"], paths["manifests_dir"], paths["images_raw_dir"]]:
        ensure_dir(path)

    print(f"[H4] Output dir: {paths['out_dir']}", flush=True)
    source_df = build_h4_source_manifest(cfg, df, force=force)
    target_accepted = _as_int(cfg, "h4_target_accepted_studies", 2000)
    max_source = _as_int(cfg, "h4_max_source_studies", -1)
    candidates_per_study = _as_int(cfg, "h4_candidates_per_study", 2)
    checkpoint_every = _as_int(cfg, "h4_checkpoint_every", 25)
    resume = _as_bool(cfg, "h4_resume", True) and not force

    if max_source > 0:
        source_work = source_df.iloc[:max_source].reset_index(drop=True)
    else:
        source_work = source_df
    print(
        f"[H4] Generation source window rows={len(source_work)} target_accepted={target_accepted} "
        f"candidates_per_study={candidates_per_study} resume={resume}",
        flush=True,
    )

    candidates_df_resume = _read_csv(paths["candidates"]) if resume else pd.DataFrame()
    if resume and not _candidate_cache_matches_source(candidates_df_resume, source_df):
        print(
            "[H4] Existing candidates do not match current source prompt/sex schema; "
            "ignoring candidate/QC resume cache for this artifact.",
            flush=True,
        )
        candidates_df_resume = pd.DataFrame()
        resume = False

    candidates = candidates_df_resume.to_dict("records") if resume else []
    qc_df_resume = _read_csv(paths["qc"]) if resume else pd.DataFrame()
    if not qc_df_resume.empty:
        qc_df_resume = _refresh_qc_thresholds_from_scores(cfg, qc_df_resume)
        qc_rows = qc_df_resume.to_dict("records")
        selected_rows = []
        for _, group in qc_df_resume.groupby(qc_df_resume["study_id"].map(_id_str), sort=False):
            selected = _select_best(group.to_dict("records"))
            if selected is not None:
                selected_record = dict(selected)
                selected_record["selection_reason"] = "best_medsiglip_sim" if selected_record.get("medsiglip_sim") is not None else "first_passing"
                selected_rows.append(selected_record)
        if target_accepted > 0:
            selected_rows = selected_rows[:target_accepted]
    else:
        qc_rows = []
        selected_rows = [] if not resume else _read_csv(paths["selected"]).to_dict("records")

    candidate_by_key = {(_id_str(x.get("study_id")), int(x.get("candidate_idx", -1))): x for x in candidates}
    qc_by_key = {(_id_str(x.get("study_id")), int(x.get("candidate_idx", -1))): x for x in qc_rows}
    selected_by_study = {_id_str(x.get("study_id")): x for x in selected_rows}
    print(
        f"[H4] Resume state: candidates={len(candidates)} qc_rows={len(qc_rows)} selected={len(selected_by_study)}",
        flush=True,
    )

    if target_accepted > 0 and len(selected_by_study) >= target_accepted:
        selected_rows = list(selected_by_study.values())
        _checkpoint_manifests(cfg, source_df, candidates, qc_rows, selected_rows)
        status_df = write_h4_source_status(cfg, source_df, candidates, qc_rows, selected_rows)
        train_counts = write_h4_training_manifests(cfg, source_df, pd.DataFrame(selected_rows), pd.DataFrame(qc_rows))
        summary = {
            "eligible_pool": int(len(source_df)),
            "source_attempt_window": int(len(source_work)),
            "raw_candidates": int(len(candidates)),
            "qc_rows": int(len(qc_rows)),
            "accepted_studies": int(len(selected_rows)),
            **train_counts,
            "target_accepted_studies": int(target_accepted),
            "candidates_per_study": int(candidates_per_study),
            "artifact_name": str(paths["artifact_name"]),
            "artifact_dir": str(paths["artifact_dir"]),
            "manifests_dir": str(paths["manifests_dir"]),
            "images_raw_dir": str(paths["images_raw_dir"]),
            "medsiglip_score_kind": str(_cfg(cfg, "h4_medsiglip_score_kind", "sigmoid_logit")),
            "medsiglip_min_sim": float(_as_float(cfg, "h4_medsiglip_min_sim", 0.00027)),
            "source_status_counts": status_df["status"].value_counts().to_dict() if not status_df.empty else {},
        }
        write_json(paths["summary"], summary)
        return summary

    pipe, device = _load_roentgen_pipeline(cfg)
    scorer = MedSigLIPScorer(cfg) if _as_bool(cfg, "h4_use_medsiglip_qc", True) else None

    for idx, row in enumerate(tqdm(source_work.to_dict("records"), desc="H4 RoentGen-v2 studies")):
        row_s = pd.Series(row)
        study_id = _id_str(row["study_id"])
        if target_accepted > 0 and len(selected_by_study) >= target_accepted:
            break
        if study_id in selected_by_study:
            continue

        missing: list[tuple[int, int, Path]] = []
        for cand_idx in range(candidates_per_study):
            key = (study_id, cand_idx)
            seed = _candidate_seed(cfg, int(row["h4_source_order"]), cand_idx)
            rel = candidate_relative_path(row, cand_idx, seed)
            out_path = paths["images_raw_dir"] / rel
            if key not in candidate_by_key or not out_path.exists():
                missing.append((cand_idx, seed, out_path))

        new_records = _generate_candidate_images(cfg, pipe, device, row_s, missing)
        for rec in new_records:
            key = (_id_str(rec["study_id"]), int(rec["candidate_idx"]))
            candidate_by_key[key] = rec
            candidates.append(rec)

        study_candidates = _candidate_rows_for_study(candidates, study_id)
        study_qc = _qc_candidates_for_study(cfg, row_s, study_candidates, scorer=scorer, existing_qc=qc_by_key)
        for rec in study_qc:
            key = (_id_str(rec["study_id"]), int(rec["candidate_idx"]))
            if key not in qc_by_key:
                qc_rows.append(rec)
            qc_by_key[key] = rec

        selected = _select_best(study_qc)
        if selected is not None:
            selected_record = dict(selected)
            selected_record["selection_reason"] = "best_medsiglip_sim" if selected_record.get("medsiglip_sim") is not None else "first_passing"
            selected_by_study[study_id] = selected_record
            selected_rows.append(selected_record)

        if checkpoint_every > 0 and (idx + 1) % checkpoint_every == 0:
            _checkpoint_manifests(cfg, source_df, candidates, qc_rows, list(selected_by_study.values()))

    selected_rows = list(selected_by_study.values())
    _checkpoint_manifests(cfg, source_df, candidates, qc_rows, selected_rows)
    status_df = write_h4_source_status(cfg, source_df, candidates, qc_rows, selected_rows)
    train_counts = write_h4_training_manifests(cfg, source_df, pd.DataFrame(selected_rows), pd.DataFrame(qc_rows))

    summary = {
        "eligible_pool": int(len(source_df)),
        "source_attempt_window": int(len(source_work)),
        "raw_candidates": int(len(candidates)),
        "qc_rows": int(len(qc_rows)),
        "accepted_studies": int(len(selected_rows)),
        **train_counts,
        "target_accepted_studies": int(target_accepted),
        "candidates_per_study": int(candidates_per_study),
        "artifact_name": str(paths["artifact_name"]),
        "artifact_dir": str(paths["artifact_dir"]),
        "manifests_dir": str(paths["manifests_dir"]),
        "images_raw_dir": str(paths["images_raw_dir"]),
        "medsiglip_score_kind": str(_cfg(cfg, "h4_medsiglip_score_kind", "sigmoid_logit")),
        "medsiglip_min_sim": float(_as_float(cfg, "h4_medsiglip_min_sim", 0.00027)),
    }
    if not status_df.empty:
        summary["source_status_counts"] = status_df["status"].value_counts().to_dict()
    write_json(paths["summary"], summary)
    return summary


def reselect_h4_from_qc(cfg: DictConfig) -> dict[str, Any]:
    paths = h4_manifest_paths(cfg)
    source_df = _read_csv(paths["eligible_pool"])
    qc_df = _read_csv(paths["qc"])
    candidates_df = _read_csv(paths["candidates"])
    if source_df.empty or qc_df.empty:
        raise FileNotFoundError("H4 source/QC manifests are missing or empty; run data.prepare_mode=h4_all first.")
    if not _candidate_cache_matches_source(candidates_df, source_df):
        raise RuntimeError(
            "H4 candidate/QC manifests do not match the current source prompt/sex schema. "
            "Run h4_all with a new data.h4_artifact_name, or regenerate this artifact."
        )

    qc_df = _refresh_qc_thresholds_from_scores(cfg, qc_df)
    selected = []
    for _, group in qc_df.groupby(qc_df["study_id"].map(_id_str), sort=False):
        best = _select_best(group.to_dict("records"))
        if best is not None:
            selected.append(best)

    target_accepted = _as_int(cfg, "h4_target_accepted_studies", 2000)
    if target_accepted > 0:
        selected = selected[:target_accepted]
    selected_df = pd.DataFrame(selected)
    _write_csv(qc_df, paths["qc"])
    _write_csv(selected_df, paths["selected"])
    train_counts = write_h4_training_manifests(cfg, source_df, selected_df, qc_df)
    status_df = write_h4_source_status(
        cfg,
        source_df,
        candidates_df.to_dict("records") if not candidates_df.empty else qc_df.to_dict("records"),
        qc_df.to_dict("records"),
        selected,
    )
    summary = {
        "accepted_studies": int(len(selected_df)),
        **train_counts,
        "artifact_name": str(paths["artifact_name"]),
        "artifact_dir": str(paths["artifact_dir"]),
        "manifests_dir": str(paths["manifests_dir"]),
        "images_raw_dir": str(paths["images_raw_dir"]),
        "medsiglip_score_kind": str(_cfg(cfg, "h4_medsiglip_score_kind", "sigmoid_logit")),
        "medsiglip_min_sim": float(_as_float(cfg, "h4_medsiglip_min_sim", 0.00027)),
        "source_status_counts": status_df["status"].value_counts().to_dict() if not status_df.empty else {},
    }
    write_json(paths["summary"], summary)
    return summary


def rescore_h4_qc_from_candidates(cfg: DictConfig) -> dict[str, Any]:
    paths = h4_manifest_paths(cfg)
    source_df = _read_csv(paths["eligible_pool"])
    candidates_df = _read_csv(paths["candidates"])
    if source_df.empty or candidates_df.empty:
        raise FileNotFoundError(
            "H4 eligible/candidate manifests are missing or empty; run data.prepare_mode=h4_all first."
        )
    if not _candidate_cache_matches_source(candidates_df, source_df):
        raise RuntimeError(
            "H4 candidate manifest does not match the current source prompt/sex schema. "
            "Run h4_all with a new data.h4_artifact_name, or regenerate this artifact."
        )

    for path in candidates_df["image_path"].astype(str).tolist()[:1]:
        if not Path(path).exists():
            raise FileNotFoundError(
                "Synthetic image files are not available at the paths stored in synthetic_candidates_manifest.csv. "
                f"First missing path example: {path}"
            )

    scorer = MedSigLIPScorer(cfg) if _as_bool(cfg, "h4_use_medsiglip_qc", True) else None
    source_by_study = {}
    source_tmp = source_df.copy()
    source_tmp["study_id"] = source_tmp["study_id"].map(_id_str)
    for row in source_tmp.to_dict("records"):
        source_by_study[_id_str(row["study_id"])] = row

    qc_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    target_accepted = _as_int(cfg, "h4_target_accepted_studies", 2000)

    grouped = candidates_df.groupby(candidates_df["study_id"].map(_id_str), sort=False)
    for study_id, group in tqdm(grouped, total=len(grouped), desc="H4 MedSigLIP QC rescore"):
        source_row = source_by_study.get(_id_str(study_id))
        if source_row is None:
            source_row = group.iloc[0].to_dict()
        source_series = pd.Series(source_row)
        study_qc = _qc_candidates_for_study(
            cfg,
            source_series,
            group.to_dict("records"),
            scorer=scorer,
            existing_qc={},
        )
        qc_rows.extend(study_qc)

        selected = _select_best(study_qc)
        if selected is not None:
            selected_record = dict(selected)
            selected_record["selection_reason"] = (
                "best_medsiglip_score" if selected_record.get("medsiglip_sim") is not None else "first_passing"
            )
            selected_rows.append(selected_record)
            if target_accepted > 0 and len(selected_rows) >= target_accepted:
                break

    qc_df = pd.DataFrame(qc_rows)
    selected_df = pd.DataFrame(selected_rows)
    _write_csv(qc_df, paths["qc"])
    _write_csv(selected_df, paths["selected"])
    train_counts = write_h4_training_manifests(cfg, source_df, selected_df, qc_df)
    status_df = write_h4_source_status(
        cfg,
        source_df,
        candidates_df.to_dict("records"),
        qc_df.to_dict("records"),
        selected_df.to_dict("records"),
    )
    summary = {
        "eligible_pool": int(len(source_df)),
        "raw_candidates_available": int(len(candidates_df)),
        "qc_rows": int(len(qc_df)),
        "accepted_studies": int(len(selected_df)),
        **train_counts,
        "target_accepted_studies": int(target_accepted),
        "medsiglip_score_kind": str(_cfg(cfg, "h4_medsiglip_score_kind", "sigmoid_logit")),
        "artifact_name": str(paths["artifact_name"]),
        "artifact_dir": str(paths["artifact_dir"]),
        "manifests_dir": str(paths["manifests_dir"]),
        "images_raw_dir": str(paths["images_raw_dir"]),
        "source_status_counts": status_df["status"].value_counts().to_dict() if not status_df.empty else {},
    }
    write_json(paths["summary"], summary)
    return summary
