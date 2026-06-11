from __future__ import annotations

import importlib
import os
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from thesis_rrg.data.readers.mimic_reports import build_or_load_mimic_cache
from thesis_rrg.utils.config_utils import resolve_experiment_data_value
from thesis_rrg.utils.io import write_parquet
from thesis_rrg.utils.seed import seed_everything



def _set_common_env(cfg: DictConfig) -> None:
    os.environ["MODEL_ID"] = str(cfg.model.model_id)
    os.environ["MIMIC_ROOT"] = str(cfg.data.mimic_root)
    os.environ["MIMIC_CACHE_DIR"] = str(cfg.data.cache_dir)
    os.environ["MIMIC_CACHE_TAG"] = str(cfg.data.cache_tag)
    os.environ["SEED"] = str(cfg.seed)
    os.environ["CHEXBERT_CACHE_DIR"] = str(cfg.data.chexbert_cache_dir)

    os.environ.setdefault("HF_HUB_READ_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")



def _load_or_build_clean_cache(cfg: DictConfig, force: bool):
    cache_dir = Path(str(cfg.data.cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)

    prepared = cache_dir / f"mimic_prepared_frontal1perstudy_sections_{cfg.data.cache_tag}.parquet"
    clean = cache_dir / f"mimic_prepared_frontal1perstudy_sections_{cfg.data.cache_tag}_clean.parquet"

    if (not force) and clean.exists():
        print(f"[prepare] Loading clean MIMIC cache: {clean}", flush=True)
        df = pd.read_parquet(clean)
        print(f"[prepare] Loaded clean MIMIC cache rows={len(df)}", flush=True)
        return df

    if (not force) and prepared.exists():
        print(f"[prepare] Loading prepared MIMIC cache: {prepared}", flush=True)
        df = pd.read_parquet(prepared)
    else:
        print("[prepare] Building MIMIC cache from source files", flush=True)
        df = build_or_load_mimic_cache(
            mimic_root=Path(str(cfg.data.mimic_root)),
            cache_dir=cache_dir,
            cache_tag=str(cfg.data.cache_tag),
        )
        print(f"[prepare] Built/loaded MIMIC cache rows={len(df)}", flush=True)
        return df

    print("[prepare] Filtering rows with existing DICOM paths", flush=True)
    exists = df["dcm_path"].astype(str).map(os.path.exists)
    df = df.loc[exists].reset_index(drop=True)
    write_parquet(df, clean)
    print(f"[prepare] Wrote clean MIMIC cache rows={len(df)}: {clean}", flush=True)
    return df



def _prepare_e2a(cfg: DictConfig, df: pd.DataFrame, force: bool) -> None:
    from thesis_rrg.hypotheses.h2 import structural as nb

    e2a_dir = Path(str(cfg.data.e2a_dir))
    e2a_dir.mkdir(parents=True, exist_ok=True)

    os.environ["E2A_DIR"] = str(cfg.data.e2a_dir)
    os.environ["E2A_BUILD_MAX_ROWS"] = str(cfg.data.e2a_build_max_rows)
    os.environ["E2A_AUG_PER_SAMPLE"] = str(cfg.data.e2a_aug_per_sample)
    os.environ["E2A_APPLY_P"] = str(cfg.data.e2a_apply_p)
    os.environ["E2A_RG_MODE"] = str(cfg.data.e2a_rg_mode)
    os.environ["E2A_USE_RADGRAPH"] = "1" if bool(cfg.data.get("e2a_use_radgraph", False)) else "0"


    os.environ["E2A_MIN_ENT_P"] = str(cfg.data.e2a_thresholds.min_ent_p)
    os.environ["E2A_MIN_ENT_R"] = str(cfg.data.e2a_thresholds.min_ent_r)
    os.environ["E2A_MIN_REL_P"] = str(cfg.data.e2a_thresholds.min_rel_p)
    os.environ["E2A_MIN_REL_R"] = str(cfg.data.e2a_thresholds.min_rel_r)

    os.environ["E2A_MAX_EXTRA_ENT"] = str(cfg.data.e2a_diffs.max_extra_ent)
    os.environ["E2A_MAX_MISSING_ENT"] = str(cfg.data.e2a_diffs.max_missing_ent)
    os.environ["E2A_MAX_EXTRA_REL"] = str(cfg.data.e2a_diffs.max_extra_rel)
    os.environ["E2A_MAX_MISSING_REL"] = str(cfg.data.e2a_diffs.max_missing_rel)
    os.environ["E2A_MAX_CHAR_DELTA"] = str(cfg.data.e2a_max_char_delta)

    importlib.reload(nb)

    # e2a_cache = Path(cfg.data.e2a_dir) / f"mimic_e2a_struct_targets_{cfg.data.cache_tag}.parquet"
    e2a_cache = Path(str(cfg.data.e2a_cache_path))
    e2a_cache.parent.mkdir(parents=True, exist_ok=True)
    if (not force) and e2a_cache.exists():
        print("E2A cache exists, skip:", e2a_cache)
        return

    df_train = df[(df["split"] == "train") & (df["ref_findings"] != "") & (df["ref_impression"] != "")].copy()
    print("df_train for E2A:", len(df_train))
    nb.build_e2a_bank(df_train, e2a_cache, max_rows=int(cfg.data.e2a_build_max_rows))



def _build_e2c_clean_shards(cfg: DictConfig, df: pd.DataFrame, force: bool) -> None:
    from thesis_rrg.hypotheses.h2 import llm_paraphrase as nb

    nb = importlib.reload(nb)
    nb.rg = None

    num_shards = int(cfg.data.e2c_num_shards)
    shards_dir = Path(str(cfg.data.e2c_shards_dir))
    shards_dir.mkdir(parents=True, exist_ok=True)

    expected = [shards_dir / f"mimic_clean_shard{sid}_of{num_shards}.parquet" for sid in range(num_shards)]
    if (not force) and all(p.exists() for p in expected):
        print("Shard files exist, skip:", shards_dir)
        return

    df_sh = nb.add_shard_id(df, num_shards=num_shards, key="study_id")
    nb.save_all_shards(df_sh, shards_dir, num_shards)



def _configure_e2c_globals(cfg: DictConfig, nb) -> None:
    nb.SEED = int(cfg.seed)
    nb.AUX_DEVICE = str(cfg.data.e2c_aux_device)
    nb.CHEXBERT_DEVICE = nb.AUX_DEVICE

    nb.USE_QUICK_SAFETY = bool(cfg.data.e2c_use_quick_safety)
    nb.USE_LEXICAL_FILTER = bool(cfg.data.e2c_use_lexical_filter)
    nb.USE_ASCII_FILTER = bool(cfg.data.e2c_use_ascii_filter)
    nb.USE_FORMAT_FILTER = bool(cfg.data.get("e2c_use_format_filter", True))
    nb.USE_RADGRAPH = bool(cfg.data.e2c_use_radgraph)
    nb.USE_CHEXBERT = bool(cfg.data.e2c_use_chexbert)
    nb.USE_MEDSIGLIP = bool(cfg.data.e2c_use_medsiglip)

    nb.LEX_MAX_SEQ_RATIO = float(cfg.data.e2c_lex_max_seq_ratio)
    nb.LEX_MAX_4GRAM_OVERLAP = float(cfg.data.e2c_lex_max_4gram_overlap)
    nb.LEX_MIN_TOKEN_CHANGE = float(cfg.data.e2c_lex_min_token_change)
    nb.FORMAT_MIN_CHAR_RATIO = float(cfg.data.get("e2c_format_min_char_ratio", 0.55))
    nb.FORMAT_MAX_CHAR_RATIO = float(cfg.data.get("e2c_format_max_char_ratio", 1.80))

    nb.MIN_ENT_P = float(cfg.data.e2a_thresholds.min_ent_p)
    nb.MIN_ENT_R = float(cfg.data.e2a_thresholds.min_ent_r)
    nb.MIN_REL_P = float(cfg.data.e2a_thresholds.min_rel_p)
    nb.MIN_REL_R = float(cfg.data.e2a_thresholds.min_rel_r)
    nb.MAX_EXTRA_ENT = int(cfg.data.e2a_diffs.max_extra_ent)
    nb.MAX_MISSING_ENT = int(cfg.data.e2a_diffs.max_missing_ent)
    nb.MAX_EXTRA_REL = int(cfg.data.e2a_diffs.max_extra_rel)
    nb.MAX_MISSING_REL = int(cfg.data.e2a_diffs.max_missing_rel)

    nb.MEDSIGLIP_MODEL_ID = str(cfg.data.e2c_medsiglip_model_id)
    nb.MEDSIGLIP_MODE = str(cfg.data.e2c_medsiglip_mode)
    nb.MEDSIGLIP_MIN_SIM = float(cfg.data.e2c_medsiglip_min_sim)
    nb.MEDSIGLIP_MAX_SIM = float(cfg.data.e2c_medsiglip_max_sim)

    nb.LLM_BATCH_SIZE = int(cfg.data.e2c_llm_batch_size)



def _prepare_e2c(cfg: DictConfig, df: pd.DataFrame):
    from thesis_rrg.hypotheses.h2 import llm_paraphrase as nb

    nb = importlib.reload(nb)
    _configure_e2c_globals(cfg, nb)

    nb.rg = None
    if nb.USE_RADGRAPH and hasattr(nb, "try_init_radgraph"):
        nb.try_init_radgraph()
        if nb.rg is None:
            raise RuntimeError(
                "E2C RadGraph filter is enabled but RadGraph failed to initialize. "
                "Disable data.e2c_use_radgraph or fix the RadGraph installation/device before regeneration."
            )


    nb.medsiglip = None
    if nb.USE_MEDSIGLIP and hasattr(nb, "try_init_medsiglip"):
        nb.try_init_medsiglip()
        if nb.medsiglip is None:
            raise RuntimeError(
                "E2C MedSigLIP filter is enabled but MedSigLIP failed to initialize. "
                "Disable data.e2c_use_medsiglip or fix the model/device before regeneration."
            )

    nb.chexbert_runner = None
    if nb.USE_CHEXBERT and hasattr(nb, "CheXbertRunner"):
        try:
            nb.chexbert_runner = nb.CheXbertRunner(device=str(nb.CHEXBERT_DEVICE))
        except Exception as e:
            nb.chexbert_runner = None
            raise RuntimeError(
                "E2C CheXbert filter is enabled but CheXbert failed to initialize. "
                "Disable data.e2c_use_chexbert or fix the CheXbert checkpoint/device before regeneration."
            ) from e

    num_shards = int(cfg.data.e2c_num_shards)
    shard_id = int(cfg.data.e2c_shard_id)

    split = str(cfg.data.e2c_aug_split).strip().lower()
    if split == "valid":
        split = "validate"

    use_pre = bool(cfg.data.e2c_use_pre_sharded_clean)
    shards_dir = Path(str(cfg.data.e2c_shards_dir))

    if use_pre and shard_id >= 0:
        shard_path = shards_dir / f"mimic_clean_shard{shard_id}_of{num_shards}.parquet"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard: {shard_path}")
        df_work = pd.read_parquet(shard_path).reset_index(drop=True)
        if "shard_id" not in df_work.columns:
            df_work["shard_id"] = shard_id
    else:
        df = nb.add_shard_id(df, num_shards=num_shards, key="study_id")
        if shard_id >= 0:
            df_work = df[df["shard_id"] == shard_id].copy().reset_index(drop=True)
        else:
            df_work = df.copy().reset_index(drop=True)

    df_work = df_work[df_work["split"] == split].copy().reset_index(drop=True)
    df_work["ref_findings"] = df_work["ref_findings"].fillna("").astype(str)
    df_work["ref_impression"] = df_work["ref_impression"].fillna("").astype(str)
    df_work = df_work[(df_work["ref_findings"].str.len() > 0) & (df_work["ref_impression"].str.len() > 0)].reset_index(drop=True)

    max_rows = int(cfg.data.e2c_max_rows)
    if max_rows > 0:
        df_work = df_work.iloc[:max_rows].reset_index(drop=True)

    out_dir = Path(str(cfg.data.e2c_out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = nb.ModelSpec(
        model_id=str(cfg.data.e2c_llm_model_id),
        kind=str(cfg.data.e2c_llm_kind),
        device=str(cfg.data.e2c_llm_device),
        dtype=str(cfg.data.e2c_llm_dtype),
        n_candidates=int(cfg.data.e2c_candidates_per_prompt),
        max_new_tokens=int(cfg.data.e2c_max_new_tokens),
        min_new_tokens=int(cfg.data.e2c_min_new_tokens),
        temperature=float(cfg.data.e2c_temperature),
        top_p=float(cfg.data.e2c_top_p),
        repetition_penalty=float(cfg.data.e2c_repetition_penalty),
        no_repeat_ngram=int(cfg.data.e2c_no_repeat_ngram),
        input_max_length=int(cfg.data.get("e2c_llm_input_max_length", 1024)),
    )
    nb.paraphraser = nb.HFParaphraser(spec)

    tag = f"{split}_shard{shard_id if shard_id >= 0 else 'ALL'}_of{num_shards}"
    checkpoint_every = int(cfg.data.e2c_checkpoint_every)
    resume = bool(cfg.data.e2c_resume)
    checkpoint_dir = out_dir / f"e2c_chunks_{tag}"

    df_cands, df_best = nb.run_paraphrase_with_filters(
        df_work,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        run_tag=tag,
        resume=resume,
    )

    cand_path = out_dir / f"e2c_candidates_{tag}.parquet"
    best_path = out_dir / f"e2c_best_{tag}.parquet"
    write_parquet(df_cands, cand_path)
    write_parquet(df_best, best_path)


def _reselect_e2c_best(cfg: DictConfig) -> None:
    from thesis_rrg.hypotheses.h2 import llm_paraphrase as nb

    nb = importlib.reload(nb)
    _configure_e2c_globals(cfg, nb)

    out_dir = Path(str(cfg.data.e2c_out_dir))
    split = str(cfg.data.e2c_aug_split).strip().lower()
    if split == "valid":
        split = "validate"

    num_shards = int(cfg.data.e2c_num_shards)
    shard_id = int(cfg.data.e2c_shard_id)
    tag = f"{split}_shard{shard_id if shard_id >= 0 else 'ALL'}_of{num_shards}"

    cand_path = out_dir / f"e2c_candidates_{tag}.parquet"
    chunk_dir = out_dir / f"e2c_chunks_{tag}"
    if cand_path.exists():
        df_cands = pd.read_parquet(cand_path)
    elif chunk_dir.exists():
        part_paths = sorted(chunk_dir.glob(f"e2c_candidates_{tag}_part*.parquet"))
        if not part_paths:
            raise FileNotFoundError(f"No E2C candidate chunks found in {chunk_dir}")
        df_cands = pd.concat([pd.read_parquet(p) for p in part_paths], ignore_index=True)
    else:
        raise FileNotFoundError(f"Missing E2C candidates: {cand_path} or {chunk_dir}")

    df_cands, df_best = nb.select_best_candidates(df_cands)
    best_path = out_dir / f"e2c_best_{tag}.parquet"
    write_parquet(df_cands, cand_path)
    write_parquet(df_best, best_path)
    print(
        "Reselected E2C:",
        "candidates=", len(df_cands),
        "keep_ok=", int(df_cands["keep_ok"].sum()) if "keep_ok" in df_cands else 0,
        "best=", len(df_best),
        "->", best_path,
    )


def _resolve_e2c_merge_shard_ids(cfg: DictConfig) -> list[int]:
    num_shards = int(cfg.data.e2c_num_shards)
    raw = cfg.data.get("e2c_merge_shard_ids", None)
    if raw in (None, "", "all"):
        return list(range(num_shards))
    if isinstance(raw, str):
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    ids = [int(x) for x in raw]
    return ids if ids else list(range(num_shards))


def _merge_e2c_shards(cfg: DictConfig):
    out_dir = Path(str(cfg.data.e2c_out_dir))
    split = str(cfg.data.e2c_aug_split).strip().lower()
    if split == "valid":
        split = "validate"

    num_shards = int(cfg.data.e2c_num_shards)
    shard_ids = _resolve_e2c_merge_shard_ids(cfg)

    paths = []
    missing = []
    for sid in shard_ids:
        p = out_dir / f"e2c_best_{split}_shard{sid}_of{num_shards}.parquet"
        if p.exists():
            paths.append(p)
        else:
            missing.append(p)

    if missing:
        raise FileNotFoundError("Missing shards:\n" + "\n".join(str(x) for x in missing))

    dfs = [pd.read_parquet(p) for p in paths]
    merged = pd.concat(dfs, ignore_index=True)

    dedup_col = "study_id" if "study_id" in merged.columns else ("ex_id" if "ex_id" in merged.columns else None)
    if dedup_col is not None:
        merged = merged.drop_duplicates(subset=[dedup_col], keep="first").reset_index(drop=True)

    merged_path = Path(str(cfg.data.e2c_merged_path))
    write_parquet(merged, merged_path)


def _prepare_h4(cfg: DictConfig, df: pd.DataFrame, mode: str, force: bool) -> None:
    from thesis_rrg.hypotheses.h4.synthetic import (
        build_h4_source_manifest,
        prepare_h4_synthetic_dataset,
        reselect_h4_from_qc,
        rescore_h4_qc_from_candidates,
    )

    if mode == "h4_source":
        source = build_h4_source_manifest(cfg, df=df, force=force)
        print("H4 eligible source pool:", len(source))
        return
    if mode == "h4_all":
        summary = prepare_h4_synthetic_dataset(cfg, df=df, force=force)
        print("H4 synthetic preparation summary:", summary)
        return
    if mode == "h4_select":
        summary = reselect_h4_from_qc(cfg)
        print("H4 manifest reselection summary:", summary)
        return
    if mode == "h4_rescore_qc":
        summary = rescore_h4_qc_from_candidates(cfg)
        print("H4 QC rescore summary:", summary)
        return
    raise ValueError(f"Unsupported H4 prepare mode: {mode}")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="prepare")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(int(cfg.seed))
    _set_common_env(cfg)

    mode = str(
        resolve_experiment_data_value(
            cfg,
            "prepare_mode",
            default="baseline_cache",
            sentinels=("", "none", "baseline_cache", None),
        )
    ).strip().lower()
    force = bool(cfg.get("force", False))

    if mode == "merge_e2c":
        _merge_e2c_shards(cfg)
        return
    if mode == "reselect_e2c":
        _reselect_e2c_best(cfg)
        return

    df = _load_or_build_clean_cache(cfg, force=force)

    if mode == "baseline_cache":
        print("Baseline cache prepared.")
    elif mode == "e2a":
        _prepare_e2a(cfg, df=df, force=force)
    elif mode == "build_e2c_shards":
        _build_e2c_clean_shards(cfg, df=df, force=force)
    elif mode == "e2c":
        _prepare_e2c(cfg, df=df)
    elif mode in {"h4_source", "h4_all", "h4_select", "h4_rescore_qc"}:
        _prepare_h4(cfg, df=df, mode=mode, force=force)
    else:
        raise ValueError("Unsupported prepare mode")


if __name__ == "__main__":
    main()
