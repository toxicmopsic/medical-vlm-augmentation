import os


import os

from pathlib import Path

MODEL_ID = os.environ.get("MODEL_ID", "google/medgemma-4b-pt")

MIMIC_ROOT = Path(os.environ.get("MIMIC_ROOT", "/storage_server/Data/PublicDatasets/MIMIC-CXR-DICOM"))

PHYSIONET_ROOT = MIMIC_ROOT / "physionet.org" / "files" / "mimic-cxr" / "2.0.0"

FILES_ROOT = PHYSIONET_ROOT / "files"

RECORD_LIST_CSV = PHYSIONET_ROOT / "cxr-record-list.csv"

SPLIT_CSV = MIMIC_ROOT / "mimic-cxr-2.0.0-split.csv"

META_CSV = MIMIC_ROOT / "mimic-cxr-2.0.0-metadata.csv"

CACHE_DIR = Path(os.environ.get("MIMIC_CACHE_DIR", "MIMIC_CACHE_DIR"))

CACHE_TAG = os.environ.get("MIMIC_CACHE_TAG", "v1")

PREPARED_CACHE = CACHE_DIR / f"mimic_prepared_frontal1perstudy_sections_{CACHE_TAG}.parquet"

CLEAN_CACHE = CACHE_DIR / f"mimic_prepared_frontal1perstudy_sections_{CACHE_TAG}_clean.parquet"

E2A_DIR = Path(os.environ.get("E2A_DIR", "E2a_CACHE"))

E2A_CACHE = E2A_DIR / f"mimic_e2a_struct_targets_{CACHE_TAG}.parquet"

SEED = int(os.environ.get("SEED", "0"))

BUILD_MAX_ROWS = int(os.environ.get("E2A_BUILD_MAX_ROWS", "100"))

AUG_PER_SAMPLE = int(os.environ.get("E2A_AUG_PER_SAMPLE", "5"))

APPLY_P = float(os.environ.get("E2A_APPLY_P", "0.5"))

E2A_RG_MODE = os.environ.get("E2A_RG_MODE", "threshold")

MIN_ENT_P = float(os.environ.get("E2A_MIN_ENT_P", "0.98"))

MIN_ENT_R = float(os.environ.get("E2A_MIN_ENT_R", "0.95"))

MIN_REL_P = float(os.environ.get("E2A_MIN_REL_P", "0.90"))

MIN_REL_R = float(os.environ.get("E2A_MIN_REL_R", "0.80"))

MAX_EXTRA_ENT = int(os.environ.get("E2A_MAX_EXTRA_ENT", "1"))

MAX_MISSING_ENT = int(os.environ.get("E2A_MAX_MISSING_ENT", "1"))

MAX_EXTRA_REL = int(os.environ.get("E2A_MAX_EXTRA_REL", "2"))

MAX_MISSING_REL = int(os.environ.get("E2A_MAX_MISSING_REL", "1"))

MAX_CHAR_DELTA = int(os.environ.get("E2A_MAX_CHAR_DELTA", "800"))


import re

import json

import random

import numpy as np

import pandas as pd

from tqdm.auto import tqdm

from thesis_rrg.data.transforms.structural_text import (
    format_findings_impression_target,
    structural_aug_once as structural_aug_once_core,
)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

_ws = re.compile(r"\s+")

def minimal_one_line(text):
    if not isinstance(text, str):
        return ""
    return _ws.sub(" ", text.strip())


_ws_tabs = re.compile(r"[ \t]+")

_many_nl = re.compile(r"\n{3,}")

_multispace = re.compile(r"\s+")

_hidden_control = re.compile(r"[\u200b-\u200d\ufeff]")

def norm_keep_newlines(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    t = _hidden_control.sub("", t)
    t = _ws_tabs.sub(" ", t)
    t = _many_nl.sub("\n\n", t)
    return t.strip()

def norm_for_filters(text: str) -> str:
    t = norm_keep_newlines(text)
    t = t.replace("\n", " ")
    t = _multispace.sub(" ", t).strip()
    return t


import pydicom

from PIL import Image

from pydicom.pixel_data_handlers.util import apply_modality_lut

def dicom_to_uint8_grayscale(dcm_path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array  # usually int16/uint16 from DICOM decoder
    try:
        # dtype after modality LUT may become float64 depending on tags
        arr = apply_modality_lut(arr, ds)
    except Exception:
        # keep raw decoded array if LUT is unavailable/broken
        arr = np.asarray(arr)

    # unify numeric type for stable normalization math: int16/uint16/float64 -> float32
    arr_f = np.asarray(arr, dtype=np.float32)
    mn, mx = float(arr_f.min()), float(arr_f.max())

    # float32 in [0, 1]
    if mx > mn:
        arr01 = (arr_f - mn) / (mx - mn)
    else:
        arr01 = np.zeros_like(arr_f, dtype=np.float32)
    arr01 = np.clip(arr01, 0.0, 1.0)  # float32 in [0, 1]

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr01 = 1.0 - arr01  # still float32 in [0, 1]

    return (arr01 * 255.0).clip(0, 255).astype(np.uint8)  # final uint8

def pad_resize_to_square(img: Image.Image, out_size: int = 896) -> Image.Image:
    w, h = img.size
    scale = out_size / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), resample=Image.BICUBIC)

    canvas = Image.new("RGB", (out_size, out_size), (0, 0, 0))
    left = (out_size - new_w) // 2
    top = (out_size - new_h) // 2
    canvas.paste(img, (left, top))
    return canvas

def safe_open_mimic_dicom(dcm_path: Path) -> Image.Image:
    arr = dicom_to_uint8_grayscale(dcm_path)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    return pad_resize_to_square(img, out_size=896)


import re

def study_report_path(subject_id: int, study_id: int) -> Path:
    sid = str(subject_id)
    prefix = f"p{sid[:2]}"
    return FILES_ROOT / prefix / f"p{sid}" / f"s{study_id}.txt"

def extract_section(report_text: str, section_names):
    if not report_text:
        return ""
    txt = report_text.replace("\r\n", "\n").replace("\r", "\n")
    for name in section_names:
        pattern = rf"(?ims)^\s*{re.escape(name)}\s*:\s*(.*?)(?=^\s*[A-Z][A-Z0-9 /\-()]+?\s*:|\Z)"
        m = re.search(pattern, txt)
        if m:
            return minimal_one_line(m.group(1))
    return ""

def read_report_sections(subject_id: int, study_id: int):
    p = study_report_path(subject_id, study_id)
    if not p.exists():
        return {"indication": "", "findings": "", "impression": ""}
    txt = p.read_text(errors="ignore")
    indication = extract_section(txt, ["INDICATION", "HISTORY", "CLINICAL HISTORY", "REASON FOR EXAM", "EXAMINATION"])
    findings = extract_section(txt, ["FINDINGS"])
    impression = extract_section(txt, ["IMPRESSION"])
    return {"indication": indication, "findings": findings, "impression": impression}


import os

def drop_missing_dicoms_and_save_clean(df: pd.DataFrame) -> pd.DataFrame:
    dcm_path = df["dcm_path"].astype(str)
    exists = dcm_path.map(os.path.exists)
    n_missing = int((~exists).sum())
    print("Check DICOM existence. Missing:", n_missing, "/", len(df))
    if n_missing > 0:
        print("Example missing:", dcm_path[~exists].head(5).tolist())
        df = df.loc[exists].reset_index(drop=True)
    df.to_parquet(CLEAN_CACHE, index=False, engine="pyarrow", compression="snappy")
    print("Saved CLEAN_CACHE:", CLEAN_CACHE, "rows:", len(df))
    return df


def build_mimic_prompt(indication: str) -> str:
    ind = minimal_one_line(indication)
    return f"<start_of_image> {ind} findings:" if ind else "<start_of_image> findings:"

def build_target_text(findings: str, impression: str) -> str:
    return format_findings_impression_target(findings, impression)


import difflib

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+")

BULLET_RE = re.compile(r"^\s*([-*\u2022]|\d+\.|\d+\))\s+", flags=re.M)

DEPENDENCY_CUES = re.compile(
    r"\b(compared to|comparison|previous|prior|interval|again|as before|therefore|however|this|these|above)\b",
    flags=re.I
)

def split_sentences(text: str):
    t = norm_keep_newlines(text)
    if not t:
        return []
    if BULLET_RE.search(t):
        parts = [norm_keep_newlines(BULLET_RE.sub("", x, count=1)) for x in re.split(r"\n+", t)]
        parts = [x for x in parts if x]
        return parts if parts else [t]
    parts = [norm_keep_newlines(x) for x in _SENT_SPLIT.split(t) if norm_keep_newlines(x)]
    return parts if parts else [t]

def join_sentences(sents, style: int):
    sents = [norm_keep_newlines(s) for s in sents if norm_keep_newlines(s)]
    if not sents:
        return ""
    if style == 0:
        return " ".join(sents)
    if style == 1:
        return ". ".join([s.rstrip(".") for s in sents]) + ("" if sents[-1].endswith(".") else ".")
    if style == 2:
        return "; ".join([s.rstrip(".") for s in sents]) + ("" if sents[-1].endswith(".") else ".")
    if style == 3:
        return "\n".join([f"- {s.rstrip('.')}" for s in sents])
    return " ".join(sents)

def can_shuffle(sents):
    if len(sents) < 2:
        return False
    for s in sents:
        if DEPENDENCY_CUES.search(s):
            return False
    return True

def merge_short_adjacent(sents, min_len_chars=35):
    out = []
    i = 0
    while i < len(sents):
        cur = sents[i]
        if i + 1 < len(sents) and (len(cur) < min_len_chars or len(sents[i+1]) < min_len_chars):
            out.append(norm_keep_newlines(cur.rstrip(".") + "; " + sents[i+1].lstrip()))
            i += 2
        else:
            out.append(cur)
            i += 1
    return out

def structural_aug_once(text: str, rng: random.Random):
    return structural_aug_once_core(text, rng)

def show_diff(a: str, b: str, n=40):
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    diff = list(difflib.unified_diff(a_lines, b_lines, lineterm="", fromfile="orig", tofile="aug"))
    print("\n".join(diff[:n]))
    if len(diff) > n:
        print(f"... diff truncated, total lines: {len(diff)}")


LATERALITY_RE = re.compile(r"\b(right|left|bilateral)\b", flags=re.I)

SEVERITY_RE = re.compile(r"\b(mild|moderate|severe)\b", flags=re.I)

def token_set(rex, text: str):
    return set([t.lower() for t in rex.findall(text or "")])

def quick_safety_ok(src: str, cand: str):
    src1 = norm_for_filters(src)
    cand1 = norm_for_filters(cand)
    lat_src, lat_c = token_set(LATERALITY_RE, src1), token_set(LATERALITY_RE, cand1)
    sev_src, sev_c = token_set(SEVERITY_RE, src1), token_set(SEVERITY_RE, cand1)
    ok_lat = (lat_src == lat_c)
    ok_sev = (sev_src == sev_c)
    return ok_lat and ok_sev, {"lat_src": lat_src, "lat_cand": lat_c, "sev_src": sev_src, "sev_cand": sev_c}


import re

RADGRAPH_REWARD_LEVEL = os.environ.get("RADGRAPH_REWARD_LEVEL", "all")

RADGRAPH_MODEL_TYPE = os.environ.get("RADGRAPH_MODEL_TYPE", "modern-radgraph-xl")
E2A_USE_RADGRAPH = os.environ.get("E2A_USE_RADGRAPH", "0") == "1"
rg = None

BULLET_PREFIX_RE = re.compile(r"(?m)^\s*([-*\u2022]|\d+\)|\d+\.)\s+")


MULTISPACE_RE = re.compile(r"\s+")

def canonical_for_radgraph(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    t = BULLET_PREFIX_RE.sub("", t)
    t = t.replace("\n", ". ")
    t = re.sub(r"\s*;\s*", ". ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

TOK_CLEAN_RE = re.compile(r"[^a-z0-9]+")

def clean_tok(x: str) -> str:
    return TOK_CLEAN_RE.sub("", str(x).lower().strip())

def norm_tokens(tok):
    if tok is None:
        return tuple()
    if isinstance(tok, list):
        out = []
        for t in tok:
            ct = clean_tok(t)
            if ct:
                out.append(ct)
        return tuple(out)
    if isinstance(tok, str):
        parts = [clean_tok(p) for p in tok.split(" ")]
        parts = [p for p in parts if p]
        return tuple(parts)
    ct = clean_tok(tok)
    return tuple([ct]) if ct else tuple()

def rg_to_sets_v2(ann_dict):
    if not isinstance(ann_dict, dict) or len(ann_dict) == 0:
        return set(), set()

    k0 = next(iter(ann_dict.keys()))
    payload = ann_dict[k0]

    for _ in range(4):
        if isinstance(payload, dict) and "entities" in payload:
            break
        if isinstance(payload, dict) and len(payload) == 1:
            payload = next(iter(payload.values()))
            continue
        if isinstance(payload, dict) and "0" in payload and isinstance(payload["0"], dict):
            payload = payload["0"]
            continue
        break

    if not (isinstance(payload, dict) and "entities" in payload):
        return set(), set()

    entities = payload.get("entities", {}) or {}

    ent_map = {}
    ent_set = set()
    for ent_id, ent in entities.items():
        lab = str(ent.get("label", "")).strip()
        tok = norm_tokens(ent.get("tokens", []))
        key = (tok, lab)
        ent_map[str(ent_id)] = key
        ent_set.add(key)

    rel_set = set()
    for src_id, ent in entities.items():
        src_key = ent_map.get(str(src_id), None)
        if src_key is None:
            continue
        rels = ent.get("relations", []) or []
        for rel in rels:
            if not isinstance(rel, (list, tuple)) or len(rel) != 2:
                continue
            rel_type, tgt_id = str(rel[0]), str(rel[1])
            tgt_key = ent_map.get(str(tgt_id), None)
            if tgt_key is None:
                continue
            rel_set.add((src_key, rel_type, tgt_key))

    return ent_set, rel_set

def pr(ref_set, hyp_set):
    if len(ref_set) == 0 and len(hyp_set) == 0:
        return 1.0, 1.0
    if len(hyp_set) == 0:
        return 0.0, 0.0
    inter = len(ref_set & hyp_set)
    p = inter / max(1, len(hyp_set))
    r = inter / max(1, len(ref_set))
    return float(p), float(r)

def radgraph_pair_stats_v2(src: str, cand: str):
    assert rg is not None
    src_c = canonical_for_radgraph(src)
    cand_c = canonical_for_radgraph(cand)

    hyp_anns = rg([cand_c])
    ref_anns = rg([src_c])

    hyp_ent, hyp_rel = rg_to_sets_v2(hyp_anns)
    ref_ent, ref_rel = rg_to_sets_v2(ref_anns)

    ent_p, ent_r = pr(ref_ent, hyp_ent)
    rel_p, rel_r = pr(ref_rel, hyp_rel)

    return {
        "ent_p": ent_p, "ent_r": ent_r,
        "rel_p": rel_p, "rel_r": rel_r,
        "ref_ent": ref_ent, "hyp_ent": hyp_ent,
        "ref_rel": ref_rel, "hyp_rel": hyp_rel,
    }

def radgraph_ok_e2a(st):
    extra_ent = len(st["hyp_ent"] - st["ref_ent"])
    miss_ent  = len(st["ref_ent"] - st["hyp_ent"])
    extra_rel = len(st["hyp_rel"] - st["ref_rel"])
    miss_rel  = len(st["ref_rel"] - st["hyp_rel"])

    if E2A_RG_MODE == "strict":
        if extra_ent != 0:
            return False, "extra_entities"
        if miss_ent != 0:
            return False, "missing_entities"
        if miss_rel != 0:
            return False, "missing_relations"
        if extra_rel > MAX_EXTRA_REL:
            return False, "too_many_extra_relations"
        if st["rel_p"] < MIN_REL_P:
            return False, "rel_precision_low"
        if st["ent_p"] < MIN_ENT_P or st["ent_r"] < MIN_ENT_R:
            return False, "entity_pr_low"
        return True, "ok"

    # if st["ent_p"] < MIN_ENT_P:
    #     return False, "ent_precision_low"
    if st["ent_r"] < MIN_ENT_R:
        return False, "ent_recall_low"
    if st["rel_p"] < MIN_REL_P:
        return False, "rel_precision_low"
    if st["rel_r"] < MIN_REL_R:
        return False, "rel_recall_low"

    if extra_ent > MAX_EXTRA_ENT:
        return False, "too_many_extra_entities"
    if miss_ent > MAX_MISSING_ENT:
        return False, "too_many_missing_entities"
    if miss_rel > MAX_MISSING_REL:
        return False, "too_many_missing_relations"
    if extra_rel > MAX_EXTRA_REL:
        return False, "too_many_extra_relations"

    return True, "ok"

def debug_rg_diff(src: str, cand: str, top_k: int = 5):
    st = radgraph_pair_stats_v2(src, cand)
    ok, reason = radgraph_ok_e2a(st)

    extra_rel = list(st["hyp_rel"] - st["ref_rel"])
    miss_rel  = list(st["ref_rel"] - st["hyp_rel"])

    print("RadGraph stats v2:", {k: st[k] for k in ["ent_p","ent_r","rel_p","rel_r"]})
    print("Mode:", E2A_RG_MODE, "ok:", ok, "reason:", reason)
    print("Extra_rel:", len(extra_rel), "Missing_rel:", len(miss_rel))
    for x in extra_rel[:top_k]:
        print("  extra:", x)
    for x in miss_rel[:top_k]:
        print("  miss :", x)


def debug_try_k_attempts_on_one_sample(row, k=10):
    sid = int(row["study_id"])
    f = minimal_one_line(row["ref_findings"])
    i = minimal_one_line(row["ref_impression"])
    src_full = build_target_text(f, i)

    rng = random.Random(SEED + sid % 10_000_000)

    print("study_id:", sid)
    print("Trying K candidates:", k, "AUG_PER_SAMPLE currently:", AUG_PER_SAMPLE)
    ok_cnt = 0

    for j in range(k):
        af = structural_aug_once(f, rng)
        ai = structural_aug_once(i, rng)
        cand_full = build_target_text(af, ai)

        ok_q, _ = quick_safety_ok(src_full, cand_full)

        if (not E2A_USE_RADGRAPH) or rg is None:
            ok_rg, reason = True, "rg_disabled"
            st = {"ent_p": None, "ent_r": None, "rel_p": None, "rel_r": None}
        else:
            st = radgraph_pair_stats_v2(src_full, cand_full)
            ok_rg, reason = radgraph_ok_e2a(st)

        ok_all = ok_q and ok_rg
        ok_cnt += int(ok_all)

        print(f"\n[{j}] ok_q={ok_q} ok_rg={ok_rg} reason={reason}")
        if E2A_USE_RADGRAPH and rg is not None:
            print("   stats:", {k: st[k] for k in ["ent_p","ent_r","rel_p","rel_r"]})
            extra_ent = len(st["hyp_ent"] - st["ref_ent"])
            miss_ent  = len(st["ref_ent"] - st["hyp_ent"])
            extra_rel = len(st["hyp_rel"] - st["ref_rel"])
            miss_rel  = len(st["ref_rel"] - st["hyp_rel"])
            print("   diffs:", {"extra_ent": extra_ent, "miss_ent": miss_ent, "extra_rel": extra_rel, "miss_rel": miss_rel})

    print("\nPassed candidates:", ok_cnt, "/", k)


def score_candidate(stats):
    extra_ent = len(stats["hyp_ent"] - stats["ref_ent"])
    miss_ent  = len(stats["ref_ent"] - stats["hyp_ent"])
    extra_rel = len(stats["hyp_rel"] - stats["ref_rel"])
    miss_rel  = len(stats["ref_rel"] - stats["hyp_rel"])
    return (
        - 1000 * miss_ent
        - 500 * miss_rel
        - 50 * extra_ent
        - 10 * extra_rel
        + 10 * stats["ent_p"]
        + 10 * stats["ent_r"]
        + 5 * stats["rel_p"]
        + 5 * stats["rel_r"]
    )

def build_e2a_bank(df_train: pd.DataFrame, out_path: Path, max_rows: int = 5000):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_src = df_train.copy()
    if max_rows != -1 and len(df_src) > max_rows:
        df_src = df_src.sample(n=max_rows, random_state=SEED).reset_index(drop=True)

    print("Building E2a bank on rows:", len(df_src))
    accepted = []
    rejected = {}

    for r in tqdm(df_src.to_dict("records"), desc="E2a build"):
        sid = r["study_id"]
        f = minimal_one_line(r["ref_findings"])
        i = minimal_one_line(r["ref_impression"])
        if not f or not i:
            rejected["empty_ref"] = rejected.get("empty_ref", 0) + 1
            continue

        src_full = build_target_text(f, i)
        rng = random.Random(SEED + int(sid) % 10_000_000)

        best = None
        best_score = None
        best_stats = None

        for _ in range(AUG_PER_SAMPLE):
            af = structural_aug_once(f, rng)
            ai = structural_aug_once(i, rng)
            cand_full = build_target_text(af, ai)

            if not cand_full:
                rejected["empty_cand"] = rejected.get("empty_cand", 0) + 1
                continue
            if norm_for_filters(cand_full) == norm_for_filters(src_full):
                rejected["identical"] = rejected.get("identical", 0) + 1
                continue
            if abs(len(norm_for_filters(cand_full)) - len(norm_for_filters(src_full))) > MAX_CHAR_DELTA:
                rejected["len_delta"] = rejected.get("len_delta", 0) + 1
                continue

            ok_q, _ = quick_safety_ok(src_full, cand_full)
            if not ok_q:
                rejected["quick_safety"] = rejected.get("quick_safety", 0) + 1
                continue

            if E2A_USE_RADGRAPH and rg is not None:
                st = radgraph_pair_stats_v2(src_full, cand_full)
                ok_rg, reason = radgraph_ok_e2a(st)
                if not ok_rg:
                    key = "radgraph_" + reason
                    rejected[key] = rejected.get(key, 0) + 1
                    continue
                sc = score_candidate(st)
            else:
                st = {"ent_p": None, "ent_r": None, "rel_p": None, "rel_r": None, "ref_ent": set(), "hyp_ent": set(), "ref_rel": set(), "hyp_rel": set()}
                sc = 0.0

            if best is None or sc > best_score:
                best = (af, ai, cand_full)
                best_score = sc
                best_stats = st

        if best is not None:
            af, ai, cand_full = best
            accepted.append({
                "study_id": str(sid),
                "aug_findings": norm_keep_newlines(af),
                "aug_impression": norm_keep_newlines(ai),
                "aug_full": norm_keep_newlines(cand_full),
                "rg_ent_p": None if best_stats is None else best_stats.get("ent_p"),
                "rg_ent_r": None if best_stats is None else best_stats.get("ent_r"),
                "rg_rel_p": None if best_stats is None else best_stats.get("rel_p"),
                "rg_rel_r": None if best_stats is None else best_stats.get("rel_r"),
            })

    df_acc = pd.DataFrame(accepted)
    df_acc.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

    print("Saved E2a bank:", out_path)
    print("Accepted:", len(df_acc))
    print("Rejected breakdown (top 20):")
    for k, v in sorted(rejected.items(), key=lambda kv: kv[1], reverse=True)[:20]:
        print(" ", k, ":", v)

    if len(df_acc) > 0:
        print("\nExample accepted rows:")
        print(df_acc.head(3)[["study_id","rg_ent_p","rg_rel_p"]])
        print("Example aug_full (first 300 chars):")
        print(df_acc.iloc[0]["aug_full"][:300])

    return df_acc, rejected


def load_e2a_map(path: Path):
    if not path.exists():
        return {}
    df_c = pd.read_parquet(path)
    if len(df_c) == 0:
        return {}
    m = {}
    for sid, g in df_c.groupby("study_id"):
        m[str(sid)] = g["aug_full"].fillna("").astype(str).tolist()
    return m


import pandas as pd

def _ent_to_str(ent):
    # ent = (tokens_tuple, label)
    toks, lab = ent
    txt = " ".join(toks) if isinstance(toks, tuple) else str(toks)
    return f"{lab} :: {txt}"

def _rel_to_str(rel):
    # rel = ((src_tokens,label), rel_type, (tgt_tokens,label))
    src, rel_type, tgt = rel
    return f"{_ent_to_str(src)}  --{rel_type}-->  {_ent_to_str(tgt)}"

def _sorted_ents(s):
    return sorted(list(s), key=lambda e: (_ent_to_str(e)))

def _sorted_rels(s):
    return sorted(list(s), key=lambda r: (_rel_to_str(r)))

def explain_radgraph_diff_for_study(study_id: str, top_k: int = 25):
    assert rg is not None, "RadGraph не инициализирован (rg is None)."
    assert study_id in textaug_map, "study_id нет в textaug_map."

    # 1) исходный текст из df_train
    row = df_train[df_train["study_id"].astype(str) == str(study_id)].iloc[0]
    src_full = build_target_text(row["ref_findings"], row["ref_impression"])

    # 2) аугмент (берём первый вариант)
    cand_full = textaug_map[str(study_id)][0]

    print("study_id:", study_id)
    print("\nSOURCE (orig) first 500 chars:\n", src_full[:500], "...\n")
    print("CANDIDATE (aug) first 500 chars:\n", cand_full[:500], "...\n")

    # 3) RadGraph stats + множества
    st = radgraph_pair_stats_v2(src_full, cand_full)

    E_ref, E_hyp = st["ref_ent"], st["hyp_ent"]
    R_ref, R_hyp = st["ref_rel"], st["hyp_rel"]

    E_inter = E_ref & E_hyp
    R_inter = R_ref & R_hyp

    E_extra = E_hyp - E_ref
    E_miss  = E_ref - E_hyp

    R_extra = R_hyp - R_ref
    R_miss  = R_ref - R_hyp

    # 4) вычислим p/r вручную (чтобы было видно происхождение)
    ent_p = len(E_inter) / max(1, len(E_hyp))
    ent_r = len(E_inter) / max(1, len(E_ref))
    rel_p = len(R_inter) / max(1, len(R_hyp))
    rel_r = len(R_inter) / max(1, len(R_ref))

    print("Counts:")
    print("  |E_ref| =", len(E_ref), " |E_hyp| =", len(E_hyp), " |E_inter| =", len(E_inter))
    print("  |R_ref| =", len(R_ref), " |R_hyp| =", len(R_hyp), " |R_inter| =", len(R_inter))

    print("\nComputed metrics (by counts):")
    print("  ent_p = |E_inter| / |E_hyp| =", f"{len(E_inter)}/{max(1,len(E_hyp))} =", ent_p)
    print("  ent_r = |E_inter| / |E_ref| =", f"{len(E_inter)}/{max(1,len(E_ref))} =", ent_r)
    print("  rel_p = |R_inter| / |R_hyp| =", f"{len(R_inter)}/{max(1,len(R_hyp))} =", rel_p)
    print("  rel_r = |R_inter| / |R_ref| =", f"{len(R_inter)}/{max(1,len(R_ref))} =", rel_r)

    print("\nMetrics returned by radgraph_pair_stats_v2:")
    print("  ent_p:", st["ent_p"], " ent_r:", st["ent_r"], " rel_p:", st["rel_p"], " rel_r:", st["rel_r"])

    # 5) печать отличий
    print("\nExtra entities (E_hyp \\ E_ref):", len(E_extra))
    for x in _sorted_ents(E_extra)[:top_k]:
        print(" ", _ent_to_str(x))

    print("\nMissing entities (E_ref \\ E_hyp):", len(E_miss))
    for x in _sorted_ents(E_miss)[:top_k]:
        print(" ", _ent_to_str(x))

    print("\nExtra relations (R_hyp \\ R_ref):", len(R_extra))
    for x in _sorted_rels(R_extra)[:top_k]:
        print(" ", _rel_to_str(x))

    print("\nMissing relations (R_ref \\ R_hyp):", len(R_miss))
    for x in _sorted_rels(R_miss)[:top_k]:
        print(" ", _rel_to_str(x))


import torch

from torch.utils.data import Dataset

from transformers import AutoProcessor

import difflib

def show_short_diff(a: str, b: str, n=15):
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    diff = list(difflib.unified_diff(a_lines, b_lines, lineterm="", fromfile="orig", tofile="aug"))
    print("\n".join(diff[:n]))
    if len(diff) > n:
        print(f"... diff truncated, total lines: {len(diff)}")

class MimicSFTDataset_E2a(Dataset):
    def __init__(self, df_: pd.DataFrame, processor: AutoProcessor, textaug_map: dict, apply_p: float, seed: int, debug_first_n: int = 3, max_text_len: int = 768):
        self.df = df_.reset_index(drop=True)
        self.processor = processor
        self.textaug_map = textaug_map
        self.apply_p = float(apply_p)
        self.seed = int(seed)
        self.debug_first_n = int(debug_first_n)
        self.max_text_len = int(max_text_len)

    def __len__(self):
        return len(self.df)

    def choose_target(self, idx: int, study_id, ref_findings: str, ref_impression: str):
        base = build_target_text(ref_findings, ref_impression)
        alts = self.textaug_map.get(str(study_id), [])
        rng = random.Random(self.seed * 1_000_003 + idx)
        use_aug = (len(alts) > 0) and (rng.random() < self.apply_p)
        chosen = rng.choice(alts) if use_aug else base
        return base, chosen, use_aug, len(alts)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx].to_dict()
        img = safe_open_mimic_dicom(Path(r["dcm_path"]))

        prompt = build_mimic_prompt(r["indication"])
        base, chosen, use_aug, n_alts = self.choose_target(idx, r["study_id"], r["ref_findings"], r["ref_impression"])

        full_text = f"{prompt} {norm_for_filters(chosen)}".strip()

        enc = self.processor(
            text=full_text,
            images=img,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_text_len,
            do_resize=False,
        )

        prompt_ids = self.processor.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_text_len,
        )["input_ids"]
        prompt_len = len(prompt_ids)

        input_ids = enc["input_ids"][0]
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        if idx < self.debug_first_n:
            print("\n=== DEBUG DATASET ITEM ===")
            print("idx:", idx, "study_id:", r["study_id"])
            print("n_alts:", n_alts, "use_aug:", use_aug, "APPLY_P:", APPLY_P)
            print("prompt:", prompt[:200], "...")
            print("base_target:", base[:250], "...")
            print("chosen_target(one-line):", norm_for_filters(chosen)[:250], "...")
            if use_aug:
                print("diff (first lines):")
                show_short_diff(base, norm_for_filters(chosen), n=15)

        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "pixel_values": enc["pixel_values"][0],
            "labels": labels,
        }


import os

import math

from uuid import uuid4

import torch

from transformers import AutoModelForImageTextToText, TrainingArguments, Trainer

from peft import LoraConfig, get_peft_model

FT_OUT_ROOT = Path(os.environ.get("FT_OUT_ROOT", "E2a_finetune_outputs"))

USE_E2A_ON_TRAIN = os.environ.get("USE_E2A_ON_TRAIN", "1") == "1"

FT_MAX_TRAIN = int(os.environ.get("FT_MAX_TRAIN", "-1"))

FT_MAX_VALID = int(os.environ.get("FT_MAX_VALID", "2000"))

FT_BS = int(os.environ.get("FT_BS", "1"))

FT_GAS = int(os.environ.get("FT_GAS", "16"))

FT_LR = float(os.environ.get("FT_LR", "1e-4"))

FT_WD = float(os.environ.get("FT_WD", "0.0"))

FT_WARMUP = float(os.environ.get("FT_WARMUP", "0.03"))

FT_SCHED = os.environ.get("FT_SCHED", "cosine")

FT_EPOCHS = float(os.environ.get("FT_EPOCHS", "1.0"))

FT_LOG_STEPS = int(os.environ.get("FT_LOG_STEPS", "50"))

FT_EVAL_STEPS = int(os.environ.get("FT_EVAL_STEPS", "500"))

FT_SAVE_STEPS = int(os.environ.get("FT_SAVE_STEPS", "500"))

EVAL_MAX_NEW_TOKENS = int(os.environ.get("EVAL_MAX_NEW_TOKENS", "256"))

GEN_KWARGS = dict(do_sample=False, num_beams=1, max_new_tokens=EVAL_MAX_NEW_TOKENS)


from typing import List, Dict, Any

class DataCollatorPad:
    def __init__(self, processor_):
        self.processor = processor_

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]

        padded = self.processor.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )
        max_len = padded["input_ids"].shape[1]

        padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, lab in enumerate(labels):
            padded_labels[i, : lab.shape[0]] = lab

        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)

        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "pixel_values": pixel_values,
            "labels": padded_labels,
        }

def attach_lora(model_):
    cfg = LoraConfig(
        r=int(os.environ.get("LORA_R", "8")),
        lora_alpha=int(os.environ.get("LORA_ALPHA", "16")),
        lora_dropout=float(os.environ.get("LORA_DROPOUT", "0.05")),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    m = get_peft_model(model_, cfg)
    print("[LoRA] Trainable parameters:")
    m.print_trainable_parameters()
    return m


import evaluate

from tqdm.auto import tqdm

def parse_pred_findings_impression(gen_text: str):
    t = norm_for_filters(gen_text)
    if not t:
        return "", ""
    m = re.search(r"(?i)\bimpression\s*:\s*", t)
    if m:
        findings = norm_for_filters(t[:m.start()])
        impression = norm_for_filters(t[m.end():])
        return findings, impression
    return t, ""

def move_to_device(batch, device, dtype):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if k == "pixel_values":
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out

def generate_reports(model, df_eval: pd.DataFrame, batch_size: int = 1, max_new_tokens: int = 256):
    model.eval()
    device = next(model.parameters()).device

    rows = df_eval.to_dict("records")
    raw_preds, preds_find, preds_impr = [], [], []
    refs_find, refs_impr, keys = [], [], []

    for i in tqdm(range(0, len(rows), batch_size), desc="Generate"):
        batch_rows = rows[i:i+batch_size]
        images = [safe_open_mimic_dicom(Path(r["dcm_path"])) for r in batch_rows]
        prompts = [build_mimic_prompt(r["indication"]) for r in batch_rows]

        enc = processor(text=prompts, images=images, return_tensors="pt", padding=True, do_resize=False)
        enc = move_to_device(enc, device, DTYPE)

        gen_ids = model.generate(**enc, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens)
        input_len = enc["input_ids"].shape[1]
        gen_text = processor.batch_decode(gen_ids[:, input_len:], skip_special_tokens=True)
        gen_text = [norm_for_filters(t) for t in gen_text]

        for r, t in zip(batch_rows, gen_text):
            raw_preds.append(t)
            pf, pi = parse_pred_findings_impression(t)
            preds_find.append(pf)
            preds_impr.append(pi)
            refs_find.append(norm_for_filters(r["ref_findings"]))
            refs_impr.append(norm_for_filters(r["ref_impression"]))
            keys.append(str(r["study_id"]))

    out = pd.DataFrame({
        "study_id": keys,
        "raw_pred": raw_preds,
        "ref_findings": refs_find,
        "pred_findings": preds_find,
        "ref_impression": refs_impr,
        "pred_impression": preds_impr,
    })
    out["ref_full"] = (out["ref_findings"] + " " + out["ref_impression"]).map(norm_for_filters)
    out["pred_full"] = out["raw_pred"].map(norm_for_filters)
    out["pred_full_concat"] = (out["pred_findings"] + " " + out["pred_impression"]).map(norm_for_filters)
    return out


rouge = None
bleu = None

def _ensure_text_metric_objects():
    global rouge, bleu
    if rouge is None:
        rouge = evaluate.load("rouge")
    if bleu is None:
        bleu = evaluate.load("sacrebleu")

def compute_text_metrics(preds, refs):
    _ensure_text_metric_objects()
    return {
        "n": len(preds),
        "rouge": rouge.compute(predictions=preds, references=refs, use_stemmer=True),
        "sacrebleu": bleu.compute(predictions=preds, references=[[r] for r in refs]),
    }

def compute_radgraph_f1(preds, refs, model_type="modern-radgraph-xl", reward_level="all"):
    from radgraph import F1RadGraph
    f1rg = F1RadGraph(reward_level=reward_level, model_type=model_type)
    mean_reward, _, _, _ = f1rg(hyps=preds, refs=refs)
    rg_e, rg_er, rg_bar_er = mean_reward
    return {"model_type": model_type, "n": len(preds), "rg_e": float(rg_e), "rg_er": float(rg_er), "rg_bar_er": float(rg_bar_er)}

def compute_radgraph_detail_v2(preds, refs):
    assert rg is not None, "rg is None. Initialize RadGraph earlier."
    ent_p, ent_r, rel_p, rel_r = [], [], [], []
    extra_ent, extra_rel = [], []
    miss_ent, miss_rel = [], []

    for hyp, ref in tqdm(list(zip(preds, refs)), desc="RadGraph detail v2", total=len(preds)):
        st = radgraph_pair_stats_v2(ref, hyp)  # NOTE: function expects (src, cand); we interpret src=ref, cand=hyp
        ent_p.append(st["ent_p"]); ent_r.append(st["ent_r"])
        rel_p.append(st["rel_p"]); rel_r.append(st["rel_r"])
        extra_ent.append(len(st["hyp_ent"] - st["ref_ent"]))
        extra_rel.append(len(st["hyp_rel"] - st["ref_rel"]))
        miss_ent.append(len(st["ref_ent"] - st["hyp_ent"]))
        miss_rel.append(len(st["ref_rel"] - st["hyp_rel"]))

    def mean(x): return float(np.mean(x)) if len(x) else 0.0
    return {
        "n": len(preds),
        "ent_p_mean": mean(ent_p),
        "ent_r_mean": mean(ent_r),
        "rel_p_mean": mean(rel_p),
        "rel_r_mean": mean(rel_r),
        "H_ent_extra": mean(extra_ent),
        "H_rel_extra": mean(extra_rel),
        "H_ent_missing": mean(miss_ent),
        "H_rel_missing": mean(miss_rel),
    }

def length_stats(texts):
    lens = [len((t or "").split()) for t in texts]
    return {"mean_tokens": float(np.mean(lens)), "p50": float(np.median(lens)), "p90": float(np.quantile(lens, 0.9)), "max": int(np.max(lens))}
