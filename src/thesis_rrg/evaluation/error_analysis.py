from __future__ import annotations
import re
import pandas as pd

TIME_RE = re.compile(r"at\s+\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)?", re.IGNORECASE)
IMPRESSION_RE = re.compile(r"(?i)\bimpression\s*:")
FINDINGS_RE = re.compile(r"(?i)\bfindings\s*:")
WORD_RE = re.compile(r"\b\w+\b")

def top_length_outliers(df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    if len(df) == 0:
        return df

    out = df.copy()
    out["pred_len"] = out["pred_full"].fillna("").astype(str).str.len()
    out["ref_len"] = out["ref_full"].fillna("").astype(str).str.len()
    out["len_abs_delta"] = (out["pred_len"] - out["ref_len"]).abs()
    return out.sort_values("len_abs_delta", ascending=False).head(k)

def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def _max_punct_run(text: str) -> tuple[int, str]:
    best_len = 0
    best_char = ""
    for match in re.finditer(r"([_\-])\1{9,}", text or ""):
        if len(match.group(0)) > best_len:
            best_len = len(match.group(0))
            best_char = match.group(1)
    return best_len, best_char


def _max_consecutive_ngram_repeat(text: str, n_max: int = 5) -> tuple[int, str]:
    tokens = re.findall(r"[a-z0-9:/.]+", (text or "").lower())
    best_count = 1
    best_gram = ""
    for n in range(1, n_max + 1):
        i = 0
        while i + n <= len(tokens):
            gram = tokens[i : i + n]
            j = i + n
            count = 1
            while j + n <= len(tokens) and tokens[j : j + n] == gram:
                count += 1
                j += n
            if count > best_count:
                best_count = count
                best_gram = " ".join(gram)
            i += 1
    return best_count, best_gram


def generation_pathology_report(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in df.fillna("").to_dict("records"):
        raw = str(row.get("raw_pred", ""))
        pred_findings = str(row.get("pred_findings", ""))
        pred_impression = str(row.get("pred_impression", ""))
        max_punct_len, max_punct_char = _max_punct_run(raw)
        repeat_count, repeat_gram = _max_consecutive_ngram_repeat(raw)
        impression_count = len(IMPRESSION_RE.findall(raw))
        time_count = len(TIME_RE.findall(raw))
        records.append(
            {
                "study_id": str(row.get("study_id", "")),
                "raw_chars": len(raw),
                "raw_words": _word_count(raw),
                "findings_empty": not pred_findings.strip(),
                "impression_empty": not pred_impression.strip(),
                "findings_word_le_1": _word_count(pred_findings) <= 1,
                "impression_word_le_1": _word_count(pred_impression) <= 1,
                "raw_starts_impression": bool(re.match(r"(?i)^\s*/?\s*impression\s*:", raw)),
                "raw_starts_and": bool(re.match(r"(?i)^\s*and\b", raw)),
                "raw_starts_slash": bool(re.match(r"^\s*/", raw)),
                "raw_has_findings_header": bool(FINDINGS_RE.search(raw)),
                "impression_marker_count": impression_count,
                "multiple_impression_markers": impression_count > 1,
                "no_impression_marker": impression_count == 0,
                "time_like_count": time_count,
                "has_time_like": time_count > 0,
                "max_punct_run_len": max_punct_len,
                "max_punct_run_char": max_punct_char,
                "long_punct_run": max_punct_len >= 30,
                "repeat_count": repeat_count,
                "repeat_gram": repeat_gram,
                "has_ngram_repeat_ge_5": repeat_count >= 5,
                "raw_pred_preview": raw[:240],
                "pred_findings_preview": pred_findings[:160],
                "pred_impression_preview": pred_impression[:160],
                "ref_impression_preview": str(row.get("ref_impression", ""))[:160],
            }
        )
    return pd.DataFrame.from_records(records)


def generation_pathology_summary(flags_df: pd.DataFrame) -> dict:
    if len(flags_df) == 0:
        return {"n": 0}

    bool_cols = [
        "findings_empty",
        "impression_empty",
        "findings_word_le_1",
        "impression_word_le_1",
        "raw_starts_impression",
        "raw_starts_and",
        "raw_starts_slash",
        "raw_has_findings_header",
        "multiple_impression_markers",
        "no_impression_marker",
        "has_time_like",
        "long_punct_run",
        "has_ngram_repeat_ge_5",
    ]
    out = {"n": int(len(flags_df))}
    for col in bool_cols:
        if col in flags_df:
            out[col] = int(flags_df[col].sum())
    out["time_like_total_occurrences"] = int(flags_df["time_like_count"].sum())
    out["time_like_max_one_row"] = int(flags_df["time_like_count"].max())
    out["max_punct_run_len"] = int(flags_df["max_punct_run_len"].max())
    out["max_repeat_count"] = int(flags_df["repeat_count"].max())
    return out