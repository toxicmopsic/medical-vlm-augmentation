from __future__ import annotations

from pathlib import Path

from thesis_rrg.data.cache.cache_utils import file_sha1



def fingerprint_paths(paths: list[str | Path]) -> dict[str, str]:
    out = {}
    for p in paths:
        pp = Path(p)
        if pp.exists() and pp.is_file():
            out[str(pp)] = file_sha1(pp)
    return out
