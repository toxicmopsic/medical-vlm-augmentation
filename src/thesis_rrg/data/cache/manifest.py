from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

from thesis_rrg.utils.io import write_json


@dataclass
class CacheManifest:
    cache_tag: str
    clean_cache_path: str
    prepared_cache_path: str
    row_count: int



def save_manifest(manifest: CacheManifest, path: str | Path) -> None:
    write_json(path, asdict(manifest))
