from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from thesis_rrg.constants import DEFAULT_GIT_COMMIT_FILE, DEFAULT_HYDRA_CONFIG_NAME, RUN_SUBDIRS
from thesis_rrg.utils.io import ensure_dir



def setup_run_dirs(output_dir: str | Path) -> Path:
    root = ensure_dir(output_dir)
    for rel in RUN_SUBDIRS:
        ensure_dir(root / rel)
    return root



def save_hydra_config(cfg: DictConfig, output_dir: str | Path) -> Path:
    out = Path(output_dir) / DEFAULT_HYDRA_CONFIG_NAME
    out.write_text(OmegaConf.to_yaml(cfg))
    return out



def save_git_commit(output_dir: str | Path, commit: str | None) -> Path:
    out = Path(output_dir) / DEFAULT_GIT_COMMIT_FILE
    out.write_text((commit or "unknown") + "\n")
    return out
