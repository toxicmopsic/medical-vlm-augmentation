from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from thesis_rrg.training.hf_trainer import run_hf_trainer
from thesis_rrg.training.trl_sft import run_trl_sft
from thesis_rrg.utils.hf_utils import try_get_git_commit
from thesis_rrg.utils.io import write_json, write_text
from thesis_rrg.utils.logging import setup_logging
from thesis_rrg.utils.paths import save_git_commit, save_hydra_config, setup_run_dirs
from thesis_rrg.utils.seed import seed_everything



def run_training(cfg: DictConfig) -> dict:
    output_dir = Path(str(cfg.paths.output_dir))
    setup_run_dirs(output_dir)

    logger = setup_logging(output_dir / "train.log")
    seed_everything(int(cfg.seed), deterministic=bool(cfg.get("deterministic", False)))

    save_hydra_config(cfg, output_dir)
    commit = try_get_git_commit(Path(str(cfg.paths.root_dir)))
    save_git_commit(output_dir, commit)

    backend = str(cfg.training.backend).strip().lower()
    if backend == "hf_trainer":
        result = run_hf_trainer(cfg, output_dir=output_dir, logger=logger)
    elif backend == "trl_sft":
        result = run_trl_sft(cfg, output_dir=output_dir, logger=logger)
    else:
        raise ValueError(f"Unsupported training backend: {backend}")

    # Keep a stable run artifact layout even when evaluation is triggered later.
    write_text(output_dir / "predictions" / "valid_predictions.csv", "")
    write_text(output_dir / "predictions" / "test_predictions.csv", "")
    write_json(output_dir / "metrics" / "valid_metrics.json", {"status": "not_computed_in_train_stage"})
    write_json(output_dir / "metrics" / "test_metrics.json", {"status": "not_computed_in_train_stage"})

    write_json(output_dir / "metrics" / "train_pipeline_summary.json", result)
    return result
