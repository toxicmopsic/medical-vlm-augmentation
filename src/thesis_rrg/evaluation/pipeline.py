from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from thesis_rrg.evaluation.aggregations import build_predictions_dataframe
from thesis_rrg.evaluation.error_analysis import (
    generation_pathology_report,
    generation_pathology_summary,
    top_length_outliers,
)
from thesis_rrg.evaluation.export import export_metrics, export_predictions
from thesis_rrg.evaluation.inference import run_inference
from thesis_rrg.metrics.reportgen_suite import run_reportgen_suite
from thesis_rrg.models.factory import build_model_bundle
from thesis_rrg.training.train_utils import build_dataset_bundle
from thesis_rrg.utils.hf_utils import try_get_git_commit
from thesis_rrg.utils.logging import setup_logging
from thesis_rrg.utils.paths import save_git_commit, save_hydra_config, setup_run_dirs
from thesis_rrg.utils.seed import seed_everything
from thesis_rrg.utils.config_utils import dataset_bundle_summary
from thesis_rrg.utils.io import write_json



def _resolve_eval_df(bundle, split: str):
    s = split.strip().lower()
    if s in {"val", "valid", "validate"}:
        return bundle.valid_df
    if s == "test":
        return bundle.test_df
    if s == "train":
        return bundle.train_df
    raise ValueError(f"Unsupported eval split: {split}")



def run_evaluation(cfg: DictConfig) -> dict:
    output_dir = Path(str(cfg.paths.output_dir))
    setup_run_dirs(output_dir)
    logger = setup_logging(output_dir / "eval.log")

    seed_everything(int(cfg.seed), deterministic=False)
    save_hydra_config(cfg, output_dir)
    save_git_commit(output_dir, try_get_git_commit(Path(str(cfg.paths.root_dir))))

    bundle = build_dataset_bundle(cfg)
    bundle_summary = dataset_bundle_summary(bundle)
    write_json(output_dir / "metrics" / "dataset_bundle.json", bundle_summary)
    logger.info("Dataset bundle: %s", bundle_summary)

    eval_df = _resolve_eval_df(bundle, str(cfg.data.eval_split))

    model_bundle = build_model_bundle(cfg, for_train=False)
    rows = run_inference(cfg, model_bundle.model, model_bundle.processor, df_eval=eval_df)

    pred_df = build_predictions_dataframe(rows)
    radeval_cfg = OmegaConf.to_container(cfg.metrics.get("radeval", {}), resolve=True)
    if not isinstance(radeval_cfg, dict):
        radeval_cfg = {}

    metrics = run_reportgen_suite(
        preds_findings=pred_df["pred_findings"].fillna("").astype(str).tolist(),
        refs_findings=pred_df["ref_findings"].fillna("").astype(str).tolist(),
        preds_impression=pred_df["pred_impression"].fillna("").astype(str).tolist(),
        refs_impression=pred_df["ref_impression"].fillna("").astype(str).tolist(),
        preds_full=pred_df["pred_full"].fillna("").astype(str).tolist(),
        refs_full=pred_df["ref_full"].fillna("").astype(str).tolist(),
        radgraph_model_type=str(cfg.metrics.radgraph_model_type),
        radgraph_reward_level=str(cfg.metrics.radgraph_reward_level),
        radeval_cfg=radeval_cfg,
    )

    metrics["n_eval"] = int(len(pred_df))
    metrics["split"] = str(cfg.data.eval_split)
    metrics["dataset_bundle"] = bundle_summary

    export_predictions(pred_df, output_dir / "predictions" / f"{cfg.data.eval_split}_predictions.csv")
    export_metrics(metrics, output_dir / "metrics" / f"{cfg.data.eval_split}_metrics.json")

    outliers = top_length_outliers(pred_df)
    if len(outliers):
        outliers.to_csv(output_dir / "predictions" / f"{cfg.data.eval_split}_length_outliers.csv", index=False)

    flags_df = generation_pathology_report(pred_df)
    if len(flags_df):
        flags_path = output_dir / "predictions" / f"{cfg.data.eval_split}_generation_flags.csv"
        flags_df.to_csv(flags_path, index=False)
        write_json(
            output_dir / "metrics" / f"{cfg.data.eval_split}_generation_pathology_summary.json",
            generation_pathology_summary(flags_df),
        )
        
    logger.info("Evaluation finished: %s samples", len(pred_df))
    return metrics
