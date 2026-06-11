from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from thesis_rrg.evaluation.aggregations import build_predictions_dataframe
from thesis_rrg.evaluation.export import export_predictions
from thesis_rrg.evaluation.inference import run_inference
from thesis_rrg.models.factory import build_model_bundle
from thesis_rrg.training.train_utils import build_dataset_bundle
from thesis_rrg.utils.logging import setup_logging
from thesis_rrg.utils.paths import save_hydra_config, setup_run_dirs
from thesis_rrg.utils.seed import seed_everything



def _resolve_df(bundle, split: str):
    split = split.strip().lower()
    if split in {"valid", "val", "validate"}:
        return bundle.valid_df
    if split == "test":
        return bundle.test_df
    if split == "train":
        return bundle.train_df
    raise ValueError(f"Unsupported predict split: {split}")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="predict")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    output_dir = Path(str(cfg.paths.output_dir))
    setup_run_dirs(output_dir)
    setup_logging(output_dir / "predict.log")
    save_hydra_config(cfg, output_dir)
    seed_everything(int(cfg.seed))

    bundle = build_dataset_bundle(cfg)
    print(bundle.meta)
    print('image_aug_mode=', bundle.image_aug_mode)
    print('textaug_apply_p=', bundle.textaug_apply_p, 'textaug_map_size=', len(bundle.textaug_map))
    df_eval = _resolve_df(bundle, str(cfg.data.predict_split))

    model_bundle = build_model_bundle(cfg, for_train=False)
    rows = run_inference(cfg, model_bundle.model, model_bundle.processor, df_eval=df_eval)
    pred_df = build_predictions_dataframe(rows)

    export_predictions(pred_df, output_dir / "predictions" / f"{cfg.data.predict_split}_predictions.csv")


if __name__ == "__main__":
    main()
