from __future__ import annotations

import inspect
from pathlib import Path

from transformers import Trainer, TrainingArguments

from thesis_rrg.models.factory import build_model_bundle
from thesis_rrg.models.model_utils import save_trainer_model_artifact
from thesis_rrg.training.args_utils import add_optional_training_arguments
from thesis_rrg.training.callbacks import PeftCheckpointCallback, TrainingProgressCallback
from thesis_rrg.training.checkpointing import prepare_checkpoint_dirs, write_best_checkpoint_pointer
from thesis_rrg.training.reporting import ensure_reporting_dependencies, report_to_includes_tensorboard, resolve_report_to
from thesis_rrg.training.train_utils import build_dataset_bundle, build_seq2seq_collator, build_sft_datasets
from thesis_rrg.utils.io import write_json



def run_hf_trainer(cfg, output_dir: Path, logger):
    bundle = build_dataset_bundle(cfg)
    print(bundle.meta)
    print('image_aug_mode=', bundle.image_aug_mode)
    print('textaug_apply_p=', bundle.textaug_apply_p, 'textaug_map_size=', len(bundle.textaug_map))
    model_bundle = build_model_bundle(cfg, for_train=True)

    train_ds, valid_ds, _ = build_sft_datasets(cfg, bundle, model_bundle.processor)
    collator = build_seq2seq_collator(model_bundle.processor)
    report_to = resolve_report_to(cfg)
    ensure_reporting_dependencies(report_to)

    logging_dir = None
    if report_to_includes_tensorboard(report_to):
        logging_dir = str(output_dir / "tensorboard")

    args_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(cfg.training.num_train_epochs),
        "max_steps": int(cfg.training.max_steps),
        "per_device_train_batch_size": int(cfg.training.per_device_train_batch_size),
        "per_device_eval_batch_size": int(cfg.training.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(cfg.training.gradient_accumulation_steps),
        "learning_rate": float(cfg.training.learning_rate),
        "weight_decay": float(cfg.training.weight_decay),
        "warmup_ratio": float(cfg.training.warmup_ratio),
        "lr_scheduler_type": str(cfg.training.lr_scheduler_type),
        "logging_steps": int(cfg.training.logging_steps),
        "eval_steps": int(cfg.training.eval_steps),
        "save_steps": int(cfg.training.save_steps),
        "eval_strategy": str(cfg.training.eval_strategy),
        "save_strategy": str(cfg.training.save_strategy),
        "bf16": bool(cfg.training.bf16),
        "fp16": bool(cfg.training.fp16),
        "gradient_checkpointing": bool(cfg.training.gradient_checkpointing),
        "remove_unused_columns": False,
        "report_to": report_to,
        "logging_dir": logging_dir,
        "dataloader_num_workers": int(cfg.data.num_workers),
        "save_total_limit": int(cfg.training.save_total_limit),
        "metric_for_best_model": str(cfg.training.metric_for_best_model),
    }

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    add_optional_training_arguments(args_kwargs, cfg.training, set(ta_params))

    args = TrainingArguments(**args_kwargs)

    trainer = Trainer(
        model=model_bundle.model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        callbacks=[TrainingProgressCallback(), PeftCheckpointCallback(model_bundle.processor)],
    )

    logger.info("Starting HF Trainer training")
    train_output = trainer.train()

    last_dir, _ = prepare_checkpoint_dirs(output_dir)
    save_trainer_model_artifact(trainer, model_bundle.processor, last_dir)
    write_best_checkpoint_pointer(output_dir, trainer.state.best_model_checkpoint)

    train_metrics = dict(train_output.metrics)
    write_json(output_dir / "metrics" / "train_metrics.json", train_metrics)

    logger.info("HF Trainer training finished")
    return {
        "backend": "hf_trainer",
        "train_metrics": train_metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
    }
