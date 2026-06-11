from __future__ import annotations

import inspect
from pathlib import Path

from trl import SFTConfig, SFTTrainer

from thesis_rrg.models.factory import build_model_bundle
from thesis_rrg.models.model_utils import save_trainer_model_artifact
from thesis_rrg.training.args_utils import add_optional_training_arguments
from thesis_rrg.training.callbacks import PeftCheckpointCallback
from thesis_rrg.training.reporting import ensure_reporting_dependencies, report_to_includes_tensorboard, resolve_report_to
from thesis_rrg.training.train_utils import build_chat_collator, build_chat_datasets, build_dataset_bundle
from thesis_rrg.utils.io import write_json
from thesis_rrg.utils.config_utils import dataset_bundle_summary



def run_trl_sft(cfg, output_dir: Path, logger):
    bundle = build_dataset_bundle(cfg)
    bundle_summary = dataset_bundle_summary(bundle)
    write_json(output_dir / "metrics" / "dataset_bundle.json", bundle_summary)
    logger.info("Dataset bundle: %s", bundle_summary)

    model_bundle = build_model_bundle(cfg, for_train=True)

    train_ds, valid_ds = build_chat_datasets(cfg, bundle, model_bundle.processor)
    collator = build_chat_collator(model_bundle.processor)
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
        "optim": str(cfg.training.get("optim", "adamw_torch_fused")),
        "logging_steps": int(cfg.training.logging_steps),
        "eval_steps": int(cfg.training.eval_steps),
        "eval_strategy": str(cfg.training.eval_strategy),
        "save_strategy": str(cfg.training.save_strategy),
        "save_steps": int(cfg.training.save_steps),
        "save_total_limit": int(cfg.training.save_total_limit),
        "max_grad_norm": float(cfg.training.get("max_grad_norm", 1.0)),
        "bf16": bool(cfg.training.bf16),
        "fp16": bool(cfg.training.fp16),
        "gradient_checkpointing": bool(cfg.training.gradient_checkpointing),
        "gradient_checkpointing_kwargs": {
            "use_reentrant": bool(cfg.training.get("gradient_checkpointing_use_reentrant", False))
        },
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "remove_unused_columns": False,
        "label_names": list(cfg.training.get("label_names", ["labels"])),
        "report_to": report_to,
        "logging_dir": logging_dir,
        "metric_for_best_model": str(cfg.training.metric_for_best_model),
    }
    sft_params = inspect.signature(SFTConfig.__init__).parameters
    add_optional_training_arguments(args_kwargs, cfg.training, set(sft_params))
    args = SFTConfig(**args_kwargs)

    trainer = SFTTrainer(
        model=model_bundle.model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=model_bundle.processor,
        data_collator=collator,
        callbacks=[PeftCheckpointCallback(model_bundle.processor)],
    )

    logger.info("Starting TRL SFT training")
    train_output = trainer.train()

    save_trainer_model_artifact(trainer, model_bundle.processor, output_dir / "checkpoints" / "last")

    train_metrics = dict(train_output.metrics)
    write_json(output_dir / "metrics" / "train_metrics.json", train_metrics)

    logger.info("TRL SFT training finished")
    return {
        "backend": "trl_sft",
        "train_metrics": train_metrics,
        "dataset_bundle": bundle_summary,
    }
