from __future__ import annotations

from dataclasses import dataclass

from thesis_rrg.models.loading import load_base_model
from thesis_rrg.models.lora import attach_lora, load_adapter
from thesis_rrg.models.processor_factory import build_processor


@dataclass
class ModelBundle:
    model: object
    processor: object



def build_model_bundle(cfg, for_train: bool) -> ModelBundle:
    model = load_base_model(cfg.model, cfg.training)
    processor = build_processor(str(cfg.model.model_id), use_fast=bool(cfg.model.get("processor_use_fast", False)))

    adapter_path = str(cfg.model.get("adapter_path", "")).strip()

    if for_train:
        if adapter_path:
            model = load_adapter(model, adapter_path, is_trainable=True)
        else:
            model = attach_lora(model, cfg.lora)
    elif adapter_path:
        model = load_adapter(model, adapter_path)

    return ModelBundle(model=model, processor=processor)
