from __future__ import annotations

import torch
from tqdm.auto import tqdm

from thesis_rrg.data.preprocess.report_cleaning import parse_pred_findings_impression
from thesis_rrg.training.train_utils import build_eval_loader

def _prepare_processor_for_generation(processor) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return
    # Decoder-only models generate from the last non-padding token. With
    # right-padding, batched generation can condition on pad tokens for shorter
    # prompts. Keep training padding untouched; force left-padding only here.
    tokenizer.padding_side = "left"

    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

def _build_generation_cfg(cfg, processor) -> dict:
    generation_node = cfg.model.generation if "generation" in cfg.model else cfg.generation
    generation_cfg = {
        "do_sample": bool(generation_node.do_sample),
        "num_beams": int(generation_node.num_beams),
        "max_new_tokens": int(generation_node.max_new_tokens),
        "repetition_penalty": float(generation_node.repetition_penalty),
    }

    if "min_new_tokens" in generation_node:
        generation_cfg["min_new_tokens"] = int(generation_node.min_new_tokens)

    no_repeat = int(generation_node.get("no_repeat_ngram_size", 0))
    if no_repeat > 0:
        generation_cfg["no_repeat_ngram_size"] = no_repeat

    if bool(generation_node.do_sample):
        generation_cfg["temperature"] = float(generation_node.temperature)
        generation_cfg["top_p"] = float(generation_node.top_p)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if eos_token_id is not None:
            generation_cfg["eos_token_id"] = eos_token_id
        if pad_token_id is not None:
            generation_cfg["pad_token_id"] = pad_token_id

    return generation_cfg


def run_inference(cfg, model, processor, df_eval):
    _prepare_processor_for_generation(processor)
    loader = build_eval_loader(cfg, df_eval=df_eval, processor=processor)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    generation_cfg = _build_generation_cfg(cfg, processor)

    rows = []
    progress = tqdm(loader, total=len(loader), desc="Eval inference", dynamic_ncols=True)
    for batch in progress:
        enc = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "pixel_values": batch["pixel_values"].to(device),
        }
        with torch.inference_mode():
            gen_ids = model.generate(**enc, **generation_cfg)
        input_len = enc["input_ids"].shape[1]
        gen_texts = processor.batch_decode(gen_ids[:, input_len:], skip_special_tokens=True)

        for sid, pred, rf, ri in zip(
            batch["study_id"],
            gen_texts,
            batch["ref_findings"],
            batch["ref_impression"],
        ):
            pred = str(pred).strip()
            pf, pi = parse_pred_findings_impression(pred)
            rows.append(
                {
                    "study_id": sid,
                    "raw_pred": pred,
                    "pred_findings": pf,
                    "pred_impression": pi,
                    "pred_full": pred,
                    "ref_findings": rf,
                    "ref_impression": ri,
                    "ref_full": f"{rf} {ri}".strip(),
                }
            )

    return rows
