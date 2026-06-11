from __future__ import annotations

from functools import partial
from pathlib import Path

from torch.utils.data import DataLoader

from thesis_rrg.data.collators.seq2seq_collator import Seq2SeqVisionLanguageCollator
from thesis_rrg.data.collators.vlm_chat_collator import chat_collate_fn, chats_to_tensors
from thesis_rrg.data.datasets.chat_dataset import MimicReportChatDataset, build_user_chat
from thesis_rrg.data.datasets.mimic_report_dataset import MimicEvalDataset, MimicReportSFTDataset
from thesis_rrg.data.preprocess.image_pipeline import safe_open_mimic_dicom
from thesis_rrg.data.preprocess.prompt_targets import build_mimic_prompt
from thesis_rrg.data.registry import DATA_BUILDER_REGISTRY
from thesis_rrg.data.transforms.image_augment import build_cxr_augment



def build_dataset_bundle(cfg):
    family = str(cfg.experiment.family)
    builder = DATA_BUILDER_REGISTRY.get(family)
    return builder(cfg)



def build_sft_datasets(cfg, bundle, processor):
    image_aug = build_cxr_augment(mode=bundle.image_aug_mode, seed=int(cfg.seed))
    target_format = str(cfg.prompts.get("target_format", "findings + impression"))

    train_ds = MimicReportSFTDataset(
        df_=bundle.train_df,
        processor=processor,
        is_train=True,
        max_text_len=int(cfg.data.max_text_len),
        out_size=int(cfg.data.get("image_size", 896)),
        image_augment=image_aug,
        textaug_map=bundle.textaug_map,
        textaug_apply_p=float(bundle.textaug_apply_p),
        seed=int(cfg.seed),
        prompt_template=str(cfg.prompts.prompt_template),
        target_format=target_format,
        debug_first_n=int(cfg.data.debug_first_n),
    )

    valid_ds = MimicReportSFTDataset(
        df_=bundle.valid_df,
        processor=processor,
        is_train=False,
        max_text_len=int(cfg.data.max_text_len),
        out_size=int(cfg.data.get("image_size", 896)),
        image_augment=None,
        textaug_map={},
        textaug_apply_p=0.0,
        seed=int(cfg.seed),
        prompt_template=str(cfg.prompts.prompt_template),
        target_format=target_format,
        debug_first_n=0,
    )

    test_ds = MimicReportSFTDataset(
        df_=bundle.test_df,
        processor=processor,
        is_train=False,
        max_text_len=int(cfg.data.max_text_len),
        out_size=int(cfg.data.get("image_size", 896)),
        image_augment=None,
        textaug_map={},
        textaug_apply_p=0.0,
        seed=int(cfg.seed),
        prompt_template=str(cfg.prompts.prompt_template),
        target_format=target_format,
        debug_first_n=0,
    )

    return train_ds, valid_ds, test_ds



def build_eval_loader(cfg, df_eval, processor):
    ds = MimicEvalDataset(df_eval)
    use_chat_template = bool(cfg.prompts.get("use_chat_template", False))

    def _collate(batch):
        images = [
            safe_open_mimic_dicom(Path(x["dcm_path"]), out_size=int(cfg.data.get("image_size", 896)), augment=None)
            for x in batch
        ]
        prompts = [build_mimic_prompt(x["indication"], template=str(cfg.prompts.prompt_template)) for x in batch]

        if use_chat_template:
            chats = [build_user_chat(prompt, image) for prompt, image in zip(prompts, images)]
            enc = chats_to_tensors(chats, processor=processor, add_generation_prompt=True)
            return {
                **enc,
                "study_id": [x["study_id"] for x in batch],
                "ref_findings": [x["ref_findings"] for x in batch],
                "ref_impression": [x["ref_impression"] for x in batch],
            }

        # enc = processor(text=prompts, images=images, return_tensors="pt", padding=True, do_resize=False)
        # Gemma-style multimodal processors expect one image-list per text sample.
        # For 1 image per prompt this must be shaped as [[img1], [img2], ...].
        image_batches = [[img] for img in images]

        enc = processor(text=prompts, images=image_batches, return_tensors="pt", padding=True, do_resize=False)
     
        return {
            **enc,
            "study_id": [x["study_id"] for x in batch],
            "ref_findings": [x["ref_findings"] for x in batch],
            "ref_impression": [x["ref_impression"] for x in batch],
        }

    return DataLoader(
        ds,
        batch_size=int(cfg.data.eval_batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=True,
        collate_fn=_collate,
    )



def build_seq2seq_collator(processor):
    return Seq2SeqVisionLanguageCollator(processor)



def build_chat_datasets(cfg, bundle, processor):
    prompt_template = str(cfg.prompts.prompt_template)
    target_format = str(cfg.prompts.get("target_format", "findings + impression"))
    tokenizer = processor.tokenizer

    train_ds = MimicReportChatDataset(
        bundle.train_df,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        target_format=target_format,
        out_size=int(cfg.data.get("image_size", 896)),
        is_train=True,
        textaug_map=bundle.textaug_map,
        textaug_apply_p=float(bundle.textaug_apply_p),
        seed=int(cfg.seed),
    )
    valid_ds = MimicReportChatDataset(
        bundle.valid_df,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        target_format=target_format,
        out_size=int(cfg.data.get("image_size", 896)),
        is_train=False,
        textaug_map={},
        textaug_apply_p=0.0,
        seed=int(cfg.seed),
    )

    return train_ds, valid_ds


def build_chat_collator(processor):
    return partial(chat_collate_fn, processor=processor)
