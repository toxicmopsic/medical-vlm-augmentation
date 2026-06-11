from __future__ import annotations

from typing import Any, Dict, List

import torch


def _tokenized_prefix_len(tokens: Any) -> int:
    """
    Robustly infer token length from apply_chat_template(..., tokenize=True)
    outputs across tokenizer/processor implementations.
    """
    if tokens is None:
        return 0

    if torch.is_tensor(tokens):
        if tokens.ndim == 0:
            return 0
        if tokens.ndim == 1:
            return int(tokens.shape[0])
        return int(tokens.shape[-1])

    if isinstance(tokens, dict):
        return _tokenized_prefix_len(tokens.get("input_ids"))

    if isinstance(tokens, (list, tuple)):
        if len(tokens) == 0:
            return 0
        first = tokens[0]
        if isinstance(first, (list, tuple)):
            return len(first)
        return len(tokens)

    return 0



def chats_to_tensors(chats: list, processor, add_generation_prompt: bool):
    texts = [
        processor.apply_chat_template(
            dlg,
            add_generation_prompt=add_generation_prompt,  # pyright: ignore[reportCallIssue]
            tokenize=False,  # pyright: ignore[reportCallIssue]
            add_special_tokens=True,  # Be careful: duplicate BOS and other stuff can appear  # pyright: ignore[reportCallIssue]
        )
        for dlg in chats
    ]

    images = []
    for dlg in chats:
        dlg_images = []
        for msg in dlg:
            dlg_images += [
                entry['image']  # pyright: ignore[reportArgumentType]
                for entry in msg['content']
                if entry['type'] == 'image'  # pyright: ignore[reportArgumentType]
            ]
        images.append(dlg_images)

    batch = processor(
        text=texts,
        images=images,
        return_tensors='pt',   # pyright: ignore[reportCallIssue]
        padding=True,  # pyright: ignore[reportCallIssue]
        add_special_tokens=False,  #  Otherwise sometimes it will add a second BOS token  # pyright: ignore[reportCallIssue]
    )
    return batch



def chat_collate_fn(examples: List[Dict[str, Any]], processor) -> Dict[str, torch.Tensor]:
    chats = [x["messages"] for x in examples]
    batch = chats_to_tensors(chats, processor=processor, add_generation_prompt=False)

    labels = batch["input_ids"].clone()

    token_names_for_masking = [
        "bos_token_id",
        "pad_token_id",
        "boi_token_id",
        "eoi_token_id",
        "image_token_id",
    ]

    token_ids_for_masking = []
    for token_name in token_names_for_masking:
        token_id = getattr(processor.tokenizer, token_name, None)
        if token_id is not None:
            token_ids_for_masking.append(token_id)

    for token_id in token_ids_for_masking:
        labels[labels == token_id] = -100

    tokens_without_response = [
        processor.apply_chat_template(
            dlg[:-1],
            add_generation_prompt=True,
            tokenize=True,
            add_special_tokens=True,
        )
        for dlg in chats
    ]
    lengths = [_tokenized_prefix_len(x) for x in tokens_without_response]
    for i, length in enumerate(lengths):
        labels[i, :length] = -100

    batch["labels"] = labels
    return batch
