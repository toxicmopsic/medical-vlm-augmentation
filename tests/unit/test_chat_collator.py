import torch

from thesis_rrg.data.collators.vlm_chat_collator import _tokenized_prefix_len


def test_tokenized_prefix_len_for_flat_list():
    assert _tokenized_prefix_len([1, 2, 3, 4]) == 4


def test_tokenized_prefix_len_for_nested_list():
    assert _tokenized_prefix_len([[1, 2, 3], [4, 5]]) == 3


def test_tokenized_prefix_len_for_tensor():
    assert _tokenized_prefix_len(torch.tensor([1, 2, 3])) == 3
    assert _tokenized_prefix_len(torch.tensor([[1, 2, 3]])) == 3


def test_tokenized_prefix_len_for_mapping_like_output():
    assert _tokenized_prefix_len({"input_ids": [1, 2, 3]}) == 3
    assert _tokenized_prefix_len({"input_ids": [[1, 2, 3, 4]]}) == 4
