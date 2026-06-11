import torch

from thesis_rrg.data.collators.seq2seq_collator import Seq2SeqVisionLanguageCollator


class DummyTokenizer:
    padding_side = "right"

    def pad(self, payload, padding, return_tensors):
        input_ids = payload["input_ids"]
        attention_mask = payload["attention_mask"]
        max_len = max(x.shape[0] for x in input_ids)

        ids = torch.zeros((len(input_ids), max_len), dtype=torch.long)
        mask = torch.zeros((len(input_ids), max_len), dtype=torch.long)

        for i, (row_ids, row_mask) in enumerate(zip(input_ids, attention_mask)):
            if self.padding_side == "left":
                ids[i, max_len - row_ids.shape[0] :] = row_ids
                mask[i, max_len - row_mask.shape[0] :] = row_mask
            else:
                ids[i, : row_ids.shape[0]] = row_ids
                mask[i, : row_mask.shape[0]] = row_mask

        return {"input_ids": ids, "attention_mask": mask}


class DummyProcessor:
    tokenizer = DummyTokenizer()



def test_collator_shapes():
    collator = Seq2SeqVisionLanguageCollator(DummyProcessor())

    b1 = {
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
        "pixel_values": torch.zeros((3, 4, 4)),
        "labels": torch.tensor([1, 2, -100], dtype=torch.long),
    }
    b2 = {
        "input_ids": torch.tensor([1, 2], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1], dtype=torch.long),
        "pixel_values": torch.zeros((3, 4, 4)),
        "labels": torch.tensor([1, -100], dtype=torch.long),
    }

    out = collator([b1, b2])

    assert out["input_ids"].shape == (2, 3)
    assert out["labels"].shape == (2, 3)
    assert out["pixel_values"].shape == (2, 3, 4, 4)


def test_collator_accepts_singleton_image_axis():
    collator = Seq2SeqVisionLanguageCollator(DummyProcessor())

    b1 = {
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 4, 4)),
        "labels": torch.tensor([1, 2, -100], dtype=torch.long),
    }
    b2 = {
        "input_ids": torch.tensor([1, 2], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1], dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 4, 4)),
        "labels": torch.tensor([1, -100], dtype=torch.long),
    }

    out = collator([b1, b2])
    assert out["pixel_values"].shape == (2, 3, 4, 4)


def test_collator_mixed_pixel_shapes_promotes_to_5d():
    collator = Seq2SeqVisionLanguageCollator(DummyProcessor())

    b1 = {
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
        "pixel_values": torch.zeros((3, 4, 4)),
        "labels": torch.tensor([1, 2, -100], dtype=torch.long),
    }
    b2 = {
        "input_ids": torch.tensor([1, 2], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1], dtype=torch.long),
        "pixel_values": torch.zeros((1, 3, 4, 4)),
        "labels": torch.tensor([1, -100], dtype=torch.long),
    }

    out = collator([b1, b2])
    assert out["pixel_values"].shape == (2, 1, 3, 4, 4)


def test_collator_aligns_labels_with_left_padding():
    class LeftPadProcessor:
        tokenizer = DummyTokenizer()
        tokenizer.padding_side = "left"

    collator = Seq2SeqVisionLanguageCollator(LeftPadProcessor())

    b1 = {
        "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
        "pixel_values": torch.zeros((3, 4, 4)),
        "labels": torch.tensor([10, 20, 30], dtype=torch.long),
    }
    b2 = {
        "input_ids": torch.tensor([4, 5], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1], dtype=torch.long),
        "pixel_values": torch.zeros((3, 4, 4)),
        "labels": torch.tensor([40, 50], dtype=torch.long),
    }

    out = collator([b1, b2])

    assert out["input_ids"].tolist() == [[1, 2, 3], [0, 4, 5]]
    assert out["labels"].tolist() == [[10, 20, 30], [-100, 40, 50]]
