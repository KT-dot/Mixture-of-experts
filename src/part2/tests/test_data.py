from __future__ import annotations

import torch
from datasets import Dataset

from src.part2.data import (
    TokenBlockDataset,
    choose_text_and_id_fields,
    deterministic_group_split,
    inspect_dataset_schema,
)


def test_schema_is_inspected_before_fields_are_selected() -> None:
    dataset = Dataset.from_dict(
        {
            "doc_id": [f"doc_{index // 2}@variant_{index % 2}" for index in range(40)],
            "text": [f"example {index}" for index in range(40)],
        }
    )
    schema = inspect_dataset_schema(dataset)
    assert schema["columns"] == ["doc_id", "text"]
    assert choose_text_and_id_fields(dataset, schema, {}) == ("text", "doc_id")
    splits = deterministic_group_split(
        dataset,
        "doc_id",
        {"ratios": {"train": 0.8, "validation": 0.1, "test": 0.1}},
        seed=42,
    )
    families = {
        name: {identifier.split("@", 1)[0] for identifier in split["doc_id"]}
        for name, split in splits.items()
    }
    assert families["train"].isdisjoint(families["validation"])
    assert families["train"].isdisjoint(families["test"])
    assert families["validation"].isdisjoint(families["test"])


def test_causal_labels_are_shifted_one_token() -> None:
    dataset = TokenBlockDataset(torch.tensor([[4, 5, 6, 7]], dtype=torch.long))
    item = dataset[0]
    torch.testing.assert_close(item["input_ids"], torch.tensor([4, 5, 6]))
    torch.testing.assert_close(item["labels"], torch.tensor([5, 6, 7]))
    assert dataset.training_tokens == 3
