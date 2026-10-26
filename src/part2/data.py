"""Deterministic real and synthetic causal-language-model data pipelines."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch.utils.data import Dataset

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


class DecodingTokenizer(Protocol):
    def get_vocab_size(self) -> int: ...

    def encode(self, text: str): ...

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str: ...


class TokenBlockDataset(Dataset):
    """Rows of length sequence_length+1, shifted into inputs and next-token labels."""

    def __init__(self, blocks: torch.Tensor) -> None:
        if blocks.ndim != 2 or blocks.shape[1] < 3:
            raise ValueError("blocks must have shape [examples, sequence_length+1]")
        self.blocks = blocks.to(dtype=torch.long, copy=False).contiguous()

    @property
    def sequence_length(self) -> int:
        return self.blocks.shape[1] - 1

    @property
    def training_tokens(self) -> int:
        return len(self) * self.sequence_length

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.blocks[index]
        return {"input_ids": row[:-1], "labels": row[1:]}


@dataclass
class DataBundle:
    train: TokenBlockDataset
    validation: TokenBlockDataset
    test: TokenBlockDataset
    tokenizer: DecodingTokenizer
    metadata: dict


class SyntheticTokenizer:
    """Minimal deterministic decoder used only by local synthetic tests."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def encode(self, text: str):
        raise NotImplementedError("SyntheticTokenizer supports decoding only")

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        special = set(range(len(SPECIAL_TOKENS))) if skip_special_tokens else set()
        return " ".join(f"token_{value}" for value in ids if value not in special)


def _synthetic_blocks(
    count: int, sequence_length: int, vocab_size: int, offset: int
) -> torch.Tensor:
    # A repeated, learnable modular pattern makes smoke tests stable and inexpensive.
    rows = []
    usable = vocab_size - len(SPECIAL_TOKENS)
    for row_index in range(count):
        start = (row_index * 3 + offset) % usable
        row = [
            len(SPECIAL_TOKENS) + (start + position) % usable
            for position in range(sequence_length + 1)
        ]
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


def make_synthetic_bundle(
    *,
    vocab_size: int = 32,
    sequence_length: int = 12,
    train_blocks: int = 16,
    validation_blocks: int = 4,
    test_blocks: int = 4,
    seed: int = 42,
) -> DataBundle:
    if vocab_size <= len(SPECIAL_TOKENS) + 1:
        raise ValueError("synthetic vocabulary is too small")
    return DataBundle(
        train=TokenBlockDataset(_synthetic_blocks(train_blocks, sequence_length, vocab_size, seed)),
        validation=TokenBlockDataset(
            _synthetic_blocks(validation_blocks, sequence_length, vocab_size, seed + 1)
        ),
        test=TokenBlockDataset(
            _synthetic_blocks(test_blocks, sequence_length, vocab_size, seed + 2)
        ),
        tokenizer=SyntheticTokenizer(vocab_size),
        metadata={
            "source": "synthetic",
            "seed": seed,
            "vocab_size": vocab_size,
            "sequence_length": sequence_length,
        },
    )


def inspect_dataset_schema(dataset) -> dict:
    """Inspect columns and feature types before any text field is selected."""
    columns = list(dataset.column_names)
    feature_types = {name: str(dataset.features[name]) for name in columns}
    sample = dataset[0] if len(dataset) else {}
    sample_python_types = {name: type(sample.get(name)).__name__ for name in columns}
    return {
        "row_count": len(dataset),
        "columns": columns,
        "features": feature_types,
        "sample_python_types": sample_python_types,
    }


def choose_text_and_id_fields(dataset, schema: dict, configured: dict) -> tuple[str, str]:
    """Choose fields only after schema inspection; fail rather than guess ambiguously."""
    text_field = configured.get("text_field")
    id_field = configured.get("id_field")
    columns = schema["columns"]
    if text_field is None:
        string_fields = [
            name
            for name in columns
            if "string" in schema["features"][name].lower()
            and name not in {id_field, "doc_id", "id"}
        ]
        if "text" in string_fields:
            text_field = "text"
        elif len(string_fields) == 1:
            text_field = string_fields[0]
        else:
            raise ValueError(f"cannot unambiguously choose a text field from {string_fields}")
    if id_field is None:
        for candidate in ("doc_id", "id"):
            if candidate in columns:
                id_field = candidate
                break
    if text_field not in columns or id_field not in columns:
        raise ValueError(
            f"selected fields text={text_field!r}, id={id_field!r} are not in {columns}"
        )
    if not all(isinstance(value, str) for value in dataset[: min(32, len(dataset))][text_field]):
        raise TypeError(f"selected text field {text_field!r} is not consistently string-valued")
    return text_field, id_field


def _document_family(identifier: str) -> str:
    """Keep human/AI parallel versions of one source document in one split."""
    return identifier.split("@", maxsplit=1)[0]


def deterministic_group_split(dataset, id_field: str, split_config: dict, seed: int):
    from datasets import DatasetDict

    ratios = split_config["ratios"]
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("dataset split ratios must sum to 1")
    family_rows: dict[str, list[int]] = {}
    for index, identifier in enumerate(dataset[id_field]):
        family_rows.setdefault(_document_family(str(identifier)), []).append(index)
    families = sorted(family_rows)
    random.Random(seed).shuffle(families)
    n_families = len(families)
    n_train = int(n_families * ratios["train"])
    n_validation = int(n_families * ratios["validation"])
    if min(n_train, n_validation, n_families - n_train - n_validation) < 1:
        raise ValueError("not enough document families for non-empty train/validation/test")
    family_splits = {
        "train": families[:n_train],
        "validation": families[n_train : n_train + n_validation],
        "test": families[n_train + n_validation :],
    }
    return DatasetDict(
        {
            name: dataset.select(
                sorted(index for family in selected for index in family_rows[family])
            )
            for name, selected in family_splits.items()
        }
    )


def train_or_load_tokenizer(
    train_texts, tokenizer_config: dict, path: Path, expected_manifest: dict
):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    if tuple(tokenizer_config["special_tokens"]) != SPECIAL_TOKENS:
        raise ValueError(f"special_tokens must be exactly {SPECIAL_TOKENS}")
    manifest_path = path.with_name("tokenizer_manifest.json")
    if path.exists():
        if not manifest_path.exists():
            raise ValueError("cached tokenizer has no reproducibility manifest")
        observed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if observed_manifest != expected_manifest:
            raise ValueError(
                "cached tokenizer was built from different data or settings; "
                "use a clean output directory"
            )
        tokenizer = Tokenizer.from_file(str(path))
        if tokenizer.get_vocab_size() != tokenizer_config["vocab_size"]:
            raise ValueError("cached tokenizer vocabulary does not match configuration")
        return tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    # Seed the vocabulary with all 256 byte symbols so no input byte can become <unk>;
    # without this, bytes unseen in the training split (e.g. rare accents, CJK) are lost.
    trainer = trainers.BpeTrainer(
        vocab_size=tokenizer_config["vocab_size"],
        min_frequency=tokenizer_config["min_frequency"],
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(train_texts, trainer=trainer, length=len(train_texts))
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    manifest_path.write_text(
        json.dumps(expected_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return tokenizer


def tokenizer_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def texts_to_blocks(
    texts: list[str], tokenizer: DecodingTokenizer, sequence_length: int
) -> TokenBlockDataset:
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    if bos is None or eos is None:
        raise ValueError("tokenizer is missing BOS/EOS special tokens")
    stream: list[int] = []
    for start in range(0, len(texts), 256):
        encodings = tokenizer.encode_batch(texts[start : start + 256])
        for encoding in encodings:
            stream.append(bos)
            stream.extend(encoding.ids)
            stream.append(eos)
    row_length = sequence_length + 1
    row_count = (len(stream) - 1) // sequence_length
    if row_count < 1:
        raise ValueError("tokenized split is too small for one causal block")
    usable = row_count * sequence_length + 1
    values = torch.tensor(stream[:usable], dtype=torch.long)
    blocks = values.unfold(0, row_length, sequence_length).contiguous()
    return TokenBlockDataset(blocks)


def prepare_real_data(config: dict, output_directory: Path) -> DataBundle:
    """Download, inspect, split, tokenize, and block the configured HF dataset.

    This function is reached only for a requested real run. Synthetic verification
    does not call Hugging Face or download any dataset.
    """
    from datasets import DatasetDict, load_dataset
    from huggingface_hub import HfApi

    dataset_config = config["dataset"]
    cache_directory = Path(dataset_config["cache_directory"]).expanduser()
    # Resolve a moving branch/tag to an immutable Hub commit before downloading.
    resolved_revision = (
        HfApi()
        .dataset_info(repo_id=dataset_config["name"], revision=dataset_config.get("revision"))
        .sha
    )
    loaded = load_dataset(
        dataset_config["name"],
        revision=resolved_revision,
        cache_dir=str(cache_directory),
    )
    split_schemas = {name: inspect_dataset_schema(split) for name, split in loaded.items()}
    reference_split = loaded["train"] if "train" in loaded else next(iter(loaded.values()))
    text_field, id_field = choose_text_and_id_fields(
        reference_split,
        split_schemas[next(iter(loaded.keys())) if "train" not in loaded else "train"],
        dataset_config,
    )
    official = all(name in loaded for name in ("train", "validation", "test"))
    if official:
        split_dataset = DatasetDict(
            {name: loaded[name] for name in ("train", "validation", "test")}
        )
        split_method = "official"
    else:
        if "train" not in loaded or len(loaded) != 1:
            raise ValueError(
                "partial official splits are unsupported; expected one or three splits"
            )
        split_dataset = deterministic_group_split(
            loaded["train"], id_field, dataset_config["split"], config["experiment"]["seed"]
        )
        split_method = "seeded_document_family"
    tokenizer_path = output_directory / "tokenizer.json"
    tokenizer_manifest = {
        "dataset": dataset_config["name"],
        "resolved_revision": resolved_revision,
        "training_split_fingerprint": split_dataset["train"]._fingerprint,
        "selected_text_field": text_field,
        "configuration": config["tokenizer"],
        "initial_alphabet": "byte_level_256",
    }
    tokenizer = train_or_load_tokenizer(
        split_dataset["train"][text_field],
        config["tokenizer"],
        tokenizer_path,
        tokenizer_manifest,
    )
    sequence_length = config["training"]["sequence_length"]
    blocks = {
        name: texts_to_blocks(split_dataset[name][text_field], tokenizer, sequence_length)
        for name in ("train", "validation", "test")
    }
    metadata = {
        "source": dataset_config["name"],
        "configured_revision": dataset_config.get("revision"),
        "resolved_revision": resolved_revision,
        "dataset_fingerprints": {name: split_dataset[name]._fingerprint for name in split_dataset},
        "inspected_schema": split_schemas,
        "selected_text_field": text_field,
        "selected_id_field": id_field,
        "split_method": split_method,
        "split_rows": {name: len(split_dataset[name]) for name in split_dataset},
        "block_counts": {name: len(blocks[name]) for name in blocks},
        "prediction_tokens": {name: blocks[name].training_tokens for name in blocks},
        "tokenizer_sha256": tokenizer_sha256(tokenizer_path),
        "unknown_token_counts": {
            name: int(blocks[name].blocks.eq(tokenizer.token_to_id("<unk>")).sum())
            for name in blocks
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "dataset_schema.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return DataBundle(
        train=blocks["train"],
        validation=blocks["validation"],
        test=blocks["test"],
        tokenizer=tokenizer,
        metadata=metadata,
    )


def fixed_training_order(example_count: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(example_count, generator=generator).tolist()
