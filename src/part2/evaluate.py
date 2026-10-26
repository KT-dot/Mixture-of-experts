"""Validation loss and controlled continuation BLEU evaluation."""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator

import sacrebleu
import torch

from .data import TokenBlockDataset


def precision_context(device: torch.device, precision: str):
    if precision == "float32":
        return contextlib.nullcontext()
    if precision == "bfloat16" and device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    if device.type != "cuda":
        raise ValueError(f"{precision} evaluation requires CUDA")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[precision]
    return torch.autocast(device_type="cuda", dtype=dtype)


def _batches(
    dataset: TokenBlockDataset, batch_size: int, max_examples: int | None = None
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    limit = len(dataset) if max_examples is None else min(len(dataset), max_examples)
    for start in range(0, limit, batch_size):
        rows = dataset.blocks[start : min(start + batch_size, limit)]
        yield rows[:, :-1], rows[:, 1:]


@torch.inference_mode()
def evaluate_loss(
    model,
    dataset: TokenBlockDataset,
    *,
    batch_size: int,
    device: torch.device,
    precision: str,
    max_examples: int | None = None,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for input_ids, labels in _batches(dataset, batch_size, max_examples):
        input_ids, labels = input_ids.to(device), labels.to(device)
        with precision_context(device, precision):
            output = model(input_ids, labels)
        tokens = int(labels.ne(-100).sum().item())
        total_loss += float(output["loss"].float().item()) * tokens
        total_tokens += tokens
    if was_training:
        model.train()
    if total_tokens == 0:
        raise ValueError("evaluation dataset contains no supervised tokens")
    mean_loss = total_loss / total_tokens
    return mean_loss, math.exp(min(mean_loss, 80.0))


@torch.inference_mode()
def evaluate_continuation_bleu(
    model,
    dataset: TokenBlockDataset,
    tokenizer,
    *,
    optimizer_name: str,
    training_tokens_seen: int,
    checkpoint_fraction: float,
    device: torch.device,
    prefix_fraction: float,
    generation_tokens: int,
    max_examples: int | None,
    batch_size: int,
) -> tuple[float, list[dict]]:
    """Greedily complete fixed prefixes and compute one corpus SacreBLEU score.

    BLEU is retained because the assignment requests it. These short open-ended
    continuations can have many valid wordings, so the metric is a controlled
    optimizer comparison and is not treated as a complete LM-quality measure.
    ``max_examples=None`` scores every block of the test split.
    """
    if not 0 < prefix_fraction < 1:
        raise ValueError("prefix_fraction must lie in (0, 1)")
    was_training = model.training
    model.eval()
    hypotheses: list[str] = []
    references: list[str] = []
    records: list[dict] = []
    count = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    prefix_length = max(1, int(dataset.sequence_length * prefix_fraction))
    # The prompt plus continuation must fit the model context (sequence_length), which
    # also guarantees the reference slice lies inside the block.
    continuation_length = min(generation_tokens, dataset.sequence_length - prefix_length)
    if continuation_length < 1:
        raise ValueError("prefix leaves no room for a continuation in the context window")
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch = dataset.blocks[start:stop]
        prompts = batch[:, :prefix_length].to(device)
        generated_batch = model.greedy_generate(prompts, continuation_length).cpu()
        for local_index, index in enumerate(range(start, stop)):
            prompt_ids = batch[local_index, :prefix_length].tolist()
            reference_ids = batch[
                local_index, prefix_length : prefix_length + continuation_length
            ].tolist()
            generated_ids = generated_batch[local_index, -continuation_length:].tolist()
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            reference_text = tokenizer.decode(reference_ids, skip_special_tokens=True)
            generation_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            references.append(reference_text)
            hypotheses.append(generation_text)
            records.append(
                {
                    "optimizer": optimizer_name,
                    "checkpoint_fraction": checkpoint_fraction,
                    "training_tokens_seen": training_tokens_seen,
                    "example_index": index,
                    "prefix_tokens": prefix_length,
                    "continuation_tokens": continuation_length,
                    "prompt": prompt_text,
                    "reference": reference_text,
                    "generation": generation_text,
                }
            )
    if was_training:
        model.train()
    if not hypotheses:
        raise ValueError("BLEU evaluation requires at least one test example")
    score = sacrebleu.corpus_bleu(hypotheses, [references]).score
    return float(score), records
