"""Scoring of generated continuations against ROCStories references.

The assignment names perplexity, accuracy, precision, recall and F1 without fixing a
protocol for open-ended text, so the definitions used here are explicit:

* perplexity (model-based, via ``forward()``): ``exp(total NLL / total tokens)`` of a
  continuation given its prompt, under the unmodified model distribution (T=1, no
  truncation). Computed for the reference continuation (strategy-independent) and for
  each strategy's own generations.
* accuracy: position-wise token accuracy -- the fraction of reference token positions
  i whose generated token at position i is identical (missing positions count wrong).
* precision / recall / F1: bag-of-words overlap (SQuAD-style) between normalised
  generated and reference words, computed per story and macro-averaged.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter

import torch
from torch.nn import functional as F

_PUNCTUATION = re.compile(f"[{re.escape(string.punctuation)}]")


def normalise_words(text: str) -> list[str]:
    return _PUNCTUATION.sub(" ", text.lower()).split()


def overlap_scores(generated: str, reference: str) -> tuple[float, float, float]:
    """Per-example (precision, recall, F1) of word multisets."""
    hypothesis = normalise_words(generated)
    target = normalise_words(reference)
    if not hypothesis and not target:
        return 1.0, 1.0, 1.0
    if not hypothesis or not target:
        return 0.0, 0.0, 0.0
    common = sum((Counter(hypothesis) & Counter(target)).values())
    if common == 0:
        return 0.0, 0.0, 0.0
    precision = common / len(hypothesis)
    recall = common / len(target)
    return precision, recall, 2 * precision * recall / (precision + recall)


def positional_token_accuracy(generated_ids: list[int], reference_ids: list[int]) -> float:
    if not reference_ids:
        raise ValueError("reference continuation is empty")
    matches = sum(
        1
        for index, token in enumerate(reference_ids)
        if index < len(generated_ids) and generated_ids[index] == token
    )
    return matches / len(reference_ids)


def distinct_n(texts: list[str], n: int) -> float:
    """Corpus-level unique n-grams / total n-grams over normalised words."""
    total = 0
    unique: set[tuple[str, ...]] = set()
    for text in texts:
        words = normalise_words(text)
        grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def repeated_trigram_fraction(texts: list[str]) -> float:
    """Fraction of word trigrams that already occurred earlier in the same text."""
    repeated = 0
    total = 0
    for text in texts:
        words = normalise_words(text)
        seen: set[tuple[str, ...]] = set()
        for i in range(len(words) - 2):
            gram = tuple(words[i : i + 3])
            repeated += gram in seen
            seen.add(gram)
            total += 1
    return repeated / total if total else 0.0


@torch.inference_mode()
def continuation_nll(
    model,
    prompts: list[list[int]],
    continuations: list[list[int]],
    *,
    pad_token_id: int,
    vocab_size: int,
    batch_size: int,
) -> list[tuple[float, int]]:
    """Per-example (summed NLL in nats, token count) of each continuation given its prompt.

    Sequences are right-padded; only continuation positions contribute. Empty
    continuations return (0.0, 0).
    """
    device = next(model.parameters()).device
    results: list[tuple[float, int]] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch_continuations = continuations[start : start + batch_size]
        rows = [p + c for p, c in zip(batch_prompts, batch_continuations, strict=True)]
        width = max(len(row) for row in rows)
        ids = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        labels = torch.full((len(rows), width), -100, dtype=torch.long)
        for index, (prompt, continuation) in enumerate(
            zip(batch_prompts, batch_continuations, strict=True)
        ):
            row = prompt + continuation
            ids[index, : len(row)] = torch.tensor(row)
            mask[index, : len(row)] = 1
            if continuation:
                labels[index, len(prompt) : len(row)] = torch.tensor(continuation)
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
        logits[..., vocab_size:] = float("-inf")
        targets = labels[:, 1:]
        token_nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(targets.shape)
        valid = targets.ne(-100)
        sums = (token_nll * valid).sum(dim=1).tolist()
        counts = valid.sum(dim=1).tolist()
        results.extend(zip(sums, counts, strict=True))
    return results


def corpus_perplexity(nll: list[tuple[float, int]]) -> float:
    total_nll = sum(value for value, _ in nll)
    total_tokens = sum(count for _, count in nll)
    return math.exp(total_nll / total_tokens) if total_tokens else math.nan


def mean_and_sem(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, math.nan
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))
