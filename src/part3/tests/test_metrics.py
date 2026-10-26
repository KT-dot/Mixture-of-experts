from __future__ import annotations

import math

import pytest

from src.part3.metrics import (
    corpus_perplexity,
    distinct_n,
    overlap_scores,
    positional_token_accuracy,
    repeated_trigram_fraction,
)


def test_overlap_scores_are_bag_of_words() -> None:
    precision, recall, f1 = overlap_scores("The cat sat, the end!", "the cat ran")
    # generated words: the cat sat the end (5); reference: the cat ran (3); common = 2
    assert precision == pytest.approx(2 / 5)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 * 0.4 * (2 / 3) / (0.4 + 2 / 3))
    assert overlap_scores("", "something") == (0.0, 0.0, 0.0)


def test_positional_accuracy_counts_missing_positions_wrong() -> None:
    assert positional_token_accuracy([1, 2, 9], [1, 2, 3, 4]) == 0.5
    assert positional_token_accuracy([1, 2, 3, 4, 5, 6], [1, 2, 3, 4]) == 1.0


def test_diversity_and_perplexity_helpers() -> None:
    assert distinct_n(["a b a b"], 1) == 0.5
    assert repeated_trigram_fraction(["a b c a b c"]) == pytest.approx(1 / 4)
    assert corpus_perplexity([(2.0, 1), (4.0, 3)]) == pytest.approx(math.exp(6.0 / 4))
