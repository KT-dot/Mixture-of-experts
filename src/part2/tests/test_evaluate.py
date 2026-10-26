from __future__ import annotations

import torch

from src.part2.data import SyntheticTokenizer, TokenBlockDataset
from src.part2.evaluate import evaluate_continuation_bleu
from src.part2.model import DenseTransformer, ModelConfig


def test_continuation_fits_context_window_at_the_boundary() -> None:
    # Regression: the continuation allowance used the block length (sequence_length+1),
    # asking a context-8 model for a 4-token prompt plus 5 new tokens.
    torch.manual_seed(0)
    model = DenseTransformer(
        ModelConfig(
            vocab_size=16,
            d_model=16,
            n_heads=4,
            n_layers=1,
            mlp_hidden_size=32,
            max_sequence_length=8,
            dropout=0.0,
        )
    )
    blocks = torch.randint(4, 16, (3, 9))
    _, records = evaluate_continuation_bleu(
        model,
        TokenBlockDataset(blocks),
        SyntheticTokenizer(16),
        optimizer_name="adamw",
        training_tokens_seen=0,
        checkpoint_fraction=1.0,
        device=torch.device("cpu"),
        prefix_fraction=0.5,
        generation_tokens=64,
        max_examples=None,
        batch_size=2,
    )
    assert len(records) == 3
    assert {record["prefix_tokens"] for record in records} == {4}
    assert {record["continuation_tokens"] for record in records} == {4}
