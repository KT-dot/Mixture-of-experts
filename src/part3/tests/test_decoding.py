from __future__ import annotations

import itertools
import math

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from src.part3.decoding import (
    IncrementalLM,
    beam_search_decode,
    greedy_decode,
    sample_decode,
    top_k_filter,
    top_p_filter,
)

VOCAB = 48
LOGIT_WIDTH = 64  # wider than the tokenizer, as with Pythia's padded output layer


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = GPTNeoXConfig(
        vocab_size=LOGIT_WIDTH,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
        rotary_pct=0.5,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    network = GPTNeoXForCausalLM(config).eval()
    network.generate = None  # any accidental use fails loudly
    return network


def lm_for(model) -> IncrementalLM:
    return IncrementalLM(model, pad_token_id=0, vocab_size=VOCAB)


def no_stop() -> torch.Tensor:
    return torch.zeros(LOGIT_WIDTH, dtype=torch.bool)


PROMPTS = [[5, 9, 13, 2, 7], [11, 3], [8, 8, 30, 21]]


def reference_greedy(model, prompt: list[int], steps: int) -> list[int]:
    """Uncached, unpadded argmax decoding by re-running the whole sequence."""
    sequence = list(prompt)
    with torch.inference_mode():
        for _ in range(steps):
            logits = model(input_ids=torch.tensor([sequence])).logits[0, -1, :VOCAB]
            sequence.append(int(logits.argmax()))
    return sequence[len(prompt) :]


def test_cached_left_padded_greedy_matches_uncached_reference(model) -> None:
    result = greedy_decode(lm_for(model), PROMPTS, max_new_tokens=6, stop_mask=no_stop())
    for prompt, generated in zip(PROMPTS, result.sequences, strict=True):
        assert generated == reference_greedy(model, prompt, 6)
    assert result.forward_calls == 6


def test_cached_step_logits_match_full_forward(model) -> None:
    lm = lm_for(model)
    logits, state = lm.start(PROMPTS)
    token = logits.argmax(-1)
    stepped = lm.step(token, state)
    for row, prompt in enumerate(PROMPTS):
        full = model(input_ids=torch.tensor([prompt + [int(token[row])]])).logits[0, -1, :VOCAB]
        torch.testing.assert_close(stepped[row, :VOCAB], full, atol=1e-4, rtol=1e-4)
    assert torch.isinf(stepped[:, VOCAB:]).all()


def test_beam_width_one_equals_greedy(model) -> None:
    stop = no_stop()
    stop[[4, 17]] = True
    greedy = greedy_decode(lm_for(model), PROMPTS, max_new_tokens=10, stop_mask=stop)
    beam = beam_search_decode(
        lm_for(model), PROMPTS, beam_width=1, max_new_tokens=10, stop_mask=stop
    )
    assert beam.sequences == greedy.sequences
    assert beam.stopped == greedy.stopped


def test_degenerate_sampling_equals_greedy(model) -> None:
    greedy = greedy_decode(lm_for(model), PROMPTS, max_new_tokens=5, stop_mask=no_stop())
    for kwargs in ({"top_k": 1}, {"top_p": 1e-6}):
        sampled = sample_decode(
            lm_for(model),
            PROMPTS,
            max_new_tokens=5,
            stop_mask=no_stop(),
            generator=torch.Generator().manual_seed(3),
            **kwargs,
        )
        assert sampled.sequences == greedy.sequences


def test_sampling_is_seed_reproducible_and_seed_dependent(model) -> None:
    def run(seed: int):
        return sample_decode(
            lm_for(model),
            PROMPTS,
            max_new_tokens=8,
            stop_mask=no_stop(),
            generator=torch.Generator().manual_seed(seed),
        ).sequences

    assert run(1) == run(1)
    assert run(1) != run(2)
    assert all(token < VOCAB for row in run(4) for token in row)


def test_exhaustive_beam_finds_the_best_sequence(model) -> None:
    # With width >= |V|^(T-1) no prefix is ever pruned, so beam search must return the
    # exact maximum-probability continuation (length penalty 0 = raw log-probability).
    prompt = [5, 9, 13]
    steps = 2
    best_score, best = -math.inf, None
    with torch.inference_mode():
        for candidate in itertools.product(range(VOCAB), repeat=steps):
            ids = torch.tensor([prompt + list(candidate)])
            logp = model(input_ids=ids).logits[0, :, :VOCAB].float().log_softmax(-1)
            score = sum(float(logp[len(prompt) - 1 + i, t]) for i, t in enumerate(candidate))
            if score > best_score:
                best_score, best = score, list(candidate)
    result = beam_search_decode(
        lm_for(model),
        [prompt],
        beam_width=VOCAB,
        max_new_tokens=steps,
        stop_mask=no_stop(),
        length_penalty=0.0,
    )
    assert result.sequences[0] == best


def test_beam_search_rows_are_independent_of_batching(model) -> None:
    stop = no_stop()
    stop[[4, 17]] = True
    batched = beam_search_decode(
        lm_for(model), PROMPTS, beam_width=3, max_new_tokens=8, stop_mask=stop
    )
    for prompt, sequence in zip(PROMPTS, batched.sequences, strict=True):
        alone = beam_search_decode(
            lm_for(model), [prompt], beam_width=3, max_new_tokens=8, stop_mask=stop
        )
        assert alone.sequences[0] == sequence


def test_top_k_and_top_p_filters() -> None:
    logits = torch.log(torch.tensor([[0.5, 0.2, 0.15, 0.1, 0.05]]))
    kept_k = torch.isfinite(top_k_filter(logits, 2))
    assert kept_k.tolist() == [[True, True, False, False, False]]
    # Mass reaches 0.7 only after the 2nd token and 0.85 after the 3rd.
    assert torch.isfinite(top_p_filter(logits, 0.7)).tolist() == [[True, True, False, False, False]]
    assert torch.isfinite(top_p_filter(logits, 0.71)).tolist() == [[True, True, True, False, False]]
    shuffled = logits[:, [3, 0, 4, 2, 1]]
    assert torch.isfinite(top_p_filter(shuffled, 0.7)).tolist() == [
        [False, True, False, False, True]
    ]


def test_banned_tokens_are_never_emitted(model) -> None:
    unconstrained = greedy_decode(lm_for(model), PROMPTS, max_new_tokens=6, stop_mask=no_stop())
    banned = torch.zeros(LOGIT_WIDTH, dtype=torch.bool)
    banned[sorted({token for row in unconstrained.sequences for token in row})] = True
    lm = IncrementalLM(model, pad_token_id=0, vocab_size=VOCAB, banned_mask=banned)
    runs = [
        greedy_decode(lm, PROMPTS, max_new_tokens=6, stop_mask=no_stop()),
        beam_search_decode(lm, PROMPTS, beam_width=3, max_new_tokens=6, stop_mask=no_stop()),
        sample_decode(
            lm,
            PROMPTS,
            max_new_tokens=6,
            stop_mask=no_stop(),
            generator=torch.Generator().manual_seed(0),
        ),
    ]
    for result in runs:
        assert not any(banned[token] for row in result.sequences for token in row)
