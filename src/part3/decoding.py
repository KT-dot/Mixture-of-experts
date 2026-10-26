"""Decoding strategies built only on a causal LM's ``forward()``.

``model.generate()`` is never called. Every strategy drives the model one step at a
time through :class:`IncrementalLM`, which owns left padding, position ids and the
key/value cache, so the strategies themselves only see next-token logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class DecodeState:
    cache: object
    attention_mask: torch.Tensor
    next_position: torch.Tensor
    forward_calls: int = 0


@dataclass
class DecodeResult:
    """Generated ids per example (stop token included, padding removed)."""

    sequences: list[list[int]]
    stopped: list[bool]
    forward_calls: int


class IncrementalLM:
    """Step-wise wrapper around a Hugging Face causal LM's forward pass.

    Prompts are left-padded so that every row's next token is at the last position;
    position ids are recomputed from the attention mask so padding never shifts the
    rotary positions of real tokens. Logits beyond the tokenizer's vocabulary (Pythia
    pads its output layer to a multiple of 128) are masked out, as are any ids in
    ``banned_mask`` (an output-format constraint applied identically to every strategy).
    """

    def __init__(
        self,
        model,
        *,
        pad_token_id: int,
        vocab_size: int,
        banned_mask: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.banned_mask = banned_mask

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        if logits.shape[-1] > self.vocab_size:
            logits[..., self.vocab_size :] = float("-inf")
        if self.banned_mask is not None:
            logits = logits.masked_fill(self.banned_mask, float("-inf"))
        return logits

    def left_pad(self, prompts: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        if not prompts or any(len(prompt) == 0 for prompt in prompts):
            raise ValueError("every prompt must contain at least one token")
        width = max(len(prompt) for prompt in prompts)
        ids = torch.full((len(prompts), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(prompts), width), dtype=torch.long)
        for row, prompt in enumerate(prompts):
            ids[row, width - len(prompt) :] = torch.tensor(prompt, dtype=torch.long)
            mask[row, width - len(prompt) :] = 1
        return ids.to(self.device), mask.to(self.device)

    def start(self, prompts: list[list[int]]) -> tuple[torch.Tensor, DecodeState]:
        input_ids, attention_mask = self.left_pad(prompts)
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        state = DecodeState(
            cache=output.past_key_values,
            attention_mask=attention_mask,
            next_position=position_ids[:, -1] + 1,
            forward_calls=1,
        )
        return self._mask_logits(output.logits[:, -1]), state

    def step(self, tokens: torch.Tensor, state: DecodeState) -> torch.Tensor:
        state.attention_mask = torch.cat(
            (state.attention_mask, state.attention_mask.new_ones((tokens.shape[0], 1))), dim=1
        )
        output = self.model(
            input_ids=tokens[:, None],
            attention_mask=state.attention_mask,
            position_ids=state.next_position[:, None],
            past_key_values=state.cache,
            use_cache=True,
        )
        state.cache = output.past_key_values
        state.next_position = state.next_position + 1
        state.forward_calls += 1
        return self._mask_logits(output.logits[:, -1])

    @staticmethod
    def reorder(state: DecodeState, index: torch.Tensor) -> None:
        """Select/duplicate batch rows (beam expansion and beam re-ranking)."""
        state.cache.reorder_cache(index)
        state.attention_mask = state.attention_mask.index_select(0, index)
        state.next_position = state.next_position.index_select(0, index)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k highest logits per row (ties at the k-th value are kept)."""
    if k < 1:
        raise ValueError("top-k requires k >= 1")
    k = min(k, logits.shape[-1])
    threshold = logits.topk(k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Keep the smallest highest-probability set whose total mass reaches p."""
    if not 0 < p <= 1:
        raise ValueError("top-p requires 0 < p <= 1")
    sorted_logits, order = logits.sort(dim=-1, descending=True)
    probabilities = sorted_logits.softmax(dim=-1)
    # Mass strictly before each token; the most likely token always survives.
    mass_before = probabilities.cumsum(dim=-1) - probabilities
    remove_sorted = mass_before >= p
    remove = remove_sorted.scatter(-1, order, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def _truncate(tokens: list[int], stop_ids: set[int]) -> tuple[list[int], bool]:
    for index, token in enumerate(tokens):
        if token in stop_ids:
            return tokens[: index + 1], True
    return tokens, False


def _step_decode(
    lm: IncrementalLM,
    prompts: list[list[int]],
    *,
    max_new_tokens: int,
    stop_mask: torch.Tensor,
    choose,
) -> DecodeResult:
    """Shared loop for greedy and sampling: one token per row per forward call."""
    logits, state = lm.start(prompts)
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=logits.device)
    chosen: list[torch.Tensor] = []
    for step in range(max_new_tokens):
        token = choose(logits)
        token = token.masked_fill(finished, lm.pad_token_id)
        chosen.append(token)
        finished |= stop_mask[token]
        if bool(finished.all()) or step + 1 == max_new_tokens:
            break
        logits = lm.step(token, state)
    stop_ids = set(torch.nonzero(stop_mask).flatten().tolist())
    matrix = torch.stack(chosen, dim=1).tolist()
    sequences, stopped = zip(*(_truncate(row, stop_ids) for row in matrix), strict=True)
    return DecodeResult(list(sequences), list(stopped), state.forward_calls)


@torch.inference_mode()
def greedy_decode(
    lm: IncrementalLM, prompts: list[list[int]], *, max_new_tokens: int, stop_mask: torch.Tensor
) -> DecodeResult:
    return _step_decode(
        lm,
        prompts,
        max_new_tokens=max_new_tokens,
        stop_mask=stop_mask,
        choose=lambda logits: logits.argmax(dim=-1),
    )


@torch.inference_mode()
def sample_decode(
    lm: IncrementalLM,
    prompts: list[list[int]],
    *,
    max_new_tokens: int,
    stop_mask: torch.Tensor,
    generator: torch.Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> DecodeResult:
    """Ancestral sampling with optional top-k and/or top-p (nucleus) truncation."""
    if temperature <= 0:
        raise ValueError("temperature must be positive; use greedy_decode for T=0")

    def choose(logits: torch.Tensor) -> torch.Tensor:
        logits = logits / temperature
        if top_k is not None:
            logits = top_k_filter(logits, top_k)
        if top_p is not None:
            logits = top_p_filter(logits, top_p)
        probabilities = logits.softmax(dim=-1)
        return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)

    return _step_decode(
        lm, prompts, max_new_tokens=max_new_tokens, stop_mask=stop_mask, choose=choose
    )


@torch.inference_mode()
def beam_search_decode(
    lm: IncrementalLM,
    prompts: list[list[int]],
    *,
    beam_width: int,
    max_new_tokens: int,
    stop_mask: torch.Tensor,
    length_penalty: float = 1.0,
) -> DecodeResult:
    """Batched beam search with a KV cache.

    Each step expands every live beam by the whole vocabulary and keeps the 2k best
    (beam, token) pairs of each example by cumulative log-probability. A pair ending in
    a stop token becomes a finished hypothesis if it ranks inside the top k; the other
    pairs fill the k live beams. An example is done once it holds k finished
    hypotheses (``early_stopping=True`` semantics); at the length limit its live beams
    are added as unfinished hypotheses. The winner maximises
    ``sum_logprob / generated_length ** length_penalty``. With k=1 this is exactly
    greedy decoding.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    batch = len(prompts)
    k = beam_width
    logits, state = lm.start(prompts)
    device = logits.device
    expand = torch.arange(batch, device=device).repeat_interleave(k)
    lm.reorder(state, expand)
    logits = logits.index_select(0, expand)
    beam_scores = torch.zeros((batch, k), device=device)
    beam_scores[:, 1:] = float("-inf")  # all k beams start identical; keep one
    beam_tokens: list[list[int]] = [[] for _ in range(batch * k)]
    finished: list[list[tuple[float, list[int], bool]]] = [[] for _ in range(batch)]
    done = [False] * batch
    stop_list = stop_mask.tolist()
    vocab = logits.shape[-1]

    for step in range(max_new_tokens):
        log_probs = F.log_softmax(logits, dim=-1)
        candidates = (beam_scores.view(-1, 1) + log_probs).view(batch, k * vocab)
        top_scores, top_index = candidates.topk(2 * k, dim=1)
        top_scores_list = top_scores.tolist()
        top_beam_list = (top_index // vocab).tolist()
        top_token_list = (top_index % vocab).tolist()
        next_scores = torch.full((batch, k), float("-inf"), device=device)
        next_tokens = torch.full((batch, k), lm.pad_token_id, dtype=torch.long, device=device)
        next_source = torch.arange(batch * k, device=device).view(batch, k).clone()
        new_beam_tokens: list[list[int]] = [[] for _ in range(batch * k)]
        for example in range(batch):
            if done[example]:
                continue
            filled = 0
            for rank in range(2 * k):
                score = top_scores_list[example][rank]
                if score == float("-inf"):
                    break
                source = example * k + top_beam_list[example][rank]
                token = top_token_list[example][rank]
                tokens = beam_tokens[source] + [token]
                if stop_list[token]:
                    if rank < k:
                        normalised = score / (len(tokens) ** length_penalty)
                        finished[example].append((normalised, tokens, True))
                    continue
                slot = example * k + filled
                next_scores[example, filled] = score
                next_tokens[example, filled] = token
                next_source[example, filled] = source
                new_beam_tokens[slot] = tokens
                filled += 1
                if filled == k:
                    break
            if len(finished[example]) >= k:
                done[example] = True
        beam_tokens = new_beam_tokens
        beam_scores = next_scores
        if all(done) or step + 1 == max_new_tokens:
            break
        lm.reorder(state, next_source.view(-1))
        logits = lm.step(next_tokens.view(-1), state)

    sequences: list[list[int]] = []
    stopped: list[bool] = []
    for example in range(batch):
        hypotheses = list(finished[example])
        if not done[example]:
            for beam in range(k):
                score = float(beam_scores[example, beam])
                tokens = beam_tokens[example * k + beam]
                if tokens and score != float("-inf"):
                    normalised = score / (len(tokens) ** length_penalty)
                    hypotheses.append((normalised, tokens, False))
        best = max(hypotheses, key=lambda item: item[0])
        sequences.append(best[1])
        stopped.append(best[2])
    return DecodeResult(sequences, stopped, state.forward_calls)
