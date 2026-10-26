"""Dense decoder-only Transformer adapted from the Part 1 Model 1 architecture."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    mlp_hidden_size: int = 2048
    max_sequence_length: int = 256
    dropout: float = 0.1
    tie_embeddings: bool = True

    @classmethod
    def from_mapping(cls, values: dict, vocab_size: int | None = None) -> ModelConfig:
        clean = dict(values)
        if vocab_size is not None:
            clean["vocab_size"] = vocab_size
        return cls(**clean)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_model // self.n_heads % 2:
            raise ValueError("attention head size must be even for rotary embeddings")
        if self.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2")


class DenseMLP(nn.Module):
    """Part 1 Model 1's standard two-layer GELU feed-forward network."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.up = nn.Linear(config.d_model, config.mlp_hidden_size, bias=False)
        self.down = nn.Linear(config.mlp_hidden_size, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_size = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        positions = torch.arange(config.max_sequence_length)
        frequencies = 10000.0 ** (-torch.arange(0, self.head_size, 2).float() / self.head_size)
        angles = torch.outer(positions, frequencies)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def _apply_rope(self, tensor: torch.Tensor, position_offset: int) -> torch.Tensor:
        length = tensor.shape[-2]
        stop = position_offset + length
        cosine = self.rope_cos[position_offset:stop].to(tensor.dtype)[None, None]
        sine = self.rope_sin[position_offset:stop].to(tensor.dtype)[None, None]
        even, odd = tensor[..., ::2], tensor[..., 1::2]
        return torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        ).flatten(-2)

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch, length, width = x.shape
        query, key, value = (
            self.qkv(x)
            .view(batch, length, 3, self.n_heads, self.head_size)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        cached_length = 0 if cache is None else cache[0].shape[-2]
        query = self._apply_rope(query, cached_length)
        key = self._apply_rope(key, cached_length)
        if cache is not None:
            if length != 1:
                raise ValueError("cached attention accepts one new token at a time")
            key = torch.cat((cache[0], key), dim=-2)
            value = torch.cat((cache[1], value), dim=-2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=cache is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = self.out(attended.transpose(1, 2).reshape(batch, length, width))
        return output, (key, value)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp = DenseMLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attended, next_cache = self.attention(self.attention_norm(x), cache)
        x = x + self.dropout(attended)
        return x + self.dropout(self.mlp(self.mlp_norm(x))), next_cache


class DenseTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> dict[str, object]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        cached_length = 0 if cache is None else cache[0][0].shape[-2]
        if cached_length + input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("input exceeds configured maximum sequence length")
        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError("cache must contain one entry per Transformer block")
        hidden = self.dropout(self.embedding(input_ids))
        next_cache = []
        for index, block in enumerate(self.blocks):
            hidden, block_cache = block(hidden, None if cache is None else cache[index])
            if use_cache:
                next_cache.append(block_cache)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels and input_ids must have the same shape")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss, "cache": next_cache if use_cache else None}

    @torch.no_grad()
    def greedy_generate(self, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate with the model's forward pass and deterministic greedy decoding."""
        if prompt.ndim != 2:
            raise ValueError("prompt must have shape [batch, sequence]")
        if prompt.shape[1] + max_new_tokens > self.config.max_sequence_length:
            raise ValueError("prompt and continuation exceed the model context window")
        generated = prompt
        output = self(prompt, use_cache=True)
        for step in range(max_new_tokens):
            next_token = output["logits"][:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if step + 1 < max_new_tokens:
                output = self(next_token, cache=output["cache"], use_cache=True)
        return generated


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
