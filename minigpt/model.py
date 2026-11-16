"""Milestones 2-4: the transformer itself.

Rules:
  - Do NOT use nn.MultiheadAttention, nn.TransformerEncoderLayer or
    F.scaled_dot_product_attention. Writing attention by hand is the point.
  - nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout and F.softmax are allowed.

Check your work with:  pytest tests/test_model.py
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64  # maximum context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """Milestone 2. Multi-head self-attention where position t sees only positions <= t.

    Input and output shape: (B, T, n_embd).

    Before you write this, answer on paper:
      1. Why divide the attention scores by sqrt(head_dim)?
      2. Why is the mask applied before the softmax and not after?
      3. What is the shape of the score tensor for B=2, T=8, n_head=4?
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # TODO
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


class MLP(nn.Module):
    """Milestone 3. Linear(n_embd -> 4*n_embd), GELU, Linear(4*n_embd -> n_embd), dropout."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


class Block(nn.Module):
    """Milestone 3. Pre-norm residual block:

        x = x + attn(ln_1(x))
        x = x + mlp(ln_2(x))

    Question: what breaks in a deep network if you remove the two `x +` terms?
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


class GPT(nn.Module):
    """Milestone 4. Token embedding + learned position embedding, n_layer blocks,
    a final LayerNorm, and a linear head to vocab_size.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # TODO
        raise NotImplementedError

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: (B, T) token ids, with T <= block_size.

        Returns (logits, loss). logits has shape (B, T, vocab_size). loss is
        the mean cross-entropy against `targets` (shape (B, T)), or None when
        targets is None.
        """
        # TODO
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Append `max_new_tokens` sampled tokens to idx (B, T) and return it.

        Each step: crop the context to the last block_size tokens, take the
        logits at the final position, divide by temperature, optionally keep
        only the top_k logits, sample one token with torch.multinomial.
        """
        # TODO
        raise NotImplementedError
