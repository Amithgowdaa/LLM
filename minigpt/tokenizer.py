"""Milestone 1: character-level tokenizer and batch sampling.

Check your work with:  pytest tests/test_tokenizer.py
"""

import torch


class CharTokenizer:
    """Maps each distinct character in `text` to an integer id.

    Ids must be assigned in sorted character order, so the same text always
    gives the same vocabulary.
    """

    def __init__(self, text: str):
        # TODO: build the vocabulary and the two lookup tables (char -> id, id -> char).
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, s: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` random windows from a 1-D tensor of token ids.

    Returns (x, y), both of shape (batch_size, block_size), where y is x
    shifted one position to the right: y[b, t] is the token that follows
    x[b, t] in `data`.

    Before you write this, answer on paper: what is the largest valid start
    index, and why is it not len(data) - block_size?
    """
    # TODO
    raise NotImplementedError
