import torch

from minigpt.tokenizer import CharTokenizer, get_batch

TEXT = "hello world, hello gate"


def test_vocab_is_sorted_unique_chars():
    tok = CharTokenizer(TEXT)
    assert tok.vocab_size == len(set(TEXT))
    assert tok.encode(" ") == [0]  # space sorts first in this text
    assert tok.encode("w") == [tok.vocab_size - 1]


def test_roundtrip():
    tok = CharTokenizer(TEXT)
    ids = tok.encode("hello gate")
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == "hello gate"


def test_batch_shapes_and_dtype():
    data = torch.arange(1000)
    x, y = get_batch(data, block_size=16, batch_size=8)
    assert x.shape == (8, 16) and y.shape == (8, 16)
    assert x.dtype == torch.long and y.dtype == torch.long


def test_targets_are_inputs_shifted_by_one():
    data = torch.arange(1000)  # token at position i is i, so the next token is x + 1
    x, y = get_batch(data, block_size=16, batch_size=32)
    assert torch.equal(y, x + 1)


def test_batch_never_reads_past_the_end():
    data = torch.arange(10)
    for _ in range(200):
        x, y = get_batch(data, block_size=9, batch_size=4)
        assert torch.equal(x[0], torch.arange(9))  # only one window fits
        assert torch.equal(y[0], torch.arange(1, 10))


def test_batch_covers_the_last_window():
    data = torch.arange(12)
    starts = set()
    for _ in range(300):
        x, _ = get_batch(data, block_size=8, batch_size=4)
        starts.update(x[:, 0].tolist())
    assert starts == {0, 1, 2, 3}


def test_generator_makes_batches_reproducible():
    data = torch.arange(1000)
    a, _ = get_batch(data, 16, 8, generator=torch.Generator().manual_seed(0))
    b, _ = get_batch(data, 16, 8, generator=torch.Generator().manual_seed(0))
    assert torch.equal(a, b)
