import inspect
import math

import torch

import minigpt.model as model_module
from minigpt.model import GPT, MLP, Block, CausalSelfAttention, GPTConfig

CFG = GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=4, n_embd=32)


def test_no_shortcuts():
    src = inspect.getsource(model_module)
    code = "\n".join(line.split("#")[0] for line in src.split('"""')[::2])
    for banned in ("MultiheadAttention", "scaled_dot_product_attention", "TransformerEncoder"):
        assert banned not in code, f"{banned} is not allowed; write attention by hand"


# Milestone 2 -----------------------------------------------------------------


def test_attention_shape():
    attn = CausalSelfAttention(CFG)
    x = torch.randn(2, 10, CFG.n_embd)
    assert attn(x).shape == x.shape


def test_attention_is_causal():
    torch.manual_seed(0)
    attn = CausalSelfAttention(CFG).eval()
    x = torch.randn(1, 10, CFG.n_embd)
    x2 = x.clone()
    x2[0, 6:] = torch.randn(4, CFG.n_embd)  # change only the future
    out, out2 = attn(x), attn(x2)
    assert torch.allclose(out[0, :6], out2[0, :6], atol=1e-6), "positions < 6 saw the future"
    assert not torch.allclose(out[0, 6:], out2[0, 6:])


def test_attention_uses_the_past():
    torch.manual_seed(0)
    attn = CausalSelfAttention(CFG).eval()
    x = torch.randn(1, 10, CFG.n_embd)
    x2 = x.clone()
    x2[0, 0] = torch.randn(CFG.n_embd)  # change only the first token
    assert not torch.allclose(attn(x)[0, 9], attn(x2)[0, 9]), "last position ignores the past"


# Milestone 3 -----------------------------------------------------------------


def test_mlp_and_block_shapes():
    x = torch.randn(2, 10, CFG.n_embd)
    assert MLP(CFG)(x).shape == x.shape
    assert Block(CFG)(x).shape == x.shape


def test_block_has_a_gradient_path_through_the_residual():
    block = Block(CFG)
    x = torch.randn(1, 4, CFG.n_embd, requires_grad=True)
    for p in block.parameters():
        torch.nn.init.zeros_(p)  # kill both sub-layers; only the skip path remains
    assert torch.allclose(block(x), x, atol=1e-6)


# Milestone 4 -----------------------------------------------------------------


def test_gpt_logits_and_loss():
    torch.manual_seed(0)
    model = GPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (4, 16))
    logits, loss = model(idx)
    assert logits.shape == (4, 16, CFG.vocab_size)
    assert loss is None
    _, loss = model(idx, targets=idx)
    assert loss.ndim == 0


def test_initial_loss_is_near_uniform():
    torch.manual_seed(0)
    model = GPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (16, 16))
    tgt = torch.randint(0, CFG.vocab_size, (16, 16))
    _, loss = model(idx, tgt)
    expected = math.log(CFG.vocab_size)
    assert abs(loss.item() - expected) < 0.5, (
        f"untrained loss should be about ln(vocab_size) = {expected:.2f}, got {loss.item():.2f}; "
        "check your weight initialisation"
    )


def test_gpt_is_causal_end_to_end():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    idx = torch.randint(0, CFG.vocab_size, (1, 16))
    idx2 = idx.clone()
    idx2[0, 10:] = (idx2[0, 10:] + 1) % CFG.vocab_size
    a, _ = model(idx)
    b, _ = model(idx2)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-5)


def test_position_matters():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    idx = torch.full((1, 8), 7)  # the same token at every position
    logits, _ = model(idx)
    assert not torch.allclose(logits[0, 0], logits[0, 1], atol=1e-6), "no position information"


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    model = GPT(CFG)
    idx = torch.randint(0, CFG.vocab_size, (4, 16))
    tgt = torch.randint(0, CFG.vocab_size, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = model(idx, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, f"could not memorise 64 tokens (loss {loss.item():.3f})"


def test_generate():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    start = torch.zeros((2, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=40)  # 41 > block_size, so cropping is required
    assert out.shape == (2, 41)
    assert torch.equal(out[:, :1], start)
    assert out.min() >= 0 and out.max() < CFG.vocab_size


def test_generate_top_k_1_is_greedy():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    start = torch.zeros((1, 1), dtype=torch.long)
    a = model.generate(start, max_new_tokens=10, top_k=1)
    b = model.generate(start, max_new_tokens=10, top_k=1)
    assert torch.equal(a, b)
    logits, _ = model(start)
    assert a[0, 1].item() == logits[0, -1].argmax().item()
