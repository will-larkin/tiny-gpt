import pytest
import torch
from tiny_gpt.model import (
    CharTokenizer,
    GPTConfig,
    CausalSelfAttention,
    FeedForward,
    TransformerBlock,
    TinyGPT,
    make_batches,
)


def test_tokenizer():
    corpus = "hello world! 123"
    tokenizer = CharTokenizer(corpus)
    assert tokenizer.vocab_size == len(set(corpus))

    encoded = tokenizer.encode(corpus)
    assert isinstance(encoded, torch.Tensor)
    assert encoded.dtype == torch.long
    assert len(encoded) == len(corpus)

    decoded = tokenizer.decode(encoded)
    assert decoded == corpus


def test_modular_imports():
    from tiny_gpt.config import GPTConfig as SubConfig
    from tiny_gpt.tokenizer import CharTokenizer as SubTokenizer
    from tiny_gpt.attention import CausalSelfAttention as SubAttention
    from tiny_gpt.block import FeedForward as SubFFN, TransformerBlock as SubBlock
    from tiny_gpt.data import make_batches as sub_make_batches
    from tiny_gpt.model import TinyGPT as SubGPT

    assert SubConfig is GPTConfig
    assert SubTokenizer is CharTokenizer
    assert SubAttention is CausalSelfAttention
    assert SubFFN is FeedForward
    assert SubBlock is TransformerBlock
    assert sub_make_batches is make_batches
    assert SubGPT is TinyGPT


def test_causal_self_attention_shape():
    config = GPTConfig(vocab_size=20, block_size=16, d_model=32, num_heads=4)
    attn = CausalSelfAttention(config)
    x = torch.randn(2, 10, 32)
    out = attn(x)
    assert out.shape == (2, 10, 32)


def test_feed_forward_shape():
    config = GPTConfig(vocab_size=20, block_size=16, d_model=32, d_ff=64)
    ff = FeedForward(config)
    x = torch.randn(2, 10, 32)
    out = ff(x)
    assert out.shape == (2, 10, 32)


def test_transformer_block_shape():
    config = GPTConfig(vocab_size=20, block_size=16, d_model=32, num_heads=4, d_ff=64)
    block = TransformerBlock(config)
    x = torch.randn(2, 10, 32)
    out = block(x)
    assert out.shape == (2, 10, 32)


def test_tiny_gpt_forward_and_loss():
    config = GPTConfig(
        vocab_size=25,
        block_size=16,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
    )
    model = TinyGPT(config)
    idx = torch.randint(0, 25, (2, 12))

    # Without targets
    logits, loss = model(idx)
    assert logits.shape == (2, 12, 25)
    assert loss is None

    # With targets
    targets = torch.randint(0, 25, (2, 12))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 12, 25)
    assert loss is not None
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0.0


def test_tiny_gpt_backward():
    config = GPTConfig(
        vocab_size=25,
        block_size=16,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
    )
    model = TinyGPT(config)
    idx = torch.randint(0, 25, (2, 8))
    targets = torch.randint(0, 25, (2, 8))

    _, loss = model(idx, targets)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} grad contains NaN"


def test_make_batches():
    data = torch.arange(100, dtype=torch.long)
    batch_size = 4
    block_size = 8
    x, y = make_batches(data, batch_size, block_size)

    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    # y must be shifted by 1 relative to x
    assert torch.all(y[:, :-1] == x[:, 1:])


def test_tiny_gpt_generate():
    config = GPTConfig(
        vocab_size=20,
        block_size=16,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=1,
    )
    model = TinyGPT(config)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=5)

    assert generated.shape == (1, 13)
    assert torch.all(generated[:, :3] == prompt)


def test_overfitting_single_batch():
    """Verify that model capacity is sufficient to overfit a small batch rapidly."""
    torch.manual_seed(42)
    config = GPTConfig(
        vocab_size=20,
        block_size=8,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
    )
    model = TinyGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    x = torch.randint(0, 20, (2, 8))
    y = torch.randint(0, 20, (2, 8))

    losses = []
    for _ in range(50):
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Initial loss should be around ln(20) ~ 3.0, and decrease substantially
    assert losses[-1] < losses[0] * 0.2
