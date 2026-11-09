# Tiny GPT in PyTorch

A clean, minimal, and modern GPT-style language model implemented in PyTorch. Includes character tokenization, causal multi-head self-attention, pre-norm transformer blocks, AdamW optimization, and autoregressive text generation with temperature and top-k sampling.

## How to run

Run the end-to-end training and text generation demo scaffold:

```bash
uv run python -m tiny_gpt.scaffold
```

Or run automated tests:

```bash
uv run pytest -v
```

## Module Structure

The codebase is organized into clean, focused modules under `src/tiny_gpt/`:

- **[`config.py`](src/tiny_gpt/config.py)**: `GPTConfig` dataclass for hyperparameters.
- **[`tokenizer.py`](src/tiny_gpt/tokenizer.py)**: `CharTokenizer` for character-level encoding and decoding.
- **[`attention.py`](src/tiny_gpt/attention.py)**: `CausalSelfAttention` multi-head causal attention layer.
- **[`block.py`](src/tiny_gpt/block.py)**: `FeedForward` MLP and `TransformerBlock` pre-norm residual layer.
- **[`data.py`](src/tiny_gpt/data.py)**: `make_batches` batch sampling helper.
- **[`model.py`](src/tiny_gpt/model.py)**: `TinyGPT` model assembly, forward pass, and autoregressive `generate`.
- **[`scaffold.py`](src/tiny_gpt/scaffold.py)**: End-to-end training and sampling demo.
