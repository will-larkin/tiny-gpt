"""Configuration definitions for TinyGPT."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Configuration for TinyGPT architecture."""

    vocab_size: int
    block_size: int = 16
    d_model: int = 32
    num_heads: int = 4
    d_ff: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    norm_eps: float = 1e-5
