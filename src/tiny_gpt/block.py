"""Transformer block and Feed-Forward sublayers."""

from __future__ import annotations

import torch
import torch.nn as nn

from tiny_gpt.attention import CausalSelfAttention
from tiny_gpt.norm import RMSNorm
from tiny_gpt.config import GPTConfig


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Transformer block with pre-layer normalization and residual connections."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
