"""Causal self-attention mechanisms."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_gpt.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention layer."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.d_model % config.num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.d_model = config.d_model

        # Q, K, V combined linear projection
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask buffer
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Compute Q, K, V
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=2)
        # Reshape to (B, num_heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.bias[:, :, :T, :T], float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        y = attn_weights @ v  # (B, num_heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(y))
