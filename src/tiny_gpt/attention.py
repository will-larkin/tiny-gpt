"""Causal self-attention mechanisms."""

from __future__ import annotations

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
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Compute Q, K, V
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=2)
        # Reshape to (B, num_heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.attn_dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(y))
