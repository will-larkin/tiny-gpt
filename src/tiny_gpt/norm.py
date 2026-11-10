"""Root Mean Square Layer Normalization (RMSNorm)."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        if not isinstance(dim, int) and hasattr(dim, "d_model"):
            eps = getattr(dim, "norm_eps", eps)
            dim = dim.d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + self.eps)
        return (x_norm.to(input_dtype)) * self.weight
