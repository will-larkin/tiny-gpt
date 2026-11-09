"""Dataset and batching utilities for TinyGPT."""

from __future__ import annotations

from typing import Optional, Tuple
import torch


def make_batches(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of inputs and targets from tokenized data."""
    max_idx = len(data) - block_size
    ix = torch.randint(0, max_idx, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y
