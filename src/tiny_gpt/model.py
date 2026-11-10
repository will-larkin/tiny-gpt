"""TinyGPT model architecture."""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from tiny_gpt.attention import CausalSelfAttention
from tiny_gpt.block import FeedForward, TransformerBlock
from tiny_gpt.config import GPTConfig
from tiny_gpt.data import make_batches
from tiny_gpt.norm import RMSNorm
from tiny_gpt.tokenizer import CharTokenizer


class TinyGPT(nn.Module):
    """Minimal GPT language model in PyTorch."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        # Apply normal weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for token IDs.

        Args:
            idx: (B, T) LongTensor of token indices.
            targets: (B, T) Optional LongTensor of target token indices.

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar loss Tensor if targets provided, else None
        """
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block_size is {self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)

        # Embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, d_model)
        pos_emb = self.pos_embedding(pos)    # (T, d_model)
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively generate new tokens from a prefix prompt.

        Args:
            idx: (B, T) LongTensor of prompt token indices.
            max_new_tokens: Number of tokens to generate.
            temperature: Softmax temperature (> 0.0).
            top_k: Optional top-k filtering.

        Returns:
            idx: (B, T + max_new_tokens) LongTensor.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


__all__ = [
    "TinyGPT",
    "GPTConfig",
    "CharTokenizer",
    "CausalSelfAttention",
    "FeedForward",
    "TransformerBlock",
    "RMSNorm",
    "make_batches",
]
