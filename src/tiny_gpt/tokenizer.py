"""Character-level tokenizer for TinyGPT."""

from __future__ import annotations

from typing import List, Optional, Union
import torch


class CharTokenizer:
    """Character-level tokenizer for text data."""

    def __init__(self, text: str) -> None:
        chars = sorted(list(set(text)))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """Encode string to 1D Tensor of token IDs."""
        ids = [self.char_to_id[ch] for ch in text if ch in self.char_to_id]
        return torch.tensor(ids, dtype=torch.long, device=device)

    def decode(self, ids: Union[torch.Tensor, List[int]]) -> str:
        """Decode a sequence of token IDs to string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join([self.id_to_char[i] for i in ids if i in self.id_to_char])
