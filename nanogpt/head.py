
import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """Single masked self-attention head."""

    def __init__(self, n_embd: int, block_size: int, head_size: int, dropout: float) -> None:
        """Create projection layers and causal mask for one attention head."""
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.head_size = head_size
        self.tril: torch.Tensor
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply masked self-attention to input embeddings."""
        _, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        weights = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        out = weights @ v
        return out
