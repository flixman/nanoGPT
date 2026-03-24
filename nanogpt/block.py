
import torch
import torch.nn as nn

from .head import Head


class MultiHeadAttention(nn.Module):
    """Parallel composition of multiple attention heads."""

    def __init__(self, num_heads: int, n_embd: int, block_size: int, head_size: int, dropout: float) -> None:
        """Create multiple attention heads and output projection."""
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, block_size, head_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate head outputs and project back to embedding size."""
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network used inside Transformer blocks."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        """Create the MLP expansion and projection layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MLP to each token embedding independently."""
        return self.net(x)


class Block(nn.Module):
    """Transformer block with pre-norm attention and feed-forward layers."""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        """Create one Transformer block."""
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, n_embd, block_size, head_size, dropout)
        self.ffw = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run residual attention and residual feed-forward passes."""
        # first part: self attention
        x = x + self.sa(self.ln1(x))
        # second part: feed forward
        x = x + self.ffw(self.ln2(x))

        return x
