from pathlib import Path
from typing import Generator, Self
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from .block import Block


logger = logging.getLogger('gpt')


class GPTLanguageModel(nn.Module):
    """Character-level GPT language model."""

    def __init__(self, vocab_size: int, n_embd: int, block_size: int, n_layer: int, n_head: int, dropout: float, precision_bits: int = 32) -> None:
        """Initialize token/position embeddings and Transformer stack."""
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "n_embd": n_embd,
            "block_size": block_size,
            "n_layer": n_layer,
            "n_head": n_head,
            "dropout": dropout,
            "precision_bits": precision_bits,
        }
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info("Using device %s", self.device)
        self.precision_bits = precision_bits
        self.precision_dtype = self._precision_dtype(precision_bits)
        self.to(self.device, dtype=self.precision_dtype)

    @staticmethod
    def _precision_dtype(precision_bits: int) -> torch.dtype:
        """Map a bit-width choice to a PyTorch floating-point dtype."""
        if precision_bits == 32:
            return torch.float32
        if precision_bits == 16:
            return torch.bfloat16
        raise ValueError(f"Unsupported precision_bits={precision_bits}. Expected 32 or 16.")
    
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute logits and optional cross-entropy loss for input tokens."""
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits.float(), targets)
        
        return logits, loss

    def generate(self, top_k: int, max_new_tokens: int, start_token: int, temperature: float) -> Generator[torch.Tensor]:
        """Autoregressively generate max_new_tokens from a token context."""
        context = torch.tensor([[start_token]], dtype=torch.long, device=self.device)
        block_size = self.config['block_size']

        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # get a tensor with the size of the vocabulary, in which each component is the
                # contains what model thinks is the likelyhood of being the next token given the current token
                logits, _ = self(context)
                logits = logits[:, -1, :]
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                # Apply softmax to obtain probabilities from the scores
                probs = F.softmax(logits.float(), dim=-1)

                # given those probabilities, select a sample (tensor index). That will be decoded into a token.
                idx_next = torch.multinomial(probs, num_samples=1)
                yield idx_next

                # update the context with the token previously generated and go on.
                context = torch.cat((context, idx_next), dim=1)[:, -block_size:]
    
    def save_model(self, state_path: Path) -> None:
        """Persist model weights and architecture metadata to disk."""
        checkpoint = {
            "model_config": self.config,
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, state_path)
    
    @classmethod
    def load_model(cls, state_path: Path, precision_bits: int | None = None) -> Self:
        """Load model checkpoint and place model on its configured device."""
        checkpoint = torch.load(state_path, map_location="cpu")

        try:
            config = checkpoint["model_config"]
            state_dict = checkpoint["state_dict"]
        except Exception:
            raise ValueError("Invalid checkpoint format. Re-train or re-save the model with the current save_model implementation.")

        if precision_bits is not None:
            config = {**config, "precision_bits": precision_bits}

        model = cls(**config)
        model.load_state_dict(state_dict)
        return model.to(model.device, dtype=model.precision_dtype)
