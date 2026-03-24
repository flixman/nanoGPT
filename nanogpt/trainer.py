import logging

import torch
from torch.utils.data import DataLoader

from .gpt_language_model import GPTLanguageModel


logger = logging.getLogger('trainer')


class CharDataset(torch.utils.data.Dataset):
    """Dataset of fixed-length token windows for next-token prediction."""

    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        """Store token sequence and window length."""
        self.data = data          # tensor of token ids
        self.block_size = block_size

    def __len__(self) -> int:
        """Return number of valid training windows."""
        return len(self.data) - self.block_size

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one input-target pair from a windowed token slice."""
        x = self.data[i:i+self.block_size]
        y = self.data[i+1:i+self.block_size+1]
        return x, y


class Trainer:
    """Training helper for preparing datasets and optimizing the model."""

    def __init__(self, model: GPTLanguageModel, tokens: list[int], ratio: float, batch_size: int, block_size: int) -> None:
        """Move model to its configured device."""
        self.model = model.to(model.device)
        self.batch_size = batch_size
        self.block_size = block_size

        """Split tokens into train/validation sets and build dataloaders."""
        data = torch.tensor(tokens, dtype=torch.long)
        n = int(ratio * len(data))
        train_data = data[:n]
        val_data = data[n:]

        self.train_ds = CharDataset(train_data, block_size)
        self.val_ds = CharDataset(val_data, block_size)
        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=batch_size, pin_memory=True)

    def _log_memory_requirements(self) -> None:
        """Log parameter count and estimated memory requirements."""
        n_params = sum(p.numel() for p in self.model.parameters())

        config = getattr(self.model, "config", {})
        n_embd = int(config.get("n_embd", 0))
        n_layer = int(config.get("n_layer", 0))
        vocab_size = int(config.get("vocab_size", 0))
        precision_bits = getattr(self.model, "precision_bits", 32)

        dtype_bytes = precision_bits / 8
        weight_bytes = n_params * dtype_bytes
        grad_bytes = n_params * dtype_bytes
        adam_state_bytes = n_params * 2 * 4

        tokens_per_batch = self.batch_size * self.block_size
        hidden_activation_bytes = tokens_per_batch * n_embd * dtype_bytes * (n_layer * 6 + 2)
        logits_bytes = tokens_per_batch * vocab_size * dtype_bytes

        weight_memory_mib = weight_bytes / (1024 ** 2)
        startup_memory_mib = (weight_bytes + grad_bytes + hidden_activation_bytes + logits_bytes) / (1024 ** 2)
        first_step_memory_mib = (weight_bytes + grad_bytes + hidden_activation_bytes + logits_bytes + adam_state_bytes) / (1024 ** 2)

        logger.info(f"Model created with {n_params / 1e6:.2f}M parameters.")
        logger.info(f"Weight memory: ~{weight_memory_mib:.2f} MiB at {precision_bits}-bit precision.")
        logger.info(f"Estimated minimum memory to start training: ~{startup_memory_mib:.2f} MiB.")
        logger.info(f"Estimated memory including AdamW state: ~{first_step_memory_mib:.2f} MiB.")

    def train(self, max_iters: int, eval_interval: int, learning_rate: float, eval_iters: int) -> GPTLanguageModel:
        """Train the model and periodically report train/validation loss."""
        self._log_memory_requirements()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        @torch.no_grad()
        def estimate_loss() -> dict[str, torch.Tensor]:
            """Estimate mean loss for train and validation splits."""
            out: dict[str, torch.Tensor] = {}
            self.model.eval()

            for split, loader in [('train', self.train_loader), ('val', self.val_loader)]:
                losses = torch.zeros(eval_iters)
                it = iter(loader)

                for k in range(eval_iters):
                    try:
                        X, Y = next(it)
                    except StopIteration:
                        it = iter(loader)
                        X, Y = next(it)

                    X, Y = X.to(self.model.device), Y.to(self.model.device)
                    _, loss = self.model(X, Y)
                    losses[k] = loss.item()

                out[split] = losses.mean()

            self.model.train()
            return out

        train_iter = iter(self.train_loader)
        for iter_num in range(max_iters):
            if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
                losses = estimate_loss()
                logger.info(f"Iteration {iter_num}: train {losses['train']:.4f}, val {losses['val']:.4f}")

            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                xb, yb = next(train_iter)

            xb, yb = xb.to(self.model.device, non_blocking=True), yb.to(self.model.device, non_blocking=True)

            _, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        return self.model
