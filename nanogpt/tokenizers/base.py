from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self


class Tokenizer(ABC):
    """Common interface for text tokenizers used by training and generation."""

    @classmethod
    def cli_options(cls) -> dict[str, dict[str, Any]]:
        """Return tokenizer-specific argparse metadata keyed by option name."""
        return {}

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the number of token ids supported by this tokenizer."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """Decode token ids back into text."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist tokenizer state to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Self:
        """Restore tokenizer state from disk."""
