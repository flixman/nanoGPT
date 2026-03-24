import json
from pathlib import Path

from ..base import Tokenizer
from .vocabulary import Vocabulary


class CharTokenizer(Tokenizer):
    """Tokenizer that maps individual characters to token ids."""

    @classmethod
    def cli_options(cls) -> dict[str, dict[str, object]]:
        """Character tokenization does not require extra CLI options."""
        return {}

    def __init__(
        self,
        training_set_path: Path | None = None,
        *,
        dataset_path: Path | None = None,
        tokens: list[str] | None = None,
        text: str = "",
        **_ignored,
    ) -> None:
        """Initialize tokenizer from a training text file or a token list."""
        if dataset_path is not None and training_set_path is None:
            training_set_path = dataset_path

        if training_set_path is not None:
            with open(training_set_path, encoding="utf-8") as f:
                self.text = f.read()
            tokens = sorted(set(self.text))
        elif tokens is not None:
            self.text = text
        else:
            raise ValueError("Provide either training_set_path or tokens.")

        self._vocabulary = Vocabulary(tokens)

    @property
    def input(self) -> str:
        """Return the full training text."""
        return self.text

    @property
    def tokens(self) -> list[str]:
        """Return sorted unique characters discovered in the input text."""
        return self._vocabulary.dump()

    @property
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""
        return len(self._vocabulary)

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""
        return self._vocabulary.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode token ids back into text."""
        return self._vocabulary.decode(tokens)

    def save(self, path: Path) -> None:
        """Save tokenizer tokens to a JSON file."""
        payload = {
            "tokenizer_type": "char",
            "tokens": self.tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        """Load tokenizer tokens from a JSON file."""
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            tokens = payload.get("tokens", [])
        else:
            tokens = payload
        return cls(tokens=tokens)