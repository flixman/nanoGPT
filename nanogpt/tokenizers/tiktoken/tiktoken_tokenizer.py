import json
from pathlib import Path
from typing import Any

try:
    import tiktoken
except ImportError as exc:  # pragma: no cover - runtime dependency
    tiktoken: Any = None
    _TIKTOKEN_IMPORT_ERROR = exc
else:
    tiktoken: Any = tiktoken
    _TIKTOKEN_IMPORT_ERROR = None

from ..base import Tokenizer


class TiktokenTokenizer(Tokenizer):
    """Tokenizer backed by the `tiktoken` library."""

    @classmethod
    def cli_options(cls) -> dict[str, dict[str, object]]:
        """Expose tokenizer-specific CLI options for training."""
        return {
            "encoding_name": {
                "default": "cl100k_base",
                "help": "tiktoken encoding name",
            },
        }

    def __init__(self, encoding_name: str = "cl100k_base", dataset_path: Path | None = None, **_ignored) -> None:
        """Initialize a tiktoken encoding by name."""
        if tiktoken is None:
            raise ImportError(
                "The 'tiktoken' package is required for TiktokenTokenizer. "
                "Install it with `pip install tiktoken`."
            ) from _TIKTOKEN_IMPORT_ERROR

        self.encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    @property
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""
        return self._encoding.n_vocab

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""
        return self._encoding.encode(text, allowed_special="all")

    def decode(self, tokens: list[int]) -> str:
        """Decode token ids back into text."""
        return self._encoding.decode(tokens)

    def save(self, path: Path) -> None:
        """Save the tokenizer configuration to a JSON file."""
        payload = {
            "tokenizer_type": "tiktoken",
            "encoding_name": self.encoding_name,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TiktokenTokenizer":
        """Load tokenizer configuration from a JSON file."""
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            encoding_name = payload.get("encoding_name", "cl100k_base")
        else:
            encoding_name = "cl100k_base"

        return cls(encoding_name=encoding_name)
