import json
from pathlib import Path
from typing import Any

try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import decoders, models, pre_tokenizers, trainers
except ImportError as exc:  # pragma: no cover - runtime dependency
    HFTokenizer: Any = None
    decoders: Any = None
    models: Any = None
    pre_tokenizers: Any = None
    trainers: Any = None
    _TOKENIZERS_IMPORT_ERROR = exc
else:
    HFTokenizer = HFTokenizer
    _TOKENIZERS_IMPORT_ERROR = None

from ..base import Tokenizer


class BpeTokenizer(Tokenizer):
    """Corpus-trained byte-level BPE tokenizer."""

    @classmethod
    def cli_options(cls) -> dict[str, dict[str, object]]:
        """Expose tokenizer-specific CLI options for training."""
        return {
            "vocab_size": {
                "type": int,
                "default": 8000,
                "help": "Target tokenizer vocab size",
            },
            "min_frequency": {
                "type": int,
                "default": 2,
                "help": "Minimum frequency for BPE merges",
            },
            "special_tokens": {
                "nargs": "*",
                "default": "",
                "help": "Special tokens to reserve in the tokenizer",
            },
        }

    def __init__(
        self,
        training_set_path: Path | None = None,
        *,
        dataset_path: Path | None = None,
        tokenizer_json: str | None = None,
        vocab_size: int = 8000,
        min_frequency: int = 2,
        special_tokens: list[str] | None = None,
        **_ignored,
    ) -> None:
        """Initialize from corpus or from a saved tokenizer JSON string."""
        if HFTokenizer is None:
            raise ImportError(
                "The 'tokenizers' package is required for BpeTokenizer. "
                "Install it with `pip install tokenizers`."
            ) from _TOKENIZERS_IMPORT_ERROR

        if dataset_path is not None and training_set_path is None:
            training_set_path = dataset_path

        self.requested_vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens

        if tokenizer_json is not None:
            self._tokenizer = HFTokenizer.from_str(tokenizer_json)
            return

        if training_set_path is None:
            raise ValueError("Provide either training_set_path or tokenizer_json.")

        self._tokenizer = HFTokenizer(models.BPE(unk_token="[UNK]"))
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self._tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.special_tokens,
            show_progress=False,
        )
        self._tokenizer.train([str(training_set_path)], trainer)

    @property
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""
        return self._tokenizer.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""
        return self._tokenizer.encode(text).ids

    def decode(self, tokens: list[int]) -> str:
        """Decode token ids back into text."""
        return self._tokenizer.decode(tokens)

    def save(self, path: Path) -> None:
        """Save the tokenizer configuration to a JSON file."""
        payload = {
            "tokenizer_type": "bpe",
            "requested_vocab_size": self.requested_vocab_size,
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
            "tokenizer_json": self._tokenizer.to_str(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BpeTokenizer":
        """Load tokenizer configuration from a JSON file."""
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        return cls(
            tokenizer_json=payload["tokenizer_json"],
            vocab_size=payload.get("requested_vocab_size", payload["vocab_size"]),
            min_frequency=payload["min_frequency"],
            special_tokens=payload.get("special_tokens"),
        )
