import json
from pathlib import Path


class Vocabulary:
    """Bidirectional mapping between characters and token ids."""

    def __init__(self, training_set: list[str]) -> None:
        """Build stoi/itos maps from a list of unique tokens."""
        self._stoi: dict[str, int] = {c: i for i, c in enumerate(training_set)}
        self._itos: dict[int, str] = dict(zip(self._stoi.values(), self._stoi.keys()))

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self._stoi)

    def dump(self) -> list[str]:
        """Return ordered token list for persistence."""
        return list(self._stoi.keys())

    def encode(self, input: str) -> list[int]:
        """Encode text into token ids."""
        return [self._stoi[x] for x in input]

    def decode(self, input: list[int]) -> str:
        """Decode token ids back to text."""
        return ''.join(self._itos[x] for x in input)

    def save(self, path: Path) -> None:
        """Save vocabulary tokens to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.dump(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """Load vocabulary tokens from a JSON file."""
        with open(path, encoding="utf-8") as f:
            tokens = json.load(f)
        return Vocabulary(tokens)
