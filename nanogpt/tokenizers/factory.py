import json
import importlib
import inspect
from functools import lru_cache
from pathlib import Path
import pkgutil

from .base import Tokenizer


class TokenizerFactory:
    """Create tokenizer instances by name."""

    @staticmethod
    @lru_cache(maxsize=1)
    def _registry() -> dict[str, type[Tokenizer]]:
        """Discover tokenizer implementations from tokenizer subpackages."""
        registry: dict[str, type[Tokenizer]] = {}
        package_dir = Path(__file__).resolve().parent

        for module_info in pkgutil.iter_modules([str(package_dir)]):
            if not module_info.ispkg or module_info.name in {"base", "factory"}:
                continue

            try:
                module = importlib.import_module(f"{__package__}.{module_info.name}")
            except ImportError:
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if obj is Tokenizer:
                    continue
                if not issubclass(obj, Tokenizer):
                    continue
                if not obj.__name__.lower().endswith("tokenizer"):
                    continue

                registry[module_info.name.lower()] = obj
                break

        return registry

    @classmethod
    def available_tokenizers(cls) -> tuple[str, ...]:
        """Return tokenizer names that can be instantiated in this environment."""
        return tuple(sorted(cls._registry()))

    @classmethod
    def get(cls, tokenizer_name: str) -> type[Tokenizer]:
        """Return the tokenizer class for a registered tokenizer name."""
        tokenizer_key = tokenizer_name.lower()
        registry = cls._registry()
        if tokenizer_key not in registry:
            available = ", ".join(sorted(registry))
            raise ValueError(f"Unknown tokenizer '{tokenizer_name}'. Available tokenizers: {available}")
        return registry[tokenizer_key]

    @classmethod
    def create(
        cls,
        tokenizer_name: str,
        **kwargs,
    ) -> Tokenizer:
        """Instantiate a tokenizer from constructor arguments."""
        return cls.get(tokenizer_name)(**kwargs)

    @classmethod
    def load(cls, path: Path) -> Tokenizer:
        """Load a tokenizer from disk by inspecting the tokenizer file."""
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        tokenizer_key = str(payload["tokenizer_type"]).lower()

        tokenizer_cls = cls._registry().get(tokenizer_key)
        if tokenizer_cls is None:
            available = ", ".join(sorted(cls._registry()))
            raise ValueError(f"Tokenizer '{tokenizer_key}' is not available. Available tokenizers: {available}")

        return tokenizer_cls.load(path)
