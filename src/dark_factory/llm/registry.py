"""Lazy provider registry.

No provider SDK is imported until it is actually selected by config, so
`import dark_factory` never touches any of them and a downstream repo using
only `openai_compatible`/`mock` needs no extra package.
"""

from __future__ import annotations

import difflib
import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from dark_factory.llm.errors import ProviderNotInstalledError, UnknownProviderError

if TYPE_CHECKING:
    from dark_factory.config.model import ResolvedLLM
    from dark_factory.llm.client import LLMClient

# provider name -> "module.path:ClassName", imported on first use only.
_BUILTIN: dict[str, str] = {
    "mock": "dark_factory.llm.providers.mock:MockClient",
    "anthropic": "dark_factory.llm.providers.anthropic:AnthropicClient",
    "openai": "dark_factory.llm.providers.openai:OpenAIClient",
    "openai_compatible": "dark_factory.llm.providers.openai_compatible:OpenAICompatibleClient",
}

_REGISTERED: dict[str, Callable[[ResolvedLLM], LLMClient]] = {}


def register(name: str, factory: Callable[[ResolvedLLM], LLMClient]) -> None:
    """Register a provider factory in-process (tests, private adapters)."""
    _REGISTERED[name] = factory


def known_providers() -> list[str]:
    """List every registered provider name, built-in and in-process."""
    return sorted({*_BUILTIN, *_REGISTERED})


def resolve(name: str) -> Callable[[ResolvedLLM], LLMClient]:
    """Look up the client factory for a provider, importing it lazily.

    Args:
        name: The configured provider name, e.g. "anthropic".

    Returns:
        A callable that builds an `LLMClient` from a `ResolvedLLM` spec.

    Raises:
        UnknownProviderError: If `name` matches no registered provider.
        ProviderNotInstalledError: If the provider's module fails to import.
    """
    if name in _REGISTERED:
        return _REGISTERED[name]

    dotted = _BUILTIN.get(name)
    if dotted is None:
        suggestion = difflib.get_close_matches(name, known_providers(), n=1)
        raise UnknownProviderError(name, known_providers(), suggestion[0] if suggestion else None)

    module_path, _, class_name = dotted.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        # dark_factory's own built-in adapters use stdlib HTTP and have no
        # extra dependency, so this only fires for third-party/private
        # adapters registered under a builtin-looking dotted path.
        raise ProviderNotInstalledError(name, str(exc)) from exc
    return cast("Callable[[ResolvedLLM], LLMClient]", getattr(module, class_name))
