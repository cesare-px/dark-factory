"""Provider-agnostic LLM client protocol, registry, and metering."""

from dark_factory.llm.client import LLMClient, MeteredLLMClient, build_client
from dark_factory.llm.errors import (
    LLMAuthError,
    LLMBadRequestError,
    LLMContextLengthError,
    LLMError,
    LLMRateLimitError,
    LLMTransientError,
    ProviderNotInstalledError,
    UnknownProviderError,
)
from dark_factory.llm.registry import known_providers, register, resolve
from dark_factory.llm.types import LLMRequest, LLMResponse, Message, TokenUsage

__all__ = [
    "LLMAuthError",
    "LLMBadRequestError",
    "LLMClient",
    "LLMContextLengthError",
    "LLMError",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMTransientError",
    "Message",
    "MeteredLLMClient",
    "ProviderNotInstalledError",
    "TokenUsage",
    "UnknownProviderError",
    "build_client",
    "known_providers",
    "register",
    "resolve",
]
