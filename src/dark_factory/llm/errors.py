"""Provider-blind error taxonomy.

Every adapter maps its native error shape into these so pipeline escalation
logic (retry vs. block vs. surface-to-human) never branches on which
provider raised.
"""

from __future__ import annotations


class LLMError(RuntimeError):
    """Base class for all LLM-call failures."""


class LLMAuthError(LLMError):
    """Invalid or missing credentials."""


class LLMRateLimitError(LLMError):
    """429-shaped response; caller may retry with backoff."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        """Record the provider's suggested retry delay, if it sent one."""
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LLMTransientError(LLMError):
    """5xx / connection failure; caller may retry."""


class LLMBadRequestError(LLMError):
    """4xx other than auth/rate-limit; retrying won't help."""


class LLMContextLengthError(LLMError):
    """The request exceeded the model's context window."""


class ProviderNotInstalledError(LLMError):
    """A registered provider's module could not be imported."""

    def __init__(self, provider: str, detail: str) -> None:
        """Build the error with the underlying import failure's detail."""
        self.provider = provider
        self.detail = detail
        super().__init__(
            f"provider {provider!r} could not be loaded: {detail}\n"
            f"  (built-in providers -- mock, anthropic, openai, openai_compatible -- need no "
            f"extra package; this looks like a custom adapter registered under that name)"
        )


class UnknownProviderError(LLMError):
    """A configured provider name matches no registered provider."""

    def __init__(self, name: str, known: list[str], suggestion: str | None) -> None:
        """Build the error with the known-provider list and an optional did-you-mean."""
        self.name = name
        hint = f" did you mean {suggestion!r}?" if suggestion else ""
        super().__init__(
            f"unknown provider {name!r}.{hint} known providers: {', '.join(sorted(known))}"
        )
