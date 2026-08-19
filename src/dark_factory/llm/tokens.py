"""Client-side token estimation for preflight budgeting.

Only ever used as a pessimistic upper bound before a real call (when a
provider offers no count-tokens endpoint) or to estimate usage after a call
whose response omitted a usage block. Never trust this over provider-reported
usage.
"""

from __future__ import annotations

from dark_factory.llm.types import LLMRequest, TokenUsage

# Code and structured text tokenize worse than prose; err pessimistic so a
# preflight check never under-reserves budget.
DEFAULT_CHARS_PER_TOKEN = 3.4


def estimate_tokens(text: str, *, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Estimate a pessimistic token count for `text`."""
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


def estimate_request_tokens(
    request: LLMRequest, *, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN
) -> int:
    """Estimate a pessimistic input-token count for an entire `LLMRequest`."""
    total_chars = len(request.system or "")
    total_chars += sum(len(m.content) for m in request.messages)
    return estimate_tokens("x" * total_chars, chars_per_token=chars_per_token)


def estimate_usage_from_text(prompt_text: str, completion_text: str) -> TokenUsage:
    """Build an estimated `TokenUsage` from raw prompt/completion text."""
    return TokenUsage(
        input_tokens=estimate_tokens(prompt_text),
        output_tokens=estimate_tokens(completion_text),
        is_estimated=True,
    )
