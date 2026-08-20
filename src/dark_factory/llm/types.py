"""Provider-agnostic request/response/usage types.

No provider SDK types leak past this module -- every adapter translates its
native request/response shape into these dataclasses at the boundary, which
is what lets agents, the budget tracker, and tests stay provider-blind.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """One tool the model may call, described as a JSON-schema input contract."""

    name: str
    description: str
    input_schema: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One invocation of a tool, as requested by the model."""

    id: str
    name: str
    input: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    """The result of running one `ToolCall`, fed back to the model."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class TextBlock:
    """Plain text content within a message."""

    text: str


ContentBlock = TextBlock | ToolCall | ToolResultBlock


@dataclass(frozen=True, slots=True)
class Message:
    """One chat message in a provider-agnostic request.

    `content` is plain text for an ordinary turn, or a tuple of content
    blocks for a tool-use turn: the model's own `ToolCall`s from a prior
    response, or `ToolResultBlock`s reporting what running them produced.
    """

    role: Role
    content: str | tuple[ContentBlock, ...]


def message_text(content: str | tuple[ContentBlock, ...]) -> str:
    """Best-effort plain-text rendering of `Message.content`.

    For providers/estimators that don't understand tool content blocks yet
    (openai, openai_compatible, mock -- tool support is Anthropic-only for
    now). Never raises; a tool call/result renders as a short bracketed
    placeholder rather than being silently dropped.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ToolCall):
            parts.append(f"[tool_call {block.name}]")
        else:
            parts.append(block.content)
    return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts for one LLM call, actual or estimated."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    # True when the provider's response carried no usage block and these
    # numbers were estimated client-side (e.g. some openai_compatible
    # endpoints). Surfaced in run reports so cost figures can be flagged.
    is_estimated: bool = False

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Sum two usages, propagating `is_estimated` if either is estimated."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            is_estimated=self.is_estimated or other.is_estimated,
        )

    @property
    def billable_output(self) -> int:
        """Output tokens plus reasoning tokens, which providers bill together."""
        return self.output_tokens + self.reasoning_tokens


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """A provider-agnostic completion request."""

    messages: tuple[Message, ...]
    system: str | None = None
    max_output_tokens: int = 4096
    temperature: float | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    tools: tuple[ToolDefinition, ...] = ()


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A provider-agnostic completion response."""

    text: str
    usage: TokenUsage
    provider: str
    model: str
    finish_reason: str = "stop"
    tool_calls: tuple[ToolCall, ...] = ()
    raw: Any = field(default=None, repr=False, compare=False)
