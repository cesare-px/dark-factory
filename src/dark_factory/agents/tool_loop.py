"""Shared model-driven tool-use loop for the Planner and Developer agents.

Given an LLM client, a tool registry, and a starting message, repeatedly
calls the model, executes whatever tools it asks for, and feeds the results
back -- until it calls its designated "finish" tool, or `max_iterations` is
exhausted. Every round trip is one `MeteredLLMClient.complete()` call, so
the existing per-call budget preflight check bounds total spend exactly as
it already does for a single-shot call; no new budget mechanism is needed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dark_factory.llm.client import MeteredLLMClient
from dark_factory.llm.types import (
    LLMRequest,
    Message,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
)


class ToolLoopLimitExceededError(RuntimeError):
    """The tool-use loop ended without the model calling its finish tool."""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One callable tool.

    Its LLM-facing definition plus the Python handler that runs it, called
    with the model's `tool_call.input` as kwargs.
    """

    definition: ToolDefinition
    handler: Callable[..., str]


@dataclass(frozen=True, slots=True)
class ToolLoopResult:
    """The outcome of a finished tool-use loop."""

    finish_input: dict[str, Any]
    iterations: int


def run_tool_loop(
    llm: MeteredLLMClient,
    *,
    system: str,
    first_message: str,
    tools: tuple[ToolSpec, ...],
    finish_tool: ToolDefinition,
    max_iterations: int,
    max_output_tokens: int = 4096,
    temperature: float | None = 0.0,
    metadata: dict[str, str] | None = None,
) -> ToolLoopResult:
    """Drive one tool-use conversation to completion.

    Raises `ToolLoopLimitExceededError` if the model exhausts
    `max_iterations` without calling `finish_tool`, or if it ever stops
    without calling any tool at all -- a plain-text stop is treated as a
    dead end, not a silent success, so the caller's own retry loop (e.g.
    developer's build-test-fix loop) decides what happens next rather than
    this driver guessing.
    """
    tool_defs = (*(t.definition for t in tools), finish_tool)
    handlers = {t.definition.name: t.handler for t in tools}
    messages: list[Message] = [Message("user", first_message)]

    for iteration in range(1, max_iterations + 1):
        response = llm.complete(
            LLMRequest(
                system=system,
                messages=tuple(messages),
                tools=tool_defs,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                metadata=metadata or {},
            )
        )

        assistant_blocks: list[Any] = []
        if response.text:
            assistant_blocks.append(TextBlock(response.text))
        assistant_blocks.extend(response.tool_calls)
        if assistant_blocks:
            messages.append(Message("assistant", tuple(assistant_blocks)))

        finish_call = next((tc for tc in response.tool_calls if tc.name == finish_tool.name), None)
        if finish_call is not None:
            return ToolLoopResult(finish_input=dict(finish_call.input), iterations=iteration)

        if not response.tool_calls:
            raise ToolLoopLimitExceededError("model stopped without calling the finish tool")

        results = []
        for call in response.tool_calls:
            handler = handlers.get(call.name)
            if handler is None:
                results.append(
                    ToolResultBlock(call.id, f"unknown tool: {call.name}", is_error=True)
                )
                continue
            try:
                output = handler(**call.input)
                results.append(ToolResultBlock(call.id, output))
            except Exception as exc:
                results.append(ToolResultBlock(call.id, str(exc), is_error=True))
        messages.append(Message("user", tuple(results)))

    raise ToolLoopLimitExceededError(f"exceeded {max_iterations} tool-use iterations")
