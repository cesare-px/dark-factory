"""Fixture-driven tests of each adapter's pure request/response mapping.

These need zero network access and no provider SDK: build_payload/parse_usage
/parse_response are plain functions over dicts, exercised here against
recorded response shapes.
"""

from dark_factory.config.model import ResolvedLLM
from dark_factory.llm.providers import anthropic, openai
from dark_factory.llm.types import (
    LLMRequest,
    Message,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)


def _spec(provider: str, model: str) -> ResolvedLLM:
    return ResolvedLLM(
        provider=provider,
        model=model,
        preset=None,
        api_key_env="X",
        base_url=None,
        max_output_tokens=1024,
        temperature=0.0,
        timeout_seconds=30,
        max_attempts=1,
        extra={},
    )


def test_anthropic_build_payload_includes_system_and_messages():
    spec = _spec("anthropic", "claude-sonnet-5")
    request = LLMRequest(messages=(Message("user", "hi"),), system="be terse")
    payload = anthropic.build_payload(request, spec)
    assert payload["model"] == "claude-sonnet-5"
    assert payload["system"] == "be terse"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_parse_response_extracts_text_and_usage():
    spec = _spec("anthropic", "claude-sonnet-5")
    raw = {
        "content": [{"type": "text", "text": "hello"}],
        "usage": {
            "input_tokens": 5,
            "output_tokens": 3,
            "cache_read_input_tokens": 1,
            "cache_creation_input_tokens": 2,
        },
        "model": "claude-sonnet-5-20260101",
        "stop_reason": "end_turn",
    }
    response = anthropic.parse_response(raw, spec)
    assert response.text == "hello"
    assert response.usage.input_tokens == 5
    assert response.usage.output_tokens == 3
    assert response.usage.cached_input_tokens == 1
    assert response.usage.cache_write_tokens == 2
    assert response.model == "claude-sonnet-5-20260101"
    assert response.finish_reason == "end_turn"


def test_anthropic_parse_response_concatenates_multiple_text_blocks():
    spec = _spec("anthropic", "claude-sonnet-5")
    raw = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "usage": {}}
    response = anthropic.parse_response(raw, spec)
    assert response.text == "ab"


def test_anthropic_build_payload_includes_tool_definitions():
    spec = _spec("anthropic", "claude-sonnet-5")
    tool = ToolDefinition(
        name="read_file", description="Read a file", input_schema={"type": "object"}
    )
    request = LLMRequest(messages=(Message("user", "hi"),), tools=(tool,))
    payload = anthropic.build_payload(request, spec)
    assert payload["tools"] == [
        {"name": "read_file", "description": "Read a file", "input_schema": {"type": "object"}}
    ]


def test_anthropic_build_payload_translates_content_blocks():
    spec = _spec("anthropic", "claude-sonnet-5")
    request = LLMRequest(
        messages=(
            Message(
                "assistant",
                (
                    TextBlock("checking"),
                    ToolCall(id="t1", name="read_file", input={"path": "a.py"}),
                ),
            ),
            Message("user", (ToolResultBlock(tool_call_id="t1", content="contents of a.py"),)),
        )
    )
    payload = anthropic.build_payload(request, spec)
    assert payload["messages"][0]["content"] == [
        {"type": "text", "text": "checking"},
        {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}},
    ]
    assert payload["messages"][1]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "contents of a.py"}
    ]


def test_anthropic_build_payload_marks_error_tool_results():
    spec = _spec("anthropic", "claude-sonnet-5")
    request = LLMRequest(
        messages=(
            Message(
                "user", (ToolResultBlock(tool_call_id="t1", content="no such file", is_error=True),)
            ),
        )
    )
    payload = anthropic.build_payload(request, spec)
    assert payload["messages"][0]["content"][0]["is_error"] is True


def test_anthropic_parse_response_extracts_tool_calls():
    spec = _spec("anthropic", "claude-sonnet-5")
    raw = {
        "content": [
            {"type": "text", "text": "let me check"},
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}},
        ],
        "usage": {},
        "stop_reason": "tool_use",
    }
    response = anthropic.parse_response(raw, spec)
    assert response.text == "let me check"
    assert response.finish_reason == "tool_use"
    assert response.tool_calls == (ToolCall(id="t1", name="read_file", input={"path": "a.py"}),)


def test_openai_build_payload_prepends_system_message():
    spec = _spec("openai", "gpt-5")
    request = LLMRequest(messages=(Message("user", "hi"),), system="be terse")
    payload = openai.build_payload(request, spec)
    assert payload["messages"][0] == {"role": "system", "content": "be terse"}
    assert payload["messages"][1] == {"role": "user", "content": "hi"}


def test_openai_parse_response_reads_nested_usage_details():
    spec = _spec("openai", "gpt-5")
    raw = {
        "choices": [{"message": {"content": "hi there"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
        "model": "gpt-5",
    }
    response = openai.parse_response(raw, spec, provider="openai")
    assert response.text == "hi there"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 4
    assert response.usage.cached_input_tokens == 2
    assert response.usage.reasoning_tokens == 1


def test_openai_parse_response_tolerates_missing_usage_block():
    spec = _spec("openai", "gpt-5")
    raw = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}
    response = openai.parse_response(raw, spec, provider="openai_compatible")
    assert response.usage.is_estimated is True
    assert response.usage.input_tokens == 0
