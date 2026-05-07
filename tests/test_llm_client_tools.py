"""Tests for the tool-using path on LLMClient. The OpenAI SDK is mocked;
these only verify our request/response wiring, not the upstream API."""
from unittest.mock import MagicMock, patch

import pytest

from ssdataagent.agent.llm_client import (
    OpenAICompatibleClient,
    ToolCallRequest,
    ToolUsingResponse,
    _parse_openai_tool_message,
)
from ssdataagent.config import LLMConfig


def _cfg() -> LLMConfig:
    return LLMConfig(
        provider="openai", base_url="https://api.openai.com/v1",
        api_key="k", model="gpt-5.4-2026-03-05", temperature=0.7, max_tokens=512,
    )


# A fake OpenAI tool_call object — uses attribute access, matches SDK shape.
def _fake_tool_call(call_id: str, name: str, args_json: str):
    fn = MagicMock(name=name, arguments=args_json)
    fn.name = name           # MagicMock.name is special-cased; force it
    fn.arguments = args_json
    tc = MagicMock(id=call_id, type="function", function=fn)
    return tc


def test_parse_openai_tool_message_text_only():
    msg = MagicMock(content="just an answer", tool_calls=None)
    out = _parse_openai_tool_message(msg)
    assert out.content == "just an answer"
    assert out.tool_calls == []
    assert out.assistant_message == {"role": "assistant", "content": "just an answer"}


def test_parse_openai_tool_message_single_tool_call():
    tc = _fake_tool_call("call_1", "describe_column", '{"col": "age"}')
    msg = MagicMock(content="", tool_calls=[tc])
    out = _parse_openai_tool_message(msg)
    assert out.content == ""
    assert out.tool_calls == [
        ToolCallRequest(id="call_1", name="describe_column", arguments={"col": "age"})
    ]
    assert out.assistant_message["role"] == "assistant"
    assert out.assistant_message["tool_calls"][0]["id"] == "call_1"
    assert out.assistant_message["tool_calls"][0]["function"]["name"] == "describe_column"


def test_parse_openai_tool_message_multiple_tool_calls():
    tcs = [
        _fake_tool_call("a", "list_columns", "{}"),
        _fake_tool_call("b", "describe_column", '{"col": "income"}'),
    ]
    msg = MagicMock(content="thinking", tool_calls=tcs)
    out = _parse_openai_tool_message(msg)
    assert [t.name for t in out.tool_calls] == ["list_columns", "describe_column"]
    assert out.tool_calls[0].arguments == {}
    assert out.tool_calls[1].arguments == {"col": "income"}


def test_parse_openai_tool_message_handles_dict_input():
    """Tool messages are sometimes serialized to dicts (transcripts, replays)."""
    msg = {
        "content": "",
        "tool_calls": [
            {"id": "x", "type": "function",
             "function": {"name": "fit_marginal", "arguments": '{"col":"age","family":"kde"}'}},
        ],
    }
    out = _parse_openai_tool_message(msg)
    assert out.tool_calls[0].name == "fit_marginal"
    assert out.tool_calls[0].arguments == {"col": "age", "family": "kde"}


def test_parse_openai_tool_message_records_arg_parse_failure():
    """Malformed JSON arguments should be recorded, not raise — the
    orchestrator nudges the agent via the tool result."""
    tc = _fake_tool_call("c", "fit_conditional", "not valid json {")
    msg = MagicMock(content="", tool_calls=[tc])
    out = _parse_openai_tool_message(msg)
    assert out.tool_calls[0].arguments["_parse_error"] == "JSONDecodeError"
    assert out.tool_calls[0].arguments["_raw_arguments"] == "not valid json {"


def test_chat_with_tools_passes_tools_through():
    """End-to-end through the OpenAI client: tools list + tool_choice get
    forwarded; response is parsed into ToolUsingResponse."""
    cfg = _cfg()
    tc = _fake_tool_call("call_99", "list_columns", "{}")
    fake_msg = MagicMock(content="", tool_calls=[tc])
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=fake_msg)]

    tools = [{
        "type": "function",
        "function": {
            "name": "list_columns",
            "description": "Return column metadata.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }]

    with patch("ssdataagent.agent.llm_client.OpenAI") as Sdk:
        Sdk.return_value.chat.completions.create.return_value = fake_resp
        client = OpenAICompatibleClient(cfg)
        out = client.chat_with_tools(
            [{"role": "user", "content": "what's in the data"}],
            tools=tools,
            system="you are an analyst",
        )
        kwargs = Sdk.return_value.chat.completions.create.call_args.kwargs

    assert isinstance(out, ToolUsingResponse)
    assert kwargs["tools"] is tools
    assert kwargs["tool_choice"] == "auto"
    # System prepended just like the non-tool path.
    assert kwargs["messages"][0] == {"role": "system", "content": "you are an analyst"}
    # Tool call surfaced.
    assert out.tool_calls[0].name == "list_columns"


def test_chat_with_tools_round_trip_history_compatibility():
    """The assistant_message returned must be appendable to the next
    request's `messages` list verbatim — that's the OpenAI tool-calling
    contract."""
    cfg = _cfg()
    tc = _fake_tool_call("xyz", "describe_column", '{"col":"age"}')
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content="", tool_calls=[tc]))]
    with patch("ssdataagent.agent.llm_client.OpenAI") as Sdk:
        Sdk.return_value.chat.completions.create.return_value = fake_resp
        client = OpenAICompatibleClient(cfg)
        out = client.chat_with_tools([{"role": "user", "content": "go"}], tools=[])

    next_history = [
        {"role": "user", "content": "go"},
        out.assistant_message,
        {"role": "tool", "tool_call_id": "xyz", "content": '{"col":"age","mean":42}'},
    ]
    # Every entry has a role; tool message references the call id we got.
    assert all("role" in m for m in next_history)
    assert next_history[2]["tool_call_id"] == out.tool_calls[0].id


def test_anthropic_chat_with_tools_raises_until_implemented():
    from ssdataagent.agent.llm_client import AnthropicCompatibleClient
    cfg = LLMConfig(provider="anthropic", base_url="x", api_key="k", model="claude")
    with patch("ssdataagent.agent.llm_client.Anthropic"):
        client = AnthropicCompatibleClient(cfg)
    with pytest.raises(NotImplementedError, match="EXP-006"):
        client.chat_with_tools([], tools=[])
