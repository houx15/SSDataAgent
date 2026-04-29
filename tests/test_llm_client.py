from unittest.mock import MagicMock, patch

import pytest

from ssdataagent.agent.llm_client import (
    AnthropicCompatibleClient,
    OpenAICompatibleClient,
    build_client,
)
from ssdataagent.config import LLMConfig


def _openai_cfg() -> LLMConfig:
    return LLMConfig(
        provider="openai", base_url="https://example.com",
        api_key="k", model="m", temperature=0.5, max_tokens=128,
    )


def _anthropic_cfg() -> LLMConfig:
    return LLMConfig(
        provider="anthropic", base_url="https://anthropic.example",
        api_key="k", model="claude", temperature=0.5, max_tokens=128,
    )


def test_build_client_openai():
    with patch("ssdataagent.agent.llm_client.OpenAI"):
        assert isinstance(build_client(_openai_cfg()), OpenAICompatibleClient)


def test_build_client_anthropic():
    with patch("ssdataagent.agent.llm_client.Anthropic"):
        assert isinstance(build_client(_anthropic_cfg()), AnthropicCompatibleClient)


def test_openai_chat_returns_text():
    cfg = _openai_cfg()
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content="hi", reasoning_content=None))]
    with patch("ssdataagent.agent.llm_client.OpenAI") as Sdk:
        Sdk.return_value.chat.completions.create.return_value = fake_resp
        client = OpenAICompatibleClient(cfg)
        out = client.chat([{"role": "user", "content": "hello"}], system="be brief")
    assert out == "hi"
    Sdk.assert_called_once_with(api_key="k", base_url="https://example.com")


def test_openai_chat_falls_back_to_reasoning_content():
    """For DeepSeek-style reasoning models that return only reasoning_content."""
    cfg = _openai_cfg()
    fake_msg = MagicMock(content="", reasoning_content="thoughts here")
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=fake_msg)]
    with patch("ssdataagent.agent.llm_client.OpenAI") as Sdk:
        Sdk.return_value.chat.completions.create.return_value = fake_resp
        client = OpenAICompatibleClient(cfg)
        out = client.chat([{"role": "user", "content": "hi"}])
    assert "thoughts here" in out


def test_openai_system_prepended_when_provided():
    cfg = _openai_cfg()
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content="ok", reasoning_content=None))]
    with patch("ssdataagent.agent.llm_client.OpenAI") as Sdk:
        Sdk.return_value.chat.completions.create.return_value = fake_resp
        client = OpenAICompatibleClient(cfg)
        client.chat([{"role": "user", "content": "x"}], system="SYS")
        kwargs = Sdk.return_value.chat.completions.create.call_args.kwargs
        assert kwargs["messages"][0] == {"role": "system", "content": "SYS"}


def test_anthropic_chat_returns_text():
    cfg = _anthropic_cfg()
    fake_resp = MagicMock()
    fake_resp.content = [MagicMock(text="hello world")]
    with patch("ssdataagent.agent.llm_client.Anthropic") as Sdk:
        Sdk.return_value.messages.create.return_value = fake_resp
        client = AnthropicCompatibleClient(cfg)
        out = client.chat([{"role": "user", "content": "hi"}], system="brief")
    assert out == "hello world"


def test_openai_retries_on_transient_connection_error():
    from openai import APIConnectionError
    cfg = _openai_cfg()
    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content="ok", reasoning_content=None))]
    transient = APIConnectionError(request=MagicMock())
    with patch("ssdataagent.agent.llm_client.OpenAI") as Sdk, \
         patch("ssdataagent.agent.llm_client.time.sleep"):
        Sdk.return_value.chat.completions.create.side_effect = [transient, transient, fake_resp]
        client = OpenAICompatibleClient(cfg, max_retries=3)
        out = client.chat([{"role": "user", "content": "x"}])
    assert out == "ok"
    assert Sdk.return_value.chat.completions.create.call_count == 3


def test_openai_gives_up_after_max_retries():
    from openai import APIConnectionError
    cfg = _openai_cfg()
    transient = APIConnectionError(request=MagicMock())
    with patch("ssdataagent.agent.llm_client.OpenAI") as Sdk, \
         patch("ssdataagent.agent.llm_client.time.sleep"):
        Sdk.return_value.chat.completions.create.side_effect = transient
        client = OpenAICompatibleClient(cfg, max_retries=2)
        with pytest.raises(APIConnectionError):
            client.chat([{"role": "user", "content": "x"}])
    assert Sdk.return_value.chat.completions.create.call_count == 3  # initial + 2 retries


def test_unknown_provider_rejected():
    cfg = LLMConfig(
        provider="nope",  # type: ignore[arg-type]
        base_url="x", api_key="k", model="m",
    )
    with pytest.raises(RuntimeError, match="provider"):
        build_client(cfg)
