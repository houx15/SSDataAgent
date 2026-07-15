"""OpenRouter delivers upstream-provider failures as HTTP 200 with `choices:
null` and an `error` object, not a non-2xx status. `resp.choices[0]` then raises
a bare 'NoneType object is not subscriptable' that hides the cause — which is
exactly how llama-4-scout's first EXP-007 run died. `_first_message` must turn
that into a diagnosable error carrying the provider's own message.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ssdataagent.agent.llm_client import OpenAICompatibleClient


def test_first_message_surfaces_provider_error_when_choices_is_null():
    resp = SimpleNamespace(choices=None,
                           error={"message": "llama-4-scout: upstream timeout"})
    with pytest.raises(RuntimeError, match="upstream timeout"):
        OpenAICompatibleClient._first_message(resp)


def test_first_message_handles_empty_choices_list():
    resp = SimpleNamespace(choices=[], error=None)
    with pytest.raises(RuntimeError, match="no choices"):
        OpenAICompatibleClient._first_message(resp)


def test_first_message_returns_the_message_on_a_normal_response():
    msg = SimpleNamespace(content="hi", tool_calls=None)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    assert OpenAICompatibleClient._first_message(resp) is msg
