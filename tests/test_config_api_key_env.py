"""Per-provider API keys.

Running the same agent against OpenRouter and a first-party endpoint in one batch
means two different keys. The name of the env var is configurable; the key VALUE
still only ever comes from the environment, never from a config file.

The security property that matters: an experiment pointed at a third-party
base_url must never be handed the first-party key. Silently falling back to
LLM_API_KEY when the named var is unset would ship that key to whoever owns the
other endpoint, so the miss is a hard error.
"""
from __future__ import annotations

import pytest

from ssdataagent.config import load_llm_config


@pytest.fixture()
def yaml_only(tmp_path):
    p = tmp_path / "llm.yaml"
    p.write_text("provider: openai\nbase_url: https://api.openai.com/v1\nmodel: gpt-5.4\n")
    return p


def test_defaults_to_llm_api_key(monkeypatch, yaml_only):
    monkeypatch.setenv("LLM_API_KEY", "first-party-key")
    cfg = load_llm_config(yaml_path=yaml_only, env_path=None)
    assert cfg.api_key == "first-party-key"


def test_api_key_env_override_selects_a_different_variable(monkeypatch, yaml_only):
    monkeypatch.setenv("LLM_API_KEY", "first-party-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    cfg = load_llm_config(
        yaml_path=yaml_only, env_path=None,
        overrides={"api_key_env": "OPENROUTER_API_KEY",
                   "base_url": "https://openrouter.ai/api/v1",
                   "model": "anthropic/claude-sonnet-4.5"},
    )
    assert cfg.api_key == "openrouter-key"
    assert cfg.base_url == "https://openrouter.ai/api/v1"


def test_missing_named_key_raises_instead_of_leaking_the_first_party_key(
    monkeypatch, yaml_only
):
    """The whole point. LLM_API_KEY is set and OPENROUTER_API_KEY is not — a
    fallback here would POST the first-party key to openrouter.ai."""
    monkeypatch.setenv("LLM_API_KEY", "first-party-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        load_llm_config(
            yaml_path=yaml_only, env_path=None,
            overrides={"api_key_env": "OPENROUTER_API_KEY",
                       "base_url": "https://openrouter.ai/api/v1"},
        )
