import pytest

from ssdataagent.config import load_llm_config


@pytest.mark.live_llm
def test_openai_api_reachable():
    """Verify the configured OpenAI-compatible endpoint accepts a chat call.

    The configured deepseek-v4-flash is a reasoning model that emits
    reasoning_content separately from content. We only assert the API responds
    with a structured ChatCompletion (proving auth, base_url, and model name
    are all valid); content emission is verified at higher max_tokens in
    integration tests.
    """
    cfg = load_llm_config()
    assert cfg.provider == "openai", "Phase-0 connectivity test assumes openai-compatible"

    from openai import OpenAI

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
        temperature=0.0,
        max_tokens=512,  # generous to allow reasoning + answer
    )
    assert resp.choices, "no choices in response"
    assert resp.usage and resp.usage.total_tokens > 0, "no usage in response"
    msg = resp.choices[0].message
    text = (msg.content or "") + (getattr(msg, "reasoning_content", "") or "")
    assert text, f"both content and reasoning_content empty: {resp}"
