import pytest

from ssdataagent.config import LLMConfig, load_llm_config


def test_load_from_yaml_only(tmp_path, monkeypatch):
    yaml = tmp_path / "llm.yaml"
    yaml.write_text(
        "provider: openai\n"
        "base_url: https://api.example.com\n"
        "model: foo-1\n"
        "temperature: 0.5\n"
        "max_tokens: 1024\n"
    )
    for v in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "from-env")
    cfg = load_llm_config(yaml_path=yaml, env_path=None)
    assert isinstance(cfg, LLMConfig)
    assert cfg.provider == "openai"
    assert cfg.base_url == "https://api.example.com"
    assert cfg.model == "foo-1"
    assert cfg.temperature == 0.5
    assert cfg.max_tokens == 1024
    assert cfg.api_key == "from-env"


def test_env_overrides_yaml(tmp_path, monkeypatch):
    yaml = tmp_path / "llm.yaml"
    yaml.write_text(
        "provider: openai\nbase_url: https://yaml.example/v1\nmodel: yaml-model\n"
        "temperature: 1.0\nmax_tokens: 4096\n"
    )
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_BASE_URL", "https://env.example")
    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    cfg = load_llm_config(yaml_path=yaml, env_path=None)
    assert cfg.provider == "anthropic"
    assert cfg.base_url == "https://env.example"
    assert cfg.model == "env-model"
    assert cfg.api_key == "env-key"


def test_missing_api_key_raises(tmp_path, monkeypatch):
    yaml = tmp_path / "llm.yaml"
    yaml.write_text("provider: openai\nbase_url: x\nmodel: y\n")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        load_llm_config(yaml_path=yaml, env_path=None)


def test_unknown_provider_raises(tmp_path, monkeypatch):
    yaml = tmp_path / "llm.yaml"
    yaml.write_text("provider: bogus\nbase_url: x\nmodel: y\n")
    monkeypatch.setenv("LLM_API_KEY", "k")
    with pytest.raises(RuntimeError, match="LLM_PROVIDER"):
        load_llm_config(yaml_path=yaml, env_path=None)
