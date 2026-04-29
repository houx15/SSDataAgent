from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAML = REPO_ROOT / "config" / "llm.yaml"


@dataclass(frozen=True)
class LLMConfig:
    provider: Literal["openai", "anthropic"]
    base_url: str
    api_key: str
    model: str
    temperature: float = 1.0
    max_tokens: int = 4096


_SENTINEL = object()


def load_llm_config(
    yaml_path: Path | None = None,
    env_path: Path | None | object = _SENTINEL,
) -> LLMConfig:
    """Load LLM configuration with precedence env > yaml > defaults.

    The API key is *only* read from env (LLM_API_KEY); never from yaml.

    Args:
        yaml_path: yaml file to read, defaults to ``config/llm.yaml``.
        env_path: .env file to load before reading env vars. Pass ``None`` to
            skip loading any .env (useful in tests). Default is the repo root
            ``.env``.
    """
    if env_path is _SENTINEL:
        env_path = REPO_ROOT / ".env"
    if env_path is not None:
        load_dotenv(env_path, override=False)
    yaml_path = yaml_path or DEFAULT_YAML
    data: dict = {}
    if yaml_path.exists():
        with yaml_path.open() as f:
            data = yaml.safe_load(f) or {}

    provider = os.environ.get("LLM_PROVIDER", data.get("provider", "openai"))
    base_url = os.environ.get("LLM_BASE_URL", data.get("base_url", ""))
    model = os.environ.get("LLM_MODEL", data.get("model", ""))
    api_key = os.environ.get("LLM_API_KEY", "")
    temperature = float(data.get("temperature", 1.0))
    max_tokens = int(data.get("max_tokens", 4096))

    if not api_key:
        raise RuntimeError(
            "LLM_API_KEY not found in environment. Set it in .env or export it."
        )
    if provider not in ("openai", "anthropic"):
        raise RuntimeError(f"Unknown LLM_PROVIDER: {provider!r}")

    return LLMConfig(
        provider=provider,  # type: ignore[arg-type]
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
