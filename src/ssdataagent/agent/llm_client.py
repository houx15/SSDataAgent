from __future__ import annotations

import time
from typing import Protocol

from anthropic import Anthropic
from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from ssdataagent.config import LLMConfig


_TRANSIENT = (APIConnectionError, RateLimitError)
_DEFAULT_MAX_RETRIES = 4
_BACKOFF_BASE_S = 2.0


class LLMClient(Protocol):
    def chat(self, messages: list[dict], system: str | None = None) -> str: ...


class OpenAICompatibleClient:
    def __init__(self, cfg: LLMConfig, max_retries: int = _DEFAULT_MAX_RETRIES):
        self.cfg = cfg
        self.max_retries = max_retries
        self._sdk = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    def chat(self, messages: list[dict], system: str | None = None) -> str:
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}, *msgs]
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._sdk.chat.completions.create(
                    model=self.cfg.model,
                    messages=msgs,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                )
                msg = resp.choices[0].message
                content = msg.content or ""
                if not content:
                    content = getattr(msg, "reasoning_content", "") or ""
                return content
            except _TRANSIENT as e:
                last_err = e
                if attempt == self.max_retries:
                    break
                time.sleep(_BACKOFF_BASE_S * (2 ** attempt))
            except APIError as e:
                # Treat 5xx as transient; 4xx as fatal.
                status = getattr(e, "status_code", None)
                if status and 500 <= status < 600 and attempt < self.max_retries:
                    last_err = e
                    time.sleep(_BACKOFF_BASE_S * (2 ** attempt))
                    continue
                raise
        assert last_err is not None
        raise last_err


class AnthropicCompatibleClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._sdk = Anthropic(api_key=cfg.api_key, base_url=cfg.base_url)

    def chat(self, messages: list[dict], system: str | None = None) -> str:
        kwargs = dict(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        if system:
            kwargs["system"] = system
        resp = self._sdk.messages.create(**kwargs)
        parts = []
        for block in getattr(resp, "content", []):
            t = getattr(block, "text", None)
            if t:
                parts.append(t)
        return "".join(parts)


def build_client(cfg: LLMConfig) -> LLMClient:
    if cfg.provider == "openai":
        return OpenAICompatibleClient(cfg)
    if cfg.provider == "anthropic":
        return AnthropicCompatibleClient(cfg)
    raise RuntimeError(f"unknown provider: {cfg.provider!r}")
