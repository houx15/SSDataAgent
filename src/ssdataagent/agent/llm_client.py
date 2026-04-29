from __future__ import annotations

from typing import Protocol

from anthropic import Anthropic
from openai import OpenAI

from ssdataagent.config import LLMConfig


class LLMClient(Protocol):
    def chat(self, messages: list[dict], system: str | None = None) -> str: ...


class OpenAICompatibleClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._sdk = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    def chat(self, messages: list[dict], system: str | None = None) -> str:
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}, *msgs]
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
