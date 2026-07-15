from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from anthropic import Anthropic
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

from ssdataagent.config import LLMConfig


_TRANSIENT = (APIConnectionError, RateLimitError, APITimeoutError)
_DEFAULT_MAX_RETRIES = 4
_BACKOFF_BASE_S = 2.0
# Per-request hard timeout. The pilot once hung for 2+ hours on a stuck
# proxy connection because the SDK's default 600s default was indistinguishable
# from a long deepseek-v4-flash reasoning chain. 300s is generous for
# reasoning calls but bounds the pathological case.
_REQUEST_TIMEOUT_S = 300.0


@dataclass
class ToolCallRequest:
    """One function call the assistant emitted in a tool-using turn."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolUsingResponse:
    """A single tool-using LLM turn. Either content (final answer when no
    tools were called) or tool_calls (still iterating). Both can be present.
    `assistant_message` is the OpenAI-shaped dict the caller should append
    verbatim to the next request's `messages` list."""
    content: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    assistant_message: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    def chat(self, messages: list[dict], system: str | None = None) -> str: ...

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
        tool_choice: str = "auto",
    ) -> ToolUsingResponse: ...


class OpenAICompatibleClient:
    def __init__(self, cfg: LLMConfig, max_retries: int = _DEFAULT_MAX_RETRIES):
        self.cfg = cfg
        self.max_retries = max_retries
        self._sdk = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=_REQUEST_TIMEOUT_S,
        )

    @staticmethod
    def _first_message(resp):
        """Return resp.choices[0].message, or raise with the provider's own error.

        OpenRouter proxies many providers and, when the upstream one errors,
        returns HTTP 200 with `choices: null` and an `error` object rather than a
        non-2xx status. `resp.choices[0]` then raises a bare
        'NoneType object is not subscriptable' that hides the real cause. Surface
        the provider message instead so a model that rejects the request (bad
        tool schema, context overflow, provider outage) is diagnosable.
        """
        choices = getattr(resp, "choices", None)
        if not choices:
            err = getattr(resp, "error", None)
            detail = ""
            if err is not None:
                detail = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise RuntimeError(
                f"LLM response carried no choices (provider error: {detail or 'none given'})"
            )
        return choices[0].message

    def _token_kwarg(self) -> str:
        # OpenAI's GPT-5 / o-series chat-completions endpoint rejects the
        # legacy `max_tokens` and requires `max_completion_tokens`. DeepSeek's
        # OpenAI-compatible endpoint still wants `max_tokens`.
        return (
            "max_completion_tokens"
            if "api.openai.com" in (self.cfg.base_url or "")
            else "max_tokens"
        )

    def _retry_loop(self, do_call):
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return do_call()
            except _TRANSIENT as e:
                last_err = e
                if attempt == self.max_retries:
                    break
                time.sleep(_BACKOFF_BASE_S * (2 ** attempt))
            except APIError as e:
                status = getattr(e, "status_code", None)
                if status and 500 <= status < 600 and attempt < self.max_retries:
                    last_err = e
                    time.sleep(_BACKOFF_BASE_S * (2 ** attempt))
                    continue
                raise
        assert last_err is not None
        raise last_err

    def chat(self, messages: list[dict], system: str | None = None) -> str:
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}, *msgs]

        def call():
            resp = self._sdk.chat.completions.create(
                model=self.cfg.model,
                messages=msgs,
                temperature=self.cfg.temperature,
                **{self._token_kwarg(): self.cfg.max_tokens},
            )
            msg = self._first_message(resp)
            content = msg.content or ""
            if not content:
                content = getattr(msg, "reasoning_content", "") or ""
            return content

        return self._retry_loop(call)

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
        tool_choice: str = "auto",
    ) -> ToolUsingResponse:
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}, *msgs]

        def call():
            resp = self._sdk.chat.completions.create(
                model=self.cfg.model,
                messages=msgs,
                tools=tools,
                tool_choice=tool_choice,
                temperature=self.cfg.temperature,
                **{self._token_kwarg(): self.cfg.max_tokens},
            )
            return self._first_message(resp)

        msg = self._retry_loop(call)
        return _parse_openai_tool_message(msg)


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

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
        tool_choice: str = "auto",
    ) -> ToolUsingResponse:
        # Anthropic uses a different tool-use envelope than OpenAI. EXP-006
        # spike targets gpt-5.4 only; add this when the project needs to run
        # the tool-using path on Claude.
        raise NotImplementedError(
            "chat_with_tools is not implemented for the Anthropic provider yet. "
            "EXP-006 spike runs on gpt-5.4 (OpenAI-shaped). Wire this if/when "
            "we move the tool path to Claude."
        )


def _parse_openai_tool_message(msg) -> ToolUsingResponse:
    """Turn an OpenAI ChatCompletionMessage into our internal shape.
    `msg` may be either an SDK object or, in tests, a plain dict."""
    if isinstance(msg, dict):
        content = msg.get("content") or ""
        tool_calls_raw = msg.get("tool_calls") or []
    else:
        content = getattr(msg, "content", None) or ""
        tool_calls_raw = getattr(msg, "tool_calls", None) or []

    tool_calls: list[ToolCallRequest] = []
    serialized_tool_calls: list[dict] = []
    for tc in tool_calls_raw:
        if isinstance(tc, dict):
            tc_id = tc["id"]
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args_json = fn.get("arguments", "{}") or "{}"
        else:
            tc_id = tc.id
            name = tc.function.name
            args_json = tc.function.arguments or "{}"
        try:
            args = json.loads(args_json) if isinstance(args_json, str) else dict(args_json)
        except json.JSONDecodeError:
            args = {"_raw_arguments": args_json, "_parse_error": "JSONDecodeError"}
        tool_calls.append(ToolCallRequest(id=tc_id, name=name, arguments=args))
        serialized_tool_calls.append({
            "id": tc_id,
            "type": "function",
            "function": {"name": name, "arguments": args_json if isinstance(args_json, str) else json.dumps(args_json)},
        })

    assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
    if serialized_tool_calls:
        assistant_message["tool_calls"] = serialized_tool_calls
    return ToolUsingResponse(
        content=content,
        tool_calls=tool_calls,
        assistant_message=assistant_message,
    )


def build_client(cfg: LLMConfig) -> LLMClient:
    if cfg.provider == "openai":
        return OpenAICompatibleClient(cfg)
    if cfg.provider == "anthropic":
        return AnthropicCompatibleClient(cfg)
    raise RuntimeError(f"unknown provider: {cfg.provider!r}")
