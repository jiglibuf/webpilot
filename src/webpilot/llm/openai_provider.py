"""OpenAI Chat Completions provider - and every OpenAI-compatible endpoint.

The official ``openai`` package speaks the same protocol as DeepSeek,
OpenRouter, vLLM, llama.cpp, LM Studio, Ollama's ``/v1`` shim and most
gateways, so one provider covers them all; only ``base_url`` and ``model``
change.  :class:`~webpilot.llm.local_provider.LocalProvider` subclasses this for
the local-model quirks.

Mapping done here:

* ``tools=[spec.to_openai()]`` + ``tool_choice="auto"``;
* ``response.choices[0].message.tool_calls`` -> :class:`ToolCall` with the JSON
  arguments parsed; malformed JSON never raises (the raw string is kept and a
  parse-error note is attached, see :mod:`webpilot.llm.base`);
* ``response.usage`` -> :class:`Usage`, including
  ``prompt_tokens_details.cached_tokens`` (DeepSeek reports its cache hits as
  ``prompt_cache_hit_tokens`` instead - both are understood);
* provider errors -> :class:`LLMError` with ``status``/``retryable``.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import openai

from ..errors import LLMError
from ..types import LLMMessage, LLMResponse, ToolCall, ToolSpec, Usage
from .base import (
    PLACEHOLDER_API_KEY,
    BaseLLM,
    is_retryable_status,
    parse_tool_arguments,
    set_tool_call_diagnostics,
)

__all__ = ["OpenAIProvider"]


class OpenAIProvider(BaseLLM):
    """Chat Completions provider built on the official ``openai`` client."""

    provider = "openai"
    _default_provider = "openai"

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._client: openai.OpenAI | None = None

    # ------------------------------------------------------------- plumbing

    @property
    def client(self) -> openai.OpenAI:
        """Lazily built SDK client.

        Laziness matters: ``build_llm()`` must not touch the network (and must not
        fail on a machine without credentials), so the client is created on the
        first actual request.
        """
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=self.api_key or PLACEHOLDER_API_KEY,
                base_url=self.base_url or None,
                timeout=self.request_timeout,
                # our own retry loop is the only retry policy in this project
                max_retries=0,
            )
        return self._client

    def _close_client(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover - closing must never raise
                pass
            self._client = None

    # ---------------------------------------------------------------- request

    def build_request(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSpec],
        *,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """The exact kwargs handed to ``chat.completions.create`` (test seam)."""
        request: dict[str, Any] = {
            "model": self.model,
            "messages": self.to_openai_messages(messages, system),
            "temperature": temperature,
            "timeout": self.request_timeout,
        }
        if max_tokens:
            request["max_tokens"] = int(max_tokens)
        if tools:
            request["tools"] = [spec.to_openai() for spec in tools]
            request["tool_choice"] = "auto"
        return request

    def _complete_once(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSpec],
        *,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        request = self.build_request(
            messages, tools, system=system, temperature=temperature, max_tokens=max_tokens
        )
        raw = self.client.chat.completions.create(**request)
        return self.parse_response(raw)

    # --------------------------------------------------------------- response

    def parse_response(self, raw: Any) -> LLMResponse:
        """Map a Chat Completions payload onto :class:`LLMResponse`."""
        choices = getattr(raw, "choices", None) or []
        if not choices:
            raise LLMError(f"{self.provider}: response contains no choices", retryable=True)
        choice = choices[0]
        message = getattr(choice, "message", None)
        text = _as_text(getattr(message, "content", None))
        calls = self.parse_tool_calls(getattr(message, "tool_calls", None))
        response = LLMResponse(
            text=text,
            tool_calls=calls,
            usage=self.parse_usage(getattr(raw, "usage", None)),
            stop_reason=str(getattr(choice, "finish_reason", "") or ""),
            raw=raw,
        )
        return self.postprocess(response)

    def parse_tool_calls(self, raw_calls: Any) -> list[ToolCall]:
        """Map ``message.tool_calls`` -> :class:`ToolCall` (never raises)."""
        calls: list[ToolCall] = []
        for index, item in enumerate(raw_calls or []):
            function = getattr(item, "function", None)
            name = str(getattr(function, "name", "") or "")
            raw_arguments = getattr(function, "arguments", None)
            if isinstance(raw_arguments, (dict, list)):
                raw_arguments = json.dumps(raw_arguments, ensure_ascii=False)
            args, note = parse_tool_arguments(raw_arguments)
            call = ToolCall(
                name=name,
                args=args,
                id=str(getattr(item, "id", "") or f"call_{index}"),
            )
            set_tool_call_diagnostics(
                call,
                parse_error=note,
                raw_arguments=raw_arguments if isinstance(raw_arguments, str) else None,
                source="provider",
            )
            calls.append(call)
        return calls

    def parse_usage(self, raw_usage: Any) -> Usage:
        """Map ``response.usage`` -> :class:`Usage`, cache hits included."""
        if raw_usage is None:
            return Usage(calls=1)
        prompt = int(getattr(raw_usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(raw_usage, "completion_tokens", 0) or 0)
        return Usage(prompt, completion, self._cached_tokens(raw_usage), 1)

    @staticmethod
    def _cached_tokens(raw_usage: Any) -> int:
        details = getattr(raw_usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details is not None else None
        if cached is None:
            # DeepSeek / some gateways use a flatter name
            cached = getattr(raw_usage, "prompt_cache_hit_tokens", None)
        if cached is None:
            extra = getattr(raw_usage, "model_extra", None) or {}
            cached = extra.get("cached_tokens") or extra.get("prompt_cache_hit_tokens")
        try:
            return int(cached or 0)
        except (TypeError, ValueError):
            return 0

    def postprocess(self, response: LLMResponse) -> LLMResponse:
        """Hook for subclasses to fix up a parsed response (see LocalProvider)."""
        return response

    # ---------------------------------------------------------------- errors

    def normalise_error(self, exc: BaseException) -> LLMError:
        """Map ``openai`` exceptions onto :class:`LLMError`."""
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, openai.APITimeoutError):
            return LLMError(
                f"{self.provider}: request timed out after {self.request_timeout}s "
                f"(model {self.model!r})",
                status=None,
                retryable=True,
            )
        if isinstance(exc, openai.APIConnectionError):
            return LLMError(
                f"{self.provider}: cannot reach {self.base_url or 'api.openai.com'} "
                f"({exc}). Is the endpoint running?",
                status=None,
                retryable=True,
            )
        if isinstance(exc, openai.APIStatusError):
            status = getattr(exc, "status_code", None) or getattr(
                getattr(exc, "response", None), "status_code", None
            )
            retryable = is_retryable_status(status)
            return LLMError(
                f"{self.provider}: HTTP {status} from {self.base_url or 'api.openai.com'}: {_short(exc)}",
                status=status,
                retryable=retryable,
            )
        if isinstance(exc, openai.OpenAIError):
            return LLMError(f"{self.provider}: {type(exc).__name__}: {_short(exc)}", retryable=True)
        return super().normalise_error(exc)


def _as_text(content: Any) -> str:
    """Content may be a string, ``None`` or a list of content parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                parts.append(str(text))
        return "".join(parts)
    return str(content)


def _short(exc: BaseException, limit: int = 300) -> str:
    message = str(exc).replace("\n", " ").strip()
    return message if len(message) <= limit else message[: limit - 1] + "…"
