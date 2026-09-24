"""Anthropic Messages API provider.

The Messages API differs from Chat Completions in three ways that matter here:

1. the system prompt is a **top-level parameter**, not a message;
2. tool calls come back as ``tool_use`` content blocks (``{id, name, input}``)
   with ``stop_reason="tool_use"``, and the assistant turn must be echoed back
   **with those blocks, ids included**;
3. there is no ``tool`` role - every result is a ``tool_result`` block inside the
   *next user turn*, keyed by ``tool_use_id``, and all results for one assistant
   turn travel in a single user message.

:func:`webpilot.llm.base.to_anthropic_messages` performs (1) and (3); this module
does the request/response plumbing.
"""

from __future__ import annotations

import inspect
from typing import Any, Sequence

import anthropic

from ..errors import LLMError
from ..types import LLMMessage, LLMResponse, ToolCall, ToolSpec, Usage
from .base import (
    PLACEHOLDER_API_KEY,
    BaseLLM,
    is_retryable_status,
    parse_tool_arguments,
    set_tool_call_diagnostics,
)

__all__ = ["AnthropicProvider"]

#: Anthropic requires ``max_tokens``; this is the fallback when the caller passes 0.
DEFAULT_MAX_TOKENS = 2048


def sdk_accepts_temperature() -> bool:
    """Does the installed SDK's Messages API take a ``temperature``?

    It depends on the SDK version (newer ones moved sampling out of the typed
    surface), so we ask instead of assuming - and simply leave the parameter out
    when the endpoint would not understand it.
    """
    try:
        from anthropic.resources.messages.messages import Messages  # type: ignore

        if "temperature" in inspect.signature(Messages.create).parameters:
            return True
    except Exception:  # pragma: no cover - SDK layout changed
        pass
    try:  # pragma: no cover - fallback path
        client = anthropic.Anthropic(api_key=PLACEHOLDER_API_KEY)
        return "temperature" in inspect.signature(client.messages.create).parameters
    except Exception:
        return False


class AnthropicProvider(BaseLLM):
    """Messages-API provider built on the official ``anthropic`` client."""

    provider = "anthropic"
    _default_provider = "anthropic"
    #: cached answer to :meth:`supports_temperature` (``None`` = not probed yet)
    _temperature_supported: bool | None = None

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self._client: anthropic.Anthropic | None = None
        if self.temperature is not None and self.temperature != 0.0:
            # Anthropic accepts 0 < temperature <= 1 (0.0 is fine and means greedy).
            self.temperature = min(max(self.temperature, 0.0), 1.0)

    # ------------------------------------------------------------- plumbing

    def supports_temperature(self) -> bool:
        """Whether ``temperature`` may be sent (cached per process)."""
        if AnthropicProvider._temperature_supported is None:
            AnthropicProvider._temperature_supported = sdk_accepts_temperature()
        return bool(AnthropicProvider._temperature_supported)

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazily built SDK client (no network / no credentials at build time)."""
        if self._client is None:
            self._client = anthropic.Anthropic(
                api_key=self.api_key or PLACEHOLDER_API_KEY,
                base_url=self.base_url or None,
                timeout=self.request_timeout,
                max_retries=0,  # BaseLLM owns the retry policy
            )
        return self._client

    def _close_client(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover
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
        """The exact kwargs handed to ``messages.create`` (test seam)."""
        wire_messages, wire_system = self.to_anthropic_messages(messages, system)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": wire_messages,
            "max_tokens": int(max_tokens) or DEFAULT_MAX_TOKENS,
            "timeout": self.request_timeout,
        }
        if wire_system:
            request["system"] = wire_system
        if temperature is not None and self.supports_temperature():
            request["temperature"] = float(temperature)
        if tools:
            request["tools"] = [spec.to_anthropic() for spec in tools]
            request["tool_choice"] = {"type": "auto"}
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
        raw = self.client.messages.create(**request)
        return self.parse_response(raw)

    # --------------------------------------------------------------- response

    def parse_response(self, raw: Any) -> LLMResponse:
        """Map a Messages payload onto :class:`LLMResponse`."""
        blocks = getattr(raw, "content", None) or []
        texts: list[str] = []
        calls: list[ToolCall] = []
        for index, block in enumerate(blocks):
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text = getattr(block, "text", "") or ""
                if text:
                    texts.append(text)
            elif block_type == "tool_use":
                calls.append(self.parse_tool_use(block, index))
        return LLMResponse(
            text="".join(texts),
            tool_calls=calls,
            usage=self.parse_usage(getattr(raw, "usage", None)),
            stop_reason=str(getattr(raw, "stop_reason", "") or ""),
            raw=raw,
        )

    def parse_tool_use(self, block: Any, index: int = 0) -> ToolCall:
        """Map one ``tool_use`` block -> :class:`ToolCall` (never raises)."""
        name = str(getattr(block, "name", "") or "")
        raw_input = getattr(block, "input", None)
        if isinstance(raw_input, str):
            args, note = parse_tool_arguments(raw_input)
            raw_repr: str | None = raw_input
        elif raw_input is None:
            args, note, raw_repr = {}, None, None
        else:
            args, note, raw_repr = dict(raw_input), None, None
        call = ToolCall(name=name, args=args, id=str(getattr(block, "id", "") or f"toolu_{index}"))
        set_tool_call_diagnostics(call, parse_error=note, raw_arguments=raw_repr, source="provider")
        return call

    def parse_usage(self, raw_usage: Any) -> Usage:
        """``input_tokens``/``output_tokens``/``cache_read_input_tokens``."""
        if raw_usage is None:
            return Usage(calls=1)
        return Usage(
            int(getattr(raw_usage, "input_tokens", 0) or 0),
            int(getattr(raw_usage, "output_tokens", 0) or 0),
            int(getattr(raw_usage, "cache_read_input_tokens", 0) or 0),
            1,
        )

    # ---------------------------------------------------------------- errors

    def normalise_error(self, exc: BaseException) -> LLMError:
        """Map ``anthropic`` exceptions onto :class:`LLMError`."""
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, anthropic.APITimeoutError):
            return LLMError(
                f"{self.provider}: request timed out after {self.request_timeout}s "
                f"(model {self.model!r})",
                status=None,
                retryable=True,
            )
        if isinstance(exc, anthropic.APIConnectionError):
            return LLMError(
                f"{self.provider}: cannot reach {self.base_url or 'api.anthropic.com'}: {exc}",
                status=None,
                retryable=True,
            )
        if isinstance(exc, anthropic.APIStatusError):
            status = getattr(exc, "status_code", None)
            message = getattr(exc, "message", None) or str(exc)
            return LLMError(
                f"{self.provider}: HTTP {status} from {self.base_url or 'api.anthropic.com'}: "
                f"{str(message).replace(chr(10), ' ')[:300]}",
                status=status,
                retryable=is_retryable_status(status),
            )
        if isinstance(exc, anthropic.AnthropicError):
            return LLMError(f"{self.provider}: {type(exc).__name__}: {exc}", retryable=True)
        return super().normalise_error(exc)
