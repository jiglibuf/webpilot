"""``FakeLLM`` - the deterministic, offline provider.

Used by the agent-loop tests, by the CLI in demo mode and anywhere a scripted
model is more useful than a real one.  Two ways to drive it:

* ``FakeLLM(responses=[...])`` - a list of :class:`LLMResponse` returned in order,
  the **last one repeating** forever (so a script can end with a ``finish`` call
  and stay there);
* ``FakeLLM(responder=callable)`` - ``responder(messages, tools) -> LLMResponse``
  called on every request; a responder that takes ``(messages, tools, system)``
  is detected and given the system prompt too (handy for prompt assertions).

Every request is recorded in :attr:`FakeLLM.requests` as a plain dict
(``messages``, ``tools``, ``system``, ``temperature``, ``max_tokens``) so tests
can assert what the agent actually asked for.  Usage accumulates in
:attr:`FakeLLM.usage`; :meth:`count_tokens` comes from ``BaseLLM``.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Iterable, Sequence

from ..config import Config
from ..types import LLMMessage, LLMResponse, ToolCall, ToolSpec, Usage
from .base import BaseLLM

__all__ = ["FakeLLM"]

Responder = Callable[..., LLMResponse]


def _responder_arity(responder: Responder) -> int:
    try:
        signature = inspect.signature(responder)
    except (TypeError, ValueError):  # pragma: no cover - builtins / C callables
        return 2
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    if any(parameter.kind == parameter.VAR_POSITIONAL for parameter in signature.parameters.values()):
        return 2
    return len(positional)


class FakeLLM(BaseLLM):
    """Scripted provider: no network, fully deterministic, records everything."""

    provider = "fake"
    _default_provider = "fake"

    def __init__(
        self,
        responses: Iterable[LLMResponse] | None = None,
        responder: Responder | None = None,
        model: str = "fake-1",
        *,
        config: Config | None = None,
        usage: Usage | None = None,
    ) -> None:
        super().__init__(config, model=model)
        self.responses: list[LLMResponse] = list(responses or [])
        self.responder = responder
        self.model = model
        self.requests: list[dict[str, Any]] = []
        self.usage = usage or Usage()
        self._cursor = 0

    # ------------------------------------------------------------------ API

    def add(self, response: LLMResponse) -> "FakeLLM":
        """Append one more scripted response (chainable)."""
        self.responses.append(response)
        return self

    def reset(self) -> "FakeLLM":
        """Forget the script cursor, the recorded requests and the usage."""
        self._cursor = 0
        self.requests.clear()
        self.usage = Usage()
        return self

    def last_request(self) -> dict[str, Any] | None:
        return self.requests[-1] if self.requests else None

    @property
    def calls(self) -> int:
        return len(self.requests)

    # -------------------------------------------------------------- provider

    def _complete_once(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSpec],
        *,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        self.requests.append(
            {
                "messages": list(messages),
                "tools": list(tools),
                "system": system,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "model": self.model,
            }
        )
        response = self._next_response(list(messages), list(tools), system)
        return response

    def _next_response(
        self,
        messages: list[LLMMessage],
        tools: list[ToolSpec],
        system: str | None,
    ) -> LLMResponse:
        if self.responder is not None:
            if _responder_arity(self.responder) >= 3:
                response = self.responder(messages, tools, system)
            else:
                response = self.responder(messages, tools)
        elif self.responses:
            index = min(self._cursor, len(self.responses) - 1)
            self._cursor += 1
            response = self.responses[index]
        else:
            # Nothing scripted: behave like an agent that has nothing left to do.
            response = LLMResponse(
                text="nothing scripted",
                tool_calls=[ToolCall(name="finish", args={"success": True, "answer": "nothing scripted"})],
                stop_reason="tool_calls",
            )
        if not isinstance(response, LLMResponse):  # pragma: no cover - guard rail
            raise TypeError(
                f"FakeLLM responder must return LLMResponse, got {type(response).__name__}"
            )
        if not response.usage.calls:
            response.usage.calls = 1
        return response
