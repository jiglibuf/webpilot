"""Local / self-hosted models: Ollama, vLLM, llama.cpp, LM Studio, any gateway.

These servers expose an **OpenAI-compatible** endpoint, so the request side is
identical to :class:`~webpilot.llm.openai_provider.OpenAIProvider`.  What differs
is operational rather than protocol:

* no API key - a dummy placeholder is sent (most servers ignore it);
* a sensible default endpoint, ``http://localhost:11434/v1`` (Ollama);
* small models often "call tools" by *writing JSON in the content* instead of
  filling the ``tool_calls`` field.  :meth:`LocalProvider.postprocess` scans the
  text for a fenced/inline/array JSON object carrying a ``name`` and
  ``arguments`` and turns it into a real :class:`ToolCall`, restricted to the
  tools that were actually offered (so prose like ``{"answer": "42"}`` is left
  alone).
"""

from __future__ import annotations

from typing import Any, Sequence

from ..errors import LLMError
from ..types import LLMMessage, LLMResponse, ToolSpec
from .base import PLACEHOLDER_API_KEY, tool_calls_from_text
from .openai_provider import OpenAIProvider

__all__ = ["LocalProvider"]

#: Ollama's OpenAI-compatible endpoint.
DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"


class LocalProvider(OpenAIProvider):
    """OpenAI-compatible provider for local models, with tolerant parsing."""

    provider = "ollama"
    _default_provider = "ollama"

    def __init__(self, config: Any = None, **kwargs: Any) -> None:
        if not kwargs.get("base_url") and not getattr(config, "base_url", None):
            kwargs["base_url"] = DEFAULT_LOCAL_BASE_URL
        if not kwargs.get("api_key") and not getattr(config, "api_key", None):
            # most local servers ignore the key, but the SDK needs one to exist
            kwargs["api_key"] = PLACEHOLDER_API_KEY
        super().__init__(config, **kwargs)
        if not self.base_url:
            self.base_url = DEFAULT_LOCAL_BASE_URL
        #: names of the tools offered on the current request (set per call;
        #: ``[]`` means "none offered", ``None`` means "unknown, do not filter")
        self._offered_tools: list[str] | None = None

    # ---------------------------------------------------------------- request

    def _complete_once(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSpec],
        *,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        # Remember what we offered: the text-fallback must not invent tool names.
        self._offered_tools = [spec.name for spec in tools]
        response = super()._complete_once(
            messages, tools, system=system, temperature=temperature, max_tokens=max_tokens
        )
        if not response.tool_calls and not response.text:
            raise LLMError(
                f"{self.provider}: empty response from {self.base_url or DEFAULT_LOCAL_BASE_URL} - "
                "does this model/server support the OpenAI chat completions API?",
                retryable=True,
            )
        return response

    # --------------------------------------------------------------- response

    def postprocess(self, response: LLMResponse) -> LLMResponse:
        """Recover tool calls that arrived as JSON inside the message content."""
        if response.tool_calls or not response.text:
            return response
        recovered = tool_calls_from_text(
            response.text,
            known_names=self._offered_tools,
            call_id_prefix="call_local",
        )
        if recovered:
            response.tool_calls = recovered
            # normalise: the model did ask for a tool, whatever the server said
            response.stop_reason = "tool_calls"
        return response
