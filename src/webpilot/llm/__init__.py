"""The LLM provider layer.

Uniform tool-calling interface (``webpilot.types.LLMClient``) over every backend
webpilot supports: OpenAI Chat Completions, the Anthropic Messages API, any
OpenAI-compatible endpoint (DeepSeek, OpenRouter, Ollama, vLLM, llama.cpp,
LM Studio) and a deterministic offline fake.

Typical use::

    from webpilot.llm import build_llm
    llm = build_llm(config)                  # dispatch + credential check
    response = llm.complete(messages, tools, system="...")
    for call in response.tool_calls:
        ...

Nothing here talks to the network at import or build time: SDK clients are
created lazily on the first request, which keeps the offline test-suite (and the
``fake`` provider) free of credentials.
"""

from __future__ import annotations

from .anthropic_provider import AnthropicProvider
from .base import (
    DEFAULT_BACKOFF_BASE,
    DEFAULT_BACKOFF_MAX,
    DEFAULT_RETRY_JITTER,
    PLACEHOLDER_API_KEY,
    RETRYABLE_STATUS_CODES,
    BaseLLM,
    is_retryable_status,
    iter_json_objects,
    parse_tool_arguments,
    set_tool_call_diagnostics,
    to_anthropic_messages,
    to_openai_messages,
    tool_call_parse_error,
    tool_call_raw_arguments,
    tool_call_source,
    tool_calls_from_text,
)
from .fake import FakeLLM
from .local_provider import DEFAULT_LOCAL_BASE_URL, LocalProvider
from .openai_provider import OpenAIProvider
from .registry import BUILDERS, build_llm, provider_names

__all__ = [
    # providers
    "BaseLLM",
    "OpenAIProvider",
    "AnthropicProvider",
    "LocalProvider",
    "FakeLLM",
    "build_llm",
    "provider_names",
    "BUILDERS",
    "DEFAULT_LOCAL_BASE_URL",
    # wire-format helpers
    "to_openai_messages",
    "to_anthropic_messages",
    "parse_tool_arguments",
    "tool_calls_from_text",
    "iter_json_objects",
    "set_tool_call_diagnostics",
    "tool_call_parse_error",
    "tool_call_raw_arguments",
    "tool_call_source",
    "is_retryable_status",
    "RETRYABLE_STATUS_CODES",
    "PLACEHOLDER_API_KEY",
    "DEFAULT_BACKOFF_BASE",
    "DEFAULT_BACKOFF_MAX",
    "DEFAULT_RETRY_JITTER",
]
