"""Provider dispatch: ``build_llm(config) -> LLMClient``.

One place decides which concrete provider a :class:`~webpilot.config.Config`
maps to, and one place validates credentials *before* any network call, with an
error message that names the environment variable the user has to set.

+--------------------+--------------------------------------------------------+
| ``config.provider``| built by                                                |
+====================+========================================================+
| ``openai``         | :class:`OpenAIProvider`                                 |
| ``anthropic``      | :class:`AnthropicProvider`                              |
| ``deepseek``       | :class:`OpenAIProvider` (DeepSeek base_url)             |
| ``openrouter``     | :class:`OpenAIProvider` (OpenRouter base_url)           |
| ``ollama``         | :class:`LocalProvider`                                  |
| ``openai_compatible`` | :class:`LocalProvider` (vLLM / llama.cpp / LM Studio) |
| ``fake``           | :class:`FakeLLM` (tests, offline demos)                 |
+--------------------+--------------------------------------------------------+

SDK clients are created lazily by the providers, so ``build_llm`` never touches
the network and never fails on a machine without keys for *other* providers.
"""

from __future__ import annotations

from typing import Callable

from ..config import PROVIDERS, Config
from ..errors import ConfigError
from ..types import LLMClient
from .anthropic_provider import AnthropicProvider
from .base import BaseLLM
from .fake import FakeLLM
from .local_provider import LocalProvider
from .openai_provider import OpenAIProvider

__all__ = ["build_llm", "BUILDERS", "provider_names"]

#: provider -> factory.  ``openai``/``deepseek``/``openrouter`` share the Chat
#: Completions provider; only the endpoint in the config differs.
BUILDERS: dict[str, Callable[[Config], BaseLLM]] = {
    "openai": OpenAIProvider,
    "deepseek": OpenAIProvider,
    "openrouter": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": LocalProvider,
    "openai_compatible": LocalProvider,
    "fake": lambda config: FakeLLM(model=config.model or "fake-1", config=config),
}

#: providers whose endpoint must be configured explicitly (no sane default host)
_COMPAT_PROVIDERS = frozenset({"ollama", "openai_compatible"})


def provider_names() -> list[str]:
    """Names ``build_llm`` accepts (kept in sync with :data:`PROVIDERS`)."""
    return sorted(BUILDERS)


def build_llm(config: Config) -> LLMClient:
    """Build the provider described by ``config``.

    Raises :class:`~webpilot.errors.ConfigError` for an unknown provider or a
    missing API key (naming the environment variable to set).
    """
    name = str(getattr(config, "provider", "") or "").strip().lower()
    if name not in BUILDERS:
        raise ConfigError(
            f"unknown provider {name or config.provider!r}; "
            f"choose one of: {', '.join(provider_names())} "
            f"(env WEBPILOT_PROVIDER, or --provider)"
        )
    if name not in PROVIDERS:
        # config.Config validates this too, but build_llm may be handed any object.
        raise ConfigError(f"provider {name!r} is not registered in webpilot.config.PROVIDERS")
    if getattr(config, "provider", None) != name and callable(getattr(config, "copy_with", None)):
        # normalise "OpenAI " / "ANTHROPIC" so Config's own lookups keep working
        config = config.copy_with(provider=name)

    if bool(PROVIDERS[name].get("needs_key")):
        validate = getattr(config, "validate_credentials", None)
        if callable(validate):
            validate()  # raises ConfigError naming the env var
        elif not getattr(config, "api_key", None):
            raise ConfigError(
                f"provider {name!r} needs an API key: set {PROVIDERS[name]['key_env']} "
                f"(env, .env or --api-key)."
            )

    if name in _COMPAT_PROVIDERS and not getattr(config, "base_url", None):
        raise ConfigError(
            f"provider {name!r} needs an endpoint: set WEBPILOT_BASE_URL or --base-url "
            f"(e.g. http://localhost:11434/v1)"
        )

    return BUILDERS[name](config)
