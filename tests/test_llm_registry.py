"""Registry tests: provider dispatch and credential validation.

``build_llm`` is the single place where a :class:`~webpilot.config.Config`
becomes a concrete provider.  These tests pin the three things that matter:

* every provider name in :data:`webpilot.config.PROVIDERS` builds something that
  looks like an ``LLMClient`` (offline, without a key for the local/fake ones);
* an unknown provider raises :class:`ConfigError` (never ``KeyError``);
* a missing key raises :class:`ConfigError` that *names the environment variable*
  the user has to set, before any network call is attempted.

Nothing here opens a socket: the providers build their SDK clients lazily.
"""

from __future__ import annotations

import pytest

from webpilot.config import PROVIDERS, Config, config_from_env
from webpilot.errors import ConfigError
from webpilot.llm import (
    AnthropicProvider,
    FakeLLM,
    LocalProvider,
    OpenAIProvider,
    build_llm,
    provider_names,
)
from webpilot.llm.registry import BUILDERS

KEYED_PROVIDERS = sorted(name for name, info in PROVIDERS.items() if info["needs_key"])
KEYLESS_PROVIDERS = sorted(name for name, info in PROVIDERS.items() if not info["needs_key"])

#: everything ``LLMClient`` (a non-runtime-checkable Protocol) requires
PROTOCOL_ATTRS = ("provider", "model", "complete", "count_tokens")


@pytest.fixture(autouse=True)
def _no_env_keys(monkeypatch: pytest.MonkeyPatch):
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
                 "OPENROUTER_API_KEY", "WEBPILOT_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def make(**kwargs) -> Config:
    """A Config with a key, so only the test's own credential state matters."""
    kwargs.setdefault("api_key", "test-key")
    return Config(**kwargs)


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", sorted(PROVIDERS))
def test_every_provider_name_builds(name: str):
    llm = build_llm(make(provider=name))

    assert llm.provider == name          # the configured name is preserved
    assert llm.model                     # a model is always resolved
    assert all(hasattr(llm, attr) for attr in PROTOCOL_ATTRS)
    assert llm.count_tokens("hello world") > 0
    assert llm.usage.total == 0
    llm.close()                          # never built a client, must not raise


def test_provider_names_match_the_builders():
    assert provider_names() == sorted(BUILDERS)
    assert provider_names() == sorted(PROVIDERS)


def test_dispatch_map_is_the_expected_one():
    assert isinstance(build_llm(make(provider="openai")), OpenAIProvider)
    assert isinstance(build_llm(make(provider="anthropic")), AnthropicProvider)
    assert isinstance(build_llm(make(provider="deepseek")), OpenAIProvider)
    assert isinstance(build_llm(make(provider="openrouter")), OpenAIProvider)
    assert isinstance(build_llm(make(provider="ollama")), LocalProvider)
    assert isinstance(build_llm(make(provider="openai_compatible")), LocalProvider)
    assert isinstance(build_llm(make(provider="fake")), FakeLLM)
    # the Anthropic client must never be handed an OpenAI-shaped provider
    assert not isinstance(build_llm(make(provider="deepseek")), AnthropicProvider)


def test_deepseek_and_openrouter_keep_their_endpoints():
    deepseek = build_llm(make(provider="deepseek"))
    assert deepseek.base_url == PROVIDERS["deepseek"]["base_url"] == "https://api.deepseek.com/v1"
    assert deepseek.model == "deepseek-chat"

    openrouter = build_llm(make(provider="openrouter"))
    assert openrouter.base_url == PROVIDERS["openrouter"]["base_url"]
    assert openrouter.model == "anthropic/claude-sonnet-4"

    # an explicit override wins (self-hosted gateway in front of DeepSeek)
    custom = build_llm(make(provider="deepseek", base_url="http://127.0.0.1:9000/v1"))
    assert custom.base_url == "http://127.0.0.1:9000/v1"


def test_local_providers_default_to_a_local_endpoint_without_a_key():
    for name in ("ollama", "openai_compatible"):
        llm = build_llm(Config(provider=name))          # api_key deliberately missing
        assert isinstance(llm, LocalProvider)
        assert llm.provider == name
        assert llm.base_url and llm.base_url.startswith("http://localhost")
        assert llm.api_key                              # dummy key, servers ignore it


def test_fake_provider_needs_no_credentials_and_is_deterministic():
    llm = build_llm(Config(provider="fake"))
    assert isinstance(llm, FakeLLM)
    assert llm.model == "fake-1"
    assert llm.requests == []
    assert llm.complete([]).tool_calls[0].name == "finish"   # empty script -> finishes


def test_provider_name_is_normalised():
    cfg = Config(provider="fake", api_key="test-key")
    cfg.provider = " OpenAI "
    llm = build_llm(cfg)
    assert isinstance(llm, OpenAIProvider) and llm.provider == "openai"


# --------------------------------------------------------------------------- #
# failures
# --------------------------------------------------------------------------- #

def test_unknown_provider_raises_config_error():
    cfg = Config(provider="fake")
    cfg.provider = "mistral-large-from-a-fork"

    with pytest.raises(ConfigError) as excinfo:
        build_llm(cfg)

    message = str(excinfo.value)
    assert "mistral-large-from-a-fork" in message
    for name in PROVIDERS:
        assert name in message                 # lists the supported providers


@pytest.mark.parametrize("name", KEYED_PROVIDERS)
def test_missing_key_raises_config_error_naming_the_env_var(name: str):
    cfg = Config(provider=name, api_key=None)
    assert cfg.api_key is None

    with pytest.raises(ConfigError) as excinfo:
        build_llm(cfg)

    message = str(excinfo.value)
    assert PROVIDERS[name]["key_env"] in message          # the actionable bit
    assert name in message
    assert "api" in message.lower() or "key" in message.lower()


@pytest.mark.parametrize("name", KEYED_PROVIDERS)
def test_keyed_provider_builds_once_the_key_is_present(name: str):
    llm = build_llm(Config(provider=name, api_key="sk-test"))
    assert llm.api_key == "sk-test"
    llm.close()


@pytest.mark.parametrize("name", KEYLESS_PROVIDERS)
def test_keyless_provider_never_needs_a_key(name: str):
    build_llm(Config(provider=name))          # must not raise


def test_openai_compatible_without_base_url_is_an_actionable_error():
    cfg = Config(provider="openai_compatible")
    cfg.base_url = None                       # what a hand-built config may look like

    with pytest.raises(ConfigError) as excinfo:
        build_llm(cfg)

    message = str(excinfo.value)
    assert "WEBPILOT_BASE_URL" in message and "--base-url" in message


def test_config_from_env_credentials_flow_into_the_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    cfg = config_from_env(env={"OPENAI_API_KEY": "sk-from-env"}, overrides={"provider": "openai"})
    llm = build_llm(cfg)
    assert isinstance(llm, OpenAIProvider) and llm.api_key == "sk-from-env"

    # and without the variable, the same flow fails with a helpful message
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError) as excinfo:
        build_llm(config_from_env(env={}, overrides={"provider": "openai"}))
    assert "OPENAI_API_KEY" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# offline safety
# --------------------------------------------------------------------------- #

def test_build_llm_never_constructs_an_sdk_client(monkeypatch: pytest.MonkeyPatch):
    """Building a provider must not touch the network or need credentials."""
    import anthropic
    import openai

    def explode(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("build_llm must not create an SDK client")

    monkeypatch.setattr(openai, "OpenAI", explode)
    monkeypatch.setattr(anthropic, "Anthropic", explode)

    for name in PROVIDERS:
        llm = build_llm(make(provider=name))
        assert llm.model
