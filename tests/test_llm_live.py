"""Live smoke tests - deselected by default, run with ``pytest -m live``.

These are the *only* tests that may touch the internet.  Each one is skipped
unless its provider key is present in the environment, so a plain ``pytest`` run
(``addopts = -m 'not live'``) never even collects them as passing.

What they check (cheaply - one short completion per provider):

* the request actually leaves the machine and is accepted (auth + endpoint);
* the tool-calling round trip works against the real API: the model is offered a
  single ``subtract`` tool and must call it with valid JSON arguments;
* usage comes back and is turned into :class:`~webpilot.types.Usage`.

They deliberately do not assert on model *wording*, only on the contract.
"""

from __future__ import annotations

import os

import pytest

from webpilot.config import Config
from webpilot.llm import BaseLLM, build_llm
from webpilot.types import LLMMessage, ToolSpec

pytestmark = pytest.mark.live

TOOLS = [
    ToolSpec(
        name="subtract",
        description="Subtract b from a and return the result.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    )
]

PROMPT = [LLMMessage(role="user", content="Use the subtract tool to compute 41 minus 1.")]


def _key(name: str) -> str | None:
    value = os.environ.get(name)
    return value or None


def _run(provider: str, key_env: str, model: str | None = None, base_url: str | None = None):
    """Build the configured provider and assert one tool call round trip."""
    config = Config(
        provider=provider,
        model=model or "",
        api_key=_key(key_env),
        base_url=base_url or "",
        request_timeout=60.0,
        llm_max_retries=2,          # the real endpoint may 429 once
        max_output_tokens=256,
    )
    llm = build_llm(config)
    response = llm.complete(PROMPT, TOOLS, system="You are a calculator. Call the tool.", temperature=0.0)

    assert isinstance(llm, BaseLLM)
    assert response.usage.prompt_tokens > 0, "the provider reported no input tokens"
    assert response.usage.calls == 1
    assert llm.count_tokens(response.text) >= 0
    assert response.tool_calls, f"{provider} did not call the tool: {response.text!r}"
    call = response.tool_calls[0]
    assert call.name == "subtract"
    assert set(call.args) >= {"a", "b"}, f"arguments were not usable: {call.args!r}"
    assert call.id, "tool calls need an id so the result can be paired"
    llm.close()
    return response


@pytest.mark.skipif(not _key("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set")
def test_openai_live_smoke():
    _run("openai", "OPENAI_API_KEY")


@pytest.mark.skipif(not _key("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set")
def test_anthropic_live_smoke():
    # Anthropic needs the tool history rebuilt the Anthropic way: send the
    # assistant turn back with its tool_use block and the result as a user turn.
    from webpilot.types import LLMMessage as Msg

    config = Config(provider="anthropic", api_key=_key("ANTHROPIC_API_KEY"),
                    request_timeout=60.0, llm_max_retries=2, max_output_tokens=256)
    llm = build_llm(config)
    first = llm.complete(PROMPT, TOOLS, system="You are a calculator. Call the tool.")
    assert first.tool_calls, f"anthropic did not call the tool: {first.text!r}"

    call = first.tool_calls[0]
    follow_up = llm.complete(
        [
            *PROMPT,
            Msg(role="assistant", content=first.text, tool_calls=first.tool_calls),
            Msg(role="tool", content="40", tool_call_id=call.id),
        ],
        TOOLS,
        system="You are a calculator. Call the tool.",
    )
    assert follow_up.usage.prompt_tokens > 0
    assert follow_up.text or follow_up.tool_calls, "empty follow-up response"
    llm.close()


@pytest.mark.skipif(not _key("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY is not set")
def test_deepseek_live_smoke():
    _run("deepseek", "DEEPSEEK_API_KEY", model="deepseek-chat")


@pytest.mark.skipif(not _key("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY is not set")
def test_openrouter_live_smoke():
    _run("openrouter", "OPENROUTER_API_KEY")


@pytest.mark.skipif(
    not _key("WEBPILOT_LIVE_LOCAL"),
    reason="no local model endpoint requested (set WEBPILOT_LIVE_LOCAL=1)",
)
def test_local_live_smoke():
    _run("ollama", "WEBPILOT_FAKE_KEY", base_url=_key("WEBPILOT_BASE_URL") or "http://localhost:11434/v1")
