"""Offline tests for the LLM provider layer.

Everything here runs against a tiny in-process ``http.server`` that serves canned
OpenAI-shaped and Anthropic-shaped payloads, so the OpenAI/Anthropic/DeepSeek
mapping code is exercised through the *real* SDKs without a network or an API
key.  The scripted failures (429 then 200, 500, 400, a hung request) prove the
retry policy and the error normalisation.

Absolute rules asserted here:
* a malformed tool-call argument string never raises - it yields a parse-error
  note and keeps the raw text;
* retries happen exactly ``llm_max_retries`` times, never more;
* a 4xx that is not a rate limit fails immediately with ``retryable=False``.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from webpilot.config import Config
from webpilot.errors import LLMError
from webpilot.llm import (
    AnthropicProvider,
    FakeLLM,
    LocalProvider,
    OpenAIProvider,
    PLACEHOLDER_API_KEY,
    parse_tool_arguments,
    to_anthropic_messages,
    to_openai_messages,
    tool_call_parse_error,
    tool_call_raw_arguments,
    tool_call_source,
    tool_calls_from_text,
)
from webpilot.types import LLMMessage, LLMResponse, ToolCall, ToolSpec, Usage

# --------------------------------------------------------------------------- #
# Canned HTTP server
# --------------------------------------------------------------------------- #


@dataclass
class Canned:
    """One scripted HTTP response."""

    status: int = 200
    body: Any = field(default_factory=dict)
    delay: float = 0.0
    content_type: str = "application/json"


def _encode(body: Any) -> bytes:
    if isinstance(body, bytes):
        return body
    if isinstance(body, str):
        return body.encode("utf-8")
    return json.dumps(body).encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "webpilot-canned/1.0"

    def log_message(self, *args: Any) -> None:  # keep pytest output clean
        pass

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        owner: "CannedServer" = self.server.owner  # type: ignore[attr-defined]
        try:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            text = raw.decode("utf-8", "replace")
            try:
                body = json.loads(text or "{}")
            except ValueError:
                body = {"__unparsed__": text}
            entry = owner.record(
                {
                    "path": self.path,
                    "headers": {key.lower(): value for key, value in self.headers.items()},
                    "body": body,
                    "raw": text,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            owner.errors.append(exc)
            entry = Canned(500, {"error": {"message": str(exc)}})
        if entry.delay:
            time.sleep(entry.delay)
        payload = _encode(entry.body)
        try:
            self.send_response(entry.status)
            self.send_header("content-type", entry.content_type)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):  # pragma: no cover
            self.close_connection = True


class CannedServer:
    """Scripted HTTP endpoint; the last entry repeats once the script is out."""

    def __init__(self) -> None:
        self.script: list[Canned] = []
        self.requests: list[dict[str, Any]] = []
        self.errors: list[BaseException] = []
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.daemon_threads = True
        self._httpd.owner = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self._thread.start()

    # -- lifecycle -------------------------------------------------------- #
    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)

    # -- API -------------------------------------------------------------- #
    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    def set_script(self, *entries: Any) -> "CannedServer":
        self.script = [entry if isinstance(entry, Canned) else Canned(*entry) for entry in entries]
        return self

    def record(self, request: dict[str, Any]) -> Canned:
        self.requests.append(request)
        if not self.script:
            return Canned(200, {})
        return self.script[min(len(self.requests) - 1, len(self.script) - 1)]

    # -- assertions helpers ----------------------------------------------- #
    @property
    def count(self) -> int:
        return len(self.requests)

    def bodies(self) -> list[dict[str, Any]]:
        return [request["body"] for request in self.requests]

    def last_body(self) -> dict[str, Any]:
        return self.requests[-1]["body"]

    def headers(self, index: int = -1) -> dict[str, str]:
        return self.requests[index]["headers"]


@pytest.fixture
def server():
    canned = CannedServer()
    try:
        yield canned
    finally:
        canned.stop()


@pytest.fixture(autouse=True)
def _no_env_keys(monkeypatch: pytest.MonkeyPatch):
    """No ambient credentials: every test must build its own config."""
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# Canned payloads
# --------------------------------------------------------------------------- #

TOOLS = [
    ToolSpec(
        name="click",
        description="Click an element by its id.",
        parameters={
            "type": "object",
            "properties": {"element_id": {"type": "integer"}, "intent": {"type": "string"}},
            "required": ["element_id", "intent"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="type_text",
        description="Type into a field.",
        parameters={
            "type": "object",
            "properties": {"element_id": {"type": "integer"}, "text": {"type": "string"}},
            "required": ["element_id", "text"],
            "additionalProperties": False,
        },
    ),
]


def openai_payload(*, content: str = "on it", tool_calls: list[dict[str, Any]] | None = None,
                   finish_reason: str = "tool_calls", usage: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, "tool_calls": tool_calls or []},
            }
        ],
        "usage": usage if usage is not None else {
            "prompt_tokens": 120, "completion_tokens": 18, "total_tokens": 138,
            "prompt_tokens_details": {"cached_tokens": 64},
        },
    }


def openai_call(name: str, arguments: str, call_id: str = "call_1") -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def anthropic_payload(*, blocks: list[dict[str, Any]] | None = None, stop_reason: str = "tool_use",
                      usage: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": blocks if blocks is not None else [
            {"type": "text", "text": "clicking now"},
            {"type": "tool_use", "id": "toolu_1", "name": "click", "input": {"element_id": 3, "intent": "open"}},
        ],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage if usage is not None else {
            "input_tokens": 500, "output_tokens": 40, "cache_read_input_tokens": 128,
        },
    }


OPENAI_ERROR = {"error": {"message": "rate limited", "type": "rate_limit_error"}}
ANTHROPIC_ERROR = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}


# --------------------------------------------------------------------------- #
# helpers that build providers against the canned server
# --------------------------------------------------------------------------- #

#: kwargs that belong to the provider (retry policy), not to ``Config``.
_PROVIDER_KWARGS = ("backoff_base", "backoff_max", "retry_jitter")


def _split(overrides: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    config_kwargs = {key: value for key, value in overrides.items() if key not in _PROVIDER_KWARGS}
    provider_kwargs = {key: overrides[key] for key in _PROVIDER_KWARGS if key in overrides}
    provider_kwargs.setdefault("backoff_base", 0.0)  # tests never want to actually wait
    return config_kwargs, provider_kwargs


def openai_provider(server: CannedServer, **overrides: Any) -> OpenAIProvider:
    params: dict[str, Any] = {
        "provider": "openai",
        "model": "gpt-test",
        "api_key": "test-key",
        "base_url": server.base_url,
        "request_timeout": 5.0,
        "llm_max_retries": 3,
    }
    config_kwargs, provider_kwargs = _split(overrides)
    params.update(config_kwargs)
    return OpenAIProvider(Config(**params), **provider_kwargs)


def anthropic_provider(server: CannedServer, **overrides: Any) -> AnthropicProvider:
    params: dict[str, Any] = {
        "provider": "anthropic",
        "model": "claude-test",
        "api_key": "test-key",
        "base_url": server.base_url,
        "request_timeout": 5.0,
        "llm_max_retries": 3,
    }
    config_kwargs, provider_kwargs = _split(overrides)
    params.update(config_kwargs)
    return AnthropicProvider(Config(**params), **provider_kwargs)


def local_provider(server: CannedServer, **overrides: Any) -> LocalProvider:
    params: dict[str, Any] = {
        "provider": "ollama",
        "model": "qwen3:8b",
        "base_url": server.base_url,
        "request_timeout": 5.0,
        "llm_max_retries": 0,
    }
    config_kwargs, provider_kwargs = _split(overrides)
    params.update(config_kwargs)
    return LocalProvider(Config(**params), **provider_kwargs)


USER = [LLMMessage(role="user", content="book the cheapest flight")]


# ======================================================================== #
# 1. message conversion (pure, no HTTP)
# ======================================================================== #

def test_to_openai_messages_roundtrips_tool_calls():
    messages = [
        LLMMessage(role="system", content="be careful"),
        LLMMessage(role="user", content="click it"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(name="click", args={"element_id": 3, "intent": "open"}, id="call_9")],
        ),
        LLMMessage(role="tool", content="OK: clicked", tool_call_id="call_9", name="click"),
    ]
    wire = to_openai_messages(messages, "you are webpilot")

    assert [entry["role"] for entry in wire] == ["system", "system", "user", "assistant", "tool"]
    assert wire[0]["content"] == "you are webpilot"
    call = wire[3]["tool_calls"][0]
    assert call["id"] == "call_9" and call["type"] == "function"
    assert call["function"]["name"] == "click"
    assert json.loads(call["function"]["arguments"]) == {"element_id": 3, "intent": "open"}
    # the tool result keeps its id, which is what lets the model pair them up
    assert wire[4]["tool_call_id"] == "call_9"
    assert wire[4]["content"] == "OK: clicked"


def test_to_openai_messages_synthesises_missing_ids():
    wire = to_openai_messages([LLMMessage(role="assistant", tool_calls=[ToolCall(name="click", args={})])])
    assert wire[0]["tool_calls"][0]["id"] == "call_0"


def test_to_openai_messages_keeps_empty_content_for_assistant_tool_turns():
    wire = to_openai_messages([LLMMessage(role="assistant", tool_calls=[ToolCall(name="click", id="c")])])
    assert wire[0]["content"] == ""  # some OpenAI-compatible servers reject null


def test_to_anthropic_messages_rebuilds_history_the_anthropic_way():
    messages = [
        LLMMessage(role="system", content="be careful"),
        LLMMessage(role="user", content="click it"),
        LLMMessage(
            role="assistant",
            content="clicking",
            tool_calls=[
                ToolCall(name="click", args={"element_id": 3}, id="toolu_a"),
                ToolCall(name="setup", args={}, id="toolu_b"),
            ],
        ),
        LLMMessage(role="tool", content="OK: clicked", tool_call_id="toolu_a"),
        LLMMessage(role="tool", content="FAILED: nope", tool_call_id="toolu_b", meta={"is_error": True}),
    ]
    wire, system = to_anthropic_messages(messages, "you are webpilot")

    assert system == "you are webpilot\n\nbe careful"
    assert all(entry["role"] in ("user", "assistant") for entry in wire)
    assert wire[0] == {"role": "user", "content": "click it"}
    assistant = wire[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "text", "text": "clicking"}
    assert assistant["content"][1] == {"type": "tool_use", "id": "toolu_a", "name": "click", "input": {"element_id": 3}}
    assert assistant["content"][2]["id"] == "toolu_b"
    # both results travel in ONE user turn, keyed by tool_use_id
    results = wire[2]
    assert results["role"] == "user"
    assert [block["tool_use_id"] for block in results["content"]] == ["toolu_a", "toolu_b"]
    assert results["content"][0]["content"] == "OK: clicked"
    assert results["content"][1]["is_error"] is True
    # no system message, no tool role leaked into the wire format
    assert not any(entry["role"] == "tool" for entry in wire)


def test_to_anthropic_messages_drops_empty_content_and_fences_orphan_results():
    wire, system = to_anthropic_messages(
        [
            LLMMessage(role="user", content=""),
            LLMMessage(role="assistant", content=""),
            LLMMessage(role="tool", content="stray result"),
        ]
    )
    assert system is None
    assert len(wire) == 1
    assert wire[0]["role"] == "user"
    # an unattributable result cannot claim a tool_use_id Anthropic never saw
    assert all(block["type"] != "tool_result" for block in wire[0]["content"])
    assert "stray result" in wire[0]["content"][0]["text"]


def test_to_anthropic_messages_merges_user_turns_with_results_first():
    wire, _ = to_anthropic_messages(
        [
            LLMMessage(role="assistant", content="", tool_calls=[ToolCall(name="click", id="toolu_a")]),
            LLMMessage(role="tool", content="ok", tool_call_id="toolu_a"),
            LLMMessage(role="user", content="now scroll"),
        ]
    )
    assert len(wire) == 2
    kinds = [block["type"] for block in wire[1]["content"]]
    assert kinds == ["tool_result", "text"]  # tool results must come first


# ======================================================================== #
# 2. tolerance helpers
# ======================================================================== #

@pytest.mark.parametrize(
    "raw, expected, has_note",
    [
        ('{"a": 1}', {"a": 1}, False),
        ("{}", {}, False),
        ({'a': 2}, {"a": 2}, False),
        (None, {}, False),
        ("", {}, False),
        ('{"element_id": 3, ', {}, True),          # truncated JSON
        ("not json at all", {}, True),
        ('"[1, 2]"', {}, True),                     # valid JSON, wrong shape
        ('"{\\"a\\": 1}"', {"a": 1}, False),        # double-encoded object
        ("[1, 2, 3]", {}, True),
    ],
)
def test_parse_tool_arguments_never_raises(raw, expected, has_note):
    args, note = parse_tool_arguments(raw)
    assert args == expected
    assert (note is not None) is has_note


def test_parse_tool_arguments_recovers_truncated_object():
    args, note = parse_tool_arguments('{"element_id": 3, "intent": ')
    assert note is None or "recovered" in note or "could not parse" in note
    assert isinstance(args, dict)


def test_tool_calls_from_text_handles_fences_tags_arrays_and_flat_args():
    fenced = tool_calls_from_text('sure!\n```json\n{"name": "click", "arguments": {"element_id": 3}}\n```')
    assert [(call.name, call.args) for call in fenced] == [("click", {"element_id": 3})]
    assert tool_call_source(fenced[0]) == "text"

    tagged = tool_calls_from_text('<tool_call>{"tool": "type_text", "args": {"element_id": 4, "text": "hi"}}</tool_call>')
    assert tagged[0].name == "type_text" and tagged[0].args["text"] == "hi"

    array = tool_calls_from_text('[{"name": "click", "arguments": {"element_id": 1}}, {"name": "click", "arguments": {"element_id": 2}}]')
    assert [call.args["element_id"] for call in array] == [1, 2]

    flat = tool_calls_from_text('{"name": "click", "element_id": 7, "intent": "x"}')
    assert flat[0].args == {"element_id": 7, "intent": "x"}

    wrapped = tool_calls_from_text('{"function": {"name": "click", "arguments": {"element_id": 9}}}')
    assert wrapped[0].args == {"element_id": 9}


def test_tool_calls_from_text_respects_known_names_and_ignores_prose():
    assert tool_calls_from_text('{"name": "delete_everything", "arguments": {}}', known_names=["click"]) == []
    assert tool_calls_from_text('The answer is {"answer": 42}.') == []
    assert tool_calls_from_text("no json here") == []
    # duplicates collapsed
    once = tool_calls_from_text('{"name": "click", "arguments": {"element_id": 3}}\n{"name": "click", "arguments": {"element_id": 3}}')
    assert len(once) == 1


# ======================================================================== #
# 3. OpenAI provider
# ======================================================================== #

def test_openai_maps_tool_calls_text_and_usage(server: CannedServer):
    server.set_script(
        Canned(200, openai_payload(
            tool_calls=[openai_call("click", '{"element_id": 3, "intent": "open"}', "call_a")],
        ))
    )
    llm = openai_provider(server)
    response = llm.complete(USER, TOOLS, system="sys", temperature=0.2, max_tokens=256)

    assert isinstance(response, LLMResponse)
    assert response.text == "on it"
    assert response.stop_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "click" and call.id == "call_a"
    assert call.args == {"element_id": 3, "intent": "open"}
    assert tool_call_parse_error(call) is None
    assert response.usage == Usage(120, 18, 64, 1)
    # accumulated on the instance as well
    assert llm.usage.total == 138 and llm.usage.cached_tokens == 64
    assert llm.request_count == 1 and llm.retries == 0
    assert llm.count_tokens("hello world") > 0
    llm.close()


def test_openai_request_shape_is_openai_compatible(server: CannedServer):
    server.set_script(Canned(200, openai_payload()))
    llm = openai_provider(server)
    llm.complete(
        [
            LLMMessage(role="user", content="click it"),
            LLMMessage(role="assistant", content="", tool_calls=[ToolCall(name="click", args={"element_id": 3}, id="call_1")]),
            LLMMessage(role="tool", content="OK: clicked", tool_call_id="call_1"),
        ],
        TOOLS,
        system="you are webpilot",
    )
    body = server.last_body()
    assert body["model"] == "gpt-test"
    assert body["tool_choice"] == "auto"
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 2048  # Config.max_output_tokens
    assert [tool["function"]["name"] for tool in body["tools"]] == ["click", "type_text"]
    assert body["tools"][0]["type"] == "function"
    assert body["messages"][0] == {"role": "system", "content": "you are webpilot"}
    assert body["messages"][-1]["role"] == "tool" and body["messages"][-1]["tool_call_id"] == "call_1"
    assert server.headers()["authorization"] == "Bearer test-key"
    assert server.requests[0]["path"].endswith("/chat/completions")


def test_openai_omits_tools_when_none_offered(server: CannedServer):
    server.set_script(Canned(200, openai_payload(tool_calls=[], finish_reason="stop")))
    llm = openai_provider(server)
    llm.complete(USER, [], system=None)
    body = server.last_body()
    assert "tools" not in body and "tool_choice" not in body
    assert body["messages"] == [{"role": "user", "content": "book the cheapest flight"}]


def test_openai_malformed_arguments_do_not_raise(server: CannedServer):
    server.set_script(
        Canned(200, openai_payload(
            tool_calls=[
                openai_call("click", '{"element_id": 3, ', "call_bad"),
                openai_call("type_text", '{"element_id": 4, "text": "hi"}', "call_ok"),
            ]
        ))
    )
    llm = openai_provider(server)
    response = llm.complete(USER, TOOLS)  # must not raise

    bad, good = response.tool_calls
    assert bad.name == "click" and bad.id == "call_bad"
    assert bad.args == {}
    assert tool_call_parse_error(bad) is not None
    assert "element_id" in (tool_call_raw_arguments(bad) or "")
    assert good.args == {"element_id": 4, "text": "hi"}
    assert tool_call_parse_error(good) is None


def test_openai_retries_429_once_then_succeeds(server: CannedServer):
    server.set_script(
        Canned(429, OPENAI_ERROR),
        Canned(200, openai_payload(tool_calls=[openai_call("click", "{}", "call_1")])),
    )
    sleeps: list[float] = []
    llm = openai_provider(server, llm_max_retries=3, request_timeout=0.15, backoff_base=0.01, retry_jitter=0.0)
    llm._sleep = sleeps.append

    response = llm.complete(USER, TOOLS)

    assert response.tool_calls[0].name == "click"
    assert server.count == 2          # exactly one retry, not the whole budget
    assert llm.retries == 1 and llm.request_count == 2
    assert sleeps == pytest.approx([0.01])   # and it actually backed off
    assert llm.usage.calls == 1       # a retried failure is not usage


def test_openai_gives_up_after_exactly_max_retries(server: CannedServer):
    server.set_script(Canned(429, OPENAI_ERROR))  # the script repeats its last entry
    sleeps: list[float] = []
    llm = openai_provider(server, llm_max_retries=2, backoff_base=0.01, retry_jitter=0.0)
    llm._sleep = sleeps.append

    with pytest.raises(LLMError) as excinfo:
        llm.complete(USER, TOOLS)

    error = excinfo.value
    assert error.status == 429 and error.retryable is True
    assert "429" in str(error)
    assert server.count == 3          # 1 try + 2 retries: no more, no less
    assert llm.retries == 2
    # exponential backoff: 0.01, 0.02 (retry_jitter disabled for the assertion)
    assert sleeps == pytest.approx([0.01, 0.02])


def test_openai_backoff_adds_jitter_without_exceeding_double(server: CannedServer):
    server.set_script(Canned(429, OPENAI_ERROR))
    seen: list[float] = []
    llm = openai_provider(server, llm_max_retries=1)
    llm.backoff_base, llm.retry_jitter = 1.0, 0.5
    llm._sleep = seen.append

    with pytest.raises(LLMError):
        llm.complete(USER, TOOLS)

    assert len(seen) == 1
    assert 1.0 <= seen[0] <= 1.5      # base + up to 50% jitter


def test_openai_retries_5xx(server: CannedServer):
    server.set_script(Canned(503, {"error": {"message": "overloaded"}}), Canned(200, openai_payload()))
    llm = openai_provider(server, llm_max_retries=2)
    llm._sleep = lambda _delay: None

    llm.complete(USER, TOOLS)
    assert server.count == 2


def test_openai_400_fails_fast_without_retrying(server: CannedServer):
    server.set_script(Canned(400, {"error": {"message": "tools are not supported", "type": "invalid_request_error"}}))
    llm = openai_provider(server, llm_max_retries=3)
    llm._sleep = lambda _delay: pytest.fail("must not back off on a 400")

    with pytest.raises(LLMError) as excinfo:
        llm.complete(USER, TOOLS)

    error = excinfo.value
    assert error.status == 400
    assert error.retryable is False
    assert "400" in str(error)
    assert server.count == 1          # no retry at all


def test_openai_401_fails_fast(server: CannedServer):
    server.set_script(Canned(401, {"error": {"message": "bad key"}}))
    llm = openai_provider(server, llm_max_retries=3)
    with pytest.raises(LLMError) as excinfo:
        llm.complete(USER, TOOLS)
    assert excinfo.value.status == 401 and excinfo.value.retryable is False
    assert server.count == 1


def test_openai_timeout_is_normalised_to_a_retryable_llm_error(server: CannedServer):
    server.set_script(Canned(200, openai_payload(), delay=1.2))
    llm = openai_provider(server, request_timeout=0.25, llm_max_retries=0)

    with pytest.raises(LLMError) as excinfo:
        llm.complete(USER, TOOLS)

    error = excinfo.value
    assert error.retryable is True and error.status is None
    assert "timed out" in str(error)
    assert llm.request_count == 1


def test_openai_timeout_is_retried(server: CannedServer):
    server.set_script(Canned(200, openai_payload(), delay=1.2), Canned(200, openai_payload(), delay=1.2))
    llm = openai_provider(server, request_timeout=0.25, llm_max_retries=1)
    llm._sleep = lambda _delay: None

    with pytest.raises(LLMError):
        llm.complete(USER, TOOLS)
    assert llm.request_count == 2     # timed out, retried, timed out again


def test_openai_connection_error_is_retryable_and_actionable():
    llm = OpenAIProvider(
        Config(provider="openai", model="m", api_key="k", base_url="http://127.0.0.1:1/v1",
               request_timeout=2.0, llm_max_retries=1),
        backoff_base=0.0,
    )
    llm._sleep = lambda _delay: None

    with pytest.raises(LLMError) as excinfo:
        llm.complete(USER, TOOLS)

    error = excinfo.value
    assert error.retryable is True and error.status is None
    assert "127.0.0.1:1" in str(error)
    assert llm.request_count == 2     # the retry also happened


def test_openai_empty_choices_raises_retryable_error(server: CannedServer):
    server.set_script(Canned(200, {"id": "x", "choices": []}))
    llm = openai_provider(server, llm_max_retries=0)
    with pytest.raises(LLMError) as excinfo:
        llm.complete(USER, TOOLS)
    assert excinfo.value.retryable is True


def test_openai_usage_accumulates_across_calls(server: CannedServer):
    server.set_script(
        Canned(200, openai_payload(tool_calls=[], finish_reason="stop")),
        Canned(200, openai_payload(tool_calls=[], finish_reason="stop", usage={"prompt_tokens": 10, "completion_tokens": 5})),
    )
    llm = openai_provider(server)
    first = llm.complete(USER, TOOLS)
    second = llm.complete(USER, TOOLS)

    assert first.usage.prompt_tokens == 120 and second.usage.prompt_tokens == 10
    assert llm.usage == Usage(130, 23, 64, 2)
    assert llm.count_tokens("") == 0


def test_openai_reads_deepseek_style_cache_hits(server: CannedServer):
    server.set_script(Canned(200, openai_payload(usage={
        "prompt_tokens": 50, "completion_tokens": 5, "prompt_cache_hit_tokens": 32,
    })))
    llm = OpenAIProvider(
        Config(provider="deepseek", model="deepseek-chat", api_key="k", base_url=server.base_url,
               request_timeout=5.0, llm_max_retries=0)
    )
    response = llm.complete(USER, TOOLS)
    assert response.usage.cached_tokens == 32
    assert llm.provider == "deepseek"          # provider name comes from the config
    # without an override the DeepSeek endpoint from config.PROVIDERS is used
    assert OpenAIProvider(Config(provider="deepseek", api_key="k")).base_url == "https://api.deepseek.com/v1"


# ======================================================================== #
# 4. Anthropic provider
# ======================================================================== #

def test_anthropic_maps_tool_use_blocks_and_usage(server: CannedServer):
    server.set_script(Canned(200, anthropic_payload(blocks=[
        {"type": "text", "text": "let me click"},
        {"type": "tool_use", "id": "toolu_1", "name": "click", "input": {"element_id": 3}},
    ])))
    llm = anthropic_provider(server)
    response = llm.complete(USER, TOOLS, system="sys")

    assert response.text == "let me click"
    assert response.stop_reason == "tool_use"
    call = response.tool_calls[0]
    assert (call.name, call.args, call.id) == ("click", {"element_id": 3}, "toolu_1")
    assert response.usage == Usage(500, 40, 128, 1)
    assert llm.usage.cached_tokens == 128
    llm.close()


def test_anthropic_multiple_tool_use_blocks_keep_order_and_ids(server: CannedServer):
    server.set_script(Canned(200, anthropic_payload(blocks=[
        {"type": "tool_use", "id": "toolu_a", "name": "click", "input": {"element_id": 1}},
        {"type": "tool_use", "id": "toolu_b", "name": "type_text", "input": {"element_id": 2, "text": "hi"}},
    ])))
    llm = anthropic_provider(server)
    response = llm.complete(USER, TOOLS)
    assert [call.id for call in response.tool_calls] == ["toolu_a", "toolu_b"]
    assert response.text == ""


def test_anthropic_request_shape(server: CannedServer):
    server.set_script(Canned(200, anthropic_payload()))
    llm = anthropic_provider(server)
    llm.complete(
        [
            LLMMessage(role="user", content="click it"),
            LLMMessage(role="assistant", content="", tool_calls=[ToolCall(name="click", args={"element_id": 3}, id="toolu_a")]),
            LLMMessage(role="tool", content="OK: clicked", tool_call_id="toolu_a"),
        ],
        TOOLS,
        system="you are webpilot",
        temperature=0.3,
        max_tokens=512,
    )
    body = server.last_body()
    assert body["model"] == "claude-test"
    assert body["system"] == "you are webpilot"          # top-level, not a message
    assert body["max_tokens"] == 512
    assert body["tool_choice"] == {"type": "auto"}
    assert body["tools"][0]["input_schema"]["type"] == "object"   # Anthropic schema key
    assert "strict" not in body["tools"][0]                       # no OpenAI-ism leaks
    assert [entry["role"] for entry in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][1]["content"][0]["type"] == "tool_use"
    assert body["messages"][1]["content"][0]["id"] == "toolu_a"
    assert body["messages"][2]["content"][0]["tool_use_id"] == "toolu_a"
    assert server.requests[0]["headers"]["anthropic-version"]
    assert server.requests[0]["headers"]["x-api-key"] == "test-key"
    # temperature is only sent when the installed SDK's Messages API takes it
    assert ("temperature" in body) is llm.supports_temperature()


def test_anthropic_sends_temperature_when_the_sdk_supports_it(server: CannedServer, monkeypatch: pytest.MonkeyPatch):
    """The kwarg is built when supported; the installed SDK rejects it, so we
    assert on the request builder rather than sending it."""
    monkeypatch.setattr(AnthropicProvider, "_temperature_supported", True)
    monkeypatch.setattr(AnthropicProvider, "supports_temperature", lambda self: True)
    llm = anthropic_provider(server)
    request = llm.build_request(USER, TOOLS, system=None, temperature=0.3, max_tokens=64)
    assert request["temperature"] == 0.3
    assert not server.requests          # nothing was sent


def test_anthropic_omits_temperature_when_the_sdk_dropped_it(server: CannedServer, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(AnthropicProvider, "supports_temperature", lambda self: False)
    server.set_script(Canned(200, anthropic_payload()))
    llm = anthropic_provider(server)
    llm.complete(USER, TOOLS, temperature=0.3)
    body = server.last_body()
    assert "temperature" not in body      # an unsupported kwarg would be a TypeError
    assert body["messages"] and body["max_tokens"] > 0


def test_anthropic_temperature_probe_matches_the_installed_sdk():
    from webpilot.llm.anthropic_provider import sdk_accepts_temperature

    probe = sdk_accepts_temperature()
    assert isinstance(probe, bool)
    assert AnthropicProvider(Config(provider="anthropic", api_key="k")).supports_temperature() is probe


def test_anthropic_omits_system_and_tools_when_absent(server: CannedServer):
    server.set_script(Canned(200, anthropic_payload(blocks=[{"type": "text", "text": "hi"}], stop_reason="end_turn")))
    llm = anthropic_provider(server)
    llm.complete(USER, [], system=None)
    body = server.last_body()
    assert "system" not in body and "tools" not in body and "tool_choice" not in body
    assert body["messages"] == [{"role": "user", "content": "book the cheapest flight"}]


def test_anthropic_tolerant_tool_use_input(server: CannedServer):
    server.set_script(Canned(200, anthropic_payload(blocks=[
        {"type": "tool_use", "id": "toolu_str", "name": "click", "input": '{"element_id": 8, "intent": "go"}'},
        {"type": "tool_use", "id": "toolu_none", "name": "type_text", "input": None},
        {"type": "tool_use", "id": "toolu_bad", "name": "click", "input": '{"element_id": '},
    ])))
    llm = anthropic_provider(server)
    response = llm.complete(USER, TOOLS)          # must not raise

    string_input, none_input, bad_input = response.tool_calls
    assert string_input.args == {"element_id": 8, "intent": "go"}
    assert none_input.args == {}
    assert bad_input.args == {}
    assert tool_call_parse_error(bad_input) is not None
    assert "element_id" in (tool_call_raw_arguments(bad_input) or "")


def test_anthropic_retries_429_then_succeeds(server: CannedServer):
    server.set_script(Canned(429, ANTHROPIC_ERROR), Canned(200, anthropic_payload()))
    llm = anthropic_provider(server, llm_max_retries=2, backoff_base=0.01, retry_jitter=0.0)
    sleeps: list[float] = []
    llm._sleep = sleeps.append

    response = llm.complete(USER, TOOLS)

    assert response.tool_calls[0].name == "click"
    assert server.count == 2 and llm.retries == 1
    assert sleeps == pytest.approx([0.01])


def test_anthropic_400_is_not_retryable(server: CannedServer):
    server.set_script(Canned(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "bad schema"}}))
    llm = anthropic_provider(server, llm_max_retries=3)
    llm._sleep = lambda _delay: pytest.fail("must not retry a 400")

    with pytest.raises(LLMError) as excinfo:
        llm.complete(USER, TOOLS)

    assert excinfo.value.status == 400 and excinfo.value.retryable is False
    assert server.count == 1


# ======================================================================== #
# 5. Local provider
# ======================================================================== #

def test_local_provider_defaults_need_no_key():
    llm = LocalProvider(Config(provider="ollama", model="qwen3:8b"))
    assert llm.base_url == "http://localhost:11434/v1"
    assert llm.api_key == PLACEHOLDER_API_KEY      # dummy key, server ignores it
    assert llm.provider == "ollama"
    assert Config(provider="ollama").needs_key is False

    # also constructible with nothing at all
    bare = LocalProvider()
    assert bare.base_url == "http://localhost:11434/v1" and bare.api_key


def test_local_provider_keeps_a_configured_key_and_endpoint():
    llm = LocalProvider(Config(provider="openai_compatible", api_key="proxy-token",
                               base_url="http://127.0.0.1:8000/v1"))
    assert llm.api_key == "proxy-token"                    # authenticating gateways exist
    assert llm.base_url == "http://127.0.0.1:8000/v1"


def test_local_provider_posts_the_dummy_key(server: CannedServer):
    server.set_script(Canned(200, openai_payload(tool_calls=[], finish_reason="stop", content="hello")))
    llm = local_provider(server)
    llm.complete(USER, TOOLS)
    assert server.headers()["authorization"] == f"Bearer {PLACEHOLDER_API_KEY}"


def test_local_provider_parses_tool_call_emitted_as_content(server: CannedServer):
    server.set_script(Canned(200, openai_payload(
        content='Of course!\n```json\n{"name": "click", "arguments": {"element_id": 3, "intent": "open"}}\n```',
        tool_calls=[], finish_reason="stop",
    )))
    llm = local_provider(server)
    response = llm.complete(USER, TOOLS)

    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "click" and call.args == {"element_id": 3, "intent": "open"}
    assert call.id and tool_call_source(call) == "text"
    assert response.stop_reason == "tool_calls"   # normalised so the loop dispatches it


def test_local_provider_parses_json_array_of_calls_and_tags(server: CannedServer):
    server.set_script(Canned(200, openai_payload(
        content='<tool_call>{"tool": "type_text", "args": {"element_id": 4, "text": "hi"}}</tool_call>\n'
                '[{"name": "click", "arguments": {"element_id": 5}}]',
        tool_calls=[], finish_reason="stop",
    )))
    llm = local_provider(server)
    response = llm.complete(USER, TOOLS)
    assert [call.name for call in response.tool_calls] == ["type_text", "click"]
    assert response.tool_calls[1].args == {"element_id": 5}


def test_local_provider_ignores_tool_calls_for_unoffered_tools(server: CannedServer):
    server.set_script(Canned(200, openai_payload(
        content='{"name": "delete_account", "arguments": {}}',
        tool_calls=[], finish_reason="stop",
    )))
    llm = local_provider(server)
    response = llm.complete(USER, TOOLS)
    assert response.tool_calls == []
    assert response.text == '{"name": "delete_account", "arguments": {}}'


def test_local_provider_is_silent_when_no_tools_were_offered(server: CannedServer):
    """A JSON object in the text is *not* a tool call when there were no tools."""
    server.set_script(Canned(200, openai_payload(
        content='result: {"name": "click", "arguments": {"element_id": 1}}',
        tool_calls=[], finish_reason="stop",
    )))
    llm = local_provider(server)
    response = llm.complete(USER, [])          # extractor sub-agent style call
    assert response.tool_calls == []
    assert response.text.startswith("result:")


def test_local_provider_ignores_plain_prose_json(server: CannedServer):
    server.set_script(Canned(200, openai_payload(
        content='The price is {"amount": 42, "currency": "EUR"}.',
        tool_calls=[], finish_reason="stop",
    )))
    llm = local_provider(server)
    assert llm.complete(USER, TOOLS).tool_calls == []


def test_local_provider_empty_response_is_a_retryable_error(server: CannedServer):
    server.set_script(Canned(200, openai_payload(content="", tool_calls=[], finish_reason="stop")))
    llm = local_provider(server, llm_max_retries=0)
    with pytest.raises(LLMError) as excinfo:
        llm.complete(USER, TOOLS)
    assert excinfo.value.retryable is True
    assert "empty response" in str(excinfo.value)


def test_local_provider_keeps_provider_tool_calls_untouched(server: CannedServer):
    server.set_script(Canned(200, openai_payload(tool_calls=[openai_call("click", '{"element_id": 1}', "call_p")])))
    llm = local_provider(server)
    response = llm.complete(USER, TOOLS)
    assert response.tool_calls[0].id == "call_p"       # no text fallback kicked in
    assert tool_call_source(response.tool_calls[0]) == "provider"


# ======================================================================== #
# 6. FakeLLM
# ======================================================================== #

def test_fake_llm_import_path_and_basic_shape():
    from webpilot.llm.fake import FakeLLM as Direct

    assert Direct is FakeLLM
    llm = FakeLLM()
    assert llm.provider == "fake" and llm.model == "fake-1"
    assert llm.count_tokens("some text") > 0
    llm.close()


def test_fake_llm_returns_scripted_responses_in_order_and_repeats_the_last():
    script = [
        LLMResponse(text="one", tool_calls=[ToolCall(name="click", args={"element_id": 1}, id="c1")], stop_reason="tool_calls"),
        LLMResponse(text="two", tool_calls=[ToolCall(name="finish", args={"success": True, "answer": "done"})]),
    ]
    llm = FakeLLM(script)

    assert llm.complete(USER, TOOLS).text == "one"
    assert llm.complete(USER, TOOLS).text == "two"
    assert llm.complete(USER, TOOLS).text == "two"      # last one repeats
    assert llm.calls == 3


def test_fake_llm_records_requests_and_accumulates_usage():
    llm = FakeLLM([LLMResponse(text="ok", usage=Usage(10, 2, 0, 1))])
    llm.complete(USER, TOOLS, system="sys", temperature=0.4, max_tokens=99)

    request = llm.last_request()
    assert request is not None
    assert request["system"] == "sys"
    assert request["temperature"] == 0.4 and request["max_tokens"] == 99
    assert request["messages"][0].content == "book the cheapest flight"
    assert [tool.name for tool in request["tools"]] == ["click", "type_text"]
    assert llm.usage == Usage(10, 2, 0, 1)
    assert llm.responses[0].usage.calls == 1


def test_fake_llm_responder_can_drive_the_conversation():
    seen: list[Any] = []

    def responder(messages, tools):
        seen.append((len(messages), [tool.name for tool in tools]))
        if len(messages) < 2:
            return LLMResponse(tool_calls=[ToolCall(name="click", args={"element_id": 2}, id="c1")])
        return LLMResponse(tool_calls=[ToolCall(name="finish", args={"success": True, "answer": "ok"})])

    llm = FakeLLM(responder=responder)
    first = llm.complete([LLMMessage(role="user", content="go")], TOOLS)
    second = llm.complete([LLMMessage(role="user", content="go"), LLMMessage(role="assistant", tool_calls=first.tool_calls)], TOOLS)

    assert first.tool_calls[0].name == "click"
    assert second.tool_calls[0].name == "finish"
    assert seen == [(1, ["click", "type_text"]), (2, ["click", "type_text"])]


def test_fake_llm_three_arg_responder_receives_the_system_prompt():
    captured: dict[str, Any] = {}

    def responder(messages, tools, system):
        captured["system"] = system
        return LLMResponse(text="ok")

    FakeLLM(responder=responder).complete(USER, TOOLS, system="you are webpilot")
    assert captured["system"] == "you are webpilot"


def test_fake_llm_without_a_script_finishes_instead_of_hanging():
    response = FakeLLM().complete(USER, TOOLS)
    assert response.tool_calls[0].name == "finish"
    assert response.tool_calls[0].args["success"] is True


def test_fake_llm_reset_and_add():
    llm = FakeLLM([LLMResponse(text="a")])
    llm.complete(USER, TOOLS)
    llm.add(LLMResponse(text="b"))
    llm.reset()
    assert llm.requests == [] and llm.usage.total == 0
    assert llm.complete(USER, TOOLS).text == "a"


def test_fake_llm_never_touches_the_network(monkeypatch: pytest.MonkeyPatch):
    import openai
    import anthropic

    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: pytest.fail("FakeLLM must stay offline"))
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: pytest.fail("FakeLLM must stay offline"))

    llm = FakeLLM([LLMResponse(text="ok", tool_calls=[ToolCall(name="finish", args={})])])
    assert llm.complete(USER, TOOLS).text == "ok"
    assert llm.usage.calls == 1


# ======================================================================== #
# 7. conftest contract
# ======================================================================== #

def test_conftest_fake_llm_fixture_points_at_our_class(fake_llm):
    assert fake_llm is FakeLLM
    assert fake_llm([LLMResponse(text="x")]).complete(USER, TOOLS).text == "x"
