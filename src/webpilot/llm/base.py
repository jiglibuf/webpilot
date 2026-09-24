"""``BaseLLM`` - the shared machinery every provider builds on.

The provider layer has one job: turn *one* uniform tool-calling interface
(``LLMClient`` from :mod:`webpilot.types`) into whatever dialect a concrete API
speaks.  Everything that is *not* dialect specific lives here:

* **retries** - exponential backoff with jitter on 429 / 5xx / timeouts, bounded
  by ``Config.llm_max_retries``; every other failure fails fast (a 400 or a 401
  will not get better by trying again);
* **error normalisation** - provider exceptions (``openai.RateLimitError``,
  ``anthropic.APITimeoutError``, plain socket errors, ...) become one
  :class:`webpilot.errors.LLMError` carrying ``status`` and ``retryable``, so the
  agent loop only ever catches one exception type;
* **timeouts** - ``Config.request_timeout`` is applied per request, both on the
  SDK client and per call;
* **usage accumulation** - each call returns its own :class:`Usage`; the total
  spend is accumulated on the instance (``llm.usage``);
* **token counting** - delegates to :func:`webpilot.tokenizer.count_tokens`;
* **message conversion** - :meth:`BaseLLM.to_openai_messages` and
  :meth:`BaseLLM.to_anthropic_messages` translate the uniform ``LLMMessage``
  list (including ``tool_call_id`` round-tripping) into each wire format.  The
  two are deliberately separate methods: OpenAI keeps tool results as their own
  ``tool`` messages keyed by ``tool_call_id``, while Anthropic has no tool role —
  results travel inside a *user* turn as ``tool_result`` blocks that must
  immediately follow the ``tool_use`` blocks of the previous assistant turn.

Subclasses implement :meth:`BaseLLM._complete_once` (one request, no retrying)
and :meth:`BaseLLM.normalise_error` (their SDK's exception hierarchy).
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from ..config import Config
from ..errors import LLMError
from ..tokenizer import count_tokens
from ..types import LLMMessage, LLMResponse, ToolCall, ToolSpec, Usage

__all__ = [
    "BaseLLM",
    "to_openai_messages",
    "to_anthropic_messages",
    "parse_tool_arguments",
    "iter_json_objects",
    "tool_calls_from_text",
    "set_tool_call_diagnostics",
    "tool_call_parse_error",
    "tool_call_raw_arguments",
    "tool_call_source",
    "is_retryable_status",
    "RETRYABLE_STATUS_CODES",
    "DEFAULT_BACKOFF_BASE",
    "DEFAULT_BACKOFF_MAX",
    "DEFAULT_RETRY_JITTER",
]

#: Retry policy defaults (seconds).  ``backoff_base * 2**attempt`` capped by
#: ``backoff_max``, plus up to ``retry_jitter`` * delay of random jitter so that
#: parallel agents do not hammer a rate-limited endpoint in lockstep.
DEFAULT_BACKOFF_BASE = 1.5
DEFAULT_BACKOFF_MAX = 30.0
DEFAULT_RETRY_JITTER = 0.5

#: Statuses worth a second attempt: rate limits, request timeouts, server errors.
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504, 507, 529})

#: The official SDKs refuse to build a client without *some* credential string,
#: so local/unauthenticated endpoints get this placeholder.  Real key validation
#: lives in :func:`webpilot.llm.registry.build_llm`.
PLACEHOLDER_API_KEY = "webpilot-none"


def is_retryable_status(status: int | None) -> bool:
    """True for 429, 408 and every 5xx (the documented retry contract)."""
    if status is None:
        return False
    return status in RETRYABLE_STATUS_CODES or 500 <= status < 600


# --------------------------------------------------------------------------- #
# JSON tolerance helpers
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"```[ \t]*(?:json|JSON|json5)?[ \t]*\r?\n?(.*?)```", re.DOTALL)
_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")

#: keys a local model may use for the tool name / the arguments object.
_NAME_KEYS = ("name", "tool", "tool_name", "toolName", "function", "action", "recipient")
_ARG_KEYS = ("arguments", "args", "parameters", "input", "kwargs", "parameters_json")


def iter_json_objects(text: str) -> Iterator[tuple[dict[str, Any], str]]:
    """Yield ``(object, source_text)`` for every JSON object embedded in ``text``.

    Scanning is tolerant on purpose: local models wrap their tool call in prose,
    in a markdown fence or in ``<tool_call>`` tags, and half of them emit a JSON
    array at the top level.
    """
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            value = json.loads(stripped)
        except ValueError:
            value = None
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    yield dict(item), json.dumps(item)
            return
    idx = 0
    while True:
        start = text.find("{", idx)
        if start < 0:
            return
        try:
            value, end = decoder.raw_decode(text[start:])
        except ValueError:
            idx = start + 1
            continue
        if isinstance(value, Mapping):
            yield dict(value), text[start : start + end]
            idx = start + end
        else:
            idx = start + 1


#: ``ToolCall`` (defined in ``webpilot.types``, which is frozen) has no field for
#: "the model sent us something we could not parse".  Rather than inventing a
#: parallel type we attach these two facts as attributes and expose them through
#: :func:`tool_call_parse_error` / :func:`tool_call_raw_arguments` so callers never
#: have to touch ``getattr`` themselves.
_PARSE_ERROR_ATTR = "parse_error"
_RAW_ARGS_ATTR = "raw_arguments"
_SOURCE_ATTR = "source"


def set_tool_call_diagnostics(
    call: ToolCall,
    *,
    parse_error: str | None = None,
    raw_arguments: str | None = None,
    source: str | None = None,
) -> ToolCall:
    """Attach diagnostics to a :class:`ToolCall` (see module docstring)."""
    if parse_error is not None:
        setattr(call, _PARSE_ERROR_ATTR, parse_error)
    if raw_arguments is not None:
        setattr(call, _RAW_ARGS_ATTR, raw_arguments)
    if source is not None:
        setattr(call, _SOURCE_ATTR, source)
    return call


def tool_call_parse_error(call: ToolCall) -> str | None:
    """Why the arguments could not be parsed, or ``None`` when they were fine."""
    return getattr(call, _PARSE_ERROR_ATTR, None)


def tool_call_raw_arguments(call: ToolCall) -> str | None:
    """The raw argument string exactly as the provider sent it."""
    return getattr(call, _RAW_ARGS_ATTR, None)


def tool_call_source(call: ToolCall) -> str | None:
    """``"text"`` when the call was recovered from prose rather than a tool field."""
    return getattr(call, _SOURCE_ATTR, None)


def parse_tool_arguments(raw: Any) -> tuple[dict[str, Any], str | None]:
    """Turn raw tool-call arguments into ``(args, parse_error_note)``.

    Never raises: a provider that sends truncated JSON (models truncate mid
    object all the time) yields ``({}, "<why>")`` so the caller can keep the raw
    string, flag it and let the agent loop decide what to do.
    """
    if isinstance(raw, Mapping):
        return dict(raw), None
    if raw is None:
        return {}, None
    text = str(raw).strip()
    if not text:
        return {}, None
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        for candidate, _src in iter_json_objects(text):
            if candidate:
                return candidate, f"arguments were not valid JSON ({exc}); recovered an embedded object"
        return {}, f"could not parse tool arguments as JSON: {exc}"
    if isinstance(value, Mapping):
        return dict(value), None
    if isinstance(value, str):
        # double-encoded arguments ("{\"a\": 1}" sent as a JSON string)
        return parse_tool_arguments(value)
    return {}, f"tool arguments must be a JSON object, got {type(value).__name__}"


def _candidate_to_call(raw: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Normalise one loose JSON object into ``(tool_name, arguments)`` or None."""
    name: str | None = None
    args: Any = None
    have_args = False
    for key in _NAME_KEYS:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            name = value.strip()
            break
        if isinstance(value, Mapping):  # {"function": {"name": ..., "arguments": ...}}
            inner = value
            inner_name = inner.get("name")
            if isinstance(inner_name, str) and inner_name.strip():
                name = inner_name.strip()
            for arg_key in _ARG_KEYS:
                if arg_key in inner:
                    args, have_args = inner[arg_key], True
                    break
            break
    if not name or not _NAME_RE.match(name):
        return None
    for key in _ARG_KEYS:
        if key in raw:
            args, have_args = raw[key], True
            break
    if not have_args:
        # flat arguments: {"name": "click", "element_id": 3}
        args = {k: v for k, v in raw.items() if k not in _NAME_KEYS and k != "id"}
    parsed, _note = parse_tool_arguments(args)
    return name, parsed


def tool_calls_from_text(
    text: str,
    *,
    known_names: Iterable[str] | None = None,
    call_id_prefix: str = "call_text",
) -> list[ToolCall]:
    """Recover tool calls a model emitted as *text* instead of via ``tool_calls``.

    Handles fenced blocks, ``<tool_call>`` tags, bare inline objects and JSON
    arrays of calls.  ``known_names`` (when given) restricts the accepted names
    to the tools actually offered, which is what keeps prose like
    ``{"answer": "12"}`` from being mistaken for an action.
    """
    text = text or ""
    allowed = set(known_names) if known_names is not None else None
    candidates: list[Mapping[str, Any]] = []
    seen_sources: set[str] = set()

    def _collect(block: str) -> None:
        for obj, src in iter_json_objects(block):
            if src in seen_sources:
                continue
            seen_sources.add(src)
            candidates.append(obj)

    for match in list(_FENCE_RE.finditer(text)) + list(_TAG_RE.finditer(text)):
        _collect(match.group(1))
    without_blocks = _FENCE_RE.sub(" ", _TAG_RE.sub(" ", text))
    _collect(without_blocks)

    calls: list[ToolCall] = []
    seen_calls: set[str] = set()
    for raw in candidates:
        parsed = _candidate_to_call(raw)
        if parsed is None:
            continue
        name, args = parsed
        if allowed is not None and name not in allowed:
            continue
        key = name + ":" + json.dumps(args, sort_keys=True, default=str)
        if key in seen_calls:
            continue
        seen_calls.add(key)
        raw_id = raw.get("id")
        call = ToolCall(
            name=name,
            args=args,
            id=raw_id if isinstance(raw_id, str) and raw_id else f"{call_id_prefix}_{len(calls)}",
        )
        set_tool_call_diagnostics(
            call,
            raw_arguments=json.dumps(args, ensure_ascii=False, default=str),
            source="text",
        )
        calls.append(call)
    return calls


# --------------------------------------------------------------------------- #
# Wire-format conversions
# --------------------------------------------------------------------------- #

def to_openai_messages(
    messages: Sequence[LLMMessage],
    system: str | None = None,
) -> list[dict[str, Any]]:
    """Convert uniform messages to Chat Completions ``messages``.

    Tool results stay their own ``{"role": "tool", "tool_call_id": ...}`` entries
    and assistant turns carry ``tool_calls`` - see
    :func:`to_anthropic_messages` for the very different Anthropic shape.
    """
    out: list[dict[str, Any]] = []
    if system and system.strip():
        out.append({"role": "system", "content": system})
    for message in messages:
        role = message.role
        if role == "system":
            if message.content and message.content.strip():
                out.append({"role": "system", "content": message.content})
            continue
        if role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id or f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.args or {}, ensure_ascii=False, default=str),
                        },
                    }
                    for index, call in enumerate(message.tool_calls)
                ]
            if message.name:
                entry["name"] = message.name
            out.append(entry)
            continue
        if role == "tool":
            entry = {"role": "tool", "content": message.content or ""}
            entry["tool_call_id"] = message.tool_call_id or ""
            if message.name:
                entry["name"] = message.name
            out.append(entry)
            continue
        # user (and anything unknown) - keep the text verbatim
        out.append({"role": "user", "content": message.content or ""})
    return out


def _merge_turn(out: list[dict[str, Any]], role: str, content: Any) -> None:
    """Append a turn, merging it into the previous one when the role repeats.

    Anthropic rejects empty text blocks and requires every ``tool_result`` block
    of a user turn to come *before* any prose, so merges keep that order (and
    several results for one assistant turn stay in a single user message).
    """
    if not content:
        return
    blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
    if not blocks:
        return
    if out and out[-1]["role"] == role:
        previous = out[-1]["content"]
        previous_blocks = previous if isinstance(previous, list) else (
            [{"type": "text", "text": previous}] if previous else []
        )
        if blocks[0].get("type") == "tool_result":
            out[-1]["content"] = blocks + previous_blocks
        else:
            out[-1]["content"] = previous_blocks + blocks
        return
    out.append({"role": role, "content": content if isinstance(content, list) else blocks[0]["text"]})


def to_anthropic_messages(
    messages: Sequence[LLMMessage],
    system: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Convert uniform messages to the Anthropic Messages shape.

    Returns ``(messages, system)`` - Anthropic takes the system prompt as a
    top-level parameter, not as a message.  Tool results are rebuilt the
    Anthropic way: the assistant turn keeps its ``tool_use`` blocks (with ids)
    and the *next* user turn carries the matching ``tool_result`` blocks, all in
    one message, keyed by ``tool_use_id``.
    """
    out: list[dict[str, Any]] = []
    system_parts: list[str] = []
    if system and system.strip():
        system_parts.append(system)
    pending_results: list[dict[str, Any]] = []   # tool_result blocks for the next user turn
    pending_orphans: list[str] = []              # results with no id: kept as plain text

    def flush_results() -> None:
        blocks = list(pending_results)
        if pending_orphans:
            blocks.append({"type": "text", "text": "\n".join(pending_orphans)})
        if blocks:
            _merge_turn(out, "user", blocks)
        pending_results.clear()
        pending_orphans.clear()

    for index, message in enumerate(messages):
        role = message.role
        if role == "system":
            if message.content and message.content.strip():
                system_parts.append(message.content)
            continue
        if role == "tool":
            if not message.tool_call_id:
                # A tool result without an id cannot be attributed to a tool_use
                # block, and Anthropic rejects unknown ids - so it travels as text.
                pending_orphans.append(f"[tool result with no tool_call_id] {message.content or ''}")
            else:
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content or "",
                }
                if message.meta.get("is_error"):
                    block["is_error"] = True
                pending_results.append(block)
            continue
        flush_results()
        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            if message.content and message.content.strip():
                blocks.append({"type": "text", "text": message.content})
            for call_index, call in enumerate(message.tool_calls):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id or f"toolu_{index}_{call_index}",
                        "name": call.name,
                        "input": call.args or {},
                    }
                )
            if blocks:
                _merge_turn(out, "assistant", blocks)
            continue
        if message.content and message.content.strip():
            _merge_turn(out, "user", message.content)
    flush_results()
    merged_system = "\n\n".join(part for part in system_parts if part.strip()) or None
    return out, merged_system


# --------------------------------------------------------------------------- #
# BaseLLM
# --------------------------------------------------------------------------- #

class BaseLLM:
    """Retry/timeout/usage/conversion harness shared by all providers.

    Subclasses must set :attr:`provider` (a short name) and implement
    :meth:`_complete_once` plus, when their SDK has its own exceptions,
    :meth:`normalise_error`.
    """

    #: value of ``LLMClient.provider`` when no config says otherwise.
    provider: str = "base"
    #: provider name used to build a throwaway :class:`Config` when none is given.
    _default_provider: str = "fake"

    def __init__(
        self,
        config: Config | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        request_timeout: float | None = None,
        max_retries: int | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        backoff_max: float = DEFAULT_BACKOFF_MAX,
        retry_jitter: float = DEFAULT_RETRY_JITTER,
    ) -> None:
        if config is None:
            config = Config(
                provider=provider or self._default_provider,
                model=model or "",
                api_key=api_key,
                base_url=base_url,
            )
        overrides: dict[str, Any] = {}
        if model:
            overrides["model"] = model
        if api_key is not None:
            overrides["api_key"] = api_key
        if base_url:
            overrides["base_url"] = base_url
        if request_timeout is not None:
            overrides["request_timeout"] = float(request_timeout)
        if max_retries is not None:
            overrides["llm_max_retries"] = int(max_retries)
        if max_output_tokens is not None:
            overrides["max_output_tokens"] = int(max_output_tokens)
        if overrides:
            config = config.copy_with(**overrides)

        self.config = config
        self.provider = provider or config.provider
        self.model = config.model
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.request_timeout = float(config.request_timeout)
        self.max_retries = max(0, int(config.llm_max_retries))
        self.temperature = float(config.temperature if temperature is None else temperature)
        self.max_output_tokens = int(config.max_output_tokens)

        self.backoff_base = float(backoff_base)
        self.backoff_max = float(backoff_max)
        self.retry_jitter = float(retry_jitter)

        #: cumulative usage across every successful call (see ``Usage.add``)
        self.usage = Usage()
        #: number of HTTP attempts made, including the ones that were retried
        self.request_count = 0
        self.retries = 0
        #: injectable sleep, so tests can observe the backoff without waiting
        self._sleep: Callable[[float], None] = time.sleep

    # ---------------------------------------------------------------- public

    def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSpec] | None = None,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Run one chat completion, retrying transient failures.

        Returns the provider's :class:`LLMResponse`; the per-call usage is also
        accumulated into :attr:`usage`.
        """
        msgs = list(messages or [])
        specs = list(tools or [])
        temp = self.temperature if temperature is None else float(temperature)
        limit = int(max_tokens or self.max_output_tokens)
        response = self._with_retries(
            lambda: self._complete_once(msgs, specs, system=system, temperature=temp, max_tokens=limit)
        )
        if not response.usage.calls:
            response.usage.calls = 1
        self.usage = self.usage.add(response.usage)
        return response

    def count_tokens(self, text: str) -> int:
        """Token estimate, delegated to :mod:`webpilot.tokenizer`."""
        return count_tokens(text)

    def close(self) -> None:
        """Release the underlying SDK client, if one was created."""
        self._close_client()

    def __enter__(self) -> "BaseLLM":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ------------------------------------------------------- message helpers

    def to_openai_messages(
        self,
        messages: Sequence[LLMMessage],
        system: str | None = None,
    ) -> list[dict[str, Any]]:
        """Shared helper: uniform messages -> Chat Completions wire format."""
        return to_openai_messages(messages, system)

    def to_anthropic_messages(
        self,
        messages: Sequence[LLMMessage],
        system: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Shared helper: uniform messages -> Anthropic wire format.

        Deliberately a *separate* method from :meth:`to_openai_messages`: the two
        APIs disagree about how tool results are represented.
        """
        return to_anthropic_messages(messages, system)

    # ------------------------------------------------------------ retry core

    def _with_retries(self, call: Callable[[], LLMResponse]) -> LLMResponse:
        """Call ``call`` up to ``1 + max_retries`` times on retryable errors."""
        attempts = self.max_retries + 1
        last: LLMError | None = None
        for attempt in range(attempts):
            self.request_count += 1
            try:
                return call()
            except Exception as exc:  # normalised below - nothing escapes raw
                error = self.normalise_error(exc)
                last = error
                if not error.retryable or attempt + 1 >= attempts:
                    if error is exc:
                        raise
                    raise error from exc
                delay = self._retry_delay(attempt)
                self.retries += 1
                if delay > 0:
                    self._sleep(delay)
        raise last or LLMError(f"{self.provider}: call failed", retryable=False)

    def _retry_delay(self, attempt: int) -> float:
        """``backoff_base * 2**attempt`` (capped) plus up to ``retry_jitter``."""
        delay = min(self.backoff_max, self.backoff_base * (2.0 ** attempt))
        if delay <= 0 or self.retry_jitter <= 0:
            return max(0.0, delay)
        return delay * (1.0 + random.uniform(0.0, self.retry_jitter))

    # ------------------------------------------------------------ overriding

    def _complete_once(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSpec],
        *,
        system: str | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        raise NotImplementedError

    def normalise_error(self, exc: BaseException) -> LLMError:
        """Map any exception onto :class:`LLMError` (``status``/``retryable``)."""
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, TimeoutError):
            return LLMError(
                f"{self.provider}: request timed out after {self.request_timeout}s: {exc}",
                status=None,
                retryable=True,
            )
        if isinstance(exc, (ConnectionError, OSError)):
            return LLMError(f"{self.provider}: connection failed: {exc}", status=None, retryable=True)
        return LLMError(f"{self.provider}: {type(exc).__name__}: {exc}", status=None, retryable=False)

    def _close_client(self) -> None:
        """Hook: close the SDK client (no-op when the provider has none)."""
