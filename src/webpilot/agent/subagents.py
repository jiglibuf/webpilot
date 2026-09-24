"""Sub-agents: page reading, recovery advice and history compaction.

All three share the same contract:

* they run on their **own message list** and never see the main conversation;
* they always report a ``Usage`` so the run telemetry can attribute token spend;
* they **degrade gracefully** - a broken model, a malformed answer or a missing
  LLM produces a conservative result (an empty answer, generic advice, or a
  deterministic summary) instead of an exception the loop would have to handle.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Sequence

from ..config import Config
from ..types import (
    ExtractionResult,
    FailureContext,
    RecoveryProposal,
    StepRecord,
    ToolCall,
    Usage,
)
from ..tokenizer import count_tokens, truncate_to_tokens
from . import prompts
from .tools import SPEC_BY_NAME, TOOL_NAMES, validate_args

#: Alternative actions the recovery advisor may propose (per the contract).
MAX_ALTERNATIVES = 3


# --------------------------------------------------------------------------- #
# Lenient parsing helpers
# --------------------------------------------------------------------------- #

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model answer, however wrapped.

    Models return JSON inside code fences, with a prose preamble, or with an
    apology before it.  Anything that cannot be parsed returns ``None`` and the
    caller falls back to prose handling.
    """
    if not text:
        return None
    candidates: list[str] = []
    for match in _FENCE.finditer(text):
        candidates.append(match.group(1))
    candidates.append(text.strip())
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        span = _balanced_object(candidate)
        if span is not None:
            try:
                parsed = json.loads(span)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
    return None


def _balanced_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` block, ignoring braces in strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_tool_calls(raw: Any) -> list[ToolCall]:
    """Turn a lenient list of ``{"name":..., "args": {...}}`` into valid ToolCalls.

    Anything that is not a known tool with valid arguments is dropped, so a
    hallucinated action can never reach the executor.
    """
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for item in raw:
        if isinstance(item, ToolCall):
            name, args = item.name, item.args
        elif isinstance(item, dict):
            name = item.get("name") or item.get("tool") or item.get("tool_name")
            args = item.get("args") or item.get("arguments") or item.get("input") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
        elif isinstance(item, str):
            name, args = item, {}
        else:
            continue
        if not isinstance(name, str) or name.strip() not in SPEC_BY_NAME:
            continue
        clean, error = validate_args(name.strip(), args or {})
        if error:
            continue
        calls.append(ToolCall(name=name.strip(), args=clean))
        if len(calls) >= MAX_ALTERNATIVES:
            break
    return calls


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #


class ExtractorSubAgent:
    """Answers one question about the content of the current page.

    It is the *only* place in the agent that may look at raw markup, and it
    returns a short answer - never the page - to the main loop.
    """

    name = "extractor"

    def __init__(self, llm: Any | None, config: Config | None = None) -> None:
        self.llm = llm
        self.config = config
        self.usage = Usage()
        self.calls = 0
        self.failures = 0

    # ------------------------------------------------------------------ #
    def answer(self, question: str, page_text: str, html_excerpt: str = "") -> ExtractionResult:
        question = (question or "").strip()
        if not question:
            return ExtractionResult(answer="", evidence="", confidence=0.0, usage=Usage(), subagent=self.name)
        if self.llm is None:
            self.failures += 1
            return ExtractionResult(
                answer="",
                evidence="",
                confidence=0.0,
                usage=Usage(),
                subagent=self.name,
            )
        budget = int(getattr(self.config, "extractor_token_budget", 24_000) or 24_000)
        text, _ = truncate_to_tokens(page_text or "", budget)
        remaining = max(0, budget - count_tokens(text))
        markup = ""
        if html_excerpt and remaining > 200:
            markup, _ = truncate_to_tokens(html_excerpt or "", remaining)
        messages = [
            _user(prompts.extractor_user_prompt(question, text, markup)),
        ]
        max_tokens = int(getattr(self.config, "subagent_max_output_tokens", 1024) or 1024)
        try:
            response = self.llm.complete(
                messages,
                [],
                system=prompts.extractor_system_prompt(),
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception:
            self.failures += 1
            self.calls += 1
            return ExtractionResult(
                answer="",
                evidence="",
                confidence=0.0,
                usage=Usage(calls=1),
                subagent=self.name,
            )
        self.calls += 1
        usage = getattr(response, "usage", None) or Usage()
        self.usage = self.usage.add(usage)
        result = self._parse(getattr(response, "text", "") or "", usage)
        return result

    # ------------------------------------------------------------------ #
    def _parse(self, text: str, usage: Usage) -> ExtractionResult:
        payload = extract_json_object(text)
        if payload is not None:
            answer = str(payload.get("answer") or "").strip()
            evidence = str(payload.get("evidence") or "").strip()
            confidence = _as_float(payload.get("confidence"), 0.0)
            if answer.lower() in ("none", "null", "n/a", "not present", "unknown"):
                answer = ""
            if not answer:
                confidence = 0.0
            return ExtractionResult(
                answer=answer,
                evidence=evidence[:600],
                confidence=max(0.0, min(1.0, confidence)),
                usage=usage,
                subagent=self.name,
            )
        # tolerant path: the model answered in prose
        prose = text.strip()
        if not prose:
            return ExtractionResult(answer="", usage=usage, subagent=self.name)
        lowered = prose.lower()
        absent = any(
            marker in lowered
            for marker in ("not present", "no information", "does not contain", "cannot find", "not found in")
        )
        if absent:
            return ExtractionResult(answer="", evidence=prose[:300], confidence=0.0, usage=usage, subagent=self.name)
        sentence = prose.split("\n")[0].strip()
        sentence = sentence[:400]
        confidence = _as_float(_first_number(prose), 0.35)
        return ExtractionResult(
            answer=sentence,
            evidence="",
            confidence=max(0.0, min(1.0, confidence)),
            usage=usage,
            subagent=self.name,
        )


# --------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------- #


class RecoverySubAgent:
    """Diagnoses a repeated failure and proposes alternative actions."""

    name = "recovery"

    def __init__(self, llm: Any | None, config: Config | None = None) -> None:
        self.llm = llm
        self.config = config
        self.usage = Usage()
        self.calls = 0
        self.failures = 0

    # ------------------------------------------------------------------ #
    def propose(self, failure: FailureContext) -> RecoveryProposal:
        if self.llm is None:
            return self._generic(failure)
        max_tokens = int(getattr(self.config, "subagent_max_output_tokens", 1024) or 1024)
        messages = [
            _user(prompts.recovery_user_prompt(failure, _render_page(failure), TOOL_NAMES)),
        ]
        try:
            response = self.llm.complete(
                messages,
                [],
                system=prompts.recovery_system_prompt(),
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception:
            self.failures += 1
            self.calls += 1
            return self._generic(failure, Usage(calls=1))
        self.calls += 1
        usage = getattr(response, "usage", None) or Usage()
        self.usage = self.usage.add(usage)
        payload = extract_json_object(getattr(response, "text", "") or "")
        if payload is None:
            prose = (getattr(response, "text", "") or "").strip()
            if not prose:
                return self._generic(failure, usage)
            return RecoveryProposal(
                advice=prose[:600],
                alternative_actions=[],
                give_up=False,
                usage=usage,
                subagent=self.name,
            )
        advice = str(payload.get("advice") or payload.get("diagnosis") or "").strip()
        alternatives = parse_tool_calls(
            payload.get("alternative_actions")
            or payload.get("alternatives")
            or payload.get("actions")
            or []
        )
        give_up = bool(payload.get("give_up") or payload.get("giveUp") or False)
        if not advice:
            advice = (
                "The previous approach kept failing. Change strategy: re-read the page, "
                "look for another control, or ask the human."
            )
        return RecoveryProposal(
            advice=advice[:800],
            alternative_actions=alternatives[:MAX_ALTERNATIVES],
            give_up=give_up,
            usage=usage,
            subagent=self.name,
        )

    def _generic(self, failure: FailureContext, usage: Usage | None = None) -> RecoveryProposal:
        call = getattr(failure, "tool_call", None)
        name = getattr(call, "name", "the action")
        return RecoveryProposal(
            advice=(
                f"{name} failed {getattr(failure, 'attempts', 0)} times "
                f"({getattr(failure, 'error', 'no error reported')}). "
                "Do not repeat it: re-read the page, pick a different element, go back, "
                "wait for the page to settle, or ask the human what to do."
            ),
            alternative_actions=[],
            give_up=False,
            usage=usage or Usage(),
            subagent=self.name,
        )


# --------------------------------------------------------------------------- #
# Summarizer
# --------------------------------------------------------------------------- #


class Summarizer:
    """Compresses old steps into a running summary for the context manager."""

    name = "summarizer"

    def __init__(self, llm: Any | None, config: Config | None = None) -> None:
        self.llm = llm
        self.config = config
        self.usage = Usage()
        self.calls = 0
        self.failures = 0

    # ------------------------------------------------------------------ #
    def summarize(
        self,
        task: str,
        steps: Iterable[StepRecord],
        previous_summary: str = "",
    ) -> str:
        steps = list(steps)
        if not steps:
            return previous_summary.strip()
        if self.llm is None:
            return deterministic_summary(steps, previous_summary)
        max_tokens = int(getattr(self.config, "subagent_max_output_tokens", 1024) or 1024)
        messages = [_user(prompts.summarizer_user_prompt(task, steps, previous_summary))]
        try:
            response = self.llm.complete(
                messages,
                [],
                system=prompts.summarizer_system_prompt(),
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception:
            self.failures += 1
            self.calls += 1
            return deterministic_summary(steps, previous_summary)
        self.calls += 1
        usage = getattr(response, "usage", None) or Usage()
        self.usage = self.usage.add(usage)
        text = (getattr(response, "text", "") or "").strip()
        if not text:
            return deterministic_summary(steps, previous_summary)
        return text


def deterministic_summary(steps: Sequence[StepRecord], previous_summary: str = "") -> str:
    """Bullet summary used when no LLM is configured (or it failed)."""
    lines: list[str] = []
    if previous_summary.strip():
        lines.append(previous_summary.strip()[:400])
    for step in steps:
        lines.append("- " + step.digest())
    return "\n".join(lines)[:2_000]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _user(text: str):
    from ..types import LLMMessage

    return LLMMessage(role="user", content=text)


def _render_page(failure: FailureContext) -> str:
    page = getattr(failure, "page", None)
    if page is None:
        return ""
    from .tools import fallback_render_page

    return fallback_render_page(page, budget_tokens=1_200, mode="full")


def _first_number(text: str) -> float | None:
    match = re.search(r"(0?\.\d+|[01](?:\.\d+)?)", text)
    if not match:
        return None
    value = _as_float(match.group(1), 0.0)
    return value if 0.0 <= value <= 1.0 else None


__all__ = [
    "ExtractorSubAgent",
    "RecoverySubAgent",
    "Summarizer",
    "deterministic_summary",
    "extract_json_object",
    "parse_tool_calls",
    "MAX_ALTERNATIVES",
]
