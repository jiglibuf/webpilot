"""The agent loop: ``Agent.run(task) -> RunResult``.

One iteration is: drain what the human typed -> assemble the prompt -> call the
model -> execute the tool calls it asked for (in order) -> fold the results and a
fresh page observation back into the conversation -> repeat until the model calls
``finish``, a limit is hit, or something unrecoverable happens.

Everything the loop does is (a) reported through ``AgentUI`` as a ``RunEvent``
and (b) written as one JSON line to the run transcript, which is the evidence
artifact for the demo.  ``RunResult.success`` is only true when the model claimed
success *and* the final page observation does not contradict the claim.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from ..config import Config
from ..types import (
    ActionContext,
    ActionRisk,
    FailureContext,
    LLMMessage,
    LLMResponse,
    PageModel,
    RunEvent,
    RunResult,
    SnapshotMode,
    StepRecord,
    ToolCall,
    ToolResult,
    Usage,
)
from . import prompts, tools
from .memory import ContextManager
from .subagents import ExtractorSubAgent, RecoverySubAgent, Summarizer
from .tools import DispatchContext, ToolRegistry

#: After this many *consecutive* hard model failures the run gives up.
MAX_LLM_FAILURES = 3

#: Phrases in an alert/dialog/title that contradict a claimed success.
CONTRADICTION_MARKERS = (
    "error",
    "invalid",
    "failed",
    "failure",
    "denied",
    "not found",
    "incorrect",
    "unable to",
    "cannot",
    "can not",
    "can't",
    "try again",
    "rejected",
    "problem",
    "oops",
    "404",
    "403",
    "500",
)

_TITLE_MARKERS = ("error", "not found", "404", "problem")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


class _NullUI:
    """Fallback UI: renders nothing, approves nothing, answers nothing."""

    def emit(self, event: RunEvent) -> None:  # pragma: no cover - trivial
        return None

    def confirm(self, prompt: str, details: str) -> bool:  # pragma: no cover
        return False

    def ask(self, question: str) -> str:  # pragma: no cover
        return ""

    def drain_instructions(self) -> list[str]:  # pragma: no cover
        return []


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _fmt_args(args: Any, limit: int = 240) -> str:
    if not isinstance(args, dict):
        text = str(args)
    else:
        text = ", ".join(f"{key}={value!r}" for key, value in args.items())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _resolve_renderer(session: Any) -> Callable[..., str] | None:
    """Prefer the renderer the browser session/package provides, if any."""
    if session is not None:
        for attribute in ("render_page_model", "render"):
            candidate = getattr(session, attribute, None)
            if callable(candidate):
                return candidate
    try:  # optional integration: the browser package owns the real renderer
        from webpilot.browser.snapshot import render_page_model  # type: ignore

        return render_page_model
    except Exception:
        return None


def _raw_provider_for(session: Any, max_chars: int = 120_000) -> Callable[[], str] | None:
    """A lazily-called reader for raw markup, used only by the extractor."""
    if session is None or not hasattr(session, "raw_html"):
        return None

    def _read() -> str:
        try:
            return session.raw_html(max_chars)
        except TypeError:
            return session.raw_html()
        except Exception:
            return ""

    return _read


def _tool_text(result: ToolResult, limit: int = 20_000) -> str:
    """What the model reads for a tool result.

    ``ToolResult.to_llm_text()`` is the one-line contract; agent-local tools may
    additionally attach a payload the model came for (``page_text`` from
    ``page_outline``), which is appended here so it actually reaches the context.
    """
    text = result.to_llm_text()
    payload = result.data.get("page_text") if isinstance(result.data, dict) else None
    if payload:
        text += "\nPAGE OUTLINE:\n" + str(payload)
    return text if len(text) <= limit else text[:limit]


def _call_count(agent: Any) -> int:
    """How many times a sub-agent was invoked (lenient about how it counts)."""
    value = getattr(agent, "calls", 0)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return 0


def _usage_dict(usage: Usage | None) -> dict[str, int]:
    if usage is None:
        return {}
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cached_tokens": usage.cached_tokens,
        "calls": usage.calls,
        "total": usage.prompt_tokens + usage.completion_tokens,
    }


# --------------------------------------------------------------------------- #
# Transcript
# --------------------------------------------------------------------------- #


class _Transcript:
    """Append-only JSONL evidence log (one line per event and per step)."""

    def __init__(self, config: Config) -> None:
        directory = Path(config.transcript_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"run-{_timestamp()}.jsonl"
        if path.exists():  # two runs in the same microsecond: never clobber evidence
            for suffix in range(1, 1_000):
                candidate = directory / f"run-{_timestamp()}-{suffix}.jsonl"
                if not candidate.exists():
                    path = candidate
                    break
        self.path = path
        self._handle = self.path.open("w", encoding="utf-8")
        self.events = 0
        self.steps = 0

    def _line(self, payload: dict[str, Any]) -> None:
        self._handle.write(_json(payload) + "\n")
        self._handle.flush()

    def write_event(self, event: RunEvent) -> None:
        self.events += 1
        self._line(
            {
                "type": "event",
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "kind": event.kind,
                "level": event.level,
                "message": event.message,
                "data": event.data,
            }
        )

    def write_step(self, step: StepRecord) -> None:
        self.steps += 1
        self._line(
            {
                "type": "step",
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "index": step.index,
                "tool": step.tool_call.name,
                "arguments": step.tool_call.args,
                "ok": step.ok,
                "summary": step.result_summary,
                "page_url": step.page_url,
                "page_title": step.page_title,
                "page_hash": step.page_hash,
                "page_changed": step.page_changed,
                "duration_s": round(step.duration_s, 3),
                "risk": step.risk,
                "confirmed": step.confirmed,
                "usage": {
                    "prompt_tokens": step.prompt_tokens,
                    "completion_tokens": step.completion_tokens,
                    "total": step.prompt_tokens + step.completion_tokens,
                },
                "screenshot_path": step.screenshot_path,
            }
        )

    def write_result(self, result: RunResult) -> None:
        self._line(
            {
                "type": "result",
                "ts": datetime.now().isoformat(timespec="milliseconds"),
                "task": result.task,
                "success": result.success,
                "answer": result.answer,
                "steps": result.steps,
                "duration_s": round(result.duration_s, 3),
                "error": result.error,
                "subagent_calls": result.subagent_calls,
                "confirmations": result.confirmations,
                "recoveries": result.recoveries,
                "usage": _usage_dict(result.usage),
                "usage_main": _usage_dict(getattr(result.usage, "main", None)),
                "usage_subagents": _usage_dict(getattr(result.usage, "subagents", None)),
            }
        )

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:  # pragma: no cover - already closed
            pass


@dataclass
class _Outcome:
    success: bool
    answer: str
    error: str | None
    reason: str = ""


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #


class Agent:
    """Drives one browser tab through a task with a tool-calling LLM."""

    def __init__(
        self,
        config: Config,
        llm: Any,
        session: Any = None,
        executor: Any = None,
        ui: Any = None,
        policy: Any = None,
        registry: ToolRegistry | None = None,
        context: ContextManager | None = None,
        extractor: ExtractorSubAgent | None = None,
        recovery: RecoverySubAgent | None = None,
        summarizer: Summarizer | None = None,
        renderer: Callable[..., str] | None = None,
        raw_provider: Callable[[], str] | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.session = session
        self.executor = executor
        self.ui = ui if ui is not None else _NullUI()
        self.policy = policy
        self.renderer = renderer if renderer is not None else _resolve_renderer(session)

        self.extractor = extractor if extractor is not None else (
            ExtractorSubAgent(llm, config) if llm is not None else None
        )
        self.recovery = recovery if recovery is not None else (
            RecoverySubAgent(llm, config) if llm is not None else None
        )
        self.summarizer = summarizer if summarizer is not None else (
            Summarizer(llm, config) if llm is not None else None
        )
        self.context = context if context is not None else ContextManager(
            config, summarizer=self.summarizer
        )

        if registry is None:
            registry = ToolRegistry(
                executor,
                config=config,
                ui=self.ui,
                policy=policy,
                extractor=self.extractor,
                context=self.context,
                renderer=self.renderer,
                session=session,
                raw_provider=raw_provider or _raw_provider_for(session),
            )
        else:
            registry.bind(
                ui=self.ui,
                config=config,
                extractor=self.extractor,
                context=self.context,
                renderer=self.renderer,
                raw_provider=raw_provider,
            )
        registry.bind(page_provider=self._provide_page_block)
        self.registry = registry
        self.context.set_page_provider(self._render_for_memory)
        self._system_prompt = prompts.system_prompt(config)

        # run state (reset by run())
        self._task = ""
        self._steps = 0
        self._model: PageModel | None = None
        self._previous_model: PageModel | None = None
        self._block = ""
        self._ctx = DispatchContext()
        self._pending: list[str] = []
        self._failure_counts: dict[str, int] = {}
        self._repeat_counts: Counter = Counter()
        self._recovered_keys: set[tuple[str, str]] = set()
        self._consecutive_failures = 0
        self._llm_failures = 0
        self._confirmations = 0
        self._recoveries = 0
        self._render_errors = 0
        self.main_usage = Usage()
        self._transcript: _Transcript | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self, task: str) -> RunResult:
        task = (task or "").strip()
        started = time.monotonic()
        self._reset_run(task)
        self.config.ensure_dirs()
        transcript = _Transcript(self.config)
        self._transcript = transcript
        self._emit(
            RunEvent(
                "start",
                f"starting run for task: {task}",
                {
                    "task": task,
                    "model": getattr(self.llm, "model", ""),
                    "provider": getattr(self.llm, "provider", ""),
                    "max_steps": self.config.max_steps,
                    "tools": list(tools.TOOL_NAMES),
                },
            )
        )

        self._model, self._block = self._initial_observation()
        self._ctx.page = self._model
        try:
            outcome = self._loop(task, started)
        except Exception as exc:  # a bug must still produce an honest result
            self._emit(
                RunEvent(
                    "error",
                    f"the run aborted: {type(exc).__name__}: {exc}",
                    {"error": str(exc), "type": type(exc).__name__},
                    level="error",
                )
            )
            outcome = _Outcome(
                False,
                self._partial_answer(f"the run aborted: {exc}"),
                f"{type(exc).__name__}: {exc}",
                "crash",
            )

        duration = time.monotonic() - started
        subagent_usage = self._subagent_usage()
        usage = self.main_usage.add(subagent_usage)
        # Attribution: ``RunResult.usage`` is the total; the two dynamic
        # attributes below split it between the main loop and the sub-agents.
        usage.main = self.main_usage  # type: ignore[attr-defined]
        usage.subagents = subagent_usage  # type: ignore[attr-defined]

        subagent_calls = self._subagent_call_count()
        result = RunResult(
            task=task,
            success=outcome.success,
            answer=outcome.answer,
            steps=self._steps,
            duration_s=duration,
            usage=usage,
            transcript_path=str(transcript.path),
            error=outcome.error,
            subagent_calls=subagent_calls,
            confirmations=self._confirmations,
            recoveries=self._recoveries,
        )
        self._emit(
            RunEvent(
                "finish",
                ("SUCCESS: " if result.success else "INCOMPLETE: ") + (result.answer or ""),
                {
                    "success": result.success,
                    "reason": outcome.reason,
                    "steps": result.steps,
                    "duration_s": round(duration, 2),
                    "usage": _usage_dict(usage),
                    "usage_main": _usage_dict(self.main_usage),
                    "usage_subagents": _usage_dict(subagent_usage),
                    "error": result.error,
                },
                level="success" if result.success else "warn",
            )
        )
        transcript.write_result(result)
        transcript.close()
        return result

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _reset_run(self, task: str) -> None:
        self._task = task
        self._steps = 0
        self._model = None
        self._previous_model = None
        self._block = ""
        self._ctx = DispatchContext(task=task)
        self._pending = []
        self._failure_counts = {}
        self._repeat_counts = Counter()
        self._recovered_keys = set()
        self._consecutive_failures = 0
        self._llm_failures = 0
        self._confirmations = 0
        self._recoveries = 0
        self._render_errors = 0
        self.main_usage = Usage()
        self.registry.finished = False
        self.registry.finish_success = False
        self.registry.finish_answer = ""

    # ------------------------------------------------------------------ #
    # Page plumbing
    # ------------------------------------------------------------------ #
    def _initial_observation(self) -> tuple[PageModel | None, str]:
        model = self._snapshot()
        if model is None:
            self._emit(
                RunEvent(
                    "info",
                    "no page observation available (no browser session attached)",
                    {"page": None},
                    level="warn",
                )
            )
            return None, ""
        block = self._render(model, "full", None)
        self._emit_page(model, changed=True)
        return model, block

    def _snapshot(self) -> PageModel | None:
        session = self.session
        if session is None:
            return None
        try:
            return session.snapshot(budget_tokens=self.context.page_budget(), mode="full")
        except TypeError:
            try:
                return session.snapshot(self.context.page_budget())
            except Exception as exc:
                self._emit_snapshot_error(exc)
                return None
        except Exception as exc:
            self._emit_snapshot_error(exc)
            return None

    def _emit_snapshot_error(self, exc: Exception) -> None:
        self._emit(
            RunEvent(
                "error",
                f"could not read the page: {type(exc).__name__}: {exc}",
                {"error": str(exc)},
                level="error",
            )
        )

    def _render(self, model: PageModel, mode: SnapshotMode, previous: PageModel | None) -> str:
        budget = self.context.page_budget()
        renderer = self.renderer
        if renderer is not None:
            try:
                return renderer(model, budget_tokens=budget, mode=mode, previous=previous)
            except Exception as exc:
                self._render_errors += 1
                if self._render_errors == 1:
                    self._emit(
                        RunEvent(
                            "error",
                            f"the page renderer failed ({type(exc).__name__}: {exc}); "
                            "falling back to the built-in renderer",
                            {"error": str(exc)},
                            level="warn",
                        )
                    )
        return tools.fallback_render_page(model, budget_tokens=budget, mode=mode, previous=previous)

    def _provide_page_block(self, mode: str, _filter: str = "") -> tuple[PageModel | None, str]:
        """Callback the tool registry uses for ``page_outline``."""
        model = self._model
        if model is None and self.session is not None:
            model = self._snapshot()
            if model is not None:
                self._model = model
        if model is None:
            return None, ""
        if mode == "text":
            return model, self._render(model, "text", None)
        if mode == "delta":
            return model, self._render(model, "delta", self._previous_model)
        return model, self._render(model, "full", None)

    def _render_for_memory(self, page: PageModel, mode: str = "full") -> str:
        """Callback the context manager uses to re-render a dropped page block."""
        if mode == "text":
            return self._render(page, "text", None)
        if mode == "delta":
            return self._render(page, "delta", self._previous_model)
        return self._render(page, "full", None)

    def _refresh_after(self, result: ToolResult, index: int) -> tuple[str, bool]:
        """Adopt the page the action produced; return ``(block, changed)``."""
        previous = self._model
        model = result.page if result.page is not None else previous
        if result.page is None and result.page_changed and self.session is not None:
            model = self._snapshot() or previous
        changed = model is not None and (
            previous is None or model.page_hash() != previous.page_hash()
        )
        self._ctx.page = model
        self._ctx.step = index
        self._ctx.history.append(f"#{index} {result.summary}")
        if model is None:
            return "", changed
        if not changed:
            self._model = model
            self._emit_page(model, changed=False)
            return "", False
        mode: SnapshotMode = (
            "delta" if previous is not None and previous.url == model.url else "full"
        )
        block = self._render(model, mode, previous)
        self._previous_model = previous
        self._model = model
        self._block = block
        self._emit_page(model, changed=True)
        return block, True

    def _emit_page(self, model: PageModel | None, *, changed: bool) -> None:
        if model is None:
            return
        self._emit(
            RunEvent(
                "page",
                f"page: {model.url} | {model.title} | generation {model.generation}"
                + ("" if changed else " (unchanged)"),
                {
                    "url": model.url,
                    "title": model.title,
                    "generation": model.generation,
                    "hash": model.page_hash(),
                    "changed": changed,
                    "elements": len(model.elements),
                    "alerts": list(model.alerts),
                    "truncated": model.truncated,
                },
                level="info" if changed else "info",
            )
        )

    # ------------------------------------------------------------------ #
    # Model calls
    # ------------------------------------------------------------------ #
    def _collect_pending(self) -> list[str]:
        pending = list(self._pending)
        self._pending = []
        for instruction in self.ui.drain_instructions() or []:
            text = (instruction or "").strip()
            if not text:
                continue
            self._emit(
                RunEvent("user", f"human instruction: {text}", {"instruction": text})
            )
            pending.append(prompts.human_instruction_message(text))
        return pending

    def _call_model(self, messages: Sequence[LLMMessage]) -> LLMResponse | None:
        try:
            response = self.llm.complete(
                messages,
                tools.TOOL_SPECS,
                system=self._system_prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_output_tokens,
            )
        except Exception as exc:
            self._llm_failures += 1
            self._emit(
                RunEvent(
                    "error",
                    f"model call failed ({type(exc).__name__}): {exc} "
                    f"[{self._llm_failures}/{MAX_LLM_FAILURES}]",
                    {"error": str(exc), "attempt": self._llm_failures},
                    level="error",
                )
            )
            if self._llm_failures >= MAX_LLM_FAILURES:
                return None
            return LLMResponse(text="(the model call failed; retrying)")
        self._llm_failures = 0
        if response is None:
            return LLMResponse()
        if not isinstance(response, LLMResponse):
            return LLMResponse(text=str(response))
        return response

    # ------------------------------------------------------------------ #
    # The loop
    # ------------------------------------------------------------------ #
    def _loop(self, task: str, started: float) -> _Outcome:
        config = self.config
        while True:
            if config.task_timeout_s and (time.monotonic() - started) > config.task_timeout_s:
                return _Outcome(
                    False,
                    self._partial_answer(
                        f"the time limit of {config.task_timeout_s:.0f}s was reached"
                    ),
                    "time limit reached",
                    "timeout",
                )
            if self._steps >= config.max_steps:
                return _Outcome(
                    False,
                    self._partial_answer(prompts.step_limit_message(config.max_steps)),
                    f"reached the step limit ({config.max_steps})",
                    "max_steps",
                )

            pending = self._collect_pending()
            self._ctx.page = self._model  # keep the policy's view of the page current
            self._ctx.task = task
            messages = self.context.build_messages(
                task=task, page=self._model, page_block=self._block, pending=pending
            )
            stats = self.context.stats()
            self._emit(
                RunEvent(
                    "context",
                    f"prompt ~{stats['prompt_tokens_estimate']} tokens "
                    f"(page {stats['page_tokens']}, page budget {stats['page_token_budget']}, "
                    f"history {stats['history_tokens']}, compactions {stats['compactions']})",
                    stats,
                )
            )

            response = self._call_model(messages)
            if response is None:
                return _Outcome(
                    False,
                    self._partial_answer("the language model is unavailable"),
                    "the language model failed repeatedly",
                    "llm_error",
                )
            self.main_usage = self.main_usage.add(response.usage)
            text = (response.text or "").strip()
            calls = list(response.tool_calls or [])
            self._emit(
                RunEvent(
                    "thinking",
                    text[:1_500] or "(no reasoning text)",
                    {
                        "text": text,
                        "tool_calls": [call.name for call in calls],
                        "usage": _usage_dict(response.usage),
                    },
                )
            )

            if not calls:
                self._consecutive_failures += 1
                self._emit(
                    RunEvent(
                        "error",
                        "the model produced no tool call; asking it to act",
                        {"consecutive_failures": self._consecutive_failures},
                        level="warn",
                    )
                )
                self._pending.append(prompts.no_tool_call_message())
                if self._consecutive_failures >= config.max_consecutive_failures:
                    return _Outcome(
                        False,
                        self._partial_answer(prompts.failure_limit_message(self._consecutive_failures)),
                        "the model stopped calling tools",
                        "no_tool_calls",
                    )
                continue

            for position, call in enumerate(calls):
                self._steps += 1
                self._execute_one(
                    call,
                    self._steps,
                    assistant_text=text if position == 0 else "",
                    turn_usage=response.usage if position == 0 else None,
                )
                if self.registry.finished:
                    return self._finish_outcome()
                if self._consecutive_failures >= config.max_consecutive_failures:
                    return _Outcome(
                        False,
                        self._partial_answer(
                            prompts.failure_limit_message(self._consecutive_failures)
                        ),
                        f"{self._consecutive_failures} consecutive actions failed",
                        "failure_limit",
                    )

    # ------------------------------------------------------------------ #
    # One tool call
    # ------------------------------------------------------------------ #
    def _execute_one(
        self,
        call: ToolCall,
        index: int,
        assistant_text: str = "",
        turn_usage: Usage | None = None,
    ) -> ToolResult:
        if call.id is None:
            call.id = f"call_{index}"
        self._emit(
            RunEvent(
                "tool_call",
                f"{call.name}({_fmt_args(call.args)})",
                {"tool": call.name, "arguments": dict(call.args), "step": index},
            )
        )

        ctx = self._ctx
        ctx.step = index
        risk, action_ctx, refusal = self._assess_risk(call, ctx)
        confirmed: bool | None = None
        denied = refusal

        if risk is not None:
            self._emit(
                RunEvent(
                    "risk",
                    f"risk {risk.level}: {call.name} "
                    + ("; ".join(risk.reasons) if risk.reasons else "(no specific rule)"),
                    {
                        "tool": call.name,
                        "level": risk.level,
                        "reasons": list(risk.reasons),
                        "rules": list(risk.matched_rules),
                    },
                    level="warn" if risk.level != "safe" else "info",
                )
            )
        if denied is None and risk is not None and risk.requires_confirmation:
            approved, reason = self._authorize(call, risk, action_ctx)
            confirmed = approved
            if not approved:
                denied = reason

        if denied is not None:
            human = denied.startswith("human:")
            reason = denied.split(":", 1)[1] if human else denied
            summary = (
                f"the human denied this action: {reason}"
                if human
                else f"refused: {reason}"
            )
            result = ToolResult(
                ok=False,
                summary=summary,
                error=summary,
                recovery_hint=(
                    "Do not retry this exact action. Look for a legitimate alternative, "
                    "or use ask_user to find out what the human wants."
                ),
                data={"denied": True, "human": human},
            )
            self.registry.denied += 1
            duration = 0.0
        else:
            started = time.monotonic()
            result = self.registry.dispatch(call, ctx)
            duration = time.monotonic() - started

        block, changed = self._refresh_after(result, index)
        page = self._model
        step = StepRecord(
            index=index,
            tool_call=call,
            result_summary=result.summary,
            ok=result.ok,
            page_url=page.url if page else "",
            page_title=page.title if page else "",
            page_hash=page.page_hash() if page else "",
            page_changed=changed,
            duration_s=duration,
            prompt_tokens=turn_usage.prompt_tokens if turn_usage else 0,
            completion_tokens=turn_usage.completion_tokens if turn_usage else 0,
            risk=risk.level if risk else "safe",
            confirmed=confirmed,
            screenshot_path=result.screenshot_path,
        )
        self._emit(
            RunEvent(
                "tool_result",
                result.summary[:1_000],
                {
                    "tool": call.name,
                    "ok": result.ok,
                    "step": index,
                    "error": result.error,
                    "recovery_hint": result.recovery_hint,
                    "data": result.data,
                    "page_changed": changed,
                    "duration_s": round(duration, 3),
                },
                level="info" if result.ok else "warn",
            )
        )
        self.context.record(
            step,
            tool_text=_tool_text(result),
            assistant_text=assistant_text,
            page_block=block,
            tool_tokens=self.context.page_budget() if call.name == "page_outline" else None,
        )
        if self._transcript is not None:
            self._transcript.write_step(step)
        self._account(call, result, index)
        return result

    def _assess_risk(
        self, call: ToolCall, ctx: DispatchContext
    ) -> tuple[ActionRisk | None, ActionContext, str | None]:
        """Classify the call.  Returns ``(risk, action_context, refusal)``.

        ``refusal`` is set when the call must not be executed at all: a secret
        typed by the agent, or an action the policy could not classify (an
        unclassified destructive-looking call must never run).
        """
        action_ctx = ctx.action_context(call)
        if call.name == "type_text" and action_ctx.is_password_field and not self.config.allow_password_typing:
            return (
                None,
                action_ctx,
                "password fields are filled by the human, never by the agent",
            )
        if self.policy is None:
            return ActionRisk(level="safe"), action_ctx, None
        try:
            risk = self.policy.classify(call, action_ctx)
        except Exception as exc:
            return (
                None,
                action_ctx,
                f"the security policy could not classify this action ({type(exc).__name__}: {exc})",
            )
        if isinstance(risk, str):
            risk = ActionRisk(level=risk)  # type: ignore[arg-type]
        if not isinstance(risk, ActionRisk):
            return None, action_ctx, "the security policy returned an unusable decision"
        return risk, action_ctx, None

    def _authorize(self, call: ToolCall, risk: ActionRisk, action_ctx: ActionContext) -> tuple[bool, str]:
        self._confirmations += 1
        details = (
            f"tool: {call.name}({_fmt_args(call.args)})\n"
            f"risk: {risk.level}\n"
            f"reasons: {'; '.join(risk.reasons) or '(none)'}\n"
            f"page: {action_ctx.url}\n"
            f"element: {action_ctx.element.name if action_ctx.element else '(none)'}"
        )
        prompt_text = f"Allow this {risk.level} action?"
        self._emit(
            RunEvent(
                "confirm",
                f"asking the human to confirm a {risk.level} action: {call.name}",
                {"tool": call.name, "level": risk.level, "reasons": list(risk.reasons)},
                level="warn",
            )
        )
        try:
            approved, reason = self.policy.authorize(call, risk, action_ctx, self.ui.confirm)
        except Exception as exc:
            approved, reason = False, f"the policy could not authorize this action ({exc})"
        approved = bool(approved)
        message = reason or ("approved by the human" if approved else "not approved")
        self._emit(
            RunEvent(
                "confirm",
                f"human {'approved' if approved else 'denied'} the action: {message}",
                {"tool": call.name, "approved": approved, "details": details},
                level="info" if approved else "warn",
            )
        )
        if approved:
            return True, ""
        return False, f"human:{message}"

    # ------------------------------------------------------------------ #
    # Failure handling
    # ------------------------------------------------------------------ #
    def _account(self, call: ToolCall, result: ToolResult, index: int) -> None:
        signature = call.signature()
        if result.ok:
            self._failure_counts.pop(signature, None)
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
            attempts = self._failure_counts.get(signature, 0) + 1
            self._failure_counts[signature] = attempts
            if attempts >= max(2, int(self.config.max_attempts_per_action)):
                self._run_recovery(call, result, attempts, index)

        page_hash = self._model.page_hash() if self._model is not None else ""
        if not page_hash:
            return
        key = (page_hash, signature)
        self._repeat_counts[key] += 1
        repeats = self._repeat_counts[key]
        limit = max(2, int(self.config.loop_detection_repeats))
        if repeats >= limit:
            self._pending.append(
                prompts.loop_warning_message(f"{call.name}({_fmt_args(call.args)})", repeats)
            )
            self._emit(
                RunEvent(
                    "error",
                    f"loop detected: {call.name} repeated {repeats} times on an unchanged page",
                    {"tool": call.name, "repeats": repeats, "page_hash": page_hash},
                    level="warn",
                )
            )
        if repeats == limit + 1 and key not in self._recovered_keys:
            self._recovered_keys.add(key)
            self._run_recovery(call, ToolResult(ok=False, summary=result.summary, error="the action is not changing the page"), repeats, index)

    def _run_recovery(self, call: ToolCall, result: ToolResult, attempts: int, index: int) -> None:
        if self.recovery is None:
            self._pending.append(
                prompts.failure_advice_message(
                    f"{call.name} failed {attempts} times ({result.error or result.summary}). "
                    "Change the approach instead of repeating it."
                )
            )
            return
        failure = FailureContext(
            task=self._task,
            tool_call=call,
            error=result.error or result.summary,
            attempts=attempts,
            page=self._model,
            history=[step.digest() for step in self.context.steps[-6:]],
        )
        try:
            proposal = self.recovery.propose(failure)
        except Exception as exc:  # the sub-agent must never break the run
            self._emit(
                RunEvent(
                    "error",
                    f"the recovery sub-agent failed: {type(exc).__name__}: {exc}",
                    {"error": str(exc)},
                    level="warn",
                )
            )
            return
        self._recoveries += 1
        alternatives = [f"{alt.name}({_fmt_args(alt.args)})" for alt in proposal.alternative_actions]
        self._emit(
            RunEvent(
                "subagent",
                f"recovery advice: {proposal.advice}",
                {
                    "subagent": "recovery",
                    "advice": proposal.advice,
                    "alternatives": alternatives,
                    "give_up": proposal.give_up,
                    "attempts": attempts,
                    "usage": _usage_dict(proposal.usage),
                },
                level="warn",
            )
        )
        if proposal.give_up:
            self._pending.append(prompts.give_up_message(proposal.advice))
        else:
            self._pending.append(prompts.failure_advice_message(proposal.advice, alternatives))

    # ------------------------------------------------------------------ #
    # Finishing
    # ------------------------------------------------------------------ #
    def _finish_outcome(self) -> _Outcome:
        answer = (self.registry.finish_answer or "").strip()
        success = bool(self.registry.finish_success)
        contradiction = self._contradiction(self._model)
        error = None
        if success and contradiction:
            success = False
            error = f"the final page contradicts the reported success: {contradiction}"
            answer = (
                answer
                + f"\n[not verified: after finishing, the page still showed: {contradiction}]"
            ).strip()
            self._emit(
                RunEvent(
                    "error",
                    f"finish(success=true) rejected: the page still shows {contradiction!r}",
                    {"contradiction": contradiction},
                    level="error",
                )
            )
        return _Outcome(success, answer or "(the agent finished without an answer)", error, "finish")

    def _contradiction(self, page: PageModel | None) -> str | None:
        """Evidence that the page does not support a claimed success."""
        if page is None:
            return None
        for text in list(page.alerts) + list(page.dialogs):
            lowered = text.lower()
            if any(marker in lowered for marker in CONTRADICTION_MARKERS):
                return text if len(text) <= 200 else text[:197] + "…"
        title = (page.title or "").lower()
        if title and any(marker in title for marker in _TITLE_MARKERS):
            return f"page title: {page.title}"
        invalid = next((element for element in page.elements if element.invalid), None)
        if invalid is not None:
            return f"the field {invalid.name!r} is still marked invalid"
        return None

    def _partial_answer(self, note: str) -> str:
        lines = [note]
        digests = [step.digest() for step in self.context.steps[-5:]]
        if digests:
            lines.append("Last actions:")
            lines.extend(digests)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def _subagent_call_count(self) -> int:
        total = int(getattr(self.registry, "subagent_calls", 0) or 0)
        seen: set[int] = set()
        for agent in (
            self.recovery,
            self.extractor,
            self.summarizer,
            getattr(self.context, "summarizer", None),
        ):
            if agent is None or id(agent) in seen:
                continue
            seen.add(id(agent))
            if agent is getattr(self.registry, "extractor", None):
                continue  # the registry already counted these
            total += _call_count(agent)
        return total

    def _subagent_usage(self) -> Usage:
        """Token spend of everything outside the main loop (no double counting)."""
        usage = Usage()
        seen: set[int] = set()
        candidates = [
            self.extractor,
            self.recovery,
            self.summarizer,
            getattr(self.context, "summarizer", None),
        ]
        for agent in candidates:
            if agent is None or id(agent) in seen:
                continue
            seen.add(id(agent))
            value = getattr(agent, "usage", None)
            if isinstance(value, Usage):
                usage = usage.add(value)
        registry_usage = getattr(self.registry, "extractor_usage", None)
        tracked = any(
            isinstance(getattr(agent, "usage", None), Usage) and getattr(agent, "usage").total
            for agent in candidates
            if agent is not None
        )
        if not tracked and isinstance(registry_usage, Usage):
            usage = usage.add(registry_usage)
        return usage

    def _emit(self, event: RunEvent) -> None:
        try:
            self.ui.emit(event)
        except Exception:  # a broken UI must not kill the run
            pass
        if self._transcript is not None:
            self._transcript.write_event(event)


__all__ = ["Agent", "MAX_LLM_FAILURES", "CONTRADICTION_MARKERS"]
