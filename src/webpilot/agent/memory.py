"""Context management: keep the conversation inside the token budget.

The eight strategies (see ``docs/CONTEXT.md`` for the rationale):

1. the page is represented by a compact model, never by markup - this module only
   ever handles already-rendered page text;
2. the page-block budget *shrinks* as the history grows
   (``page_token_budget`` -> ``page_token_budget_min``);
3. a sliding window of the last ``config.history_window`` steps stays verbatim,
   older steps are folded into a running summary (LLM summarizer when available,
   deterministic bullets otherwise);
4. every individual tool result is truncated to
   ``config.max_tool_result_tokens``;
5. a repeated page hash renders as ``page unchanged since step N`` instead of a
   second copy of the same block;
6. a repeated identical tool-call signature injects a hard warning;
7. the scratchpad (``notes``) is always present, whatever else is dropped;
8. on budget pressure the manager drops, in order: page block -> old tool texts ->
   old steps -> summary -> notes -> everything but the task.  ``build_messages``
   never returns more than ``config.context_token_budget`` tokens
   (``tests/test_memory.py::test_stress_never_exceeds_budget`` pins this with 200
   synthetic steps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..config import Config
from ..tokenizer import count_tokens as default_count_tokens, truncate_to_tokens
from ..types import LLMMessage, PageModel, SnapshotMode, StepRecord, Usage
from . import prompts

#: Hard cap on the scratchpad so a chatty model cannot crowd out the page.
NOTES_TOKEN_CAP = 320
#: Above this many repeats of the same signature the warning is re-issued.
_WARN_EVERY = 4


@dataclass
class _Entry:
    """One recorded step plus the text that belongs in the conversation."""

    step: StepRecord
    tool_text: str = ""
    assistant_text: str = ""
    page_block: str = ""

    def signature(self) -> str:
        return self.step.tool_call.signature()


def _fallback_summary(previous: str, steps: Sequence[StepRecord], limit: int = 1_800) -> str:
    """Deterministic compaction used when no summarizer is configured."""
    lines: list[str] = []
    if previous.strip():
        lines.append(previous.strip()[:400])
    for step in steps:
        lines.append("- " + step.digest())
    text = "\n".join(lines)
    return text[:limit]


class ContextManager:
    """Owns the durable conversation, the scratchpad and the token budget."""

    def __init__(
        self,
        config: Config,
        count_tokens: Callable[[str], int] | None = None,
        summarizer: Any | None = None,
    ) -> None:
        self.config = config
        self.count = count_tokens or default_count_tokens
        self.summarizer = summarizer

        self._entries: list[_Entry] = []
        self._notes: str = ""
        self._summary: str = ""
        self._task: str = ""
        self._last_messages = 0
        self._summarized_upto = 0
        self.compactions = 0
        self.dropped_items = 0
        self.duplicate_pages = 0
        self.warnings_injected = 0
        self.prompt_tokens = 0
        self.page_tokens = 0
        self.summary_usage = Usage()
        self._prompt_tokens = 0
        self._page_tokens = 0
        self._pending_warnings: list[str] = []
        self._last_signature = ""
        self._signature_repeats = 0
        self._warned_at = 0
        self._page_provider: Callable[[PageModel, str], str] | None = None
        self._renderer: Callable[..., str] | None = None

    # ------------------------------------------------------------------ #
    # Wiring
    # ------------------------------------------------------------------ #
    def set_page_provider(self, provider: Callable[[PageModel, str], str] | None) -> None:
        """Callback used to re-render the page when no verbatim block is visible.

        ``provider(page, mode) -> str``.  The agent passes its own renderer here;
        without it the manager falls back to ``tools.fallback_render_page``.
        """
        self._page_provider = provider

    # ------------------------------------------------------------------ #
    # Scratchpad
    # ------------------------------------------------------------------ #
    @property
    def notes(self) -> str:
        return self._notes

    def update_notes(self, text: str) -> str:
        """Append a line to the scratchpad (``clear`` resets it).  Returns it."""
        text = (text or "").strip()
        if text.lower() in ("clear", "clear()", "reset"):
            self._notes = ""
            return self._notes
        marker = "…[older notes trimmed]"
        lines = [
            line
            for line in self._notes.splitlines()
            if line.strip() and not line.startswith("…")
        ]
        if text:
            lines.append("- " + text)
        trimmed = False
        while len(lines) > 1 and self.count("\n".join(lines) + "\n" + marker) > NOTES_TOKEN_CAP:
            lines.pop(0)
            trimmed = True
        notes = "\n".join(lines)
        if self.count(notes) > NOTES_TOKEN_CAP:  # a single very long note
            notes, _ = truncate_to_tokens(notes, NOTES_TOKEN_CAP, marker="…")
        elif trimmed:
            notes = notes + "\n" + marker
        self._notes = notes
        return self._notes

    def clear_notes(self) -> None:
        self._notes = ""

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def step_count(self) -> int:
        return len(self._entries)

    @property
    def steps(self) -> list[StepRecord]:
        return [entry.step for entry in self._entries]

    @property
    def summary(self) -> str:
        return self._summary

    def record(
        self,
        step: StepRecord,
        tool_text: str = "",
        assistant_text: str = "",
        page_block: str = "",
        tool_tokens: int | None = None,
    ) -> None:
        """Store one executed step (and the observation that followed it).

        ``tool_tokens`` overrides the per-result truncation limit: a
        ``page_outline`` result carries a whole page and is allowed the page
        budget instead of ``config.max_tool_result_tokens``.
        """
        max_tokens = max(24, int(tool_tokens or self.config.max_tool_result_tokens))
        text, _ = truncate_to_tokens(tool_text or "", max_tokens)
        assistant, _ = truncate_to_tokens(assistant_text or "", max_tokens)
        entry = _Entry(
            step=step,
            tool_text=prompts.tool_result_message(step.index, step.tool_call.name, text),
            assistant_text=assistant,
            page_block=page_block or "",
        )
        if self._entries and self._entries[-1].step.page_hash and entry.step.page_hash:
            if self._entries[-1].step.page_hash == entry.step.page_hash:
                self.duplicate_pages += 1

        signature = entry.signature()
        if signature == self._last_signature:
            self._signature_repeats += 1
        else:
            self._last_signature = signature
            self._signature_repeats = 1
            self._warned_at = 0
            self._pending_warnings = [
                w for w in self._pending_warnings if signature not in w
            ]
        if self._signature_repeats >= 2:
            issue = self._signature_repeats == 2 or (
                self._signature_repeats - self._warned_at
            ) >= _WARN_EVERY
            if issue:
                self._warned_at = self._signature_repeats
                self._pending_warnings.append(
                    prompts.loop_warning_message(
                        _short_action(entry.step), self._signature_repeats
                    )
                )
        self._entries.append(entry)

    # ------------------------------------------------------------------ #
    # Budgets
    # ------------------------------------------------------------------ #
    def page_budget(self) -> int:
        """Current page-block token budget: full at the start, shrinking later."""
        high = int(self.config.page_token_budget)
        low = min(int(self.config.page_token_budget_min), high)
        if high <= low:
            return high
        trigger = max(1, int(self.config.context_token_budget * self.config.compaction_trigger))
        used = self._history_tokens()
        ratio = min(1.0, used / trigger)
        budget = high - int((high - low) * ratio)
        return max(low, min(high, budget))

    def _budget(self) -> int:
        return max(1, int(self.config.context_token_budget))

    def _history_tokens(self) -> int:
        total = self.count(self._summary)
        for entry in self._entries[self._summarized_upto :]:
            total += self.count(entry.tool_text) + self.count(entry.assistant_text)
            total += self.count(entry.page_block)
        return total

    # ------------------------------------------------------------------ #
    # Compaction
    # ------------------------------------------------------------------ #
    def _window_entries(self) -> list[_Entry]:
        window = max(0, int(self.config.history_window))
        return self._entries[-window:] if window else []

    def _maybe_fold(self) -> None:
        """Fold the steps that fell out of the sliding window into the summary."""
        window = max(0, int(self.config.history_window))
        fold_end = max(0, len(self._entries) - window)
        foldable = self._entries[self._summarized_upto : fold_end]
        if not foldable:
            return
        trigger = self.config.compaction_trigger * self.config.context_token_budget
        forced = self._history_tokens() > trigger
        if len(foldable) < max(1, window) and not forced:
            return
        self._fold(foldable)
        self._summarized_upto = fold_end

    def _fold(self, foldable: Sequence[_Entry]) -> None:
        steps = [entry.step for entry in foldable]
        text = ""
        if self.summarizer is not None:
            try:
                text = self.summarizer.summarize(self._task, steps, self._summary) or ""
                # cumulative, never summed again: the loop attributes this usage
                self.summary_usage = getattr(self.summarizer, "usage", self.summary_usage) or self.summary_usage
            except Exception:
                text = ""
        if not text.strip():
            text = _fallback_summary(self._summary, steps)
        self._summary = text.strip()[:6_000]
        self.compactions += 1

    # ------------------------------------------------------------------ #
    # Message assembly
    # ------------------------------------------------------------------ #
    def build_messages(
        self,
        *,
        task: str = "",
        page: PageModel | None = None,
        page_block: str = "",
        pending: list[str] | None = None,
        extra: list[LLMMessage] | None = None,
    ) -> list[LLMMessage]:
        """Assemble the full prompt for the next model call."""
        if task:
            self._task = task
        self._maybe_fold()

        window = self._window_entries()
        work: dict[str, Any] = {
            "task": LLMMessage(role="user", content=prompts.first_user_message(self._task)),
            "summary": (
                LLMMessage(role="user", content=prompts.summary_message(self._summary))
                if self._summary.strip()
                else None
            ),
            "window": self._window_messages(window),
            "notes": LLMMessage(role="user", content=prompts.scratchpad_message(self._notes)),
            "obs": LLMMessage(role="user", content=self._observation_message(page, page_block, window)),
            "pending": [LLMMessage(role="user", content=p) for p in (pending or []) if p],
            "extra": list(extra or []),
        }
        for warning in self._take_warnings():
            work["pending"].append(LLMMessage(role="user", content=prompts.warning_message(warning)))

        messages = self._compose(work)
        self._page_tokens = self._count_message(work["obs"])
        budget = self._budget()
        if self._measure(messages) > budget:
            self._squeeze(work, budget)
            messages = self._compose(work)
        if self._measure(messages) > budget:
            self._hard_fit(messages, budget)
        self._prompt_tokens = self._measure(messages)
        self.page_tokens = self._page_tokens
        self.prompt_tokens = self._prompt_tokens
        self._last_messages = len(messages)
        return messages

    # -- parts ----------------------------------------------------------- #
    def _window_messages(self, window: Sequence[_Entry]) -> list[LLMMessage]:
        messages: list[LLMMessage] = []
        for entry in window:
            call = entry.step.tool_call
            call_id = call.id or f"call_{entry.step.index}"
            if call.id is None:
                call.id = call_id
            messages.append(
                LLMMessage(role="assistant", content=entry.assistant_text, tool_calls=[call])
            )
            messages.append(
                LLMMessage(
                    role="tool",
                    content=entry.tool_text,
                    tool_call_id=call_id,
                    name=call.name,
                )
            )
            if entry.page_block:
                messages.append(
                    LLMMessage(
                        role="user",
                        content=prompts.current_page_message(
                            entry.page_block,
                            generation=None,
                            step=entry.step.index,
                        ),
                    )
                )
        return messages

    def _observation_message(
        self,
        page: PageModel | None,
        page_block: str,
        window: Sequence[_Entry],
    ) -> str:
        """The current state: a fresh block, or a pointer at one already shown."""
        if page is None:
            return prompts.no_page_message()
        visible = next((entry for entry in reversed(window) if entry.page_block), None)
        current_hash = page.page_hash()
        if visible is not None and visible.step.page_hash == current_hash:
            self.dropped_items += 1
            return prompts.page_marker_message(
                step=visible.step.index, page_hash=current_hash, unchanged=True
            )
        if page_block.strip():
            if visible is not None and visible.page_block.strip() == page_block.strip():
                return prompts.page_marker_message(
                    step=visible.step.index,
                    page_hash=current_hash,
                    generation=page.generation,
                    unchanged=False,
                )
            return prompts.current_page_message(
                page_block, generation=page.generation, url=page.url, title=page.title
            )
        return prompts.current_page_message(
            self._render(page, "full"), generation=page.generation, url=page.url, title=page.title
        )

    def _render(self, page: PageModel, mode: SnapshotMode) -> str:
        if self._page_provider is not None:
            try:
                text = self._page_provider(page, mode)
                if text:
                    return text
            except Exception:
                pass
        from .tools import fallback_render_page  # local import: avoids a cycle

        return fallback_render_page(page, budget_tokens=self.page_budget(), mode=mode)

    # -- measuring ------------------------------------------------------- #
    def _count_message(self, message: LLMMessage) -> int:
        total = self.count(message.content)
        for call in message.tool_calls:
            total += self.count(call.signature())
        return total

    def _measure(self, messages: Sequence[LLMMessage]) -> int:
        return sum(self._count_message(message) for message in messages)

    # -- squeezing ------------------------------------------------------- #
    def _compose(self, work: Mapping[str, Any]) -> list[LLMMessage]:
        messages: list[LLMMessage] = [work["task"]]
        if work["summary"] is not None:
            messages.append(work["summary"])
        messages.extend(work["window"])
        messages.append(work["notes"])
        messages.append(work["obs"])
        messages.extend(work["pending"])
        messages.extend(work["extra"])
        return messages

    def _squeeze(self, work: dict[str, Any], budget: int) -> None:
        """Drop the cheapest-to-lose content until the prompt fits ``budget``."""
        window: list[LLMMessage] = work["window"]
        obs: LLMMessage = work["obs"]
        notes: LLMMessage = work["notes"]
        summary: LLMMessage | None = work["summary"]
        pending: list[LLMMessage] = work["pending"]
        extra: list[LLMMessage] = work["extra"]
        task: LLMMessage = work["task"]
        floor = max(48, budget // 8)

        def over() -> bool:
            return self._measure(self._compose(work)) > budget

        # 1. shrink the current page block
        if over() and self._count_message(obs) > floor:
            obs.content = truncate_to_tokens(obs.content, floor, marker="…[page truncated]")[0]
            self.dropped_items += 1
        # 2. shrink the tool texts of the remaining window, oldest first
        per_tool = max(20, int(self.config.max_tool_result_tokens) // 3)
        for message in window:
            if not over():
                break
            if message.role == "tool" and self._count_message(message) > per_tool:
                message.content = truncate_to_tokens(
                    message.content, per_tool, marker="…[truncated]"
                )[0]
                self.dropped_items += 1
        # 3. drop whole older steps (each group starts at an assistant message so
        #    a tool message can never be orphaned)
        while over() and window:
            starts = [i for i, message in enumerate(window) if message.role == "assistant"]
            if len(starts) <= 1:
                break
            cut = starts[1]
            del window[0:cut]
            self.dropped_items += cut
        if over() and window:
            window.clear()
            self.dropped_items += 1
        # 4. drop the summary, then the scratchpad, then extra messages
        if over() and summary is not None:
            self.dropped_items += 1
            summary.content = truncate_to_tokens(summary.content, max(40, budget // 12))[0]
        if over() and summary is not None:
            work["summary"] = None
            self.dropped_items += 1
        if over() and self._count_message(notes) > 40:
            notes.content = truncate_to_tokens(notes.content, 40, marker="…")[0]
            self.dropped_items += 1
        while over() and len(pending) > 1:
            pending.pop(0)
            self.dropped_items += 1
        while over() and extra:
            extra.pop()
            self.dropped_items += 1
        if over() and self._count_message(obs) > 48:
            obs.content = truncate_to_tokens(obs.content, 48, marker="…")[0]
            self.dropped_items += 1
        if over() and self._count_message(task) > 48:
            task.content = truncate_to_tokens(task.content, 48, marker="…")[0]
            self.dropped_items += 1
        if self.dropped_items:
            self.compactions += 1

    def _hard_fit(self, messages: list[LLMMessage], budget: int) -> None:
        """Last-resort guarantee: never return more than ``budget`` tokens."""
        guard = 0
        while messages and self._measure(messages) > budget and guard < 512:
            guard += 1
            index = max(range(len(messages)), key=lambda i: self._count_message(messages[i]))
            message = messages[index]
            before = self._count_message(message)
            if before == 0:
                if len(messages) > 1:
                    messages.pop(index)
                    self.dropped_items += 1
                    continue
                break
            excess = self._measure(messages) - budget
            target = max(0, before - excess - 2)  # margin: the estimate can overshoot
            message.content = truncate_to_tokens(message.content, target, marker="")[0]
            if message.tool_calls:
                # an assistant message without its tool_calls would orphan the
                # matching tool message, so the whole group goes
                ids = {call.id for call in message.tool_calls}
                message.tool_calls = []
                messages[:] = [
                    m for m in messages if not (m.role == "tool" and m.tool_call_id in ids)
                ]
            if self._count_message(message) >= before and message.content:
                message.content = message.content[: max(1, len(message.content) // 2)]
            self.dropped_items += 1

    def _take_warnings(self) -> list[str]:
        warnings, self._pending_warnings = self._pending_warnings, []
        self.warnings_injected += len(warnings)
        return warnings

    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, Any]:
        """Token/compaction telemetry for the CLI and the transcript."""
        window = self._window_entries()
        return {
            "steps": self.step_count(),
            "prompt_tokens_estimate": self._prompt_tokens,
            "page_tokens": self._page_tokens,
            "history_tokens": self._history_tokens(),
            "compactions": self.compactions,
            "dropped_items": self.dropped_items,
            "budget": self._budget(),
            "page_token_budget": self.page_budget(),
            "summary_tokens": self.count(self._summary),
            "notes_tokens": self.count(self._notes),
            "window_steps": len(window),
            "history_window": int(self.config.history_window),
            "duplicate_pages": self.duplicate_pages,
            "warnings_injected": self.warnings_injected,
            "summarizer": self.summarizer is not None,
            "messages": self._last_messages,
        }


def _short_action(step: StepRecord) -> str:
    call = step.tool_call
    args = ", ".join(f"{k}={v!r}" for k, v in call.args.items())
    text = f"{call.name}({args})"
    return text if len(text) <= 120 else text[:117] + "…"


__all__ = ["ContextManager", "NOTES_TOKEN_CAP"]
