"""Tool schemas (`TOOL_SPECS`) and the dispatch layer (`ToolRegistry`).

Two families of tools exist:

* **browser tools** (``goto``, ``click``, ``type_text``, ``press_key``,
  ``scroll``, ``go_back``, ``wait_for``, ``tabs``, ``screenshot``) are delegated
  to a duck-typed executor that only has to expose
  ``execute(ToolCall) -> ToolResult``.  That is the real
  ``webpilot.browser.actions.ActionExecutor`` in production and a tiny stub in
  the tests, which keeps this package free of any browser import.
* **agent-local tools** (``page_outline``, ``ask_page``, ``notes``,
  ``ask_user``, ``finish``) are implemented here because they manipulate the
  agent's own state: the page renderer, the extractor sub-agent, the context
  manager's scratchpad and the UI.

Schema rules (from ``docs/INTERFACES.md``): every property is listed in
``required`` and the model must pass an explicit value for optional arguments
(``element_id: 0`` means "the focused element", ``""`` means "unused string").
``additionalProperties`` is ``False`` because the providers send these schemas
with ``strict: true``, which the OpenAI-compatible APIs reject otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..config import Config
from ..types import (
    ActionContext,
    Element,
    ExtractionResult,
    PageModel,
    RunEvent,
    SnapshotMode,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)
from ..tokenizer import count_tokens, truncate_to_tokens
from . import prompts

# --------------------------------------------------------------------------- #
# Schema helpers
# --------------------------------------------------------------------------- #


def _obj(**props: dict[str, Any]) -> dict[str, Any]:
    """Build a strict JSON-Schema object with every property required."""
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def _str(description: str, *, enum: Sequence[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string", "description": description}
    if enum:
        schema["enum"] = list(enum)
    return schema


def _int(description: str, *, minimum: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "integer", "description": description}
    if minimum is not None:
        schema["minimum"] = minimum
    return schema


def _num(description: str, *, minimum: float | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "number", "description": description}
    if minimum is not None:
        schema["minimum"] = minimum
    return schema


def _bool(description: str) -> dict[str, Any]:
    return {"type": "boolean", "description": description}


# --------------------------------------------------------------------------- #
# The tool table
# --------------------------------------------------------------------------- #

TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        "page_outline",
        "Re-read the current page in a compact, token-bounded form. mode='full' "
        "lists every addressable element and the visible text, mode='delta' lists "
        "only what changed since your previous observation (elements that stayed the "
        "same keep the ids you already saw), mode='text' returns readable text "
        "without the element list. filter keeps only lines containing that substring "
        "(pass an empty string for no filtering). Use it when you need ids you can no "
        "longer see, or to find something in a long page.",
        _obj(
            mode=_str(
                "How much of the page to return: 'full', 'delta' or 'text'.",
                enum=["full", "delta", "text"],
            ),
            filter=_str(
                "Case-insensitive substring filter applied to the returned lines; "
                "pass an empty string for the whole page."
            ),
        ),
    ),
    ToolSpec(
        "goto",
        "Navigate the current tab to the given address and observe the resulting "
        "page. Only use an address that the task or the current page actually gives "
        "you; never guess one.",
        _obj(url=_str("Absolute address to open in the current tab.")),
    ),
    ToolSpec(
        "click",
        "Click one addressable element by the integer id it has in the latest page "
        "observation. intent is a one-line statement of why you are clicking, shown "
        "to the human in the audit trail.",
        _obj(
            element_id=_int("Element id from the latest observation.", minimum=0),
            intent=_str("Why this click, in a few words."),
        ),
    ),
    ToolSpec(
        "type_text",
        "Type text into a form field. element_id is the id from the latest "
        "observation, or 0 to type into the element that currently has focus. "
        "submit=true presses Enter afterwards (handy for search fields); clear=true "
        "empties the field first. Never type secrets: passwords and payment data are "
        "entered by the human.",
        _obj(
            element_id=_int("Element id of the field, or 0 for the focused element.", minimum=0),
            text=_str("The exact text to type."),
            submit=_bool("Press Enter after typing."),
            clear=_bool("Clear the field before typing."),
        ),
    ),
    ToolSpec(
        "press_key",
        "Send one keyboard key to the page: Enter, Escape, Tab, PageDown, PageUp, "
        "ArrowDown, ArrowUp, ArrowLeft, ArrowRight, Home, End or Backspace.",
        _obj(key=_str("Key name to press.")),
    ),
    ToolSpec(
        "scroll",
        "Scroll the page. direction is one of up, down, top, bottom; amount is the "
        "number of pixels for up/down (0 scrolls about one screenful).",
        _obj(
            direction=_str(
                "Scroll direction.", enum=["up", "down", "top", "bottom"]
            ),
            amount=_int("Pixels to scroll for up/down; 0 means one screenful.", minimum=0),
        ),
    ),
    ToolSpec(
        "go_back",
        "Go back to the previous page in this tab (browser Back).",
        _obj(),
    ),
    ToolSpec(
        "wait_for",
        "Wait for time or for content to appear. seconds is a fixed delay, text is "
        "an optional substring to wait for, timeout_seconds bounds that wait. Use it "
        "when the page is still loading or when content appears after a delay.",
        _obj(
            seconds=_num("Fixed delay in seconds (0 if you only wait for text).", minimum=0),
            text=_str("Substring to wait for; pass an empty string to wait only for time."),
            timeout_seconds=_num("How long to wait for the text before giving up.", minimum=0),
        ),
    ),
    ToolSpec(
        "tabs",
        "Manage browser tabs. action='list' reports the open tabs, 'open' opens url "
        "in a new tab, 'switch' focuses the tab with the given index, 'close' closes "
        "the tab with that index and returns to another one.",
        _obj(
            action=_str("Tab operation.", enum=["list", "open", "switch", "close"]),
            index=_int("Tab index for switch/close; 0 otherwise.", minimum=0),
            url=_str("Address to open for action='open'; empty string otherwise."),
        ),
    ),
    ToolSpec(
        "ask_page",
        "Ask a focused question about the readable content of the current page and "
        "get back a short answer with a quoted snippet and a confidence value. Use it "
        "for facts buried in long text (a value, a date, a name, a confirmation, an "
        "error message) instead of reading the whole page yourself.",
        _obj(question=_str("The precise question about the current page.")),
    ),
    ToolSpec(
        "screenshot",
        "Save a picture of the current page for the human and for the audit trail. "
        "label is a short name for the file.",
        _obj(label=_str("Short label for the file name.")),
    ),
    ToolSpec(
        "notes",
        "Update your persistent scratchpad. The scratchpad stays in your context "
        "even after older steps are compacted away, so write down what you must not "
        "forget: progress so far, values read from the page, constraints, the current "
        "sub-goal. The text is appended as a new line; pass the single word clear to "
        "erase the scratchpad.",
        _obj(text=_str("Line to append to the scratchpad, or 'clear' to reset it.")),
    ),
    ToolSpec(
        "ask_user",
        "Pause and ask the human a question in the terminal. Use it when the task "
        "needs something only the human has or can decide (a login, a one-time code, "
        "a choice between options, permission for a risky step). Never type "
        "credentials or payment data into the page yourself.",
        _obj(question=_str("The question to put to the human.")),
    ),
    ToolSpec(
        "finish",
        "End the run. success must be true only if the page state proves the task is "
        "complete. answer is the final result for the human in plain language: the "
        "value or confirmation you observed when successful, or an honest account of "
        "what you achieved and what blocked you when not.",
        _obj(
            success=_bool("True only if the page proves the task is complete."),
            answer=_str("Final answer for the human."),
        ),
    ),
]

TOOL_NAMES: list[str] = [spec.name for spec in TOOL_SPECS]
SPEC_BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}

#: Tools that go to the browser executor.
BROWSER_TOOLS: frozenset[str] = frozenset(
    {"goto", "click", "type_text", "press_key", "scroll", "go_back", "wait_for", "tabs", "screenshot"}
)
#: Tools implemented in this module.
LOCAL_TOOLS: frozenset[str] = frozenset({"page_outline", "ask_page", "notes", "ask_user", "finish"})

VALID_TOOL_HINT = "valid tool names: " + ", ".join(TOOL_NAMES)


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #


def _invalid(message: str, call: ToolCall | None = None) -> ToolResult:
    name = getattr(call, "name", "?")
    return ToolResult(
        ok=False,
        summary=f"{message} ({VALID_TOOL_HINT})",
        error=message,
        recovery_hint=(
            f"{name!r} is not a usable tool call. Call one of the tools with the exact "
            "argument names and types from its schema, or finish if the task is over."
        ),
        data={"valid_tools": list(TOOL_NAMES)},
    )


def _coerce(key: str, value: Any, schema: Mapping[str, Any]) -> tuple[Any, str | None]:
    """Coerce one argument to the declared type; return ``(value, error)``."""
    kind = schema.get("type")
    if kind == "string":
        if value is None:
            value = ""
        if isinstance(value, bool):
            return None, f"{key} must be a string, got a boolean"
        if isinstance(value, (int, float)):
            value = str(value)
        if not isinstance(value, str):
            return None, f"{key} must be a string, got {type(value).__name__}"
        enum = schema.get("enum")
        if enum:
            candidate = value.strip()
            if candidate in enum:
                value = candidate
            else:
                lowered = {str(e).lower(): e for e in enum}
                match = lowered.get(candidate.lower())
                if match is None:
                    return None, f"{key} must be one of {list(enum)}, got {value!r}"
                value = match
        return value, None

    if kind == "integer":
        if value is None or (isinstance(value, str) and not value.strip()):
            value = 0
        elif isinstance(value, bool):
            return None, f"{key} must be an integer, got a boolean"
        elif isinstance(value, str):
            text = value.strip()
            if re.fullmatch(r"[+-]?\d+", text):
                value = int(text)
            elif re.fullmatch(r"[+-]?\d+\.0+", text):
                value = int(float(text))
            else:
                return None, f"{key} must be an integer, got {value!r}"
        elif isinstance(value, float):
            if float(value).is_integer():
                value = int(value)
            else:
                return None, f"{key} must be an integer, got {value!r}"
        elif not isinstance(value, int):
            return None, f"{key} must be an integer, got {type(value).__name__}"
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            return None, f"{key} must be >= {minimum}, got {value}"
        return value, None

    if kind == "number":
        if value is None or (isinstance(value, str) and not value.strip()):
            value = 0.0
        elif isinstance(value, bool):
            return None, f"{key} must be a number, got a boolean"
        elif isinstance(value, str):
            try:
                value = float(value.strip())
            except ValueError:
                return None, f"{key} must be a number, got {value!r}"
        elif not isinstance(value, (int, float)):
            return None, f"{key} must be a number, got {type(value).__name__}"
        value = float(value)
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            return None, f"{key} must be >= {minimum}, got {value}"
        return value, None

    if kind == "boolean":
        if value is None:
            value = False
        elif isinstance(value, bool):
            pass
        elif isinstance(value, str):
            text = value.strip().lower()
            if text in ("true", "yes", "1", "y"):
                value = True
            elif text in ("false", "no", "0", "n", ""):
                value = False
            else:
                return None, f"{key} must be a boolean, got {value!r}"
        elif isinstance(value, (int, float)):
            value = bool(value)
        else:
            return None, f"{key} must be a boolean, got {type(value).__name__}"
        return value, None

    return value, None


def validate_args(name: str, args: Any) -> tuple[dict[str, Any], str | None]:
    """Normalise ``args`` against the schema of ``name``.

    Returns ``(clean_args, error)``.  Optional-but-explicit arguments are the rule:
    a missing one is a malformed call, a ``null`` becomes the documented sentinel
    (``""``/``0``/``false``) so a chatty model is still usable.
    """
    spec = SPEC_BY_NAME.get(name)
    if spec is None:
        return {}, f"unknown tool {name!r}"
    if args is None:
        args = {}
    if not isinstance(args, Mapping):
        return {}, f"arguments for {name} must be a JSON object, got {type(args).__name__}"
    props: dict[str, Any] = spec.parameters.get("properties", {})
    clean: dict[str, Any] = {}
    missing: list[str] = []
    problems: list[str] = []
    for key, schema in props.items():
        if key not in args:
            missing.append(key)
            continue
        value, error = _coerce(key, args[key], schema)
        if error:
            problems.append(error)
        else:
            clean[key] = value
    if missing:
        return {}, (
            f"{name} is missing required argument(s): {', '.join(missing)}"
            f" (expected {_arg_summary(spec)})"
        )
    if problems:
        return {}, f"{name}: " + "; ".join(problems)
    return clean, None


def _arg_summary(spec: ToolSpec) -> str:
    props: dict[str, Any] = spec.parameters.get("properties", {})
    if not props:
        return "no arguments"
    return ", ".join(f"{key}: {schema.get('type', 'any')}" for key, schema in props.items())


def extra_arg_names(name: str, args: Any) -> list[str]:
    """Keys the model passed that the schema does not declare (ignored, reported)."""
    spec = SPEC_BY_NAME.get(name)
    if spec is None or not isinstance(args, Mapping):
        return []
    props = spec.parameters.get("properties", {})
    return [key for key in args if key not in props]


# --------------------------------------------------------------------------- #
# Dispatch context
# --------------------------------------------------------------------------- #


@dataclass
class DispatchContext:
    """Mutable view of the run that agent-local tools need.

    The loop owns one instance and keeps ``page``/``history`` up to date; the
    registry only reads it.  ``task`` is carried so sub-agents can be told what
    the run is really about.
    """

    page: PageModel | None = None
    task: str = ""
    step: int = 0
    history: list[str] = field(default_factory=list)

    # -- helpers -----------------------------------------------------------
    @property
    def current_url(self) -> str:
        return self.page.url if self.page is not None else ""

    @property
    def current_title(self) -> str:
        return self.page.title if self.page is not None else ""

    def element_for(self, call: ToolCall) -> Element | None:
        """Resolve ``call.args['element_id']`` against the current page model."""
        if self.page is None:
            return None
        raw = call.args.get("element_id") if isinstance(call.args, Mapping) else None
        try:
            element_id = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if element_id <= 0:
            return None
        return self.page.by_id(element_id)

    def is_password_call(self, call: ToolCall) -> bool:
        if call.name != "type_text":
            return False
        element = self.element_for(call)
        return bool(element is not None and element.type == "password")

    def action_context(self, call: ToolCall | None = None) -> ActionContext:
        element = self.element_for(call) if call is not None else None
        return ActionContext(
            url=self.current_url,
            page_title=self.current_title,
            element=element,
            is_password_field=bool(element is not None and element.type == "password"),
            recent_history=list(self.history[-6:]),
        )


# --------------------------------------------------------------------------- #
# Fallback page rendering
# --------------------------------------------------------------------------- #


def fallback_render_page(
    model: PageModel,
    *,
    budget_tokens: int = 2_000,
    mode: SnapshotMode = "full",
    previous: PageModel | None = None,
) -> str:
    """Minimal renderer used only when no real renderer was wired in.

    ``webpilot.browser.snapshot.render_page_model`` is the production renderer and
    is preferred everywhere; this keeps the agent usable (and testable) without the
    browser package installed.  It follows the same textual contract.
    """
    lines: list[str] = [f"URL: {model.url} | Title: {model.title} | generation {model.generation}"]
    for alert in model.alerts[:8]:
        lines.append(f"ALERTS: {alert}")
    for dialog in model.dialogs[:4]:
        lines.append(f"DIALOG: {dialog}")
    shown = 0
    if mode != "text":
        for element in model.elements:
            if mode == "delta" and previous is not None:
                known = previous.by_id(element.id)
                if known is not None and known.dom_hash == element.dom_hash:
                    continue
            lines.append(element.line())
            shown += 1
        if mode == "delta" and previous is not None:
            skipped = max(0, len(model.elements) - shown)
            if skipped:
                lines.append(f"(unchanged: {skipped} elements, use ids from the previous snapshot)")
    if model.text.strip():
        lines.append("TEXT:")
        lines.extend(f"  {chunk}" for chunk in model.text.splitlines()[:40])
    text = "\n".join(lines)
    target = max(32, budget_tokens)
    trimmed, cut = truncate_to_tokens(text, target)
    for _ in range(4):  # the estimate can overshoot by a token or two
        if not cut or count_tokens(trimmed) <= budget_tokens:
            break
        target = max(16, target - 8)
        trimmed, cut = truncate_to_tokens(text, target)
    return trimmed


def _filter_lines(block: str, needle: str) -> str:
    """Keep only the lines containing ``needle`` (case-insensitive)."""
    if not needle.strip():
        return block
    lowered = needle.strip().lower()
    kept = [line for line in block.splitlines() if lowered in line.lower()]
    if not kept:
        return f"(no line of the current page observation contains {needle!r})"
    return "\n".join(kept)


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #


class ToolRegistry:
    """Executes tool calls and turns every failure into a ``ToolResult``.

    ``dispatch`` never raises: an unknown tool, a malformed argument, a crashing
    executor or a missing dependency all come back as ``ok=False`` with a message
    that tells the model what to do instead.
    """

    def __init__(
        self,
        executor: Any = None,
        config: Config | None = None,
        ui: Any = None,
        policy: Any = None,
        extractor: Any = None,
        context: Any = None,
        *,
        renderer: Callable[..., str] | None = None,
        page_provider: Callable[[str, str], tuple[PageModel | None, str]] | None = None,
        raw_provider: Callable[[], str] | None = None,
        session: Any = None,
    ) -> None:
        self.executor = executor
        self.config = config
        self.ui = ui
        self.policy = policy
        self.extractor = extractor
        self.context = context
        self.renderer = renderer
        self.page_provider = page_provider
        self.raw_provider = raw_provider
        self.session = session

        # -- bookkeeping the loop reads after a dispatch --------------------
        self.calls = 0
        self.errors = 0
        self.subagent_calls = 0
        self.extractor_usage = Usage()
        self.finished = False
        self.finish_success = False
        self.finish_answer = ""
        self.finish_calls = 0
        self.denied = 0

    # ------------------------------------------------------------------ #
    def bind(self, **attrs: Any) -> "ToolRegistry":
        """Fill in collaborators that were not supplied at construction time."""
        for key, value in attrs.items():
            if value is not None and getattr(self, key, None) is None:
                setattr(self, key, value)
        return self

    # ------------------------------------------------------------------ #
    def dispatch(self, call: ToolCall, ctx: DispatchContext | None = None) -> ToolResult:
        """Route one tool call.  Guaranteed not to raise."""
        ctx = ctx if ctx is not None else DispatchContext()
        if isinstance(call, Mapping):  # tolerate a plain dict from a provider
            call = ToolCall(
                name=str(call.get("name") or ""),
                args=dict(call.get("args") or {}),
                id=call.get("id"),  # type: ignore[arg-type]
            )
        if not isinstance(call, ToolCall):
            return _invalid(f"tool call must be an object with a name, got {type(call).__name__}")

        name = (call.name or "").strip()
        if name not in SPEC_BY_NAME:
            return _invalid(f"unknown tool {call.name!r}", call)
        clean, error = validate_args(name, call.args)
        if error:
            return _invalid(error, call)

        extra = extra_arg_names(name, call.args)
        normalised = ToolCall(name=name, args=clean, id=call.id)
        self.calls += 1
        try:
            if name in BROWSER_TOOLS:
                result = self._browser(normalised)
            else:
                result = self._local(name, normalised, ctx)
        except Exception as exc:  # defensive: a tool must never break the loop
            self.errors += 1
            result = ToolResult(
                ok=False,
                summary=f"{name} failed internally: {type(exc).__name__}: {exc}",
                error=str(exc),
                recovery_hint="This is an internal failure of the tool, not the page. Try a different tool.",
            )
        if not isinstance(result, ToolResult):
            result = ToolResult(ok=bool(result), summary=str(result) or f"{name} returned nothing")
        if extra:
            result.data.setdefault("ignored_arguments", extra)
        if not result.summary:
            result.summary = f"{name} completed"
        return result

    # ------------------------------------------------------------------ #
    def _browser(self, call: ToolCall) -> ToolResult:
        if self.executor is None:
            return ToolResult(
                ok=False,
                summary="no browser is attached, so page actions are unavailable",
                error="no executor",
                recovery_hint=(
                    "Use agent-local tools (notes, ask_user, finish) or wait until a "
                    "browser is connected."
                ),
            )
        executor = self.executor
        result = executor.execute(call)
        if isinstance(result, Mapping):
            result = ToolResult(**dict(result))
        return result

    # ------------------------------------------------------------------ #
    def _local(self, name: str, call: ToolCall, ctx: DispatchContext) -> ToolResult:
        if name == "page_outline":
            return self._page_outline(call, ctx)
        if name == "ask_page":
            return self._ask_page(call, ctx)
        if name == "notes":
            return self._notes(call, ctx)
        if name == "ask_user":
            return self._ask_user(call)
        if name == "finish":
            return self._finish(call)
        return _invalid(f"tool {name!r} is not implemented", call)

    # -- page_outline ---------------------------------------------------- #
    def _page_outline(self, call: ToolCall, ctx: DispatchContext) -> ToolResult:
        mode: str = call.args["mode"]
        needle: str = call.args["filter"]
        model: PageModel | None = None
        text = ""
        if self.page_provider is not None:
            model, text = self.page_provider(mode, needle)
        elif ctx.page is not None:
            model = ctx.page
            budget = getattr(self.config, "page_token_budget", 2_000) if self.config else 2_000
            text = self._render(model, budget_tokens=budget, mode="full", previous=None)
        if model is None:
            return ToolResult(
                ok=False,
                summary="there is no page to read yet",
                error="no page observation",
                recovery_hint="Open a page first, or continue with the tools that do not need one.",
            )
        text = _filter_lines(text, needle)
        note = f" (filtered by {needle!r})" if needle.strip() else ""
        return ToolResult(
            ok=True,
            summary=f"page outline attached ({mode}{note}, {len(model.elements)} elements on the page)",
            data={
                "mode": mode,
                "generation": model.generation,
                "url": model.url,
                "page_text": text,
            },
            page=model,
            page_changed=False,
        )

    # -- ask_page -------------------------------------------------------- #
    def _ask_page(self, call: ToolCall, ctx: DispatchContext) -> ToolResult:
        question: str = call.args["question"]
        if self.extractor is None:
            return ToolResult(
                ok=False,
                summary="page reading is unavailable in this run",
                error="no extractor configured",
                recovery_hint="Work from the page observation you have, or ask the human.",
            )
        page_text = ""
        if self.page_provider is not None:
            _, page_text = self.page_provider("text", "")
        elif ctx.page is not None:
            page_text = ctx.page.text
        html_excerpt = ""
        if self.raw_provider is not None:
            # Raw markup is read *here only*, handed to the sub-agent, and never
            # enters the main conversation (see the assignment's hard rules).
            try:
                html_excerpt = str(self.raw_provider() or "")
            except Exception:
                html_excerpt = ""
        result = self.extractor.answer(question, page_text, html_excerpt)
        if not isinstance(result, ExtractionResult):
            result = ExtractionResult(answer=str(result))
        self.subagent_calls += 1
        self.extractor_usage = self.extractor_usage.add(result.usage)
        found = bool(result.answer.strip())
        summary = prompts.extraction_summary(result.answer, result.confidence, found)
        if self.ui is not None:
            self.ui.emit(
                RunEvent(
                    "subagent",
                    f"ask_page: {question} -> {result.answer.strip() or '(not present)'}",
                    {
                        "subagent": "extractor",
                        "question": question,
                        "answer": result.answer,
                        "confidence": result.confidence,
                        "evidence": result.evidence,
                    },
                    level="info",
                )
            )
        return ToolResult(
            ok=True,
            summary=summary,
            data={
                "question": question,
                "answer": result.answer,
                "evidence": result.evidence,
                "confidence": result.confidence,
                "found": found,
            },
        )

    # -- notes ----------------------------------------------------------- #
    def _notes(self, call: ToolCall, ctx: DispatchContext) -> ToolResult:
        text: str = call.args["text"]
        if self.context is None:
            return ToolResult(
                ok=False,
                summary="the scratchpad is unavailable in this run",
                error="no context manager",
            )
        notes = self.context.update_notes(text)
        return ToolResult(
            ok=True,
            summary=f"scratchpad updated ({len(notes)} characters kept)",
            data={"notes": notes},
        )

    # -- ask_user -------------------------------------------------------- #
    def _ask_user(self, call: ToolCall) -> ToolResult:
        question: str = call.args["question"]
        if self.ui is None:
            return ToolResult(
                ok=False,
                summary="no human is available to answer right now",
                error="no UI",
                recovery_hint="Continue with what the page gives you, or finish honestly.",
            )
        answer = str(self.ui.ask(question) or "").strip()
        self.ui.emit(
            RunEvent("user", f"human answered: {answer or '(no answer)'}", {"question": question, "answer": answer})
        )
        return ToolResult(
            ok=True,
            summary=f"the human answered: {answer or '(no answer given)'}",
            data={"question": question, "answer": answer},
        )

    # -- finish ---------------------------------------------------------- #
    def _finish(self, call: ToolCall) -> ToolResult:
        self.finished = True
        self.finish_calls += 1
        self.finish_success = bool(call.args["success"])
        self.finish_answer = str(call.args["answer"])
        return ToolResult(
            ok=True,
            summary=f"finish requested (success={self.finish_success})",
            data={"success": self.finish_success, "answer": self.finish_answer},
        )

    # ------------------------------------------------------------------ #
    def _render(
        self,
        model: PageModel,
        *,
        budget_tokens: int,
        mode: SnapshotMode = "full",
        previous: PageModel | None = None,
    ) -> str:
        renderer = self.renderer
        if renderer is not None:
            try:
                return renderer(model, budget_tokens=budget_tokens, mode=mode, previous=previous)
            except Exception:
                pass
        return fallback_render_page(model, budget_tokens=budget_tokens, mode=mode, previous=previous)


__all__ = [
    "TOOL_SPECS",
    "TOOL_NAMES",
    "SPEC_BY_NAME",
    "BROWSER_TOOLS",
    "LOCAL_TOOLS",
    "VALID_TOOL_HINT",
    "DispatchContext",
    "ToolRegistry",
    "validate_args",
    "extra_arg_names",
    "fallback_render_page",
]
