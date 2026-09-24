"""Core types shared by every webpilot subsystem.

This module is the *spine* of the project: it defines the data structures that
the browser layer, the LLM layer, the agent loop, the security layer and the UI
exchange.  Every subsystem may import from here, nothing here imports from them
(only the standard library), which keeps the import graph acyclic.

Read ``docs/INTERFACES.md`` for the behavioural contracts that go with these
types.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Literal, Protocol, Sequence

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

ElementRole = Literal[
    "button", "link", "textbox", "checkbox", "radio", "select", "option",
    "menuitem", "tab", "switch", "slider", "file", "submit", "label", "other",
]
RiskLevel = Literal["safe", "caution", "destructive"]
ProviderName = Literal[
    "openai", "anthropic", "deepseek", "openrouter", "ollama",
    "openai_compatible", "fake",
]
SnapshotMode = Literal["full", "delta", "text"]


# --------------------------------------------------------------------------- #
# Page model - the compact, token-bounded view of a web page
# --------------------------------------------------------------------------- #

@dataclass
class Element:
    """One interactive (or otherwise addressable) node of the page.

    ``id`` is the *only* handle the model ever uses to target an element; it is
    assigned by the page sniffer on every snapshot generation and never
    hard-coded anywhere in the codebase.
    """

    id: int
    role: ElementRole
    name: str                      # accessible name, whitespace-normalised, <= 120 chars
    tag: str                       # lower-case tag name
    type: str | None = None        # input type attribute, if any
    value: str | None = None       # current value (inputs/options/selected text)
    placeholder: str | None = None
    href: str | None = None        # for links: absolute URL trimmed to host + path
    disabled: bool = False
    checked: bool | None = None
    required: bool = False
    expanded: bool | None = None
    invalid: bool = False
    in_viewport: bool = True
    frame: int = 0                 # index into PageModel.frames; 0 = main frame
    options: list[str] = field(default_factory=list)   # for select/combobox
    dom_hash: str = ""             # role+name+value+state hash, used for delta rendering
    selector_hint: str = ""        # human-readable CSS path, for logs only, never for dispatch
    note: str = ""                 # e.g. "overlapped by #cookie-banner"

    def line(self) -> str:
        """One-line rendering contract used by every renderer and by tests."""
        bits = [f"[{self.id}]", self.role, repr(self.name)]
        if self.value:
            bits.append(f"value={self.value!r}")
        if self.placeholder and not self.value:
            bits.append(f"placeholder={self.placeholder!r}")
        if self.href:
            bits.append(f"href={self.href}")
        if self.type and self.type not in ("text",):
            bits.append(f"type={self.type}")
        if self.options:
            bits.append("options=[" + ", ".join(repr(o) for o in self.options[:12]) + "]")
        state = []
        if self.disabled:
            state.append("disabled")
        if self.checked is not None:
            state.append("checked" if self.checked else "unchecked")
        if self.required:
            state.append("required")
        if self.invalid:
            state.append("invalid")
        if not self.in_viewport:
            state.append("off-screen")
        if self.note:
            state.append(self.note)
        if state:
            bits.append("(" + ", ".join(state) + ")")
        return " ".join(bits)


@dataclass
class TabInfo:
    index: int
    url: str
    title: str
    active: bool = False


@dataclass
class PageModel:
    """Snapshot of one moment of one page - the agent's entire perception."""

    url: str
    title: str
    generation: int
    elements: list[Element] = field(default_factory=list)
    text: str = ""                       # ranked visible text, already truncated
    alerts: list[str] = field(default_factory=list)   # toasts, validation errors, dialogs
    dialogs: list[str] = field(default_factory=list)  # blocking modal text (native + overlaid)
    scroll: dict[str, Any] = field(default_factory=dict)   # y, max_y, viewport_h, at_bottom
    tabs: list[TabInfo] = field(default_factory=list)
    frames: list[str] = field(default_factory=list)   # frame urls, index 0 = main frame
    focused_id: int | None = None
    truncated: bool = False              # text or elements were dropped to fit the budget
    dropped_elements: int = 0
    text_dropped_chars: int = 0
    token_estimate: int = 0
    full_text_chars: int = 0             # size of the raw text before truncation
    html_chars: int = 0                  # size of the raw HTML (what we *did not* send)
    errors: list[str] = field(default_factory=list)   # console errors / failed requests
    screenshot_path: str | None = None
    captured_at: float = field(default_factory=time.time)

    def by_id(self, element_id: int) -> Element | None:
        for el in self.elements:
            if el.id == element_id:
                return el
        return None

    def page_hash(self) -> str:
        """Hash of the parts of the page an agent cares about (for loop detection)."""
        import hashlib

        h = hashlib.sha256()
        h.update(self.url.encode())
        h.update(self.title.encode())
        for el in self.elements:
            h.update((el.dom_hash or el.line()).encode())
        h.update(self.text[:2000].encode())
        return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Tool calling
# --------------------------------------------------------------------------- #

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]        # JSON Schema object
    strict: bool = True

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": self.strict,
            },
        }

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str | None = None

    def signature(self) -> str:
        """Stable string used to detect repeated actions (loop detection)."""
        return self.name + "(" + ",".join(f"{k}={self.args[k]!r}" for k in sorted(self.args)) + ")"


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cached_tokens + other.cached_tokens,
            self.calls + other.calls,
        )

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class ToolResult:
    """What every tool returns to the agent loop."""

    ok: bool
    summary: str                               # one line, always shown to the model
    data: dict[str, Any] = field(default_factory=dict)
    page: PageModel | None = None              # fresh page model produced by the action
    page_changed: bool = False
    error: str | None = None
    recovery_hint: str | None = None
    screenshot_path: str | None = None

    def to_llm_text(self) -> str:
        parts = [("OK: " if self.ok else "FAILED: ") + self.summary]
        if not self.ok and self.recovery_hint:
            parts.append("hint: " + self.recovery_hint)
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# LLM layer
# --------------------------------------------------------------------------- #

@dataclass
class LLMMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = ""
    raw: Any = None


class LLMClient(Protocol):
    """Every provider implementation (OpenAI, Anthropic, OpenAI-compatible, fake)."""

    provider: str
    model: str

    def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSpec] | None = None,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse: ...

    def count_tokens(self, text: str) -> int: ...


# --------------------------------------------------------------------------- #
# Security layer
# --------------------------------------------------------------------------- #

@dataclass
class ActionRisk:
    level: RiskLevel = "safe"
    reasons: list[str] = field(default_factory=list)
    matched_rules: list[str] = field(default_factory=list)

    @property
    def requires_confirmation(self) -> bool:
        return self.level == "destructive"


@dataclass
class ActionContext:
    """Everything the policy engine may look at besides the tool call itself."""

    url: str = ""
    page_title: str = ""
    element: Element | None = None
    is_password_field: bool = False
    recent_history: list[str] = field(default_factory=list)


ConfirmFn = Callable[[str, str], bool]      # (prompt, details) -> approved?


# --------------------------------------------------------------------------- #
# Sub-agents
# --------------------------------------------------------------------------- #

@dataclass
class ExtractionResult:
    answer: str
    evidence: str = ""
    confidence: float = 0.0
    usage: Usage = field(default_factory=Usage)
    subagent: str = "extractor"


@dataclass
class FailureContext:
    task: str
    tool_call: ToolCall
    error: str
    attempts: int
    page: PageModel | None = None
    history: list[str] = field(default_factory=list)


@dataclass
class RecoveryProposal:
    advice: str
    alternative_actions: list[ToolCall] = field(default_factory=list)
    give_up: bool = False
    usage: Usage = field(default_factory=Usage)
    subagent: str = "recovery"


# --------------------------------------------------------------------------- #
# Agent run bookkeeping
# --------------------------------------------------------------------------- #

@dataclass
class StepRecord:
    index: int
    tool_call: ToolCall
    result_summary: str
    ok: bool
    page_url: str = ""
    page_title: str = ""
    page_hash: str = ""
    page_changed: bool = False
    duration_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    risk: str = "safe"
    confirmed: bool | None = None
    note: str = ""
    screenshot_path: str | None = None

    def digest(self) -> str:
        status = "ok" if self.ok else "FAIL"
        return f"#{self.index} {self.tool_call.name} -> {status}: {self.result_summary}"


@dataclass
class RunEvent:
    """Everything the UI needs to render. ``kind`` drives formatting."""

    kind: Literal[
        "start", "thinking", "tool_call", "tool_result", "page", "risk",
        "confirm", "subagent", "user", "error", "finish", "context", "info",
    ]
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    level: str = "info"          # info | warn | error | success


@dataclass
class RunResult:
    task: str
    success: bool
    answer: str
    steps: int
    duration_s: float
    usage: Usage
    transcript_path: str | None = None
    error: str | None = None
    subagent_calls: int = 0
    confirmations: int = 0
    recoveries: int = 0


# --------------------------------------------------------------------------- #
# UI contracts (implemented by the terminal UI, faked by tests)
# --------------------------------------------------------------------------- #

class AgentUI(Protocol):
    def emit(self, event: RunEvent) -> None: ...

    def confirm(self, prompt: str, details: str) -> bool: ...

    def ask(self, question: str) -> str: ...

    def drain_instructions(self) -> list[str]:
        """Instructions the human typed while the agent was working (may be empty)."""
        ...
