"""``webpilot`` command line entry point and the rich terminal UI.

Two responsibilities live here:

* :class:`TerminalUI` - the only thing in the project that is allowed to print to
  stdout.  It renders the live trace (tool calls *with their arguments*, the risk
  verdict, the confirmation panel, the page delta, sub-agent calls and token
  usage) and it owns the human channel: a daemon stdin reader thread that queues
  instructions typed while the agent is working, plus the blocking y/N prompt
  used by the security policy.
* :func:`main` - argparse via :func:`webpilot.config.config_from_args` (the flags
  are *not* duplicated here), exit-code policy and the ``run``/interactive modes.

Everything heavy (browser, LLM providers, the agent loop, the context manager) is
imported lazily inside the small ``_build_*`` factories below, so ``webpilot
--help`` / ``--version`` and the unit tests work with no API key, no Chromium and
no other subsystem importable.  The factories are also the seam the tests use to
substitute fakes.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import traceback
from collections import deque
from typing import Any, Iterable, Sequence

from .config import Config, config_from_args
from .errors import ConfigError, WebpilotError
from .security.policy import redact_args, redact_secrets
from .tokenizer import count_tokens
from .types import AgentUI, RunEvent, RunResult, Usage

try:  # rich is a hard dependency, but --help/--version must survive without it
    from rich import box
    from rich.console import Console, Group
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _RICH_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on the environment
    box = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment]
    Group = Panel = Table = Text = None  # type: ignore[assignment]
    escape = lambda text: text  # type: ignore[assignment]
    _RICH_IMPORT_ERROR = exc


# --------------------------------------------------------------------------- #
# versioning
# --------------------------------------------------------------------------- #

def _detect_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("webpilot")
        except PackageNotFoundError:
            pass
    except Exception:  # pragma: no cover - importlib is stdlib, defensive only
        pass
    return "0.1.0"


__version__ = _detect_version()

#: subcommand words accepted for compatibility with ``webpilot run "task"``
_SUBCOMMANDS = ("run", "start", "task")

_RISK_STYLE = {"safe": "green", "caution": "yellow", "destructive": "bold red"}
_LEVEL_STYLE = {"info": "cyan", "warn": "yellow", "warning": "yellow",
                "error": "red", "success": "green"}

#: bare tokens that can only be an answer to a yes/no question.  A human who
#: types "y" *before* the confirmation panel appears must not lose that answer
#: (and a bare "y" is never a sensible instruction for the agent).
_ANSWER_TOKENS = {"y", "yes", "yeah", "ok", "да", "д", "n", "no", "nope", "нет", "н"}
#: how long an answer typed slightly too early stays valid
_PRE_ANSWER_TTL = 30.0


# --------------------------------------------------------------------------- #
# small formatting helpers
# --------------------------------------------------------------------------- #

def _clip(text: str, limit: int) -> str:
    text = str(text).replace("\n", " ⏎ ")
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


def _fmt_value(value: Any, limit: int = 60) -> str:
    if isinstance(value, str):
        return repr(_clip(value, limit))
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return repr(value)
    try:
        return _clip(json.dumps(value, ensure_ascii=False), limit)
    except (TypeError, ValueError):
        return _clip(repr(value), limit)


def fmt_args(args: dict[str, Any] | None, limit: int = 220) -> str:
    """Render tool arguments for the trace, with credentials redacted."""
    safe = redact_args(args or {})
    parts = [f"{k}={_fmt_value(v)}" for k, v in safe.items()]
    return _clip(", ".join(parts), limit) if parts else ""


def _as_usage(value: Any) -> Usage | None:
    if isinstance(value, Usage):
        return value
    if isinstance(value, dict):
        try:
            return Usage(
                prompt_tokens=int(value.get("prompt_tokens", 0) or 0),
                completion_tokens=int(value.get("completion_tokens", 0) or 0),
                cached_tokens=int(value.get("cached_tokens", 0) or 0),
                calls=int(value.get("calls", 0) or 0),
            )
        except (TypeError, ValueError):
            return None
    return None


def _fmt_usage(usage: Usage) -> str:
    calls = f"{usage.calls} call" + ("s" if usage.calls != 1 else "")
    cached = f", {usage.cached_tokens:,} cached" if usage.cached_tokens else ""
    return (
        f"{usage.prompt_tokens:,} prompt + {usage.completion_tokens:,} completion "
        f"= {usage.total:,} tokens ({calls}{cached})"
    )


def _fmt_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m {rest:.0f}s"


def _default_console() -> Any:
    if Console is None:
        raise ConfigError(
            "the 'rich' package is required for the terminal UI: pip install rich"
        )
    return Console()


# --------------------------------------------------------------------------- #
# the terminal UI
# --------------------------------------------------------------------------- #

class TerminalUI(AgentUI):
    """rich-based live trace + the human input channel.

    Parameters
    ----------
    console:
        a ``rich.console.Console`` (tests pass ``Console(record=True)``).
    verbose:
        also render model reasoning, risk notes for safe actions and keep more
        context detail.
    stdin:
        the stream the human types into (defaults to ``sys.stdin``).
    interactive:
        force the input thread on/off; ``None`` means ``stdin.isatty()``.  A
        pipe or a capture device means *not* interactive: the UI then never
        blocks and ``confirm()`` answers NO (the safe default).
    scripted_answers:
        deterministic answers for ``confirm()``/``ask()`` (test seam; consumed
        left to right).
    input_thread:
        start the stdin reader immediately (it otherwise starts with the run).
    """

    def __init__(
        self,
        console: Any = None,
        *,
        verbose: bool = False,
        stdin: Any = None,
        interactive: bool | None = None,
        scripted_answers: Sequence[str] | None = None,
        input_thread: bool = False,
        answer_timeout: float = 300.0,
    ) -> None:
        self.console = console if console is not None else _default_console()
        self.verbose = bool(verbose)
        self._stdin = stdin if stdin is not None else sys.stdin
        self.interactive = self._is_interactive() if interactive is None else bool(interactive)
        self.answer_timeout = float(answer_timeout)

        self._lock = threading.RLock()
        self._instructions: deque[str] = deque()
        self._answers: "queue.Queue[str]" = queue.Queue()
        self._scripted: deque[str] = deque(scripted_answers or [])
        self._pre_answers: deque[tuple[float, str]] = deque(maxlen=5)
        self._prompting = threading.Event()
        self._stop = threading.Event()
        self._skip = threading.Event()
        self._closed = threading.Event()
        self._reader: threading.Thread | None = None

        #: public counters/state, also used by ``\t``
        self.events: list[RunEvent] = []
        self.usage = Usage()
        self.final_usage: Usage | None = None
        self._task = ""
        self.steps = 0
        self.tool_calls = 0
        self.confirmations = 0
        self.approvals = 0
        self.denials = 0
        self.instructions: list[str] = []
        self.risk_notes = 0
        self._header_rendered = False
        if input_thread:
            self.start_input_reader()

    # ------------------------------------------------------------------ #
    # input plumbing
    # ------------------------------------------------------------------ #
    def _is_interactive(self) -> bool:
        try:
            return bool(self._stdin is not None and self._stdin.isatty())
        except Exception:  # closed/broken stream
            return False

    @property
    def stop_requested(self) -> bool:
        """The human typed ``\\q``; the loop may finish early."""
        return self._stop.is_set()

    def request_stop(self) -> None:
        self._stop.set()

    def take_skip(self) -> bool:
        """True once per ``\\s``: skip the action that is about to run."""
        if self._skip.is_set():
            self._skip.clear()
            return True
        return False

    def start_input_reader(self) -> None:
        """Start the daemon stdin reader (idempotent, never started when dumb)."""
        if not self.interactive or self._closed.is_set():
            return
        if self._reader is not None and self._reader.is_alive():
            return
        self._reader = threading.Thread(
            target=self._read_loop, name="webpilot-stdin", daemon=True
        )
        self._reader.start()

    def _read_loop(self) -> None:  # pragma: no cover - exercised via tests too
        stream = self._stdin
        while not self._closed.is_set():
            try:
                line = stream.readline()
            except Exception:
                break
            if line == "" or line is None:  # EOF (pipe closed, tests, Ctrl-D)
                break
            text = str(line).strip()
            if not text:
                continue
            try:
                self._handle_line(text)
            except Exception:  # a bad line must never kill the UI
                continue
        self._reader_done()

    def _reader_done(self) -> None:
        with self._lock:
            self._print("[dim]stdin closed: no more human instructions[/dim]")

    def _handle_line(self, text: str) -> None:
        if self._prompting.is_set():
            # A question is waiting: the answer goes to the answer queue.
            self._answers.put(text)
            return
        if text.strip().lower() in _ANSWER_TOKENS:
            # typed a fraction of a second before the panel appeared: keep it
            # for the next question instead of losing it or feeding it to the agent
            self._pre_answers.append((time.monotonic(), text))
            self._print(
                f"[magenta]⌨ recorded {escape(_clip(text, 20))!r} as the answer to the next question[/magenta]"
            )
            return
        if text.startswith("\\"):
            cmd = text[1:].strip().lower()
            if cmd in ("q", "quit", "stop", "exit", "выход"):
                self._stop.set()
                self._print("[bold red]\\q[/bold red] stop requested — the agent will halt at the next step")
            elif cmd in ("s", "skip"):
                self._skip.set()
                self._print("[bold yellow]\\s[/bold yellow] skip marker set for the next action")
            elif cmd in ("t", "stats"):
                self._print_stats()
            elif cmd in ("h", "help", "?"):
                self._print_help()
            else:
                self._print(f"[yellow]unknown command {escape(text)!r} (\\h for help)[/yellow]")
            return
        with self._lock:
            self._instructions.append(text)
            self.instructions.append(text)
        self._print(f"[magenta]⌨ instruction queued:[/magenta] {escape(_clip(text, 120))}")

    def drain_instructions(self) -> list[str]:
        """Instructions typed while the agent worked.  Never blocks."""
        with self._lock:
            items = list(self._instructions)
            self._instructions.clear()
        return items

    def _read_line(self, prompt_text: str) -> str:
        """One line of human input for a blocking question.

        Raises :class:`EOFError` when no answer can be obtained, so the caller
        can fall back to its safe default instead of hanging forever.
        """
        if self._scripted:
            return self._scripted.popleft()
        while self._pre_answers:
            recorded, value = self._pre_answers.popleft()
            if time.monotonic() - recorded <= _PRE_ANSWER_TTL:
                with self._lock:
                    self.console.print(
                        f"[dim]using the answer you typed a moment ago: {escape(_clip(value, 20))!r}[/dim]"
                    )
                return value
        with self._lock:
            self.console.print(prompt_text, end="")
        if self._reader is not None and self._reader.is_alive():
            self._prompting.set()
            try:
                try:
                    return self._answers.get(timeout=self.answer_timeout)
                except queue.Empty as exc:
                    raise EOFError("no answer from the human in time") from exc
            finally:
                self._prompting.clear()
        stream = self._stdin
        if stream is None:
            raise EOFError("no stdin available")
        line = stream.readline()
        if line == "" or line is None:
            raise EOFError("stdin is at EOF")
        return str(line).strip()

    def close(self) -> None:
        """Stop the reader thread (the daemon dies with the process anyway)."""
        self._closed.set()
        self._prompting.set()  # unblock a waiting question
        reader, self._reader = self._reader, None
        if reader is not None and reader.is_alive():
            reader.join(timeout=0.2)

    # ------------------------------------------------------------------ #
    # AgentUI: questions
    # ------------------------------------------------------------------ #
    def confirm(self, prompt: str, details: str = "") -> bool:
        """Render the destructive-action panel and read a yes/no answer."""
        with self._lock:
            self.confirmations += 1
            body: list[Any] = [Text.from_markup(f"[bold red]{escape(prompt)}[/bold red]")]
            if details:
                body.append(Text(details, style="default"))
            self.console.print(
                Panel(
                    Group(*body),
                    title="[bold red]⚠ confirmation required[/bold red]",
                    subtitle="destructive action — approve only if this is what you want",
                    border_style="red",
                    box=box.HEAVY,
                )
            )
        if not self.interactive and not self._scripted:
            with self._lock:
                self.console.print(
                    "[bold yellow]non-interactive session: no answer possible, "
                    "defaulting to NO (safe default)[/bold yellow]"
                )
            self.denials += 1
            return False
        while True:
            try:
                raw = self._read_line("Approve this action? [y/N] ").strip().lower()
            except EOFError:
                with self._lock:
                    self.console.print("[yellow]no answer available: defaulting to NO[/yellow]")
                self.denials += 1
                return False
            if raw in ("y", "yes", "yeah", "да", "д"):
                with self._lock:
                    self.console.print("[bold green]approved by the human[/bold green]")
                self.approvals += 1
                return True
            if raw in ("", "n", "no", "nope", "нет", "н"):
                with self._lock:
                    self.console.print("[bold red]denied by the human[/bold red]")
                self.denials += 1
                return False
            with self._lock:
                self.console.print("[yellow]please answer 'y' or 'n'[/yellow]")

    def ask(self, question: str) -> str:
        """Ask the human something that is not a confirmation."""
        self.emit(RunEvent(kind="user", message=question, data={"question": question}))
        if not self.interactive and not self._scripted:
            with self._lock:
                self.console.print("[yellow]non-interactive: returning an empty answer[/yellow]")
            return ""
        try:
            return self._read_line("answer (Enter to skip)> ").strip()
        except EOFError:
            return ""

    def prompt_for_task(self) -> str | None:
        """Interactive mode: ask for the task when none was given."""
        if not self.interactive and not self._scripted:
            return None
        with self._lock:
            self.console.print("[bold]Describe the task for the agent[/bold] (empty line aborts):")
        try:
            line = self._read_line("task> ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        return line or None

    # ------------------------------------------------------------------ #
    # AgentUI: the trace
    # ------------------------------------------------------------------ #
    def emit(self, event: RunEvent) -> None:
        """Render one event (``_r_<kind>``, falling back to ``_r_info``)."""
        with self._lock:
            if len(self.events) < 10_000:
                self.events.append(event)
            delta = self._absorb(event)
            renderer = getattr(self, f"_r_{event.kind}", None) or self._r_info
            try:
                renderer(event)
                if delta is not None and event.kind != "finish" and delta.total:
                    self._print(
                        f"     [dim]tokens: +{delta.prompt_tokens:,} prompt / "
                        f"+{delta.completion_tokens:,} completion "
                        f"(cumulative {self.usage.total:,})[/dim]"
                    )
            except Exception as exc:  # rendering must never break the run
                self._print(f"[red]ui render error: {type(exc).__name__}: {exc}[/red]")

    def _absorb(self, event: RunEvent) -> Usage | None:
        """Accumulate token usage.  Returns the delta this event contributed."""
        data = event.data or {}
        usage = _as_usage(data.get("usage"))
        result = data.get("result")
        if usage is None and result is not None:
            usage = _as_usage(getattr(result, "usage", None))
        if event.kind == "finish":
            # the run summary is authoritative; never double count it on top of
            # the per-step usage events
            if usage is not None:
                self.final_usage = usage
                if usage.total >= self.usage.total:
                    self.usage = usage
            return None
        delta: Usage | None = None
        if usage is not None:
            delta = usage
        elif data.get("prompt_tokens") or data.get("completion_tokens"):
            delta = Usage(
                prompt_tokens=int(data.get("prompt_tokens") or 0),
                completion_tokens=int(data.get("completion_tokens") or 0),
                cached_tokens=int(data.get("cached_tokens") or 0),
                calls=1,
            )
        if delta is not None:
            self.usage = self.usage.add(delta)
        if event.kind == "tool_call":
            self.tool_calls += 1
        if event.kind == "risk":
            level = data.get("level") or getattr(data.get("risk"), "level", "safe")
            if str(level) != "safe":
                self.risk_notes += 1
        return delta

    def _print(self, renderable: Any) -> None:
        self.console.print(renderable)

    def _picked(self, data: dict[str, Any], *names: str, default: str = "") -> str:
        for name in names:
            value = data.get(name)
            if value not in (None, ""):
                return str(value)
        return default

    # -- individual renderers ------------------------------------------- #
    # One method per RunEvent.kind: ``_r_<kind>``.  `emit` dispatches by name and
    # falls back to `_r_info`, so the loop may add event kinds without touching
    # this class.  Every renderer is defensive: it tolerates missing keys and
    # never raises into the run (a broken line is reported, the run continues).
    # The renderers accept both the CLI's own event shapes and the loop's.
    def _r_start(self, event: RunEvent) -> None:
        data = dict(event.data or {})
        self.start_input_reader()
        task = self._picked(data, "task", default=event.message)
        if self._header_rendered:
            # the loop emits its own start event right after run_task's header
            if task and task != self._task:
                self._print(f"[bold cyan]▶[/bold cyan] {escape(task)}")
            return
        self._task = task
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        table.add_row("task", escape(task))
        provider = data.get("provider")
        if provider:
            table.add_row("provider", f"{escape(str(provider))} · {escape(str(data.get('model', '?')))}")
        for key, label in (
            ("confirm_mode", "confirm mode"),
            ("max_steps", "step budget"),
            ("transcript_dir", "transcripts"),
            ("profile", "profile"),
        ):
            if data.get(key) not in (None, ""):
                table.add_row(label, escape(str(data[key])))
        if data.get("headless") is not None:
            table.add_row(
                "browser",
                "[yellow]headless[/yellow]" if data.get("headless") else "visible (headful)",
            )
        mode = "interactive (\\h for help, \\q to stop)" if self.interactive else "no stdin: instructions disabled"
        table.add_row("human channel", mode)
        self.console.print(
            Panel(table, title="[bold]webpilot[/bold]",
                  subtitle=f"v{__version__}", border_style="cyan", box=box.ROUNDED)
        )
        self._header_rendered = True

    def _r_thinking(self, event: RunEvent) -> None:
        text = self._picked(event.data or {}, "text", default=event.message)
        if not text:
            return
        self._print(f"[dim italic]… {escape(_clip(text, 200))}[/dim italic]")

    def _r_tool_call(self, event: RunEvent) -> None:
        data = dict(event.data or {})
        self.steps += 1
        # the loop may hand over a ToolCall object instead of pre-split keys
        call = data.get("call") or data.get("tool_call")
        if call is not None and hasattr(call, "name"):
            name = str(getattr(call, "name"))
            args = getattr(call, "args", {}) or {}
        else:
            name = self._picked(data, "tool", "tool_name", "name", default=event.message or "?")
            args = data.get("args") or data.get("arguments") or {}
        number = data.get("step") if isinstance(data.get("step"), int) else self.steps
        rendered = fmt_args(args if isinstance(args, dict) else {"args": args})
        line = f"[dim]{number:>2}[/dim] [bold cyan]→ {escape(name)}[/bold cyan]({rendered})"
        self._print(line)

    def _r_tool_result(self, event: RunEvent) -> None:
        data = dict(event.data or {})
        ok = bool(data.get("ok", event.level != "error"))
        summary = self._picked(data, "summary", "message", default=event.message)
        mark, style = ("✓", "green") if ok else ("✗", "red")
        line = f"   [{style}]{mark}[/{style}] {escape(_clip(redact_secrets(summary), 240))}"
        duration = data.get("duration_s")
        if isinstance(duration, (int, float)) and duration:
            line += f" [dim]({_fmt_duration(float(duration))})[/dim]"
        self._print(line)
        if not ok and data.get("recovery_hint"):
            self._print(f"     [yellow]hint: {escape(str(data['recovery_hint']))}[/yellow]")
        if data.get("screenshot_path"):
            self._print(f"     [dim]screenshot: {escape(str(data['screenshot_path']))}[/dim]")

    def _r_risk(self, event: RunEvent) -> None:
        data = dict(event.data or {})
        risk = data.get("risk")
        if risk is not None and hasattr(risk, "level"):          # an ActionRisk object
            data.setdefault("level", getattr(risk, "level", "caution"))
            data.setdefault("reasons", list(getattr(risk, "reasons", []) or []))
            data.setdefault("rules", list(getattr(risk, "matched_rules", []) or []))
        level = str(data.get("level") or "caution")
        if level == "safe" and not self.verbose:
            return
        reasons = [str(r) for r in (data.get("reasons") or [])]
        rules = [str(r) for r in (data.get("rules") or data.get("matched_rules") or [])]
        style = _RISK_STYLE.get(level, "yellow")
        head = f"   [{style}]⚠ risk: {level}[/{style}]"
        if rules:
            head += f" [dim]({escape(', '.join(rules))})[/dim]"
        self._print(head)
        for reason in reasons[:3 if not self.verbose else 8]:
            self._print(f"     [dim]· {escape(reason)}[/dim]")

    def _r_confirm(self, event: RunEvent) -> None:
        # confirm() renders its own panel; these are the loop's trace lines about
        # the same exchange (it calls policy.authorize(call, risk, ctx, ui.confirm))
        data = dict(event.data or {})
        tool = self._picked(data, "tool", default="")
        if "approved" in data:
            ok = bool(data.get("approved"))
            mark, style = ("✔", "green") if ok else ("✖", "red")
            self._print(
                f"   [{style}]{mark}[/{style}] [dim]{escape(_clip(event.message, 180))}[/dim]"
            )
            if not ok and data.get("details"):
                self._print(Text(str(data["details"]), style="dim"))
            return
        level = self._picked(data, "level", default="destructive")
        self._print(f"   [bold red]⏳ asking the human to confirm[/bold red] [dim]{escape(tool)} ({escape(level)})[/dim]")
        for reason in [str(r) for r in (data.get("reasons") or [])][:2]:
            self._print(f"     [dim]· {escape(reason)}[/dim]")

    def _r_page(self, event: RunEvent) -> None:
        data = dict(event.data or {})
        url = self._picked(data, "url", default="")
        title = self._picked(data, "title", default="")
        line = "   [cyan]page[/cyan]"
        if url:
            line += f" {escape(_clip(url, 120))}"
        if title:
            line += f" [dim]— {escape(_clip(title, 80))}[/dim]"
        changed = data.get("page_changed", data.get("changed"))
        if isinstance(changed, bool):
            line += " [green](changed)[/green]" if changed else " [dim](unchanged)[/dim]"
        self._print(line)
        for key, label in (("added", "added"), ("removed", "removed")):
            items = data.get(key)
            if isinstance(items, (list, tuple, set)) and items:
                joined = ", ".join(str(i) for i in list(items)[:6])
                self._print(f"     [dim]{label}: {escape(_clip(joined, 160))}[/dim]")
        if isinstance(changed, (list, tuple, set)) and changed:
            joined = ", ".join(str(i) for i in list(changed)[:6])
            self._print(f"     [green]changed:[/green] [dim]{escape(_clip(joined, 160))}[/dim]")
        elif isinstance(changed, dict) and changed:
            joined = ", ".join(f"{k}={v}" for k, v in list(changed.items())[:6])
            self._print(f"     [green]changed:[/green] [dim]{escape(_clip(joined, 160))}[/dim]")
        delta = data.get("delta") or data.get("page_delta")
        if delta:
            text = ", ".join(str(d) for d in delta) if isinstance(delta, (list, tuple)) else str(delta)
            self._print(f"     [dim]delta: {escape(_clip(text, 200))}[/dim]")
        facts: list[str] = []
        if data.get("generation") is not None:
            facts.append(f"generation {data['generation']}")
        if data.get("elements") is not None:
            facts.append(f"{data['elements']} element(s)")
        if data.get("page_tokens"):
            facts.append(f"{data['page_tokens']} page tokens")
        if data.get("truncated"):
            facts.append("truncated")
        if data.get("hash"):
            facts.append(str(data["hash"])[:8])
        if facts:
            self._print(f"     [dim]{escape(', '.join(facts))}[/dim]")
        for alert in [str(a) for a in (data.get("alerts") or [])][:3]:
            self._print(f"     [yellow]alert: {escape(_clip(alert, 160))}[/yellow]")

    def _r_subagent(self, event: RunEvent) -> None:
        data = dict(event.data or {})
        name = self._picked(data, "subagent", "name", default="subagent")
        text = self._picked(data, "answer", "advice", "summary", default=event.message)
        confidence = data.get("confidence")
        line = f"   [magenta]↳ {escape(name)}[/magenta] {escape(_clip(text, 200))}"
        if isinstance(confidence, (int, float)):
            line += f" [dim](confidence {float(confidence):.2f})[/dim]"
        self._print(line)
        if data.get("evidence"):
            self._print(f"     [dim]evidence: {escape(_clip(str(data['evidence']), 200))}[/dim]")
        if data.get("question"):
            self._print(f"     [dim]asked: {escape(_clip(str(data['question']), 160))}[/dim]")

    def _r_user(self, event: RunEvent) -> None:
        text = self._picked(event.data or {}, "text", "answer", "instruction", default=event.message)
        kind = (event.data or {}).get("kind") or ("answer" if (event.data or {}).get("answer") else "instruction")
        self._print(f"   [magenta]⌨ human[/magenta] [dim]({escape(str(kind))})[/dim] {escape(_clip(text, 200))}")

    def _r_context(self, event: RunEvent) -> None:
        data = dict(event.data or {})
        bits: list[str] = []
        for key in ("steps", "history_tokens", "page_tokens", "compactions", "dropped_items"):
            if data.get(key) is not None:
                bits.append(f"{key}={data[key]}")
        body = ", ".join(bits) if bits else self._picked(data, "text", default=event.message)
        self._print(f"   [blue]θ context[/blue] [dim]{escape(_clip(body, 200))}[/dim]")

    def _r_info(self, event: RunEvent) -> None:
        style = _LEVEL_STYLE.get(str(event.level), "cyan")
        message = self._picked(event.data or {}, "text", default=event.message)
        if message:
            self._print(f"   [{style}]{escape(_clip(message, 300))}[/{style}]")

    def _r_error(self, event: RunEvent) -> None:
        data = dict(event.data or {})
        message = self._picked(data, "error", "text", default=event.message)
        self.console.print(
            Panel(Text(redact_secrets(str(message))), title="error", border_style="red", box=box.ROUNDED)
        )
        if data.get("traceback") and self.verbose:
            self._print(Text(str(data["traceback"]), style="dim"))

    def _r_finish(self, event: RunEvent) -> None:
        data = dict(event.data or {})
        result = data.get("result")
        if result is None and not data.get("final"):
            # the loop's own "the model called finish()" event: one line, the
            # full summary panel comes from run_task when the run is over
            success = bool(data.get("success", event.level != "error"))
            mark = "[green]finish: success[/green]" if success else "[yellow]finish: incomplete[/yellow]"
            line = f"   {mark} [dim]{escape(_clip(event.message, 200))}[/dim]"
            bits: list[str] = []
            if data.get("steps") is not None:
                bits.append(f"{data['steps']} step(s)")
            if isinstance(data.get("duration_s"), (int, float)):
                bits.append(_fmt_duration(float(data["duration_s"])))
            if data.get("reason"):
                bits.append(str(data["reason"]))
            if bits:
                line += f" [dim]({escape(', '.join(bits))})[/dim]"
            self._print(line)
            return
        success = bool(getattr(result, "success", data.get("success", True)))
        answer = str(getattr(result, "answer", data.get("answer", event.message)) or "")
        steps = getattr(result, "steps", data.get("steps", self.steps))
        duration = getattr(result, "duration_s", data.get("duration_s"))
        usage = _as_usage(getattr(result, "usage", data.get("usage"))) or self.final_usage or self.usage
        transcript = getattr(result, "transcript_path", data.get("transcript_path"))
        error = getattr(result, "error", data.get("error"))
        if result is None and not any(
            k in data for k in ("answer", "success", "steps", "transcript_path")
        ):
            # the loop's own "I called finish()" event: keep it compact
            self._print(
                f"   [{'green' if success else 'yellow'}]finish tool called[/] "
                f"[dim]{escape(_clip(event.message, 200))}[/dim]"
            )
            return
        rows = Table.grid(padding=(0, 2))
        rows.add_column(style="bold")
        rows.add_column()
        rows.add_row("answer", escape(answer or "(no answer)"))
        if error:
            rows.add_row("error", f"[red]{escape(str(error))}[/red]")
        rows.add_row("steps", f"{steps} tool call(s)"
                              + (f", {getattr(result, 'confirmations', 0)} confirmation(s)"
                                 if getattr(result, "confirmations", 0) else ""))
        if isinstance(duration, (int, float)) and duration:
            rows.add_row("duration", _fmt_duration(float(duration)))
        rows.add_row("tokens", _fmt_usage(usage if usage is not None else Usage()))
        if transcript:
            rows.add_row("transcript", escape(str(transcript)))
        self.console.print(
            Panel(
                rows,
                title="[bold green]✔ finished[/bold green]" if success
                else "[bold red]✖ failed[/bold red]",
                border_style="green" if success else "red",
                box=box.ROUNDED,
            )
        )

    # -- interaction helpers -------------------------------------------- #
    def _print_help(self) -> None:
        table = Table(title="human channel", box=box.SIMPLE, show_header=False)
        table.add_column(style="bold cyan")
        table.add_column()
        for key, what in (
            ("<text>", "queued as an instruction for the agent (delivered before its next step)"),
            ("\\q", "stop the run at the next step"),
            ("\\s", "skip the action that is about to run"),
            ("\\t", "print run stats"),
            ("\\h", "this help"),
        ):
            table.add_row(key, what)
        with self._lock:
            self.console.print(table)

    def _print_stats(self) -> None:
        usage = self.usage
        table = Table(title="run stats", box=box.SIMPLE, show_header=False)
        table.add_column(style="bold cyan")
        table.add_column()
        table.add_row("steps", str(self.steps))
        table.add_row("tool calls", str(self.tool_calls))
        table.add_row("risk flags", str(self.risk_notes))
        table.add_row("confirmations", f"{self.confirmations} asked, {self.approvals} approved, {self.denials} denied")
        table.add_row("instructions", str(len(self.instructions)))
        table.add_row("tokens", _fmt_usage(usage))
        with self._lock:
            self.console.print(table)

    def stats(self) -> dict[str, Any]:
        """Machine-readable stats (also rendered by ``\\t``)."""
        return {
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "events": len(self.events),
            "risk_notes": self.risk_notes,
            "confirmations": self.confirmations,
            "approvals": self.approvals,
            "denials": self.denials,
            "instructions": len(self.instructions),
            "prompt_tokens": self.usage.prompt_tokens,
            "completion_tokens": self.usage.completion_tokens,
            "tokens": self.usage.total,
            "final_tokens": self.final_usage.total if self.final_usage else None,
            "llm_calls": self.usage.calls,
            "stop_requested": self.stop_requested,
        }


# --------------------------------------------------------------------------- #
# wiring (all heavy imports are lazy so --help never needs them)
# --------------------------------------------------------------------------- #

def _build_llm(config: Config) -> Any:
    from .llm.registry import build_llm

    return build_llm(config)


def _build_session(config: Config, ui: AgentUI) -> Any:
    from .browser.session import BrowserSession

    session = BrowserSession(config, ui)
    session.start()
    session.new_page()
    return session


def _build_executor(session: Any, config: Config) -> Any:
    from .browser.actions import ActionExecutor

    return ActionExecutor(session, config)


def _build_agent(
    config: Config, *, llm: Any, session: Any, executor: Any, ui: AgentUI, policy: Any, **extra: Any
) -> Any:
    from .agent.loop import Agent

    return Agent(config, llm=llm, session=session, executor=executor, ui=ui, policy=policy, **extra)


def _build_extras(config: Config, llm: Any) -> dict[str, Any]:
    """Best-effort ContextManager + sub-agents.

    The agent loop can build its own defaults, so a missing or differently
    shaped module must not break the CLI: it degrades to "let the loop decide".
    """
    try:
        from .agent.memory import ContextManager
        from .agent.subagents import ExtractorSubAgent, RecoverySubAgent, Summarizer
    except ImportError:
        return {}
    try:
        summarizer = Summarizer(llm, config) if llm is not None else None
        return {
            "context": ContextManager(config, count_tokens=count_tokens, summarizer=summarizer),
            "extractor": ExtractorSubAgent(llm, config),
            "recovery": RecoverySubAgent(llm, config),
        }
    except Exception as exc:  # noqa: BLE001 - optional wiring, never fatal
        if config.verbose:
            print(f"[webpilot] context/sub-agents unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return {}


def _stop_session(session: Any) -> None:
    try:
        session.stop()
    except Exception:  # noqa: BLE001 - shutting down must not mask the run result
        pass


def run_task(config: Config, task: str, ui: AgentUI | None = None) -> RunResult:
    """Run one task end to end and render the result.

    Wiring: ``llm.registry.build_llm`` -> ``BrowserSession`` -> ``ActionExecutor``
    -> ``SecurityPolicy`` -> ``Agent`` (+ ``ContextManager``/sub-agents), all
    imported lazily.
    """
    from .security.policy import SecurityPolicy

    config.validate_credentials()
    config.ensure_dirs()
    ui = ui if ui is not None else TerminalUI(verbose=config.verbose)
    started = time.time()

    ui.emit(
        RunEvent(
            kind="start",
            message=task,
            data={
                "task": task,
                "provider": config.provider,
                "model": config.model,
                "headless": config.headless,
                "max_steps": config.max_steps,
                "confirm_mode": config.confirm_mode,
                "transcript_dir": str(config.transcript_dir),
                "profile": str(config.user_data_dir),
            },
        )
    )

    llm = _build_llm(config)
    policy = SecurityPolicy(config, llm=llm if config.llm_risk_check else None)
    session: Any = None
    try:
        session = _build_session(config, ui)
        executor = _build_executor(session, config)
        extras = _build_extras(config, llm)
        agent = _build_agent(
            config, llm=llm, session=session, executor=executor, ui=ui, policy=policy, **extras
        )
        result = agent.run(task)
    finally:
        if session is not None:
            _stop_session(session)

    if not isinstance(result, RunResult):  # defensive: a stub returning junk
        raise WebpilotError(f"the agent returned {type(result).__name__}, expected RunResult")
    if not result.duration_s:
        result.duration_s = time.time() - started

    ui.emit(
        RunEvent(
            kind="finish",
            message=result.answer,
            level="success" if result.success else "error",
            data={
                "result": result,
                "final": True,
                "success": result.success,
                "answer": result.answer,
                "steps": result.steps,
                "duration_s": result.duration_s,
                "usage": result.usage,
                "transcript_path": result.transcript_path,
                "error": result.error,
            },
        )
    )
    return result


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def _strip_subcommand(argv: list[str]) -> list[str]:
    """``webpilot run "task"`` -> ``webpilot "task"``.

    Only a *bare* leading ``run``/``start``/``task`` token is dropped, so a task
    that literally starts with that word still works when quoted.
    """
    if len(argv) >= 2 and argv[0] in _SUBCOMMANDS:
        return argv[1:]
    if len(argv) == 1 and argv[0] in _SUBCOMMANDS:
        return []
    return argv


def _wants_version(argv: Iterable[str]) -> bool:
    return any(a in ("--version", "-V") for a in argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.  Returns the process exit code (0/1/2/130)."""
    args = list(sys.argv[1:] if argv is None else argv)
    args = _strip_subcommand(args)

    if _wants_version(args):
        print(f"webpilot {__version__}")
        return 0

    try:
        config = config_from_args(args)
    except SystemExit as exc:  # argparse handled --help / a usage error
        return int(exc.code or 0)
    except ConfigError as exc:
        print(f"webpilot: configuration error: {exc}", file=sys.stderr)
        return 2

    try:
        ui = TerminalUI(verbose=config.verbose)
    except ConfigError as exc:
        print(f"webpilot: {exc}", file=sys.stderr)
        return 2

    try:
        config.validate_credentials()
    except ConfigError as exc:
        print(f"webpilot: {exc}", file=sys.stderr)
        return 2

    task = str(getattr(config, "task", "") or "").strip()
    if not task:
        task = ui.prompt_for_task() or ""
    if not task:
        print(
            "webpilot: no task given.\n"
            "  usage: webpilot [run] \"<task>\" [--provider openai] [--model gpt-4.1-mini] ...\n"
            "  see:   webpilot --help",
            file=sys.stderr,
        )
        return 2

    try:
        result = run_task(config, task, ui=ui)
    except ConfigError as exc:
        print(f"webpilot: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        ui.emit(RunEvent(kind="error", message="interrupted by the human (Ctrl-C)", level="warn"))
        return 130
    except Exception as exc:  # noqa: BLE001 - last resort: report, do not traceback-dump
        ui.emit(
            RunEvent(
                kind="error",
                message=f"{type(exc).__name__}: {exc}",
                level="error",
                data={"traceback": traceback.format_exc()},
            )
        )
        if config.verbose:
            traceback.print_exc()
        return 1
    finally:
        if isinstance(ui, TerminalUI):
            ui.close()

    if result.success:
        return 0
    ui.emit(RunEvent(kind="info", message=result.error or "task did not succeed", level="warn"))
    return 1


if __name__ == "__main__":  # pragma: no cover - `python -m webpilot.cli`
    raise SystemExit(main())
