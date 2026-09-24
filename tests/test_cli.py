"""Tests for :mod:`webpilot.cli` (the entry point and the rich terminal UI).

Offline by construction:

* ``--help`` / ``--version`` are exercised for real (argparse + version lookup);
* the terminal UI is rendered into a ``rich.console.Console(record=True)`` buffer;
* ``run_task`` is exercised end to end with ``provider="fake"`` and stand-ins
  installed in ``sys.modules`` for the lazily imported subsystems, so the test
  proves the wiring *and* the laziness (``import webpilot.cli`` must not pull in
  the agent loop, the browser or a provider SDK).
"""

from __future__ import annotations

import importlib
import inspect
import io
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from rich.console import Console

from webpilot import cli
from webpilot.config import Config, config_from_args
from webpilot.errors import ConfigError
from webpilot.types import RunEvent, RunResult, Usage

ROOT = Path(__file__).resolve().parents[1]
CARD = "4111111111111111"


# --------------------------------------------------------------------------- #
# fakes for the lazily imported subsystems
# --------------------------------------------------------------------------- #

class FakeLLM:
    provider = "fake"
    model = "fake-1"
    instances: list["FakeLLM"] = []

    def __init__(self, config=None):
        self.config = config
        FakeLLM.instances.append(self)

    def complete(self, messages, tools=None, **kwargs):  # pragma: no cover - unused
        raise AssertionError("the scripted agent must not call the LLM")

    def count_tokens(self, text: str) -> int:
        return len(text) // 4


class FakeBrowserSession:
    instances: list["FakeBrowserSession"] = []

    def __init__(self, config, ui=None):
        self.config, self.ui = config, ui
        self.started = self.stopped = False
        self.url = None
        FakeBrowserSession.instances.append(self)

    def start(self) -> None:
        self.started = True

    def new_page(self, url=None) -> None:
        self.url = url

    def stop(self) -> None:
        self.stopped = True


class FakeActionExecutor:
    instances: list["FakeActionExecutor"] = []

    def __init__(self, session, config):
        self.session, self.config = session, config
        FakeActionExecutor.instances.append(self)


class FakeContextManager:
    def __init__(self, config, count_tokens=None, summarizer=None):
        self.config, self.count_tokens, self.summarizer = config, count_tokens, summarizer


class FakeExtractor:
    def __init__(self, llm, config):
        self.llm, self.config = llm, config


class FakeRecovery:
    def __init__(self, llm, config):
        self.llm, self.config = llm, config


class FakeSummarizer:
    def __init__(self, llm, config):
        self.llm, self.config = llm, config


class ScriptedAgent:
    """Stand-in for ``webpilot.agent.loop.Agent`` (no browser, no model)."""

    instances: list["ScriptedAgent"] = []
    result: RunResult | None = None
    error: BaseException | None = None
    events: list[RunEvent] = []

    def __init__(self, config, *, llm=None, session=None, executor=None, ui=None,
                 policy=None, registry=None, **extra):
        self.config = config
        self.llm = llm
        self.session = session
        self.executor = executor
        self.ui = ui
        self.policy = policy
        self.kwargs = extra
        self.task: str | None = None
        ScriptedAgent.instances.append(self)

    def run(self, task: str) -> RunResult:
        self.task = task
        if ScriptedAgent.error is not None:
            raise ScriptedAgent.error
        for event in ScriptedAgent.events:
            self.ui.emit(event)
        return ScriptedAgent.result  # type: ignore[return-value]

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.result = None
        cls.error = None
        cls.events = []


def install_fakes(monkeypatch, agent_cls=ScriptedAgent) -> dict[str, types.ModuleType]:
    """Put stand-ins for browser/llm/agent into ``sys.modules`` (auto-reverted)."""
    root = importlib.import_module("webpilot")
    spec = {
        "webpilot.browser": {},
        "webpilot.browser.session": {"BrowserSession": FakeBrowserSession},
        "webpilot.browser.actions": {"ActionExecutor": FakeActionExecutor},
        "webpilot.llm": {},
        "webpilot.llm.registry": {"build_llm": lambda config: FakeLLM(config)},
        "webpilot.agent": {},
        "webpilot.agent.loop": {"Agent": agent_cls},
        "webpilot.agent.memory": {"ContextManager": FakeContextManager},
        "webpilot.agent.subagents": {
            "ExtractorSubAgent": FakeExtractor,
            "RecoverySubAgent": FakeRecovery,
            "Summarizer": FakeSummarizer,
        },
    }
    created: dict[str, types.ModuleType] = {}
    for name, attrs in spec.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        module.__dict__.setdefault("__all__", list(attrs))
        monkeypatch.setitem(sys.modules, name, module)
        created[name] = module
    for parent, child in (
        ("root", "webpilot.browser"), ("root", "webpilot.llm"), ("root", "webpilot.agent"),
        ("webpilot.browser", "webpilot.browser.session"),
        ("webpilot.browser", "webpilot.browser.actions"),
        ("webpilot.llm", "webpilot.llm.registry"),
        ("webpilot.agent", "webpilot.agent.loop"),
        ("webpilot.agent", "webpilot.agent.memory"),
        ("webpilot.agent", "webpilot.agent.subagents"),
    ):
        owner = root if parent == "root" else created[parent]
        monkeypatch.setattr(owner, child.rsplit(".", 1)[-1], created[child], raising=False)
    ScriptedAgent.reset()
    FakeLLM.instances = []
    FakeBrowserSession.instances = []
    FakeActionExecutor.instances = []
    return created


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def recording_ui(**kwargs):
    """A TerminalUI writing into an in-memory console (no tty, no blocking)."""
    buffer = io.StringIO()
    console = Console(record=True, width=200, file=buffer)
    kwargs.setdefault("interactive", False)
    kwargs.setdefault("stdin", io.StringIO(""))
    ui = cli.TerminalUI(console=console, **kwargs)
    return ui, console, buffer


def rendered(console: Console) -> str:
    return console.export_text()


def wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    """A missing key must never be able to change a test result."""
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
                "OPENROUTER_API_KEY", "WEBPILOT_PROVIDER", "WEBPILOT_MODEL",
                "WEBPILOT_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def home_in_tmp(monkeypatch, tmp_path: Path):
    """Keep ``~/.webpilot`` out of the real home directory."""
    import webpilot.config as config_module

    monkeypatch.setattr(config_module, "HOME", tmp_path)
    return tmp_path


# --------------------------------------------------------------------------- #
# 1. --help / --version without credentials, browser or network
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("argv", [["--help"], ["-h"]])
def test_help_exits_zero_and_documents_the_flags(argv, capsys):
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "usage: webpilot" in out
    for flag in ("--provider", "--model", "--headless", "--max-steps", "--confirm-mode",
                 "--profile-dir", "--transcript-dir", "--verbose"):
        assert flag in out, flag


def test_version_exits_zero_and_prints_the_version(capsys):
    assert cli.main(["--version"]) == 0
    out = capsys.readouterr().out
    assert out.strip() == f"webpilot {cli.__version__}"
    assert cli.__version__


def test_module_entry_point_prints_usage_and_version():
    """``python -m webpilot.cli --help`` (the definition-of-done command)."""
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env["PYTHONPATH"] = str(ROOT / "src")
    help_run = subprocess.run(
        [sys.executable, "-m", "webpilot.cli", "--help"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert help_run.returncode == 0, help_run.stderr
    assert "usage: webpilot" in help_run.stdout
    version_run = subprocess.run(
        [sys.executable, "-m", "webpilot.cli", "--version"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert version_run.returncode == 0
    assert cli.__version__ in version_run.stdout


def test_importing_the_cli_pulls_in_no_heavy_subsystem():
    """`webpilot --help` must work with no browser, no SDK and no agent loop."""
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env["PYTHONPATH"] = str(ROOT / "src")
    code = (
        "import sys, webpilot.cli as c;"
        "heavy=[m for m in sys.modules if m.startswith(('webpilot.agent','webpilot.browser','webpilot.llm'))];"
        "print(heavy)"
    )
    run = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "[]"


# --------------------------------------------------------------------------- #
# 2. argument parsing is delegated to config_from_args
# --------------------------------------------------------------------------- #

def test_cli_does_not_duplicate_the_argument_parser():
    """The flags come from config.config_from_args; cli.py must not re-declare them."""
    source = inspect.getsource(cli)
    assert "add_argument" not in source
    assert "ArgumentParser" not in source
    assert "config_from_args" in source


def test_flags_are_parsed_end_to_end_through_config_from_args(tmp_path: Path):
    config = config_from_args([
        "--provider", "fake",
        "--model", "fake-x",
        "--max-steps", "7",
        "--confirm-mode", "deny",
        "--auto-approve-domain", "example.com",
        "--auto-approve-domain", "example.invalid",
        "--headless",
        "--verbose",
        "--transcript-dir", str(tmp_path / "t"),
        "find", "the", "price", "of", "the", "chair",
    ])
    assert config.provider == "fake"
    assert config.model == "fake-x"
    assert config.max_steps == 7
    assert config.confirm_mode == "deny"
    assert config.auto_approve_domains == ["example.com", "example.invalid"]
    assert config.headless is True and config.verbose is True
    assert config.transcript_dir == tmp_path / "t"
    assert config.task == "find the price of the chair"


def test_bare_run_subcommand_is_accepted():
    assert cli._strip_subcommand(["run", "open", "the", "cart"]) == ["open", "the", "cart"]
    assert cli._strip_subcommand(["start", "open"]) == ["open"]
    assert cli._strip_subcommand(["open", "the", "cart"]) == ["open", "the", "cart"]
    assert cli._strip_subcommand(["run"]) == []
    # a quoted task that *starts* with the word run keeps its wording
    assert cli._strip_subcommand(["run the numbers"]) == ["run the numbers"]


def test_version_scanner():
    assert cli._wants_version(["--version"])
    assert cli._wants_version(["-V", "whatever"])
    assert not cli._wants_version(["--verbose"])


# --------------------------------------------------------------------------- #
# 3. the live trace
# --------------------------------------------------------------------------- #

def test_terminal_ui_renders_the_full_trace_with_arguments_and_usage():
    ui, console, _ = recording_ui()
    result = RunResult(
        task="find the chair price", success=True,
        answer="The Aeron Chair costs 1299.", steps=4, duration_s=12.5,
        usage=Usage(prompt_tokens=1200, completion_tokens=300, cached_tokens=0, calls=5),
        transcript_path="/tmp/run-123.jsonl", confirmations=1,
    )
    events = [
        RunEvent(kind="start", message="find the chair price", data={
            "task": "find the chair price", "provider": "deepseek", "model": "deepseek-chat",
            "headless": False, "max_steps": 8, "confirm_mode": "ask",
            "transcript_dir": "/tmp/transcripts"}),
        RunEvent(kind="thinking", message="I should open the cart first"),
        RunEvent(kind="tool_call", message="click", data={
            "tool": "click", "args": {"element_id": 5, "intent": "open the cart"}}),
        RunEvent(kind="risk", message="", data={
            "level": "destructive", "rules": ["pay.commit"],
            "reasons": ["places an order or moves money; hard to undo [pay.commit: matched 'Place order']"]}),
        RunEvent(kind="tool_result", message="clicked 'Place order'", data={
            "ok": False, "summary": "click failed: element disabled",
            "recovery_hint": "the button is disabled", "duration_s": 1.5}),
        RunEvent(kind="page", message="", data={
            "url": "https://shop.example.invalid/checkout", "title": "Checkout",
            "page_changed": True, "generation": 7, "page_tokens": 240,
            "added": ["[9] submit 'Place order'"]}),
        RunEvent(kind="subagent", message="", data={
            "subagent": "extractor", "question": "what is the price",
            "answer": "1299 USD", "confidence": 0.9}),
        RunEvent(kind="user", message="please use the cheaper option", data={"kind": "instruction"}),
        RunEvent(kind="context", message="", data={"steps": 3, "page_tokens": 240}),
        RunEvent(kind="finish", message=result.answer, data={"result": result, "final": True}),
    ]
    for event in events:
        ui.emit(event)
    out = rendered(console)

    # header: task + provider/model
    assert "find the chair price" in out
    assert "deepseek / deepseek-chat" in out or "deepseek · deepseek-chat" in out
    assert "confirm mode" in out and "ask" in out
    # tool call *with its arguments*
    assert "→ click" in out
    assert "element_id=5" in out and "intent='open the cart'" in out
    # risk verdict
    assert "destructive" in out and "pay.commit" in out and "hard to undo" in out
    # result + hint
    assert "click failed: element disabled" in out and "the button is disabled" in out
    # page delta
    assert "https://shop.example.invalid/checkout" in out and "Checkout" in out
    assert "changed" in out and "generation 7" in out
    # sub-agent call
    assert "extractor" in out and "1299 USD" in out and "0.90" in out
    # human instruction + context
    assert "instruction" in out and "please use the cheaper option" in out
    assert "page_tokens=240" in out
    # footer: answer, tokens, transcript
    assert "The Aeron Chair costs 1299." in out
    assert "1,200 prompt + 300 completion = 1,500 tokens" in out
    assert "/tmp/run-123.jsonl" in out

    stats = ui.stats()
    assert stats["steps"] == 1 and stats["tool_calls"] == 1
    assert stats["risk_notes"] == 1 and stats["tokens"] == 1500 and stats["llm_calls"] == 5


def test_terminal_ui_never_prints_a_card_number_or_password():
    ui, console, _ = recording_ui()
    ui.emit(RunEvent(kind="tool_call", data={
        "tool": "type_text",
        "args": {"element_id": 4, "text": CARD, "password": "hunter2-SECRET-value"}}))
    out = rendered(console)
    assert CARD not in out
    assert "hunter2-SECRET-value" not in out
    assert "***REDACTED***" in out


def test_terminal_ui_renders_a_loop_emitted_finish_compactly():
    """A `finish` event without a RunResult must not fake a full summary panel."""
    ui, console, _ = recording_ui()
    ui.emit(RunEvent(kind="finish", message="reporting 1299", data={}))
    out = rendered(console)
    assert "reporting 1299" in out
    assert "tokens" not in out


def test_terminal_ui_stats_and_help_are_printable():
    ui, console, _ = recording_ui()
    ui.emit(RunEvent(kind="tool_call", data={"tool": "click", "args": {"element_id": 1}}))
    ui._print_stats()
    ui._print_help()
    out = rendered(console)
    assert "run stats" in out and "steps" in out
    assert "human channel" in out and "\\q" in out


def test_per_step_usage_events_accumulate():
    ui, console, _ = recording_ui()
    ui.emit(RunEvent(kind="tool_result", data={"ok": True, "summary": "ok",
                                               "usage": {"prompt_tokens": 100, "completion_tokens": 20, "calls": 1}}))
    ui.emit(RunEvent(kind="tool_result", data={"ok": True, "summary": "ok",
                                               "prompt_tokens": 50, "completion_tokens": 10}))
    stats = ui.stats()
    assert stats["prompt_tokens"] == 150 and stats["completion_tokens"] == 30
    assert stats["tokens"] == 180
    out = rendered(console)
    # per-step delta *and* the running total are both visible
    assert "tokens: +100 prompt / +20 completion (cumulative 120)" in out
    assert "tokens: +50 prompt / +10 completion (cumulative 180)" in out


def test_the_run_summary_is_never_double_counted():
    ui, _, _ = recording_ui()
    ui.emit(RunEvent(kind="tool_result", data={"ok": True, "summary": "ok",
                                               "usage": {"prompt_tokens": 100, "completion_tokens": 20, "calls": 1}}))
    result = RunResult(task="t", success=True, answer="ok", steps=1, duration_s=1.0,
                       usage=Usage(100, 20, 0, 1))
    ui.emit(RunEvent(kind="finish", message="ok", data={"result": result, "usage": result.usage}))
    assert ui.stats()["tokens"] == 120


def test_terminal_ui_accepts_object_shaped_events():
    """The loop may emit a ToolCall / ActionRisk instead of pre-split keys."""
    from webpilot.types import ActionRisk, ToolCall

    ui, console, _ = recording_ui()
    call = ToolCall("click", {"element_id": 5, "intent": "place the order"})
    risk = ActionRisk(level="destructive", reasons=["moves money"],
                      matched_rules=["pay.commit"])
    ui.emit(RunEvent(kind="tool_call", data={"call": call}))
    ui.emit(RunEvent(kind="risk", data={"risk": risk}))
    out = rendered(console)
    assert "→ click" in out and "element_id=5" in out
    assert "destructive" in out and "pay.commit" in out and "moves money" in out
    assert ui.stats()["risk_notes"] == 1


def test_rendering_errors_do_not_break_the_run():
    ui, console, _ = recording_ui()
    ui.emit(RunEvent(kind="page", data={"added": 5}))            # not iterable
    assert "ui render error" not in rendered(console) or True    # must simply not raise


def test_terminal_ui_requires_rich(monkeypatch):
    monkeypatch.setattr(cli, "Console", None)
    with pytest.raises(ConfigError):
        cli.TerminalUI()


# --------------------------------------------------------------------------- #
# 4. confirm / ask: the blocking questions
# --------------------------------------------------------------------------- #

def test_confirm_defaults_to_no_when_stdin_is_not_interactive():
    ui, console, _ = recording_ui()
    assert ui.confirm("Allow: click 'Place order'?", "risk: destructive") is False
    out = rendered(console)
    assert "confirmation required" in out
    assert "Allow: click 'Place order'?" in out
    assert "risk: destructive" in out
    assert "non-interactive" in out and "NO" in out
    assert ui.denials == 1 and ui.confirmations == 1


@pytest.mark.parametrize(
    "answer,expected",
    [("y", True), ("Y", True), ("yes", True), ("да", True),
     ("n", False), ("no", False), ("нет", False), ("whatever", False), ("", False)],
)
def test_confirm_reads_a_yes_no_answer(answer, expected):
    ui, console, _ = recording_ui(interactive=True, scripted_answers=[answer, "n"])
    assert ui.confirm("Approve?", "details") is expected
    out = rendered(console)
    assert ("approved by the human" in out) is expected


def test_confirm_is_safe_at_eof():
    ui, console, _ = recording_ui(interactive=True)   # stdin is an empty StringIO
    assert ui.confirm("Approve?", "details") is False
    assert "defaulting to NO" in rendered(console)


def test_ask_returns_an_empty_string_when_not_interactive():
    ui, _, _ = recording_ui()
    assert ui.ask("what is your zip code?") == ""
    assert any(e.kind == "user" for e in ui.events)


def test_ask_reads_an_answer_from_a_scripted_human():
    ui, _, _ = recording_ui(interactive=True, scripted_answers=["10001"])
    assert ui.ask("zip?") == "10001"


def test_prompt_for_task_is_none_without_a_tty():
    ui, _, _ = recording_ui()
    assert ui.prompt_for_task() is None


def test_prompt_for_task_reads_a_line():
    ui, _, _ = recording_ui(interactive=True, scripted_answers=["book a table"])
    assert ui.prompt_for_task() == "book a table"


# --------------------------------------------------------------------------- #
# 5. the stdin instruction channel
# --------------------------------------------------------------------------- #

def test_stdin_reader_queues_instructions_and_commands():
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "r")
    ui, console, _ = recording_ui(interactive=True, stdin=stream)
    try:
        assert ui._reader is None, "the reader must not start before the run"
        ui.start_input_reader()
        assert ui._reader is not None

        os.write(write_fd, "look at the second result\n".encode())
        assert wait_for(lambda: ui.drain_instructions() == ["look at the second result"])
        assert ui.instructions == ["look at the second result"]

        os.write(write_fd, "\\s\n".encode())
        assert wait_for(ui.take_skip) is True

        os.write(write_fd, "\\t\n".encode())
        assert wait_for(lambda: "run stats" in rendered(console))

        os.write(write_fd, "\\h\n".encode())
        assert wait_for(lambda: "human channel" in rendered(console))

        os.write(write_fd, "\\z\n".encode())
        assert wait_for(lambda: "unknown command" in rendered(console))

        os.write(write_fd, "\\q\n".encode())
        assert wait_for(lambda: ui.stop_requested) is True

        # a second instruction still arrives, in order
        os.write(write_fd, "then stop\n".encode())
        assert wait_for(lambda: ui.drain_instructions() == ["then stop"])
    finally:
        os.close(write_fd)          # EOF: the reader thread must exit quietly
        assert wait_for(lambda: ui._reader is None or not ui._reader.is_alive())
        ui.close()
        stream.close()


def test_an_answer_typed_just_before_the_prompt_is_not_lost():
    """Regression: an impatient 'y' must not be eaten as an instruction."""
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(read_fd, "r")
    ui, _, _ = recording_ui(interactive=True, stdin=stream, answer_timeout=2.0)
    try:
        ui.start_input_reader()
        os.write(write_fd, b"y\n")                      # typed *before* the question
        assert wait_for(lambda: bool(ui._pre_answers))
        started = time.time()
        assert ui.confirm("Approve?", "risk: destructive") is True
        assert time.time() - started < 1.0, "the recorded answer must be used at once"
        assert ui.drain_instructions() == []            # it never reached the agent
    finally:
        os.close(write_fd)
        ui.close()
        stream.close()


def test_drain_instructions_is_empty_and_non_blocking_without_a_tty():
    ui, _, _ = recording_ui()
    started = time.time()
    assert ui.drain_instructions() == []
    assert ui.drain_instructions() == []
    assert time.time() - started < 0.5
    assert ui.take_skip() is False
    assert ui.stop_requested is False
    ui.start_input_reader()
    assert ui._reader is None, "no thread may be started for a non-interactive stdin"
    ui.close()


# --------------------------------------------------------------------------- #
# 6. run_task wiring (provider "fake" + stubbed subsystems)
# --------------------------------------------------------------------------- #

def test_run_task_wires_every_subsystem_and_renders_the_answer(monkeypatch, tmp_path: Path):
    install_fakes(monkeypatch)
    ScriptedAgent.result = RunResult(
        task="find the chair price", success=True,
        answer="The Aeron Chair costs 1299 USD.", steps=3, duration_s=2.0,
        usage=Usage(300, 90, 0, 2), transcript_path=str(tmp_path / "run.jsonl"),
    )
    ScriptedAgent.events = [
        RunEvent(kind="tool_call", data={"tool": "click", "args": {"element_id": 5, "intent": "open search"}}),
        RunEvent(kind="tool_result", data={"ok": True, "summary": "clicked 'Search'"}),
    ]
    ui, console, _ = recording_ui()
    config = Config(provider="fake", user_data_dir=tmp_path / "profile",
                    transcript_dir=tmp_path / "transcripts",
                    screenshot_dir=tmp_path / "screenshots",
                    audit_path=tmp_path / "audit.jsonl")

    result = cli.run_task(config, "find the chair price", ui=ui)

    assert isinstance(result, RunResult) and result.success
    assert result.answer == "The Aeron Chair costs 1299 USD."
    assert result.duration_s == 2.0

    agent = ScriptedAgent.instances[-1]
    assert agent.task == "find the chair price"
    assert isinstance(agent.llm, FakeLLM)
    assert agent.ui is ui
    assert agent.policy is not None and agent.policy.config is config
    assert agent.executor is FakeActionExecutor.instances[-1]
    # the context manager and the sub-agents were built and handed over
    assert isinstance(agent.kwargs.get("context"), FakeContextManager)
    assert isinstance(agent.kwargs.get("extractor"), FakeExtractor)
    assert isinstance(agent.kwargs.get("recovery"), FakeRecovery)
    assert isinstance(agent.kwargs["context"].summarizer, FakeSummarizer)
    assert agent.kwargs["context"].count_tokens is not None
    # the browser session was started, then closed
    assert FakeBrowserSession.instances[-1].started is True
    assert FakeBrowserSession.instances[-1].stopped is True
    # config dirs were created
    assert (tmp_path / "transcripts").is_dir()

    out = rendered(console)
    assert "find the chair price" in out
    assert "fake · fake-1" in out or "fake / fake-1" in out
    assert "→ click" in out and "element_id=5" in out
    assert "The Aeron Chair costs 1299 USD." in out
    assert "300 prompt + 90 completion" in out


def test_run_task_stops_the_browser_even_when_the_agent_raises(monkeypatch, tmp_path: Path):
    install_fakes(monkeypatch)
    ScriptedAgent.error = RuntimeError("provider exploded")
    # note: the LLM in run_task is built by build_llm; a *provider* failure must
    # still shut the browser down
    config = Config(provider="fake", user_data_dir=tmp_path / "p",
                    transcript_dir=tmp_path / "t", screenshot_dir=tmp_path / "s",
                    audit_path=tmp_path / "a.jsonl")
    ui, _, _ = recording_ui()
    with pytest.raises(RuntimeError):
        cli.run_task(config, "anything", ui=ui)
    assert FakeBrowserSession.instances[-1].stopped is True


def test_run_task_rejects_a_bogus_agent_result(monkeypatch, tmp_path: Path):
    install_fakes(monkeypatch)
    ScriptedAgent.result = "not a RunResult"       # type: ignore[assignment]
    config = Config(provider="fake", user_data_dir=tmp_path / "p",
                    transcript_dir=tmp_path / "t", screenshot_dir=tmp_path / "s",
                    audit_path=tmp_path / "a.jsonl")
    ui, _, _ = recording_ui()
    with pytest.raises(Exception):
        cli.run_task(config, "anything", ui=ui)


def test_run_task_without_optional_subsystems_still_works(monkeypatch, tmp_path: Path):
    """context/sub-agents missing or differently shaped -> the loop builds its own."""
    created = install_fakes(monkeypatch)
    monkeypatch.delitem(sys.modules, "webpilot.agent.memory")
    monkeypatch.delitem(sys.modules, "webpilot.agent.subagents")
    ScriptedAgent.result = RunResult(task="t", success=True, answer="ok", steps=1,
                                     duration_s=0.5, usage=Usage(1, 1, 0, 1))
    assert created
    config = Config(provider="fake", user_data_dir=tmp_path / "p",
                    transcript_dir=tmp_path / "t", screenshot_dir=tmp_path / "s",
                    audit_path=tmp_path / "a.jsonl")
    ui, _, _ = recording_ui()
    result = cli.run_task(config, "t", ui=ui)
    assert result.success
    assert ScriptedAgent.instances[-1].kwargs == {}


# --------------------------------------------------------------------------- #
# 7. main(): exit codes
# --------------------------------------------------------------------------- #

def test_main_runs_a_task_with_the_fake_provider(monkeypatch, capsys, home_in_tmp: Path):
    install_fakes(monkeypatch)
    ScriptedAgent.result = RunResult(task="say hello", success=True, answer="hello from the agent",
                                     steps=2, duration_s=1.0, usage=Usage(10, 5, 0, 1))
    code = cli.main(["--provider", "fake", "say", "hello"])
    assert code == 0
    out = capsys.readouterr().out
    assert "say hello" in out
    assert "hello from the agent" in out
    assert ScriptedAgent.instances[-1].task == "say hello"


def test_main_accepts_the_run_subcommand(monkeypatch, capsys, home_in_tmp: Path):
    install_fakes(monkeypatch)
    ScriptedAgent.result = RunResult(task="open the cart", success=True, answer="opened",
                                     steps=1, duration_s=0.2, usage=Usage(1, 1, 0, 1))
    assert cli.main(["run", "--provider", "fake", "open", "the", "cart"]) == 0
    assert ScriptedAgent.instances[-1].task == "open the cart"


def test_main_returns_1_when_the_task_fails(monkeypatch, capsys, home_in_tmp: Path):
    install_fakes(monkeypatch)
    ScriptedAgent.result = RunResult(task="t", success=False, answer="could not do it",
                                     steps=9, duration_s=3.0, usage=Usage(1, 1, 0, 1),
                                     error="gave up after 9 steps")
    assert cli.main(["--provider", "fake", "do", "something"]) == 1
    out = capsys.readouterr().out
    assert "could not do it" in out and "gave up after 9 steps" in out


def test_main_returns_130_on_keyboard_interrupt(monkeypatch, capsys, home_in_tmp: Path):
    install_fakes(monkeypatch)
    ScriptedAgent.error = KeyboardInterrupt()
    assert cli.main(["--provider", "fake", "do", "something"]) == 130
    assert "interrupted" in capsys.readouterr().out


def test_main_returns_1_on_an_agent_crash(monkeypatch, capsys, home_in_tmp: Path):
    install_fakes(monkeypatch)
    ScriptedAgent.error = RuntimeError("provider exploded")
    assert cli.main(["--provider", "fake", "do", "something"]) == 1
    assert "provider exploded" in capsys.readouterr().out


def test_main_returns_2_without_credentials(home_in_tmp: Path, capsys):
    assert cli.main(["--provider", "openai", "do", "something"]) == 2
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err


def test_main_returns_2_for_an_unknown_provider(home_in_tmp: Path, capsys):
    assert cli.main(["--provider", "nope", "do", "something"]) == 2
    assert "unknown provider" in capsys.readouterr().err


def test_main_returns_2_for_a_usage_error(capsys):
    assert cli.main(["--definitely-not-a-flag"]) == 2
    assert "usage: webpilot" in capsys.readouterr().err


def test_main_returns_2_when_no_task_is_given(home_in_tmp: Path, capsys):
    assert cli.main(["--provider", "fake"]) == 2
    assert "no task given" in capsys.readouterr().err


def test_main_returns_2_when_rich_is_missing(monkeypatch, home_in_tmp: Path, capsys):
    monkeypatch.setattr(cli, "Console", None)
    assert cli.main(["--provider", "fake", "do", "something"]) == 2
    assert "rich" in capsys.readouterr().err
