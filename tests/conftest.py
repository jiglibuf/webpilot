"""Shared pytest fixtures.

Design rules for the whole test-suite:
* no network, no API keys, no interactivity;
* the only "site" is the bundled local store (``tests/fixtures/site``);
* browser tests run headless for speed, but they exercise the same code paths
  as the headful demo, including the persistent-profile launch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from webpilot.config import Config  # noqa: E402
from webpilot import config as config_module  # noqa: E402
from webpilot.types import AgentUI, RunEvent  # noqa: E402


@pytest.fixture(autouse=True)
def _no_local_dotenv(monkeypatch):
    """Keep the suite independent of the developer's own ``.env`` file.

    ``config_from_env()`` without an explicit mapping reads ``.env`` from the
    working directory (that is what makes ``webpilot "task"`` work right after a
    clone).  Tests must not inherit a real key from it, so the *default path* is
    neutered while an explicit path still works — tests that check the parser
    pass their own file.
    """
    real = config_module._read_dotenv

    def guarded(path=None):
        return {} if path is None else real(path)

    monkeypatch.setattr(config_module, "_read_dotenv", guarded)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def fixture_site():
    """Start the bundled store once per session; yields its base URL."""
    from tests.fixtures.site.server import serve_background

    server, url = serve_background(0)
    yield url
    server.shutdown()
    server.server_close()


@pytest.fixture
def make_config(tmp_path: Path, fixture_site: str):
    """Factory for a deterministic Config (fake provider, temp dirs, headless)."""

    def _make(**kwargs) -> Config:
        defaults = dict(
            provider="fake",
            model="fake-1",
            headless=True,
            user_data_dir=tmp_path / "profile",
            transcript_dir=tmp_path / "transcripts",
            screenshot_dir=tmp_path / "screenshots",
            audit_path=tmp_path / "audit.jsonl",
            confirm_mode="ask",
            max_steps=8,
            typing_delay_ms=0,
            settle_timeout_ms=600,
            action_timeout_ms=6_000,
            nav_timeout_ms=15_000,
        )
        defaults.update(kwargs)
        cfg = Config(**defaults)
        cfg.ensure_dirs()
        return cfg

    return _make


@pytest.fixture
def config(make_config) -> Config:
    return make_config()


@pytest.fixture
def browser_session(config, fixture_site):
    """A started BrowserSession; closed afterwards."""
    from webpilot.browser.session import BrowserSession

    session = BrowserSession(config)
    session.start()
    session.new_page(fixture_site)
    try:
        yield session
    finally:
        session.stop()


class RecordingUI(AgentUI):
    """Agent UI double that records everything and answers scripted questions."""

    def __init__(self, approvals: list[bool] | None = None, answers: list[str] | None = None):
        self.events: list[RunEvent] = []
        self.prompts: list[tuple[str, str]] = []
        self.questions: list[str] = []
        self._approvals = list(approvals or [])
        self._answers = list(answers or [])
        self.instructions: list[str] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)

    def confirm(self, prompt: str, details: str) -> bool:
        self.prompts.append((prompt, details))
        return self._approvals.pop(0) if self._approvals else True

    def ask(self, question: str) -> str:
        self.questions.append(question)
        return self._answers.pop(0) if self._answers else "no answer"

    def drain_instructions(self) -> list[str]:
        drained, self.instructions = self.instructions, []
        return drained

    # convenience for assertions
    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]

    def text(self) -> str:
        return "\n".join(e.message for e in self.events)


@pytest.fixture
def ui() -> RecordingUI:
    return RecordingUI()


@pytest.fixture
def fake_llm():
    from webpilot.llm.fake import FakeLLM

    return FakeLLM


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
