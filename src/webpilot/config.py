"""Runtime configuration: CLI flags + environment + .env, validated once.

Nothing in the project reads ``os.environ`` directly - everything goes through
``Config`` so that tests can build a fully deterministic configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import ConfigError

HOME = Path.home()

#: provider -> (env var for the key, default base_url, default model)
PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "base_url": None,
        "model": "gpt-4.1-mini",
        "needs_key": True,
    },
    "anthropic": {
        "key_env": "ANTHROPIC_API_KEY",
        "base_url": None,
        "model": "claude-sonnet-4-5",
        "needs_key": True,
    },
    "deepseek": {
        "key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "needs_key": True,
    },
    "openrouter": {
        "key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "anthropic/claude-sonnet-4",
        "needs_key": True,
    },
    "ollama": {
        "key_env": "OLLAMA_API_KEY",
        "base_url": "http://localhost:11434/v1",
        "model": "qwen3:8b",
        "needs_key": False,
    },
    "openai_compatible": {          # vLLM / llama.cpp / LM Studio / any gateway
        "key_env": "OPENAI_API_KEY",
        "base_url": "http://localhost:8000/v1",
        "model": "local-model",
        "needs_key": False,
    },
    "fake": {                       # deterministic offline provider (tests)
        "key_env": "WEBPILOT_FAKE_KEY",
        "base_url": None,
        "model": "fake-1",
        "needs_key": False,
    },
}


@dataclass
class Config:
    # ---- LLM -------------------------------------------------------------
    provider: str = "openai"
    model: str = ""
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 2048
    request_timeout: float = 120.0
    llm_max_retries: int = 3
    subagent_model: str | None = None       # None -> same model as the main loop
    subagent_max_output_tokens: int = 1024
    llm_risk_check: bool = False            # second opinion for "caution" actions

    # ---- browser ----------------------------------------------------------
    headless: bool = False                  # the assignment wants a visible browser
    browser_channel: str | None = None      # "chrome" to reuse the installed Chrome
    browser_executable: str | None = None
    user_data_dir: Path = field(default_factory=lambda: HOME / ".webpilot" / "profile")
    persist_session: bool = True
    window_size: tuple[int, int] = (1440, 900)
    typing_delay_ms: int = 20
    action_timeout_ms: int = 20_000
    nav_timeout_ms: int = 45_000
    settle_timeout_ms: int = 3_000          # how long to wait for the page to settle
    locale: str = "en-US"
    browser_args: list[str] = field(default_factory=list)

    # ---- context management ----------------------------------------------
    page_token_budget: int = 2_600
    page_token_budget_min: int = 700
    context_token_budget: int = 60_000      # what we allow the prompt to grow to
    history_window: int = 12                # steps kept verbatim
    max_tool_result_tokens: int = 350
    compaction_trigger: float = 0.75        # compact when prompt > ratio * budget
    extractor_token_budget: int = 24_000    # sub-agent budget (it reads raw text)

    # ---- run --------------------------------------------------------------
    max_steps: int = 60
    max_attempts_per_action: int = 2
    max_consecutive_failures: int = 4
    loop_detection_repeats: int = 3
    task_timeout_s: float = 0.0             # 0 = no wall-clock limit

    # ---- security ---------------------------------------------------------
    confirm_mode: str = "ask"               # ask | allow | deny
    auto_approve_domains: list[str] = field(default_factory=list)
    allow_password_typing: bool = False     # the human logs in manually instead
    allow_destructive_tools: bool = True    # if False, destructive calls are refused outright

    # ---- artifacts --------------------------------------------------------
    transcript_dir: Path = field(default_factory=lambda: HOME / ".webpilot" / "transcripts")
    screenshot_dir: Path = field(default_factory=lambda: HOME / ".webpilot" / "screenshots")
    audit_path: Path = field(default_factory=lambda: HOME / ".webpilot" / "audit.jsonl")
    save_screenshots: bool = True
    verbose: bool = False

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        info = PROVIDERS.get(self.provider)
        if info is None:
            raise ConfigError(
                f"unknown provider {self.provider!r}; known: {', '.join(sorted(PROVIDERS))}"
            )
        if not self.model:
            self.model = info["model"]
        if not self.base_url:
            self.base_url = info["base_url"]
        if isinstance(self.user_data_dir, str):
            self.user_data_dir = Path(self.user_data_dir).expanduser()
        for attr in ("transcript_dir", "screenshot_dir", "audit_path"):
            value = getattr(self, attr)
            if isinstance(value, str):
                setattr(self, attr, Path(value).expanduser())
        if self.confirm_mode not in ("ask", "allow", "deny"):
            raise ConfigError("confirm_mode must be one of: ask, allow, deny")

    # ------------------------------------------------------------------ #
    @property
    def needs_key(self) -> bool:
        return bool(PROVIDERS[self.provider]["needs_key"])

    def key_env(self) -> str:
        return str(PROVIDERS[self.provider]["key_env"])

    def validate_credentials(self) -> None:
        if self.needs_key and not self.api_key:
            raise ConfigError(
                f"provider {self.provider!r} needs an API key: set {self.key_env()} "
                f"(env, .env or --api-key)."
            )

    def ensure_dirs(self) -> None:
        for path in (self.transcript_dir, self.screenshot_dir):
            Path(path).mkdir(parents=True, exist_ok=True)
        Path(self.audit_path).parent.mkdir(parents=True, exist_ok=True)
        if self.persist_session:
            Path(self.user_data_dir).mkdir(parents=True, exist_ok=True)

    def copy_with(self, **changes: Any) -> "Config":
        return replace(self, **changes)


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader (no dependency on python-dotenv at runtime)."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def config_from_env(
    env: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Build a Config from environment variables plus explicit overrides."""
    env = dict(os.environ if env is None else env)
    overrides = dict(overrides or {})
    provider = overrides.pop("provider", env.get("WEBPILOT_PROVIDER", "openai"))
    info = PROVIDERS.get(provider, {})
    api_key = overrides.pop("api_key", None) or env.get(str(info.get("key_env", "")))
    base_url = overrides.pop("base_url", None) or env.get("WEBPILOT_BASE_URL") or info.get("base_url")
    model = overrides.pop("model", None) or env.get("WEBPILOT_MODEL") or info.get("model", "")
    cfg = Config(provider=provider, model=model, api_key=api_key, base_url=base_url, **overrides)
    return cfg


def config_from_args(args: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> Config:
    """Parse CLI arguments (``webpilot`` CLI) into a Config."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="webpilot",
        description="Autonomous LLM agent that drives a visible Chrome browser.",
    )
    parser.add_argument("task", nargs="*", help="task for the agent (interactive prompt if omitted)")
    parser.add_argument("--provider", default=None, help=f"one of: {', '.join(sorted(PROVIDERS))}")
    parser.add_argument("--model", default=None, help="model name for the provider")
    parser.add_argument("--api-key", default=None, help="API key (defaults to the provider env var)")
    parser.add_argument("--base-url", default=None, help="override the provider endpoint")
    parser.add_argument("--headless", action="store_true", help="run the browser headless")
    parser.add_argument("--browser-channel", default=None, help="chrome | chromium | msedge")
    parser.add_argument("--profile-dir", default=None, help="persistent browser profile directory")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--confirm-mode", choices=["ask", "allow", "deny"], default=None,
                        help="how to treat destructive actions")
    parser.add_argument("--auto-approve-domain", action="append", default=None,
                        help="domain whose destructive actions need no confirmation (repeatable)")
    parser.add_argument("--page-budget", type=int, default=None, help="tokens per page snapshot")
    parser.add_argument("--context-budget", type=int, default=None, help="total prompt token budget")
    parser.add_argument("--transcript-dir", default=None)
    parser.add_argument("--no-screenshots", action="store_true")
    parser.add_argument("--llm-risk-check", action="store_true",
                        help="let the model double-check 'caution' actions")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--version", action="store_true")
    ns = parser.parse_args(list(args) if args is not None else None)

    overrides: dict[str, Any] = {}
    if ns.provider: overrides["provider"] = ns.provider
    if ns.model: overrides["model"] = ns.model
    if ns.api_key: overrides["api_key"] = ns.api_key
    if ns.base_url: overrides["base_url"] = ns.base_url
    if ns.headless: overrides["headless"] = True
    if ns.browser_channel: overrides["browser_channel"] = ns.browser_channel
    if ns.profile_dir: overrides["user_data_dir"] = Path(ns.profile_dir)
    if ns.max_steps: overrides["max_steps"] = ns.max_steps
    if ns.confirm_mode: overrides["confirm_mode"] = ns.confirm_mode
    if ns.auto_approve_domain: overrides["auto_approve_domains"] = list(ns.auto_approve_domain)
    if ns.page_budget: overrides["page_token_budget"] = ns.page_budget
    if ns.context_budget: overrides["context_token_budget"] = ns.context_budget
    if ns.transcript_dir: overrides["transcript_dir"] = Path(ns.transcript_dir)
    if ns.no_screenshots: overrides["save_screenshots"] = False
    if ns.llm_risk_check: overrides["llm_risk_check"] = True
    if ns.verbose: overrides["verbose"] = True

    cfg = config_from_env(env=env, overrides=overrides)
    cfg.task = " ".join(ns.task).strip()  # type: ignore[attr-defined]
    return cfg
