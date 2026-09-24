# webpilot — internal interfaces (build contract)

**Every module must conform to this document.** It exists so that the modules can be
written in parallel without drift. If an implementation needs a change to a signature
listed here, change this file in the same commit and keep the change additive.

Hard rules (from the assignment, enforced by review):

1. **No hard-coded selectors.** No `page.click("a[data-qa=...]")`, no `.btn-primary`
   anywhere in `src/`. Elements are addressed only through the integer `id` produced by
   the page sniffer.
2. **No pre-authored plans / site knowledge.** No code or prompt may contain a URL path,
   a button label, or a step list tied to a particular site ("vacancies are at /vacancies").
   Prompts describe *capabilities*, never *sites*.
3. **No page dumps into the model context.** Raw HTML never reaches the main agent loop;
   it is read only inside the extractor sub-agent, which returns a short answer.
4. Everything the human sees (tool calls with arguments, page state, risk prompts) goes
   through `AgentUI`; nothing prints to stdout directly except `cli.py`.

## Package layout

```
src/webpilot/
  types.py            # shared dataclasses/protocols            (DONE, do not re-implement)
  errors.py           # exception hierarchy                     (DONE)
  config.py           # Config dataclass + loaders              (DONE)
  tokenizer.py        # count_tokens(text) -> int               (DONE)
  app.py              # wiring: build Config -> session/llm/agent/ui   (DONE)
  browser/
    __init__.py
    dom_snapshot.js   # injected page sniffer (no Python)
    session.py        # BrowserSession: Playwright persistent context (headful)
    snapshot.py       # JS result -> PageModel, render_page_model()
    actions.py        # tool call -> real browser action + verification
  llm/
    __init__.py
    base.py           # BaseLLM: shared retry/timeout/counting helpers
    openai_provider.py
    anthropic_provider.py
    local_provider.py # Ollama / vLLM / llama.cpp / LM Studio (OpenAI-compatible)
    fake.py           # deterministic scripted provider for tests
    registry.py       # build_llm(config) -> LLMClient
  agent/
    __init__.py
    tools.py          # TOOL_SPECS + ToolRegistry
    prompts.py        # system prompts (capability-only, site-agnostic)
    memory.py         # ContextManager: token budget, compaction, scratchpad
    subagents.py      # ExtractorSubAgent, RecoverySubAgent, Summarizer
    loop.py           # Agent.run(task) -> RunResult
  security/
    __init__.py
    policy.py         # SecurityPolicy.classify/authorize + audit log
  cli.py              # argparse + rich terminal UI, stdin instruction channel
tests/
  conftest.py         # fixtures (DONE)
  fixtures/site/      # tiny local website used by browser tests (DONE)
```

## browser/dom_snapshot.js

A single IIFE evaluated with `page.evaluate`. It must be **idempotent** (it may run many
times on the same page) and must return a plain JSON object:

```json
{
  "url": "...", "title": "...",
  "elements": [{
     "id": 1, "role": "button", "name": "Add to cart", "tag": "button",
     "type": null, "value": null, "placeholder": null, "href": null,
     "disabled": false, "checked": null, "required": false, "expanded": null,
     "invalid": false, "inViewport": true, "options": [],
     "domHash": "ab12...", "selectorHint": "form > button",
     "frameIndex": 0, "note": "", "isPassword": false
  }],
  "text": [{"kind": "heading|paragraph|list|table|alert|label", "text": "...", "inViewport": true}],
  "alerts": ["..."],
  "dialogs": ["..."],
  "scroll": {"y": 0, "maxY": 2400, "viewportH": 800, "atBottom": false},
  "focusedId": 3,
  "htmlChars": 152340,
  "fullTextChars": 8123,
  "errors": ["console error text ..."]
}
```

Requirements:

* Mark every interactive node with `data-webpilot-id="<n>"` (a stable *attribute* so the
  Python side can find it again with a generated selector — the selector is generated
  from the id at action time, never hand-written).
* Interactive = `a[href]`, `button`, `input` (not hidden), `select`, `textarea`, `summary`,
  `label[for]`, `[role]` in the interactive set, `[contenteditable]`, `[onclick]`,
  `[tabindex]` >= 0, or computed `cursor: pointer` on a small leaf node.
* Skip invisible nodes (`display:none`, `visibility:hidden`, `opacity:0`, zero box) —
  but still count them in `droppedElements` of the returned object.
* Pierce open shadow roots; walk same-origin iframes (`frameIndex` per frame, main = 0).
* Accessible name priority: `aria-label` -> `aria-labelledby` target -> `<label>` ->
  `placeholder` -> `value` -> `alt`/`title` -> visible `innerText`, whitespace-collapsed,
  nbsp normalised to a space, truncated to 120 chars.
* `text`: visible text blocks with their semantic kind, **excluding** `script`, `style`,
  `noscript`, `svg`, and nodes already represented as an interactive element's name.
  Keep headings and `[role=alert]` messages, they are high-signal.
* `errors`: hook `window.onerror` / console errors through a small collector the JS
  installs once (guard with a `window.__webpilotInstalled` flag).

## browser/session.py — class `BrowserSession`

```python
class BrowserSession:
    def __init__(self, config: Config, ui: AgentUI | None = None) -> None: ...
    def start(self) -> None: ...            # launch persistent context, headful by default
    def stop(self) -> None: ...
    def new_page(self, url: str | None = None) -> None: ...
    @property
    def pages(self) -> list[Any]: ...
    def snapshot(self, *, budget_tokens: int, mode: SnapshotMode = "full") -> PageModel: ...
    def raw_text(self, max_chars: int = 120_000) -> str: ...
    def raw_html(self, max_chars: int = 200_000) -> str: ...
    def screenshot(self, label: str = "shot", full_page: bool = False) -> str: ...
    def is_blocked_by_auth(self) -> bool: ...   # heuristic: password field present / login form
```

Persistent sessions: `launch_persistent_context(config.user_data_dir, headless=config.headless, channel=config.browser_channel, viewport=None when headful, args=[...])`.
Extra flags needed for demos/docker: `--no-sandbox --disable-dev-shm-usage --start-maximized`,
`--disable-blink-features=AutomationControlled` (only if it does not break normal use).
Native dialogs (`alert/confirm/prompt`) must never hang a run: attach a handler that records
the text into the snapshot's `dialogs` and — unless the security policy says otherwise —
dismisses them. Must expose `pending_dialogs: list[str]`.

## browser/actions.py — class `ActionExecutor`

```python
class ActionExecutor:
    def __init__(self, session: BrowserSession, config: Config) -> None: ...
    def execute(self, call: ToolCall) -> ToolResult: ...
```

* Resolve `element_id` -> `[data-webpilot-id="N"]` inside the right frame, then
  `scroll_into_view_if_needed()`; refuse if the resolved box is covered by another element
  (`document.elementFromPoint` check) and try one cheap recovery (press Escape / click the
  topmost overlay's close target) before reporting failure.
* Human-like interaction: `page.mouse.click(x, y)` on the box centre for clicks;
  `locator.press_sequentially(text, delay=config.typing_delay_ms)` for typing.
* **Verify every action.** Capture `page_hash`, url, title before/after; wait for the
  network to settle (`wait_for_load_state("networkidle")` best-effort with a short timeout)
  or for the DOM hash to change. If nothing changed, return `ok=False` with a precise
  diagnostic and `recovery_hint` ("element disabled", "overlay intercepts clicks",
  "form rejected the value", "page needs waiting").
* Never type into a password field: return `ok=False` with
  `recovery_hint="password fields are filled by the human"` and let the loop call `ask_user`.
* Return a fresh `PageModel` in `ToolResult.page` (full mode) for every action that can
  change the page, and `page_changed=True/False` accordingly.

## browser/snapshot.py

```python
def build_page_model(raw: dict, *, generation: int, budget_tokens: int,
                     mode: SnapshotMode = "full", previous: PageModel | None = None,
                     count_tokens: Callable[[str], int] | None = None) -> PageModel: ...
def render_page_model(model: PageModel, *, budget_tokens: int,
                      mode: SnapshotMode = "full", previous: PageModel | None = None) -> str: ...
```

Rendering contract (asserted in tests):

```
URL: https://site/path | Title: Sign in | generation 7 | scroll 0/2400
ALERTS: ...
DIALOG: ...
[3] textbox 'Username' placeholder='Username'
[4] textbox 'Password' type=password
[5] submit 'Login'
TEXT:
heading: ...
paragraph: ...
```

Budget algorithm: the budget is split `alerts/dialogs` (always, they are cheap) ->
interactive elements in viewport -> interactive elements off-screen (dropped first when the
budget is tight, counted in `dropped_elements`) -> text blocks in viewport -> text off-screen.
`truncated=True` whenever anything was cut. Never exceed `budget_tokens`.

`mode="delta"`: render only elements whose `dom_hash` changed vs `previous`, plus a
`(unchanged: N elements, use ids from the previous snapshot)` line.

`mode="text"`: no element list, text only (used by the extractor sub-agent).

## llm/ — providers

`BaseLLM` (in `base.py`) implements: retries with jittered backoff on 429/5xx/timeouts,
per-call timeout, `Usage` accumulation, `count_tokens` (tiktoken-ish heuristic — see
`tokenizer.py`), and normalisation of provider errors into `webpilot.errors.LLMError`.

* `openai_provider.OpenAIProvider` — Chat Completions with `tools=[...]`,
  maps `tool_calls` -> `ToolCall`, `response.usage` -> `Usage`. Works for OpenAI and any
  OpenAI-compatible endpoint (DeepSeek, OpenRouter, vLLM, Ollama, llama.cpp, LM Studio) via
  `base_url`.
* `anthropic_provider.AnthropicProvider` — Messages API with `tools=[...]`,
  maps `tool_use` blocks -> `ToolCall`, `stop_reason="tool_use"`, usage from
  `input_tokens`/`output_tokens` (+ `cache_read_input_tokens`).
* `local_provider.LocalProvider` — thin subclass of the OpenAI provider with local defaults
  (`base_url=http://localhost:11434/v1`, no key required, tolerant JSON parsing for
  models that emit tool calls as text).
* `fake.FakeLLM` — deterministic, scripted via a list of `LLMResponse`s or a callable;
  used by the loop tests, records every request for assertions.
* `registry.build_llm(config) -> LLMClient` — provider dispatch + key validation with an
  actionable error message.

All providers must accept the same `messages` (with `tool_call_id` round-tripping) and
return `LLMResponse`. Tool schemas come from `agent/tools.py` via `ToolSpec`.

## agent/tools.py

`TOOL_SPECS: list[ToolSpec]` — exactly these tools, JSON-Schema `strict`-compatible
(no `additionalProperties`, every property listed in `required`):

| tool | arguments | purpose |
|---|---|---|
| `page_outline` | `mode` ("full"\|"delta"\|"text"), `filter` (optional substring) | refresh/extend the compact page model |
| `goto` | `url` | navigate current tab |
| `click` | `element_id`, `intent` | click an element by its id |
| `type_text` | `element_id` (0 = focused element), `text`, `submit`, `clear` | type into a field |
| `press_key` | `key` | keyboard keys (Enter, Escape, Tab, PageDown, arrow keys) |
| `scroll` | `direction` ("up"\|"down"\|"top"\|"bottom"), `amount` (px) | scroll the page |
| `go_back` | — | browser back |
| `wait_for` | `seconds`, `text` (optional), `timeout_seconds` | wait for time/DOM/text |
| `tabs` | `action` ("list"\|"open"\|"switch"\|"close"), `index`, `url` | tab management |
| `ask_page` | `question` | extractor sub-agent: answer a question about the current page |
| `screenshot` | `label` | save a PNG (for the human / audit) |
| `notes` | `text` | update the scratchpad memory that survives context compaction |
| `ask_user` | `question` | pause and ask the human |
| `finish` | `success`, `answer` | end the run and report the result |

`ToolRegistry.dispatch(call: ToolCall, ctx) -> ToolResult` maps names to the executor,
handles the agent-local tools (`notes`, `ask_user`, `finish`, `ask_page`, `page_outline`),
and rejects unknown/malformed calls with a helpful error instead of raising.

## agent/memory.py — class `ContextManager`

Responsible for *staying inside the context window*:

```python
class ContextManager:
    def __init__(self, config: Config, count_tokens: Callable[[str], int],
                 summarizer: Summarizer | None = None) -> None: ...
    def step_count(self) -> int: ...
    def record(self, step: StepRecord) -> None: ...
    def build_messages(self, *, task: str, page: PageModel, page_block: str,
                       pending: list[str], extra: list[LLMMessage] | None = None) -> list[LLMMessage]: ...
    def stats(self) -> dict[str, Any]: ...
```

Strategies (documented in `docs/CONTEXT.md`, implemented here):

1. compact 1:1 page rendering instead of HTML (delegated to the sniffer);
2. page-block budget that shrinks as the conversation grows (`page_token_budget`),
   with `mode="delta"` rendering once the page is already in context;
3. sliding window of the last `config.history_window` steps in full, older steps
   collapsed into a running summary (LLM summariser, deterministic fallback if no LLM);
4. per-step tool results truncated to `config.max_tool_result_tokens`;
5. duplicate-step suppression: a repeated identical page hash renders as
   `"page unchanged since step N"`;
6. loops: repeated `ToolCall.signature()` twice in a row injects a hard warning message.

`stats()` returns at least `{steps, prompt_tokens_estimate, page_tokens, history_tokens,
compactions, dropped_items}`.

## agent/subagents.py

* `ExtractorSubAgent(llm, config)` — `answer(question, page_text, html_excerpt) -> ExtractionResult`.
  Prompt must force: answer only from the supplied content, cite the exact snippet as
  evidence, empty answer + confidence 0 when absent. Runs with its own (large) token budget,
  its own message list, `temperature=0`; it never sees the agent's conversation.
* `RecoverySubAgent(llm, config)` — `propose(FailureContext) -> RecoveryProposal`; sees the
  task, the failing tool call, the last error(s), the compact page model and the step
  digests; returns advice + up to 3 alternative `ToolCall`s + `give_up`.
* `Summarizer(llm, config)` — `summarize(task, steps, previous_summary) -> str`; used by the
  context manager when the history has to be compacted. Deterministic fallback when
  `llm is None`.

All sub-agents report their own `Usage` so the run telemetry can attribute token spend.

## agent/loop.py — class `Agent`

```python
class Agent:
    def __init__(self, config: Config, llm: LLMClient, session: BrowserSession,
                 executor: ActionExecutor, ui: AgentUI, policy: SecurityPolicy | None = None,
                 registry: ToolRegistry | None = None, context: ContextManager | None = None,
                 extractor: ExtractorSubAgent | None = None,
                 recovery: RecoverySubAgent | None = None) -> None: ...
    def run(self, task: str) -> RunResult: ...
```

Loop requirements:

* system prompt + task -> model -> tool calls -> policy check -> execute -> record step ->
  fresh page block -> repeat until `finish`, `config.max_steps`, or a fatal error.
* Handle multiple tool calls per assistant turn (execute in order, append one `tool` message
  per call).
* **Security gate**: every call goes through `policy.classify()`; `destructive` calls require
  `ui.confirm()`; denied calls are returned to the model as a failure with
  `"human denied this action"` so it can look for another way (never silently skipped).
* **Error handling**: consecutive failures of the same signature -> `RecoverySubAgent`;
  page-hash loop detection (same hash + same signature 3x) -> injected warning, then
  recovery; `max_steps` reached -> `finish` with an honest partial answer.
* **Human in the loop**: `ui.drain_instructions()` before every model call; non-empty
  instructions are appended as a user message. `ask_user` blocks on `ui.ask()`.
* Persist a transcript: `config.transcript_dir/run-<timestamp>.jsonl` (one JSON line per
  event/step, including token usage) — this is the evidence artifact for the demo.
* `RunResult` must be honest: `success=True` only when the model called `finish(success=true)`
  **and** a final page observation does not contradict it.

## security/policy.py — class `SecurityPolicy`

```python
class SecurityPolicy:
    def __init__(self, config: Config, llm: LLMClient | None = None) -> None: ...
    def classify(self, call: ToolCall, ctx: ActionContext) -> ActionRisk: ...
    def authorize(self, call: ToolCall, risk: ActionRisk, ctx: ActionContext,
                  confirm: ConfirmFn) -> tuple[bool, str]: ...
```

* Deterministic rule engine over (action kind, element role/name, field type, URL) covering
  at least: payments/checkout, deleting data, sending messages/mail, publishing/posting,
  account or permission changes, password/credential entry, file upload, irreversible form
  submits, downloads of executables.
* Optional LLM second opinion for `caution`-level cases when `config.llm_risk_check` is on
  (cheap model, short prompt) — must degrade gracefully when no LLM is configured.
* `authorize()` calls `confirm(prompt, details)` for `destructive`; respects
  `config.confirm_mode` (`ask` | `allow` | `deny`) and `config.auto_approve_domains`.
* Audit trail: append-only JSONL `config.audit_path` with timestamp, url, tool call,
  risk, decision, and the element description. Passwords/secrets are never logged.
* Classify by *meaning*, never by site: the rules must work on an unseen site (test this).

## cli.py

* `webpilot run "task text"` and interactive mode `webpilot` (prompt for a task).
* Live, readable trace: each tool call with its arguments, the risk decision, the page
  delta (URL + changed elements), token usage per step and cumulative, sub-agent calls.
* A stdin reader thread pushes human instructions into the UI queue *while the agent works*
  (`\q` to stop, `\s` to skip to the next step, `\t` for stats).
* `--headless`, `--model`, `--provider`, `--max-steps`, `--confirm-mode`, `--profile-dir`,
  `--transcript-dir`, `--verbose` flags; `--help` documents everything.
* Exit codes: 0 success, 1 task failure, 2 config/credential error, 130 interrupted.

## Testing

* `pytest -q` must pass with **no network and no API keys**. Live tests are marked
  `@pytest.mark.live` and skipped by default (`-m live` to run).
* Browser tests use the bundled local site (`tests/conftest.py::fixture_site`) and run
  headless; they assert on `PageModel`/`ToolResult` contents, not on screenshots.
* Every module added here needs at least: one happy-path test, one failure-path test, and
  one test that pins the *anti-hardcoding* invariant where it is testable
  (e.g. `test_snapshot_ids_are_assignable_without_selectors`,
  `test_policy_flags_unseen_checkout_flow_by_meaning`).
