"""Tests for ``webpilot.agent.tools`` and ``webpilot.agent.prompts`` (offline).

No browser, no network, no API key: the executor is a stub and the sub-agents are
replaced by recorders.  The last three tests pin the assignment's
anti-hardcoding invariant: prompts and tool descriptions may describe
*capabilities*, never a site, an address, a selector or a step list.
"""

from __future__ import annotations

import re
from pathlib import Path

from webpilot.agent import prompts
from webpilot.agent.memory import ContextManager
from webpilot.agent.tools import (
    BROWSER_TOOLS,
    LOCAL_TOOLS,
    SPEC_BY_NAME,
    TOOL_NAMES,
    TOOL_SPECS,
    DispatchContext,
    ToolRegistry,
    extra_arg_names,
    fallback_render_page,
    validate_args,
)
from webpilot.tokenizer import count_tokens
from webpilot.types import (
    Element,
    ExtractionResult,
    PageModel,
    ToolCall,
    ToolResult,
    Usage,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_TOOL_NAMES = {
    "page_outline",
    "goto",
    "click",
    "type_text",
    "press_key",
    "scroll",
    "go_back",
    "wait_for",
    "tabs",
    "ask_page",
    "screenshot",
    "notes",
    "ask_user",
    "finish",
}

#: Patterns that would mean a prompt knows a particular site, flow or selector.
FORBIDDEN_PATTERNS = [
    "http",
    "https",
    "www.",
    "//",
    "saucedemo",
    "hh.ru",
    "amazon",
    "ebay",
    "google.",
    "youtube",
    "linkedin",
    "wikipedia",
    "booking.com",
    "data-qa",
    "data-testid",
    "data-webpilot",
    "queryselector",
    "xpath",
    "getbyrole",
    ".btn",
    "btn-primary",
    "#cookie",
    "/vacanc",
    "/login",
    "/signin",
    "/cart",
    "/checkout",
    "/search",
    "/product",
    "step 1:",
    "step 2:",
    "step 1 -",
    "first click",
    "then click",
    "add to cart",
    "sign in with",
]


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class StubExecutor:
    """Duck-typed stand-in for ``browser.actions.ActionExecutor``."""

    def __init__(self, results: list[ToolResult] | None = None, default: ToolResult | None = None):
        self.results = list(results or [])
        self.default = default or ToolResult(ok=True, summary="did it")
        self.calls: list[ToolCall] = []

    def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        if self.results:
            return self.results.pop(0)
        return self.default


class ExplodingExecutor:
    def execute(self, call: ToolCall) -> ToolResult:
        raise RuntimeError("the browser died")


class StubExtractor:
    def __init__(self, result: ExtractionResult | None = None):
        self.result = result or ExtractionResult(answer="42", evidence="the answer is 42", confidence=0.9)
        self.calls: list[tuple[str, str, str]] = []
        self.usage = Usage()

    def answer(self, question: str, page_text: str, html_excerpt: str = "") -> ExtractionResult:
        self.calls.append((question, page_text, html_excerpt))
        return self.result


def make_page(**kwargs) -> PageModel:
    defaults = dict(
        url="about:blank",
        title="A page",
        generation=1,
        elements=[Element(id=1, role="button", name="Primary", tag="button", dom_hash="h1")],
        text="Some readable text.",
    )
    defaults.update(kwargs)
    return PageModel(**defaults)  # type: ignore[arg-type]


def make_registry(config, ui, **kwargs) -> ToolRegistry:
    kwargs.setdefault("executor", StubExecutor())
    return ToolRegistry(config=config, ui=ui, **kwargs)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


def test_tool_specs_match_the_interface_table():
    assert set(TOOL_NAMES) == EXPECTED_TOOL_NAMES
    assert len(TOOL_SPECS) == len(EXPECTED_TOOL_NAMES)
    assert BROWSER_TOOLS | LOCAL_TOOLS == EXPECTED_TOOL_NAMES
    assert BROWSER_TOOLS & LOCAL_TOOLS == set()
    for spec in TOOL_SPECS:
        assert spec.description.strip(), spec.name
        assert len(spec.description) > 40, spec.name
        assert spec.strict is True


def test_schemas_are_strict_and_every_property_is_required():
    for spec in TOOL_SPECS:
        params = spec.parameters
        assert params["type"] == "object", spec.name
        assert params["additionalProperties"] is False, spec.name
        properties = params["properties"]
        assert isinstance(properties, dict)
        assert params["required"] == list(properties), spec.name
        for key, schema in properties.items():
            assert schema.get("type") in ("string", "integer", "number", "boolean"), (spec.name, key)
            assert schema.get("description"), (spec.name, key)
    # the two provider payload shapes must stay well formed
    for spec in TOOL_SPECS:
        openai_payload = spec.to_openai()
        assert openai_payload["type"] == "function"
        assert openai_payload["function"]["parameters"] is spec.parameters
        assert spec.to_anthropic()["input_schema"] is spec.parameters


def test_optional_arguments_demand_an_explicit_sentinel():
    """Optional information is passed as a sentinel value, never by omission."""
    element_id = SPEC_BY_NAME["type_text"].parameters["properties"]["element_id"]
    assert "0" in element_id["description"]
    filter_arg = SPEC_BY_NAME["page_outline"].parameters["properties"]["filter"]
    assert "empty string" in filter_arg["description"]
    text_arg = SPEC_BY_NAME["wait_for"].parameters["properties"]["text"]
    assert "empty string" in text_arg["description"]
    intent = SPEC_BY_NAME["click"].parameters["properties"]["intent"]
    assert "why" in intent["description"].lower()
    assert SPEC_BY_NAME["page_outline"].parameters["properties"]["mode"]["enum"] == [
        "full",
        "delta",
        "text",
    ]
    assert SPEC_BY_NAME["scroll"].parameters["properties"]["direction"]["enum"] == [
        "up",
        "down",
        "top",
        "bottom",
    ]
    assert SPEC_BY_NAME["tabs"].parameters["properties"]["action"]["enum"] == [
        "list",
        "open",
        "switch",
        "close",
    ]
    assert SPEC_BY_NAME["go_back"].parameters["properties"] == {}


def test_no_selector_or_secret_instruction_in_a_tool_description():
    for spec in TOOL_SPECS:
        blob = (spec.description + " " + " ".join(
            str(schema.get("description", "")) for schema in spec.parameters["properties"].values()
        )).lower()
        for pattern in FORBIDDEN_PATTERNS:
            assert pattern not in blob, (spec.name, pattern)


# --------------------------------------------------------------------------- #
# Argument handling
# --------------------------------------------------------------------------- #


def test_unknown_tool_is_rejected_with_the_valid_names():
    registry = make_registry(None, None)
    result = registry.dispatch(ToolCall(name="teleport", args={}))
    assert result.ok is False
    for name in TOOL_NAMES:
        assert name in result.summary
    assert result.data["valid_tools"] == TOOL_NAMES


def test_malformed_arguments_are_rejected_with_a_helpful_message():
    registry = make_registry(None, None)
    for call in (
        ToolCall(name="click", args={}),                                   # missing args
        ToolCall(name="click", args={"element_id": -3, "intent": "x"}),    # below minimum
        ToolCall(name="click", args={"element_id": "abc", "intent": "x"}),  # not an integer
        ToolCall(name="scroll", args={"direction": "sideways", "amount": 10}),
        ToolCall(name="type_text", args={"element_id": 1, "text": 5, "submit": "maybe", "clear": False}),
        ToolCall(name="finish", args={"success": "perhaps", "answer": "x"}),
        ToolCall(name="goto", args="not-an-object"),                       # type: ignore[arg-type]
    ):
        result = registry.dispatch(call)
        assert result.ok is False, call
        assert TOOL_NAMES[0] in result.summary


def test_chatty_models_are_coerced_not_rejected():
    clean, error = validate_args("click", {"element_id": "3", "intent": "open"})
    assert error is None and clean["element_id"] == 3
    clean, error = validate_args(
        "type_text", {"element_id": 0, "text": 7, "submit": "true", "clear": "false"}
    )
    assert error is None
    assert clean == {"element_id": 0, "text": "7", "submit": True, "clear": False}
    clean, error = validate_args("page_outline", {"mode": "FULL", "filter": None})
    assert error is None and clean == {"mode": "full", "filter": ""}
    clean, error = validate_args("wait_for", {"seconds": None, "text": "", "timeout_seconds": 5})
    assert error is None and clean["seconds"] == 0.0


def test_extra_arguments_are_ignored_but_reported():
    assert extra_arg_names("click", {"element_id": 1, "intent": "x", "oops": 2}) == ["oops"]
    registry = make_registry(None, None)
    result = registry.dispatch(
        ToolCall(name="notes", args={"text": "hi", "oops": 1}),
    )
    assert result.data.get("ignored_arguments") == ["oops"]


def test_dispatch_never_raises_on_garbage():
    registry = make_registry(None, None)
    for call in (ToolCall(name="", args={}), ToolCall(name="click", args=None)):  # type: ignore[arg-type]
        assert registry.dispatch(call).ok is False
    assert registry.dispatch({"name": "notes", "args": {"text": "from a dict"}}).ok is False  # type: ignore[arg-type]
    assert registry.dispatch(None).ok is False  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Browser tools delegate
# --------------------------------------------------------------------------- #


def test_browser_tools_delegate_to_the_executor():
    executor = StubExecutor([ToolResult(ok=True, summary="clicked", page_changed=True)])
    registry = ToolRegistry(executor=executor)
    result = registry.dispatch(ToolCall(name="click", args={"element_id": "2", "intent": "open"}))
    assert result.ok is True
    assert [call.name for call in executor.calls] == ["click"]
    assert executor.calls[0].args == {"element_id": 2, "intent": "open"}
    assert registry.calls == 1


def test_executor_exception_becomes_a_failed_result_not_a_crash():
    registry = ToolRegistry(executor=ExplodingExecutor())
    result = registry.dispatch(ToolCall(name="click", args={"element_id": 1, "intent": "x"}))
    assert result.ok is False
    assert "RuntimeError" in result.summary
    assert registry.errors == 1


def test_missing_executor_is_a_helpful_failure():
    registry = ToolRegistry()
    result = registry.dispatch(ToolCall(name="goto", args={"url": "about:blank"}))
    assert result.ok is False
    assert "browser" in result.summary


# --------------------------------------------------------------------------- #
# Agent-local tools
# --------------------------------------------------------------------------- #


def test_notes_tool_updates_the_context_scratchpad(config):
    context = ContextManager(config)
    registry = ToolRegistry(context=context, ui=None)
    result = registry.dispatch(ToolCall(name="notes", args={"text": "found the price"}))
    assert result.ok is True
    assert "found the price" in result.data["notes"]
    assert "found the price" in context.notes


def test_notes_tool_without_context_fails_clearly():
    registry = ToolRegistry()
    result = registry.dispatch(ToolCall(name="notes", args={"text": "x"}))
    assert result.ok is False and "scratchpad" in result.summary


def test_ask_user_returns_the_human_answer(ui):
    registry = ToolRegistry(ui=ui)
    ui._answers = ["the code is on my phone"]  # type: ignore[attr-defined]
    result = registry.dispatch(ToolCall(name="ask_user", args={"question": "which code?"}))
    assert result.ok is True
    assert "the code is on my phone" in result.summary
    assert ui.questions == ["which code?"]
    assert any(event.kind == "user" for event in ui.events)


def test_finish_sets_the_completion_flag():
    registry = ToolRegistry()
    result = registry.dispatch(ToolCall(name="finish", args={"success": True, "answer": "all done"}))
    assert result.ok is True
    assert registry.finished is True
    assert registry.finish_success is True
    assert registry.finish_answer == "all done"


def test_page_outline_uses_the_rendered_page_callback():
    page = make_page(elements=[Element(id=7, role="link", name="Details", tag="a")])
    rendered = []

    def provider(mode: str, needle: str):
        rendered.append((mode, needle))
        return page, "URL: about:blank\n[7] link 'Details'\n[8] button 'Other'"

    registry = ToolRegistry(page_provider=provider)
    result = registry.dispatch(ToolCall(name="page_outline", args={"mode": "full", "filter": "Details"}))
    assert rendered == [("full", "Details")]
    assert result.ok is True
    assert "[7]" in result.data["page_text"]
    assert "[8]" not in result.data["page_text"]
    assert result.page is page
    assert result.page_changed is False


def test_page_outline_without_a_page_fails_helpfully():
    registry = ToolRegistry(page_provider=lambda mode, needle: (None, ""))
    result = registry.dispatch(ToolCall(name="page_outline", args={"mode": "delta", "filter": ""}))
    assert result.ok is False and "no page" in result.summary


def test_page_outline_falls_back_to_the_context_page(config):
    page = make_page()
    registry = ToolRegistry(config=config, context=ContextManager(config))
    ctx = DispatchContext(page=page, task="t")
    result = registry.dispatch(ToolCall(name="page_outline", args={"mode": "full", "filter": ""}), ctx)
    assert result.ok is True
    assert "URL: about:blank" in result.data["page_text"]


def test_ask_page_delegates_to_the_extractor_and_never_returns_markup(ui):
    extractor = StubExtractor(
        ExtractionResult(answer="USD 12.50", evidence="price USD 12.50", confidence=0.8,
                         usage=Usage(prompt_tokens=300, completion_tokens=20, calls=1))
    )
    page = make_page(text="Price: USD 12.50")
    raw = ["<html><body><div class='x'>Price: USD 12.50</div></body></html>"]
    registry = ToolRegistry(
        ui=ui,
        extractor=extractor,
        raw_provider=lambda: raw[0],
        page_provider=lambda mode, needle: (page, page.text),
    )
    result = registry.dispatch(ToolCall(name="ask_page", args={"question": "what is the price?"}))
    assert result.ok is True
    assert "USD 12.50" in result.summary
    assert result.data["answer"] == "USD 12.50"
    assert result.data["found"] is True
    assert "html" not in result.summary.lower()
    question, page_text, html = extractor.calls[0]
    assert question == "what is the price?"
    assert "Price: USD 12.50" in page_text
    assert html.startswith("<html>")
    assert registry.subagent_calls == 1
    assert registry.extractor_usage.prompt_tokens == 300
    assert any(event.kind == "subagent" for event in ui.events)


def test_ask_page_reports_absence_and_missing_extractor():
    absent = StubExtractor(ExtractionResult(answer="", evidence="", confidence=0.0))
    registry = ToolRegistry(extractor=absent, page_provider=lambda mode, needle: (make_page(), "text"))
    result = registry.dispatch(ToolCall(name="ask_page", args={"question": "who won?"}))
    assert result.ok is True
    assert result.data["found"] is False
    assert "does not contain" in result.summary
    bare = ToolRegistry()
    failed = bare.dispatch(ToolCall(name="ask_page", args={"question": "who won?"}))
    assert failed.ok is False and "unavailable" in failed.summary


def test_a_broken_extractor_does_not_break_dispatch():
    class Boom:
        usage = Usage()

        def answer(self, *args, **kwargs):
            raise RuntimeError("model exploded")

    registry = ToolRegistry(extractor=Boom(), page_provider=lambda mode, needle: (make_page(), "t"))
    result = registry.dispatch(ToolCall(name="ask_page", args={"question": "q"}))
    assert result.ok is False
    assert "RuntimeError" in result.summary


# --------------------------------------------------------------------------- #
# Rendering fallback
# --------------------------------------------------------------------------- #


def test_fallback_renderer_follows_the_rendering_contract():
    page = make_page(
        url="about:blank",
        title="A page",
        generation=3,
        alerts=["Something went wrong"],
        dialogs=["Are you sure?"],
        elements=[
            Element(id=3, role="textbox", name="Query", tag="input", placeholder="Query"),
            Element(id=5, role="submit", name="Go", tag="button"),
        ],
        text="heading: Hello\nparagraph: Body",
    )
    text = fallback_render_page(page, budget_tokens=500)
    assert text.splitlines()[0] == "URL: about:blank | Title: A page | generation 3"
    assert "ALERTS: Something went wrong" in text
    assert "DIALOG: Are you sure?" in text
    assert "[3] textbox 'Query' placeholder='Query'" in text
    assert "TEXT:" in text
    assert count_tokens(text) <= 500


def test_fallback_renderer_respects_the_budget():
    page = make_page(
        elements=[Element(id=i, role="button", name=f"Button {i}", tag="button") for i in range(1, 400)],
        text="x " * 5_000,
    )
    text = fallback_render_page(page, budget_tokens=200)
    assert count_tokens(text) <= 200


# --------------------------------------------------------------------------- #
# Prompts: capability-only, site-agnostic
# --------------------------------------------------------------------------- #


def _prompt_strings() -> list[str]:
    source = (REPO_ROOT / "src" / "webpilot" / "agent" / "prompts.py").read_text(encoding="utf-8")
    page = make_page(
        url="about:blank",
        title="A page",
        alerts=["a"],
        dialogs=["b"],
        elements=[Element(id=1, role="button", name="Go", tag="button")],
    )
    from webpilot.types import FailureContext, StepRecord

    step = StepRecord(index=1, tool_call=ToolCall("click", {"element_id": 1, "intent": "open"}), result_summary="ok", ok=True)
    failure = FailureContext(task="find a value", tool_call=step.tool_call, error="nothing changed", attempts=2, page=page, history=[step.digest()])
    return [
        source,
        prompts.SYSTEM_PROMPT,
        prompts.system_prompt(ConfigStub()),
        prompts.EXTRACTOR_SYSTEM_PROMPT,
        prompts.RECOVERY_SYSTEM_PROMPT,
        prompts.SUMMARIZER_SYSTEM_PROMPT,
        prompts.extractor_user_prompt("what is the total?", "some page text", "<html></html>"),
        prompts.recovery_user_prompt(failure, "URL: about:blank", TOOL_NAMES),
        prompts.summarizer_user_prompt("do a thing", [step], "earlier"),
        prompts.first_user_message("do a thing"),
        prompts.scratchpad_message("notes here"),
        prompts.summary_message("summary here"),
        prompts.current_page_message("block", generation=1, url="about:blank", title="t", step=2),
        prompts.page_marker_message(step=2, page_hash="abc", unchanged=True),
        prompts.page_marker_message(step=2, page_hash="abc", generation=3, unchanged=False),
        prompts.no_page_message(),
        prompts.human_instruction_message("stop and look"),
        prompts.warning_message("careful"),
        prompts.denial_message("no"),
        prompts.failure_advice_message("try the menu", ["click(element_id=1, intent='open')"]),
        prompts.give_up_message("nothing left"),
        prompts.no_tool_call_message(),
        prompts.loop_warning_message("click(element_id=1)", 3),
        prompts.step_limit_message(10),
        prompts.failure_limit_message(4),
        prompts.extraction_summary("42", 0.9, True),
        prompts.tool_result_message(1, "click", "OK"),
    ]


class ConfigStub:
    max_steps = 12
    history_window = 8


def test_prompts_are_site_agnostic():
    """No prompt may know a site, an address, a selector or a fixed step list."""
    for text in _prompt_strings():
        lowered = text.lower()
        for pattern in FORBIDDEN_PATTERNS:
            assert pattern not in lowered, (pattern, text[:200])
    # and no bare path-like fragment (`/something`) either
    source = (REPO_ROOT / "src" / "webpilot" / "agent" / "prompts.py").read_text(encoding="utf-8")
    assert not re.search(r"/[a-z][a-z0-9_-]{2,}", source), "a URL path leaked into prompts.py"


def test_prompts_describe_the_loop_and_the_rules():
    text = prompts.SYSTEM_PROMPT
    for fragment in (
        "OBSERVE",
        "DECIDE",
        "VERIFY",
        "ALERTS",
        "DIALOG",
        "element ids",
        "notes",
        "ask_user",
        "finish",
        "never",
        "ask_page",
    ):
        assert fragment.lower() in text.lower(), fragment
    assert "ALERTS" in text and "DIALOG" in text
    assert "success=true" in text.lower() or "success=true" in text


def test_prompt_builders_carry_the_task_and_the_observation():
    first = prompts.first_user_message("  order the cheapest item  ")
    assert "order the cheapest item" in first
    observation = prompts.current_page_message("URL: about:blank", generation=4, url="about:blank", title="T", step=9)
    assert "[step 9]" in observation and "generation 4" in observation and "URL: about:blank" in observation
    marker = prompts.page_marker_message(step=5, page_hash="deadbeef", unchanged=True)
    assert "unchanged since step 5" in marker and "deadbeef" in marker
    scratch = prompts.scratchpad_message("")
    assert scratch and "empty" in scratch  # the scratchpad is always present


def test_extractor_and_recovery_prompts_demand_evidence():
    extractor = prompts.EXTRACTOR_SYSTEM_PROMPT.lower()
    assert "evidence" in extractor
    assert "confidence" in extractor
    assert "empty answer" in extractor or "not present" in extractor
    assert "json" in extractor
    recovery = prompts.RECOVERY_SYSTEM_PROMPT.lower()
    assert "alternative" in recovery and "give_up" in recovery


def test_summarizer_prompt_keeps_the_important_parts():
    text = prompts.SUMMARIZER_SYSTEM_PROMPT.lower()
    assert "task" in text and "accomplished" in text and "bullets" in text
