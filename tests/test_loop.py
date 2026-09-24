"""Tests for ``webpilot.agent.loop.Agent`` (offline).

Everything here runs without a browser, a network or an API key: the executor is a
stub that returns canned ``ToolResult``/``PageModel`` objects and the model is a
scripted ``LLMClient``.  The tests assert on the ``RunResult`` fields, on the JSONL
transcript and on the emitted ``RunEvent`` stream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from webpilot.agent.loop import Agent
from webpilot.agent.subagents import ExtractorSubAgent, RecoverySubAgent, Summarizer
from webpilot.agent.tools import TOOL_NAMES, ToolRegistry, fallback_render_page
from webpilot.errors import LLMError
from webpilot.tokenizer import count_tokens
from webpilot.types import (
    ActionRisk,
    Element,
    ExtractionResult,
    FailureContext,
    LLMMessage,
    LLMResponse,
    PageModel,
    RecoveryProposal,
    ToolCall,
    ToolResult,
    Usage,
)

UNSET = object()


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class ScriptedLLM:
    """Tiny deterministic LLMClient: canned responses, every request recorded."""

    provider = "scripted"
    model = "scripted-1"

    def __init__(self, script: list[Any], *, fail_times: int = 0, repeat_last: bool = False):
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.fail_times = fail_times
        self.repeat_last = repeat_last
        self.last: Any = None

    def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[Any] | None = None,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        self.requests.append(
            {
                "messages": [
                    LLMMessage(
                        role=message.role,
                        content=message.content,
                        tool_calls=list(message.tool_calls),
                        tool_call_id=message.tool_call_id,
                        name=message.name,
                    )
                    for message in messages
                ],
                "system": system,
                "tools": list(tools or []),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self.fail_times > 0:
            self.fail_times -= 1
            raise LLMError("the provider is unreachable", status=500, retryable=True)
        if self.script:
            item = self.script.pop(0)
        elif self.repeat_last and self.last is not None:
            item = self.last
        else:
            item = LLMResponse(
                text="(the script ran out)",
                tool_calls=[ToolCall("finish", {"success": False, "answer": "the script ran out"})],
            )
        if callable(item):
            item = item(self.requests[-1])
        self.last = item
        return item

    def count_tokens(self, text: str) -> int:
        return count_tokens(text)


class StubExecutor:
    def __init__(self, results: list[ToolResult] | None = None, default: ToolResult | None = None):
        self.results = list(results or [])
        self.default = default or ToolResult(ok=True, summary="did it", page=PAGE_B, page_changed=True)
        self.calls: list[ToolCall] = []

    def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        if self.results:
            return self.results.pop(0)
        return self.default


class StubSession:
    """Duck-typed BrowserSession: snapshot() plus (optionally) a renderer."""

    def __init__(self, pages: list[PageModel] | None = None, page: PageModel | None = None):
        self.pages = list(pages or [])
        self.page = page or PAGE_B
        self.snapshots = 0
        self.raw_reads = 0

    def snapshot(self, *, budget_tokens: int, mode: str = "full") -> PageModel:
        self.snapshots += 1
        if self.pages:
            return self.pages.pop(0)
        return self.page

    def raw_html(self, max_chars: int = 120_000) -> str:
        self.raw_reads += 1
        return "<html><body>markup that must never reach the main conversation</body></html>"

    def raw_text(self, max_chars: int = 120_000) -> str:
        raise AssertionError("the agent loop must not read raw page text")


class StubPolicy:
    """Mirrors SecurityPolicy: classify() then authorize(call, risk, ctx, confirm)."""

    def __init__(self, level: str = "destructive", approve: bool = True, reason: str = "not now", explode: bool = False):
        self.level = level
        self.approve = approve
        self.reason = reason
        self.explode = explode
        self.classified: list[ToolCall] = []
        self.authorized: list[ToolCall] = []

    def classify(self, call: ToolCall, ctx: Any) -> ActionRisk:
        if self.explode:
            raise RuntimeError("the rule engine broke")
        self.classified.append(call)
        if call.name in ("click", "type_text", "goto"):
            return ActionRisk(level=self.level, reasons=["test rule"], matched_rules=["test"])
        return ActionRisk(level="safe")

    def authorize(self, call: ToolCall, risk: ActionRisk, ctx: Any, confirm: Any):
        self.authorized.append(call)
        # the real policy asks the human through the UI; this stub mirrors that
        answer = confirm(f"confirm the {risk.level} action {call.name}?", "details")
        self.last_confirm_answer = answer
        if self.approve:
            return True, ""
        return False, self.reason


class StubRecovery:
    def __init__(self, advice: str = "the overlay is in the way; dismiss it", give_up: bool = False):
        self.advice = advice
        self.give_up = give_up
        self.calls = 0
        self.failures: list[FailureContext] = []
        self.usage = Usage()

    def propose(self, failure: FailureContext) -> RecoveryProposal:
        self.calls += 1
        self.failures.append(failure)
        call_usage = Usage(prompt_tokens=111, completion_tokens=22, calls=1)
        self.usage = self.usage.add(call_usage)
        return RecoveryProposal(
            advice=self.advice,
            alternative_actions=[ToolCall("press_key", {"key": "Escape"})],
            give_up=self.give_up,
            usage=call_usage,
        )


class StubExtractor:
    def __init__(self, answer: str = "42"):
        self.answer_text = answer
        self.calls: list[tuple[str, str, str]] = []
        self.usage = Usage()

    def answer(self, question: str, page_text: str, html_excerpt: str = "") -> ExtractionResult:
        self.calls.append((question, page_text, html_excerpt))
        call_usage = Usage(prompt_tokens=300, completion_tokens=20, calls=1)
        self.usage = self.usage.add(call_usage)
        return ExtractionResult(
            answer=self.answer_text, evidence="snippet", confidence=0.9, usage=call_usage
        )


class StubSummarizer:
    def __init__(self):
        self.calls = 0
        self.usage = Usage()

    def summarize(self, task: str, steps: Any, previous_summary: str = "") -> str:
        self.calls += 1
        self.usage = self.usage.add(Usage(prompt_tokens=50, completion_tokens=10, calls=1))
        return "SUMMARY: " + "; ".join(step.digest() for step in steps)[:300]


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #

PAGE_A = PageModel(
    url="about:blank",
    title="Start",
    generation=1,
    elements=[Element(id=1, role="button", name="Primary", tag="button", dom_hash="a1")],
    text="start page",
)

PAGE_B = PageModel(
    url="about:blank",
    title="Second",
    generation=2,
    elements=[
        Element(id=2, role="textbox", name="Query", tag="input", dom_hash="b1"),
        Element(id=3, role="submit", name="Go", tag="button", dom_hash="b2"),
    ],
    text="second page",
)

PAGE_ERROR = PageModel(
    url="about:blank",
    title="Sign in",
    generation=3,
    elements=[Element(id=4, role="textbox", name="User", tag="input", invalid=True, dom_hash="e1")],
    alerts=["Invalid username or password."],
)

PAGE_PASSWORD = PageModel(
    url="about:blank",
    title="Sign in",
    generation=1,
    elements=[
        Element(id=5, role="textbox", name="Password", tag="input", type="password", dom_hash="p1"),
        Element(id=6, role="submit", name="Sign in", tag="button", dom_hash="p2"),
    ],
)


def make_agent(
    config,
    ui,
    script: Any,
    *,
    executor: Any = None,
    session: Any = None,
    policy: Any = None,
    extractor: Any = None,
    recovery: Any = None,
    summarizer: Any = None,
    renderer: Any = UNSET,
    raw_provider: Any = None,
    llm: ScriptedLLM | None = None,
) -> tuple[Agent, ScriptedLLM, StubExecutor]:
    """Build an Agent whose only real parts are the loop and the context manager."""
    client = llm if llm is not None else (script if isinstance(script, ScriptedLLM) else ScriptedLLM(script))
    stub = executor if executor is not None else StubExecutor()
    agent = Agent(
        config,
        client,
        session=session,
        executor=stub,
        ui=ui,
        policy=policy,
        extractor=extractor if extractor is not None else StubExtractor(),
        recovery=recovery if recovery is not None else StubRecovery(),
        summarizer=summarizer if summarizer is not None else StubSummarizer(),
        renderer=fallback_render_page if renderer is UNSET else renderer,
        raw_provider=raw_provider,
    )
    return agent, client, stub


def click(element_id: int = 1, intent: str = "open", call_id: str | None = None) -> LLMResponse:
    return LLMResponse(
        text="clicking the element",
        tool_calls=[ToolCall("click", {"element_id": element_id, "intent": intent}, id=call_id or f"c{element_id}")],
        usage=Usage(prompt_tokens=100, completion_tokens=10, calls=1),
    )


def finish(success: bool = True, answer: str = "done") -> LLMResponse:
    return LLMResponse(
        text="finishing",
        tool_calls=[ToolCall("finish", {"success": success, "answer": answer}, id="fin")],
        usage=Usage(prompt_tokens=120, completion_tokens=8, calls=1),
    )


def request_text(llm: ScriptedLLM, index: int = -1) -> str:
    request = llm.requests[index]
    blob = (request["system"] or "") + "\n"
    for message in request["messages"]:
        blob += f"{message.role}: {message.content}\n"
        blob += " ".join(call.signature() for call in message.tool_calls) + "\n"
    return blob


def transcript_lines(result) -> list[dict]:
    path = Path(result.transcript_path)
    assert path.exists(), "the transcript must be written"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_happy_path_ends_in_a_verified_finish(config, ui):
    executor = StubExecutor([ToolResult(ok=True, summary="clicked the primary button", page=PAGE_B, page_changed=True)])
    agent, llm, _ = make_agent(config, ui, [click(1, "open"), finish(True, "the item was added")], executor=executor)
    result = agent.run("add the item to the cart")

    assert result.success is True
    assert result.steps == 2
    assert result.answer == "the item was added"
    assert result.error is None
    assert result.transcript_path and Path(result.transcript_path).exists()
    assert [call.name for call in executor.calls] == ["click"]
    assert result.usage.total > 0
    assert result.usage.main.total == result.usage.total  # no sub-agent ran
    assert result.subagent_calls == 0

    # the second request carries the tool result and the fresh observation
    second = request_text(llm, 1)
    assert "OK: clicked the primary button" in second
    assert "[3] submit 'Go'" in second
    kinds = ui.kinds()
    for kind in ("start", "thinking", "tool_call", "tool_result", "page", "context", "risk", "finish"):
        assert kind in kinds, kind
    assert llm.requests[0]["tools"], "the tool schemas must be offered to the model"
    assert "add the item to the cart" in request_text(llm, 0)


def test_finish_success_is_rejected_when_the_page_contradicts_it(config, ui):
    executor = StubExecutor(
        [ToolResult(ok=True, summary="submitted the form", page=PAGE_ERROR, page_changed=True)]
    )
    agent, _, _ = make_agent(config, ui, [click(3, "submit"), finish(True, "you are signed in")], executor=executor)
    result = agent.run("sign in")

    assert result.success is False
    assert "you are signed in" in result.answer
    assert "not verified" in result.answer
    assert "Invalid username or password." in (result.error or "")
    assert any(event.kind == "error" for event in ui.events)


def test_an_honest_failure_finish_is_reported_as_incomplete(config, ui):
    executor = StubExecutor([ToolResult(ok=False, summary="nothing happened", error="element disabled")])
    agent, _, _ = make_agent(config, ui, [click(1, "try"), finish(False, "the button was disabled")], executor=executor)
    result = agent.run("press the button")
    assert result.success is False
    assert result.answer == "the button was disabled"
    assert result.steps == 2


def test_multiple_tool_calls_in_one_turn_run_in_order(config, ui):
    executor = StubExecutor([ToolResult(ok=True, summary="clicked", page=PAGE_B, page_changed=True)])
    turn = LLMResponse(
        text="doing three things",
        tool_calls=[
            ToolCall("notes", {"text": "remember the button id"}, id="n1"),
            ToolCall("click", {"element_id": 1, "intent": "open"}, id="c1"),
            ToolCall("press_key", {"key": "Escape"}, id="k1"),
        ],
        usage=Usage(prompt_tokens=200, completion_tokens=20, calls=1),
    )
    agent, llm, _ = make_agent(config, ui, [turn, finish(True, "all three done")], executor=executor)
    result = agent.run("do three things")
    assert result.steps == 4
    assert [call.name for call in executor.calls] == ["click", "press_key"]
    assert "remember the button id" in agent.context.notes
    # one tool message per call, all in the same turn
    third = request_text(llm, 1)
    assert "[step 1] notes" in third and "[step 2] click" in third and "[step 3] press_key" in third
    assert third.count("assistant: doing three things") == 1  # the reasoning is not duplicated


def test_run_without_a_session_still_works(config, ui):
    agent, llm, _ = make_agent(
        config, ui, [LLMResponse(text="thinking", tool_calls=[ToolCall("notes", {"text": "no page needed"})]), finish(True, "recorded")],
    )
    result = agent.run("just remember something")
    assert result.success is True
    assert any(event.kind == "info" and "no page observation" in event.message for event in ui.events)


# --------------------------------------------------------------------------- #
# Malformed and unknown tools
# --------------------------------------------------------------------------- #


def test_unknown_tool_is_reported_with_the_valid_names(config, ui):
    executor = StubExecutor()
    agent, llm, _ = make_agent(
        config,
        ui,
        [
            LLMResponse(text="trying", tool_calls=[ToolCall("teleport", {"where": "elsewhere"}, id="t1")]),
            finish(False, "could not continue"),
        ],
        executor=executor,
    )
    result = agent.run("do the impossible")
    assert executor.calls == []
    assert result.steps == 2
    second = request_text(llm, 1)
    assert "unknown tool 'teleport'" in second
    for name in TOOL_NAMES:
        assert name in second


def test_malformed_arguments_never_reach_the_executor(config, ui):
    executor = StubExecutor()
    agent, llm, _ = make_agent(
        config,
        ui,
        [
            LLMResponse(text="trying", tool_calls=[ToolCall("click", {"element_id": 1}, id="c1")]),
            finish(False, "gave up"),
        ],
        executor=executor,
    )
    agent.run("click something")
    assert executor.calls == []
    assert "missing required argument" in request_text(llm, 1)


# --------------------------------------------------------------------------- #
# Security gate
# --------------------------------------------------------------------------- #


def test_a_denied_destructive_action_is_fed_back_and_not_executed(config, ui):
    executor = StubExecutor()
    policy = StubPolicy(level="destructive", approve=False, reason="not without my approval")
    agent, llm, _ = make_agent(
        config,
        ui,
        [click(1, "delete the account"), finish(False, "the human declined")],
        executor=executor,
        policy=policy,
    )
    result = agent.run("delete the account")

    assert executor.calls == [], "a denied call must never be executed"
    assert policy.authorized, "the policy must have been asked to authorize"
    assert result.confirmations == 1
    assert ui.prompts and "confirm" in ui.prompts[0][0]
    second = request_text(llm, 1)
    assert "the human denied this action" in second
    assert "not without my approval" in second
    assert any(event.kind == "risk" and event.data.get("level") == "destructive" for event in ui.events)
    assert any(event.kind == "confirm" for event in ui.events)


def test_an_approved_destructive_action_does_execute(config, ui):
    executor = StubExecutor([ToolResult(ok=True, summary="deleted", page=PAGE_B, page_changed=True)])
    policy = StubPolicy(level="destructive", approve=True)
    agent, _, _ = make_agent(
        config, ui, [click(1, "proceed"), finish(True, "done")], executor=executor, policy=policy
    )
    result = agent.run("proceed")
    assert [call.name for call in executor.calls] == ["click"]
    assert result.confirmations == 1


def test_a_policy_that_cannot_classify_blocks_the_action(config, ui):
    executor = StubExecutor()
    policy = StubPolicy(explode=True)
    agent, llm, _ = make_agent(
        config, ui, [click(1, "x"), finish(False, "blocked")], executor=executor, policy=policy
    )
    agent.run("do something risky")
    assert executor.calls == []
    assert "could not classify" in request_text(llm, 1)


def test_password_fields_are_never_typed_by_the_agent(config, ui):
    executor = StubExecutor()
    session = StubSession(page=PAGE_PASSWORD)
    script = [
        LLMResponse(
            text="typing the password",
            tool_calls=[ToolCall("type_text", {"element_id": 5, "text": "hunter2", "submit": True, "clear": True}, id="t1")],
        ),
        finish(False, "needs the human"),
    ]
    agent, llm, _ = make_agent(config, ui, script, executor=executor, session=session)
    result = agent.run("sign in")
    assert executor.calls == []
    text = request_text(llm, 1)
    assert "password fields are filled by the human" in text
    assert "FAILED: refused" in text
    assert result.success is False


# --------------------------------------------------------------------------- #
# Failure handling: recovery and loops
# --------------------------------------------------------------------------- #


def test_repeated_failure_asks_the_recovery_subagent_and_injects_its_advice(make_config, ui):
    config = make_config(max_attempts_per_action=2, max_consecutive_failures=9, loop_detection_repeats=99)
    session = StubSession(page=PAGE_A)
    executor = StubExecutor(
        [ToolResult(ok=False, summary="nothing changed", error="the element is covered by an overlay")] * 4
    )
    recovery = StubRecovery(advice="dismiss the newsletter overlay first")
    script = [click(1, "open"), click(1, "open"), click(1, "open"), finish(False, "still blocked")]
    agent, llm, _ = make_agent(config, ui, script, executor=executor, session=session, recovery=recovery)
    result = agent.run("open the menu")

    assert recovery.calls >= 1
    assert result.recoveries >= 1
    assert recovery.failures[0].tool_call.name == "click"
    assert recovery.failures[0].attempts >= 2
    assert "the element is covered" in recovery.failures[0].error
    assert recovery.failures[0].page is not None
    injected = request_text(llm, 2)
    assert "RECOVERY ADVISOR" in injected
    assert "dismiss the newsletter overlay first" in injected
    assert "press_key(key='Escape')" in injected  # the proposed alternative is offered
    assert any(event.kind == "subagent" for event in ui.events)
    assert result.usage.subagents.prompt_tokens >= 111


def test_recovery_with_give_up_tells_the_model_to_ask_or_finish(make_config, ui):
    config = make_config(max_attempts_per_action=2, max_consecutive_failures=9, loop_detection_repeats=99)
    executor = StubExecutor([ToolResult(ok=False, summary="nope", error="dead end")] * 3)
    recovery = StubRecovery(advice="no route left", give_up=True)
    script = [click(1, "x"), click(1, "x"), finish(False, "no route")]
    agent, llm, _ = make_agent(config, ui, script, executor=executor, recovery=recovery)
    agent.run("do it")
    injected = request_text(llm, 2)
    assert "exhausted" in injected
    assert "ask_user" in injected


def test_a_repeating_page_and_action_triggers_the_loop_warning(make_config, ui):
    config = make_config(loop_detection_repeats=3, max_attempts_per_action=99, max_consecutive_failures=99)
    executor = StubExecutor([ToolResult(ok=True, summary="clicked the same thing", page=PAGE_A)] * 6)
    script = [click(1, "open")] * 4 + [finish(False, "stuck")]
    agent, llm, _ = make_agent(config, ui, script, executor=executor)
    result = agent.run("open the thing")

    assert any(
        event.kind == "error" and "loop detected" in event.message for event in ui.events
    )
    warned = request_text(llm, 3)
    assert "repeated the same action" in warned
    assert result.recoveries >= 1  # recovery follows the warning
    assert "RECOVERY ADVISOR" in request_text(llm, 4)


def test_max_steps_produces_an_honest_partial_answer(make_config, ui):
    config = make_config(max_steps=2, loop_detection_repeats=99, max_attempts_per_action=99, max_consecutive_failures=99)
    executor = StubExecutor(default=ToolResult(ok=True, summary="clicked", page=PAGE_B))
    agent, _, _ = make_agent(config, ui, [click(1, "a"), click(1, "b"), click(1, "c")], executor=executor)
    result = agent.run("keep going forever")

    assert result.steps == 2
    assert result.success is False
    assert "limit" in result.answer.lower()
    assert "step limit" in (result.error or "")
    assert len(executor.calls) == 2


def test_max_consecutive_failures_ends_the_run_honestly(make_config, ui):
    config = make_config(max_consecutive_failures=2, max_attempts_per_action=99, loop_detection_repeats=99, max_steps=20)
    executor = StubExecutor(default=ToolResult(ok=False, summary="failed", error="broken"))
    agent, _, _ = make_agent(config, ui, [click(1, "a")] * 6, executor=executor)
    result = agent.run("try until it works")

    assert result.success is False
    assert "consecutive" in (result.error or "")
    assert result.steps <= 3


def test_a_model_that_never_calls_a_tool_is_asked_to_act_then_gives_up(make_config, ui):
    config = make_config(max_consecutive_failures=2)
    agent, llm, _ = make_agent(config, ui, [LLMResponse(text="I think the answer is 42"), LLMResponse(text="maybe 43")])
    result = agent.run("find the answer")
    assert result.success is False
    assert "did not call a tool" in request_text(llm, 1)
    assert result.steps == 0


# --------------------------------------------------------------------------- #
# Human in the loop
# --------------------------------------------------------------------------- #


def test_a_human_instruction_is_injected_before_the_next_model_call(config, ui):
    def first(request):
        ui.instructions.append("ignore the sidebar and use the search box")
        return click(1, "open")

    agent, llm, _ = make_agent(config, ui, [first, finish(True, "done")])
    agent.run("find the item")
    assert ui.instructions == []
    injected = request_text(llm, 1)
    assert "INSTRUCTION FROM THE HUMAN" in injected
    assert "ignore the sidebar" in injected
    assert any(event.kind == "user" for event in ui.events)


def test_ask_user_blocks_on_the_ui_and_the_answer_comes_back(config, ui):
    ui._answers = ["the code is 8842"]  # type: ignore[attr-defined]
    script = [
        LLMResponse(
            text="asking",
            tool_calls=[ToolCall("ask_user", {"question": "what is the one-time code?"}, id="u1")],
        ),
        finish(True, "signed in with the human's code"),
    ]
    agent, llm, _ = make_agent(config, ui, script)
    result = agent.run("sign in")
    assert ui.questions == ["what is the one-time code?"]
    assert "the code is 8842" in request_text(llm, 1)
    assert result.success is True


# --------------------------------------------------------------------------- #
# Page handling and perception
# --------------------------------------------------------------------------- #


def test_the_loop_never_reads_raw_markup(config, ui):
    session = StubSession(page=PAGE_A)
    executor = StubExecutor([ToolResult(ok=True, summary="clicked", page=PAGE_B, page_changed=True)])
    agent, _, _ = make_agent(config, ui, [click(1, "open"), finish(True, "done")], executor=executor, session=session)
    result = agent.run("click the primary control")
    assert session.raw_reads == 0, "raw markup must not be touched outside ask_page"
    assert result.success is True
    assert session.snapshots >= 1


def test_ask_page_uses_the_extractor_and_no_markup_enters_the_conversation(config, ui):
    session = StubSession(page=PAGE_A)
    executor = StubExecutor()
    extractor = StubExtractor(answer="USD 42.50")
    script = [
        LLMResponse(text="asking", tool_calls=[ToolCall("ask_page", {"question": "what is the total?"}, id="a1")]),
        finish(True, "the total is USD 42.50"),
    ]
    agent, llm, _ = make_agent(
        config, ui, script, executor=executor, session=session, extractor=extractor
    )
    result = agent.run("report the total")
    assert session.raw_reads == 1
    assert extractor.calls and extractor.calls[0][0] == "what is the total?"
    assert "<html>" in extractor.calls[0][2]
    conversation = request_text(llm, 1)
    assert "USD 42.50" in conversation
    assert "<html>" not in conversation
    assert "<body>" not in conversation
    assert any(event.kind == "subagent" for event in ui.events)
    assert result.subagent_calls >= 1
    assert result.usage.subagents.prompt_tokens >= 300
    assert result.usage.total == result.usage.main.total + result.usage.subagents.total


def test_the_session_renderer_is_used_when_available(config, ui):
    class RenderingSession(StubSession):
        def render_page_model(self, model, *, budget_tokens, mode="full", previous=None):
            return f"RENDERED-BY-SESSION mode={mode} generation={model.generation}"

    session = RenderingSession(page=PAGE_A)
    executor = StubExecutor([ToolResult(ok=True, summary="clicked", page=PAGE_B, page_changed=True)])
    agent, llm, _ = make_agent(
        config, ui, [click(1, "open"), finish(True, "done")], executor=executor, session=session, renderer=None
    )
    agent.run("click it")
    assert "RENDERED-BY-SESSION mode=full" in request_text(llm, 0)
    assert "RENDERED-BY-SESSION mode=delta" in request_text(llm, 1)


def test_a_failing_renderer_falls_back_instead_of_crashing(config, ui):
    def broken_renderer(model, *, budget_tokens, mode="full", previous=None):
        raise TypeError("wrong signature")

    executor = StubExecutor([ToolResult(ok=True, summary="clicked", page=PAGE_B, page_changed=True)])
    agent, llm, _ = make_agent(
        config, ui, [click(1, "open"), finish(True, "done")], executor=executor, renderer=broken_renderer
    )
    result = agent.run("click it")
    assert result.success is True
    assert "[2] textbox 'Query'" in request_text(llm, 1)
    assert any(event.kind == "error" and "renderer failed" in event.message for event in ui.events)


def test_identical_pages_are_reported_as_unchanged(config, ui):
    executor = StubExecutor([ToolResult(ok=True, summary="clicked", page=PAGE_A, page_changed=False)])
    agent, llm, _ = make_agent(config, ui, [click(1, "open"), finish(True, "done")], executor=executor)
    agent.run("click it")
    assert "unchanged since step 1" in request_text(llm, 1)


def test_page_outline_puts_the_page_into_the_conversation(make_config, ui):
    """The point of page_outline is that the model *sees* the page again."""
    config = make_config(max_tool_result_tokens=12, page_token_budget=600)
    page = PageModel(
        url="about:blank",
        title="Long page",
        generation=5,
        elements=[
            Element(id=i, role="button", name=f"Control {i}", tag="button", dom_hash=f"d{i}")
            for i in range(1, 26)
        ],
        text="lots of text " * 50,
    )
    session = StubSession(page=page)
    executor = StubExecutor()
    script = [
        LLMResponse(text="re-reading", tool_calls=[ToolCall("page_outline", {"mode": "full", "filter": ""}, id="p1")]),
        finish(True, "read it"),
    ]
    agent, llm, _ = make_agent(config, ui, script, executor=executor, session=session)
    result = agent.run("read the page again")
    conversation = request_text(llm, 1)
    assert "PAGE OUTLINE:" in conversation
    assert "[25] button 'Control 25'" in conversation
    assert "URL: about:blank" in conversation
    assert any(event.kind == "tool_result" for event in ui.events)
    assert result.success is True


# --------------------------------------------------------------------------- #
# Transcript, events, usage, context budget
# --------------------------------------------------------------------------- #


def test_the_transcript_is_jsonl_with_usage_per_step(config, ui):
    executor = StubExecutor([ToolResult(ok=True, summary="clicked", page=PAGE_B, page_changed=True)])
    agent, _, _ = make_agent(config, ui, [click(1, "open"), finish(True, "all good")], executor=executor)
    result = agent.run("click and finish")

    lines = transcript_lines(result)
    kinds = [line.get("kind") for line in lines if line["type"] == "event"]
    for kind in ("start", "tool_call", "tool_result", "finish", "context", "page"):
        assert kind in kinds, kind
    steps = [line for line in lines if line["type"] == "step"]
    assert len(steps) == 2
    assert steps[0]["tool"] == "click"
    assert steps[0]["arguments"] == {"element_id": 1, "intent": "open"}
    assert steps[0]["ok"] is True
    assert steps[0]["usage"]["prompt_tokens"] == 100
    assert steps[0]["usage"]["total"] == 110
    assert steps[1]["tool"] == "finish"
    final = [line for line in lines if line["type"] == "result"][-1]
    assert final["success"] is True
    assert final["answer"] == "all good"
    assert final["usage"]["total"] > 0
    assert final["usage_main"]["total"] > 0
    assert Path(result.transcript_path).parent == config.transcript_dir


def test_every_meaningful_moment_emits_an_event(config, ui):
    session = StubSession(page=PAGE_A)
    executor = StubExecutor([ToolResult(ok=False, summary="failed", error="overlay")] * 2)
    policy = StubPolicy(level="destructive", approve=False, reason="no")
    recovery = StubRecovery()
    extractor = StubExtractor()
    script = [
        LLMResponse(text="t", tool_calls=[ToolCall("ask_page", {"question": "q"}, id="a")]),
        LLMResponse(text="t", tool_calls=[ToolCall("click", {"element_id": 1, "intent": "x"}, id="c1")]),
        LLMResponse(text="t", tool_calls=[ToolCall("click", {"element_id": 1, "intent": "x"}, id="c2")]),
        finish(False, "blocked"),
    ]
    agent, _, _ = make_agent(
        config, ui, script, executor=executor, session=session, policy=policy, recovery=recovery, extractor=extractor
    )
    result = agent.run("do the impossible")
    kinds = set(ui.kinds())
    for kind in ("start", "thinking", "tool_call", "tool_result", "page", "risk", "confirm", "subagent", "context", "finish"):
        assert kind in kinds, kind
    assert result.success is False


def test_subagent_usage_is_attributed_separately(config, ui):
    extractor = StubExtractor()
    script = [
        LLMResponse(text="t", tool_calls=[ToolCall("ask_page", {"question": "q"}, id="a")]),
        finish(True, "done"),
    ]
    agent, _, _ = make_agent(config, ui, script, extractor=extractor)
    result = agent.run("read something")
    assert result.usage.subagents.prompt_tokens >= 300
    assert result.usage.main.prompt_tokens >= 100
    assert result.usage.total == result.usage.main.total + result.usage.subagents.total
    assert result.usage.calls >= 2


def test_context_budget_is_respected_over_a_long_run(make_config, ui):
    config = make_config(
        context_token_budget=2_500,
        history_window=4,
        max_steps=25,
        page_token_budget=400,
        page_token_budget_min=80,
        max_tool_result_tokens=50,
        loop_detection_repeats=99,
        max_attempts_per_action=99,
        max_consecutive_failures=99,
        compaction_trigger=0.7,
    )
    executor = StubExecutor(
        default=ToolResult(ok=True, summary="clicked a control " + "x" * 200, page=PAGE_B, page_changed=True)
    )
    agent, llm, _ = make_agent(config, ui, [click(1, "go")], executor=executor, llm=ScriptedLLM([click(1, "go")], repeat_last=True))
    result = agent.run("keep clicking until the budget says stop")
    assert result.steps == 25
    for request in llm.requests:
        content = sum(count_tokens(message.content) for message in request["messages"])
        assert content <= config.context_token_budget
    stats = agent.context.stats()
    assert stats["compactions"] > 0
    assert stats["prompt_tokens_estimate"] <= config.context_token_budget
    assert any(event.kind == "context" for event in ui.events)


def test_every_step_records_a_page_hash_and_a_risk_level(config, ui):
    executor = StubExecutor([ToolResult(ok=True, summary="clicked", page=PAGE_B, page_changed=True)])
    policy = StubPolicy(level="caution", approve=True)
    agent, _, _ = make_agent(config, ui, [click(1, "open"), finish(True, "ok")], executor=executor, policy=policy)
    result = agent.run("click it")
    steps = [line for line in transcript_lines(result) if line["type"] == "step"]
    assert steps and all(line["page_hash"] for line in steps)
    assert steps[0]["risk"] == "caution"
    assert steps[0]["confirmed"] is None, "only destructive actions are put to the human"


# --------------------------------------------------------------------------- #
# Model failures and defaults
# --------------------------------------------------------------------------- #


def test_a_broken_model_produces_an_honest_result(config, ui):
    llm = ScriptedLLM([], fail_times=5)
    agent, _, _ = make_agent(config, ui, None, llm=llm)
    result = agent.run("do something")
    assert result.success is False
    assert "model" in (result.error or "").lower()
    assert any(event.kind == "error" for event in ui.events)
    assert result.transcript_path and Path(result.transcript_path).exists()


def test_default_subagents_are_built_from_the_main_model(config, ui):
    agent, llm, _ = make_agent(config, ui, [finish(True, "done")])
    default_agent = Agent(config, llm, ui=ui)
    assert isinstance(default_agent.extractor, ExtractorSubAgent)
    assert isinstance(default_agent.recovery, RecoverySubAgent)
    assert isinstance(default_agent.summarizer, Summarizer)
    assert isinstance(default_agent.registry, ToolRegistry)
    assert default_agent.context is not None
    assert agent.registry is not None


def test_the_same_agent_can_run_twice(config, ui):
    executor = StubExecutor(
        [
            ToolResult(ok=True, summary="clicked once", page=PAGE_B, page_changed=True),
            ToolResult(ok=True, summary="clicked again", page=PAGE_A, page_changed=True),
        ]
    )
    llm = ScriptedLLM([click(1, "first"), finish(True, "first run done")], repeat_last=True)
    agent, _, _ = make_agent(config, ui, None, executor=executor, llm=llm)
    first = agent.run("first task")
    # fresh script for the second run: the loop must reset its state cleanly
    llm.script = [click(1, "second"), finish(True, "second run done")]
    second = agent.run("second task")

    assert first.success is True and first.steps == 2
    assert second.success is True and second.steps == 2
    assert second.answer == "second run done"
    assert first.transcript_path != second.transcript_path
    assert second.usage.calls == 2  # usage is not carried over from the first run
