"""Tests for ``webpilot.agent.subagents`` (offline).

Every sub-agent is exercised with its own tiny scripted LLM client (never the
provider package), including the ugly paths: fenced JSON, prose instead of JSON,
hallucinated tools, a crashing model and no model at all.
"""

from __future__ import annotations

from typing import Any, Sequence

from webpilot.agent.subagents import (
    MAX_ALTERNATIVES,
    ExtractorSubAgent,
    RecoverySubAgent,
    Summarizer,
    deterministic_summary,
    extract_json_object,
    parse_tool_calls,
)
from webpilot.agent.tools import TOOL_NAMES
from webpilot.tokenizer import count_tokens
from webpilot.types import (
    Element,
    FailureContext,
    LLMMessage,
    LLMResponse,
    PageModel,
    StepRecord,
    ToolCall,
    Usage,
)


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class ScriptedLLM:
    """Minimal LLMClient: canned responses, no network, records every request."""

    provider = "scripted"
    model = "scripted-1"

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

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
                "tools": list(tools or []),
                "system": system,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        item = self.script.pop(0) if self.script else LLMResponse(text="")
        return item

    def count_tokens(self, text: str) -> int:
        return count_tokens(text)


class ExplodingLLM:
    provider = "boom"
    model = "boom-1"

    def complete(self, *args, **kwargs):
        raise RuntimeError("provider exploded")

    def count_tokens(self, text: str) -> int:
        return count_tokens(text)


def make_step(index: int = 1, *, ok: bool = True, summary: str = "did something") -> StepRecord:
    return StepRecord(
        index=index,
        tool_call=ToolCall("click", {"element_id": index, "intent": "open"}, id=f"call_{index}"),
        result_summary=summary,
        ok=ok,
        page_url="about:blank",
        page_hash=f"h{index}",
    )


def make_page() -> PageModel:
    return PageModel(
        url="about:blank",
        title="A page",
        generation=2,
        elements=[Element(id=1, role="button", name="Primary", tag="button", dom_hash="d1")],
        text="Some readable text.",
    )


# --------------------------------------------------------------------------- #
# JSON handling
# --------------------------------------------------------------------------- #


def test_extract_json_object_handles_the_shapes_models_return():
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('Sure! Here it is:\n{"a": {"b": "}"}}\nHope that helps') == {
        "a": {"b": "}"}
    }
    assert extract_json_object("no json at all") is None
    assert extract_json_object("") is None


def test_parse_tool_calls_validates_names_and_arguments():
    calls = parse_tool_calls(
        [
            {"name": "click", "args": {"element_id": "3", "intent": "open"}},
            {"name": "teleport", "args": {}},
            {"name": "click", "args": {"element_id": 1}},          # missing intent
            {"name": "scroll", "args": '{"direction": "down", "amount": 300}'},
            {"tool": "notes", "arguments": {"text": "remember this"}},
            "go_back",
        ]
    )
    assert [call.name for call in calls] == ["click", "scroll", "notes"]
    assert calls[0].args == {"element_id": 3, "intent": "open"}
    assert calls[1].args == {"direction": "down", "amount": 300}
    # the list is capped at MAX_ALTERNATIVES, so the trailing string is not reached
    assert len(calls) == MAX_ALTERNATIVES
    assert parse_tool_calls(["go_back", {"name": "goto", "args": {"url": "about:blank"}}])[0].args == {}


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #


def test_extractor_returns_answer_evidence_confidence_and_usage(config):
    llm = ScriptedLLM(
        [
            LLMResponse(
                text='{"answer": "USD 42.50", "evidence": "Total: USD 42.50", "confidence": 0.82}',
                usage=Usage(prompt_tokens=500, completion_tokens=30, calls=1),
            )
        ]
    )
    extractor = ExtractorSubAgent(llm, config)
    result = extractor.answer("what is the total?", "Cart total: USD 42.50", "<html>Total: USD 42.50</html>")
    assert result.answer == "USD 42.50"
    assert result.evidence == "Total: USD 42.50"
    assert result.confidence == 0.82
    assert result.usage.prompt_tokens == 500
    assert result.subagent == "extractor"
    # its own message list: one user message, temperature 0, a system prompt, no tools
    request = llm.requests[0]
    assert len(request["messages"]) == 1 and request["messages"][0].role == "user"
    assert request["temperature"] == 0.0
    assert request["tools"] == []
    assert "question" in request["system"].lower() or "read one web page" in request["system"].lower()
    assert extractor.usage.prompt_tokens == 500
    assert extractor.calls == 1


def test_extractor_accepts_fenced_json_and_prose():
    fenced = ScriptedLLM(
        [LLMResponse(text='```json\n{"answer": "Tuesday", "evidence": "on Tuesday", "confidence": 0.5}\n```')]
    )
    assert ExtractorSubAgent(fenced, None).answer("when?", "delivered on Tuesday").answer == "Tuesday"

    prose = ScriptedLLM([LLMResponse(text="The total is 128.00 dollars.")])
    result = ExtractorSubAgent(prose, None).answer("total?", "Total 128.00")
    assert "128.00" in result.answer
    assert 0.0 <= result.confidence <= 1.0
    assert result.evidence == ""


def test_extractor_returns_empty_when_the_answer_is_absent():
    for text in (
        '{"answer": "", "evidence": "", "confidence": 0.0}',
        '{"answer": "not present", "evidence": "", "confidence": 0.9}',
        "I cannot find that information in the provided content.",
    ):
        llm = ScriptedLLM([LLMResponse(text=text)])
        result = ExtractorSubAgent(llm, None).answer("how many?", "nothing relevant here")
        assert result.answer == ""
        assert result.confidence == 0.0


def test_extractor_degrades_gracefully():
    broken = ExtractorSubAgent(ExplodingLLM(), None)
    result = broken.answer("what is the price?", "some text")
    assert result.answer == ""
    assert result.confidence == 0.0
    assert broken.failures == 1

    without_model = ExtractorSubAgent(None, None)
    result = without_model.answer("what is the price?", "some text")
    assert result.answer == ""
    assert without_model.failures == 1

    empty_question = ExtractorSubAgent(ScriptedLLM([]), None)
    assert empty_question.answer("   ", "text").answer == ""
    assert empty_question.calls == 0


def test_extractor_keeps_its_content_within_the_extractor_budget(make_config):
    config = make_config(extractor_token_budget=300, subagent_max_output_tokens=64)
    llm = ScriptedLLM([LLMResponse(text='{"answer": "x", "evidence": "", "confidence": 0.1}')])
    extractor = ExtractorSubAgent(llm, config)
    extractor.answer("q", "page text " * 5_000, "<div>" * 5_000)
    request = llm.requests[0]
    assert count_tokens(request["messages"][0].content) <= config.extractor_token_budget + 80
    assert request["max_tokens"] == 64


# --------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------- #


def test_recovery_parses_advice_alternatives_and_give_up(config):
    llm = ScriptedLLM(
        [
            LLMResponse(
                text=(
                    '{"advice": "the overlay is swallowing the click; dismiss it first",'
                    ' "alternative_actions": ['
                    '{"name": "press_key", "args": {"key": "Escape"}},'
                    '{"name": "click", "args": {"element_id": 9, "intent": "dismiss"}}'
                    "],"
                    ' "give_up": false}'
                ),
                usage=Usage(prompt_tokens=800, completion_tokens=60, calls=1),
            )
        ]
    )
    recovery = RecoverySubAgent(llm, config)
    failure = FailureContext(
        task="order the cheapest item",
        tool_call=ToolCall("click", {"element_id": 4, "intent": "open"}),
        error="overlay intercepts clicks",
        attempts=3,
        page=make_page(),
        history=[make_step(1).digest(), make_step(2, ok=False, summary="click failed").digest()],
    )
    proposal = recovery.propose(failure)
    assert "overlay" in proposal.advice
    assert [call.name for call in proposal.alternative_actions] == ["press_key", "click"]
    assert proposal.give_up is False
    assert proposal.usage.prompt_tokens == 800
    assert proposal.subagent == "recovery"
    assert recovery.usage.prompt_tokens == 800
    # the prompt shows the task, the failing call, the error and the page
    request = llm.requests[0]["messages"][0].content
    assert "order the cheapest item" in request
    assert "click(element_id=4, intent='open')" in request
    assert "overlay intercepts clicks" in request
    assert "[1] button 'Primary'" in request
    assert "AVAILABLE TOOLS" in request
    for name in TOOL_NAMES:
        assert name in request


def test_recovery_drops_invalid_alternatives_and_caps_them(config):
    llm = ScriptedLLM(
        [
            LLMResponse(
                text=(
                    '{"advice": "try again differently", "alternative_actions": ['
                    '{"name": "teleport", "args": {}},'
                    '{"name": "scroll", "args": {"direction": "down", "amount": 0}},'
                    '{"name": "scroll", "args": {"direction": "wrong", "amount": 0}},'
                    '{"name": "press_key", "args": {"key": "Escape"}},'
                    '{"name": "notes", "args": {"text": "remember"}},'
                    '{"name": "go_back", "args": {}}'
                    "]}"
                )
            )
        ]
    )
    proposal = RecoverySubAgent(llm, config).propose(
        FailureContext(task="t", tool_call=ToolCall("click", {"element_id": 1, "intent": "x"}), error="e", attempts=2)
    )
    assert len(proposal.alternative_actions) == MAX_ALTERNATIVES
    assert [call.name for call in proposal.alternative_actions] == ["scroll", "press_key", "notes"]


def test_recovery_handles_prose_and_broken_models(config):
    prose = RecoverySubAgent(ScriptedLLM([LLMResponse(text="Just scroll down a bit.")]), config)
    proposal = prose.propose(
        FailureContext(task="t", tool_call=ToolCall("click", {"element_id": 1, "intent": "x"}), error="e", attempts=2)
    )
    assert "scroll" in proposal.advice
    assert proposal.alternative_actions == []

    broken = RecoverySubAgent(ExplodingLLM(), config)
    proposal = broken.propose(
        FailureContext(task="t", tool_call=ToolCall("click", {"element_id": 2, "intent": "x"}), error="nothing changed", attempts=4)
    )
    assert proposal.advice
    assert "nothing changed" in proposal.advice
    assert proposal.alternative_actions == []
    assert proposal.give_up is False
    assert broken.failures == 1

    without_model = RecoverySubAgent(None, config)
    proposal = without_model.propose(
        FailureContext(task="t", tool_call=ToolCall("goto", {"url": "about:blank"}), error="timeout", attempts=3)
    )
    assert "goto" in proposal.advice and proposal.usage.calls == 0


def test_recovery_give_up_is_honoured(config):
    llm = ScriptedLLM(
        [LLMResponse(text='{"advice": "nothing left to try", "alternative_actions": [], "give_up": true}')]
    )
    proposal = RecoverySubAgent(llm, config).propose(
        FailureContext(task="t", tool_call=ToolCall("scroll", {"direction": "down", "amount": 0}), error="stuck", attempts=5)
    )
    assert proposal.give_up is True


# --------------------------------------------------------------------------- #
# Summarizer
# --------------------------------------------------------------------------- #


def test_summarizer_uses_the_model_and_records_usage(config):
    llm = ScriptedLLM(
        [LLMResponse(text="- did A\n- found 12\n- next: try B", usage=Usage(prompt_tokens=900, completion_tokens=40, calls=1))]
    )
    summarizer = Summarizer(llm, config)
    text = summarizer.summarize("the task", [make_step(1), make_step(2)], "earlier summary")
    assert text.startswith("- did A")
    assert summarizer.usage.prompt_tokens == 900
    request = llm.requests[0]
    assert "the task" in request["messages"][0].content
    assert "earlier summary" in request["messages"][0].content
    assert request["temperature"] == 0.0


def test_summarizer_falls_back_deterministically(config):
    steps = [make_step(1, summary="first thing"), make_step(2, ok=False, summary="second thing")]
    without_model = Summarizer(None, config)
    text = without_model.summarize("the task", steps, "")
    assert "first thing" in text and "second thing" in text
    assert text.count("- ") >= 2

    broken = Summarizer(ExplodingLLM(), config)
    fallback = broken.summarize("the task", steps, "previous")
    assert "previous" in fallback
    assert broken.failures == 1

    empty = Summarizer(ScriptedLLM([LLMResponse(text="   ")]), config)
    assert "first thing" in empty.summarize("the task", steps, "")


def test_deterministic_summary_keeps_the_previous_summary():
    text = deterministic_summary([make_step(3, summary="third")], "OLDER FACTS")
    assert text.startswith("OLDER FACTS")
    assert "third" in text
    assert deterministic_summary([], "nothing new") == "nothing new"


def test_summarizer_failure_never_raises(config):
    class NastyLLM:
        provider = "nasty"
        model = "nasty"

        def complete(self, *args, **kwargs):
            raise KeyboardInterrupt  # not an Exception subclass

        def count_tokens(self, text: str) -> int:
            return count_tokens(text)

    # KeyboardInterrupt is intentionally *not* swallowed: only provider errors are,
    # and the loop above that calls a summarizer has its own guard.
    try:
        Summarizer(NastyLLM(), config).summarize("t", [make_step(1)], "")
    except KeyboardInterrupt:
        pass
