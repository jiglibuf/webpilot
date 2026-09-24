"""Tests for ``webpilot.agent.memory.ContextManager`` (offline).

The context manager is what keeps a long run inside the model's window, so the
tests here are mostly invariant tests: the budget is never exceeded (200 synthetic
steps), the current page is never lost, the scratchpad survives compaction and the
message list stays a structurally valid conversation (no orphan tool messages).
"""

from __future__ import annotations

from typing import Any

from webpilot.agent.memory import NOTES_TOKEN_CAP, ContextManager
from webpilot.agent.tools import fallback_render_page
from webpilot.tokenizer import count_tokens
from webpilot.types import Element, LLMMessage, PageModel, StepRecord, ToolCall, Usage


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def make_step(
    index: int,
    *,
    name: str = "click",
    args: dict[str, Any] | None = None,
    ok: bool = True,
    summary: str = "did something useful",
    page_hash: str = "",
    url: str = "about:blank",
) -> StepRecord:
    call = ToolCall(
        name=name,
        args=args if args is not None else {"element_id": index, "intent": "open"},
        id=f"call_{index}",
    )
    return StepRecord(
        index=index,
        tool_call=call,
        result_summary=summary,
        ok=ok,
        page_url=url,
        page_title="A page",
        page_hash=page_hash,
    )


def make_page(generation: int = 1, *, elements: int = 3, marker: str = "") -> PageModel:
    return PageModel(
        url="about:blank",
        title="A page",
        generation=generation,
        elements=[
            Element(id=i, role="button", name=f"Button {i}{marker}", tag="button", dom_hash=f"d{i}{marker}")
            for i in range(1, elements + 1)
        ],
        text="readable text " * 40,
    )


def contents(messages: list[LLMMessage]) -> str:
    return "\n".join(message.content for message in messages)


# --------------------------------------------------------------------------- #
# Stats and budgets
# --------------------------------------------------------------------------- #


def test_stats_exposes_the_documented_keys(config):
    manager = ContextManager(config)
    stats = manager.stats()
    for key in (
        "steps",
        "prompt_tokens_estimate",
        "page_tokens",
        "history_tokens",
        "compactions",
        "dropped_items",
    ):
        assert key in stats, key
    assert stats["steps"] == 0
    assert manager.step_count() == 0


def test_page_budget_shrinks_as_the_history_grows(make_config):
    manager = ContextManager(make_config(page_token_budget=2_000, page_token_budget_min=300))
    assert manager.page_budget() == 2_000
    budgets = []
    for index in range(1, 41):
        page = make_page(index)
        manager.record(
            make_step(index, page_hash=page.page_hash()),
            tool_text="x" * 400,
            assistant_text="thinking " * 20,
            page_block=fallback_render_page(page, budget_tokens=500),
        )
        budgets.append(manager.page_budget())
    assert budgets[-1] < budgets[0]
    assert budgets[-1] >= 300
    assert all(300 <= value <= 2_000 for value in budgets)
    assert budgets == sorted(budgets, reverse=True)


def test_page_budget_never_drops_below_the_base_when_the_min_is_larger(make_config):
    manager = ContextManager(make_config(page_token_budget=200, page_token_budget_min=900))
    assert manager.page_budget() == 200


# --------------------------------------------------------------------------- #
# Sliding window, folding, summary
# --------------------------------------------------------------------------- #


def test_window_keeps_the_last_steps_verbatim_and_folds_older_ones(make_config):
    config = make_config(history_window=4, context_token_budget=60_000, compaction_trigger=0.9)
    manager = ContextManager(config)
    for index in range(1, 21):
        page = make_page(index, marker=str(index))
        manager.record(
            make_step(index, summary=f"step number {index}", page_hash=page.page_hash()),
            tool_text=f"RESULT-{index}-END",
            assistant_text=f"REASONING-{index}-END",
            page_block=f"URL: about:blank | Title: A page | generation {index}",
        )
    page = make_page(21, marker="21")
    messages = manager.build_messages(
        task="do the thing", page=page, page_block=fallback_render_page(page, budget_tokens=300)
    )
    text = contents(messages)
    assert manager.step_count() == 20
    stats = manager.stats()
    assert stats["window_steps"] <= 4
    assert stats["compactions"] >= 1
    assert "RESULT-20-END" in text
    assert "RESULT-1-END" not in text
    assert manager.summary.strip()
    assert any("compressed" in message.content for message in messages)


def test_deterministic_summary_when_no_summarizer_is_configured(make_config):
    config = make_config(history_window=2)
    manager = ContextManager(config, summarizer=None)
    for index in range(1, 7):
        page = make_page(index, marker=str(index))
        manager.record(make_step(index, summary=f"unique-{index}", page_hash=page.page_hash()), tool_text="t")
    manager.build_messages(task="t", page=make_page(7, marker="7"), page_block="block")
    assert "unique-1" in manager.summary
    assert manager.stats()["compactions"] >= 1
    assert manager.stats()["summarizer"] is False


def test_summarizer_is_used_and_its_usage_is_reported(make_config):
    class StubSummarizer:
        def __init__(self):
            self.calls: list[tuple[str, int, str]] = []
            self.usage = Usage(prompt_tokens=100, completion_tokens=20, calls=1)

        def summarize(self, task, steps, previous_summary=""):
            self.calls.append((task, len(list(steps)), previous_summary))
            return "SUMMARY: " + "; ".join(step.digest() for step in steps)[:200]

    summarizer = StubSummarizer()
    manager = ContextManager(make_config(history_window=2), summarizer=summarizer)
    for index in range(1, 7):
        page = make_page(index, marker=str(index))
        manager.record(make_step(index, page_hash=page.page_hash()), tool_text="t")
    manager.build_messages(task="the task", page=make_page(7, marker="7"), page_block="block")
    assert summarizer.calls
    assert summarizer.calls[0][0] == "the task"
    assert manager.summary.startswith("SUMMARY:")
    assert manager.summary_usage.prompt_tokens >= 100


def test_a_broken_summarizer_falls_back_to_bullets(make_config):
    class Boom:
        usage = Usage()

        def summarize(self, *args, **kwargs):
            raise RuntimeError("no")

    manager = ContextManager(make_config(history_window=2), summarizer=Boom())
    for index in range(1, 5):
        page = make_page(index, marker=str(index))
        manager.record(make_step(index, summary=f"digest-{index}", page_hash=page.page_hash()), tool_text="t")
    manager.build_messages(task="t", page=make_page(5, marker="5"), page_block="block")
    assert "digest-1" in manager.summary


# --------------------------------------------------------------------------- #
# Truncation, duplicates, warnings, scratchpad
# --------------------------------------------------------------------------- #


def test_tool_results_are_truncated_to_the_limit(make_config):
    config = make_config(max_tool_result_tokens=50)
    manager = ContextManager(config)
    page = make_page(1)
    manager.record(
        make_step(1, page_hash=page.page_hash()),
        tool_text="ok " * 5_000,
        assistant_text="reasoning " * 5_000,
        page_block="block",
    )
    messages = manager.build_messages(task="t", page=page, page_block="block")
    tools_msgs = [message for message in messages if message.role == "tool"]
    assert tools_msgs
    assert count_tokens(tools_msgs[0].content) <= 60
    assistants = [message for message in messages if message.role == "assistant"]
    assert assistants and count_tokens(assistants[0].content) <= 60


def test_duplicate_page_hash_renders_as_unchanged(make_config):
    manager = ContextManager(make_config())
    page = make_page(1)
    block = fallback_render_page(page, budget_tokens=400)
    manager.record(make_step(1, page_hash=page.page_hash()), tool_text="clicked", page_block=block)
    messages = manager.build_messages(task="t", page=page, page_block=block)
    text = contents(messages)
    assert "unchanged since step 1" in text
    assert manager.stats()["dropped_items"] >= 1
    # the same hash twice in a row is counted as a duplicate page
    manager.record(make_step(2, page_hash=page.page_hash(), name="press_key", args={"key": "Escape"}), tool_text="esc")
    assert manager.stats()["duplicate_pages"] == 1


def test_current_page_is_never_lost_when_its_block_leaves_the_window(make_config):
    config = make_config(history_window=1)
    manager = ContextManager(config)
    for index in range(1, 6):
        page = make_page(index, marker=str(index))
        manager.record(
            make_step(index, page_hash=f"stale-{index}"),
            tool_text="t",
            page_block=f"OLD BLOCK {index}",
        )
    page = make_page(9, marker="9")
    fresh = fallback_render_page(page, budget_tokens=400)
    messages = manager.build_messages(task="t", page=page, page_block=fresh)
    text = contents(messages)
    assert "Button 1" in text  # element ids of the *current* page are present
    assert fresh.splitlines()[0] in text


def test_page_provider_is_used_when_no_block_is_visible(make_config):
    manager = ContextManager(make_config(history_window=0))
    manager.set_page_provider(lambda page, mode: f"RE-RENDERED-{mode}")
    manager.record(make_step(1, page_hash="x"), tool_text="t", page_block="OLD")
    messages = manager.build_messages(task="t", page=make_page(2), page_block="")
    assert "RE-RENDERED-full" in contents(messages)


def test_repeated_identical_tool_call_injects_a_warning(make_config):
    manager = ContextManager(make_config())
    page = make_page(1)
    args = {"element_id": 4, "intent": "open"}
    for index in (1, 2):
        manager.record(make_step(index, args=args, ok=False, summary="nothing happened", page_hash=page.page_hash()), tool_text="FAILED")
    messages = manager.build_messages(task="t", page=page, page_block="block")
    text = contents(messages)
    assert "repeated the same action" in text
    assert manager.stats()["warnings_injected"] >= 1
    # the warning is not repeated on the next turn unless it happens again
    again = manager.build_messages(task="t", page=page, page_block="block")
    assert "repeated the same action" not in contents(again)


def test_scratchpad_is_always_present_and_survives_compaction(make_config):
    config = make_config(history_window=2)
    manager = ContextManager(config)
    manager.update_notes("the code was 1234")
    for index in range(1, 11):
        page = make_page(index, marker=str(index))
        manager.record(make_step(index, page_hash=page.page_hash()), tool_text="t")
    manager.build_messages(task="t", page=make_page(11, marker="11"), page_block="block")
    messages = manager.build_messages(task="t", page=make_page(12, marker="12"), page_block="block")
    text = contents(messages)
    assert "SCRATCHPAD" in text
    assert "the code was 1234" in text
    assert manager.stats()["compactions"] >= 1
    empty = ContextManager(config)
    assert "SCRATCHPAD" in contents(empty.build_messages(task="t"))
    assert "(empty)" in contents(empty.build_messages(task="t"))


def test_scratchpad_appends_caps_and_clears(make_config):
    manager = ContextManager(make_config())
    manager.update_notes("first")
    manager.update_notes("second")
    assert manager.notes.splitlines() == ["- first", "- second"]
    for index in range(200):
        manager.update_notes(f"entry {index} " + "y" * 40)
    assert count_tokens(manager.notes) <= NOTES_TOKEN_CAP
    assert manager.notes.startswith("- entry ")
    manager.update_notes("clear")
    assert manager.notes == ""


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def test_task_page_pending_and_extra_messages_are_included(config):
    manager = ContextManager(config)
    page = make_page(3)
    extra = [LLMMessage(role="user", content="EXTRA-MESSAGE")]
    messages = manager.build_messages(
        task="find the total",
        page=page,
        page_block="BLOCK",
        pending=["HUMAN: hurry up"],
        extra=extra,
    )
    assert messages[0].role == "user" and "find the total" in messages[0].content
    text = contents(messages)
    assert "HUMAN: hurry up" in text
    assert "EXTRA-MESSAGE" in text
    assert "BLOCK" in text
    assert messages[-1].content == "EXTRA-MESSAGE"
    # a pending item is not sticky: the next build without it does not repeat it
    assert "HUMAN: hurry up" not in contents(
        manager.build_messages(task="find the total", page=page, page_block="BLOCK")
    )


def test_no_page_observation_is_stated_explicitly(config):
    manager = ContextManager(config)
    messages = manager.build_messages(task="t", page=None, page_block="")
    assert "none available" in contents(messages)


def test_window_messages_are_structurally_valid(config):
    manager = ContextManager(config)
    page = make_page(1)
    manager.record(make_step(1, page_hash=page.page_hash()), tool_text="ok", page_block="block")
    messages = manager.build_messages(task="t", page=page, page_block="block")
    seen: set[str] = set()
    for message in messages:
        if message.role == "assistant":
            seen.update(call.id for call in message.tool_calls if call.id)
        elif message.role == "tool":
            assert message.tool_call_id in seen
            assert message.name


# --------------------------------------------------------------------------- #
# The invariant that matters
# --------------------------------------------------------------------------- #


def test_stress_two_hundred_steps_never_exceed_the_budget(make_config):
    config = make_config(
        context_token_budget=3_000,
        history_window=6,
        page_token_budget=500,
        page_token_budget_min=100,
        max_tool_result_tokens=60,
        compaction_trigger=0.7,
    )
    manager = ContextManager(config)
    notes_seen = False
    for index in range(1, 201):
        page = make_page(index % 7 + 1, marker=str(index % 7))
        block = fallback_render_page(page, budget_tokens=manager.page_budget())
        manager.record(
            make_step(index, page_hash=page.page_hash(), summary="s" * 200),
            tool_text="result " + "y" * 400,
            assistant_text="thinking " + "z" * 300,
            page_block=block,
        )
        manager.update_notes(f"note {index}")
        messages = manager.build_messages(
            task="stress the context manager",
            page=page,
            page_block=block,
            pending=[f"pending warning {index} " + "p" * 100],
        )
        content_only = sum(count_tokens(message.content) for message in messages)
        assert content_only <= config.context_token_budget
        assert manager.stats()["prompt_tokens_estimate"] <= config.context_token_budget
        assert messages and messages[0].content
        seen: set[str] = set()
        for message in messages:
            if message.role == "assistant":
                seen.update(call.id for call in message.tool_calls if call.id)
            elif message.role == "tool":
                assert message.tool_call_id in seen
        if "SCRATCHPAD" in contents(messages):
            notes_seen = True
    stats = manager.stats()
    assert stats["steps"] == 200
    assert stats["compactions"] > 1
    assert stats["dropped_items"] > 0
    assert notes_seen
