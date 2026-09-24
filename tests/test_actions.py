"""ActionExecutor tests: every tool against the bundled local store.

The point of these tests is not "the click happened" but "the executor knew
whether it happened": each one asserts on the verification verdict
(``ok``/``page_changed``/``recovery_hint``), which is what keeps the agent loop
from looping on silent no-ops.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from webpilot.browser.actions import (
    AGENT_LOCAL_TOOLS,
    WEBPILOT_ATTR,
    ActionExecutor,
    selector_for,
)
from webpilot.types import ToolCall


@pytest.fixture
def executor(browser_session, config) -> ActionExecutor:
    return ActionExecutor(browser_session, config)


def element_named(model, name: str, role: str | None = None):
    for element in model.elements:
        if name.lower() in element.name.lower() and (role is None or element.role == role):
            return element
    return None


def shadow_contains(browser_session, needle: str) -> bool:
    """Test-side oracle: is ``needle`` visible inside any open shadow root?"""
    return bool(browser_session.page.evaluate(
        """(needle) => {
            let found = false;
            document.querySelectorAll("*").forEach((el) => {
              if (el.shadowRoot && (el.shadowRoot.textContent || "").includes(needle)) found = true;
            });
            return found;
        }""",
        needle,
    ))


# --------------------------------------------------------------------------- #
# addressing (the anti-hardcoding invariant)
# --------------------------------------------------------------------------- #

def test_element_selectors_are_generated_from_the_id_only():
    assert selector_for(7) == '[data-webpilot-id="7"]'
    assert selector_for(0) == '[data-webpilot-id="0"]'
    assert WEBPILOT_ATTR == "data-webpilot-id"


def test_browser_layer_has_no_hardcoded_selectors_or_site_paths():
    """The assignment's hard rule, pinned as a test.

    Forbidden anywhere under ``src/webpilot/browser``:
      * a quoted id selector (``"#submit"``) or class selector (``".btn-primary"``);
      * a ``data-*`` attribute selector other than webpilot's own id attribute;
      * an attribute selector with a literal value (``[name=q]``);
      * an absolute URL pointing at a real site (``https://shop.example/cart``).
    Allowed: the generated ``[data-webpilot-id=...]`` form, tag/role/aria
    selectors, and standard pseudo-classes.
    """
    root = Path(__file__).resolve().parents[1] / "src" / "webpilot" / "browser"
    sources = sorted([*root.glob("*.py"), *root.glob("*.js")])
    assert sources, "no browser sources found"

    forbidden = {
        # a selector that starts with an id / class ("#submit", ".btn-primary");
        # the lookahead keeps harmless separators like "; ".join(...) out of it
        "quoted id selector": re.compile(r"""["']\s*#[A-Za-z][\w-]*["'](?!\s*\.)"""),
        "quoted class selector": re.compile(r"""["']\s*\.[A-Za-z][\w-]*["'](?!\s*\.)"""),
        "foreign data attribute": re.compile(r"\[\s*data-(?!webpilot-id)"),
        "literal attribute selector": re.compile(
            r"\[\s*(?:name|id|href|placeholder|class|type|value)\s*=\s*[A-Za-z0-9]"
        ),
        "hard-coded site url": re.compile(r"https?://(?!127\.0\.0\.1|localhost)[A-Za-z0-9.-]+/"),
    }
    problems = []
    for path in sources:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for label, pattern in forbidden.items():
                if pattern.search(line):
                    problems.append(f"{path.name}:{number} {label}: {line.strip()[:100]}")
    assert problems == [], "\n".join(problems)


def test_a_stale_element_id_is_refused_with_a_hint(executor, browser_session, fixture_site):
    browser_session.goto(fixture_site)
    model = browser_session.snapshot(budget_tokens=1200)
    home_link = element_named(model, "Home", role="link")

    browser_session.goto(f"{fixture_site}/cart")
    browser_session.snapshot(budget_tokens=800)
    result = executor.execute(ToolCall("click", {"element_id": home_link.id}))

    assert result.ok is False
    assert "not part of the current page snapshot" in result.summary
    assert "stale" in (result.recovery_hint or "")
    assert "page_outline" in (result.recovery_hint or "")
    assert result.error


def test_missing_or_malformed_arguments_never_raise(executor):
    for call in (
        ToolCall("click", {}),
        ToolCall("click", {"element_id": "abc"}),
        ToolCall("click", {"element_id": -4}),
        ToolCall("goto", {}),
        ToolCall("type_text", {"element_id": 1}),
        ToolCall("press_key", {}),
        ToolCall("scroll", {"direction": "sideways"}),
        ToolCall("tabs", {"action": "explode"}),
    ):
        result = executor.execute(call)
        assert result.ok is False
        assert result.summary
        assert result.recovery_hint


def test_agent_local_tools_are_rejected_not_executed(executor):
    for name in sorted(AGENT_LOCAL_TOOLS):
        result = executor.execute(ToolCall(name, {"text": "x"}))
        assert result.ok is False
        assert "agent loop" in result.summary
        assert result.page is None

    unknown = executor.execute(ToolCall("teleport", {}))
    assert unknown.ok is False and "unknown tool" in unknown.summary


# --------------------------------------------------------------------------- #
# click
# --------------------------------------------------------------------------- #

def test_click_follows_a_link_and_reports_the_change(executor, browser_session, fixture_site):
    browser_session.goto(fixture_site)
    model = browser_session.snapshot(budget_tokens=1200)
    link = element_named(model, "Search", role="link")

    result = executor.execute(ToolCall("click", {"element_id": link.id, "intent": "open search"}))

    assert result.ok is True
    assert result.page_changed is True
    assert result.page is not None
    assert result.page.url.endswith("/search")
    assert "Search" in result.summary


def test_click_that_changes_nothing_is_reported_as_a_failure(executor, browser_session, fixture_site):
    browser_session.goto(f"{fixture_site}/modal")
    model = browser_session.snapshot(budget_tokens=1500)
    opener = element_named(model, "Open newsletter dialog", role="button")
    assert executor.execute(ToolCall("click", {"element_id": opener.id})).ok is True

    model = browser_session.snapshot(budget_tokens=1500)
    subscribe = element_named(model, "Subscribe", role="button")
    result = executor.execute(ToolCall("click", {"element_id": subscribe.id}))

    assert result.ok is False
    assert result.page_changed is False
    assert "did not change" in result.summary
    assert result.recovery_hint and "wait_for" in result.recovery_hint


def test_click_on_a_disabled_control_is_refused(executor, browser_session, fixture_site):
    browser_session.goto(f"{fixture_site}/checkout")
    model = browser_session.snapshot(budget_tokens=1500)
    submit = element_named(model, "Place order", role="submit")

    # disable it the way a page would, then let the sniffer see it
    browser_session.page.locator(selector_for(submit.id)).first.evaluate(
        "(el) => { el.disabled = true; }"
    )
    browser_session.snapshot(budget_tokens=1500)

    result = executor.execute(ToolCall("click", {"element_id": submit.id}))

    assert result.ok is False
    assert "disabled" in result.summary.lower()
    assert "disabled" in (result.recovery_hint or "").lower()


def test_overlay_interception_is_recovered_and_the_click_is_retried(
    executor, browser_session, fixture_site
):
    browser_session.goto(f"{fixture_site}/modal")
    model = browser_session.snapshot(budget_tokens=1500)
    opener = element_named(model, "Open newsletter dialog", role="button")
    executor.execute(ToolCall("click", {"element_id": opener.id}))

    model = browser_session.snapshot(budget_tokens=1500)
    home = element_named(model, "Home", role="link")
    result = executor.execute(ToolCall("click", {"element_id": home.id}))

    assert result.ok is True, result.summary
    assert result.page_changed is True
    assert result.data["recovered"], "the overlay must have been dismissed first"
    assert browser_session.last_model.url.rstrip("/") == fixture_site.rstrip("/")


# --------------------------------------------------------------------------- #
# type_text
# --------------------------------------------------------------------------- #

def test_type_text_fills_a_field_and_submits(executor, browser_session, fixture_site):
    browser_session.goto(f"{fixture_site}/search")
    model = browser_session.snapshot(budget_tokens=1200)
    box = element_named(model, "Search products", role="textbox")

    result = executor.execute(ToolCall("type_text", {
        "element_id": box.id, "text": "desk", "submit": True, "clear": True,
    }))

    assert result.ok is True
    assert result.page_changed is True
    assert "q=desk" in browser_session.last_model.url
    assert "Results for desk" in browser_session.last_model.text


def test_type_text_without_submit_keeps_the_value_and_says_so(
    executor, browser_session, fixture_site
):
    browser_session.goto(f"{fixture_site}/search")
    model = browser_session.snapshot(budget_tokens=1200)
    box = element_named(model, "Search products", role="textbox")

    result = executor.execute(ToolCall("type_text", {
        "element_id": box.id, "text": "lamp", "submit": False, "clear": True,
    }))

    assert result.ok is True
    assert "typed into" in result.summary
    assert browser_session.page.locator(selector_for(box.id)).first.input_value() == "lamp"


def test_typing_into_a_password_field_is_refused(executor, browser_session, fixture_site):
    browser_session.goto(f"{fixture_site}/login")
    model = browser_session.snapshot(budget_tokens=1200)
    password = next(el for el in model.elements if el.type == "password")

    result = executor.execute(ToolCall("type_text", {
        "element_id": password.id, "text": "s3cr3t-value", "submit": True, "clear": True,
    }))

    assert result.ok is False
    assert result.recovery_hint == "password fields are filled by the human"
    assert "password" in result.summary.lower()
    value = browser_session.page.locator(selector_for(password.id)).first.input_value()
    assert value == "", "nothing may be typed into a password field"


# --------------------------------------------------------------------------- #
# shadow DOM and iframes
# --------------------------------------------------------------------------- #

def test_shadow_dom_element_is_clickable(executor, browser_session, fixture_site):
    browser_session.goto(f"{fixture_site}/shadow")
    model = browser_session.snapshot(budget_tokens=1500)
    button = element_named(model, "Shadow action", role="button")
    assert button is not None, [el.line() for el in model.elements]

    result = executor.execute(ToolCall("click", {"element_id": button.id}))

    assert result.ok is True, result.summary
    assert shadow_contains(browser_session, "shadow clicked")


def test_iframe_element_is_clickable(executor, browser_session, fixture_site):
    browser_session.goto(f"{fixture_site}/iframe")
    model = browser_session.snapshot(budget_tokens=2000)
    note = element_named(model, "Note", role="textbox")
    assert note is not None and note.frame == 1

    typed = executor.execute(ToolCall("type_text", {
        "element_id": note.id, "text": "hello from outside", "submit": False, "clear": True,
    }))
    assert typed.ok is True, typed.summary

    model = browser_session.snapshot(budget_tokens=2000)
    save = element_named(model, "Save note", role="submit")
    result = executor.execute(ToolCall("click", {"element_id": save.id}))

    assert result.ok is True, result.summary
    assert result.page_changed is True
    assert "note=hello+from+outside" in browser_session.page.frames[1].url \
        or "note=hello%20from%20outside" in browser_session.page.frames[1].url


# --------------------------------------------------------------------------- #
# keys, scroll, history, waits
# --------------------------------------------------------------------------- #

def test_press_key_reports_change_and_no_change(executor, browser_session, fixture_site):
    browser_session.goto(f"{fixture_site}/search?q=desk")
    # a small viewport makes "the page can scroll" a property of the test, not of
    # the fixture's page length
    browser_session.page.set_viewport_size({"width": 800, "height": 300})
    model = browser_session.snapshot(budget_tokens=1500)
    assert model.scroll.get("max_y", 0) > 0

    moved = executor.execute(ToolCall("press_key", {"key": "PageDown"}))
    dead = executor.execute(ToolCall("press_key", {"key": "Escape"}))

    assert moved.ok is True, moved.summary
    assert moved.page_changed is True
    assert dead.ok is False
    assert "did not react" in dead.summary
    assert dead.recovery_hint


def test_scroll_moves_the_page_and_never_lies(executor, browser_session, fixture_site):
    browser_session.goto(f"{fixture_site}/search?q=desk")
    browser_session.page.set_viewport_size({"width": 800, "height": 300})
    model = browser_session.snapshot(budget_tokens=1500)
    assert model.scroll.get("max_y", 0) > 0

    down = executor.execute(ToolCall("scroll", {"direction": "down", "amount": 60}))
    assert down.ok is True, down.summary
    assert down.page_changed is True

    top = executor.execute(ToolCall("scroll", {"direction": "top", "amount": 60}))
    assert top.ok is True
    assert browser_session.snapshot(budget_tokens=800).scroll["y"] == 0

    up = executor.execute(ToolCall("scroll", {"direction": "up", "amount": 60}))
    assert up.ok is False, "scrolling up from the top must be reported, not faked"
    assert up.recovery_hint


def test_go_back_returns_or_explains(executor, browser_session, fixture_site):
    browser_session.goto(fixture_site)
    model = browser_session.snapshot(budget_tokens=1200)
    link = element_named(model, "Cart", role="link")
    executor.execute(ToolCall("click", {"element_id": link.id}))
    assert browser_session.last_model.url.endswith("/cart")

    back = executor.execute(ToolCall("go_back", {}))
    assert back.ok is True
    assert back.page_changed is True
    assert browser_session.last_model.url.rstrip("/") == fixture_site.rstrip("/")

    # walk back until the tab has no history left -> honest failure with a hint
    exhausted = back
    for _ in range(5):
        exhausted = executor.execute(ToolCall("go_back", {}))
        if not exhausted.ok:
            break
    assert exhausted.ok is False
    assert "no earlier page" in exhausted.summary
    assert exhausted.recovery_hint and "goto" in exhausted.recovery_hint


def test_wait_for_text_sees_delayed_content(executor, browser_session, fixture_site):
    browser_session.goto(f"{fixture_site}/lazy")
    model = browser_session.snapshot(budget_tokens=1200)
    load = element_named(model, "Load recommendations", role="button")

    executor.execute(ToolCall("click", {"element_id": load.id}))
    result = executor.execute(ToolCall("wait_for", {"text": "Recommended:", "timeout_seconds": 4}))

    assert result.ok is True, result.summary
    assert "Recommended:" in browser_session.snapshot(budget_tokens=1200).text


def test_wait_for_missing_text_times_out_with_a_hint(executor, browser_session, fixture_site):
    browser_session.goto(fixture_site)
    result = executor.execute(ToolCall("wait_for", {
        "text": "this phrase is not on the page", "timeout_seconds": 0.6,
    }))

    assert result.ok is False
    assert "never appeared" in result.summary
    assert result.recovery_hint


def test_wait_for_seconds_is_cheap_and_reported(executor, browser_session, fixture_site):
    browser_session.goto(fixture_site)
    result = executor.execute(ToolCall("wait_for", {"seconds": 0.2, "timeout_seconds": 1}))

    assert result.ok is True
    assert result.data["waited_s"] == pytest.approx(0.2, abs=0.3)


# --------------------------------------------------------------------------- #
# outline, tabs, screenshots
# --------------------------------------------------------------------------- #

def test_page_outline_modes_and_filter(executor, browser_session, fixture_site):
    browser_session.goto(f"{fixture_site}/checkout")
    full = executor.execute(ToolCall("page_outline", {"mode": "full"}))
    text = executor.execute(ToolCall("page_outline", {"mode": "text"}))
    filtered = executor.execute(ToolCall("page_outline", {"mode": "full", "filter": "zip"}))

    assert full.ok is True and full.page is not None
    assert len(full.page.elements) >= 3
    assert text.ok is True and text.page is not None
    assert len(text.page.elements) == len(full.page.elements), "the model keeps its ids"

    from webpilot.browser.snapshot import render_page_model

    rendered_text = render_page_model(text.page, budget_tokens=600, mode="text")
    assert not any(line.startswith("[") for line in rendered_text.splitlines())
    assert "TEXT:" in rendered_text
    assert filtered.ok is True and filtered.data["filter"] == "zip"
    assert 0 < len(filtered.page.elements) < len(full.page.elements)
    assert all("zip" in el.line().lower() for el in filtered.page.elements)


def test_tabs_tool_lists_opens_switches_and_closes(executor, browser_session, fixture_site):
    listed = executor.execute(ToolCall("tabs", {"action": "list"}))
    assert listed.ok is True
    assert len(listed.data["tabs"]) == 1

    opened = executor.execute(ToolCall("tabs", {"action": "open", "url": f"{fixture_site}/cart"}))
    assert opened.ok is True
    assert len(browser_session.pages) == 2
    assert browser_session.last_model.url.endswith("/cart")

    switched = executor.execute(ToolCall("tabs", {"action": "switch", "index": 0}))
    assert switched.ok is True
    assert len(browser_session.pages) == 2
    assert browser_session.last_model.url.rstrip("/") == fixture_site.rstrip("/")

    closed = executor.execute(ToolCall("tabs", {"action": "close", "index": 1}))
    assert closed.ok is True
    assert len(browser_session.pages) == 1


def test_screenshot_tool_writes_a_file(executor, config, fixture_site):
    executor.session.goto(fixture_site)
    result = executor.execute(ToolCall("screenshot", {"label": "actions-test"}))

    assert result.ok is True
    assert result.screenshot_path and Path(result.screenshot_path).exists()
    assert Path(result.screenshot_path).parent == Path(config.screenshot_dir)


def test_goto_accepts_relative_urls_and_refuses_nonsense(executor, browser_session, fixture_site):
    browser_session.goto(fixture_site)
    browser_session.snapshot(budget_tokens=800)

    relative = executor.execute(ToolCall("goto", {"url": "/cart"}))
    assert relative.ok is True
    assert browser_session.last_model.url.endswith("/cart")

    nonsense = executor.execute(ToolCall("goto", {"url": "find me some shoes"}))
    assert nonsense.ok is False
    assert "url" in (nonsense.recovery_hint or "").lower()


def test_every_action_returns_a_fresh_page_model_when_it_can(executor, browser_session, fixture_site):
    browser_session.goto(fixture_site)
    model = browser_session.snapshot(budget_tokens=1200)
    generation_before = model.generation
    link = element_named(model, "Lazy", role="link")

    result = executor.execute(ToolCall("click", {"element_id": link.id}))

    assert result.page is not None
    assert result.page.generation > generation_before
    assert result.page.url.endswith("/lazy")


# --------------------------------------------------------------------------- #
# input delivery: the native path comes first, an in-page dispatch is the
# fallback for environments where the browser drops synthetic input
#
# These tests simulate "the browser delivered nothing" by neutralising exactly
# the Playwright call the executor uses, so the rest of execute() (resolution,
# safety checks, verification) runs for real.
# --------------------------------------------------------------------------- #

def _explode(*args, **kwargs):
    raise AssertionError("the in-page fallback must not run in this test")


def _neutralised_delivery(*args, **kwargs):
    """Stand-in for a Playwright input call the browser never delivers."""
    return None


def test_click_reports_the_native_path_and_never_falls_back(
    executor, browser_session, fixture_site, monkeypatch
):
    browser_session.goto(fixture_site)
    model = browser_session.snapshot(budget_tokens=1200)
    link = element_named(model, "Search", role="link")

    clicks = []
    real_click = executor._native_mouse_click
    monkeypatch.setattr(executor, "_native_mouse_click",
                        lambda x, y: (clicks.append((x, y)), real_click(x, y)))
    monkeypatch.setattr(executor, "_dom_click_fallback", _explode)

    result = executor.execute(ToolCall("click", {"element_id": link.id}))

    assert result.ok is True, result.summary
    assert result.data["input_method"] == "native"
    assert "did not deliver" not in result.summary
    assert clicks, "the real mouse click must be the first thing tried"


def test_click_falls_back_to_an_in_page_sequence(
    executor, browser_session, fixture_site, monkeypatch
):
    """A handler-driven button: the mouse sequence dispatched in the page works."""
    browser_session.goto(fixture_site)
    model = browser_session.snapshot(budget_tokens=1200)
    button = element_named(model, "Show cookie banner", role="button")
    assert button is not None, [el.line() for el in model.elements]
    browser_session.page.evaluate("""() => {
        window.__webpilotEvents = [];
        ['pointerdown', 'mousedown', 'pointerup', 'mouseup', 'click'].forEach((type) => {
            document.addEventListener(type, () => window.__webpilotEvents.push(type), true);
        });
    }""")
    monkeypatch.setattr(executor, "_native_mouse_click", _neutralised_delivery)

    result = executor.execute(ToolCall("click", {"element_id": button.id}))

    assert result.ok is True, result.summary
    assert result.data["input_method"] == "dom_fallback"
    assert "did not deliver" in result.summary
    assert result.data["input_fallback"]["mechanism"] == "sequence"
    assert browser_session.page.evaluate(
        "() => document.getElementById('banner').style.display"
    ) == "block"
    seen = browser_session.page.evaluate("() => window.__webpilotEvents")
    for expected in ("pointerdown", "mousedown", "pointerup", "mouseup", "click"):
        assert expected in seen, f"{expected} was not dispatched in the page"


def test_click_fallback_follows_a_link(
    executor, browser_session, fixture_site, monkeypatch
):
    """Neutralising the native mouse click must not cost us a working link."""
    browser_session.goto(fixture_site)
    model = browser_session.snapshot(budget_tokens=1200)
    link = element_named(model, "Search", role="link")
    monkeypatch.setattr(executor, "_native_mouse_click", _neutralised_delivery)

    result = executor.execute(ToolCall("click", {"element_id": link.id}))

    assert result.ok is True, result.summary
    assert result.data["input_method"] == "dom_fallback"
    assert "did not deliver" in result.summary
    assert browser_session.last_model.url.endswith("/search")


def test_click_fallback_finishes_with_element_click_when_the_sequence_does_nothing(
    executor, browser_session, fixture_site, monkeypatch
):
    """If the dispatched sequence produces nothing, element.click() is the last resort."""
    import webpilot.browser.actions as actions

    browser_session.goto(fixture_site)
    model = browser_session.snapshot(budget_tokens=1200)
    link = element_named(model, "Search", role="link")
    monkeypatch.setattr(executor, "_native_mouse_click", _neutralised_delivery)
    monkeypatch.setattr(actions, "_JS_DOM_CLICK",
                        "(el, args) => ({hit: 'ignored', dispatched: []})")

    result = executor.execute(ToolCall("click", {"element_id": link.id}))

    assert result.ok is True, result.summary
    assert result.data["input_method"] == "dom_fallback"
    assert result.data["input_fallback"]["mechanism"] == "element.click()"
    assert browser_session.last_model.url.endswith("/search")


def test_click_fallback_walks_into_an_open_shadow_root(
    executor, browser_session, fixture_site, monkeypatch
):
    browser_session.goto(f"{fixture_site}/shadow")
    model = browser_session.snapshot(budget_tokens=1500)
    button = element_named(model, "Shadow action", role="button")
    assert button is not None, [el.line() for el in model.elements]
    monkeypatch.setattr(executor, "_native_mouse_click", _neutralised_delivery)

    result = executor.execute(ToolCall("click", {"element_id": button.id}))

    assert result.ok is True, result.summary
    assert result.data["input_method"] == "dom_fallback"
    assert shadow_contains(browser_session, "shadow clicked")
    # elementFromPoint alone would retarget to the shadow host, so this pins the
    # walk into the open shadow root: the hit target must be the button itself.
    hit = result.data["input_fallback"]["hit"]
    assert hit.startswith("button"), f"the hit test must find the button, got {hit!r}"


def test_click_fallback_reports_failure_when_neither_path_works(
    executor, browser_session, fixture_site, monkeypatch
):
    """A button with no handler: both paths do nothing and the result says so."""
    browser_session.goto(f"{fixture_site}/modal")
    model = browser_session.snapshot(budget_tokens=1500)
    opener = element_named(model, "Open newsletter dialog", role="button")
    assert executor.execute(ToolCall("click", {"element_id": opener.id})).ok is True

    model = browser_session.snapshot(budget_tokens=1500)
    subscribe = element_named(model, "Subscribe", role="button")
    monkeypatch.setattr(executor, "_native_mouse_click", _neutralised_delivery)

    result = executor.execute(ToolCall("click", {"element_id": subscribe.id}))

    assert result.ok is False
    assert result.page_changed is False
    assert result.data["fallback_attempted"] is True
    assert "did not change" in result.summary
    assert result.recovery_hint and "wait_for" in result.recovery_hint


def test_click_fallback_refuses_a_password_field(
    executor, browser_session, fixture_site, monkeypatch
):
    browser_session.goto(f"{fixture_site}/login")
    model = browser_session.snapshot(budget_tokens=1200)
    password = next(el for el in model.elements if el.type == "password")
    browser_session.page.evaluate("""() => {
        window.__webpilotEvents = 0;
        ['pointerdown', 'mousedown', 'click'].forEach((type) => {
            document.addEventListener(type, () => { window.__webpilotEvents += 1; }, true);
        });
    }""")
    monkeypatch.setattr(executor, "_native_mouse_click", _neutralised_delivery)

    result = executor.execute(ToolCall("click", {"element_id": password.id}))

    assert result.ok is False
    assert result.data["fallback_attempted"] is False, "nothing may be dispatched at a password field"
    assert "password" in result.summary.lower()
    assert "password" in (result.recovery_hint or "").lower()
    assert browser_session.page.evaluate("() => window.__webpilotEvents") == 0


def test_click_fallback_is_not_reached_for_a_stale_id(
    executor, browser_session, fixture_site, monkeypatch
):
    browser_session.goto(fixture_site)
    model = browser_session.snapshot(budget_tokens=1200)
    home_link = element_named(model, "Home", role="link")

    browser_session.goto(f"{fixture_site}/cart")
    browser_session.snapshot(budget_tokens=800)
    monkeypatch.setattr(executor, "_native_mouse_click", _neutralised_delivery)
    monkeypatch.setattr(executor, "_dom_click_fallback", _explode)

    result = executor.execute(ToolCall("click", {"element_id": home_link.id}))

    assert result.ok is False
    assert "not part of the current page snapshot" in result.summary
    assert "stale" in (result.recovery_hint or "")


def test_type_text_reports_the_native_path_and_never_falls_back(
    executor, browser_session, fixture_site, monkeypatch
):
    browser_session.goto(f"{fixture_site}/search")
    model = browser_session.snapshot(budget_tokens=1200)
    box = element_named(model, "Search products", role="textbox")
    monkeypatch.setattr(executor, "_dom_type_fallback", _explode)

    result = executor.execute(ToolCall("type_text", {
        "element_id": box.id, "text": "lamp", "submit": False, "clear": True,
    }))

    assert result.ok is True, result.summary
    assert result.data["input_method"] == "native"
    assert "did not deliver" not in result.summary
    assert browser_session.page.locator(selector_for(box.id)).first.input_value() == "lamp"


def test_type_text_falls_back_to_the_in_page_value_setter(
    executor, browser_session, fixture_site, monkeypatch
):
    browser_session.goto(f"{fixture_site}/search")
    model = browser_session.snapshot(budget_tokens=1200)
    box = element_named(model, "Search products", role="textbox")
    browser_session.page.evaluate("""() => {
        window.__webpilotInputs = 0;
        document.addEventListener('input', () => { window.__webpilotInputs += 1; }, true);
    }""")
    monkeypatch.setattr(executor, "_native_type", _neutralised_delivery)

    result = executor.execute(ToolCall("type_text", {
        "element_id": box.id, "text": "desk", "submit": False, "clear": True,
    }))

    assert result.ok is True, result.summary
    assert result.data["input_method"] == "dom_fallback"
    assert "did not deliver" in result.summary
    assert result.data["input_fallback"]["set_via"] == "native-setter"
    assert browser_session.page.locator(selector_for(box.id)).first.input_value() == "desk"
    assert browser_session.page.evaluate("() => window.__webpilotInputs") >= 1, \
        "the page's own JS must see an input event"


def test_type_text_fallback_submits_the_form_in_the_page(
    executor, browser_session, fixture_site, monkeypatch
):
    browser_session.goto(f"{fixture_site}/search")
    model = browser_session.snapshot(budget_tokens=1200)
    box = element_named(model, "Search products", role="textbox")
    monkeypatch.setattr(executor, "_native_type", _neutralised_delivery)
    monkeypatch.setattr(executor, "_native_locator_press", _neutralised_delivery)

    result = executor.execute(ToolCall("type_text", {
        "element_id": box.id, "text": "desk", "submit": True, "clear": True,
    }))

    assert result.ok is True, result.summary
    assert result.data["input_method"] == "dom_fallback"
    assert result.data["submitted"] is True
    assert result.data["input_fallback"]["form_submitted"] is True
    assert "did not deliver" in result.summary
    assert "q=desk" in browser_session.last_model.url


def test_type_text_fallback_still_refuses_a_password_field(
    executor, browser_session, fixture_site, monkeypatch
):
    browser_session.goto(f"{fixture_site}/login")
    model = browser_session.snapshot(budget_tokens=1200)
    password = next(el for el in model.elements if el.type == "password")
    monkeypatch.setattr(executor, "_native_type", _neutralised_delivery)
    monkeypatch.setattr(executor, "_dom_type_fallback", _explode)

    result = executor.execute(ToolCall("type_text", {
        "element_id": password.id, "text": "s3cr3t-value", "submit": True, "clear": True,
    }))

    assert result.ok is False
    assert result.recovery_hint == "password fields are filled by the human"
    value = browser_session.page.locator(selector_for(password.id)).first.input_value()
    assert value == "", "the in-page fallback must not fill a password field"


def test_type_text_fallback_honours_allow_password_typing(
    config, browser_session, fixture_site, monkeypatch
):
    """config.allow_password_typing must mean the same thing on both paths."""
    import dataclasses

    executor = ActionExecutor(browser_session, dataclasses.replace(config, allow_password_typing=True))
    browser_session.goto(f"{fixture_site}/login")
    model = browser_session.snapshot(budget_tokens=1200)
    password = next(el for el in model.elements if el.type == "password")
    monkeypatch.setattr(executor, "_native_type", _neutralised_delivery)
    monkeypatch.setattr(executor, "_native_locator_press", _neutralised_delivery)

    result = executor.execute(ToolCall("type_text", {
        "element_id": password.id, "text": "opt-in-value", "submit": False, "clear": True,
    }))

    assert result.ok is True, result.summary
    assert result.data["input_method"] == "dom_fallback"
    value = browser_session.page.locator(selector_for(password.id)).first.input_value()
    assert value == "opt-in-value"


def test_press_key_reports_the_native_path_and_never_falls_back(
    executor, browser_session, fixture_site, monkeypatch
):
    browser_session.goto(f"{fixture_site}/search?q=desk")
    browser_session.page.set_viewport_size({"width": 800, "height": 300})
    model = browser_session.snapshot(budget_tokens=1500)
    assert model.scroll.get("max_y", 0) > 0
    monkeypatch.setattr(executor, "_dom_press_fallback", _explode)

    result = executor.execute(ToolCall("press_key", {"key": "PageDown"}))

    assert result.ok is True, result.summary
    assert result.data["input_method"] == "native"
    assert "did not deliver" not in result.summary


def test_press_key_falls_back_to_an_in_page_enter(
    executor, browser_session, fixture_site, monkeypatch
):
    browser_session.goto(f"{fixture_site}/search")
    model = browser_session.snapshot(budget_tokens=1200)
    box = element_named(model, "Search products", role="textbox")
    browser_session.page.evaluate("""(selector) => {
        // window.name survives the navigation the submit causes, so the test can
        // still tell that the page received the key events.
        window.name = "";
        document.addEventListener('keydown', (event) => {
            window.name = window.name + event.key + ',';
        }, true);
        document.querySelector(selector).focus();
    }""", selector_for(box.id))
    monkeypatch.setattr(executor, "_native_press", _neutralised_delivery)

    result = executor.execute(ToolCall("press_key", {"key": "Enter"}))

    assert result.ok is True, result.summary
    assert result.data["input_method"] == "dom_fallback"
    assert "did not deliver" in result.summary
    assert result.data["input_fallback"]["form_submitted"] is True
    assert "q=" in browser_session.last_model.url, "requestSubmit must have submitted the form"
    assert "Enter," in browser_session.page.evaluate("() => window.name")
