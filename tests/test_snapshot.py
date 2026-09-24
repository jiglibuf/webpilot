"""Unit tests for the page model: parsing, rendering and the token budget.

These tests never touch a browser - they feed ``build_page_model`` the exact JSON
shape ``dom_snapshot.js`` produces, which is what pins the interface between the
sniffer and the Python side.
"""

from __future__ import annotations

import pytest

from webpilot.browser.snapshot import build_page_model, render_page_model
from webpilot.tokenizer import count_tokens

BUDGET = 1200


# --------------------------------------------------------------------------- #
# fixtures: raw payloads in the sniffer's JSON shape
# --------------------------------------------------------------------------- #

def _element(element_id: int, *, role: str, name: str, tag: str = "div", **extra) -> dict:
    base = {
        "id": element_id,
        "role": role,
        "name": name,
        "tag": tag,
        "type": None,
        "value": None,
        "placeholder": None,
        "href": None,
        "disabled": False,
        "checked": None,
        "required": False,
        "expanded": None,
        "invalid": False,
        "inViewport": True,
        "options": [],
        "domHash": f"hash{element_id}",
        "selectorHint": tag,
        "frameIndex": 0,
        "note": "",
        "isPassword": False,
    }
    base.update(extra)
    return base


def login_payload() -> dict:
    """The page from the rendering contract in docs/INTERFACES.md."""
    return {
        "url": "https://site/path",
        "title": "Sign in",
        "elements": [
            _element(3, role="textbox", name="Username", tag="input", placeholder="Username"),
            _element(4, role="textbox", name="Password", tag="input", type="password",
                     value="hunter2-secret"),
            _element(5, role="submit", name="Login", tag="button", type="submit"),
        ],
        "text": [
            {"kind": "heading", "text": "Sign in", "inViewport": True},
            {"kind": "paragraph", "text": "Use your account to continue.", "inViewport": True},
        ],
        "alerts": ["Invalid username or password."],
        "dialogs": ["Join the newsletter"],
        "scroll": {"y": 0, "maxY": 2400, "viewportH": 800, "atBottom": False},
        "focusedId": 3,
        "htmlChars": 152340,
        "fullTextChars": 8123,
        "droppedElements": 2,
        "frames": ["https://site/path"],
        "errors": [],
    }


def crowded_payload(in_viewport: int = 3, off_screen: int = 40, text_blocks: int = 30) -> dict:
    elements = [
        _element(100 + i, role="button", name=f"In-view button number {i} with a fairly long label",
                 tag="button")
        for i in range(in_viewport)
    ]
    elements += [
        _element(200 + i, role="link", name=f"Off-screen link {i} " + "x" * 40, tag="a",
                 href="https://site/path", inViewport=False)
        for i in range(off_screen)
    ]
    text = [
        {"kind": "paragraph", "text": f"Text block {i} " + "y" * 60, "inViewport": i < 4}
        for i in range(text_blocks)
    ]
    return {
        "url": "https://site/path",
        "title": "Crowded",
        "elements": elements,
        "text": text,
        "alerts": [],
        "dialogs": [],
        "scroll": {"y": 0, "maxY": 100, "viewportH": 900, "atBottom": False},
        "htmlChars": 999,
        "fullTextChars": 4242,
        "droppedElements": 0,
        "frames": ["https://site/path"],
        "errors": [],
    }


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def test_build_page_model_maps_the_sniffer_payload():
    model = build_page_model(login_payload(), generation=7, budget_tokens=BUDGET)

    assert model.url == "https://site/path"
    assert model.title == "Sign in"
    assert model.generation == 7
    assert [el.id for el in model.elements] == [3, 4, 5]
    assert model.elements[0].role == "textbox"
    assert model.elements[0].placeholder == "Username"
    assert model.elements[2].role == "submit"
    assert model.elements[2].tag == "button"
    assert model.focused_id == 3
    assert model.alerts == ["Invalid username or password."]
    assert model.dialogs == ["Join the newsletter"]
    assert model.scroll == {"y": 0, "max_y": 2400, "viewport_h": 800, "at_bottom": False}
    assert model.html_chars == 152340
    assert model.full_text_chars == 8123
    assert model.token_estimate > 0


def test_password_values_never_enter_the_model():
    model = build_page_model(login_payload(), generation=1, budget_tokens=BUDGET)
    password = model.by_id(4)

    assert password is not None
    assert password.type == "password"
    assert password.value is None, "a password typed by the human must not be modelled"
    rendered = render_page_model(model, budget_tokens=BUDGET)
    assert "hunter2-secret" not in rendered
    assert "[4] textbox 'Password' type=password" in rendered


def test_names_are_normalised_and_truncated():
    payload = login_payload()
    payload["elements"] = [
        _element(1, role="button", name="  Save\u00a0 \u00a0 and\tcontinue  "),
        _element(2, role="button", name="L" * 300),
    ]
    model = build_page_model(payload, generation=1, budget_tokens=BUDGET)

    assert model.by_id(1).name == "Save and continue"
    assert len(model.by_id(2).name) == 120
    assert set(model.by_id(2).name) == {"L"}


def test_unknown_roles_degrade_to_other_and_ids_are_required():
    payload = login_payload()
    payload["elements"] = [
        _element(9, role="glorp", name="weird"),
        _element(0, role="button", name="no id"),
    ]
    model = build_page_model(payload, generation=1, budget_tokens=BUDGET)

    assert [el.id for el in model.elements] == [9]
    assert model.elements[0].role == "other"


def test_invisible_nodes_do_not_mark_the_model_truncated():
    model = build_page_model(login_payload(), generation=1, budget_tokens=BUDGET)

    assert model.truncated is False
    assert model.dropped_elements == 0
    assert model.text_dropped_chars == 0


# --------------------------------------------------------------------------- #
# rendering contract
# --------------------------------------------------------------------------- #

def test_render_page_model_follows_the_contract():
    model = build_page_model(login_payload(), generation=7, budget_tokens=BUDGET)
    rendered = render_page_model(model, budget_tokens=BUDGET)
    lines = rendered.splitlines()

    assert lines[0] == "URL: https://site/path | Title: Sign in | generation 7 | scroll 0/2400"
    assert lines[1] == "ALERTS: Invalid username or password."
    assert lines[2] == "DIALOG: Join the newsletter"
    assert "[3] textbox 'Username' placeholder='Username'" in lines
    assert "[4] textbox 'Password' type=password" in lines
    assert "[5] submit 'Login'" in lines
    # element lines come before the text section, the label is its own line
    assert lines.index("TEXT:") > max(i for i, ln in enumerate(lines) if ln.startswith("[5]"))
    assert "heading: Sign in" in lines
    assert "paragraph: Use your account to continue." in lines
    assert lines.index("heading: Sign in") > lines.index("TEXT:")


def test_text_mode_renders_no_element_lines():
    model = build_page_model(login_payload(), generation=7, budget_tokens=BUDGET)
    rendered = render_page_model(model, budget_tokens=BUDGET, mode="text")

    assert "[3]" not in rendered
    assert "TEXT:" in rendered
    assert "heading: Sign in" in rendered
    assert "DIALOG: Join the newsletter" in rendered


def test_delta_mode_renders_only_changed_elements():
    previous = build_page_model(login_payload(), generation=1, budget_tokens=BUDGET)
    payload = login_payload()
    payload["elements"][2]["domHash"] = "changed"          # only the Login button moved
    current = build_page_model(payload, generation=2, budget_tokens=BUDGET, previous=previous)

    rendered = render_page_model(current, budget_tokens=400, mode="delta", previous=previous)

    assert "[5] submit 'Login'" in rendered
    assert "[3] textbox 'Username'" not in rendered
    assert "(unchanged: 2 elements, use ids from the previous snapshot)" in rendered


def test_delta_without_a_previous_model_falls_back_to_full():
    model = build_page_model(login_payload(), generation=1, budget_tokens=BUDGET)
    rendered = render_page_model(model, budget_tokens=BUDGET, mode="delta", previous=None)

    assert "[3] textbox 'Username'" in rendered
    assert "unchanged" not in rendered


def test_filter_is_applied_by_page_outline_not_by_the_renderer():
    """The renderer has no filter: narrowing a page is the executor's job."""
    model = build_page_model(login_payload(), generation=1, budget_tokens=BUDGET)
    rendered = render_page_model(model, budget_tokens=BUDGET)

    assert "Username" in rendered and "Password" in rendered


# --------------------------------------------------------------------------- #
# budget
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("budget", [8, 16, 32, 64, 128, 320, 900])
def test_render_never_exceeds_the_budget(budget: int):
    model = build_page_model(crowded_payload(), generation=1, budget_tokens=budget)
    rendered = render_page_model(model, budget_tokens=budget)

    assert count_tokens(rendered) <= budget, rendered[:200]


@pytest.mark.parametrize("budget", [8, 32, 120, 600])
def test_build_page_model_reports_its_own_token_estimate(budget: int):
    model = build_page_model(crowded_payload(), generation=1, budget_tokens=budget)

    assert isinstance(model.token_estimate, int)
    assert model.token_estimate <= budget


def test_tight_budget_drops_off_screen_elements_first():
    payload = crowded_payload(in_viewport=3, off_screen=40)
    crowded = build_page_model(payload, generation=1, budget_tokens=100_000)
    tight = build_page_model(payload, generation=1, budget_tokens=140)
    rendered = render_page_model(tight, budget_tokens=140)

    assert tight.truncated is True
    assert 0 < tight.dropped_elements < len(crowded.elements)
    # every in-viewport element is kept, and the ones that fall off the end of
    # the list are the off-screen ones (the in-viewport group is rendered first)
    for element in crowded.elements:
        if element.in_viewport:
            assert element.line() in rendered
    kept = [el.line() for el in crowded.elements if el.line() in rendered]
    assert kept == [el.line() for el in crowded.elements[: len(kept)]]
    assert len(kept) == len(crowded.elements) - tight.dropped_elements


def test_text_is_ranked_below_interactive_elements():
    """Contract order: elements first, then text - off-screen text goes last."""
    payload = crowded_payload(in_viewport=1, off_screen=0, text_blocks=10)
    model = build_page_model(payload, generation=1, budget_tokens=200)
    rendered = render_page_model(model, budget_tokens=200)

    assert "Text block 0" in rendered          # in-viewport text survives
    assert "Text block 9" not in rendered      # off-screen text is dropped
    assert model.text_dropped_chars > 0
    assert model.dropped_elements == 0


def test_text_dropped_chars_is_honest():
    payload = crowded_payload(in_viewport=1, off_screen=0, text_blocks=20)
    model = build_page_model(payload, generation=1, budget_tokens=400)
    rendered = render_page_model(model, budget_tokens=400)

    total = sum(len(block["text"]) for block in payload["text"])
    kept = sum(
        len(block["text"]) for block in payload["text"]
        if f"paragraph: {block['text']}" in rendered
    )
    assert model.text_dropped_chars == total - kept
    assert model.truncated is True


def test_a_custom_tokenizer_is_honoured():
    estimator = lambda text: max(1, len(text) // 3)  # noqa: E731 - deliberate, tiny stand-in
    model = build_page_model(crowded_payload(), generation=1, budget_tokens=300,
                             count_tokens=estimator)
    rendered = render_page_model(model, budget_tokens=300, count_tokens=estimator)

    assert model.token_estimate <= 300
    assert estimator(rendered) <= 300


def test_ids_are_addressable_without_any_selector_in_the_payload():
    """The anti-hardcoding invariant at the data level: nothing but an integer."""
    payload = login_payload()
    for element in payload["elements"]:
        element.pop("selectorHint", None)  # hints are for humans only
    model = build_page_model(payload, generation=1, budget_tokens=BUDGET)

    assert [el.id for el in model.elements] == [3, 4, 5]
    assert all(isinstance(el.id, int) for el in model.elements)
    assert model.page_hash() != ""
