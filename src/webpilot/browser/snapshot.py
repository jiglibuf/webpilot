"""``dom_snapshot.js`` result -> :class:`webpilot.types.PageModel`, and the
token-bounded text rendering of that model.

This module is the *only* place where the compact page representation is built,
so the rendering contract in ``docs/INTERFACES.md`` lives in exactly one
function: :func:`render_page_model`.

Two rules drive the design:

* the rendered page block must **never** exceed ``budget_tokens`` - it is fed
  straight into the model context, so an over-budget render is a context
  overflow by construction;
* everything reported (``truncated``, ``dropped_elements``,
  ``text_dropped_chars``, ``token_estimate``) has to be honest about what was
  cut, because the agent uses those numbers to decide to ask for a delta or to
  scroll.
"""

from __future__ import annotations

from typing import Any, Callable

from ..tokenizer import count_tokens as default_count_tokens
from ..types import Element, ElementRole, PageModel, SnapshotMode

__all__ = ["build_page_model", "render_page_model"]

#: Which element roles the sniffer may emit; anything else degrades to "other".
_KNOWN_ROLES: frozenset[str] = frozenset(
    {
        "button", "link", "textbox", "checkbox", "radio", "select", "option",
        "menuitem", "tab", "switch", "slider", "file", "submit", "label", "other",
    }
)

#: Tokens held back so that the truncation footer always fits.
_FOOTER_RESERVE = 28
#: Per-line overhead charged on top of each line's own token estimate; keeps the
#: running sum conservative (the real tokenizer merges across newlines).
_LINE_OVERHEAD = 2
_TITLE_MAX = 120
_URL_MAX = 200
_MAX_ALERTS_RENDERED = 6
_MAX_DIALOGS_RENDERED = 4
_MAX_ERRORS_RENDERED = 4


# --------------------------------------------------------------------------- #
# raw JS -> types
# --------------------------------------------------------------------------- #

def _first(raw: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Read the first present key (the JS uses camelCase, Python snake_case)."""
    for key in keys:
        if key in raw and raw[key] is not None:
            return raw[key]
    return default


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\u00a0", " ").replace("\r", "")
    return " ".join(text.split())


def _role_of(raw_role: Any) -> ElementRole:
    role = str(raw_role or "").strip().lower()
    if role in _KNOWN_ROLES:
        return role  # type: ignore[return-value]
    return "other"


def element_from_raw(raw: dict[str, Any]) -> Element:
    """Convert one sniffer element entry into an :class:`Element`."""
    etype = _first(raw, "type", default=None)
    etype = str(etype).lower() if etype else None
    role = _role_of(_first(raw, "role", default="other"))
    if etype == role:
        # "submit 'Login' type=submit" wastes tokens on a fact the role already
        # states; keep the type only when it adds information (password, email...).
        etype = None
    value = _first(raw, "value", default=None)
    if value is not None:
        value = _clean_text(value)[:200] or None
    if etype == "password":
        # A password value typed by the human must never enter the model
        # context (or the transcript) - mask it at the boundary.
        value = None
    options = [_clean_text(o) for o in (_first(raw, "options", default=[]) or [])]
    return Element(
        id=int(_first(raw, "id", default=0) or 0),
        role=role,
        name=_clean_text(_first(raw, "name", default=""))[:120],
        tag=_clean_text(_first(raw, "tag", default="")).lower(),
        type=etype,
        value=value,
        placeholder=(_clean_text(_first(raw, "placeholder", default="")) or None),
        href=(_first(raw, "href", default=None) or None),
        disabled=bool(_first(raw, "disabled", default=False)),
        checked=(None if _first(raw, "checked", default=None) is None
                 else bool(_first(raw, "checked"))),
        required=bool(_first(raw, "required", default=False)),
        expanded=(None if _first(raw, "expanded", default=None) is None
                  else bool(_first(raw, "expanded"))),
        invalid=bool(_first(raw, "invalid", default=False)),
        in_viewport=bool(_first(raw, "inViewport", "in_viewport", default=True)),
        frame=int(_first(raw, "frameIndex", "frame", default=0) or 0),
        options=[o for o in options if o][:25],
        dom_hash=str(_first(raw, "domHash", "dom_hash", default="") or ""),
        selector_hint=_clean_text(_first(raw, "selectorHint", "selector_hint", default=""))[:120],
        note=_clean_text(_first(raw, "note", default=""))[:80],
    )


def _scroll_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    scroll = raw.get("scroll") or {}
    if not isinstance(scroll, dict):
        return {"y": 0, "max_y": 0, "viewport_h": 0, "at_bottom": True}
    y = int(_first(scroll, "y", default=0) or 0)
    max_y = int(_first(scroll, "maxY", "max_y", default=0) or 0)
    return {
        "y": y,
        "max_y": max_y,
        "viewport_h": int(_first(scroll, "viewportH", "viewport_h", default=0) or 0),
        "at_bottom": bool(_first(scroll, "atBottom", "at_bottom", default=(y >= max_y))),
    }


def _text_blocks(raw: dict[str, Any]) -> list[tuple[str, str, bool]]:
    """Ranked ``(kind, text, in_viewport)`` blocks; in-viewport text comes first."""
    blocks: list[tuple[str, str, bool, int]] = []
    for index, item in enumerate(raw.get("text") or []):
        if not isinstance(item, dict):
            continue
        text = _clean_text(item.get("text"))
        if len(text) < 2:
            continue
        kind = _clean_text(item.get("kind")) or "paragraph"
        in_viewport = bool(item.get("inViewport", True))
        blocks.append((kind, text, in_viewport, index))
    blocks.sort(key=lambda b: (0 if b[2] else 1, b[3]))
    return [(kind, text, in_viewport) for kind, text, in_viewport, _ in blocks]


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #

def build_page_model(
    raw: dict,
    *,
    generation: int,
    budget_tokens: int,
    mode: SnapshotMode = "full",
    previous: PageModel | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> PageModel:
    """Turn one raw ``dom_snapshot.js`` result into a :class:`PageModel`.

    ``budget_tokens``/``mode``/``previous`` are used to render the model once, so
    that ``token_estimate``, ``truncated``, ``dropped_elements`` and
    ``text_dropped_chars`` describe the block the agent will actually receive.
    """
    count = count_tokens or default_count_tokens
    if not isinstance(raw, dict):
        raw = {}

    elements = [
        element_from_raw(item)
        for item in (raw.get("elements") or [])
        if isinstance(item, dict)
    ]
    # ids must be usable as the single addressing handle: drop unusable rows.
    elements = [el for el in elements if el.id > 0]

    text_blocks = _text_blocks(raw)
    alerts = [_clean_text(a) for a in (raw.get("alerts") or []) if _clean_text(a)]
    dialogs = [_clean_text(d) for d in (raw.get("dialogs") or []) if _clean_text(d)]
    errors = [_clean_text(e)[:200] for e in (raw.get("errors") or []) if _clean_text(e)]

    focused_raw = _first(raw, "focusedId", "focused_id", default=None)
    focused_id = int(focused_raw) if isinstance(focused_raw, (int, float)) and focused_raw else None

    frames = [str(f) for f in (raw.get("frames") or [])]
    if not frames:
        frames = [str(raw.get("url") or "")]

    model = PageModel(
        url=str(raw.get("url") or ""),
        title=_clean_text(raw.get("title"))[:_TITLE_MAX],
        generation=int(generation),
        elements=elements,
        text="\n".join(f"{kind}: {text}" for kind, text, _ in text_blocks),
        alerts=alerts,
        dialogs=dialogs,
        scroll=_scroll_from_raw(raw),
        tabs=[],
        frames=frames,
        focused_id=focused_id,
        full_text_chars=int(_first(raw, "fullTextChars", "full_text_chars", default=0) or 0),
        html_chars=int(_first(raw, "htmlChars", "html_chars", default=0) or 0),
        errors=errors,
    )
    # Sidecar used by the renderer for the in-viewport priority of text blocks.
    model._text_blocks = text_blocks  # type: ignore[attr-defined]
    # Invisible/uninteresting nodes the sniffer skipped.  Kept for diagnostics -
    # it does NOT set `truncated` (that flag is about the *budget*, not the page).
    model._dropped_in_sniffer = int(raw.get("droppedElements") or 0)  # type: ignore[attr-defined]

    rendered = render_page_model(
        model, budget_tokens=budget_tokens, mode=mode, previous=previous, count_tokens=count
    )
    model.token_estimate = count(rendered)
    return model


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #

def _header_line(model: PageModel) -> str:
    scroll = model.scroll or {}
    url = (model.url or "")[:_URL_MAX]
    title = _clean_text(model.title)[:_TITLE_MAX]
    return (
        f"URL: {url} | Title: {title} | generation {model.generation} "
        f"| scroll {scroll.get('y', 0)}/{scroll.get('max_y', 0)}"
    )


def _rendered_elements(
    model: PageModel, mode: SnapshotMode, previous: PageModel | None
) -> tuple[list[Element], int]:
    """Elements to render for ``mode`` plus the number of unchanged ones hidden.

    ``mode="text"`` renders no element list at all; ``mode="delta"`` renders only
    the elements whose ``dom_hash`` moved; ``mode="full"`` renders everything.
    """
    if mode == "text":
        return [], 0
    if mode == "delta" and previous is not None:
        changed = _changed_elements(model, previous)
        return changed, len(model.elements) - len(changed)
    return list(model.elements), 0


def render_page_model(
    model: PageModel,
    *,
    budget_tokens: int,
    mode: SnapshotMode = "full",
    previous: PageModel | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> str:
    """Render ``model`` as the compact page block fed to the LLM.

    Layout (see ``docs/INTERFACES.md``)::

        URL: <url> | Title: <title> | generation <n> | scroll <y>/<maxY>
        ALERTS: <alert> || <alert>
        DIALOG: <dialog text>
        [3] textbox 'Username' placeholder='Username'
        [4] textbox 'Password' type=password
        [5] submit 'Login'
        TEXT:
        heading: ...
        paragraph: ...
        (unchanged: 4 elements, use ids from the previous snapshot)

    Budget order: alerts/dialogs -> interactive elements in viewport ->
    interactive elements off-screen -> text in viewport -> text off-screen.
    Elements that do not fit stay in ``model.elements`` (their ids are still
    addressable, e.g. after a ``delta`` snapshot) but are counted in
    ``model.dropped_elements``; the text that does not fit is counted in
    ``model.text_dropped_chars``.

    ``count_tokens`` is an *additive* optional override of the estimator (the
    contract's signature is preserved); ``build_page_model`` passes its own so
    that both agree no matter which estimator the caller uses.

    Note: the function writes the honest bookkeeping fields back onto ``model``
    (``truncated``, ``dropped_elements``, ``text_dropped_chars``) so the model and
    the rendered text can never disagree.
    """
    count = count_tokens or default_count_tokens
    budget = max(1, int(budget_tokens))

    header: list[str] = [_header_line(model)]
    if model.alerts:
        header.append("ALERTS: " + " || ".join(model.alerts[:_MAX_ALERTS_RENDERED]))
    for dialog in model.dialogs[:_MAX_DIALOGS_RENDERED]:
        header.append("DIALOG: " + dialog)
    if model.errors:
        header.append("ERRORS: " + " || ".join(model.errors[:_MAX_ERRORS_RENDERED]))

    rendered_elements, unchanged = _rendered_elements(model, mode, previous)
    in_viewport_lines = [el.line() for el in rendered_elements if el.in_viewport]
    off_screen_lines = [el.line() for el in rendered_elements if not el.in_viewport]

    text_blocks: list[tuple[str, str, bool]] = list(getattr(model, "_text_blocks", []) or [])
    if not text_blocks and model.text:
        text_blocks = [
            (line.split(":", 1)[0].strip(), line.split(":", 1)[1].strip(), True)
            for line in model.text.splitlines()
            if ":" in line
        ]

    # ---- candidate lines in strict budget priority order
    items: list[tuple[str, str, int]] = []          # (group, line, weight)
    for line in in_viewport_lines:
        items.append(("element", line, 1))
    for line in off_screen_lines:
        items.append(("element", line, 1))
    text_header = "TEXT:"
    for _kind, text, in_viewport in text_blocks:
        items.append(("text", f"{_kind}: {text}", max(1, len(text))))

    head = "\n".join(header)
    text_header = "TEXT:"
    used = count(head)
    if any(group == "text" for group, _l, _w in items):
        # reserve the section label so that "text without a label" cannot happen
        used += count(text_header) + _LINE_OVERHEAD
    kept_index: set[int] = set()

    for index, (group, line, _weight) in enumerate(items):
        cost = count(line) + _LINE_OVERHEAD
        if used + cost + _FOOTER_RESERVE <= budget:
            kept_index.add(index)
            used += cost

    dropped_elements = sum(
        1 for i, (group, _l, _w) in enumerate(items) if group == "element" and i not in kept_index
    )
    dropped_chars = sum(
        weight for i, (group, _l, weight) in enumerate(items)
        if group == "text" and i not in kept_index
    )

    # ---- assemble in render order (label before the first kept text line)
    body: list[tuple[int | None, str]] = []
    label_placed = False
    for index, (group, line, _weight) in enumerate(items):
        if index not in kept_index:
            continue
        if group == "text" and not label_placed:
            body.append((None, text_header))
            label_placed = True
        body.append((index, line))
    if mode == "delta" and unchanged:
        body.append((None, f"(unchanged: {unchanged} elements, use ids from the previous snapshot)"))

    footer_bits: list[str] = []
    if dropped_elements:
        footer_bits.append(f"{dropped_elements} elements")
    if dropped_chars:
        footer_bits.append(f"{dropped_chars} text chars")
    if unchanged:
        footer_bits.append(f"{unchanged} unchanged elements hidden")
    footer = "TRUNCATED: " + ", ".join(footer_bits) if (dropped_elements or dropped_chars) else ""

    lines = list(header) + [line for _i, line in body]
    if footer:
        lines.append(footer)
    rendered = "\n".join(lines)

    # Final guard: the running sum is conservative but the joined string is the
    # ground truth.  Trim by binary search (rare, keeps the call cheap).
    if count(rendered) > budget:
        lo, hi = 0, len(lines)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count("\n".join(lines[:mid])) <= budget:
                lo = mid
            else:
                hi = mid - 1
        lines = lines[:lo]
        rendered = "\n".join(lines)
        body_kept = max(0, lo - len(header))
        kept_index = {
            index for index, _line in body[:body_kept] if index is not None
        }
        dropped_elements = sum(
            1 for i, (group, _l, _w) in enumerate(items)
            if group == "element" and i not in kept_index
        )
        dropped_chars = sum(
            weight for i, (group, _l, weight) in enumerate(items)
            if group == "text" and i not in kept_index
        )

    model.truncated = bool(dropped_elements or dropped_chars)
    model.dropped_elements = dropped_elements
    model.text_dropped_chars = dropped_chars
    return rendered


def _changed_elements(model: PageModel, previous: PageModel | None) -> list[Element]:
    """Elements whose ``dom_hash`` differs from the previous snapshot."""
    if previous is None:
        return list(model.elements)
    previous_hashes = {el.id: el.dom_hash for el in previous.elements}
    return [el for el in model.elements if previous_hashes.get(el.id) != el.dom_hash]
