"""Tool call -> real browser action, with verification.

Every action here follows the same shape:

1. **resolve** the element id handed over by the page sniffer (never a
   hand-written selector - the CSS is *generated* from the integer, see
   :func:`selector_for`);
2. **act** like a human would (mouse click at the box centre, sequential key
   presses with a delay);
3. **verify**: capture the page state (url, title, DOM signature, dialog count)
   before and after, wait for the network to settle *or* for the DOM to move, and
   when nothing happened return ``ok=False`` with a precise diagnostic plus a
   ``recovery_hint`` the agent loop can act on.
4. **only if** step 3 proved the action had no effect *and* the target is still
   actionable, retry once by dispatching the event sequence inside the page
   (``input_method="dom_fallback"``) - a workaround for environments where
   Chromium drops Playwright's synthetic input in a persistent context, never a
   replacement for the native path.  See the module-level note above the
   ``_JS_DOM_*`` snippets.

Refusing and explaining beats pretending: a silent no-op is the most expensive
failure mode an agent can have.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable
from urllib.parse import urljoin

from ..errors import BrowserError
from ..types import Element, PageModel, SnapshotMode, ToolCall, ToolResult
from .session import BrowserSession

__all__ = ["ActionExecutor", "selector_for", "WEBPILOT_ATTR"]

#: The attribute the sniffer writes.  It is the project's only addressing handle.
WEBPILOT_ATTR = "data-webpilot-id"

#: Tools the agent loop owns; the browser layer must not silently implement them.
AGENT_LOCAL_TOOLS = frozenset({"ask_page", "notes", "ask_user", "finish"})

_MAX_WAIT_S = 60.0


def selector_for(element_id: int) -> str:
    """Generate the CSS selector for a sniffer-provided element id.

    This function is the *only* place in the project allowed to turn an id into a
    selector, and it is generated from the integer - never typed by hand for a
    particular site.
    """
    return f'[{WEBPILOT_ATTR}="{int(element_id)}"]'


# --------------------------------------------------------------------------- #
# injected helpers (generic engine code - no site knowledge)
# --------------------------------------------------------------------------- #

#: Cheap "did anything move?" probe.  Installs a mutation counter once per
#: document (main frame + same-origin frames) and reports the state hashed for
#: verification.
_JS_STATE = """
(() => {
  const OBSERVE = {subtree: true, childList: true, attributes: true, characterData: true};
  const install = (node) => {
    if (!node || node.__webpilotMutations !== undefined) return;
    node.__webpilotMutations = 0;
    try {
      new MutationObserver((records) => {
        node.__webpilotMutations += records.length;
      }).observe(node, OBSERVE);
    } catch (e) { /* observation is best effort */ }
  };
  // A mutation observer on a document does NOT see inside shadow trees, so open
  // shadow roots get their own observer - otherwise a click that only touches
  // shadow content would look like "nothing happened".
  const observeTree = (root) => {
    install(root);
    let count = 0;
    let nodes = [];
    try { nodes = root.querySelectorAll("*"); } catch (e) { nodes = []; }
    for (let i = 0; i < nodes.length; i++) {
      let shadow = null;
      try { shadow = nodes[i].shadowRoot || null; } catch (e) { shadow = null; }
      if (shadow) count += observeTree(shadow);
    }
    count += (root.__webpilotMutations || 0);
    return count;
  };
  const probe = (doc) => {
    if (!doc) return null;
    const mutations = observeTree(doc);
    const de = doc.documentElement;
    const body = doc.body;
    const view = doc.defaultView;
    return {
      url: String((view && view.location ? view.location.href : "") || ""),
      title: doc.title || "",
      mutations: mutations,
      nodes: de ? de.querySelectorAll("*").length : 0,
      textLen: body ? (body.innerText || "").length : 0,
      scrollY: view ? Math.round(view.scrollY || 0) : 0,
      scrollMax: de && view ? Math.max(0, (de.scrollHeight || 0) - (view.innerHeight || 0)) : 0,
      viewportH: view ? (view.innerHeight || 0) : 0,
    };
  };
  const docs = [];
  const walk = (doc) => {
    if (!doc || docs.indexOf(doc) !== -1) return;
    docs.push(doc);
    let frames = [];
    try { frames = doc.querySelectorAll("iframe"); } catch (e) { frames = []; }
    for (let i = 0; i < frames.length; i++) {
      let inner = null;
      try { inner = frames[i].contentDocument; } catch (e) { inner = null; }
      if (inner) walk(inner);
    }
  };
  walk(document);
  const states = [];
  for (let i = 0; i < docs.length; i++) {
    const state = probe(docs[i]);
    if (state) states.push(state);
  }
  return {main: states[0] || {}, frames: states.slice(1)};
})()
"""

_JS_ELEMENT_STATE = """
(el) => {
  const out = {tag: String(el.tagName || "").toLowerCase(), type: null, value: null,
               checked: null, expanded: null, disabled: null, focused: null,
               label: ""};
  try { out.type = el.type ? String(el.type).toLowerCase() : null; } catch (e) {}
  try { if (el.value !== undefined && el.value !== null) out.value = String(el.value); } catch (e) {}
  if (typeof el.checked === "boolean") out.checked = el.checked;
  try {
    const exp = el.getAttribute("aria-expanded");
    out.expanded = exp === null ? null : exp === "true";
  } catch (e) {}
  out.disabled = !!el.disabled;
  const owner = el.ownerDocument || document;
  out.focused = owner.activeElement === el;
  out.label = String(el.getAttribute("aria-label") || (el.innerText || "") || "").trim().slice(0, 80);
  return out;
}
"""

_JS_COVER = """
(el) => {
  const r = el.getBoundingClientRect();
  const x = r.left + r.width / 2;
  const y = r.top + r.height / 2;
  const top = document.elementFromPoint(x, y);
  const result = {covered: false, x: x, y: y, by: null};
  if (!top) { result.covered = true; result.by = "nothing at that point"; return result; }
  // Inside a shadow tree the hit test retargets to the shadow *host*, which is
  // not a light-DOM ancestor: walk the host chain before crying "covered".
  const hosts = [];
  let node = el;
  let hops = 0;
  while (node && hops++ < 10) {
    const root = node.getRootNode ? node.getRootNode() : null;
    if (!root || !root.host) break;
    hosts.push(root.host);
    node = root.host;
  }
  if (top === el || el.contains(top) || top.contains(el) || hosts.indexOf(top) !== -1) return result;
  result.covered = true;
  result.by = String(top.tagName || "").toLowerCase();
  const text = (top.innerText || "").replace(/\\s+/g, " ").trim();
  if (text) result.by += ": " + text.slice(0, 80);
  return result;
}
"""

#: Text search that also looks inside open shadow roots and same-origin iframes.
_JS_HAS_TEXT = """
(needle) => {
  const hay = [];
  const walk = (doc) => {
    try { if (doc.body) hay.push(doc.body.innerText || doc.body.textContent || ""); } catch (e) {}
    try {
      doc.querySelectorAll("*").forEach((el) => { if (el.shadowRoot) walk(el.shadowRoot); });
    } catch (e) {}
    try {
      doc.querySelectorAll("iframe").forEach((f) => { if (f.contentDocument) walk(f.contentDocument); });
    } catch (e) {}
  };
  walk(document);
  const joined = hay.join(" ");
  return joined.indexOf(needle) !== -1;
}
"""

#: Shared helper: hand an id to a node the sniffer has not seen (used by the two
#: snippets below so that *every* addressable node still carries the same
#: attribute - the Python side then keeps using the generated selector form).
#: Ids only ever grow, mirroring the sniffer, so a fresh id can never collide
#: with one already on the page.
_JS_ASSIGN_ID = """
  const webpilotAssignId = (node) => {
    let highest = 0;
    let marked = [];
    try { marked = document.querySelectorAll('[data-webpilot-id]'); } catch (e) { marked = []; }
    for (let i = 0; i < marked.length; i++) {
      const value = parseInt(marked[i].getAttribute('data-webpilot-id'), 10);
      if (!isNaN(value) && value > highest) highest = value;
    }
    try {
      const stored = parseInt(window.sessionStorage.getItem('__webpilotId'), 10);
      if (!isNaN(stored) && stored > highest) highest = stored;
    } catch (e) { /* storage may be unavailable */ }
    const id = highest + 1;
    try { node.setAttribute('data-webpilot-id', String(id)); } catch (e) { return null; }
    try { window.sessionStorage.setItem('__webpilotId', String(id)); } catch (e) {}
    return id;
  };
"""

_JS_FOCUSED = (
    """
() => {"""
    + _JS_ASSIGN_ID
    + """
  let el = document.activeElement;
  let hops = 0;
  while (el && el.shadowRoot && el.shadowRoot.activeElement && hops++ < 10) {
    el = el.shadowRoot.activeElement;
  }
  if (!el || el === document.body || el === document.documentElement) return null;
  let id = parseInt(el.getAttribute('data-webpilot-id'), 10);
  if (!id || isNaN(id)) id = webpilotAssignId(el);
  if (!id) return null;
  return {id: id, tag: String(el.tagName || "").toLowerCase()};
}
"""
)

#: Finds a way out of an overlay and marks it with a fresh sniffer id, so the
#: Python side keeps using the one generated selector form.
_JS_CLOSE_TARGET = (
    """
(point) => {"""
    + _JS_ASSIGN_ID
    + """
  const top = document.elementFromPoint(point.x, point.y);
  if (!top) return null;
  const label = (node) => String(
    node.getAttribute("aria-label") || node.innerText || node.value || node.title || ""
  ).replace(/\\s+/g, " ").trim();
  let scope = top;
  for (let depth = 0; scope && depth < 7; depth++, scope = scope.parentElement) {
    let nodes = [];
    try {
      nodes = Array.prototype.slice.call(
        scope.querySelectorAll('button, [role="button"], a[href], summary, input[type="button"], input[type="submit"]')
      );
    } catch (e) { nodes = []; }
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      const text = label(node);
      if (!text) continue;
      if (!/(close|dismiss|no thanks|not now|later|skip|cancel|decline|got it|understood|no\\b|accept)/i.test(text)) continue;
      let id = parseInt(node.getAttribute('data-webpilot-id'), 10);
      if (!id || isNaN(id)) id = webpilotAssignId(node);
      if (!id) continue;
      return {id: id, text: text.slice(0, 80)};
    }
  }
  return null;
}
"""
)


# --------------------------------------------------------------------------- #
# in-page (DOM) dispatch - the fallback for lost synthetic input
# --------------------------------------------------------------------------- #
#
# **Why these exist.**  Playwright's synthetic input is not always delivered to
# the page.  With ``launch_persistent_context()`` (which this project uses for
# persistent sessions) some environments - observed on this machine, headful,
# with the bundled Chromium *and* with ``channel="chrome"`` - drop it silently:
# ``page.mouse.click(x, y)`` and ``page.keyboard.press(...)`` produce zero DOM
# events, while a JS-level ``element.click()`` executed inside the page works
# normally.  ``launch()`` (non-persistent) is unaffected, so this is a browser
# build/environment defect, not something our action code can cause.
#
# The executor therefore *always* tries the native, human-like path first and
# verifies it exactly as before.  Only when that verification shows **no
# observable effect** (and the element is still present and actionable) does it
# retry the action once with the dispatchers below, which build the event objects
# inside the page and dispatch them from there.  That is why every result carries
# ``input_method`` (``"native"`` or ``"dom_fallback"``): the agent - and a human
# reading the transcript - must never be told a fallback click was a real one.
#
# The trade-off, stated plainly: the trigger is "no effect within the settle
# window", so a page whose reaction is slower than that window can receive one
# extra in-page dispatch of an action that did land.  The retry is therefore
# capped at one attempt per action and stays behind the same verification,
# resolution and safety checks as the native path.
#
# These snippets are pure engine code: they know nothing about any site.

#: Click sequence at the element's box centre, dispatched inside the page.
#: ``document.elementFromPoint`` retargets to the shadow *host*, so the hit test
#: descends through open shadow roots to find the node a human would really hit.
_JS_DOM_CLICK = """
(el, args) => {
  const label = (node) => String(
    node.getAttribute("aria-label") || node.innerText || node.value || node.title || ""
  ).replace(/\\s+/g, " ").trim().slice(0, 60);
  const describe = (node) => {
    if (!node) return "nothing";
    const tag = String(node.tagName || "").toLowerCase();
    const text = label(node);
    return text ? tag + " " + JSON.stringify(text) : tag;
  };
  const hostChain = (node) => {
    const chain = [];
    let current = node;
    let hops = 0;
    while (current && hops++ < 10) {
      let root = null;
      try { root = current.getRootNode ? current.getRootNode() : null; } catch (e) { root = null; }
      if (!root || !root.host) break;
      chain.push(root.host);
      current = root.host;
    }
    return chain;
  };
  const related = (candidate, node) => {
    if (!candidate || !node) return false;
    if (candidate === node || node.contains(candidate) || candidate.contains(node)) return true;
    // inside a shadow tree the hit test stops at the host, which is not a
    // light-DOM ancestor - walk the host chain before declaring a miss.
    return hostChain(node).indexOf(candidate) !== -1;
  };
  const deepHit = (x, y) => {
    let node = null;
    try { node = document.elementFromPoint(x, y); } catch (e) { node = null; }
    let hops = 0;
    while (node && node.shadowRoot && hops++ < 10) {
      let inner = null;
      try { inner = node.shadowRoot.elementFromPoint(x, y); } catch (e) { inner = null; }
      if (!inner || inner === node) break;
      node = inner;
    }
    return node;
  };
  const rect = el.getBoundingClientRect();
  // The box centre the caller computed is relative to the *main* frame viewport,
  // which is wrong inside an iframe; the element's own rect is always in this
  // document's coordinates, so it is the fallback when the first point misses.
  const candidates = [[Number(args.cx), Number(args.cy)],
                      [rect.left + rect.width / 2, rect.top + rect.height / 2]];
  let hit = null;
  let cx = candidates[1][0];
  let cy = candidates[1][1];
  for (let i = 0; i < candidates.length; i++) {
    const x = candidates[i][0];
    const y = candidates[i][1];
    if (!isFinite(x) || !isFinite(y)) continue;
    const node = deepHit(x, y);
    if (related(node, el)) { hit = node; cx = x; cy = y; break; }
  }
  if (!hit) hit = el;   // nothing recognisable at either point: dispatch on the element itself
  const base = {
    bubbles: true, cancelable: true, composed: true, view: window,
    clientX: cx, clientY: cy, screenX: cx, screenY: cy, detail: 1,
  };
  const send = (type, extra) => {
    const options = Object.assign({}, base, extra || {});
    let event = null;
    try {
      event = (type.indexOf("pointer") === 0 && typeof PointerEvent === "function")
        ? new PointerEvent(type, options)
        : new MouseEvent(type, options);
    } catch (e) { return false; }
    try { return hit.dispatchEvent(event); } catch (e) { return false; }
  };
  const steps = [
    ["pointerdown", {button: 0, buttons: 1, pointerId: 1, isPrimary: true, pointerType: "mouse"}],
    ["mousedown", {button: 0, buttons: 1}],
    ["pointerup", {button: 0, buttons: 0, pointerId: 1, isPrimary: true, pointerType: "mouse"}],
    ["mouseup", {button: 0, buttons: 0}],
    ["click", {button: 0, buttons: 0}],
  ];
  const dispatched = [];
  for (let i = 0; i < steps.length; i++) {
    send(steps[i][0], steps[i][1]);
    dispatched.push(steps[i][0]);
  }
  return {hit: describe(hit), target: describe(el), dispatched: dispatched,
          x: Math.round(cx), y: Math.round(cy)};
}
"""

#: Fill a field the way a framework would notice: value through the element's own
#: ``value`` setter (so React/Vue value trackers see the change), then ``input`` +
#: ``change``, then the keystrokes, then ``blur``.  ``hardSubmit`` calls
#: ``form.requestSubmit()`` as the last resort - it is only requested when there is
#: evidence that the native input never reached the page, because a form that *did*
#: receive the native Enter must not be submitted twice.
_JS_DOM_TYPE = """
(el, args) => {
  const text = String(args.text === null || args.text === undefined ? "" : args.text);
  const clear = !!args.clear;
  const submit = !!args.submit;
  const hardSubmit = !!args.hardSubmit;
  const setValue = (value) => {
    let proto = null;
    try { proto = (el.constructor && el.constructor.prototype) || Object.getPrototypeOf(el); }
    catch (e) { proto = null; }
    let hops = 0;
    while (proto && hops++ < 6) {
      let desc = null;
      try { desc = Object.getOwnPropertyDescriptor(proto, "value"); } catch (e) { desc = null; }
      if (desc && typeof desc.set === "function") {
        desc.set.call(el, value);
        return "native-setter";
      }
      proto = Object.getPrototypeOf(proto);
    }
    try { el.value = value; return "direct"; } catch (e) { return "none"; }
  };
  const fire = (type, options) => {
    try { return el.dispatchEvent(new Event(type, options || {bubbles: true, composed: true})); }
    catch (e) { return false; }
  };
  let focused = false;
  try { el.focus({preventScroll: true}); focused = (el.ownerDocument.activeElement === el); } catch (e) {}
  let before = "";
  try { before = el.value === undefined || el.value === null ? "" : String(el.value); } catch (e) {}
  const how = setValue(clear ? text : before + text);
  fire("input", {bubbles: true, composed: true});
  fire("change", {bubbles: true, composed: true});
  let keys = 0;
  const sendKey = (type, key, extra) => {
    const options = Object.assign(
      {bubbles: true, cancelable: true, composed: true, view: window, key: key, code: key},
      extra || {}
    );
    let event = null;
    try { event = new KeyboardEvent(type, options); } catch (e) { return false; }
    let target = el;
    try { target = el.ownerDocument.activeElement || el; } catch (e) { target = el; }
    try { target.dispatchEvent(event); } catch (e) { return false; }
    keys += 1;
    return true;
  };
  for (let i = 0; i < text.length; i++) {
    const ch = text.charAt(i);
    const code = ch.charCodeAt(0);
    sendKey("keydown", ch, {charCode: code, keyCode: code, which: code});
    sendKey("keypress", ch, {charCode: code, keyCode: code, which: code});
    sendKey("keyup", ch, {charCode: code, keyCode: code, which: code});
  }
  let submitted = false;
  let form = null;
  try { form = el.form || (el.closest ? el.closest("form") : null); } catch (e) { form = null; }
  if (submit) {
    sendKey("keydown", "Enter", {charCode: 13, keyCode: 13, which: 13});
    sendKey("keypress", "Enter", {charCode: 13, keyCode: 13, which: 13});
    sendKey("keyup", "Enter", {charCode: 13, keyCode: 13, which: 13});
    if (hardSubmit && form && typeof form.requestSubmit === "function") {
      try { form.requestSubmit(); submitted = true; } catch (e) { submitted = false; }
    }
  }
  try { el.blur(); } catch (e) {}
  let after = "";
  try { after = el.value === undefined || el.value === null ? "" : String(el.value); } catch (e) {}
  return {how: how, focused: focused, keys: keys, submitted: submitted,
          hasForm: !!form, valueLen: after.length};
}
"""

#: Keydown/keypress/keyup on the focused element (or body); Enter additionally
#: asks the closest form to submit itself when the native key never arrived.
_JS_DOM_PRESS = """
(args) => {
  const key = String(args.key === null || args.key === undefined ? "" : args.key);
  let el = document.activeElement;
  let hops = 0;
  while (el && el.shadowRoot && el.shadowRoot.activeElement && hops++ < 10) {
    el = el.shadowRoot.activeElement;
  }
  const target = el || document.body || document.documentElement;
  const keys = [];
  const send = (type, extra) => {
    const options = Object.assign(
      {bubbles: true, cancelable: true, composed: true, view: window, key: key, code: key},
      extra || {}
    );
    let event = null;
    try { event = new KeyboardEvent(type, options); } catch (e) { return; }
    try { target.dispatchEvent(event); } catch (e) { return; }
    keys.push(type);
  };
  send("keydown");
  send("keypress");
  send("keyup");
  let submitted = false;
  let form = null;
  if (key === "Enter") {
    try { form = target.form || (target.closest ? target.closest("form") : null); } catch (e) { form = null; }
    if (!!args.hardSubmit && form && typeof form.requestSubmit === "function") {
      try { form.requestSubmit(); submitted = true; } catch (e) { submitted = false; }
    }
  }
  const text = String(
    (target.getAttribute && target.getAttribute("aria-label")) || target.innerText || target.value || ""
  ).replace(/\\s+/g, " ").trim().slice(0, 60);
  const tag = String(target.tagName || "").toLowerCase();
  return {target: text ? tag + " " + JSON.stringify(text) : tag, keys: keys,
          hasForm: !!form, submitted: submitted};
}
"""


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #

class _ActionFailure(Exception):
    """Internal: an action failed in a way the agent should hear about."""

    def __init__(self, summary: str, *, hint: str = "", error: str | None = None) -> None:
        super().__init__(summary)
        self.summary = summary
        self.hint = hint
        self.error = error or summary


@dataclass
class _State:
    """Hashable page state used for before/after verification."""

    url: str = ""
    title: str = ""
    digest: str = ""
    dialogs: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Fallback:
    """Outcome of the one in-page (DOM) retry allowed per action.

    ``attempted`` is False when the fallback declined to run (element gone,
    disabled, covered, credential field) - ``refused`` then says why, so the
    failure the agent sees stays the honest, precise one.
    """

    attempted: bool = False        # were events dispatched inside the page?
    succeeded: bool = False        # did the verification actually see the effect?
    note: str = ""                 # one-line explanation for the summary
    refused: str = ""              # why the fallback did not run
    hint: str = ""                 # recovery hint that overrides the default one
    stage: str = ""                # sequence | element.click() | value-setter | key-events
    detail: dict[str, Any] = field(default_factory=dict)
    state: "_State | None" = None  # page state measured after the retry
    page_changed: bool = False
    dialogs: int = 0
    value: str | None = None
    submitted: bool = False

    def describe(self) -> dict[str, Any]:
        """Compact, JSON-able description for ``ToolResult.data``."""
        info: dict[str, Any] = {"mechanism": self.stage or None, "dispatched": self.attempted}
        if self.refused:
            info["refused"] = self.refused
        info.update(self.detail)
        return info


class ActionExecutor:
    """Executes the browser tools defined in ``docs/INTERFACES.md``."""

    def __init__(self, session: BrowserSession, config: Any) -> None:
        self.session = session
        self.config = config

    # ------------------------------------------------------------------ #
    # entry point
    # ------------------------------------------------------------------ #
    def execute(self, call: ToolCall) -> ToolResult:
        """Run ``call`` and always return a :class:`ToolResult` (never raises)."""
        if not isinstance(call, ToolCall):  # pragma: no cover - defensive
            return ToolResult(ok=False, summary="malformed tool call", error="expected a ToolCall",
                              recovery_hint="call the tool with a name and an arguments object")
        name = str(call.name or "")
        args = call.args if isinstance(call.args, dict) else {}
        if name in AGENT_LOCAL_TOOLS:
            return ToolResult(
                ok=False,
                summary=f"{name} is handled by the agent loop, not by the browser executor",
                error="not a browser tool",
                recovery_hint=f"do not route {name} through ActionExecutor",
            )
        handler: Callable[[dict[str, Any]], ToolResult] | None = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return ToolResult(
                ok=False,
                summary=f"unknown tool {name!r}",
                error="unknown tool",
                recovery_hint="use one of: page_outline, goto, click, type_text, press_key, "
                              "scroll, go_back, wait_for, tabs, screenshot",
            )
        try:
            return handler(args)
        except _ActionFailure as failure:
            return self._failure(failure.summary, failure.hint, failure.error)
        except BrowserError as exc:
            return self._failure(str(exc), "the browser could not complete that",
                                 error=exc.__class__.__name__)
        except Exception as exc:  # pragma: no cover - defensive: never raise into the loop
            return self._failure(f"{name} failed unexpectedly: {exc}",
                                 "retry once, then ask for a fresh page outline",
                                 error=exc.__class__.__name__)

    # ------------------------------------------------------------------ #
    # tools
    # ------------------------------------------------------------------ #
    def _tool_page_outline(self, args: dict[str, Any]) -> ToolResult:
        mode = self._snapshot_mode(args.get("mode"))
        needle = str(args.get("filter") or "").strip()
        model = self._model(mode=mode)
        if model is None:  # pragma: no cover - defensive
            raise _ActionFailure("could not read the page", hint="the tab may have been closed")
        if needle:
            model = self._filter_model(model, needle)
        summary = (
            f"page outline ({mode}) for {model.url or 'unknown url'}: "
            f"{len(model.elements)} elements, {len(model.text)} chars of text"
            + (f", filtered by {needle!r}" if needle else "")
        )
        return ToolResult(ok=True, summary=summary, page=model,
                          data={"mode": mode, "filter": needle, "elements": len(model.elements)})

    def _tool_goto(self, args: dict[str, Any]) -> ToolResult:
        raw = str(args.get("url") or "").strip()
        if not raw:
            raise _ActionFailure("goto needs a url", hint="pass an absolute or page-relative url")
        url = self._normalise_url(raw)
        before = self._state()
        try:
            self.session.goto(url)
        except BrowserError as exc:
            raise _ActionFailure(f"could not open {url}: {exc}",
                                 hint="check the spelling, or go back to a page you were on",
                                 error="navigation failed") from exc
        after = self._await_change(before, timeout_ms=self.config.nav_timeout_ms // 4)
        model = self._model(mode="full")
        changed = after.digest != before.digest
        return ToolResult(
            ok=True,
            summary=f"opened {after.url or url} (title: {after.title or 'untitled'})",
            page=model,
            page_changed=changed,
            data={"url": after.url, "title": after.title},
        )

    def _tool_click(self, args: dict[str, Any]) -> ToolResult:
        element_id = self._element_id(args, "element_id")
        meta = self._meta(element_id)
        if meta is not None and meta.disabled:
            raise _ActionFailure(
                f"element [{element_id}] {meta.role} {meta.name!r} is disabled",
                hint="element disabled - it cannot be clicked until a form makes it active",
            )
        before = self._state()
        loc, frame_index = self._resolve(element_id)
        state_before = self._element_state(loc)
        self._scroll_into_view(loc, element_id)

        covered = self._cover(loc)
        recovered: str | None = None
        if covered.get("covered"):
            recovered = self._recover_from_overlay(loc, covered)
            if recovered is None:
                raise _ActionFailure(
                    f"click on [{element_id}] was intercepted by another element "
                    f"({covered.get('by')})",
                    hint="overlay intercepts clicks - close it first (its close control is an "
                         "element id in the next page outline)",
                )
            self._scroll_into_view(loc, element_id)
            covered = self._cover(loc)
            if covered.get("covered"):
                raise _ActionFailure(
                    f"click on [{element_id}] is still intercepted by {covered.get('by')}",
                    hint="overlay intercepts clicks - dismiss the overlay before clicking behind it",
                )

        box = loc.bounding_box()
        if not box:
            raise _ActionFailure(
                f"element [{element_id}] has no visible box",
                hint="element is not visible - scroll to it or take a fresh page outline",
            )
        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        try:
            self._native_mouse_click(cx, cy)
        except Exception as exc:
            raise _ActionFailure(f"could not click [{element_id}]: {exc}",
                                 hint="the tab may be busy; wait a moment and retry") from exc

        after = self._await_change(before, timeout_ms=self.config.settle_timeout_ms)
        dialogs = self.session.dialog_total - before.dialogs
        page_moved = after.digest != before.digest
        toggled = False
        if not page_moved and not dialogs:
            # Only ask the *element* whether it toggled when the page itself did
            # not move; after a navigation the locator is detached and any read
            # would just burn the action timeout.
            toggled = self._state_changed(state_before, self._element_state(loc))
        changed = page_moved or dialogs > 0 or toggled
        input_method = "native"
        fallback = _Fallback()

        label = self._describe(element_id, meta)
        if not changed:
            # The native click was verified to have had *no* observable effect.
            # Either the page really ignored the click, or this environment's
            # Chromium never delivered it - one in-page retry tells the two apart.
            fallback = self._dom_click_fallback(element_id, meta)
            if fallback.succeeded:
                after = fallback.state or after
                dialogs = fallback.dialogs
                input_method = "dom_fallback"
                changed = True
            else:
                return self._click_failed(label, after, covered, fallback, element_id)

        notes = []
        if recovered:
            notes.append(f"dismissed an overlay first ({recovered})")
        if dialogs:
            notes.append(f"{dialogs} native dialog(s) were answered")
        if toggled and not page_moved:
            notes.append("the control toggled")
        model = self._model(mode="full")
        url = self._live_url() or after.url or "same url"
        summary = f"clicked {label}"
        if input_method == "dom_fallback":
            summary += f" ({fallback.note})"
        summary += f" -> page changed ({url})"
        if notes:
            summary += "; " + "; ".join(notes)
        data: dict[str, Any] = {"element_id": element_id, "input_method": input_method,
                                "url": url, "dialogs": dialogs, "recovered": recovered}
        if input_method == "dom_fallback":
            data["input_fallback"] = fallback.describe()
        return ToolResult(ok=True, summary=summary, page=model, page_changed=True, data=data)

    def _click_failed(
        self, label: str, after: _State, covered: dict[str, Any], fallback: _Fallback,
        element_id: int,
    ) -> ToolResult:
        """The click did nothing natively *and* nothing in the page: report it."""
        model = self._model(mode="delta")
        summary = f"clicked {label} but the page did not change"
        if fallback.attempted:
            summary += (" (the browser did not deliver the synthetic mouse event and the "
                        "in-page dispatch had no effect either)")
        elif fallback.refused:
            summary += f" (the in-page fallback was not used: {fallback.refused})"
        data: dict[str, Any] = {
            "element_id": element_id, "url": after.url, "covered_by": covered.get("by"),
            "input_method": "native", "fallback_attempted": fallback.attempted,
            "fallback_note": fallback.note or fallback.refused,
        }
        if fallback.attempted:
            data["input_fallback"] = fallback.describe()
        return ToolResult(
            ok=False,
            summary=summary,
            error="no effect",
            recovery_hint=fallback.hint or (
                "nothing changed - the control may need a different element, "
                "or the page needs waiting (try wait_for)"
            ),
            page=model,
            data=data,
        )

    def _tool_type_text(self, args: dict[str, Any]) -> ToolResult:
        element_id = self._element_id(args, "element_id", allow_zero=True)
        text = str(args.get("text") if args.get("text") is not None else "")
        submit = bool(args.get("submit", False))
        clear = bool(args.get("clear", True))
        meta = self._meta(element_id) if element_id else None
        trust_live = False

        if element_id == 0:
            loc = None
            focused = self._focused_element()
            if focused is None:
                raise _ActionFailure(
                    "no element is focused, so element_id 0 has no target",
                    hint="pass the id of the field you want to type into",
                )
            element_id = int(focused["id"])
            loc, _ = self._resolve(element_id, trust_live=True)
            trust_live = True
            meta = self._meta(element_id)
        else:
            if meta is not None and meta.disabled:
                raise _ActionFailure(f"element [{element_id}] is disabled",
                                     hint="element disabled - enable it first")
            loc, _ = self._resolve(element_id)

        state = self._element_state(loc)
        if self._is_password(state, meta):
            return ToolResult(
                ok=False,
                summary=f"refused to type into the password field [{element_id}]",
                error="password field",
                recovery_hint="password fields are filled by the human",
                data={"element_id": element_id},
            )

        before = self._state()
        self._scroll_into_view(loc, element_id)
        probe = self._probe_timeout_ms()
        try:
            loc.click(timeout=probe)
        except Exception:
            pass  # focus is best effort; the sequential typing still lands
        if clear:
            try:
                self._native_locator_press(loc, "ControlOrMeta+a", timeout=probe)
                self._native_locator_press(loc, "Backspace", timeout=probe)
            except Exception:
                pass
        delay = max(0, int(self.config.typing_delay_ms))
        if text:
            try:
                self._native_type(loc, text, delay)
            except Exception as exc:
                raise _ActionFailure(f"could not type into [{element_id}]: {exc}",
                                     hint="the field may be readonly or replaced by the page") from exc

        typed_note = ""
        if submit:
            tag = str(state.get("tag") or "")
            if tag == "textarea":
                typed_note = " (Enter not sent: it would add a line break in a textarea)"
            else:
                try:
                    self._native_locator_press(loc, "Enter")
                except Exception as exc:
                    raise _ActionFailure(f"typed into [{element_id}] but could not submit: {exc}",
                                         hint="press the submit control with click instead") from exc

        after = self._await_change(before, timeout_ms=self.config.settle_timeout_ms)
        dialogs = self.session.dialog_total - before.dialogs
        page_moved = after.digest != before.digest
        has_value = ""
        accepted = True
        if not page_moved and not dialogs:
            state_after = self._element_state(loc)
            has_value = state_after.get("value") or ""
            accepted = (text in has_value) if text else has_value == ""
            if clear and not text:
                accepted = has_value == ""
        changed = page_moved or dialogs > 0
        input_method = "native"
        fallback: _Fallback | None = None

        if not changed and (not accepted or submit):
            # Native typing is verified to have had no effect (or a requested
            # submit changed nothing): retry once inside the page.  ``hard_submit``
            # is only passed on when the native keystrokes demonstrably never
            # reached the field, so a form that did receive them is never
            # submitted a second time.
            fallback = self._dom_type_fallback(
                element_id, meta, text, clear, submit,
                trust_live=trust_live, hard_submit=(submit and not accepted),
            )

        label = self._describe(element_id, meta)
        if fallback is not None and fallback.attempted and fallback.succeeded:
            changed = bool(fallback.page_changed)
            model = self._model(mode="full" if changed else "delta")
            summary = f"typed into {label} ({fallback.note})"
            if changed:
                url = self._live_url() or (fallback.state.url if fallback.state else after.url)
                summary += f" -> page changed ({url or 'same url'})"
            elif submit:
                summary += "; the submit did not change the page"
            if typed_note:
                summary += typed_note
            data: dict[str, Any] = {"element_id": element_id, "submitted": submit,
                                    "dialogs": fallback.dialogs, "input_method": "dom_fallback",
                                    "input_fallback": fallback.describe()}
            return ToolResult(ok=True, summary=summary, page=model, page_changed=changed,
                              data=data)

        if not accepted and not changed:
            raise _ActionFailure(
                f"the field [{element_id}] did not keep the typed value",
                hint="form rejected the value - check the field constraints in the page outline",
            )
        if fallback is not None and fallback.attempted and submit and not changed:
            raise _ActionFailure(
                f"typed into [{element_id}] but the form did not submit",
                hint="the form rejected the value or did not navigate - press the submit "
                     "control with click instead",
            )
        if not changed and not submit:
            model = self._model(mode="delta")
            return ToolResult(ok=True, summary=f"typed into {label} (field value updated)",
                              page=model, page_changed=False,
                              data={"element_id": element_id, "value_len": len(has_value),
                                    "input_method": input_method})
        model = self._model(mode="full")
        summary = f"typed into {label}" + (" and submitted" if submit else "")
        if changed:
            summary += f" -> page changed ({after.url or 'same url'})"
        if typed_note:
            summary += typed_note
        return ToolResult(ok=True, summary=summary, page=model, page_changed=changed,
                          data={"element_id": element_id, "submitted": submit,
                                "dialogs": dialogs, "input_method": input_method})

    def _tool_press_key(self, args: dict[str, Any]) -> ToolResult:
        key = str(args.get("key") or "").strip()
        if not key:
            raise _ActionFailure("press_key needs a key", hint="e.g. Enter, Escape, Tab, PageDown")
        before = self._state()
        try:
            self._native_press(key)
        except Exception as exc:
            raise _ActionFailure(f"could not press {key!r}: {exc}",
                                 hint="use a Playwright key name such as Enter or ArrowDown") from exc
        after = self._await_change(before, timeout_ms=self.config.settle_timeout_ms)
        changed = after.digest != before.digest or self.session.dialog_total > before.dialogs
        input_method = "native"
        fallback = _Fallback()
        if not changed:
            # The native key was verified to have had no effect: the browser may
            # simply not have delivered it - retry once inside the page.
            fallback = self._dom_press_fallback(key)
            if fallback.succeeded:
                after = fallback.state or after
                changed = True
                input_method = "dom_fallback"
        model = self._model(mode="full" if changed else "delta")
        if not changed:
            summary = f"pressed {key!r} but the page did not react"
            if fallback.attempted:
                summary += (" (the browser did not deliver the synthetic key event and the "
                            "in-page dispatch had no effect either)")
            return ToolResult(
                ok=False, summary=summary,
                error="no effect", page=model,
                recovery_hint="nothing changed - try clicking the element first, or use wait_for",
                data={"key": key, "input_method": "native",
                      "fallback_attempted": fallback.attempted},
            )
        summary = f"pressed {key!r}"
        url = self._live_url() or after.url or "same url"
        data: dict[str, Any] = {"key": key, "url": url, "input_method": input_method}
        if input_method == "dom_fallback":
            summary += f" ({fallback.note})"
            data["input_fallback"] = fallback.describe()
        summary += f" -> page changed ({url})"
        return ToolResult(ok=True, summary=summary, page=model, page_changed=True, data=data)

    def _tool_scroll(self, args: dict[str, Any]) -> ToolResult:
        direction = str(args.get("direction") or "down").strip().lower()
        if direction not in ("up", "down", "top", "bottom"):
            raise _ActionFailure(f"unknown scroll direction {direction!r}",
                                 hint="direction must be up, down, top or bottom")
        try:
            amount = int(args.get("amount") or 0) or int(self._state().raw.get("viewportH") or 800)
        except (TypeError, ValueError):
            amount = 800
        before = self._state()
        raw = before.raw
        max_y = int(raw.get("scrollMax") or 0)
        y = int(raw.get("scrollY") or 0)
        if direction == "top":
            target = 0
        elif direction == "bottom":
            target = max_y
        elif direction == "up":
            target = max(0, y - amount)
        else:
            target = min(max_y, y + amount)
        try:
            self.session.page.evaluate(
                "(y) => window.scrollTo({top: y, left: window.scrollX || 0, behavior: 'instant'})", target
            )
        except Exception as exc:
            raise _ActionFailure(f"could not scroll: {exc}", hint="the page may be closing") from exc
        after = self._await_change(before, timeout_ms=self.config.settle_timeout_ms)
        model = self._model(mode="full")
        moved = int(after.raw.get("scrollY") or 0) != y
        if not moved:
            reason = "the page is not scrollable" if max_y <= 0 else "already at the requested position"
            return ToolResult(
                ok=False,
                summary=f"scroll {direction} did nothing: {reason}",
                error="no effect", page=model,
                recovery_hint="no more content in that direction - look at the page outline instead",
                data={"direction": direction, "scroll_y": y, "max_y": max_y},
            )
        return ToolResult(
            ok=True,
            summary=f"scrolled {direction} to {after.raw.get('scrollY')}/{max_y}",
            page=model, page_changed=True,
            data={"direction": direction, "scroll_y": after.raw.get("scrollY"), "max_y": max_y},
        )

    def _tool_go_back(self, args: dict[str, Any]) -> ToolResult:
        before = self._state()
        try:
            self.session.page.go_back(wait_until="load", timeout=self.config.nav_timeout_ms)
        except Exception:
            pass  # go_back times out when there is no history entry; the state check decides
        after = self._await_change(before, timeout_ms=self.config.settle_timeout_ms)
        model = self._model(mode="full" if after.digest != before.digest else "delta")
        if after.url == before.url:
            return ToolResult(
                ok=False, summary="browser back did nothing (no earlier page in this tab)",
                error="no effect", page=model,
                recovery_hint="there is no history to go back to - use goto with a url from the "
                              "page outline",
                data={"url": after.url},
            )
        return ToolResult(ok=True, summary=f"went back to {after.url}", page=model,
                          page_changed=True, data={"url": after.url})

    def _tool_wait_for(self, args: dict[str, Any]) -> ToolResult:
        needle = str(args.get("text") or "")
        seconds = float(args.get("seconds") or 0)
        timeout = float(args.get("timeout_seconds") or 0) or max(seconds, 5.0)
        timeout = min(timeout, _MAX_WAIT_S)
        seconds = min(max(seconds, 0.0), _MAX_WAIT_S)
        started = time.monotonic()
        if needle:
            deadline = started + timeout
            found = False
            while time.monotonic() < deadline:
                if self._page_has_text(needle):
                    found = True
                    break
                self._sleep(0.1)
            waited = round(time.monotonic() - started, 2)
            model = self._model(mode="full" if found else "delta")
            if not found:
                return ToolResult(
                    ok=False,
                    summary=f"waited {waited}s but the text {needle[:60]!r} never appeared",
                    error="timeout", page=model,
                    recovery_hint="the text may be worded differently, appear later, or need an "
                                  "interaction - re-read the page outline",
                    data={"waited_s": waited, "text": needle},
                )
            return ToolResult(ok=True, summary=f"text {needle[:60]!r} appeared after {waited}s",
                              page=model, page_changed=True,
                              data={"waited_s": waited, "text": needle})
        self._sleep(seconds)
        model = self._model(mode="delta")
        return ToolResult(ok=True, summary=f"waited {seconds:.1f}s",
                          page=model, page_changed=False, data={"waited_s": seconds})

    def _tool_tabs(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "list").strip().lower()
        if action == "list":
            infos = self.session.tab_infos()
            lines = [f"#{t.index}{'*' if t.active else ''} {t.title or '(untitled)'} {t.url}" for t in infos]
            return ToolResult(ok=True, summary=f"{len(infos)} tab(s): " + " | ".join(lines)[:400],
                              data={"tabs": [t.__dict__ for t in infos]})
        if action == "open":
            url = str(args.get("url") or "").strip()
            before = self._state()
            self.session.new_page(self._normalise_url(url) if url else None)
            after = self._await_change(before, timeout_ms=self.config.settle_timeout_ms)
            model = self._model(mode="full")
            return ToolResult(ok=True, summary=f"opened a new tab at {after.url or 'about:blank'}",
                              page=model, page_changed=True,
                              data={"url": after.url, "index": self.session.active_index})
        if action == "switch":
            index = self._index(args.get("index"))
            try:
                self.session.switch(index)
            except BrowserError as exc:
                raise _ActionFailure(f"could not switch to tab {index}: {exc}",
                                     hint="call tabs(list) to see which indexes exist") from exc
            model = self._model(mode="full")
            return ToolResult(ok=True, summary=f"switched to tab {index} ({model.url if model else ''})",
                              page=model, page_changed=True, data={"index": index})
        if action == "close":
            index = self._index(args.get("index"), default=self.session.active_index)
            before_total = len(self.session.pages)
            self.session.close_page(index)
            model = self._model(mode="full")
            return ToolResult(
                ok=True,
                summary=f"closed tab {index} ({before_total - len(self.session.pages)} closed)",
                page=model, page_changed=True,
                data={"index": index, "open": len(self.session.pages)},
            )
        raise _ActionFailure(f"unknown tabs action {action!r}",
                             hint="action must be list, open, switch or close")

    def _tool_screenshot(self, args: dict[str, Any]) -> ToolResult:
        label = str(args.get("label") or "shot")
        try:
            path = self.session.screenshot(label)
        except BrowserError as exc:
            return ToolResult(ok=False, summary=f"screenshot failed: {exc}", error="screenshot failed",
                              recovery_hint="screenshots may be disabled by configuration")
        return ToolResult(ok=True, summary=f"screenshot saved to {path}", screenshot_path=path,
                          data={"path": path})

    # ------------------------------------------------------------------ #
    # element resolution & interaction helpers
    # ------------------------------------------------------------------ #
    def _resolve(self, element_id: int, *, trust_live: bool = False) -> tuple[Any, int]:
        """Find the locator for ``element_id``, preferring its recorded frame.

        Only ids that came out of a snapshot the agent has seen are resolvable:
        an id from a previous page is refused with a hint instead of silently
        hitting whatever node now happens to carry that number.  ``trust_live``
        skips that check for ids this module assigned to the page itself moments
        ago (the focused element, an overlay's close control).
        """
        if element_id <= 0:
            raise _ActionFailure(f"element id {element_id} is not a valid target",
                                 hint="use an id from the page outline (page_outline)")
        model = self.session.last_model
        if not trust_live and model is not None and model.by_id(element_id) is None:
            raise _ActionFailure(
                f"element [{element_id}] is not part of the current page snapshot",
                hint="that id is stale (the page moved on) - take a fresh page_outline",
            )
        page = self.session.page
        selector = selector_for(element_id)
        frames = list(page.frames)
        hint = 0
        meta = self._meta(element_id)
        if meta is not None:
            hint = int(meta.frame or 0)
        order = ([hint] if 0 <= hint < len(frames) else []) + [
            i for i in range(len(frames)) if i != hint
        ]
        for index in order:
            frame = frames[index]
            try:
                locator = frame.locator(selector)
                if locator.count() > 0:
                    return locator.first, index
            except Exception:
                continue
        raise _ActionFailure(
            f"element [{element_id}] is no longer on the page",
            hint="the page moved on - take a fresh page_outline and use a current id",
        )

    # ------------------------------------------------------------------ #
    # native input: thin wrappers, so a test (or a reviewer) can see exactly
    # which Playwright call is the human-like path - and neutralise it
    # ------------------------------------------------------------------ #
    def _native_mouse_click(self, x: float, y: float) -> None:
        """A real mouse click at ``(x, y)`` - the primary path for clicks."""
        self.session.page.mouse.click(x, y)

    def _native_type(self, loc: Any, text: str, delay: int) -> None:
        """Real, key-by-key typing into ``loc`` - the primary path for type_text."""
        loc.press_sequentially(text, delay=delay)

    def _native_locator_press(self, loc: Any, key: str, *, timeout: int | None = None) -> None:
        """Real key press scoped to a locator (clear shortcut, Backspace, Enter)."""
        if timeout is None:
            loc.press(key)
        else:
            loc.press(key, timeout=timeout)

    def _native_press(self, key: str) -> None:
        """A real key press on the page - the primary path for press_key."""
        self.session.page.keyboard.press(key)

    # ------------------------------------------------------------------ #
    # in-page (DOM) fallback: used only when the native input above was
    # verified to have had no observable effect (see the module note)
    # ------------------------------------------------------------------ #
    def _dom_click_fallback(self, element_id: int, meta: Element | None) -> _Fallback:
        """Retry a click by dispatching the mouse sequence *inside* the page.

        Called only after the native ``page.mouse.click`` was verified to have
        produced no observable effect, and only while the element is still
        resolvable, enabled, uncovered and visible - a refused, stale or covered
        action can therefore never be turned into a success by this path.
        """
        try:
            loc, _ = self._resolve(element_id)
        except _ActionFailure as exc:
            return _Fallback(refused=exc.summary, hint=exc.hint)
        self._scroll_into_view(loc, element_id)
        state = self._element_state(loc)
        if state.get("disabled") or (meta is not None and meta.disabled):
            return _Fallback(refused=f"element [{element_id}] is disabled",
                             hint="element disabled - it cannot be clicked until a form makes "
                                  "it active")
        if self._is_password(state, meta):
            # An action that bypasses the browser must not touch credentials.
            return _Fallback(refused="a password field may not be clicked in the page",
                             hint="password fields are filled by the human")
        covered = self._cover(loc)
        if covered.get("covered"):
            return _Fallback(
                refused=f"the element is now covered by {covered.get('by')}",
                hint="overlay intercepts clicks - close it first (its close control is an "
                     "element id in the next page outline)",
            )
        box = loc.bounding_box()
        if not box:
            return _Fallback(refused=f"element [{element_id}] has no visible box",
                             hint="element is not visible - scroll to it or take a fresh "
                                  "page outline")

        mark = self._state()
        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        try:
            info = dict(loc.evaluate(_JS_DOM_CLICK, {"cx": cx, "cy": cy}) or {})
        except Exception as exc:
            return _Fallback(attempted=True, refused=f"the in-page dispatch failed: {exc}",
                             hint="the element may have been replaced - take a fresh page_outline")

        stage = "sequence"
        note = "the browser did not deliver the synthetic mouse event; dispatched in the page"
        changed, after, dialogs = self._fallback_settle(mark, loc, state)
        if not changed:
            # Not every target reacts to a dispatched click (Chromium runs the
            # activation behaviour of some nodes and not others), so finish with
            # the element's own .click() before declaring defeat.
            stage = "element.click()"
            note = ("the browser did not deliver the synthetic mouse event; the in-page "
                    "sequence produced nothing, so element.click() was used instead")
            try:
                loc.evaluate("(el) => { el.click(); }")
            except Exception:
                pass
            changed, after, dialogs = self._fallback_settle(mark, loc, state)
        return _Fallback(attempted=True, succeeded=bool(changed), note=note, stage=stage,
                         detail={"hit": info.get("hit"), "dispatched": info.get("dispatched")},
                         state=after, dialogs=dialogs, page_changed=bool(changed))

    def _dom_type_fallback(
        self, element_id: int, meta: Element | None, text: str, clear: bool, submit: bool,
        *, trust_live: bool = False, hard_submit: bool = False,
    ) -> _Fallback:
        """Retry typing by setting the value and firing events inside the page.

        ``hard_submit`` asks the page to ``form.requestSubmit()``; the caller only
        turns it on when the field is demonstrably still empty, i.e. when the
        native keystrokes never arrived, so a form that already received a real
        Enter is never submitted twice.
        """
        try:
            loc, _ = self._resolve(element_id, trust_live=trust_live)
        except _ActionFailure as exc:
            return _Fallback(refused=exc.summary, hint=exc.hint)
        self._scroll_into_view(loc, element_id)
        state = self._element_state(loc)
        if state.get("disabled") or (meta is not None and meta.disabled):
            return _Fallback(refused=f"element [{element_id}] is disabled",
                             hint="element disabled - enable it first")
        if self._is_password(state, meta):
            # Exactly the native rule, including config.allow_password_typing.
            return _Fallback(refused="a password field may not be filled in the page",
                             hint="password fields are filled by the human")
        mark = self._state()
        try:
            info = dict(loc.evaluate(_JS_DOM_TYPE, {
                "text": text, "clear": clear, "submit": submit, "hardSubmit": hard_submit,
            }) or {})
        except Exception as exc:
            return _Fallback(attempted=True, refused=f"the in-page dispatch failed: {exc}",
                             hint="the field may have been replaced - take a fresh page_outline")

        after = self._await_change(mark, timeout_ms=self.config.settle_timeout_ms)
        dialogs = self.session.dialog_total - mark.dialogs
        page_changed = after.digest != mark.digest or dialogs > 0
        if page_changed:
            after = self._await_settle_url(after)
        value = str((self._element_state(loc) or {}).get("value") or "")
        landed = (text in value) if text else value == ""
        if clear and not text:
            landed = value == ""
        note = ("the browser did not deliver the synthetic keystrokes; "
                "the value was set in the page")
        if info.get("submitted"):
            note += " and the form was submitted in the page"
        return _Fallback(
            attempted=True, succeeded=bool(landed or page_changed), note=note,
            stage="value-setter",
            detail={"set_via": info.get("how"), "key_events": info.get("keys"),
                    "form_submitted": bool(info.get("submitted"))},
            state=after, dialogs=dialogs, page_changed=bool(page_changed),
            value=value, submitted=bool(info.get("submitted")),
        )

    def _dom_press_fallback(self, key: str, *, hard_submit: bool = True) -> _Fallback:
        """Retry a key press by dispatching key events inside the page.

        Unlike typing there is no in-page evidence that the native key was lost,
        so an Enter also asks the closest form to submit itself (the last resort
        the contract asks for); every other key can only be dispatched.
        """
        mark = self._state()
        try:
            info = dict(self.session.page.evaluate(
                _JS_DOM_PRESS, {"key": key, "hardSubmit": hard_submit}
            ) or {})
        except Exception as exc:
            return _Fallback(attempted=True, refused=f"the in-page dispatch failed: {exc}",
                             hint="the tab may be busy; wait a moment and retry")
        after = self._await_change(mark, timeout_ms=self.config.settle_timeout_ms)
        dialogs = self.session.dialog_total - mark.dialogs
        changed = after.digest != mark.digest or dialogs > 0
        if changed:
            after = self._await_settle_url(after)
        note = "the browser did not deliver the synthetic key event; dispatched in the page"
        if info.get("submitted"):
            note += " and the form was submitted in the page"
        return _Fallback(attempted=True, succeeded=bool(changed), note=note, stage="key-events",
                         detail={"target": info.get("target"),
                                 "form_submitted": bool(info.get("submitted"))},
                         state=after, dialogs=dialogs, page_changed=bool(changed),
                         submitted=bool(info.get("submitted")))

    def _fallback_settle(
        self, mark: _State, loc: Any, state_before: dict[str, Any]
    ) -> tuple[bool, _State, int]:
        """Settle after an in-page dispatch and say whether the page reacted.

        Reuses the exact verification the native path uses: page hash, dialog
        count, and the element's own state (checked, expanded, value, disabled).
        """
        after = self._await_change(mark, timeout_ms=self.config.settle_timeout_ms)
        dialogs = self.session.dialog_total - mark.dialogs
        moved = after.digest != mark.digest
        toggled = False
        if not moved and not dialogs:
            toggled = self._state_changed(state_before, self._element_state(loc))
        return bool(moved or dialogs > 0 or toggled), self._await_settle_url(after), dialogs

    def _await_settle_url(self, after: _State, *, timeout_ms: int | None = None) -> _State:
        """Re-read the page state when the first read landed mid-navigation.

        An in-page dispatch can start a navigation; a state read taken while the
        new document is still loading comes back with an empty url, which would
        make the summary report ``page changed ()``.  Cheap no-op otherwise.
        """
        if after.url:
            return after
        limit = self.config.nav_timeout_ms if timeout_ms is None else timeout_ms
        try:
            self.session.page.wait_for_load_state("domcontentloaded",
                                                  timeout=min(2000, int(limit)))
        except Exception:
            pass
        self._sleep(0.05)
        fresh = self._state()
        return fresh if fresh.url else after

    def _live_url(self) -> str:
        """The tab's current url without touching the page's JS."""
        try:
            return str(self.session.page.url or "")
        except Exception:
            return ""

    def _scroll_into_view(self, loc: Any, element_id: int) -> None:
        try:
            loc.scroll_into_view_if_needed(timeout=min(2000, self.config.action_timeout_ms))
        except Exception:
            pass  # not scrollable / already visible: the click attempt decides

    def _cover(self, loc: Any) -> dict[str, Any]:
        try:
            return dict(loc.evaluate(_JS_COVER, timeout=self._probe_timeout_ms()) or {})
        except Exception:
            return {"covered": False}

    def _recover_from_overlay(self, loc: Any, covered: dict[str, Any]) -> str | None:
        """One cheap attempt to get rid of an overlay covering the target.

        Tries Escape first, then the first "make this go away" control inside the
        covering element.  Returns a short description of what worked, or
        ``None`` when the target is still covered.
        """
        page = self.session.page
        try:
            self._native_press("Escape")
        except Exception:
            pass
        self._sleep(0.12)
        if not self._cover(loc).get("covered"):
            return "Escape"
        target = self._close_target(covered)
        if target is None:
            return None
        try:
            locator = page.locator(selector_for(int(target["id"]))).first
            if locator.count() == 0:
                return None
            box = locator.bounding_box()
            if box:
                page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            else:
                locator.click(timeout=min(2000, self.config.action_timeout_ms))
        except Exception:
            return None
        self._sleep(0.15)
        if self._cover(loc).get("covered"):
            return None
        return f"clicked {target.get('text')!r}"

    def _close_target(self, covered: dict[str, Any]) -> dict[str, Any] | None:
        x, y = covered.get("x"), covered.get("y")
        if x is None or y is None:
            return None
        try:
            found = self.session.page.evaluate(_JS_CLOSE_TARGET, {"x": x, "y": y})
        except Exception:
            return None
        return dict(found) if isinstance(found, dict) else None

    def _element_state(self, loc: Any) -> dict[str, Any]:
        try:
            return dict(loc.evaluate(_JS_ELEMENT_STATE, timeout=self._probe_timeout_ms()) or {})
        except Exception:
            return {}

    def _probe_timeout_ms(self) -> int:
        """Timeout for read-only probes.

        They must fail fast: an element detached by a navigation would otherwise
        make every verification wait for the full action timeout.
        """
        return max(300, min(1500, int(getattr(self.config, "action_timeout_ms", 5000))))

    def _focused_element(self) -> dict[str, Any] | None:
        try:
            found = self.session.page.evaluate(_JS_FOCUSED)
        except Exception:
            return None
        return dict(found) if isinstance(found, dict) else None

    def _page_has_text(self, needle: str) -> bool:
        try:
            return bool(self.session.page.evaluate(_JS_HAS_TEXT, needle))
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # state / verification
    # ------------------------------------------------------------------ #
    def _state(self) -> _State:
        try:
            probed = self.session.page.evaluate(_JS_STATE) or {}
        except Exception:
            probed = {}
        probed = dict(probed)
        main = dict(probed.get("main") or {})
        payload = json.dumps(probed, sort_keys=True, default=str)
        return _State(
            url=str(main.get("url") or ""),
            title=str(main.get("title") or ""),
            digest=hashlib.sha256(payload.encode()).hexdigest()[:16],
            dialogs=self.session.dialog_total,
            raw=main,
        )

    def _await_change(self, before: _State, *, timeout_ms: int) -> _State:
        """Best-effort settle + wait for the DOM signature to move."""
        page = self.session.page
        budget = max(0.15, min(1.5, (timeout_ms or 0) / 1000 / 2))
        try:
            page.wait_for_load_state("networkidle", timeout=budget * 1000)
        except Exception:
            pass
        deadline = time.monotonic() + max(0.0, (timeout_ms or 0) / 1000)
        after = self._state()
        while time.monotonic() < deadline:
            if after.digest != before.digest or after.dialogs != before.dialogs:
                break
            self._sleep(0.05)
            after = self._state()
        return after

    @staticmethod
    def _state_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
        if not before or not after:
            return False
        for key in ("checked", "expanded", "value", "disabled"):
            if before.get(key) != after.get(key):
                return True
        return False

    def _sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        try:
            self.session.page.wait_for_timeout(int(seconds * 1000))
        except Exception:
            time.sleep(min(seconds, 0.5))

    # ------------------------------------------------------------------ #
    # model plumbing
    # ------------------------------------------------------------------ #
    def _model(self, *, mode: SnapshotMode = "full") -> PageModel | None:
        try:
            return self.session.snapshot(
                budget_tokens=self.config.page_token_budget, mode=mode
            )
        except Exception:
            return None

    def _meta(self, element_id: int) -> Element | None:
        model = self.session.last_model
        return model.by_id(element_id) if model is not None else None

    def _describe(self, element_id: int, meta: Element | None) -> str:
        if meta is None:
            return f"[{element_id}]"
        return f"[{element_id}] {meta.role} {meta.name!r}"

    def _is_password(self, state: dict[str, Any], meta: Element | None) -> bool:
        if bool(getattr(self.config, "allow_password_typing", False)):
            return False
        return (state.get("type") or "") == "password" or bool(meta is not None and meta.type == "password")

    def _filter_model(self, model: PageModel, needle: str) -> PageModel:
        low = needle.lower()
        elements = [
            el for el in model.elements
            if low in el.role.lower() or low in el.name.lower()
            or low in (el.value or "").lower() or low in (el.href or "").lower()
        ]
        lines = [line for line in model.text.splitlines() if low in line.lower()]
        filtered = replace(model, elements=elements, text="\n".join(lines))
        filtered.token_estimate = 0
        return filtered

    # ------------------------------------------------------------------ #
    # small utilities
    # ------------------------------------------------------------------ #
    def _failure(self, summary: str, hint: str, error: str | None = None) -> ToolResult:
        model = self._model(mode="delta") if self.session.started else None
        return ToolResult(ok=False, summary=summary, error=error or summary,
                          recovery_hint=hint or None, page=model)

    @staticmethod
    def _snapshot_mode(value: Any) -> SnapshotMode:
        mode = str(value or "full").strip().lower()
        if mode not in ("full", "delta", "text"):
            return "full"
        return mode  # type: ignore[return-value]

    @staticmethod
    def _element_id(args: dict[str, Any], key: str, *, allow_zero: bool = False) -> int:
        raw = args.get(key, None)
        try:
            value = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise _ActionFailure(f"{key} must be an element id from the page outline",
                                 hint="call page_outline and use one of the bracketed ids")
        if value == 0 and not allow_zero:
            raise _ActionFailure(f"{key}=0 is not a target here",
                                 hint="use a real element id from the page outline")
        if value < 0:
            raise _ActionFailure(f"{key} must not be negative",
                                 hint="use a real element id from the page outline")
        return value

    @staticmethod
    def _index(value: Any, *, default: int = 0) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return default

    def _normalise_url(self, raw: str) -> str:
        """Make a URL the model produced usable, without assuming a site."""
        url = raw.strip()
        if not url:
            raise _ActionFailure("empty url", hint="pass a url to open")
        if re.match(r"^(https?|file|about|data|blob):", url, re.IGNORECASE):
            return url
        if url.startswith("//"):
            return "https:" + url
        base = self.session.last_model.url if self.session.last_model else ""
        if url.startswith(("/", "?", ".", "#")) or url.startswith(".."):
            if not base:
                raise _ActionFailure(f"cannot resolve relative url {url!r} without a current page",
                                     hint="pass an absolute url")
            return urljoin(base, url)
        if " " in url:
            raise _ActionFailure(f"{url!r} does not look like a url",
                                 hint="pass a single url; use type_text to search inside a page")
        if "." not in url.split("/")[0] and not url.startswith("localhost"):
            raise _ActionFailure(f"{url!r} does not look like a url",
                                 hint="pass a full address such as example.com/path")
        return "https://" + url
