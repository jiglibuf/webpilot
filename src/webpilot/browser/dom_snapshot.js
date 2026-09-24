/* webpilot page sniffer.
 *
 * Single idempotent IIFE evaluated with `page.evaluate`.  It walks the DOM of the
 * main frame, same-origin iframes and open shadow roots and returns a compact
 * JSON description of everything an agent can act on.
 *
 * The ONLY handle it produces is the integer `id`, written to the DOM as the
 * attribute `data-webpilot-id`.  No selector for a real site ever appears in the
 * Python source: actions re-derive `[data-webpilot-id="N"]` from that integer.
 *
 * Idempotent: running it twice on the same page yields the same ids for the same
 * nodes (existing ids are reused), and the console/error collector is installed
 * exactly once behind `window.__webpilotInstalled`.
 */
(() => {
  "use strict";

  const ATTR = "data-webpilot-id";
  const COUNTER = "__webpilotCounter";
  const NAME_MAX = 120;
  const MAX_ELEMENTS = 400;
  const MAX_TEXT_BLOCKS = 250;
  const MAX_ERRORS = 20;
  const MAX_FRAMES = 12;
  const MAX_OPTIONS = 25;

  const errors = [];
  const note = (msg) => {
    const s = String(msg == null ? "" : msg);
    if (!s) return;
    if (errors.length < MAX_ERRORS && errors.indexOf(s) === -1) errors.push(s.slice(0, 300));
  };

  /* ---------------------------------------------------------------- utils */

  const clean = (value) => {
    if (value == null) return "";
    let s = String(value).replace(/\u00a0/g, " ").replace(/\s+/g, " ").trim();
    if (s.length > NAME_MAX) s = s.slice(0, NAME_MAX - 1) + "…";
    return s;
  };

  const attr = (el, name) => {
    try {
      const v = el.getAttribute ? el.getAttribute(name) : null;
      return v == null ? "" : String(v);
    } catch (e) {
      return "";
    }
  };

  const isPasswordEl = (el) => {
    try {
      return String(el.type || "").toLowerCase() === "password";
    } catch (e) {
      return false;
    }
  };

  /** FNV-1a over a string: cheap, dependency-free, good enough for change detection. */
  const hash = (s) => {
    let h = 2166136261;
    const str = String(s);
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return (h >>> 0).toString(16).padStart(8, "0");
  };

  const cssEscape = (ident) => {
    const s = String(ident);
    if (s && /^[A-Za-z_][\w-]*$/.test(s)) return s;
    return s.replace(/[^A-Za-z0-9_-]/g, (c) => "\\" + c);
  };

  const rectOf = (el) => {
    try {
      return el.getBoundingClientRect();
    } catch (e) {
      return null;
    }
  };

  /* Walks the composed tree of `doc`: every element, plus the open shadow roots
   * of those elements, plus same-origin iframe documents.  Accessing
   * `el.shadowRoot` on a closed root throws -> handled, the node is skipped. */
  const allNodes = (doc) => {
    const out = [];
    const roots = [doc];
    let guard = 0;
    while (roots.length && guard++ < 200) {
      const root = roots.shift();
      let nodes = [];
      try {
        nodes = root.querySelectorAll("*");
      } catch (e) {
        continue;
      }
      if (!nodes) continue;
      for (let j = 0; j < nodes.length; j++) {
        const el = nodes[j];
        // Same-origin frames are walked separately (one frameIndex per frame),
        // so an iframe's nodes must not be collected twice here.
        let shadow = null;
        try {
          shadow = el.shadowRoot || null;
        } catch (e) {
          shadow = null;
        }
        if (shadow) roots.push(shadow);
        out.push(el);
      }
    }
    return out;
  };

  const selectorHint = (el) => {
    const parts = [];
    let cur = el;
    let span = 0;
    while (cur && span < 4) {
      const tag = String(cur.tagName || "").toLowerCase();
      if (!tag) break;
      if (tag === "html" || tag === "body") break;
      let seg = tag;
      const id = attr(cur, "id");
      const cls = attr(cur, "class").split(/\s+/).filter(Boolean)[0];
      const nm = attr(cur, "name");
      if (id) seg += "#" + cssEscape(id);
      else if (nm) seg += '[name="' + nm.replace(/"/g, "") + '"]';
      else if (cls) seg += "." + cssEscape(cls);
      else if (cur.parentElement) {
        const sibs = Array.prototype.filter.call(cur.parentElement.children, (s) => s.tagName === cur.tagName);
        if (sibs.length > 1) seg += ":nth-of-type(" + (sibs.indexOf(cur) + 1) + ")";
      }
      parts.unshift(seg);
      cur = cur.parentElement || null;
      span++;
    }
    const hint = parts.join(" > ");
    return hint.length > 120 ? hint.slice(0, 119) + "…" : hint;
  };

  /* --------------------------------------------------------------- labels */

  const labelTextFor = (el) => {
    const id = attr(el, "id");
    if (id) {
      const doc = el.ownerDocument || document;
      let root = null;
      try {
        root = el.getRootNode ? el.getRootNode() : doc;
      } catch (e) {
        root = doc;
      }
      if (root && root.querySelectorAll) {
        let found = "";
        try {
          found = root.querySelectorAll('[for="' + id.replace(/"/g, "") + '"]');
        } catch (e) {
          found = [];
        }
        for (let i = 0; i < found.length; i++) {
          const t = clean(found[i].textContent);
          if (t) return t;
        }
      }
    }
    let node = el.parentElement || null;
    let hops = 0;
    while (node && hops < 3) {
      const tag = String(node.tagName || "").toLowerCase();
      if (tag === "label") {
        const t = clean(node.textContent);
        if (t) return t;
        break;
      }
      node = node.parentElement || null;
      hops++;
    }
    return "";
  };

  const visibleText = (el) => {
    let s = "";
    try {
      s = el.innerText || "";
    } catch (e) {
      s = "";
    }
    if (!s) s = el.textContent || "";
    return clean(s);
  };

  /** Accessible-name priority, exactly as specified in docs/INTERFACES.md. */
  const accessibleName = (el) => {
    const tag = String(el.tagName || "").toLowerCase();
    const aria = clean(attr(el, "aria-label"));
    if (aria) return aria;

    const labelledby = attr(el, "aria-labelledby").trim();
    if (labelledby) {
      const doc = el.ownerDocument || document;
      let root = null;
      try {
        root = el.getRootNode ? el.getRootNode() : doc;
      } catch (e) {
        root = doc;
      }
      const pieces = [];
      const ids = labelledby.split(/\s+/);
      for (let i = 0; i < ids.length; i++) {
        let target = null;
        try {
          target = root && root.getElementById ? root.getElementById(ids[i]) : null;
          if (!target) target = doc.getElementById(ids[i]);
        } catch (e) {
          target = null;
        }
        if (target) {
          const t = clean(target.textContent);
          if (t) pieces.push(t);
        }
      }
      const joined = clean(pieces.join(" "));
      if (joined) return joined;
    }

    const label = labelTextFor(el);
    if (label) return label;

    const placeholder = clean(attr(el, "placeholder"));
    if (placeholder) return placeholder;

    const value = clean(el.value);
    if (value) return value;

    const alt = clean(attr(el, "alt"));
    if (alt) return alt;
    const title = clean(attr(el, "title"));
    if (title) return title;

    if (tag === "select" || tag === "textarea") {
      const inner = clean(el.textContent);
      if (inner) return inner;
    }
    return visibleText(el);
  };

  /* ------------------------------------------------------------ role/kind */

  const INTERACTIVE_ROLES = {
    button: 1, link: 1, menuitem: 1, tab: 1, switch: 1, slider: 1, checkbox: 1,
    radio: 1, textbox: 1, combobox: 1, searchbox: 1, spinbutton: 1, option: 1,
    submit: 1, "menuitemcheckbox": 1, "menuitemradio": 1, "listbox": 1,
    "dialog": 0,
  };

  const INPUT_ROLE = {
    submit: "submit", button: "button", reset: "button", image: "button",
    checkbox: "checkbox", radio: "radio", file: "file", range: "slider",
    search: "textbox", email: "textbox", tel: "textbox", url: "textbox",
    number: "textbox", text: "textbox", password: "textbox",
  };

  const kindFor = (el) => {
    const tag = String(el.tagName || "").toLowerCase();
    const role = attr(el, "role").trim().toLowerCase();
    const type = String(el.type || "").toLowerCase();
    // A <button> defaults to type=submit in the DOM even outside a form; only a
    // control that can really submit a form is reported as a submit control.
    const inForm = !!(el.form || el.closest && el.closest("form"));

    if (tag === "a" || tag === "area") return attr(el, "href") || role === "link" ? "link" : "other";
    if (tag === "button") {
      if (type === "reset") return "button";
      return type === "submit" && inForm ? "submit" : "button";
    }
    if (tag === "input") {
      if (type === "hidden") return "other";
      if (type === "submit" || type === "image") return inForm ? "submit" : "button";
      const r = INPUT_ROLE[type || "text"];
      return r || "textbox";
    }
    if (tag === "select") return el.multiple ? "select" : "select";
    if (tag === "textarea") return "textbox";
    if (tag === "summary") return "button";
    if (tag === "label") return "label";
    if (tag === "option") return "option";
    if (role) {
      if (role === "link") return "link";
      if (role === "button") return "button";
      if (role === "searchbox" || role === "textbox") return "textbox";
      if (role === "combobox") return "select";
      if (role === "dialog" || role === "treeitem" || role === "menuitem") return "menuitem";
      if (INTERACTIVE_ROLES[role] === 1) return role;
      return "other";
    }
    if (el.isContentEditable) return "textbox";
    const tabindex = attr(el, "tabindex").trim();
    if (tabindex && Number(tabindex) >= 0) return "other";
    return "other";
  };

  const VISIBLE_ROLES = {
    button: 1, link: 1, submit: 1, checkbox: 1, radio: 1, textbox: 1, select: 1,
    option: 1, menuitem: 1, tab: 1, switch: 1, slider: 1, file: 1, label: 1,
  };

  /* -------------------------------------------------------------- visible */

  const styleOf = (el) => {
    try {
      const view = (el.ownerDocument || document).defaultView;
      return view && view.getComputedStyle ? view.getComputedStyle(el) : null;
    } catch (e) {
      return null;
    }
  };

  const isVisible = (el) => {
    const tag = String(el.tagName || "").toLowerCase();
    if (tag === "input" && String(el.type || "").toLowerCase() === "hidden") return false;
    let node = el;
    let hops = 0;
    while (node && node.nodeType === 1 && hops < 40) {
      const st = styleOf(node);
      if (st) {
        if (st.display === "none" || st.visibility === "hidden" || st.visibility === "collapse") return false;
        const op = parseFloat(st.opacity);
        if (!isNaN(op) && op === 0) return false;
      }
      node = node.parentElement || null;
      hops++;
    }
    const r = rectOf(el);
    if (!r) return false;
    return r.width > 0 && r.height > 0;
  };

  /* --------------------------------------------------------------- frames */

  const frames = [];
  const frameIndexOf = (doc) => {
    for (let i = 0; i < frames.length; i++) {
      if (frames[i].doc === doc) return frames[i].index;
    }
    return 0;
  };

  const collectFrames = () => {
    frames.length = 0;
    frames.push({ doc: document, index: 0, url: location.href, top: 0 });
    const queue = [document];
    let guard = 0;
    while (queue.length && guard++ < MAX_FRAMES) {
      const doc = queue.shift();
      const parent = frames[frameIndexOf(doc)];
      if (!parent) continue;
      let framesInDoc = [];
      try {
        framesInDoc = doc.querySelectorAll("iframe, frame");
      } catch (e) {
        framesInDoc = [];
      }
      for (let i = 0; i < framesInDoc.length; i++) {
        const f = framesInDoc[i];
        const r = rectOf(f);
        let inner = null;
        try {
          inner = f.contentDocument || null;
        } catch (e) {
          inner = null;           // cross-origin: contents are unreachable
        }
        let url = "";
        try {
          url = f.contentWindow && f.contentWindow.location ? f.contentWindow.location.href : "";
        } catch (e) {
          url = attr(f, "src") || "cross-origin frame";
        }
        frames.push({
          doc: inner,
          index: frames.length,
          url: url || attr(f, "src"),
          top: parent.top + (r ? r.top : 0),
          offsetLeft: r ? r.left : 0,
          offsetTop: r ? r.top : 0,
        });
        if (inner && frames.length < MAX_FRAMES) queue.push(inner);
      }
    }
    return frames;
  };

  /* ------------------------------------------------------------------------ */
  /* id assignment                                                             */
  /* ------------------------------------------------------------------------ */

  /* Ids only ever grow - within a tab the counter lives in sessionStorage, so a
   * navigation cannot hand an old id to a new node (a stale id must stay
   * unresolvable, otherwise "click the id I saw before" would hit a stranger). */
  const readCounter = () => {
    let value = 1000;
    try {
      const stored = parseInt(window.sessionStorage.getItem("__webpilotId"), 10);
      if (!isNaN(stored) && stored > value) value = stored;
    } catch (e) { /* storage may be unavailable */ }
    if (typeof window[COUNTER] === "number" && window[COUNTER] > value) value = window[COUNTER];
    return value;
  };

  const bump = (value) => {
    window[COUNTER] = value;
    try {
      window.sessionStorage.setItem("__webpilotId", String(value));
    } catch (e) { /* storage may be unavailable */ }
  };

  if (!window[COUNTER] || typeof window[COUNTER] !== "number") window[COUNTER] = readCounter();

  const nextId = () => {
    const docRoot = document.documentElement;
    let highest = readCounter();
    if (docRoot) {
      const used = [].slice.call(docRoot.querySelectorAll("[" + ATTR + "]")).map((el) => {
        const v = parseInt(el.getAttribute(ATTR), 10);
        return isNaN(v) ? 0 : v;
      });
      if (used.length) highest = Math.max.apply(null, used.concat([highest]));
    }
    bump(highest + 1);
    return highest + 1;
  };

  const idOf = (el) => {
    const existing = attr(el, ATTR);
    const parsed = parseInt(existing, 10);
    if (existing && !isNaN(parsed)) return parsed;
    const id = nextId();
    try {
      el.setAttribute(ATTR, String(id));
    } catch (e) {
      return id;
    }
    return id;
  };

  /* ---------------------------------------------------------- text blocks */

  const TEXT_TAGS = {
    h1: "heading", h2: "heading", h3: "heading", h4: "heading", h5: "heading",
    h6: "heading", p: "paragraph", li: "list", dt: "list", dd: "list",
    blockquote: "paragraph", figcaption: "paragraph", caption: "table",
  };
  const SKIP_TAGS = {
    script: 1, style: 1, noscript: 1, svg: 1, template: 1, head: 1, iframe: 1,
    option: 1, select: 1, textarea: 1, meta: 1, link: 1,
  };

  const textKindOf = (el) => {
    const tag = String(el.tagName || "").toLowerCase();
    const role = attr(el, "role").trim().toLowerCase();
    if (role === "alert") return "alert";
    if (TEXT_TAGS[tag]) return TEXT_TAGS[tag];
    if (tag === "table") return "table";
    if (tag === "label") return "label";
    if (tag === "div" || tag === "section" || tag === "article" || tag === "span") {
      const inner = visibleText(el);
      if (!inner || inner.length <= 30) return "";
      // Only leaf-ish blocks: a wrapper whose children carry text of their own
      // would just repeat them (and the text is the expensive part of the
      // context, so duplicates are a real cost).
      let children = [];
      try {
        children = Array.prototype.slice.call(el.children || []);
      } catch (e) {
        children = [];
      }
      for (let i = 0; i < children.length; i++) {
        const child = children[i];
        const ct = String(child.tagName || "").toLowerCase();
        if (SKIP_TAGS[ct]) continue;
        if (visibleText(child).length > 30) return "";
      }
      return "paragraph";
    }
    return "";
  };

  /* --------------------------------------------------------------- collect */

  const collectFrame = (frame) => {
    const doc = frame.doc;
    if (!doc) return;
    let title = "";
    try {
      title = doc.title || "";
    } catch (e) {
      title = "";
    }
    const nodes = allNodes(doc);
    const elements = [];
    let dropped = 0;
    const seenNames = [];

    for (let i = 0; i < nodes.length; i++) {
      if (elements.length >= MAX_ELEMENTS) {
        dropped += 1;
        continue;
      }
      const el = nodes[i];
      const tag = String(el.tagName || "").toLowerCase();
      if (!tag || SKIP_TAGS[tag]) continue;
      const elType = String(el.type || "").toLowerCase();

      const role = kindFor(el);
      if (!VISIBLE_ROLES[role] && role !== "other") continue;
      if (role === "other" && !el.isContentEditable && !attr(el, "onclick") && !attr(el, "aria-label")) {
        // only keep "other" when it is a genuinely clickable leaf
        const st = styleOf(el);
        if (!st || st.cursor !== "pointer") continue;
        if (el.children && el.children.length > 2) continue;
      }
      if (!isVisible(el)) {
        dropped += 1;
        continue;
      }

      const r = rectOf(el);
      const view = (doc.defaultView) || window;
      const vh = view.innerHeight || window.innerHeight || 0;
      const name = accessibleName(el);
      const domHash = hash([role, name, el.value == null ? "" : el.value,
        el.disabled ? "d" : "", el.checked == null ? "" : String(el.checked),
        el.getAttribute && el.getAttribute("aria-expanded")].join("|"));
      // `checked` only exists on checkboxes/radios/options - reporting the
      // (always false) property of a text input would just be noise.
      const checkable = (tag === "input" && (elType === "checkbox" || elType === "radio")) ||
        role === "checkbox" || role === "radio" || role === "switch" ||
        role === "menuitemcheckbox" || role === "menuitemradio";
      const ariaChecked = attr(el, "aria-checked");
      const explicitType = el.getAttribute ? el.getAttribute("type") : null;

      const entry = {
        id: idOf(el),
        role: role,
        name: name,
        tag: tag,
        type: explicitType ? String(explicitType).toLowerCase() : null,
        value: el.value == null || el.value === "" ? null : clean(el.value),
        placeholder: attr(el, "placeholder") ? clean(attr(el, "placeholder")) : null,
        href: null,
        disabled: !!el.disabled || attr(el, "aria-disabled") === "true",
        checked: checkable ? !!el.checked
          : (ariaChecked === "" ? null : ariaChecked === "true"),
        required: !!el.required || attr(el, "aria-required") === "true",
        expanded: attr(el, "aria-expanded") === "" ? null : attr(el, "aria-expanded") === "true",
        invalid: attr(el, "aria-invalid") === "true",
        inViewport: !r ? false : r.bottom > -2 && r.top < vh + 2,
        options: [],
        domHash: domHash,
        selectorHint: selectorHint(el),
        frameIndex: frame.index,
        note: "",
        isPassword: isPasswordEl(el),
      };

      if (tag === "a" || tag === "area") {
        const href = el.href || attr(el, "href");
        if (!href) continue;
        try {
          const abs = new URL(href, (doc.defaultView && doc.defaultView.location) || location);
          entry.href = (abs.origin + abs.pathname + (abs.search || "")).slice(0, 160);
        } catch (e) {
          entry.href = String(href).slice(0, 160);
        }
      }
      if (tag === "select") {
        let opts = [];
        try {
          opts = [].slice.call(el.options || []).slice(0, MAX_OPTIONS).map((o) => clean(o.text || o.value));
        } catch (e) {
          opts = [];
        }
        entry.options = opts.filter(Boolean);
        const sel = el.selectedOptions && el.selectedOptions[0];
        if (sel) entry.value = clean(sel.text || sel.value);
      }
      if (!entry.name) {
        if (isPasswordEl(el)) entry.name = "password field";
        else if (entry.role === "textbox") entry.name = "text field";
        else if (tag === "a") entry.name = "link";
      }
      if (name && entry.inViewport) seenNames.push(name);
      elements.push(entry);
    }

    // ---- text blocks (skipping nodes already represented as an element name)
    const text = [];
    for (let i = 0; i < nodes.length; i++) {
      const el = nodes[i];
      const tag = String(el.tagName || "").toLowerCase();
      if (SKIP_TAGS[tag]) continue;
      const kind = textKindOf(el);
      if (!kind) continue;
      const val = visibleText(el);
      if (!val || val.length < 2) continue;
      if (seenNames.indexOf(val) !== -1) continue;
      const r = rectOf(el);
      const vh = (doc.defaultView && doc.defaultView.innerHeight) || window.innerHeight || 0;
      text.push({
        kind: kind,
        text: val,
        inViewport: r ? r.bottom > -2 && r.top < vh + 2 : false,
        frameIndex: frame.index,
        top: (r ? r.top : 0) + (frame.offsetTop || 0) + (doc.defaultView && doc.defaultView.scrollY ? doc.defaultView.scrollY : 0),
        order: frame.index * 1000 + i,
      });
      if (text.length >= MAX_TEXT_BLOCKS) break;
    }

    // ---- dialogs / modal overlays (role=dialog or aria-modal)
    const dialogs = [];
    for (let i = 0; i < nodes.length; i++) {
      const el = nodes[i];
      const role = attr(el, "role").trim().toLowerCase();
      const modal = attr(el, "aria-modal") === "true";
      if (role !== "dialog" && role !== "alertdialog" && !modal) continue;
      if (!isVisible(el)) continue;
      const label = clean(attr(el, "aria-label"));
      const body = clean(el.textContent);
      const text2 = clean((label ? label + " — " : "") + body);
      if (text2) dialogs.push(text2.slice(0, 300));
    }

    return { title: title, elements: elements, text: text, dropped: dropped, dialogs: dialogs };
  };

  /* ------------------------------------------------------------------ main */

  const frameList = collectFrames();
  let title = document.title || "";
  const elements = [];
  const textBlocks = [];
  const dialogs = [];
  let droppedElements = 0;

  for (let i = 0; i < frameList.length; i++) {
    const frame = frameList[i];
    let result = null;
    try {
      result = collectFrame(frame);
    } catch (e) {
      note("frame " + frame.index + " scan failed: " + (e && e.message ? e.message : e));
      continue;
    }
    if (!result) continue;
    if (i === 0) title = result.title || title;
    for (let j = 0; j < result.elements.length; j++) elements.push(result.elements[j]);
    for (let j = 0; j < result.text.length; j++) textBlocks.push(result.text[j]);
    for (let j = 0; j < result.dialogs.length; j++) dialogs.push(result.dialogs[j]);
    droppedElements += result.dropped;
  }

  textBlocks.sort((a, b) => (b.inViewport ? 1 : 0) - (a.inViewport ? 1 : 0) || a.top - b.top || a.order - b.order);
  const text = textBlocks.map((t) => ({ kind: t.kind, text: t.text, inViewport: t.inViewport }));

  const alerts = [];
  try {
    const alertNodes = document.querySelectorAll('[role="alert"], [role="status"], .error, [aria-live="assertive"]');
    for (let i = 0; i < alertNodes.length && alerts.length < 10; i++) {
      if (!isVisible(alertNodes[i])) continue;
      const t = clean(alertNodes[i].textContent);
      if (t && alerts.indexOf(t) === -1) alerts.push(t.slice(0, 300));
    }
  } catch (e) {
    note("alert scan failed: " + e);
  }

  /* scroll */
  let scroll = { y: 0, maxY: 0, viewportH: 0, atBottom: true };
  try {
    const de = document.documentElement || {};
    const maxScrollY = Math.max(0, (de.scrollHeight || 0) - (window.innerHeight || 0));
    const y = Math.max(window.scrollY || 0, 0);
    scroll = {
      y: Math.round(y),
      maxY: Math.round(maxScrollY),
      viewportH: Math.round(window.innerHeight || 0),
      atBottom: maxScrollY <= 0 ? true : y >= maxScrollY - 4,
    };
  } catch (e) {
    note("scroll read failed: " + e);
  }

  /* focused element id (search all roots; activeElement returns the host of a shadow root) */
  let focusedId = null;
  try {
    let active = document.activeElement;
    let hops = 0;
    while (active && active.shadowRoot && active.shadowRoot.activeElement && hops++ < 10) {
      active = active.shadowRoot.activeElement;
    }
    if (active && active !== document.body && attr(active, ATTR)) {
      focusedId = parseInt(attr(active, ATTR), 10);
      if (isNaN(focusedId)) focusedId = null;
    }
  } catch (e) {
    focusedId = null;
  }

  /* console errors / thrown errors, collected once per document */
  if (!window.__webpilotInstalled) {
    window.__webpilotInstalled = true;
    window.__webpilotErrors = [];
    window.addEventListener("error", (event) => {
      try {
        window.__webpilotErrors.push(String((event && (event.message || event.type)) || "error"));
      } catch (e) { /* ignore */ }
    }, true);
    window.addEventListener("unhandledrejection", (event) => {
      try {
        const reason = event && event.reason;
        window.__webpilotErrors.push("unhandled rejection: " + String(reason && reason.message ? reason.message : reason));
      } catch (e) { /* ignore */ }
    });
    const origError = console.error;
    console.error = function () {
      try {
        window.__webpilotErrors.push([].slice.call(arguments).map(String).join(" "));
      } catch (e) { /* ignore */ }
      return origError.apply(console, arguments);
    };
  }
  for (let i = 0; i < (window.__webpilotErrors || []).length && errors.length < MAX_ERRORS; i++) {
    note(window.__webpilotErrors[i]);
  }

  /* sizes */
  let htmlChars = 0;
  let fullTextChars = 0;
  try {
    const de = document.documentElement;
    htmlChars = de && de.outerHTML ? de.outerHTML.length : 0;
    const body = document.body;
    fullTextChars = ((body && (body.innerText || body.textContent)) || "").length;
  } catch (e) { /* ignore */ }

  return {
    url: location.href,
    title: clean(title),
    elements: elements,
    text: text,
    alerts: alerts,
    dialogs: dialogs,
    scroll: scroll,
    focusedId: focusedId,
    htmlChars: htmlChars,
    fullTextChars: fullTextChars,
    droppedElements: droppedElements,
    frames: frameList.map((f) => f.url || ""),
    errors: errors,
  };
})()
