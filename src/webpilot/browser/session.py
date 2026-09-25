"""The browser: one persistent Playwright context plus the page model pipeline.

Design notes (see ``docs/INTERFACES.md`` for the contract):

* **Persistent profile.** ``launch_persistent_context`` with ``config.user_data_dir``
  means cookies, logins and 2FA survive across runs: the human can log in by hand
  once (the window is visible by default) and the agent continues in that session.
* **Native dialogs never hang a run.** Every page gets a ``dialog`` handler that
  records the text and dismisses the dialog; ``pending_dialogs`` exposes what was
  seen, and the next snapshot folds it into ``PageModel.dialogs``.
* **Raw page content is not for the main loop.** :meth:`raw_text` and
  :meth:`raw_html` exist only for the extractor sub-agent, which reads them and
  returns a short, cited answer.  Nothing else may call them - keeping the HTML
  out of the agent conversation is the whole point of the page sniffer.
"""

from __future__ import annotations

import functools
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from ..errors import BrowserError
from ..tokenizer import count_tokens as default_count_tokens
from ..types import AgentUI, PageModel, RunEvent, SnapshotMode, TabInfo
from .snapshot import build_page_model

__all__ = ["BrowserSession", "snapshot_js"]

#: Anything that looks like a sign-in form.  These are *capability* words, not
#: site knowledge - they must work on a site the author has never seen.
_AUTH_URL_HINTS = ("login", "log-in", "signin", "sign-in", "auth", "session", "account")

_JS_PASSWORD_INPUTS = """
(() => {
  const count = (doc) => {
    let n = 0;
    try { n += doc.querySelectorAll('input[type="password"]').length; } catch (e) {}
    try {
      doc.querySelectorAll("*").forEach((el) => { if (el.shadowRoot) n += count(el.shadowRoot); });
    } catch (e) {}
    try {
      doc.querySelectorAll("iframe").forEach((f) => { if (f.contentDocument) n += count(f.contentDocument); });
    } catch (e) {}
    return n;
  };
  return count(document);
})()
"""


@functools.lru_cache(maxsize=1)
def snapshot_js() -> str:
    """The page sniffer source (read once per process)."""
    return Path(__file__).with_name("dom_snapshot.js").read_text(encoding="utf-8")


class BrowserSession:
    """Owns the Playwright context and turns it into :class:`PageModel` snapshots."""

    def __init__(self, config: Any, ui: AgentUI | None = None) -> None:
        self.config = config
        self.ui = ui
        self._playwright: Any = None
        self._context: Any = None
        self._active_index = 0
        self._generation = 0
        self._screenshot_seq = 0
        self._blank_page: Any = None
        self.last_model: PageModel | None = None
        #: Native dialogs seen but not yet reported in a snapshot.
        self.pending_dialogs: list[str] = []
        #: Monotonic count of every native dialog ever seen (used by actions to
        #: notice that a click opened something instead of changing the page).
        self.dialog_total = 0
        self._count_tokens: Callable[[str], int] = default_count_tokens

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    @property
    def started(self) -> bool:
        return self._context is not None

    def launch_options(self) -> dict[str, Any]:
        """The arguments ``launch_persistent_context`` is called with.

        Exposed as a method so the headful/headless difference can be tested
        without opening a window: headful gets a maximised window and no fixed
        viewport, headless gets the configured one.
        """
        cfg = self.config
        args = ["--no-sandbox", "--disable-dev-shm-usage"]
        if not cfg.headless and not _window_size_pinned(cfg.browser_args):
            # headful: give the human a real, maximised window to log in with -
            # unless the caller pinned an exact geometry.  --start-maximized makes
            # Chromium re-assert a maximised state and fight the window manager, so
            # a demo that needs the window to stay inside half of the screen would
            # watch it snap back to its default size in the middle of the run.
            args.append("--start-maximized")
        args.append("--disable-blink-features=AutomationControlled")
        args.extend(cfg.browser_args or [])

        options: dict[str, Any] = {
            "user_data_dir": str(cfg.user_data_dir),
            "headless": bool(cfg.headless),
            "args": args,
            "locale": cfg.locale,
            "accept_downloads": True,
            "timeout": float(cfg.nav_timeout_ms),
        }
        if cfg.headless:
            width, height = (cfg.window_size or (1440, 900))
            options["viewport"] = {"width": int(width), "height": int(height)}
        else:
            # viewport=None -> the OS window decides; needed for --start-maximized
            options["viewport"] = None
        if cfg.browser_channel:
            options["channel"] = cfg.browser_channel
        if cfg.browser_executable:
            options["executable_path"] = cfg.browser_executable
        return options

    def start(self) -> None:
        """Launch the persistent context (headful unless ``config.headless``)."""
        if self._context is not None:
            return
        from playwright.sync_api import sync_playwright

        self.config.ensure_dirs()
        kwargs = self.launch_options()

        self._playwright = sync_playwright().start()
        try:
            self._context = self._playwright.chromium.launch_persistent_context(**kwargs)
        except Exception as exc:  # pragma: no cover - depends on the host
            self._playwright.stop()
            self._playwright = None
            self._context = None
            raise BrowserError(f"could not launch the browser: {exc}") from exc

        cfg = self.config
        self._context.set_default_timeout(cfg.action_timeout_ms)
        self._context.set_default_navigation_timeout(cfg.nav_timeout_ms)
        for page in self._context.pages:
            self._attach(page)
            if self._blank_page is None and _is_blank(page):
                # A persistent context always opens one blank tab; remember it so
                # the first new_page() can use it instead of leaving it dangling.
                self._blank_page = page
        self._context.on("page", self._attach)
        self._emit("info", f"browser started (headless={bool(cfg.headless)})")

    def stop(self) -> None:
        """Close the context and the Playwright driver (idempotent)."""
        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass
            self._context = None
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    def __enter__(self) -> "BrowserSession":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # pages / tabs
    # ------------------------------------------------------------------ #
    @property
    def context(self) -> Any:
        if self._context is None:
            raise BrowserError("browser session is not started - call start() first")
        return self._context

    @property
    def pages(self) -> list[Any]:
        """Every open tab (the contract's ``pages``)."""
        if self._context is None:
            return []
        return list(self._context.pages)

    @property
    def active_index(self) -> int:
        pages = self.pages
        if not pages:
            return 0
        return min(self._active_index, len(pages) - 1)

    @property
    def page(self) -> Any:
        """The tab the agent is currently acting on, created on demand."""
        pages = self.pages
        if not pages:
            if self._context is None:
                raise BrowserError("browser session is not started - call start() first")
            self.new_page()
            pages = self.pages
        return pages[self.active_index]

    def new_page(self, url: str | None = None) -> None:
        """Open a new tab (optionally navigating it) and make it the active one."""
        page = self._blank_page
        if page is not None and not _is_blank(page):
            page = None
        if page is not None:
            self._blank_page = None      # reuse the tab the context opened for us
        else:
            page = self.context.new_page()
            self._attach(page)
        pages = self.pages
        try:
            self._active_index = pages.index(page)
        except ValueError:  # pragma: no cover - the page list changed under us
            self._active_index = max(0, len(pages) - 1)
        if url:
            self.goto(url)

    def switch(self, index: int) -> None:
        """Make tab ``index`` active."""
        pages = self.pages
        if not pages:
            raise BrowserError("no tabs are open")
        if not 0 <= index < len(pages):
            raise BrowserError(f"tab {index} does not exist (0..{len(pages) - 1})")
        self._active_index = index
        try:
            pages[index].bring_to_front()
        except Exception:
            pass

    def close_page(self, index: int | None = None) -> None:
        """Close a tab (the active one by default)."""
        pages = self.pages
        if not pages:
            return
        index = self.active_index if index is None else index
        if 0 <= index < len(pages):
            try:
                pages[index].close()
            except Exception:
                pass
            self._active_index = max(0, min(self._active_index, len(self.pages) - 1))

    def tab_infos(self) -> list[TabInfo]:
        """Tabs as :class:`TabInfo` rows for the page model."""
        infos: list[TabInfo] = []
        for index, page in enumerate(self.pages):
            try:
                url, title = page.url, page.title()
            except Exception:
                url, title = "", ""
            infos.append(TabInfo(index=index, url=url, title=title, active=index == self.active_index))
        return infos

    def goto(self, url: str, *, wait_until: str = "load") -> None:
        """Navigate the active tab; navigation errors become ``BrowserError``."""
        try:
            self.page.goto(url, wait_until=wait_until, timeout=self.config.nav_timeout_ms)
        except Exception as exc:
            raise BrowserError(f"navigation to {url!r} failed: {_short(exc)}") from exc

    # ------------------------------------------------------------------ #
    # observation
    # ------------------------------------------------------------------ #
    def snapshot(self, *, budget_tokens: int, mode: SnapshotMode = "full") -> PageModel:
        """Capture the current page as a token-bounded :class:`PageModel`."""
        page = self.page
        self._generation += 1
        try:
            raw = page.evaluate(snapshot_js())
        except Exception as exc:
            raise BrowserError(f"could not read the page: {_short(exc)}") from exc
        if not isinstance(raw, dict):  # pragma: no cover - defensive
            raise BrowserError("the page sniffer returned an unexpected value")

        dialogs = list(self.pending_dialogs)
        self.pending_dialogs.clear()
        if dialogs:
            raw["dialogs"] = list(raw.get("dialogs") or []) + dialogs

        previous = self.last_model
        if previous is not None and previous.url != raw.get("url"):
            previous = None  # a delta across a navigation would be misleading

        model = build_page_model(
            raw,
            generation=self._generation,
            budget_tokens=budget_tokens,
            mode=mode,
            previous=previous,
            count_tokens=self._count_tokens,
        )
        model.tabs = self.tab_infos()
        self.last_model = model
        return model

    def raw_text(self, max_chars: int = 120_000) -> str:
        """Visible text of the page, truncated to ``max_chars``.

        **Extractor sub-agent only.**  The main agent loop must never call this:
        it would put an unbounded page dump into the model context.
        """
        page = self.page
        try:
            text = page.evaluate(
                "(() => { const b = document.body; return b ? (b.innerText || b.textContent || '') : ''; })()"
            )
        except Exception as exc:
            raise BrowserError(f"could not read the page text: {_short(exc)}") from exc
        return str(text or "")[:max_chars]

    def raw_html(self, max_chars: int = 200_000) -> str:
        """Serialised DOM (what we deliberately *do not* send to the model).

        **Extractor sub-agent only** - same rule as :meth:`raw_text`.
        """
        try:
            html = self.page.content()
        except Exception as exc:
            raise BrowserError(f"could not read the page html: {_short(exc)}") from exc
        return str(html or "")[:max_chars]

    def is_blocked_by_auth(self) -> bool:
        """Heuristic: does the current page ask for credentials?

        A visible password field is the strong signal; a sign-in-looking URL or
        title together with a form is the weak one.  Nothing here is site
        specific - the words are about *what the page does*, not who serves it.
        """
        # Live check first: a stale model from the previous page must never make
        # the answer wrong.
        try:
            if int(self.page.evaluate(_JS_PASSWORD_INPUTS) or 0) > 0:
                return True
        except Exception:
            pass
        model = self.last_model
        if model is not None and model.url != _page_url(self.page):
            model = None  # the model belongs to a page we already left
        if model is None:
            try:
                model = self.snapshot(budget_tokens=self.config.page_token_budget_min)
            except Exception:
                return False
        if model is None:
            return False
        if any((el.type or "") == "password" for el in model.elements):
            return True
        haystack = f"{model.url} {model.title}".lower()
        has_login_word = any(hint in haystack for hint in _AUTH_URL_HINTS)
        has_form = any(
            el.tag in ("input", "form") or (el.type or "") in ("text", "email")
            for el in model.elements
        )
        return bool(has_login_word and has_form and len(model.elements) < 60)

    def screenshot(self, label: str = "shot", full_page: bool = False) -> str:
        """Save a PNG for the human/audit trail and return its path."""
        cfg = self.config
        if not cfg.save_screenshots:
            raise BrowserError("screenshots are disabled by configuration")
        self._screenshot_seq += 1
        directory = Path(cfg.screenshot_dir)
        directory.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(label)).strip("-") or "shot"
        path = directory / f"{int(time.time())}-{self._screenshot_seq:03d}-{safe[:60]}.png"
        try:
            self.page.screenshot(path=str(path), full_page=full_page)
        except Exception as exc:
            raise BrowserError(f"screenshot failed: {_short(exc)}") from exc
        self._emit("info", f"screenshot saved: {path}")
        if self.last_model is not None:
            self.last_model.screenshot_path = str(path)
        return str(path)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _attach(self, page: Any) -> None:
        """Wire the per-page handlers (dialogs are the important one)."""
        try:
            page.on("dialog", self._on_dialog)
        except Exception:  # pragma: no cover - defensive
            pass

    def _on_dialog(self, dialog: Any) -> None:
        """Record a native dialog and dismiss it so the run never hangs."""
        try:
            kind = getattr(dialog, "type", "") or "dialog"
            message = str(getattr(dialog, "message", "") or "")
        except Exception:
            kind, message = "dialog", ""
        text = f"{kind}: {message}".strip()
        self.dialog_total += 1
        self.pending_dialogs.append(text)
        self._emit("info", f"native dialog dismissed: {text}")
        try:
            dialog.dismiss()
        except Exception:  # pragma: no cover - already handled
            pass

    def _emit(self, kind: str, message: str, **data: Any) -> None:
        if self.ui is None:
            return
        try:
            self.ui.emit(RunEvent(kind=kind, message=message, data=data))  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - a broken UI must not stop the run
            pass


def _short(exc: Any, limit: int = 300) -> str:
    return " ".join(str(exc).split())[:limit] or exc.__class__.__name__


def _page_url(page: Any) -> str:
    """Current url of a page, or "" when the page is gone."""
    try:
        return str(page.url or "")
    except Exception:
        return ""


def _window_size_pinned(browser_args: list[str] | None) -> bool:
    """True when the caller pinned the window geometry in the browser arguments.

    A recording harness splits the screen in half and wants the browser to stay in
    its half, so it passes ``--window-size``/``--window-position``; adding
    ``--start-maximized`` on top of that is a contradiction Chromium resolves by
    re-maximising (or snapping back to a remembered size) behind the caller's back.
    """
    return any(str(arg).startswith("--window-size") for arg in (browser_args or []))


def _is_blank(page: Any) -> bool:
    """True while a tab is still the empty page the browser opened with."""
    try:
        if page.is_closed():
            return False
    except Exception:
        return False
    return _page_url(page) in ("", "about:blank")


def dump_model(model: PageModel) -> str:
    """Debug helper: the raw JSON of a model (used by tests and ``--verbose``)."""
    return json.dumps(
        {
            "url": model.url,
            "title": model.title,
            "generation": model.generation,
            "elements": [el.__dict__ for el in model.elements],
            "scroll": model.scroll,
        },
        ensure_ascii=False,
    )[:4000]
