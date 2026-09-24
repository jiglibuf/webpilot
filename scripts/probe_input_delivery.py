"""Diagnostic: does synthetic input actually reach the document?  (docs/RESEARCH.md, §8)

The agent once reported ``ok=False / no effect`` on a public SPA: the element was
found, the click was dispatched, the page did not change.  This script measures the
underlying question directly — how many DOM events a Playwright click and a keyboard
press produce — in a plain ``launch()`` browser and in a
``launch_persistent_context()`` one, on a page served by the local fixture site.

The interesting case is the *second* document: counters are installed after a click
that navigated, because that is where delivery breaks on the affected build.

Run:  python scripts/probe_input_delivery.py [--headful] [--url URL]
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright  # noqa: E402

#: Counts what the page actually receives.  Installed per document.
FEED_JS = """
() => {
  window.__feed = {clicks: 0, keys: 0, focus: document.hasFocus()};
  for (const t of ['pointerdown', 'mousedown', 'click'])
    document.addEventListener(t, () => window.__feed.clicks++, true);
  for (const t of ['keydown', 'keypress'])
    document.addEventListener(t, () => window.__feed.keys++, true);
  return true;
}
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_site(port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.fixtures.site.server", "--port", str(port)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return proc
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("fixture site did not start")


def probe_once(mode: str, url: str, headful: bool, profile: Path,
               click_selector: str, probe_selector: str) -> dict:
    """Click a link, then measure input delivery in the document it opened."""
    with sync_playwright() as p:
        browser = None
        if mode == "plain":
            browser = p.chromium.launch(headless=not headful)
            ctx = browser.new_context()
        else:
            ctx = p.chromium.launch_persistent_context(str(profile), headless=not headful)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        page.goto(url, wait_until="domcontentloaded")
        page.evaluate(FEED_JS)
        first = page.evaluate("() => window.__feed")           # counters on document #1

        page.click(click_selector)                             # synthetic click → navigation
        page.wait_for_load_state("domcontentloaded")
        page.evaluate(FEED_JS)                                 # counters on document #2

        if probe_selector:
            page.click(probe_selector)                         # does a click land here at all?
        page.keyboard.press("Tab")                             # …and do keys?
        page.keyboard.press("Tab")
        second = page.evaluate("() => window.__feed")

        result = {"mode": mode, "doc1": first, "doc2": second, "url": page.url}
        ctx.close()
        if browser:
            browser.close()
        return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", help="probe this URL instead of the local fixture site")
    ap.add_argument("--click", default="a[href^='/product/']",
                    help="element that starts the first (synthetic) click")
    ap.add_argument("--then-click", default="h1",
                    help="element clicked inside the document that click opened ('-' to skip)")
    ap.add_argument("--headful", action="store_true", help="show the browser window")
    args = ap.parse_args()

    proc = None
    if args.url:
        url = args.url
    else:
        port = free_port()
        proc = start_site(port)
        url = f"http://127.0.0.1:{port}/"

    print(f"target: {url}")
    try:
        with tempfile.TemporaryDirectory(prefix="webpilot-probe-") as tmp:
            for mode in ("plain", "persistent"):
                profile = Path(tmp) / mode
                profile.mkdir()
                r = probe_once(mode, url, args.headful, profile,
                               args.click, "" if args.then_click == "-" else args.then_click)
                print(f"  {r['mode']:<11} doc#1 clicks={r['doc1']['clicks']:<2} "
                      f"focus={r['doc1']['focus']}  |  after a synthetic click: "
                      f"clicks={r['doc2']['clicks']:<2} keys={r['doc2']['keys']:<2} "
                      f"focus={r['doc2']['focus']}  ->  {r['url']}")
        print("\nclicks/keys > 0 means input is delivered; 0 means the page never saw it.")
        print("webpilot does not depend on this outcome: browser/actions.py always tries the")
        print("native path first and repeats the action inside the page when nothing changed.")
        return 0
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
