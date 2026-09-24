"""Empirical comparison of page-perception strategies (numbers for docs/RESEARCH.md).

For every target page it measures what the model would have to read under each
strategy: raw HTML, visible text, accessibility tree, and a compact
element-indexed model ("sniffer lite", the shape webpilot uses).

Run:  python scripts/measure_perception.py [--headful]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from playwright.sync_api import sync_playwright  # noqa: E402

from webpilot.tokenizer import count_tokens  # noqa: E402

SNIFFER = r"""
() => {
  const name = (el) => {
    const t = (el.getAttribute('aria-label') || el.innerText || el.value ||
               el.getAttribute('placeholder') || el.getAttribute('title') || '');
    return t.replace(/\u00a0/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 120);
  };
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
  };
  const nodes = [...document.querySelectorAll(
    'a[href],button,input,select,textarea,summary,[role],[contenteditable],[onclick]')];
  const interactive = nodes.filter(visible);
  const total = document.querySelectorAll('*').length;
  const lines = interactive.map((el) => {
    const role = el.getAttribute('role') || el.tagName.toLowerCase();
    return `[${role}] ${name(el)}`;
  });
  const text = document.body ? document.body.innerText : '';
  return {totalNodes: total, interactive: interactive.length,
          compact: lines.join('\n'), text: text, html: document.documentElement.outerHTML};
}
"""


def ax_tree(page) -> str:
    """Dump the accessibility tree the way a CDP-based agent would read it."""
    cdp = page.context.new_cdp_session(page)
    tree = cdp.send("Accessibility.getFullAXTree")
    lines = []
    for node in tree.get("nodes", []):
        role = (node.get("role") or {}).get("value", "")
        nm = (node.get("name") or {}).get("value", "")
        if not nm and role in ("generic", "none", "StaticText"):
            continue
        lines.append(f"{role} {nm}".strip())
    return "\n".join(lines)


def measure(url: str, title: str) -> dict:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        page.wait_for_timeout(1200)
        raw = page.evaluate(SNIFFER)
        try:
            ax = ax_tree(page)
        except Exception as exc:  # pragma: no cover - CDP unavailable
            ax = f"<unavailable: {exc}>"
        browser.close()
    return {
        "title": title,
        "url": url,
        "dom_nodes": raw["totalNodes"],
        "interactive_nodes": raw["interactive"],
        "html_chars": len(raw["html"]),
        "html_tokens": count_tokens(raw["html"]),
        "text_chars": len(re.sub(r"\s+", " ", raw["text"]).strip()),
        "text_tokens": count_tokens(raw["text"]),
        "ax_chars": len(ax),
        "ax_tokens": count_tokens(ax),
        "compact_chars": len(raw["compact"]),
        "compact_tokens": count_tokens(raw["compact"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "docs" / "measurements.json"))
    ns = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from tests.fixtures.site.server import serve_background

    server, local_url = serve_background(0)
    targets = [
        ("SauceDemo login page (SPA-ish form)", "https://www.saucedemo.com/"),
        ("SauceDemo inventory", "https://www.saucedemo.com/inventory.html"),
        ("books.toscrape.com catalogue", "https://books.toscrape.com/"),
        ("Wikipedia article", "https://en.wikipedia.org/wiki/Web_browser"),
        ("bundled test store home", f"{local_url}/"),
        ("bundled test store checkout", f"{local_url}/checkout"),
    ]
    results = []
    for title, url in targets:
        try:
            row = measure(url, title)
        except Exception as exc:
            row = {"title": title, "url": url, "error": str(exc)[:200]}
        results.append(row)
        if "error" not in row:
            ratio = row["html_tokens"] / max(row["compact_tokens"], 1)
            print(f"{title:45s} html={row['html_tokens']:7d} tok  text={row['text_tokens']:6d}  "
                  f"ax={row['ax_tokens']:6d}  compact={row['compact_tokens']:5d}  "
                  f"({ratio:.1f}x smaller than HTML)")
    server.shutdown()
    Path(ns.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten to {ns.out}")


if __name__ == "__main__":
    main()
