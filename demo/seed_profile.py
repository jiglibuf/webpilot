#!/usr/bin/env python3
"""Open a site in the agent's browser profile and log in by hand.

This is deliberately *not* part of the agent: it stands in for the person who
logs in once (typing their own password) so that a later agent run continues the
saved session.  The agent itself is never allowed to type a password - it asks
the human instead (see ``docs/SECURITY.md``).

    python demo/seed_profile.py --url https://www.saucedemo.com/ \
        --user standard_user --password-env WEBPILOT_DEMO_PASSWORD

The field kinds are found generically (a password input, the first text input),
so nothing here is tied to one site.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from playwright.sync_api import sync_playwright  # noqa: E402

DEFAULT_ARGS = ["--ozone-platform=x11", "--window-size=840,900", "--no-first-run",
                "--no-default-browser-check"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--password-env", default="WEBPILOT_DEMO_PASSWORD")
    ap.add_argument("--profile-dir", default=str(Path.home() / ".webpilot" / "demo-profile"))
    ap.add_argument("--check-url-contains", default=None,
                    help="what the URL looks like after a successful login")
    args = ap.parse_args()

    password = os.environ.get(args.password_env, "")
    if not password:
        print(f"set {args.password_env} first", file=sys.stderr)
        return 2

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(args.profile_dir, headless=False,
                                                   args=DEFAULT_ARGS, viewport=None)
        page = ctx.pages[0]
        page.goto(args.url, wait_until="load")
        time.sleep(1.5)
        page.fill("input[type=text]", args.user)
        page.fill("input[type=password]", password)
        page.click("input[type=submit], button[type=submit], #login-button")
        time.sleep(2.5)
        print("now at:", page.url)
        ctx.close()
    if args.check_url_contains and args.check_url_contains not in (page.url or ""):
        print("login does not look successful", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
