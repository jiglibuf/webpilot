#!/usr/bin/env python3
"""Record a demo run of webpilot: the visible browser and the terminal side by side.

The recording session is self-contained (its own X display, own browser profile),
so it does not depend on the desktop session and is reproducible:

    +---------------------------+---------------------------+
    |  terminal: agent trace    |  Chrome: the agent's tab  |
    +---------------------------+---------------------------+

ffmpeg grabs the virtual display; this script also plays the part of the human:
it watches the run transcript and, exactly where a human would be needed, does
what the human would do - types the password into the browser after the agent
asks for help, and answers `y` at a destructive-action confirmation.

    python demo/record_demo.py --no-record --task "..."      # dry run, no video
    python demo/record_demo.py --task "..." --out demo/out/demo.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISPLAY = ":99"
SCREEN = "1680x900x24"
LEFT_W, RIGHT_W, H = 840, 840, 900

#: what the "human" does in the browser when the agent asks for a login.
#: Reloading first is deliberate: it resets focus to the document, so "Tab"
#: reliably lands on the first field of the form no matter what the agent did.
DEFAULT_HUMAN_SCRIPT = [
    {"delay": 0.5, "action": "key", "keys": "ctrl+r"},
    {"delay": 4.0, "action": "wait", "keys": ""},
    {"delay": 0.4, "action": "key", "keys": "Tab"},
    {"delay": 0.5, "action": "type", "keys": "standard_user"},
    {"delay": 0.5, "action": "key", "keys": "Tab"},
    {"delay": 0.5, "action": "type", "keys": "secret_sauce"},
    {"delay": 0.5, "action": "key", "keys": "Return"},
    {"delay": 3.0, "action": "wait", "keys": ""},
]

BROWSER_ARGS = (
    "--ozone-platform=x11 --window-position=%d,0 --window-size=%d,%d "
    "--no-first-run --no-default-browser-check --disable-infobars "
    "--disable-features=Translate,TranslateUI,TranslateRanker,AcceptCHFrame "
    "--excludeSwitches=enable-automation"
) % (LEFT_W, RIGHT_W, H)


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def log(message: str, logfile: Path | None = None) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    if logfile is not None:
        with logfile.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def ensure_xvfb(display: str) -> None:
    """Start (or restart) the virtual display so that it has the size we record."""
    wanted = SCREEN.split("x")[0] + "x" + SCREEN.split("x")[1]
    geometry = ""
    if subprocess.run(f"xdpyinfo -display {display}", shell=True, capture_output=True).returncode == 0:
        geometry = sh(f"xdpyinfo -display {display} | awk '/dimensions/ {{print $2}}'")
    if geometry == wanted:
        return
    if geometry:
        # a stale display with the wrong size: the recording would be cropped
        subprocess.run(f"pkill -f 'Xvfb {display}'", shell=True, check=False)
        time.sleep(1.5)
    subprocess.Popen(["Xvfb", display, "-screen", "0", SCREEN, "-nolisten", "tcp"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        if subprocess.run(f"xdpyinfo -display {display}", shell=True, capture_output=True).returncode == 0:
            return
        time.sleep(0.2)
    raise SystemExit(f"could not start Xvfb on {display}")


def windows(display: str) -> dict[str, str]:
    """id -> window name for every mapped top-level window."""
    out = {}
    for wid in sh(f"DISPLAY={display} xdotool search --onlyvisible --name '.*'").split():
        out[wid] = sh(f"DISPLAY={display} xdotool getwindowname {wid}").strip()
    return out


def window_class(display: str, wid: str) -> str:
    return sh(f"DISPLAY={display} xprop -id {wid} WM_CLASS")


def find_browser_window(display: str) -> str | None:
    """The browser is the mapped window that is not the recording terminal.

    Without a window manager the WM_CLASS hint is not always published, so:
    prefer an explicit chrome/chromium class, otherwise take any other mapped
    window that is at least 300 px wide.
    """
    fallback = None
    for wid, _name in windows(display).items():
        cls = window_class(display, wid).lower()
        if "xterm" in cls:
            continue
        if "chrome" in cls or "chromium" in cls:
            return wid
        geometry = sh(f"DISPLAY={display} xdotool getwindowgeometry {wid}")
        width = 0
        for token in geometry.replace("\n", " ").split():
            if token.startswith("Geometry:"):
                try:
                    width = int(geometry.split("Geometry:")[1].split("x")[0].strip())
                except (IndexError, ValueError):
                    width = 0
        if width >= 300 and fallback is None:
            fallback = wid
    return fallback


def focus(display: str, wid: str) -> None:
    sh(f"DISPLAY={display} xdotool windowfocus --sync {wid}")
    sh(f"DISPLAY={display} xdotool windowactivate --sync {wid} 2>/dev/null")


def send(display: str, wid: str, item: dict) -> None:
    keys = item.get("keys", "")
    if item.get("action") == "wait":
        time.sleep(float(item.get("delay", 0.5)))
        return
    focus(display, wid)
    if item.get("action") == "type":
        sh(f"DISPLAY={display} xdotool type --window {wid} --delay 45 {shlex.quote(keys)}")
    else:
        sh(f"DISPLAY={display} xdotool key --window {wid} {keys}")


def newest_transcript(directory: Path, after: float) -> Path | None:
    if not directory.exists():
        return None
    files = [p for p in directory.glob("run-*.jsonl") if p.stat().st_mtime >= after]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def read_new_events(path: Path, state: dict) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    offset = state.get(str(path), 0)
    chunk, state[str(path)] = text[offset:], len(text)
    events = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def main() -> int:
    ap = argparse.ArgumentParser(description="record a webpilot demo")
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default=str(ROOT / "demo" / "out" / "demo.mp4"))
    ap.add_argument("--display", default=DISPLAY)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--human", default=None, help="JSON file with the scripted human actions")
    ap.add_argument("--no-record", action="store_true", help="dry run without ffmpeg")
    ap.add_argument("--keep-profile", action="store_true")
    ap.add_argument("--transcript-dir", default=str(Path.home() / ".webpilot" / "transcripts"))
    ap.add_argument("--profile-dir", default=str(Path.home() / ".webpilot" / "demo-profile"))
    ap.add_argument("--extra-args", default="")
    ns = ap.parse_args()

    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    transcript_dir = Path(ns.transcript_dir)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    human_script = json.loads(Path(ns.human).read_text(encoding="utf-8")) if ns.human else DEFAULT_HUMAN_SCRIPT
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""

    if not ns.keep_profile:
        subprocess.run(["rm", "-rf", ns.profile_dir], check=False)

    ensure_xvfb(ns.display)
    known = set(windows(ns.display))

    env = {**os.environ, "DISPLAY": ns.display, "XDG_SESSION_TYPE": "x11",
           "WEBPILOT_BROWSER_ARGS": BROWSER_ARGS}
    env.pop("WAYLAND_DISPLAY", None)
    env.setdefault("WEBPILOT_PROVIDER", "deepseek")
    env.setdefault("WEBPILOT_MODEL", "deepseek-chat")
    if api_key:
        env.setdefault("DEEPSEEK_API_KEY", api_key)

    terminal = subprocess.Popen(
        ["xterm", "-geometry", "130x52+0+0", "-fa", "DejaVu Sans Mono", "-fs", "10",
         "-tn", "xterm-256color", "-bg", "#101216", "-fg", "#e8e8e8", "-title", "webpilot",
         "-e", "bash", "-lc",
         f"cd {shlex.quote(str(ROOT))} && source .venv/bin/activate && "
         f"clear && python -m webpilot.cli --profile-dir {shlex.quote(ns.profile_dir)} "
         f"{ns.extra_args} {shlex.quote(ns.task)}; "
         f"echo; echo '[demo] run finished'; sleep 6"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(4)
    term_window = None
    for wid in windows(ns.display):
        if "xterm" in window_class(ns.display, wid).lower():
            term_window = wid
    print("terminal window:", term_window)

    rec = None
    if not ns.no_record:
        rec = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "x11grab", "-framerate", "15", "-video_size", "1680x900",
             "-i", f"{ns.display}+0,0",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p",
             str(out)],
            stdin=subprocess.PIPE,
        )

    started = time.time()
    state: dict = {}
    logfile = out.parent / "driver.log"
    logfile.write_text("", encoding="utf-8")
    browser_window: str | None = None
    pending_human = list(human_script)
    human_done = False
    human_attempts = 0
    current_url = ""
    approvals = 0

    try:
        while time.time() - started < ns.timeout:
            if terminal.poll() is not None:
                break
            if browser_window is None:
                browser_window = find_browser_window(ns.display)
                if browser_window:
                    log(f"browser window: {browser_window} | {window_class(ns.display, browser_window)[:60]}", logfile)
            path = newest_transcript(transcript_dir, started - 5)
            if path:
                for event in read_new_events(path, state):
                    kind = event.get("event") or event.get("kind") or ""
                    data = event.get("data") or {}
                    if kind == "page" and data.get("url"):
                        current_url = str(data["url"])
                    trigger_ask = kind == "tool_call" and data.get("tool") == "ask_user"
                    trigger_confirm = kind == "confirm" and "approved" not in data
                    if trigger_ask and browser_window and (pending_human or human_attempts < 2):
                        human_attempts += 1
                        script = pending_human or DEFAULT_HUMAN_SCRIPT
                        log(f"[human] logging in in the browser (attempt {human_attempts}, window {browser_window})",
                            logfile)
                        last_url = current_url
                        time.sleep(1.0)
                        for item in script:
                            time.sleep(float(item.get("delay", 0.3)))
                            send(ns.display, browser_window, item)
                        pending_human = []
                        human_done = True
                        # only tell the agent we are done once the page really left the login screen
                        deadline = time.time() + 20
                        while time.time() < deadline and current_url == last_url:
                            time.sleep(0.5)
                            path2 = newest_transcript(transcript_dir, started - 5)
                            if path2:
                                for ev in read_new_events(path2, state):
                                    k2, d2 = ev.get("event") or ev.get("kind") or "", ev.get("data") or {}
                                    if k2 == "page" and d2.get("url"):
                                        current_url = str(d2["url"])
                        log(f"[human] login finished, page is now {current_url or '?'}", logfile)
                        if term_window:
                            time.sleep(0.8)
                            send(ns.display, term_window, {"action": "type",
                                                           "keys": "done, I logged in" if human_attempts == 1
                                                           else "I logged in again, please continue"})
                            send(ns.display, term_window, {"action": "key", "keys": "Return"})
                    elif trigger_confirm and term_window:
                        print(f"[human] {time.strftime('%H:%M:%S')} approving the destructive action")
                        time.sleep(1.2)
                        send(ns.display, term_window, {"action": "type", "keys": "y"})
                        send(ns.display, term_window, {"action": "key", "keys": "Return"})
                        approvals += 1
            time.sleep(0.3)
    finally:
        if rec is not None and rec.poll() is None:
            rec.send_signal(signal.SIGINT)
            try:
                rec.wait(timeout=25)
            except subprocess.TimeoutExpired:
                rec.kill()
        terminal.terminate()
        sh(f"DISPLAY={ns.display} pkill -f 'user-data-dir={ns.profile_dir}' 2>/dev/null")

    summary = {
        "video": str(out) if rec is not None else None,
        "video_bytes": out.stat().st_size if out.exists() else 0,
        "duration_s": round(time.time() - started, 1),
        "human_login_played": human_done,
        "confirmations_answered": approvals,
        "transcript": str(newest_transcript(transcript_dir, started - 5) or ""),
    }
    print(json.dumps(summary, indent=2))
    if ns.no_record:
        return 0
    return 0 if out.exists() and out.stat().st_size > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
