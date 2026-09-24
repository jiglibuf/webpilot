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
import shutil
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
#: attempt 2+: the agent left the page in an unknown state, so reload it and fill
#: the whole form from a known start.  Reloading resets focus to the document,
#: which makes "Tab" land on the first field of the form.
DEFAULT_HUMAN_SCRIPT = [
    {"delay": 0.5, "action": "key", "keys": "ctrl+r"},
    {"delay": 4.0, "action": "wait", "keys": ""},
    {"delay": 0.4, "action": "key", "keys": "Tab"},
    {"delay": 0.5, "action": "type", "keys": "standard_user"},
    {"delay": 0.5, "action": "key", "keys": "Tab"},
    {"delay": 0.5, "action": "type", "keys": "secret_sauce"},
    {"delay": 0.5, "action": "key", "keys": "Return"},
    {"delay": 2.0, "action": "wait", "keys": ""},
]

#: attempt 1: the agent has just typed the username, so the caret is still in that
#: field - one Tab reaches the password box.  Minimal, and it looks like a person.
HUMAN_SCRIPT_FIRST = [
    {"delay": 0.6, "action": "wait", "keys": ""},
    {"delay": 0.3, "action": "key", "keys": "Tab"},
    {"delay": 0.6, "action": "type", "keys": "secret_sauce"},
    {"delay": 0.6, "action": "key", "keys": "Return"},
    {"delay": 1.5, "action": "wait", "keys": ""},
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


def ensure_wm(display: str) -> None:
    """A window manager keeps focus/activation sane - without one, Chromium can
    ignore window activation and the 'human' keystrokes land nowhere."""
    if subprocess.run("pgrep -x openbox", shell=True, capture_output=True).returncode == 0:
        return
    if shutil.which("openbox") is None:
        return
    subprocess.Popen(["openbox", "--sm-disable"], env={**os.environ, "DISPLAY": display},
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1.5)


def windows(display: str) -> dict[str, str]:
    """id -> window name for every mapped top-level window."""
    out = {}
    for wid in sh(f"DISPLAY={display} xdotool search --onlyvisible --name '.*'").split():
        out[wid] = sh(f"DISPLAY={display} xdotool getwindowname {wid}").strip()
    return out


def window_class(display: str, wid: str) -> str:
    return sh(f"DISPLAY={display} xprop -id {wid} WM_CLASS")


def find_browser_window(display: str, profile_dir: str | None = None) -> str | None:
    """Find the browser's window.

    The reliable anchor is the process: the browser was started with
    ``--user-data-dir=<profile>``, so its pid owns the window.  Guessing by class
    or by "the window that is not the terminal" picks the wrong window as soon as
    the terminal is re-mapped or a dialog appears.
    """
    if profile_dir:
        for pid in sh(f"pgrep -f 'user-data-dir={profile_dir}'").split():
            for wid in sh(f"DISPLAY={display} xdotool search --onlyvisible --pid {pid}").split():
                if "xterm" not in window_class(display, wid).lower():
                    return wid
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
    """Activate the window without --sync: with a window manager --sync can block
    for a minute, and a slow "human" makes for a very slow demo."""
    sh(f"DISPLAY={display} xdotool windowfocus {wid} 2>/dev/null")
    sh(f"DISPLAY={display} xdotool windowactivate {wid} 2>/dev/null")
    time.sleep(0.35)


def pick_encoder() -> tuple[str, list[str]]:
    """Best H.264 encoder available here: x264 -> openh264 -> mpeg4 fallback."""
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True).stdout
    except OSError:
        out = ""
    if " libx264 " in out:
        return "libx264", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "27",
                           "-pix_fmt", "yuv420p"]
    if " libopenh264 " in out:
        return "libopenh264", ["-c:v", "libopenh264", "-b:v", "3500k", "-pix_fmt", "yuv420p"]
    return "mpeg4", ["-c:v", "mpeg4", "-qscale:v", "3", "-pix_fmt", "yuv420p"]


def have_gst(element: str) -> bool:
    try:
        out = subprocess.run(["gst-inspect-1.0", element], capture_output=True, text=True).stdout
    except OSError:
        return False
    return bool(out.strip())


def have_gst_h264() -> bool:
    try:
        out = subprocess.run(["gst-inspect-1.0", "x264enc"], capture_output=True, text=True).stdout
    except OSError:
        return False
    return "x264 H.264 Encoder" in out


def start_recorder(display: str, region_size: str, out: Path, logfile):
    """Start screen recording.  x264/GStreamer -> mp4/H.264 by default; a ``.webm``
    output uses VP9, which every Linux desktop and browser can decode without the
    patent-encumbered H.264 decoders.  Stop it by sending SIGINT."""
    if out.suffix.lower() == ".webm" and have_gst("vp9enc") and have_gst("webmmux"):
        log("[rec] encoder: gstreamer vp9enc (webm/vp9)", logfile)
        return subprocess.Popen(
            ["gst-launch-1.0", "-e", "-q",
             "ximagesrc", f"display-name={display}", "use-damage=0", "!",
             "video/x-raw,framerate=15/1", "!", "videoconvert", "!",
             "video/x-raw,format=I420", "!",
             "vp9enc", "deadline=1", "cpu-used=8", "target-bitrate=3500000", "keyframe-max-dist=30", "!",
             "webmmux", "!", "filesink", f"location={out}"],
            stdin=subprocess.DEVNULL,
        )
    if have_gst("x264enc"):
        log("[rec] encoder: gstreamer x264enc (mp4/h264)", logfile)
        return subprocess.Popen(
            ["gst-launch-1.0", "-e", "-q",
             "ximagesrc", f"display-name={display}", "use-damage=0", "!",
             "video/x-raw,framerate=15/1", "!", "videoconvert", "!",
             "video/x-raw,format=I420", "!",
             "x264enc", "speed-preset=veryfast", "tune=zerolatency",
             "bitrate=4000", "key-int-max=30", "!",
             "h264parse", "!", "mp4mux", "!", "filesink", f"location={out}"],
            stdin=subprocess.DEVNULL,
        )
    encoder, encoder_args = pick_encoder()
    log(f"[rec] encoder: ffmpeg {encoder}", logfile)
    return subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "x11grab", "-framerate", "15", "-video_size", region_size,
         "-i", f"{display}+0,0",
         *encoder_args,
         str(out)],
        stdin=subprocess.PIPE,
    )


def send(display: str, wid: str, item: dict) -> None:
    """Type into a window the way a person does: activate it, then send real X
    events (XTEST).  ``xdotool --window`` (XSendEvent) is delivered by some
    toolkits only, so it is the fallback rather than the default."""
    keys = item.get("keys", "")
    if item.get("action") == "wait":
        time.sleep(float(item.get("delay", 0.5)))
        return
    focus(display, wid)
    if item.get("action") == "type":
        cmd = f"DISPLAY={display} xdotool type --delay 45 {shlex.quote(keys)}"
        fallback = f"DISPLAY={display} xdotool type --window {wid} --delay 45 {shlex.quote(keys)}"
    else:
        cmd = f"DISPLAY={display} xdotool key {keys}"
        fallback = f"DISPLAY={display} xdotool key --window {wid} {keys}"
    sh(cmd)
    if item.get("also_window"):
        sh(fallback)


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


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="record a webpilot demo")
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default=str(ROOT / "demo" / "out" / "demo.mp4"))
    ap.add_argument("--display", default=DISPLAY)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--human", default=None, help="JSON file with the scripted human actions")
    ap.add_argument("--no-record", action="store_true", help="dry run without recording")
    ap.add_argument("--keep-profile", action="store_true")
    ap.add_argument("--transcript-dir", default=str(Path.home() / ".webpilot" / "transcripts"))
    ap.add_argument("--profile-dir", default=str(Path.home() / ".webpilot" / "demo-profile"))
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--takes", type=int, default=1,
                    help="record again if the agent does not finish successfully (flaky environments)")
    return ap.parse_args(argv)


def one_take(ns) -> dict:
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    transcript_dir = Path(ns.transcript_dir)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    human_script = json.loads(Path(ns.human).read_text(encoding="utf-8")) if ns.human else DEFAULT_HUMAN_SCRIPT
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""

    if not ns.keep_profile:
        subprocess.run(["rm", "-rf", ns.profile_dir], check=False)

    ensure_xvfb(ns.display)
    ensure_wm(ns.display)
    known = set(windows(ns.display))

    env = {**os.environ, "DISPLAY": ns.display, "XDG_SESSION_TYPE": "x11",
           "WEBPILOT_BROWSER_ARGS": BROWSER_ARGS}
    env.pop("WAYLAND_DISPLAY", None)
    env.setdefault("WEBPILOT_PROVIDER", "deepseek")
    env.setdefault("WEBPILOT_MODEL", "deepseek-chat")
    if api_key:
        env.setdefault("DEEPSEEK_API_KEY", api_key)

    logfile = out.parent / "driver.log"
    logfile.write_text("", encoding="utf-8")

    rec = None
    if not ns.no_record:
        rec = start_recorder(ns.display, "1680x900", out, logfile)
        time.sleep(1.0)

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

    started = time.time()
    state: dict = {}
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
                browser_window = find_browser_window(ns.display, ns.profile_dir)
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
                    if trigger_ask and browser_window and (pending_human or human_attempts < 3):
                        human_attempts += 1
                        script = pending_human or (
                            HUMAN_SCRIPT_FIRST if human_attempts == 1 else DEFAULT_HUMAN_SCRIPT)
                        log(f"[human] logging in in the browser (attempt {human_attempts}, window {browser_window})",
                            logfile)
                        time.sleep(1.0)
                        for item in script:
                            time.sleep(float(item.get("delay", 0.3)))
                            send(ns.display, browser_window, item)
                        pending_human = []
                        human_done = True
                        log("[human] login done, telling the agent to continue", logfile)
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
    summary.update(read_result(summary["transcript"]))
    print(json.dumps(summary, indent=2))
    return summary


def read_result(transcript: str) -> dict:
    """Final result of the agent run, so the caller knows whether to keep the take."""
    if not transcript:
        return {}
    try:
        lines = [json.loads(line) for line in Path(transcript).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    except (OSError, json.JSONDecodeError):
        return {}
    results = [d for d in lines if d.get("type") == "result"]
    if not results:
        return {}
    last = results[-1]
    return {"agent_success": bool(last.get("success")),
            "agent_answer": str(last.get("answer") or "")[:400],
            "agent_steps": last.get("steps")}


def main() -> int:
    ns = parse_args()
    takes = max(1, ns.takes)
    last: dict = {}
    for attempt in range(1, takes + 1):
        if takes > 1:
            print(f"=== take {attempt}/{takes} ===", flush=True)
        last = one_take(ns)
        if last.get("agent_success"):
            break
        if attempt < takes:
            print("[demo] this take did not finish successfully - recording another one", flush=True)
            time.sleep(3)
    ok = bool(last.get("agent_success")) and last.get("video_bytes", 0) > 0 or (
        ns.no_record and last.get("agent_success"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
