"""Grab preview frames out of the recorded demo using Chromium's own H.264 decoder.

ffmpeg-free on Fedora has no H.264 decoder, so the browser does the decoding.
"""
import base64
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

video = Path(sys.argv[1]).resolve()
stamps = [float(x) for x in (sys.argv[2:] or ["25", "70", "110"])]
out_dir = video.parent.parent.parent / "docs" / "images"
out_dir.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome",
                                args=["--autoplay-policy=no-user-gesture-required",
                                "--allow-file-access-from-files"])
    page = browser.new_page(viewport={"width": 1680, "height": 900})
    # a file:// page (Chrome refuses file:// subresources from about:blank)
    html = Path("/tmp/webpilot-frame.html")
    html.write_text(
        "<style>html,body{margin:0;background:#000}video{width:1680px;display:block}</style>"
        f"<video id=v muted preload=auto src='file://{video}'></video>")
    page.goto(html.as_uri())
    page.wait_for_function("() => { const v = document.getElementById('v');"
                          " return v.readyState >= 2 && v.videoWidth > 0; }", timeout=30000)
    for stamp in stamps:
        page.evaluate("(t) => new Promise(r => { const v = document.getElementById('v');"
                      " const done = () => { v.removeEventListener('seeked', done); r(v.currentTime); };"
                      " v.addEventListener('seeked', done); v.currentTime = t; })", stamp)
        page.wait_for_timeout(400)
        target = out_dir / f"demo-frame-{int(stamp)}s.png"
        page.screenshot(path=str(target))
        print("wrote", target, target.stat().st_size, "bytes")
    browser.close()
