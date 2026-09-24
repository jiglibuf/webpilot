"""A tiny self-contained store used by the test-suite and offline demos.

It gives the browser layer and the agent loop a real multi-page target without
touching the network: search, product pages, a cookie-backed cart, a checkout
form, client-side validation, delayed content, a modal, a shadow-root button and
an iframe.  Nothing here is known to the agent - the point is that the agent has
to discover it.

Run standalone:  python -m tests.fixtures.site.server --port 8765
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

CATALOG = [
    {"id": 1, "name": "Aeron Chair", "price": 1299.0, "rating": 5,
     "desc": "Ergonomic mesh chair with PostureFit support."},
    {"id": 2, "name": "Standing Desk", "price": 649.5, "rating": 4,
     "desc": "Electric height-adjustable desk, 120x70 cm."},
    {"id": 3, "name": "Desk Lamp", "price": 89.9, "rating": 4,
     "desc": "Warm-white LED lamp with wireless charging base."},
    {"id": 4, "name": "Mechanical Keyboard", "price": 149.0, "rating": 5,
     "desc": "75% wireless keyboard, tactile switches."},
    {"id": 5, "name": "Noise-cancelling Headphones", "price": 279.0, "rating": 3,
     "desc": "Over-ear ANC headphones, 30 h battery."},
    {"id": 6, "name": "Monitor Arm", "price": 119.0, "rating": 4,
     "desc": "Single gas-spring monitor arm up to 32 inch."},
]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 0; background: #f6f7fb; color: #16181d; }}
 header {{ background: #16181d; color: #fff; padding: 14px 24px; display: flex; gap: 18px; align-items: center; }}
 header a {{ color: #fff; text-decoration: none; opacity: .9; }}
 main {{ max-width: 900px; margin: 28px auto; padding: 0 20px; }}
 .card {{ background: #fff; border: 1px solid #e3e6ee; border-radius: 12px; padding: 18px; margin-bottom: 14px; }}
 button, .button {{ background: #2f6fed; color: #fff; border: 0; border-radius: 8px; padding: 9px 16px;
   font-size: 15px; cursor: pointer; text-decoration: none; display: inline-block; }}
 .ghost {{ background: #eef1f8; color: #16181d; }}
 input, select {{ padding: 9px; border: 1px solid #cfd4e0; border-radius: 8px; font-size: 15px; }}
 table {{ width: 100%; border-collapse: collapse; }}
 td, th {{ padding: 8px; border-bottom: 1px solid #eceff6; text-align: left; }}
 .error {{ color: #b42318; background: #fef3f2; border: 1px solid #fda29b; padding: 10px; border-radius: 8px; }}
 .price {{ font-weight: 700; }}
 #banner {{ display: none; background: #16181d; color: #fff; padding: 12px 20px; }}
 .overlay {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,.45);
   align-items: center; justify-content: center; }}
 .overlay.open {{ display: flex; }}
 .modal {{ background: #fff; border-radius: 14px; padding: 22px; max-width: 380px; }}
</style></head>
<body>
<header>
  <strong>Webpilot Test Shop</strong>
  <a href="/">Home</a>
  <a href="/search">Search</a>
  <a href="/cart">Cart</a>
  <a href="/login">Account</a>
  <a href="/lazy">Lazy</a>
  <a href="/modal">Modal</a>
  <a href="/shadow">Shadow</a>
  <a href="/iframe">Iframe</a>
</header>
<div id="banner">This site uses cookies. <button class="ghost" onclick="document.getElementById('banner').style.display='none'">Got it</button></div>
<main>{body}</main>
<script>
 window.addEventListener('error', function (e) {{ (window.__errs = window.__errs || []).push(String(e.message)); }});
</script>
</body></html>
"""


class Store:
    """Per-server state: carts keyed by the session cookie value."""

    def __init__(self) -> None:
        self.carts: dict[str, list[dict]] = {}
        self.orders: list[dict] = []
        self.lock = threading.Lock()

    def cart(self, sid: str) -> list[dict]:
        with self.lock:
            return self.carts.setdefault(sid, [])


def _page(title: str, body: str) -> bytes:
    return PAGE.format(title=title, body=body).encode()


def _product_card(p: dict) -> str:
    return (
        f'<div class="card"><a href="/product/{p["id"]}">{p["name"]}</a>'
        f'<div class="price">${p["price"]:.2f}</div>'
        f'<div>rating {p["rating"]}/5</div><div>{p["desc"]}</div></div>'
    )


class Handler(BaseHTTPRequestHandler):
    store: Store
    server_version = "WebpilotTestShop/1.0"

    # -- helpers ---------------------------------------------------------
    def _sid(self) -> str:
        cookie = self.headers.get("Cookie", "")
        for part in cookie.split(";"):
            if part.strip().startswith("sid="):
                return part.strip()[4:]
        return ""

    def _send(self, body: bytes | str, status: int = 200, ctype: str = "text/html; charset=utf-8",
              headers: dict[str, str] | None = None) -> None:
        if isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _redirect(self, location: str, headers: dict[str, str] | None = None) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _form(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode()).items()}

    def log_message(self, *args) -> None:  # keep the test output clean
        pass

    # -- routes ----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        path, query = url.path, parse_qs(url.query)

        if path == "/":
            body = ("<h1>Workspace gear</h1><p>Everything for a productive desk.</p>"
                    + "".join(_product_card(p) for p in CATALOG[:3])
                    + '<p><button id="cookie">Show cookie banner</button></p>'
                    + '<script>document.getElementById("cookie").onclick=function(){'
                      'document.getElementById("banner").style.display="block";};</script>')
            return self._send(_page("Webpilot Test Shop", body))

        if path == "/search":
            q = (query.get("q", [""])[0] or "").strip()
            hits = [p for p in CATALOG if q.lower() in p["name"].lower()] if q else CATALOG
            form = (
                '<form action="/search" method="get">'
                '<input name="q" placeholder="Search products" value="' + q + '"> '
                '<button type="submit">Search</button></form>'
            )
            if q and not hits:
                body = form + '<div class="card" role="alert">No products found for that query.</div>'
            else:
                body = form + f"<h1>Results for {q or 'everything'}</h1>" + "".join(
                    _product_card(p) for p in hits)
            return self._send(_page("Search", body))

        match = re.fullmatch(r"/product/(\d+)", path)
        if match:
            pid = int(match.group(1))
            product = next((p for p in CATALOG if p["id"] == pid), None)
            if not product:
                return self._send(_page("Not found", "<h1>Product not found</h1>"), 404)
            body = (
                f'<h1>{product["name"]}</h1><p>{product["desc"]}</p>'
                f'<p class="price">${product["price"]:.2f}</p>'
                f'<form action="/cart/add" method="post">'
                f'<input type="hidden" name="product_id" value="{pid}">'
                f'<label for="qty">Quantity</label> '
                f'<select id="qty" name="quantity"><option>1</option><option>2</option><option>3</option></select> '
                f'<button type="submit">Add to cart</button></form>'
                f'<p><a class="button ghost" href="/cart">Go to cart</a></p>'
            )
            return self._send(_page(product["name"], body))

        if path == "/cart":
            items = self.store.cart(self._sid())
            if not items:
                body = "<h1>Your cart</h1><p>Your cart is empty.</p>"
            else:
                rows = "".join(
                    f'<tr><td>{i["name"]}</td><td>{i["quantity"]}</td>'
                    f'<td>${i["price"]:.2f}</td></tr>' for i in items)
                total = sum(i["price"] * i["quantity"] for i in items)
                body = (f"<h1>Your cart</h1><table><tr><th>Item</th><th>Qty</th><th>Price</th></tr>{rows}"
                        f"<tr><td><strong>Total</strong></td><td></td><td><strong>${total:.2f}</strong></td></tr></table>"
                        '<p><a class="button" href="/checkout">Checkout</a></p>')
            return self._send(_page("Cart", body))

        if path == "/checkout":
            body = (
                "<h1>Checkout</h1>"
                '<form action="/order/place" method="post">'
                '<p><label for="fname">First name</label><br><input id="fname" name="first_name" required></p>'
                '<p><label for="lname">Last name</label><br><input id="lname" name="last_name" required></p>'
                '<p><label for="zip">ZIP / postal code</label><br><input id="zip" name="zip" required></p>'
                '<button type="submit">Place order</button></form>'
            )
            return self._send(_page("Checkout", body))

        if path == "/login":
            body = (
                "<h1>Sign in</h1>"
                '<form action="/login" method="post">'
                '<p><label for="user">Username</label><br><input id="user" name="username" autocomplete="username"></p>'
                '<p><label for="pass">Password</label><br><input id="pass" name="password" type="password"></p>'
                '<button type="submit">Sign in</button></form>'
                '<p><a href="/signup">Create an account</a></p>'
            )
            return self._send(_page("Sign in", body))

        if path == "/signup":
            return self._send(_page("Sign up", "<h1>Create an account</h1><p>Ask an administrator.</p>"))

        if path == "/account":
            return self._send(_page("Account", "<h1>My account</h1><p>Signed in as demo@example.com</p>"
                                               '<button onclick="alert(\'Order history is empty\')">Order history</button>'))

        if path == "/lazy":
            body = (
                "<h1>Delayed content</h1>"
                '<button id="load">Load recommendations</button>'
                '<div id="out">Nothing loaded yet.</div>'
                '<script>document.getElementById("load").onclick=function(){'
                'document.getElementById("out").textContent="Loading…";'
                'setTimeout(function(){document.getElementById("out").innerHTML='
                '"<strong>Recommended:</strong> Desk Lamp, Monitor Arm";},1500);};</script>'
            )
            return self._send(_page("Lazy", body))

        if path == "/modal":
            body = (
                "<h1>Modal page</h1>"
                '<div class="overlay" id="ov"><div class="modal" role="dialog" aria-label="Newsletter">'
                "<h2>Join the newsletter</h2><p>Get 10% off your first order.</p>"
                '<button id="no" class="ghost">No thanks</button> '
                '<button id="yes">Subscribe</button></div></div>'
                '<button id="open">Open newsletter dialog</button>'
                '<script>document.getElementById("open").onclick=function(){'
                'document.getElementById("ov").classList.add("open");};'
                'document.getElementById("no").onclick=function(){'
                'document.getElementById("ov").classList.remove("open");};</script>'
            )
            return self._send(_page("Modal", body))

        if path == "/shadow":
            body = (
                "<h1>Shadow DOM</h1><div id=\"host\"></div>"
                "<script>const r=document.getElementById('host').attachShadow({mode:'open'});"
                "r.innerHTML='<p>Inside the shadow root</p><button>Shadow action</button>'"
                ".replace('button','button')+'<span id=\"mark\"></span>';"
                "r.querySelector('button').onclick=function(){r.querySelector('#mark').textContent=' shadow clicked';};"
                "</script>"
            )
            return self._send(_page("Shadow", body))

        if path == "/iframe":
            body = ('<h1>Frames</h1><iframe src="/frame" width="420" height="180"></iframe>')
            return self._send(_page("Iframe", body))

        if path == "/frame":
            body = ("<h1>Inner frame</h1><p>This page lives in an iframe.</p>"
                    '<form action="/frame" method="get"><input name="note" placeholder="Note">'
                    '<button type="submit">Save note</button></form>')
            return self._send(_page("Frame", body))

        if path == "/errors":
            body = ("<h1>Console errors</h1><script>setTimeout(function(){"
                    "throw new Error('boom from the page');},100);</script>")
            return self._send(_page("Errors", body))

        return self._send(_page("Not found", "<h1>404</h1><p>Nothing here.</p>"), 404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        form = self._form()

        if path == "/cart/add":
            sid = self._sid() or "anonymous"
            product = next((p for p in CATALOG if p["id"] == int(form.get("product_id", "0"))), None)
            if not product:
                return self._send(_page("Error", '<div class="error">Unknown product.</div>'), 400)
            qty = int(form.get("quantity", "1") or 1)
            with self.store.lock:
                cart = self.store.carts.setdefault(sid, [])
                cart.append({"name": product["name"], "price": product["price"], "quantity": qty})
            return self._redirect("/cart", {"Set-Cookie": f"sid={sid}; Path=/"})

        if path == "/order/place":
            missing = [f for f in ("first_name", "last_name", "zip") if not form.get(f)]
            if missing:
                body = ('<h1>Checkout</h1><div class="error">All fields are required: '
                        + ", ".join(missing) + "</div>"
                        '<form action="/order/place" method="post">'
                        '<p><input name="first_name" placeholder="First name" required></p>'
                        '<p><input name="last_name" placeholder="Last name" required></p>'
                        '<p><input name="zip" placeholder="ZIP / postal code" required></p>'
                        '<button type="submit">Place order</button></form>')
                return self._send(_page("Checkout", body), 422)
            order = {"id": len(self.store.orders) + 1, **form}
            self.store.orders.append(order)
            return self._send(_page("Order placed",
                                    f'<h1 id="confirmation">Order #{order["id"]} placed</h1>'
                                    f'<p>Thanks {form.get("first_name")}, we emailed a receipt.</p>'))

        if path == "/login":
            if form.get("username") == "demo" and form.get("password") == "secret":
                return self._redirect("/account")
            return self._send(_page("Sign in",
                                    '<div class="error">Invalid username or password.</div>'
                                    '<form action="/login" method="post">'
                                    '<p><input name="username" placeholder="Username"></p>'
                                    '<p><input name="password" type="password" placeholder="Password"></p>'
                                    '<button type="submit">Sign in</button></form>'), 401)

        return self._send(_page("Not found", "<h1>404</h1>"), 404)

    def do_DELETE(self) -> None:  # noqa: N802
        """Used by tests that need a real destructive endpoint."""
        if urlparse(self.path).path == "/account":
            return self._send(json.dumps({"deleted": True}), 200, "application/json")
        return self._send(json.dumps({"error": "not found"}), 404, "application/json")


def make_server(port: int = 0) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"store": Store()})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def serve_background(port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Start the store on a daemon thread; returns ``(server, base_url)``."""
    server = make_server(port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def main() -> None:  # pragma: no cover - manual use
    parser = argparse.ArgumentParser(description="Webpilot test shop")
    parser.add_argument("--port", type=int, default=8765)
    ns = parser.parse_args()
    server, url = serve_background(ns.port)
    print(f"test shop on {url} (Ctrl-C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:  # pragma: no cover
        server.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
