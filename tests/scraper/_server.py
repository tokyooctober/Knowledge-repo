"""A tiny threaded HTTP server standing in for the paywalled site in login/crawler tests.

`state.authed` gates the member content. `state.flip_after` simulates the human signing
in: after that many hits to the login page it flips `authed` True, and the login page
auto-reloads (a `<script>` in the form) so the next poll sees `.member-content` — exactly
as a real site redirects you after sign-in, with no concurrent task in the test.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_LOGIN_FORM = (
    "<html><body><h1>Sign in</h1>"
    '<form><input name="username"><input type="password" name="pwd"><button>Go</button></form>'
    # a real site redirects you off the login page after sign-in; here the page polls
    # itself so the login poll sees .member-content once the server flips `authed`.
    "<script>setTimeout(function(){location.reload()}, 600)</script>"
    "</body></html>"
)

_PROSE = (
    "<p>The money supply expanded by roughly forty percent during the pandemic era, an "
    "unprecedented pace of central-bank balance-sheet growth that reshaped the outlook for "
    "inflation and real yields across the developed world. " * 6 + "</p>"
)


def _member_page(title: str) -> str:
    return f"<html><body><div class='member-content'><h1>{title}</h1>{_PROSE}</div></body></html>"


class _State:
    def __init__(self) -> None:
        self.authed = False
        self.flip_after: int | None = None
        self.login_hits = 0
        self.requests: list[str] = []


class _Handler(BaseHTTPRequestHandler):
    state: _State

    def log_message(self, *a):
        pass

    def _send(self, body: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body.encode())

    def do_GET(self):  # noqa: N802
        st = self.state
        path = self.path.split("?")[0]
        st.requests.append(path)

        if path == "/404-page":
            return self._send("<html><body><h1>Gone</h1></body></html>", 404)
        if path == "/public":
            return self._send("<html><body>a public page, no member content</body></html>")

        is_login_surface = path in ("/login", "/members/", "/premium-x", "/premium-y")
        if is_login_surface and not st.authed:
            st.login_hits += 1
            if st.flip_after is not None and st.login_hits >= st.flip_after:
                st.authed = True

        if not st.authed:
            return self._send(_LOGIN_FORM)

        if path == "/dashboard" or path == "/login":
            return self._send(_member_page("Welcome back"))  # authed but NOT the article
        if path == "/members/":
            return self._send(_member_page("Members area"))
        if path in ("/premium-x", "/premium-y"):
            return self._send(_member_page(f"Report {path[-1].upper()}"))
        return self._send("<html><body>authed, unknown page</body></html>")


@contextmanager
def fixture_site():
    state = _State()
    server = ThreadingHTTPServer(("127.0.0.1", 0), type("H", (_Handler,), {"state": state}))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield type("Site", (), {"base": base, "state": state})()
    finally:
        server.shutdown()
        thread.join(timeout=2)
