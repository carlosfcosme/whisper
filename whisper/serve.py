"""Cloud Agent helper: loopback-only startup, no model-weight download."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator, Optional

from .runtime import (
    DEFAULT_BIND_HOST,
    BindError,
    default_bind_host,
    default_device,
    refuse_non_localhost_bind,
    weight_auto_download_allowed,
)


class _HealthHandler(BaseHTTPRequestHandler):
    """GET / and GET /health. Does not load checkpoints or open the Hub."""

    def do_GET(self):
        if self.path not in {"/", "/health"}:
            self.send_error(404)
            return
        payload = {
            "status": "ok",
            "bind": self.server.server_address[0],
            "device": default_device(),
            "weights": False,
            "offline": not weight_auto_download_allowed(),
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def _ipv4_loopback_host(host: Optional[str]) -> str:
    """Resolve a bind host to IPv4 loopback, or refuse before a socket opens."""
    bind_host = default_bind_host() if host is None else host
    refuse_non_localhost_bind(bind_host)
    if bind_host.strip().lower().rstrip(".") in {"localhost", "::1"}:
        return DEFAULT_BIND_HOST
    return bind_host


def create_server(host: Optional[str] = None, port: int = 0) -> ThreadingHTTPServer:
    """Bind the Cloud Agent helper to loopback. Does not download weights.

    Non-loopback hosts (``0.0.0.0``, ``::``, LAN, public) raise ``BindError``
    before ``listen``.
    """
    bind_host = _ipv4_loopback_host(host)
    server = ThreadingHTTPServer((bind_host, int(port)), _HealthHandler)
    refuse_non_localhost_bind(server.server_address[0])
    return server


@contextmanager
def cloud_agent_startup(
    host: Optional[str] = None, port: int = 0
) -> Iterator[ThreadingHTTPServer]:
    """Start the Cloud Agent helper on loopback without fetching model weights."""
    server = create_server(host=host, port=port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cloud Agent helper (127.0.0.1 only, no weight download)"
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        server = create_server(host=args.host, port=args.port)
    except BindError as exc:
        sys.stderr.write("%s\n" % exc)
        return 2
    host, port = server.server_address[:2]
    sys.stdout.write("listening on %s:%s\n" % (host, port))
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
