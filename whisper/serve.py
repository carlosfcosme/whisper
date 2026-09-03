"""Weights-free health server bound to 127.0.0.1 only.

CLI: ``whisper serve`` or ``python3 -m whisper.serve``.
Does not call ``load_model``, does not download checkpoints, and
does not accept ``--live``. Binding an all-interface host is refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler
from typing import List, Optional

from .bind import (
    LOOPBACK_HOST,
    BindError,
    assert_no_nonloopback_listeners,
    create_loopback_httpd,
    require_loopback_host,
)

DEFAULT_PORT = 8765


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in {"/", "/health"}:
            self.send_error(404)
            return
        payload = json.dumps(
            {
                "status": "ok",
                "bind": self.server.server_address[0],
                "weights": False,
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def create_server(host: str = LOOPBACK_HOST, port: int = DEFAULT_PORT):
    """Bind the health server. Host must be 127.0.0.1."""
    require_loopback_host(host)
    httpd = create_loopback_httpd(_HealthHandler, host=host, port=port)
    assert_no_nonloopback_listeners()
    return httpd


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--live" in argv:
        print("error: --live is refused until proven", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(
        prog="whisper serve",
        description=(
            "Start a weights-free health server bound to 127.0.0.1 only. "
            "All-interface and non-loopback binds are refused."
        ),
    )
    parser.add_argument(
        "--host",
        default=LOOPBACK_HOST,
        help="bind address (default: 127.0.0.1; 127.0.0.1 only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="bind port (default: 8765)",
    )
    args = parser.parse_args(argv)
    try:
        httpd = create_server(args.host, args.port)
    except BindError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    bound_host, bound_port = httpd.server_address[:2]
    print(
        "whisper serve listening on http://%s:%s" % (bound_host, bound_port),
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
