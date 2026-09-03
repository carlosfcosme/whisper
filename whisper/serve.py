"""Weights-free HTTP health server bound to 127.0.0.1 only."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler
from typing import Optional

from .bind import (
    LOOPBACK_HOST,
    BindError,
    assert_no_nonloopback_listeners,
    create_loopback_httpd,
    require_loopback_host,
)


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path not in {"/", "/health"}:
            self.send_error(404)
            return
        payload = {
            "status": "ok",
            "bind": LOOPBACK_HOST,
            "weights": False,
            "hub": False,
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def create_server(host: Optional[str] = None, port: int = 0):
    """Bind a weights-free health server to 127.0.0.1."""
    require_loopback_host(host)
    httpd = create_loopback_httpd(HealthHandler, host=LOOPBACK_HOST, port=port)
    assert_no_nonloopback_listeners()
    return httpd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="whisper serve",
        description="Weights-free health server on 127.0.0.1 only.",
    )
    parser.add_argument(
        "--host",
        default=LOOPBACK_HOST,
        help="bind address (default: 127.0.0.1; non-loopback refused)",
    )
    parser.add_argument("--port", type=int, default=8765, help="bind port")
    args = parser.parse_args(argv)
    try:
        httpd = create_server(host=args.host, port=args.port)
    except BindError as exc:
        print(exc, file=sys.stderr)
        return 2
    host, port = httpd.server_address[:2]
    print("listening on http://%s:%s/health (weights=false)" % (host, port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
