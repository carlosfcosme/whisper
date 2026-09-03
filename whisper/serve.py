"""Health server: bind 127.0.0.1, Cache-Control no-store, no model weights."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .defaults import (
    DEFAULT_BIND_HOST,
    DEFAULT_DEVICE,
    DEFAULT_NO_STORE,
    DEFAULT_OFFLINE,
    require_loopback_host,
)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        payload = {
            "status": "ok",
            "bind": self.server.server_address[0],
            "device": DEFAULT_DEVICE,
            "offline": DEFAULT_OFFLINE,
            "no_store": DEFAULT_NO_STORE,
            "weights": False,
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def make_server(host: str = DEFAULT_BIND_HOST, port: int = 0) -> ThreadingHTTPServer:
    bound_host = require_loopback_host(host)
    return ThreadingHTTPServer((bound_host, port), _HealthHandler)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="whisper serve")
    parser.add_argument(
        "--host",
        default=DEFAULT_BIND_HOST,
        help="address to bind (127.0.0.1 only)",
    )
    parser.add_argument("--port", type=int, default=8080, help="TCP port")
    args = parser.parse_args(argv)
    try:
        httpd = make_server(args.host, args.port)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    host, port = httpd.server_address[:2]
    print(f"listening on http://{host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
