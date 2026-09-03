"""Weights-free health server. Binds 127.0.0.1 only. Does not fetch models."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .env_policy import BIND_HOST, BindError, require_bind_127_0_0_1

DEFAULT_PORT = 8765


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in {"/", "/health"}:
            self.send_error(404)
            return
        payload = json.dumps({"ok": True, "bind": BIND_HOST, "weights": False}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        return


def make_server(host: str = BIND_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Create the socket after the bind guard. Does not fetch checkpoints."""
    require_bind_127_0_0_1(host)
    return ThreadingHTTPServer((BIND_HOST, port), _HealthHandler)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Whisper localhost-only serve path (no weights, no WAN)."
    )
    parser.add_argument(
        "--host",
        default=BIND_HOST,
        help=f"bind address (must be {BIND_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"bind port (default {DEFAULT_PORT})",
    )
    args = parser.parse_args(argv)
    try:
        server = make_server(args.host, args.port)
    except BindError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    bound_host, bound_port = server.server_address
    print(f"whisper serve bound to {bound_host}:{bound_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
