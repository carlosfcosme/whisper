"""Weights-free health server. Binds 127.0.0.1 only. Does not load models."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Tuple

from .bind import LOOPBACK_HOST, BindError, require_loopback

DEFAULT_PORT = 8765


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] not in {"/", "/health"}:
            self.send_error(404, "not found")
            return
        payload = {
            "status": "ok",
            "host": LOOPBACK_HOST,
            "weights": False,
            "hub": False,
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))


def create_server(
    host: str = LOOPBACK_HOST, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    bound = require_loopback(host)
    return ThreadingHTTPServer((bound, port), HealthHandler)


def serve_forever(host: str = LOOPBACK_HOST, port: int = DEFAULT_PORT) -> None:
    server = create_server(host, port)
    bound_host, bound_port = server.server_address[:2]
    print(f"whisper serve listening on http://{bound_host}:{bound_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: Tuple[str, ...] = None) -> int:
    parser = argparse.ArgumentParser(prog="whisper serve")
    parser.add_argument(
        "--host",
        default=LOOPBACK_HOST,
        help="bind address (must be 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        require_loopback(args.host)
    except BindError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    serve_forever(args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
