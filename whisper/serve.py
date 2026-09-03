"""Weights-free health server bound to 127.0.0.1 only."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

from .bind import LOOPBACK_HOST, BindError, require_loopback_host

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
                "network": "loopback",
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def create_server(
    host: Optional[str] = None, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    host = require_loopback_host(host if host is not None else LOOPBACK_HOST)
    httpd = ThreadingHTTPServer((host, port), _HealthHandler)
    bound = httpd.server_address[0]
    try:
        require_loopback_host(bound)
    except BindError:
        httpd.server_close()
        raise
    return httpd


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whisper serve",
        description="Weights-free health server on 127.0.0.1 only.",
    )
    parser.add_argument("--host", default=LOOPBACK_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    httpd = None
    try:
        httpd = create_server(args.host, args.port)
        bound_host, bound_port = httpd.server_address[:2]
        print(
            f"whisper serve listening on http://{bound_host}:{bound_port}", flush=True
        )
        httpd.serve_forever()
        return 0
    except BindError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    finally:
        if httpd is not None:
            httpd.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
