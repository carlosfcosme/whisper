"""Weights-free health server. Binds loopback only."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

from .runtime import BIND_HOST, BindError, serve_bind_host

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


def create_server(
    host: str = BIND_HOST, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    host = serve_bind_host(host)
    return ThreadingHTTPServer((host, port), _HealthHandler)


def serve(host: Optional[str] = None, port: int = 0) -> ThreadingHTTPServer:
    return create_server(serve_bind_host(host), port)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whisper-serve",
        description="Weights-free health server bound to 127.0.0.1 only.",
    )
    parser.add_argument("--host", default=BIND_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    try:
        httpd = create_server(args.host, args.port)
    except BindError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    try:
        bound_host, bound_port = httpd.server_address[:2]
        print(
            "whisper serve listening on http://{}:{}".format(bound_host, bound_port),
            flush=True,
        )
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
