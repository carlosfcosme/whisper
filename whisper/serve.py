"""Weights-free health server bound to 127.0.0.1 only."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

from .runtime import DEFAULT_BIND_HOST, BindError, refuse_non_localhost_bind

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


def create_server(host: str = DEFAULT_BIND_HOST, port: int = DEFAULT_PORT):
    refuse_non_localhost_bind(host)
    httpd = ThreadingHTTPServer((DEFAULT_BIND_HOST, int(port)), _HealthHandler)
    actual = httpd.server_address[0]
    if actual != DEFAULT_BIND_HOST:
        httpd.server_close()
        raise BindError(
            "refusing non-localhost bind {!r}; use {}".format(actual, DEFAULT_BIND_HOST)
        )
    return httpd


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="whisper-serve")
    parser.add_argument("--host", default=DEFAULT_BIND_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    try:
        httpd = create_server(args.host, args.port)
    except BindError as exc:
        print(exc, file=sys.stderr)
        return 2
    bound_host, bound_port = httpd.server_address[:2]
    print(
        "whisper serve listening on http://{}:{}".format(bound_host, bound_port),
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
