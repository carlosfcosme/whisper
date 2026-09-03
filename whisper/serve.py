"""Localhost-only HTTP helper. Always binds 127.0.0.1. Loads no weights."""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

from .runtime import DEFAULT_BIND_HOST, BindError, serve_bind_host

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
                "device": "cpu",
                "hub": False,
                "weights": False,
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:
        return


def create_server(host=None, port=0) -> ThreadingHTTPServer:
    """Bind a weights-free health server to 127.0.0.1."""
    bind_host = serve_bind_host(host)
    httpd = ThreadingHTTPServer((bind_host, int(port)), _HealthHandler)
    bound = httpd.server_address[0]
    if bound != DEFAULT_BIND_HOST:
        httpd.server_close()
        raise BindError(
            f"Refusing bind to {bound!r}. Listeners must bind {DEFAULT_BIND_HOST}."
        )
    return httpd


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whisper serve",
        description="Weights-free health server bound to 127.0.0.1.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_BIND_HOST,
        help="bind address (default: 127.0.0.1; other hosts refused)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="bind port (default: 8765; 0 for ephemeral)",
    )
    args = parser.parse_args(argv)
    try:
        httpd = create_server(args.host, args.port)
    except BindError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    bound_host, bound_port = httpd.server_address[:2]
    print(f"whisper serve listening on http://{bound_host}:{bound_port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
