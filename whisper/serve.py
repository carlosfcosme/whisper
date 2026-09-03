"""Weights-free health server that binds 127.0.0.1 only."""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional, Tuple

from .bind import BIND_HOST, BindError, require_bind_host
from .defaults import DEFAULT_PORT

DEFAULT_HOST = BIND_HOST


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
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    """Bind a weights-free health server. Host must be 127.0.0.1."""
    host = require_bind_host(host)
    httpd = ThreadingHTTPServer((host, port), _HealthHandler)
    bound = httpd.server_address[0]
    try:
        require_bind_host(bound)
    except BindError:
        httpd.server_close()
        raise
    return httpd


def serve_forever(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> Tuple[str, int]:
    httpd = None
    try:
        httpd = create_server(host, port)
        bound_host, bound_port = httpd.server_address[:2]
        print(
            "whisper serve listening on http://{0}:{1}".format(bound_host, bound_port),
            flush=True,
        )
        httpd.serve_forever()
        return bound_host, bound_port
    finally:
        if httpd is not None:
            httpd.server_close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whisper serve",
        description="Health server bound to 127.0.0.1 only. 0.0.0.0 is refused.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="bind port (default: 8765)",
    )
    args = parser.parse_args(argv)
    try:
        serve_forever(args.host, args.port)
    except BindError as exc:
        print("error: {0}".format(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
