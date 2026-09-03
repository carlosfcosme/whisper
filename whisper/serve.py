"""Localhost-only helper server. Does not load model weights."""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

from .bind_guard import (
    DEFAULT_HOST,
    BindError,
    bind_guard,
    is_loopback_host,
    normalize_bind_host,
)

DEFAULT_PORT = 8765

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "BindError",
    "bind_guard",
    "is_loopback_host",
    "normalize_bind_host",
    "create_server",
    "main",
]


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
    """Bind a weights-free health server. Host must be loopback."""
    host = bind_guard(host)
    httpd = ThreadingHTTPServer((host, port), _HealthHandler)
    bound = httpd.server_address[0]
    if not is_loopback_host(bound):
        httpd.server_close()
        raise BindError(f"refusing non-localhost bind {bound!r}; use 127.0.0.1")
    return httpd


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whisper serve",
        description=(
            "Start a weights-free health server bound to loopback only "
            "(127.0.0.1 / ::1). Binding 0.0.0.0 is refused."
        ),
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="bind address (default: 127.0.0.1; loopback only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="bind port (default: 8765)",
    )
    args = parser.parse_args(argv)
    httpd = None
    try:
        httpd = create_server(args.host, args.port)
        bound_host, bound_port = httpd.server_address[:2]
        print(
            f"whisper serve listening on http://{bound_host}:{bound_port}",
            flush=True,
        )
        httpd.serve_forever()
    except BindError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    finally:
        if httpd is not None:
            httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
