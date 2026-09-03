"""Weights-free health server bound to loopback only.

CLI: ``whisper serve`` or ``python3 -m whisper.serve``.
Default bind is ``127.0.0.1``. All-interface hosts are refused.
Does not call ``load_model`` and does not read API keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional, Tuple

from .runtime import (
    DEFAULT_BIND_HOST,
    BindError,
    default_bind_host,
    default_device,
    is_loopback_bind_host,
    refuse_non_localhost_bind,
)

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
                "device": default_device(),
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
    host: str = DEFAULT_BIND_HOST, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    """Bind a weights-free health server. Host must be loopback."""
    refuse_non_localhost_bind(host)
    if host.strip().lower().rstrip(".") == "localhost":
        host = default_bind_host()
    httpd = ThreadingHTTPServer((host, int(port)), _HealthHandler)
    bound = httpd.server_address[0]
    if bound != DEFAULT_BIND_HOST and not is_loopback_bind_host(bound):
        httpd.server_close()
        raise BindError(
            f"Refusing non-localhost bind {bound!r}. Helper listeners must bind "
            f"{DEFAULT_BIND_HOST}."
        )
    return httpd


def serve_forever(
    host: str = DEFAULT_BIND_HOST, port: int = DEFAULT_PORT
) -> Tuple[str, int]:
    httpd = None
    try:
        httpd = create_server(host, port)
        bound_host, bound_port = httpd.server_address[:2]
        print(
            f"whisper serve listening on http://{bound_host}:{bound_port}",
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
        description=(
            "Start a weights-free health server bound to 127.0.0.1. "
            "All-interface binds are refused."
        ),
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_BIND_HOST,
        help="bind address (default: 127.0.0.1; loopback only)",
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
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
