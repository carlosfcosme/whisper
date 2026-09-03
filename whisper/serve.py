"""Weights-free local health server. Binds 127.0.0.1 only."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler
from socketserver import BaseServer
from typing import List, Optional, Tuple

try:
    from .bind import (
        LOOPBACK_HOST,
        BindError,
        create_loopback_httpd,
        require_loopback_bind,
    )
except ImportError:  # file-based CI loader (no package import / no torch)
    from bind import (  # type: ignore
        LOOPBACK_HOST,
        BindError,
        create_loopback_httpd,
        require_loopback_bind,
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


def create_server(host: str = LOOPBACK_HOST, port: int = DEFAULT_PORT) -> BaseServer:
    """Bind a weights-free health server. Host must be loopback."""
    require_loopback_bind(host)
    return create_loopback_httpd(_HealthHandler, host=host, port=port)


def serve_forever(
    host: str = LOOPBACK_HOST, port: int = DEFAULT_PORT
) -> Tuple[str, int]:
    httpd: Optional[BaseServer] = None
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
            "Start a weights-free health server bound to 127.0.0.1 only. "
            "All-interface binds are refused."
        ),
    )
    parser.add_argument(
        "--host",
        default=LOOPBACK_HOST,
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
