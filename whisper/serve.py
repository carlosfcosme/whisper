"""Local helper HTTP listener. Binds 127.0.0.1 only. Does not load weights."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .runtime import (
    DEFAULT_BIND_HOST,
    BindError,
    default_bind_host,
    default_device,
    refuse_non_localhost_bind,
)

DEFAULT_PORT = 8765


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        payload = {
            "ok": True,
            "device": default_device(),
            "host": DEFAULT_BIND_HOST,
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # noqa: A003
        return


def make_server(host: Optional[str] = None, port: int = 0) -> ThreadingHTTPServer:
    """Create a listener bound to loopback. Refuses ``0.0.0.0`` and wildcards."""
    bind_host = host if host is not None else default_bind_host()
    refuse_non_localhost_bind(bind_host)
    server = ThreadingHTTPServer((bind_host, int(port)), _HealthHandler)
    bound_host = server.server_address[0]
    try:
        refuse_non_localhost_bind(bound_host)
    except BindError:
        server.server_close()
        raise
    return server


def serve(host: Optional[str] = None, port: int = DEFAULT_PORT) -> None:
    server = make_server(host=host, port=port)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Whisper helper server (127.0.0.1 only)"
    )
    parser.add_argument(
        "--host",
        default=default_bind_host(),
        help="bind address (loopback only; default 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    refuse_non_localhost_bind(args.host)
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
