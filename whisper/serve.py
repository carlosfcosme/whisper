"""Localhost-only helper server. Does not load model weights or contact Hub."""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import BaseServer
from typing import List, Optional, Tuple

from .runtime import (
    DEFAULT_BIND_HOST,
    DEFAULT_BIND_PORT,
    BindError,
    default_bind_host,
    default_device,
    refuse_non_localhost_bind,
)

DEFAULT_HOST = DEFAULT_BIND_HOST
DEFAULT_PORT = DEFAULT_BIND_PORT


def normalize_bind_host(host: str) -> str:
    """Return a loopback IPv4 host, or raise BindError.

    ``localhost`` is rewritten to ``127.0.0.1`` (no DNS). Unspecified
    addresses such as ``0.0.0.0`` and ``::`` are refused, as are LAN and
    public hosts. IPv6 loopback ``::1`` is refused because
    ``ThreadingHTTPServer`` is AF_INET-only.
    """
    raw = (host or "").strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        raise BindError("bind host is required; use 127.0.0.1")
    if raw.lower().rstrip(".") == "localhost":
        return DEFAULT_HOST
    if raw == "::1":
        raise BindError("refusing IPv6 bind {!r}; use 127.0.0.1".format(host))
    refuse_non_localhost_bind(raw)
    return raw


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
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def create_server(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    """Bind a weights-free health server. Host must be loopback."""
    host = normalize_bind_host(host)
    httpd = ThreadingHTTPServer((host, int(port)), _HealthHandler)
    bound = httpd.server_address[0]
    try:
        refuse_non_localhost_bind(bound)
    except BindError:
        httpd.server_close()
        raise
    return httpd


def serve_forever(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> Tuple[str, int]:
    httpd: Optional[BaseServer] = None
    try:
        httpd = create_server(host, port)
        bound_host, bound_port = httpd.server_address[:2]
        print(
            "whisper serve listening on http://{}:{}".format(bound_host, bound_port),
            flush=True,
        )
        httpd.serve_forever()
        return bound_host, bound_port
    finally:
        if httpd is not None:
            httpd.server_close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whisper-serve",
        description=(
            "Start a weights-free health server bound to 127.0.0.1 only. "
            "Binding 0.0.0.0 is refused. No Hub."
        ),
    )
    parser.add_argument(
        "--host",
        default=None,
        help="bind address (default: 127.0.0.1 via default_bind_host(); loopback only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="bind port (default: 8765)",
    )
    args = parser.parse_args(argv)
    host = args.host if args.host is not None else default_bind_host()
    try:
        serve_forever(host, args.port)
    except BindError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
