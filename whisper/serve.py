"""Localhost-only helper server. Does not load model weights."""

import argparse
import ipaddress
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import BaseServer
from typing import List, Optional, Tuple

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class BindError(ValueError):
    """Raised when a bind host is not a loopback address."""


def normalize_bind_host(host: str) -> str:
    """Return a loopback bind address, or raise BindError.

    ``localhost`` is rewritten to ``127.0.0.1`` (no DNS). Unspecified
    addresses and non-loopback hosts are refused.
    """
    raw = (host or "").strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw:
        raise BindError("bind host is required; use 127.0.0.1")
    if raw.lower() == "localhost":
        return DEFAULT_HOST
    try:
        ip = ipaddress.ip_address(raw.split("%", 1)[0])
    except ValueError:
        raise BindError(f"refusing non-localhost bind {host!r}; use 127.0.0.1")
    if not ip.is_loopback:
        raise BindError(f"refusing non-localhost bind {host!r}; use 127.0.0.1")
    return str(ip)


def is_loopback_host(host: str) -> bool:
    try:
        normalize_bind_host(host)
        return True
    except BindError:
        return False


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
    host = normalize_bind_host(host)
    httpd = ThreadingHTTPServer((host, port), _HealthHandler)
    bound = httpd.server_address[0]
    if not is_loopback_host(bound):
        httpd.server_close()
        raise BindError(f"refusing non-localhost bind {bound!r}; use 127.0.0.1")
    return httpd


def serve_forever(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> Tuple[str, int]:
    httpd: Optional[BaseServer] = None
    try:
        httpd = create_server(host, port)
        bound_host, bound_port = httpd.server_address[:2]
        print(
            f"whisper serve listening on http://{bound_host}:{bound_port}", flush=True
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
            "Start a weights-free health server bound to loopback only "
            "(127.0.0.1 / ::1). All-interfaces binds are refused."
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
