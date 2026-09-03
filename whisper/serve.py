"""Weights-free health server bound to 127.0.0.1 only.

Does not call load_model, does not download checkpoints, and does not
read secrets. Wildcard and public bind hosts are refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional, Type

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
    """Bind the health server. Host must be loopback; socket is 127.0.0.1."""
    refuse_non_localhost_bind(host)
    httpd = ThreadingHTTPServer((DEFAULT_BIND_HOST, int(port)), _HealthHandler)
    actual = httpd.server_address[0]
    if actual != DEFAULT_BIND_HOST:
        httpd.server_close()
        raise BindError(
            "refusing non-localhost bind {!r}; use {}".format(actual, DEFAULT_BIND_HOST)
        )
    return httpd


def create_loopback_httpd(
    handler: Type[BaseHTTPRequestHandler],
    host: Optional[str] = None,
    port: int = 0,
) -> ThreadingHTTPServer:
    """Bind *handler* to ``127.0.0.1``. ``host`` must pass the policy."""
    chosen = DEFAULT_BIND_HOST if host is None else host
    refuse_non_localhost_bind(chosen)
    httpd = ThreadingHTTPServer((DEFAULT_BIND_HOST, int(port)), handler)
    bound = httpd.server_address[0]
    if bound != DEFAULT_BIND_HOST:
        httpd.server_close()
        raise BindError(
            "refusing non-localhost bind {!r}; use {}".format(bound, DEFAULT_BIND_HOST)
        )
    return httpd


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="whisper-serve",
        description=(
            "Start a weights-free health server bound to 127.0.0.1 only. "
            "Wildcard and public binds are refused."
        ),
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_BIND_HOST,
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
