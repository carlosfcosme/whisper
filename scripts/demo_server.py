#!/usr/bin/env python3
"""Local Whisper demo HTTP server.

Always binds to loopback (127.0.0.1). Refuses 0.0.0.0 and other non-loopback
hosts so a demo never listens on the public interface.
"""

from __future__ import annotations

import argparse
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable, Optional

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = b"whisper local demo (bound to 127.0.0.1 only)\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def validate_host(host: str) -> str:
    if host not in LOOPBACK_HOSTS:
        raise ValueError(
            "demo server must bind to 127.0.0.1 (got %r); "
            "refusing non-loopback bind" % (host,)
        )
    return host


def make_server(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    host = validate_host(host)
    return ThreadingHTTPServer((host, port), DemoHandler)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Whisper demo server (loopback / 127.0.0.1 only)"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    server = make_server(args.host, args.port)
    host, port = server.server_address[:2]
    sys.stdout.write("Serving on http://%s:%s\n" % (host, port))
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
