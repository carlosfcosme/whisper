"""Loopback-only health listener. Binds 127.0.0.1; does not load weights."""

import argparse
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from .runtime import BindError, default_bind_host, refuse_non_localhost_bind

_HEALTH_BODY = b"ok\n"


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?", 1)[0] in ("/", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(_HEALTH_BODY)))
            self.end_headers()
            self.wfile.write(_HEALTH_BODY)
            return
        self.send_error(404)

    def log_message(self, format, *args):
        return


def create_server(host=None, port=0) -> HTTPServer:
    """Return an HTTP server bound to loopback only (default ``127.0.0.1``)."""
    bind_host = default_bind_host() if host is None else host
    refuse_non_localhost_bind(bind_host)
    return HTTPServer((bind_host, int(port)), HealthHandler)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve a health endpoint bound to 127.0.0.1 (no model weights)."
    )
    parser.add_argument(
        "--host",
        default=None,
        help="bind host (loopback only; default 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="bind port (default 8765)",
    )
    args = parser.parse_args(argv)
    try:
        server = create_server(args.host, args.port)
    except BindError as exc:
        print(exc, file=sys.stderr)
        return 2
    host, port = server.server_address[:2]
    print(f"whisper serve listening on http://{host}:{port}/health", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
