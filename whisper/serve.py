"""Weights-free helper server. Binds 127.0.0.1 only."""

import argparse
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from .runtime import (
    DEFAULT_BIND_HOST,
    BindError,
    default_bind_host,
    refuse_non_localhost_bind,
)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"ok\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class LoopbackHTTPServer(HTTPServer):
    def server_bind(self):
        refuse_non_localhost_bind(self.server_address[0])
        return HTTPServer.server_bind(self)


def make_server(host=None, port=0):
    bind_host = refuse_non_localhost_bind(host or default_bind_host())
    if bind_host == "localhost":
        bind_host = DEFAULT_BIND_HOST
    return LoopbackHTTPServer((bind_host, port), _HealthHandler)


def serve_cli(argv=None):
    parser = argparse.ArgumentParser(prog="whisper serve")
    parser.add_argument(
        "--host",
        default=default_bind_host(),
        help="bind address (127.0.0.1 only)",
    )
    parser.add_argument("--port", type=int, default=8080, help="bind port")
    args = parser.parse_args(argv)

    try:
        httpd = make_server(args.host, args.port)
    except BindError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    host, port = httpd.server_address[:2]
    print("whisper serve listening on http://{}:{}".format(host, port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(serve_cli())
