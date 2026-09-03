"""Loopback health helper. Binds 127.0.0.1. Cache-Control: no-store."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from .runtime import (
    cache_control_no_store,
    default_bind_host,
    default_device,
    refuse_non_localhost_bind,
)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?", 1)[0] not in ("/", "/health"):
            self.send_error(404)
            return
        payload = {
            "ok": True,
            "bind": default_bind_host(),
            "device": default_device(),
            "cache_control": cache_control_no_store(),
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", cache_control_no_store())
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def make_server(host=None, port=0):
    """Return an ``HTTPServer`` bound to loopback. Raises ``BindError`` otherwise."""
    bind_host = default_bind_host() if host is None else host
    refuse_non_localhost_bind(bind_host)
    return HTTPServer((bind_host, int(port)), _HealthHandler)


def serve(host=None, port=0):
    httpd = make_server(host, port)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


if __name__ == "__main__":
    serve()
