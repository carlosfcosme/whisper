"""Localhost-only HTTP serve helper. Always binds 127.0.0.1 / ::1."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .localhost import serve_bind_host


class _LocalhostHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"whisper localhost-only\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def serve(host=None, port=0) -> ThreadingHTTPServer:
    """Bind a tiny HTTP server to localhost only.

    ``port=0`` lets the OS pick an ephemeral port. The caller owns shutdown.
    """
    bind_host = serve_bind_host(host)
    return ThreadingHTTPServer((bind_host, port), _LocalhostHandler)


__all__ = ["serve", "serve_bind_host"]
