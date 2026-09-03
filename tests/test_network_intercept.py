"""Python socket intercept blocks Hub/WAN; loopback stays allowed."""

import socket
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from whisper.offline import BIND_HOST, is_blocked_network_host, is_hub_host


class _OkHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"loopback-ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def test_hub_and_weight_hosts_are_blocked():
    assert is_hub_host("huggingface.co")
    assert is_hub_host("cas-bridge.xethub.hf.co")
    assert is_blocked_network_host("openaipublic.azureedge.net")
    assert is_blocked_network_host("1.1.1.1")
    assert not is_blocked_network_host("127.0.0.1")
    assert not is_blocked_network_host("localhost")


def test_connect_to_hub_is_intercepted():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="intercept"):
            sock.connect(("huggingface.co", 443))
    finally:
        sock.close()


def test_create_connection_to_weight_cdn_is_intercepted():
    with pytest.raises(RuntimeError, match="intercept"):
        socket.create_connection(("openaipublic.azureedge.net", 443), timeout=1)


def test_getaddrinfo_for_hub_is_intercepted():
    with pytest.raises(RuntimeError, match="DNS refused"):
        socket.getaddrinfo("huggingface.co", 443)


def test_getaddrinfo_for_loopback_is_allowed():
    infos = socket.getaddrinfo("127.0.0.1", 0, socket.AF_INET, socket.SOCK_STREAM)
    assert infos
    assert infos[0][4][0] == "127.0.0.1"


def test_loopback_http_roundtrip():
    server = HTTPServer((BIND_HOST, 0), _OkHandler)
    host, port = server.server_address[:2]
    assert host == "127.0.0.1"
    thread = Thread(target=server.handle_request)
    thread.daemon = True
    thread.start()
    try:
        conn = HTTPConnection(host, port, timeout=5)
        conn.request("GET", "/")
        response = conn.getresponse()
        body = response.read()
        conn.close()
        assert response.status == 200
        assert body == b"loopback-ok"
    finally:
        server.server_close()
        thread.join(timeout=5)
