"""Serve path binds 127.0.0.1 only. Does not load weights."""

import socket
from threading import Thread

import pytest

from whisper.bind import BIND_HOST, BindError
from whisper.serve import make_server


def test_make_server_binds_loopback():
    server = make_server(host=BIND_HOST, port=0)
    thread = Thread(target=server.handle_request)
    try:
        host, port = server.server_address
        assert host == "127.0.0.1"
        assert port > 0
        thread.start()
        with socket.create_connection((host, port), timeout=2) as client:
            client.sendall(b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            body = client.recv(4096).decode("utf-8")
        assert '"bind": "127.0.0.1"' in body
        assert '"hub": false' in body
        assert '"device": "cpu"' in body
    finally:
        server.server_close()
        thread.join(timeout=2)


def test_make_server_refuses_wildcard():
    with pytest.raises(BindError, match="127.0.0.1"):
        make_server(host="0.0.0.0", port=0)


def test_serve_main_refuses_wildcard():
    from whisper.serve import main

    with pytest.raises(SystemExit) as exc:
        main(["--host", "0.0.0.0", "--port", "0"])
    assert exc.value.code == 2
