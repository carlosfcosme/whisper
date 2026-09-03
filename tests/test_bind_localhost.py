"""Helpers bind 127.0.0.1 only. Does not load weights."""

import socket

import pytest

from whisper.bind import LOCALHOST, bind_host


def test_bind_host_defaults_to_localhost():
    assert LOCALHOST == "127.0.0.1"
    assert bind_host() == "127.0.0.1"
    assert bind_host(None) == "127.0.0.1"


def test_bind_host_rejects_wildcards():
    for host in ("0.0.0.0", "::", "[::]"):
        with pytest.raises(ValueError, match="127.0.0.1"):
            bind_host(host)


def test_bind_host_opens_localhost_socket():
    host = bind_host()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind((host, 0))
        bound_host, port = server.getsockname()
        assert bound_host == "127.0.0.1"
        assert port > 0
    finally:
        server.close()
