"""Helpers bind 127.0.0.1 only. Does not load weights."""

import socket
from pathlib import Path

import pytest

from whisper.bind import LOCALHOST, bind_host, listen


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


def test_listen_is_localhost_only():
    server = listen()
    try:
        bound_host, port = server.getsockname()
        assert bound_host == "127.0.0.1"
        assert port > 0
    finally:
        server.close()


def test_listen_rejects_0_0_0_0():
    with pytest.raises(ValueError, match="127.0.0.1"):
        listen(host="0.0.0.0")


def test_package_does_not_bind_wildcard():
    root = Path(__file__).resolve().parents[1] / "whisper"
    hits = []
    for path in root.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if "0.0.0.0" in line and "WILDCARD" not in line and "frozenset" not in line:
                hits.append("{0}:{1}:{2}".format(path.name, lineno, line.strip()))
    assert hits == [], "package must not listen on 0.0.0.0: {0}".format(hits)
