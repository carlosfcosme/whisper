"""Services must bind 127.0.0.1 only."""

import ast
import socket
from pathlib import Path

import pytest

import whisper
from whisper.offline import BIND_HOST, bind_loopback, require_loopback_bind

REPO_ROOT = Path(__file__).resolve().parents[1]
_WILDCARD_V4 = ".".join(("0", "0", "0", "0"))


def test_bind_host_is_loopback():
    assert BIND_HOST == "127.0.0.1"
    assert whisper.BIND_HOST == "127.0.0.1"
    assert require_loopback_bind() == "127.0.0.1"
    assert require_loopback_bind("localhost") == "127.0.0.1"


def test_require_loopback_bind_refuses_all_interfaces():
    with pytest.raises(ValueError, match="127.0.0.1"):
        require_loopback_bind(_WILDCARD_V4)
    with pytest.raises(ValueError, match="127.0.0.1"):
        require_loopback_bind("::")
    with pytest.raises(ValueError, match="127.0.0.1"):
        require_loopback_bind("8.8.8.8")


def test_socket_bind_refuses_all_interfaces():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="127.0.0.1"):
            sock.bind((_WILDCARD_V4, 0))
    finally:
        sock.close()


def test_socket_bind_loopback_is_allowed():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        host, port = bind_loopback(sock, 0)
        assert host == "127.0.0.1"
        assert port > 0
    finally:
        sock.close()


def test_package_source_does_not_bind_all_interfaces():
    offenders = []
    wildcard = _WILDCARD_V4
    for path in (REPO_ROOT / "whisper").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == wildcard:
                offenders.append(
                    "{}:{}".format(path.relative_to(REPO_ROOT), node.lineno)
                )
    assert offenders == [], "application code must not bind {}: {}".format(
        wildcard, offenders
    )
