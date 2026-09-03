import socket

import pytest

# These tests verify the loopback-only network guard installed by conftest.py:
# connections to any non-loopback host (e.g. the Hugging Face Hub) are blocked
# before a real network call happens, while loopback connections are allowed.


def test_non_loopback_connect_is_blocked():
    with pytest.raises(RuntimeError):
        socket.create_connection(("huggingface.co", 443), timeout=5)


def test_non_loopback_connect_ex_is_blocked():
    s = socket.socket()
    try:
        with pytest.raises(RuntimeError):
            s.connect_ex(("huggingface.co", 443))
    finally:
        s.close()


def test_loopback_connect_is_allowed():
    # Loopback is permitted by the guard. Port 9 (discard) is closed here, so we
    # expect a normal OSError (e.g. connection refused), not the guard's
    # RuntimeError (which is not an OSError and would propagate as a failure).
    s = socket.socket()
    try:
        with pytest.raises(OSError):
            s.connect(("127.0.0.1", 9))
    finally:
        s.close()


def test_runtime_binds_only_loopback():
    # Binding a runtime service to loopback is allowed.
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
    finally:
        s.close()
    # Binding to all interfaces (non-loopback) is blocked by the guard.
    s2 = socket.socket()
    try:
        with pytest.raises(RuntimeError):
            s2.bind(("0.0.0.0", 0))
    finally:
        s2.close()
