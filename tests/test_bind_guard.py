import socket

import pytest

from whisper.bind import install_loopback_bind_guard, require_loopback_host
from whisper.serve import make_server


@pytest.mark.parametrize("host", ["0.0.0.0", "", "*", "::", "8.8.8.8"])
def test_bind_guard_rejects_non_loopback_host(host):
    with pytest.raises(ValueError, match="127.0.0.1"):
        require_loopback_host(host)


def test_socket_bind_0_0_0_0_fails_after_guard():
    install_loopback_bind_guard()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(ValueError, match="127.0.0.1"):
            sock.bind(("0.0.0.0", 0))
    finally:
        sock.close()


def test_socket_bind_127_succeeds_after_guard():
    install_loopback_bind_guard()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        assert sock.getsockname()[0] == "127.0.0.1"
        sock.listen(1)
        assert sock.getsockname()[0] != "0.0.0.0"
    finally:
        sock.close()


def test_make_server_refuses_all_interfaces():
    with pytest.raises(ValueError, match="127.0.0.1"):
        make_server("0.0.0.0", 0)
