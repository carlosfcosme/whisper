import socket

import pytest

from whisper.loopback import (
    LOOPBACK_HOST,
    LoopbackBindError,
    bind_loopback,
    cli,
    is_loopback_url,
    require_loopback_host,
    start_loopback_server,
)


@pytest.mark.parametrize("host", [None, "127.0.0.1", "localhost", "LOCALHOST"])
def test_require_loopback_accepts_localhost(host):
    assert require_loopback_host(host) == LOOPBACK_HOST


def test_require_loopback_accepts_ipv6_loopback():
    assert require_loopback_host("::1") == "::1"
    assert require_loopback_host("[::1]") == "::1"


@pytest.mark.parametrize("host", ["0.0.0.0", "", "*", "::", "[::]", "example.com"])
def test_require_loopback_rejects_non_loopback(host):
    with pytest.raises(LoopbackBindError):
        require_loopback_host(host)


def test_bind_loopback_listens_on_127():
    sock = bind_loopback(0)
    try:
        sock.listen(1)
        host, port = sock.getsockname()[:2]
        assert host == LOOPBACK_HOST
        assert port > 0
        client = socket.create_connection((host, port), timeout=1)
        try:
            conn, addr = sock.accept()
            try:
                assert addr[0] == LOOPBACK_HOST
            finally:
                conn.close()
        finally:
            client.close()
    finally:
        sock.close()


def test_start_loopback_server_serves_file(tmp_path):
    payload = b"loopback-ok"
    (tmp_path / "hello.txt").write_bytes(payload)
    with pytest.raises(LoopbackBindError):
        start_loopback_server(str(tmp_path), host="0.0.0.0")

    httpd = start_loopback_server(str(tmp_path), host=LOOPBACK_HOST, port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == LOOPBACK_HOST
        with socket.create_connection((host, port), timeout=2) as sock:
            sock.sendall(b"GET /hello.txt HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
            body = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                body += chunk
        assert payload in body
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_cli_rejects_all_interfaces(tmp_path):
    with pytest.raises(LoopbackBindError):
        cli(["--host", "0.0.0.0", "--port", "0", str(tmp_path)])


def test_is_loopback_url():
    assert is_loopback_url("http://127.0.0.1:9/sha/tiny.pt")
    assert is_loopback_url("http://localhost/sha/tiny.pt")
    assert not is_loopback_url(
        "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors"
    )
    assert not is_loopback_url(
        "https://openaipublic.azureedge.net/main/whisper/models/abc/tiny.pt"
    )
