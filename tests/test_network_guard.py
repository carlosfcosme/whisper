import socket

import pytest

import whisper
from whisper.netguard import NetworkBlocked, is_loopback_connect, refuse_non_loopback
from whisper.serve import create_server


def test_loopback_connect_hosts_are_allowed():
    assert is_loopback_connect(("127.0.0.1", 80))
    assert is_loopback_connect(("localhost", 8765))
    refuse_non_loopback(("127.0.0.1", 9))


@pytest.mark.parametrize(
    "address",
    [
        ("0.0.0.0", 80),
        ("8.8.8.8", 53),
        ("1.1.1.1", 443),
        ("huggingface.co", 443),
        ("openaipublic.azureedge.net", 443),
        ("example.com", 80),
    ],
)
def test_wan_and_hub_connects_are_blocked(address):
    assert not is_loopback_connect(address)
    with pytest.raises(NetworkBlocked, match="127.0.0.1"):
        refuse_non_loopback(address)
    sock = socket.socket()
    try:
        with pytest.raises(NetworkBlocked):
            sock.connect(address)
    finally:
        sock.close()


def test_create_connection_blocks_wan():
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("8.8.8.8", 53), timeout=0.2)


def test_urlopen_blocks_model_download():
    import urllib.request

    with pytest.raises(RuntimeError, match="forbidden|blocked|no Hub|offline"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt",
            timeout=1,
        )
    with pytest.raises(RuntimeError, match="forbidden|blocked|no Hub"):
        urllib.request.urlopen(
            "https://huggingface.co/openai/whisper-tiny",
            timeout=1,
        )


def test_loopback_http_is_allowed():
    import json
    import urllib.request

    httpd = create_server("127.0.0.1", 0)
    thread = __import__("threading").Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        port = httpd.server_address[1]
        with urllib.request.urlopen(
            "http://127.0.0.1:{}/health".format(port), timeout=2
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["bind"] == "127.0.0.1"
        assert body["offline"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_load_model_cannot_download_over_network(tmp_path):
    with pytest.raises(RuntimeError, match="offline|no-store|no Hub|blocked"):
        whisper.load_model("tiny", device="cpu", download_root=str(tmp_path))
    assert list(tmp_path.glob("*.pt")) == []
