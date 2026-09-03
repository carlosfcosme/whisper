"""CPU-only no-store/no-fetch: WAN is blocked; loopback still works."""

import socket
import urllib.request

import pytest

import whisper
from tests.netguard import NetworkBlocked
from whisper.runtime import WeightDownloadError, default_device

HF_HUB_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"
WAN_URL = "https://openaipublic.azureedge.net/"


def test_cpu_only_default_under_blocked_network():
    assert default_device() == "cpu"
    assert whisper.default_device() == "cpu"


def test_urlopen_wan_is_blocked():
    with pytest.raises(NetworkBlocked, match="network blocked"):
        urllib.request.urlopen(HF_HUB_URL, timeout=1)


def test_urlopen_azure_cdn_is_blocked():
    with pytest.raises(NetworkBlocked, match="network blocked"):
        urllib.request.urlopen(WAN_URL, timeout=1)


def test_socket_connect_wan_is_blocked():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkBlocked, match="network blocked"):
            sock.connect(("8.8.8.8", 53))
    finally:
        sock.close()


def test_create_connection_wan_is_blocked():
    with pytest.raises(NetworkBlocked, match="network blocked"):
        socket.create_connection(("1.1.1.1", 53), timeout=1)


def test_load_model_is_no_fetch_no_store(tmp_path):
    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert list(tmp_path.iterdir()) == []
    assert list(tmp_path.rglob("*.pt")) == []
