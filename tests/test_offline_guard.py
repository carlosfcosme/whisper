import os
import socket
import urllib.request

import pytest
import torch

import whisper
from tests.offline_guard import (
    NetworkDisabledError,
    isolated_cache_root,
    user_whisper_cache,
)


def test_isolated_cache_ignores_home_whisper_dir():
    root = isolated_cache_root()
    assert root is not None
    download_root = os.path.realpath(whisper.default_download_root())
    home = user_whisper_cache()
    assert download_root != home
    assert download_root.startswith(root)
    assert os.environ["XDG_CACHE_HOME"] == root
    assert os.environ["WHISPER_ALLOW_DOWNLOAD"] == "0"


def test_load_model_does_not_use_user_weight_cache():
    home = user_whisper_cache()
    with pytest.raises(whisper.OfflineDownloadError):
        whisper.load_model("tiny", download_root=home, download=False)


def test_urlopen_model_fetch_is_refused():
    with pytest.raises(NetworkDisabledError, match="disabled in tests"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/x"
        )


def test_non_loopback_connect_is_refused():
    with pytest.raises(NetworkDisabledError, match="non-loopback connect"):
        socket.create_connection(("1.1.1.1", 443), timeout=1)


def test_loopback_bind_is_allowed():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
        assert host == "127.0.0.1"
        assert port > 0
    finally:
        sock.close()


@pytest.mark.parametrize("address", [("0.0.0.0", 0), ("", 0)])
def test_non_loopback_bind_is_refused(address):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkDisabledError, match="non-loopback bind"):
            sock.bind(address)
    finally:
        sock.close()


def test_wildcard_ipv6_bind_is_refused():
    if not socket.has_ipv6:
        pytest.skip("IPv6 not available")
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkDisabledError, match="non-loopback bind"):
            sock.bind(("::", 0))
    finally:
        sock.close()


def test_torch_hub_fetch_is_refused():
    with pytest.raises(NetworkDisabledError, match="disabled in tests"):
        torch.hub.load("pytorch/vision", "resnet18", pretrained=True)
