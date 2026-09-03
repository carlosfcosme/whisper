"""Network block for model-weight downloads. Offline: no WAN, no keys."""

from __future__ import annotations

import socket
import urllib.error
import urllib.request

import pytest


def _azure_tiny_url() -> str:
    host = "openaipublic." + "azureedge" + ".net"
    return "https://{}/main/whisper/models/deadbeef/tiny.pt".format(host)


def _hf_model_url() -> str:
    host = "huggingface" + ".co"
    return "https://{}/openai/whisper-tiny/resolve/main/model.safetensors".format(host)


def test_azure_weight_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="download model weights"):
        urllib.request.urlopen(_azure_tiny_url())


def test_hf_weight_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="HuggingFace Hub|download model weights"):
        urllib.request.urlopen(_hf_model_url())


def test_azure_weight_socket_is_blocked():
    host = "openaipublic." + "azureedge" + ".net"
    with pytest.raises(RuntimeError, match="download model weights"):
        socket.create_connection((host, 443), timeout=1)


def test_weight_suffix_urlopen_is_blocked_without_wan():
    with pytest.raises(RuntimeError, match="download model weights"):
        urllib.request.urlopen("https://example.invalid/tiny.pt")


def test_loopback_health_urlopen_is_not_blocked():
    with pytest.raises((OSError, ConnectionError, TimeoutError, urllib.error.URLError)):
        urllib.request.urlopen("http://127.0.0.1:1/health", timeout=0.2)
