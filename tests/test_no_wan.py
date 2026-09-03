"""Prove normal tests do not contact model hubs or the WAN."""

from __future__ import annotations

import socket
import urllib.request

import pytest

from tests.wan_guard import HUB_AND_CDN_MARKERS, is_forbidden_url, is_loopback_host


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/openai/whisper-tiny",
        "https://hf.co/models",
        "https://cdn-lfs.huggingface.co/file",
        "https://openaipublic.azureedge.net/main/whisper/models/x/tiny.pt",
        "https://openaipublic.blob.core.windows.net/main/whisper/models/x/tiny.pt",
        "https://example.com/tiny.pt",
    ],
)
def test_urlopen_refuses_hub_and_wan(url):
    assert is_forbidden_url(url)
    with pytest.raises(RuntimeError, match="must not contact model hubs or the WAN"):
        urllib.request.urlopen(url)


def test_urlopen_allows_file_scheme(tmp_path):
    payload = b"fixture"
    path = tmp_path / "local.bin"
    path.write_bytes(payload)
    with urllib.request.urlopen(path.as_uri()) as handle:
        assert handle.read() == payload


def test_create_connection_refuses_wan_literal():
    with pytest.raises(RuntimeError, match="must not contact model hubs or the WAN"):
        socket.create_connection(("1.1.1.1", 443), timeout=0.2)


def test_create_connection_refuses_hub_hostname():
    with pytest.raises(RuntimeError, match="must not contact model hubs or the WAN"):
        socket.create_connection(("huggingface.co", 443), timeout=0.2)


def test_loopback_host_helper():
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("localhost")
    assert is_loopback_host("::1")
    assert not is_loopback_host("huggingface.co")
    assert not is_loopback_host("8.8.8.8")
    assert not is_loopback_host("openaipublic.azureedge.net")


def test_hub_markers_cover_known_hosts():
    joined = " ".join(HUB_AND_CDN_MARKERS)
    assert "huggingface.co" in joined
    assert "hf.co" in joined
    assert "azureedge.net" in joined


def test_huggingface_hub_helpers_are_blocked_if_imported(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("huggingface_hub")

    def real_download(*args, **kwargs):
        raise AssertionError("real hf_hub_download ran")

    fake.hf_hub_download = real_download
    fake.snapshot_download = real_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    from tests.wan_guard import install_hub_client_guard

    install_hub_client_guard(monkeypatch)
    with pytest.raises(RuntimeError, match="must not contact model hubs"):
        fake.hf_hub_download("openai/whisper-tiny")
