"""Fail-on-attempt: any network or weight-download try must fail the test."""

from __future__ import annotations

import http.client
import socket
import urllib.request

import pytest

from tests.wan_guard import (
    ATTEMPTS,
    is_forbidden_url,
    is_weight_url,
    unexpected_network_attempts,
)

WEIGHT_URLS = (
    "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt",
    "https://hf.co/openai/whisper-tiny/resolve/main/model.safetensors",
    "https://openaipublic.azureedge.net/main/whisper/models/x/tiny.pt",
    "https://example.com/weights/base.pth",
)


@pytest.mark.probes_network
@pytest.mark.parametrize("url", WEIGHT_URLS)
def test_attempted_weight_urlopen_fails(url):
    assert is_weight_url(url) or is_forbidden_url(url)
    before = len(ATTEMPTS)
    with pytest.raises(RuntimeError, match="must not contact model hubs or the WAN"):
        urllib.request.urlopen(url)
    assert ATTEMPTS[before:]


@pytest.mark.probes_network
def test_attempted_urlretrieve_of_checkpoint_fails():
    url = "https://openaipublic.azureedge.net/main/whisper/models/x/tiny.pt"
    with pytest.raises(RuntimeError, match="must not contact model hubs or the WAN"):
        urllib.request.urlretrieve(url)


@pytest.mark.probes_network
def test_attempted_socket_connect_to_hub_fails():
    with pytest.raises(RuntimeError, match="must not contact model hubs or the WAN"):
        socket.socket().connect(("huggingface.co", 443))


@pytest.mark.probes_network
def test_attempted_http_client_to_cdn_fails():
    conn = http.client.HTTPSConnection("openaipublic.azureedge.net")
    with pytest.raises(RuntimeError, match="must not contact model hubs or the WAN"):
        conn.connect()


@pytest.mark.probes_network
def test_swallowed_download_attempt_is_still_recorded():
    before = len(ATTEMPTS)
    try:
        urllib.request.urlopen("https://huggingface.co/openai/whisper-tiny")
    except RuntimeError:
        pass
    recorded = ATTEMPTS[before:]
    assert recorded, "swallowed download must still be recorded"
    assert any("huggingface.co" in item for item in recorded)


def test_unmarked_test_fails_when_download_was_attempted():
    unexpected = unexpected_network_attempts(
        marker_names=["requires_cuda"],
        attempts=["https://huggingface.co/x/tiny.pt"],
    )
    assert unexpected == ["https://huggingface.co/x/tiny.pt"]


def test_probe_marker_allows_recorded_attempt():
    assert (
        unexpected_network_attempts(
            marker_names=["probes_network"],
            attempts=["https://huggingface.co/x/tiny.pt"],
        )
        == []
    )
    assert (
        unexpected_network_attempts(
            marker_names=["probes_bind"],
            attempts=["socket.bind"],
        )
        == []
    )


def test_weight_url_detection():
    assert is_weight_url("https://example.com/tiny.pt")
    assert is_weight_url("https://cdn.example/model.safetensors")
    assert not is_weight_url("https://example.com/readme.md")
    assert is_weight_url("file:///tmp/tiny.pt")
    assert not is_forbidden_url("file:///tmp/tiny.pt")
