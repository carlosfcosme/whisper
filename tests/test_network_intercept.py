"""Offline tests fail if the network interceptor lets a remote connect through."""

import socket
import urllib.request

import pytest

from tests.network_intercept import (
    NetworkIntercepted,
    deny_if_remote,
    installed,
    is_loopback_peer,
    remote_hosts,
)


def test_interceptor_is_installed():
    assert installed()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_peers_are_allowed(host):
    assert is_loopback_peer(host)
    deny_if_remote((host, 9))


@pytest.mark.parametrize("host", list(remote_hosts()))
def test_remote_connect_is_intercepted(host):
    with pytest.raises(NetworkIntercepted, match="offline intercept"):
        deny_if_remote((host, 443))
    sock = socket.socket()
    try:
        with pytest.raises(NetworkIntercepted, match="offline intercept"):
            sock.connect((host, 443))
    finally:
        sock.close()


def test_urlopen_hub_is_intercepted():
    with pytest.raises(NetworkIntercepted):
        urllib.request.urlopen("https://huggingface.co/openai/whisper", timeout=1)


def test_urlopen_azure_weight_is_intercepted():
    with pytest.raises(NetworkIntercepted):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt",
            timeout=1,
        )


def test_create_connection_remote_is_intercepted():
    with pytest.raises(NetworkIntercepted):
        socket.create_connection(("huggingface.co", 443), timeout=1)
