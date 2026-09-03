import os
import socket
import urllib.parse
import urllib.request

import pytest

import whisper

# Verifies whisper's runtime weight-loading is offline/fail-closed and that the
# environment only permits loopback connections. These rely on the loopback-only
# network guard installed by conftest.py (active for non-`requires_network` tests).

_MODEL = "tiny.en"
_MODEL_URL = whisper._MODELS[_MODEL]
_MODEL_HOST = urllib.parse.urlsplit(_MODEL_URL).hostname
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _cached_weight_path():
    default = os.path.join(os.path.expanduser("~"), ".cache")
    root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")
    return os.path.join(root, os.path.basename(_MODEL_URL))


def test_binds_only_loopback():
    # The weight host is not loopback, so any connection to it must be blocked.
    assert _MODEL_HOST not in _LOOPBACK
    # Use a literal non-loopback IP so no DNS is needed: the guard blocks connect.
    with pytest.raises(RuntimeError):
        socket.create_connection(("8.8.8.8", 443), timeout=5)
    # Loopback is permitted (port 9 is closed here, so a normal OSError, not the
    # guard's RuntimeError, is raised).
    s = socket.socket()
    try:
        with pytest.raises(OSError):
            s.connect(("127.0.0.1", 9))
    finally:
        s.close()


def test_fails_closed_without_local_weights(tmp_path):
    # No local weights + loopback-only guard: loading must raise, never silently
    # download, and must not store any checkpoint.
    with pytest.raises(Exception):
        whisper.load_model(_MODEL, device="cpu", download_root=str(tmp_path))
    assert not any(p.suffix == ".pt" for p in tmp_path.iterdir())


def test_fails_closed_with_network_disabled(tmp_path, monkeypatch):
    # Fully disable the network (even loopback): any socket connect fails. With no
    # local weights, loading must raise and store nothing (fail closed).
    def _network_disabled(*args, **kwargs):
        raise OSError("network disabled")

    monkeypatch.setattr(socket.socket, "connect", _network_disabled)
    monkeypatch.setattr(socket.socket, "connect_ex", _network_disabled)
    with pytest.raises(Exception):
        whisper.load_model(_MODEL, device="cpu", download_root=str(tmp_path))
    assert not any(p.suffix == ".pt" for p in tmp_path.iterdir())


def test_runtime_never_downloads_when_cached(monkeypatch):
    if not os.path.isfile(_cached_weight_path()):
        pytest.skip("weights not pre-cached locally")

    def _fail_download(*args, **kwargs):
        raise AssertionError("runtime attempted to download weights")

    # Any network fetch during load would call urlopen; make it fail loudly.
    monkeypatch.setattr(urllib.request, "urlopen", _fail_download)
    model = whisper.load_model(_MODEL, device="cpu")
    assert model is not None
