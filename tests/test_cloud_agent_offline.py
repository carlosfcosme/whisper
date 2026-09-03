"""Cloud Agent path: block network/model downloads; startup is loopback-only."""

import http.client
import json
import sys
import types
import urllib.request
from ipaddress import ip_address

import pytest

import whisper
from whisper.runtime import (
    ALLOW_WEIGHT_DOWNLOAD_ENV,
    BindError,
    WeightDownloadError,
    weight_auto_download_allowed,
)
from whisper.serve import cloud_agent_startup, create_server
from whisper.serve import main as serve_main

HF_HUB_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"
WAN_HOSTS = ("0.0.0.0", "::", "", "8.8.8.8", "10.0.0.1", "example.com")


def _cloud_agent_env(monkeypatch):
    """Sovereign Cloud Agent flags: offline + loopback + CPU."""
    monkeypatch.setenv("WHISPER_CPU_ONLY", "1")
    monkeypatch.setenv("WHISPER_NO_WEIGHT_DOWNLOAD", "1")
    monkeypatch.setenv("WHISPER_LOCALHOST_ONLY", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.delenv(ALLOW_WEIGHT_DOWNLOAD_ENV, raising=False)


def _patch_download_helpers(monkeypatch):
    """Patch network/model download helpers. Any call fails the test."""
    invoked = []

    def _mark(name):
        def boom(*args, **kwargs):
            invoked.append(name)
            raise AssertionError(
                "%s must not be invoked on the Cloud Agent offline path" % name
            )

        return boom

    monkeypatch.setattr(urllib.request, "urlopen", _mark("urlopen"))
    monkeypatch.setattr(urllib.request, "urlretrieve", _mark("urlretrieve"))

    requests_mod = sys.modules.get("requests")
    if requests_mod is None:
        requests_mod = types.ModuleType("requests")
        sys.modules["requests"] = requests_mod
    monkeypatch.setattr(requests_mod, "get", _mark("requests.get"), raising=False)
    monkeypatch.setattr(requests_mod, "post", _mark("requests.post"), raising=False)

    hub = sys.modules.get("huggingface_hub")
    if hub is None:
        hub = types.ModuleType("huggingface_hub")
        sys.modules["huggingface_hub"] = hub
    monkeypatch.setattr(hub, "hf_hub_download", _mark("hf_hub_download"), raising=False)
    monkeypatch.setattr(
        hub, "snapshot_download", _mark("snapshot_download"), raising=False
    )
    return invoked


def _health(host, port):
    """Loopback GET via http.client (not urlopen — that helper stays patched)."""
    conn = http.client.HTTPConnection(host, port, timeout=2)
    try:
        conn.request("GET", "/health")
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read().decode("utf-8"))
    finally:
        conn.close()


def test_cloud_agent_env_is_offline(monkeypatch):
    _cloud_agent_env(monkeypatch)
    assert weight_auto_download_allowed() is False
    assert whisper.default_device() == "cpu"
    assert whisper.default_bind_host() == "127.0.0.1"


def test_cloud_agent_startup_binds_loopback_only(monkeypatch):
    _cloud_agent_env(monkeypatch)
    invoked = _patch_download_helpers(monkeypatch)

    with cloud_agent_startup() as server:
        host, port = server.server_address[:2]
        assert host == "127.0.0.1"
        assert ip_address(host).is_loopback
        assert port > 0
        status, payload = _health(host, port)
        assert status == 200
        assert payload["status"] == "ok"
        assert payload["bind"] == "127.0.0.1"
        assert payload["device"] == "cpu"
        assert payload["weights"] is False
        assert payload["offline"] is True

    assert invoked == []


def test_cloud_agent_startup_refuses_non_loopback(monkeypatch):
    _cloud_agent_env(monkeypatch)
    invoked = _patch_download_helpers(monkeypatch)

    for host in WAN_HOSTS:
        with pytest.raises(BindError):
            create_server(host=host, port=0)
        with pytest.raises(BindError):
            with cloud_agent_startup(host=host, port=0):
                pass

    assert invoked == []


def test_cloud_agent_startup_writes_no_weight_files(monkeypatch, tmp_path):
    _cloud_agent_env(monkeypatch)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    invoked = _patch_download_helpers(monkeypatch)

    with cloud_agent_startup() as server:
        host, port = server.server_address[:2]
        status, payload = _health(host, port)
        assert status == 200
        assert payload["weights"] is False

    cache = tmp_path / "cache"
    assert not cache.exists() or not any(cache.rglob("*"))
    assert invoked == []


def test_cloud_agent_named_load_does_not_download(monkeypatch, tmp_path):
    _cloud_agent_env(monkeypatch)
    invoked = _patch_download_helpers(monkeypatch)

    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))

    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        whisper._download(HF_HUB_URL, str(tmp_path), in_memory=False)

    assert invoked == []
    assert list(tmp_path.iterdir()) == []


def test_serve_cli_refuses_wildcard(monkeypatch):
    _cloud_agent_env(monkeypatch)
    invoked = _patch_download_helpers(monkeypatch)
    assert serve_main(["--host", "0.0.0.0", "--port", "0"]) == 2
    assert invoked == []
