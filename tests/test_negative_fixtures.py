"""Offline tests for negative wildcard/network fixtures (no torch)."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = REPO_ROOT / "whisper" / ("%s.py" % name)
    spec = importlib.util.spec_from_file_location("_probe_%s" % name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wildcard_bind_host_fixture_is_refused(wildcard_bind_host):
    bind = _load("bind")
    with pytest.raises(bind.BindError, match="127.0.0.1"):
        bind.require_loopback_host(wildcard_bind_host)
    with pytest.raises(bind.BindError):
        bind.bind_tcp(host=wildcard_bind_host, port=0)
    assert bind.non_loopback_listens() == []


def test_non_loopback_host_fixture_is_refused(non_loopback_host):
    bind = _load("bind")
    with pytest.raises(bind.BindError, match="127.0.0.1"):
        bind.require_loopback_host(non_loopback_host)
    with pytest.raises(bind.BindError):
        bind.bind_tcp(host=non_loopback_host, port=0)


def test_forbidden_network_url_fixture_is_refused(forbidden_network_url):
    offline = _load("offline")
    with pytest.raises(offline.WeightDownloadError, match="weight pull is disabled"):
        offline.refuse_weight_fetch(forbidden_network_url)
    assert offline.is_remote_fetch_url(forbidden_network_url)


def test_negative_fixture_inventory_is_wired():
    text = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "WILDCARD_BIND_HOSTS" in text
    assert "NON_LOOPBACK_HOSTS" in text
    assert "FORBIDDEN_NETWORK_URLS" in text
    assert "wildcard_bind_host" in text
    assert "forbidden_network_url" in text
