import importlib.util
import socket
import sys
import urllib.request
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = _ROOT / "whisper" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


offline = _load("whisper_offline_guard", "offline.py")
bind = _load("whisper_bind_guard", "bind.py")


def test_urlopen_model_fetch_blocked():
    with offline.offline_guards():
        with pytest.raises(offline.OfflineNetworkError, match="blocked urlopen"):
            urllib.request.urlopen(
                "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
            )


def test_offline_guards_restore_when_not_nested():
    if offline._state["depth"] != 0:
        pytest.skip("session offline guards are active")
    orig = urllib.request.urlopen
    with offline.offline_guards():
        assert urllib.request.urlopen is not orig
    assert urllib.request.urlopen is orig


def test_create_connection_wan_blocked():
    with offline.offline_guards():
        with pytest.raises(offline.OfflineNetworkError, match="blocked connect"):
            socket.create_connection(("1.1.1.1", 443), timeout=0.2)


def test_create_connection_wildcard_blocked():
    with offline.offline_guards():
        with pytest.raises(offline.OfflineNetworkError):
            socket.create_connection(("0.0.0.0", 80), timeout=0.2)


def test_loopback_listen_and_connect_allowed():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        bind.listen_loopback(srv, 0)
        host, port = srv.getsockname()
        assert host == "127.0.0.1"
        with offline.offline_guards():
            client = socket.create_connection((host, port), timeout=1)
            try:
                conn, _addr = srv.accept()
                conn.close()
            finally:
                client.close()
            offline.assert_only_loopback_listeners()
    finally:
        srv.close()


def test_bind_wildcard_still_refused_offline():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with offline.offline_guards():
            with pytest.raises(bind.NonLoopbackBindError):
                bind.bind_loopback(sock, 0, host="0.0.0.0")
    finally:
        sock.close()


def test_assert_no_new_weights_detects_pt(tmp_path):
    before = offline.weight_files(tmp_path)
    (tmp_path / "tiny.pt").write_bytes(b"not-a-real-checkpoint")
    with pytest.raises(offline.OfflineNetworkError, match="tiny.pt"):
        offline.assert_no_new_weights(before, cache_dir=tmp_path)


def test_assert_no_new_weights_ok_on_empty(tmp_path):
    offline.assert_no_new_weights([], cache_dir=tmp_path)


def test_load_model_fetch_blocked(tmp_path):
    torch = pytest.importorskip("torch")
    del torch
    import whisper

    with offline.offline_guards():
        with pytest.raises(offline.OfflineNetworkError, match="blocked"):
            whisper.load_model("tiny", download_root=str(tmp_path))
    assert list(tmp_path.glob("*.pt")) == []
