import json
import threading
import urllib.request
from pathlib import Path

import pytest

from whisper.env_policy import (
    ALLOW_WEIGHT_FETCH_ENV,
    BIND_HOST,
    BindError,
    WeightFetchError,
    refuse_default_weight_fetch,
    require_bind_127_0_0_1,
    url_is_loopback,
    urlopen_for_weights,
    weight_fetch_allowed,
)
from whisper.serve import make_server

REPO_ROOT = Path(__file__).resolve().parents[1]
WILDCARD = "0.0.0.0"
CDN_URL = (
    "https://openaipublic.azureedge.net/main/whisper/models/"
    "aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/"
    "large-v3-turbo.pt"
)


def test_require_bind_accepts_127():
    assert require_bind_127_0_0_1(BIND_HOST) == BIND_HOST
    assert require_bind_127_0_0_1(" 127.0.0.1 ") == BIND_HOST


@pytest.mark.parametrize(
    "host",
    [
        WILDCARD,
        "::",
        "::1",
        "localhost",
        "127.0.0.2",
        "10.0.0.1",
        "192.168.1.1",
        "8.8.8.8",
        "openaipublic.azureedge.net",
        "",
        None,
    ],
)
def test_require_bind_rejects_non_127(host):
    with pytest.raises(BindError, match="127.0.0.1"):
        require_bind_127_0_0_1(host)


def test_weight_fetch_denied_by_default(monkeypatch):
    monkeypatch.delenv(ALLOW_WEIGHT_FETCH_ENV, raising=False)
    assert weight_fetch_allowed() is False
    monkeypatch.setenv(ALLOW_WEIGHT_FETCH_ENV, "0")
    assert weight_fetch_allowed() is False


def test_weight_fetch_allowed_only_when_opted_in(monkeypatch):
    monkeypatch.setenv(ALLOW_WEIGHT_FETCH_ENV, "1")
    assert weight_fetch_allowed() is True


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/tiny.pt",
        "http://localhost:8080/tiny.pt",
        "http://[::1]/tiny.pt",
        "file:///tmp/tiny.pt",
    ],
)
def test_loopback_urls(url):
    assert url_is_loopback(url)


@pytest.mark.parametrize(
    "url",
    [
        CDN_URL,
        "https://example.com/tiny.pt",
        "http://8.8.8.8/tiny.pt",
        "http://10.0.0.1/tiny.pt",
        "https://localhost.evil.com/tiny.pt",
    ],
)
def test_remote_urls_are_not_loopback(url):
    assert not url_is_loopback(url)


def test_refuse_default_fetch_blocks_cdn(monkeypatch):
    monkeypatch.setenv(ALLOW_WEIGHT_FETCH_ENV, "0")
    with pytest.raises(WeightFetchError, match="openaipublic.azureedge.net"):
        refuse_default_weight_fetch(CDN_URL)


def test_urlopen_does_not_touch_network_when_denied(monkeypatch):
    monkeypatch.setenv(ALLOW_WEIGHT_FETCH_ENV, "0")

    def boom(*args, **kwargs):
        raise AssertionError("urlopen must not run for a refused weight fetch")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(urllib.request, "build_opener", boom)
    with pytest.raises(WeightFetchError):
        urlopen_for_weights(CDN_URL)


def test_download_refuses_default_turbo(monkeypatch, tmp_path):
    import whisper

    monkeypatch.setenv(ALLOW_WEIGHT_FETCH_ENV, "0")

    def boom(*args, **kwargs):
        raise AssertionError("cache miss must not open the official CDN")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(WeightFetchError):
        whisper._download(whisper._MODELS["turbo"], str(tmp_path), in_memory=False)
    assert list(tmp_path.iterdir()) == []


def test_download_cache_hit_is_not_a_fetch(monkeypatch, tmp_path):
    import hashlib

    import whisper

    monkeypatch.setenv(ALLOW_WEIGHT_FETCH_ENV, "0")
    payload = b"cached-checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "tiny.pt"
    target.write_bytes(payload)
    url = f"https://openaipublic.azureedge.net/main/whisper/models/{digest}/tiny.pt"

    def boom(*args, **kwargs):
        raise AssertionError("cache hit must not open a remote URL")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert whisper._download(url, str(tmp_path), in_memory=False) == str(target)


def test_make_server_rejects_wildcard():
    with pytest.raises(BindError, match="127.0.0.1"):
        make_server(WILDCARD, 0)


def test_make_server_listens_on_127_only():
    server = make_server(BIND_HOST, 0)
    try:
        host, port = server.server_address
        assert host == BIND_HOST
        assert port > 0
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as response:
            body = json.loads(response.read())
        assert body["ok"] is True
        assert body["bind"] == BIND_HOST
        assert body["weights"] is False
    finally:
        server.shutdown()
        server.server_close()


def test_env_files_force_localhost_and_no_fetch():
    env_sh = (REPO_ROOT / ".cursor/env.sh").read_text()
    start = (REPO_ROOT / ".cursor/start.sh").read_text()
    install = (REPO_ROOT / ".cursor/install.sh").read_text()
    verify = (REPO_ROOT / ".cursor/verify.sh").read_text()
    env = json.loads((REPO_ROOT / ".cursor/environment.json").read_text())
    workflow = (REPO_ROOT / ".github/workflows/test.yml").read_text()

    assert "WHISPER_BIND_HOST=127.0.0.1" in env_sh
    assert "WHISPER_ALLOW_WEIGHT_FETCH=0" in env_sh
    assert "--host 127.0.0.1" in start
    assert WILDCARD not in start
    assert WILDCARD not in install
    assert "load_model(" not in install
    assert "source .cursor/env.sh" in install
    assert "source .cursor/env.sh" in verify
    assert "ports" not in env
    assert "bash .cursor/verify.sh" in workflow
    assert "environment-verify" in workflow


def test_start_and_cursor_scripts_forbid_wildcard():
    hits = []
    skip = {".cursor/verify.sh"}
    for path in sorted((REPO_ROOT / ".cursor").glob("*")):
        rel = str(path.relative_to(REPO_ROOT))
        if not path.is_file() or rel in skip:
            continue
        if WILDCARD in path.read_text():
            hits.append(rel)
    assert hits == [], f"{WILDCARD} must not appear in .cursor/: {hits}"


def test_serve_module_has_no_weight_path():
    source = (REPO_ROOT / "whisper/serve.py").read_text()
    assert "load_model(" not in source
    assert "_download(" not in source
    assert "openaipublic" not in source
    assert WILDCARD not in source
