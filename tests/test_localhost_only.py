import hashlib
import os
import urllib.request

import pytest

from whisper.localhost import (
    LOCALHOST_ONLY_ENV,
    RemotePullError,
    hostname_is_localhost,
    localhost_only_enabled,
    refuse_remote_pull,
    url_is_localhost,
    urlopen_maybe_localhost_only,
)

pytestmark = pytest.mark.localhost_only

WAN_URLS = (
    "https://openaipublic.azureedge.net/main/whisper/models/abc/tiny.pt",
    "https://example.com/tiny.pt",
    "http://8.8.8.8/tiny.pt",
    "http://10.0.0.1/tiny.pt",
    "http://192.168.1.10/tiny.pt",
    "http://169.254.1.1/tiny.pt",
    "https://localhost.evil.com/tiny.pt",
    "https://127.0.0.1.nip.io/tiny.pt",
)

LOCALHOST_URLS = (
    "http://127.0.0.1/tiny.pt",
    "http://127.0.0.1:8080/tiny.pt",
    "http://localhost/tiny.pt",
    "http://LOCALHOST:9000/tiny.pt",
    "http://localhost./tiny.pt",
    "http://[::1]/tiny.pt",
    "http://[::1]:8080/tiny.pt",
    "https://127.0.0.2/tiny.pt",
    "file:///tmp/tiny.pt",
)


@pytest.mark.parametrize(
    "host", ["localhost", "LOCALHOST", "localhost.", "127.0.0.1", "::1"]
)
def test_hostname_is_localhost(host):
    assert hostname_is_localhost(host)


@pytest.mark.parametrize(
    "host",
    [
        None,
        "",
        "openaipublic.azureedge.net",
        "example.com",
        "8.8.8.8",
        "10.0.0.1",
        "192.168.0.5",
        "localhost.evil.com",
    ],
)
def test_hostname_is_not_localhost(host):
    assert not hostname_is_localhost(host)


@pytest.mark.parametrize("url", LOCALHOST_URLS)
def test_url_is_localhost(url):
    assert url_is_localhost(url)


@pytest.mark.parametrize("url", WAN_URLS)
def test_url_is_remote(url):
    assert not url_is_localhost(url)


def test_localhost_only_enabled_reads_env(monkeypatch):
    monkeypatch.delenv(LOCALHOST_ONLY_ENV, raising=False)
    assert not localhost_only_enabled()
    monkeypatch.setenv(LOCALHOST_ONLY_ENV, "0")
    assert not localhost_only_enabled()
    monkeypatch.setenv(LOCALHOST_ONLY_ENV, "1")
    assert localhost_only_enabled()
    monkeypatch.setenv(LOCALHOST_ONLY_ENV, "true")
    assert localhost_only_enabled()


def test_refuse_remote_pull_is_noop_when_disabled(monkeypatch):
    monkeypatch.delenv(LOCALHOST_ONLY_ENV, raising=False)
    refuse_remote_pull(WAN_URLS[0])


@pytest.mark.parametrize("url", WAN_URLS)
def test_refuse_remote_pull_rejects_wan(monkeypatch, url):
    monkeypatch.setenv(LOCALHOST_ONLY_ENV, "1")
    with pytest.raises(RemotePullError, match="remote/WAN"):
        refuse_remote_pull(url)


@pytest.mark.parametrize("url", LOCALHOST_URLS)
def test_refuse_remote_pull_allows_localhost(monkeypatch, url):
    monkeypatch.setenv(LOCALHOST_ONLY_ENV, "1")
    refuse_remote_pull(url)


def _forbid_network(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("urlopen should not be called for a remote/WAN pull")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(urllib.request, "build_opener", boom)


def test_urlopen_refuses_wan_without_touching_network(monkeypatch):
    monkeypatch.setenv(LOCALHOST_ONLY_ENV, "1")
    _forbid_network(monkeypatch)
    with pytest.raises(RemotePullError, match="openaipublic.azureedge.net"):
        urlopen_maybe_localhost_only(WAN_URLS[0])


def test_urlopen_passthrough_when_disabled(monkeypatch):
    monkeypatch.delenv(LOCALHOST_ONLY_ENV, raising=False)
    called = {}

    class Dummy:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(url):
        called["url"] = url
        return Dummy()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    urlopen_maybe_localhost_only(WAN_URLS[0])
    assert called["url"] == WAN_URLS[0]


def test_download_refuses_wan_on_cache_miss(monkeypatch, tmp_path):
    import whisper

    monkeypatch.setenv(LOCALHOST_ONLY_ENV, "1")
    _forbid_network(monkeypatch)
    url = (
        "https://openaipublic.azureedge.net/main/whisper/models/"
        "d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/"
        "tiny.en.pt"
    )
    with pytest.raises(RemotePullError, match="remote/WAN"):
        whisper._download(url, str(tmp_path), in_memory=False)
    assert os.listdir(tmp_path) == []


def test_download_cache_hit_is_not_a_pull(monkeypatch, tmp_path):
    import whisper

    monkeypatch.setenv(LOCALHOST_ONLY_ENV, "1")
    payload = b"cached-checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "tiny.pt"
    target.write_bytes(payload)
    url = f"https://openaipublic.azureedge.net/main/whisper/models/{digest}/tiny.pt"

    def boom(*args, **kwargs):
        raise AssertionError("cache hit must not open a remote/WAN URL")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert whisper._download(url, str(tmp_path), in_memory=False) == str(target)


def test_redirect_handler_refuses_wan_target(monkeypatch):
    from whisper.localhost import LocalhostOnlyRedirectHandler

    monkeypatch.setenv(LOCALHOST_ONLY_ENV, "1")
    handler = LocalhostOnlyRedirectHandler()
    req = urllib.request.Request("http://127.0.0.1/from")
    with pytest.raises(RemotePullError, match="example.com"):
        handler.redirect_request(req, None, 302, "Found", {}, "https://example.com/to")


def test_localhost_only_blocks_wan_urlopen():
    from conftest import NetworkDownloadBlocked

    with pytest.raises(NetworkDownloadBlocked, match="example.com"):
        urllib.request.urlopen("https://example.com/tiny.pt", timeout=1)


def test_localhost_only_blocks_wan_urlretrieve(tmp_path):
    from conftest import NetworkDownloadBlocked

    dest = tmp_path / "tiny.pt"
    with pytest.raises(NetworkDownloadBlocked, match="example.com"):
        urllib.request.urlretrieve("https://example.com/tiny.pt", filename=str(dest))
    assert not dest.exists()


def test_localhost_only_allows_loopback_urlopen():
    from urllib.error import URLError

    from conftest import NetworkDownloadBlocked

    try:
        urllib.request.urlopen("http://127.0.0.1:1/", timeout=0.3)
    except NetworkDownloadBlocked:
        raise AssertionError("loopback urlopen must not be blocked")
    except (URLError, OSError, TimeoutError):
        pass
