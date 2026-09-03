import hashlib
import os
import subprocess
import urllib.request

import pytest

import whisper
from whisper.offline import (
    LOCALHOST,
    OFFLINE_FIXTURES,
    DownloadHelperCalled,
    assert_no_committed_weights,
    bind_host,
    bind_localhost,
    committed_weight_files,
    default_device,
    downloads_allowed,
    guard_download_url,
    is_hub_url,
)
from whisper.offline import main as offline_main
from whisper.offline import (
    offline_enabled,
    offline_fixture_paths,
    require_local_path,
    resolve_offline_fixture,
)


def test_offline_fixtures_are_local_files():
    paths = offline_fixture_paths()
    assert len(paths) == len(OFFLINE_FIXTURES)
    for path in paths:
        assert os.path.isfile(path)
        assert os.path.isabs(path)
        assert not path.startswith(("http://", "https://", "hf://"))
        require_local_path(path)


def test_sample_audio_fixture_rejects_hub_and_http():
    with pytest.raises(ValueError, match="no Hub"):
        require_local_path("https://huggingface.co/datasets/foo/resolve/main/jfk.flac")
    with pytest.raises(ValueError, match="no Hub"):
        require_local_path("hf://datasets/foo/jfk.flac")
    with pytest.raises(ValueError, match="no network pull"):
        require_local_path("https://example.com/jfk.flac")
    with pytest.raises(ValueError, match="no network pull"):
        require_local_path("http://127.0.0.1/jfk.flac")


def test_resolve_offline_fixture_jfk():
    path = resolve_offline_fixture("tests/jfk.flac")
    assert os.path.basename(path) == "jfk.flac"
    assert os.path.isfile(path)


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt",
        "https://cdn-lfs.huggingface.co/openai/whisper/tiny.pt",
        "https://hf.co/openai/whisper-tiny/resolve/main/tiny.pt",
        "https://hf-mirror.com/openai/whisper-tiny/resolve/main/tiny.pt",
        "hf://models/openai/whisper-tiny/tiny.pt",
    ],
)
def test_is_hub_url(url):
    assert is_hub_url(url)


def test_no_official_model_is_on_hub():
    for name, url in whisper._MODELS.items():
        assert "huggingface" not in url, name
        assert not is_hub_url(url), name


def test_guard_always_refuses_hub():
    url = "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt"
    with pytest.raises(RuntimeError, match="Hub"):
        guard_download_url(url)


def test_download_refuses_hub_without_network(monkeypatch, tmp_path):
    def boom(*_args, **_kwargs):
        raise AssertionError("urlopen must not be called for Hub URLs")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(whisper.urllib.request, "urlopen", boom)
    url = "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt"
    with pytest.raises(RuntimeError, match="Hub"):
        whisper._download(url, str(tmp_path), False)


def test_offline_is_default(monkeypatch):
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    monkeypatch.delenv("WHISPER_ALLOW_DOWNLOAD", raising=False)
    assert offline_enabled() is True
    assert downloads_allowed() is False


def test_offline_mode_refuses_cdn_cache_miss(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    monkeypatch.delenv("WHISPER_ALLOW_DOWNLOAD", raising=False)

    def boom(*_args, **_kwargs):
        raise AssertionError("urlopen must not be called when offline")

    monkeypatch.setattr(whisper.urllib.request, "urlopen", boom)
    url = whisper._MODELS["tiny"]
    with pytest.raises(RuntimeError, match="offline mode"):
        whisper._download(url, str(tmp_path), False)


def test_load_model_does_not_download_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    monkeypatch.delenv("WHISPER_ALLOW_DOWNLOAD", raising=False)
    with pytest.raises(RuntimeError, match="offline mode"):
        whisper.load_model("tiny", download_root=str(tmp_path))


def test_allow_download_reaches_urlopen_for_cdn_only(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_OFFLINE", "0")
    called = []

    def fake_urlopen(url, *_args, **_kwargs):
        called.append(url)
        raise AssertionError("network")

    monkeypatch.setattr(whisper.urllib.request, "urlopen", fake_urlopen)
    url = whisper._MODELS["tiny"]
    with pytest.raises(AssertionError, match="network"):
        whisper._download(url, str(tmp_path), False)
    assert called


def test_hub_refused_even_when_downloads_allowed(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_OFFLINE", "0")
    monkeypatch.setenv("WHISPER_ALLOW_DOWNLOAD", "1")
    url = "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt"
    with pytest.raises(RuntimeError, match="Hub"):
        whisper._download(url, str(tmp_path), False)


def test_download_helpers_fail_in_tests():
    with pytest.raises(DownloadHelperCalled, match="download helper"):
        urllib.request.urlopen("https://example.com/")
    with pytest.raises(DownloadHelperCalled, match="download helper"):
        whisper.urllib.request.urlopen("https://huggingface.co/foo")


def test_offline_cache_hit_is_not_a_pull(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    monkeypatch.delenv("WHISPER_ALLOW_DOWNLOAD", raising=False)

    def boom(*_args, **_kwargs):
        raise AssertionError("cache hit must not pull")

    monkeypatch.setattr(whisper.urllib.request, "urlopen", boom)
    data = b"offline-cache-hit"
    digest = hashlib.sha256(data).hexdigest()
    target = tmp_path / "x.pt"
    target.write_bytes(data)
    url = f"https://openaipublic.azureedge.net/{digest}/x.pt"
    result = whisper._download(url, str(tmp_path), False)
    assert result == str(target)


def test_cpu_is_default_device(monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    assert default_device() == "cpu"
    assert whisper.default_device() == "cpu"


def test_device_override_env(monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    assert default_device() == "cuda"


def test_bind_localhost_only():
    assert bind_host() == LOCALHOST
    sock = bind_localhost()
    try:
        host, port = sock.getsockname()[:2]
        assert host == LOCALHOST
        assert port > 0
        assert host != "0.0.0.0"
    finally:
        sock.close()


def test_bind_rejects_public(monkeypatch):
    monkeypatch.setenv("WHISPER_BIND", "0.0.0.0")
    with pytest.raises(ValueError, match="loopback"):
        bind_host()
    monkeypatch.setenv("WHISPER_BIND", "1.2.3.4")
    with pytest.raises(ValueError, match="loopback"):
        bind_localhost()


def test_repo_has_no_committed_weights():
    assert committed_weight_files() == []
    assert_no_committed_weights()
    assert offline_main(["--check"]) == 0


def test_committed_weights_check_detects_pt(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "ci@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "ci"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    weight = tmp_path / "tiny.pt"
    weight.write_bytes(b"not-a-real-checkpoint")
    subprocess.run(
        ["git", "add", "tiny.pt"], cwd=tmp_path, check=True, capture_output=True
    )
    hits = committed_weight_files(str(tmp_path))
    assert "tiny.pt" in hits
    with pytest.raises(AssertionError, match="tiny.pt"):
        assert_no_committed_weights(str(tmp_path))
