import hashlib
import sys

import pytest

import whisper
from whisper.loopback import LOOPBACK_HOST, start_loopback_server


def test_named_model_does_not_hit_cdn(monkeypatch, tmp_path):
    calls = []

    def boom(url, *args, **kwargs):
        calls.append(url)
        raise AssertionError("urlopen must not reach the CDN or Hub")

    monkeypatch.setattr(whisper.urllib.request, "urlopen", boom)
    with pytest.raises(whisper.RemoteDownloadError, match="Refusing remote"):
        whisper._download(whisper._MODELS["tiny"], str(tmp_path), False)
    assert calls == []


def test_huggingface_hub_url_is_rejected(monkeypatch, tmp_path):
    calls = []

    def boom(url, *args, **kwargs):
        calls.append(url)
        raise AssertionError("urlopen must not reach Hugging Face Hub")

    monkeypatch.setattr(whisper.urllib.request, "urlopen", boom)
    hub_url = (
        "https://huggingface.co/openai/whisper-tiny/resolve/"
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/model.safetensors"
    )
    with pytest.raises(whisper.RemoteDownloadError, match="Hugging Face Hub"):
        whisper._download(hub_url, str(tmp_path), False)
    assert calls == []
    assert "huggingface_hub" not in sys.modules


def test_load_model_named_without_cache_skips_network(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []

    def boom(url, *args, **kwargs):
        calls.append(url)
        raise AssertionError("load_model must not download")

    monkeypatch.setattr(whisper.urllib.request, "urlopen", boom)
    with pytest.raises(whisper.RemoteDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path / "whisper"))
    assert calls == []


def test_download_allows_loopback_http(tmp_path):
    payload = b"local-weight"
    digest = hashlib.sha256(payload).hexdigest()
    served = tmp_path / digest
    served.mkdir()
    (served / "toy.pt").write_bytes(payload)

    httpd = start_loopback_server(str(tmp_path), host=LOOPBACK_HOST, port=0)
    try:
        port = httpd.server_address[1]
        url = "http://{0}:{1}/{2}/toy.pt".format(LOOPBACK_HOST, port, digest)
        dest = whisper._download(url, str(tmp_path / "cache"), False)
        assert open(dest, "rb").read() == payload
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_checksum_mismatch_does_not_redownload(monkeypatch, tmp_path):
    target = tmp_path / "tiny.pt"
    target.write_bytes(b"stale")
    url = whisper._MODELS["tiny"]
    calls = []

    def boom(url, *args, **kwargs):
        calls.append(url)
        raise AssertionError("checksum mismatch must not re-download")

    monkeypatch.setattr(whisper.urllib.request, "urlopen", boom)
    with pytest.raises(whisper.RemoteDownloadError, match="checksum mismatch"):
        whisper._download(url, str(tmp_path), False)
    assert calls == []
