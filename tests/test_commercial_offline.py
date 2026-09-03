"""Offline commercial path: local weights, blocked downloads, loopback bind."""

import os
import threading
import urllib.request
from pathlib import Path

import pytest
import torch

import whisper
from whisper.localhost import serve_bind_host
from whisper.model import ModelDimensions, Whisper
from whisper.serve import serve

pytestmark = pytest.mark.commercial

_OFFICIAL_LOCAL_NAMES = ("tiny.en.pt", "tiny.pt")


def _official_cache_roots():
    roots = []
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        roots.append(Path(xdg) / "whisper")
    roots.append(Path.home() / ".cache" / "whisper")
    return roots


def find_local_official_checkpoint():
    """Return a cached official .pt path, or None. Does not download."""
    for root in _official_cache_roots():
        for name in _OFFICIAL_LOCAL_NAMES:
            path = root / name
            if path.is_file() and path.stat().st_size > 1_000_000:
                return path
    return None


def write_local_checkpoint(path: Path) -> Path:
    """Write a tiny random Whisper checkpoint (local weights, never from Hub)."""
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=32,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=256,
        n_text_ctx=32,
        n_text_state=32,
        n_text_head=2,
        n_text_layer=1,
    )
    model = Whisper(dims)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"dims": dims.__dict__, "model_state_dict": model.state_dict()},
        path,
    )
    return path


@pytest.fixture
def local_weight_path(tmp_path):
    """Commercial path requires weights on disk. Prefer official cache; else write local."""
    official = find_local_official_checkpoint()
    if official is not None:
        return official
    return write_local_checkpoint(tmp_path / "commercial_local.pt")


@pytest.fixture
def download_calls(monkeypatch):
    calls = []

    def guarded(url, *args, **kwargs):
        calls.append(
            url if isinstance(url, str) else getattr(url, "full_url", str(url))
        )
        raise RuntimeError(f"download blocked on commercial path: {url}")

    monkeypatch.setattr(urllib.request, "urlopen", guarded)
    return calls


def test_commercial_load_requires_local_weights(local_weight_path, download_calls):
    assert Path(local_weight_path).is_file()
    model = whisper.load_model(str(local_weight_path), device=whisper.DEFAULT_DEVICE)
    assert next(model.parameters()).device.type == "cpu"
    assert download_calls == []


def test_commercial_named_checkpoint_does_not_download(
    tmp_path, monkeypatch, download_calls
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="Offline/no-Hub|download blocked"):
        whisper.load_model("tiny")


def test_commercial_serve_is_loopback_only():
    with pytest.raises(ValueError, match="127.0.0.1"):
        serve_bind_host("0.0.0.0")
    with pytest.raises(ValueError, match="127.0.0.1"):
        serve(host="0.0.0.0", port=0)

    httpd = serve(port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        assert port > 0

        def _run():
            httpd.handle_request()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as resp:
            assert resp.read() == b"whisper localhost-only\n"
        thread.join(timeout=2)
    finally:
        httpd.server_close()


@pytest.mark.requires_local_weights
def test_commercial_transcribe_from_local_official_weights(download_calls):
    path = find_local_official_checkpoint()
    if path is None:
        pytest.skip("requires local official weights (tiny.pt / tiny.en.pt)")
    model = whisper.load_model(str(path), device=whisper.DEFAULT_DEVICE)
    audio = Path(__file__).resolve().parent / "jfk.flac"
    result = model.transcribe(str(audio), language="en", temperature=0.0)
    assert "my fellow americans" in result["text"].lower()
    assert download_calls == []
