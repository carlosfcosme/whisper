import hashlib
import urllib.request

import pytest
import torch

import whisper
from whisper.runtime import (
    DEFAULT_DEVICE,
    default_device,
    is_no_store,
    is_offline,
)
from whisper.transcribe import DEFAULT_DEVICE as CLI_DEFAULT_DEVICE


def test_default_device_is_cpu():
    assert DEFAULT_DEVICE == "cpu"
    assert whisper.DEFAULT_DEVICE == "cpu"
    assert default_device() == "cpu"
    assert CLI_DEFAULT_DEVICE == "cpu"


def test_default_device_ignores_cuda_availability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cpu"
    assert whisper.default_device() == "cpu"


def test_whisper_device_env_override(monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    assert default_device() == "cuda"
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    assert default_device() == "cpu"


def test_offline_and_no_store_default_on(monkeypatch):
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    monkeypatch.delenv("WHISPER_NO_STORE", raising=False)
    assert is_offline() is True
    assert is_no_store() is True


def test_env_can_disable_offline_and_no_store(monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "0")
    monkeypatch.setenv("WHISPER_NO_STORE", "false")
    assert is_offline() is False
    assert is_no_store() is False


def test_load_model_defaults_to_cpu_without_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    recorded = {}

    def fake_load(fp, map_location=None):
        recorded["map_location"] = map_location
        return {
            "dims": {
                "n_mels": 80,
                "n_audio_ctx": 1500,
                "n_audio_state": 8,
                "n_audio_head": 2,
                "n_audio_layer": 1,
                "n_vocab": 100,
                "n_text_ctx": 16,
                "n_text_state": 8,
                "n_text_head": 2,
                "n_text_layer": 1,
            },
            "model_state_dict": {},
        }

    class FakeWhisper:
        def __init__(self, dims):
            self.dims = dims

        def load_state_dict(self, state):
            return None

        def set_alignment_heads(self, heads):
            return None

        def to(self, device):
            recorded["to_device"] = str(device)
            return self

    monkeypatch.setattr(whisper, "Whisper", FakeWhisper)
    monkeypatch.setattr(torch, "load", fake_load)

    checkpoint = tmp_path / "local.pt"
    checkpoint.write_bytes(b"stub")
    model = whisper.load_model(str(checkpoint))
    assert recorded["map_location"] == "cpu"
    assert recorded["to_device"] == "cpu"
    assert model is not None


def test_load_named_model_offline_does_not_download(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("urlopen must not run in offline mode")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(RuntimeError, match="offline mode"):
        whisper.load_model("tiny", download_root=str(tmp_path), offline=True)


def test_download_no_store_does_not_write_cache(monkeypatch, tmp_path):
    payload = b"checkpoint-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    url = f"https://example.invalid/{digest}/tiny.pt"

    class FakeResp:
        def info(self):
            return {"Content-Length": str(len(payload))}

        def read(self, n=-1):
            if n is None or n < 0:
                data, self._data = self._data, b""
                return data
            chunk, self._data = self._data[:n], self._data[n:]
            return chunk

        def __enter__(self):
            self._data = payload
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: FakeResp())
    result = whisper._download(
        url, str(tmp_path), in_memory=False, offline=False, no_store=True
    )
    assert result == payload
    assert list(tmp_path.iterdir()) == []


def test_download_offline_missing_file_does_not_create_cache(tmp_path):
    url = "https://example.invalid/deadbeef/tiny.pt"
    with pytest.raises(RuntimeError, match="offline mode"):
        whisper._download(url, str(tmp_path), in_memory=False, offline=True)
    assert list(tmp_path.iterdir()) == []


def test_gpu_only_timing_helpers_are_optional():
    from whisper.timing import dtw_cpu

    x = __import__("numpy").random.random((4, 5)).astype("float32")
    trace = dtw_cpu(x)
    assert trace.shape[0] == 2
    assert not hasattr(test_gpu_only_timing_helpers_are_optional, "requires_cuda")


def test_no_live_or_field_brain_flags():
    from whisper.serve import build_parser
    from whisper.transcribe import cli

    serve = build_parser()
    option_strings = []
    for action in serve._actions:
        option_strings.extend(action.option_strings)
    assert "--live" not in option_strings
    assert "--api-key" not in option_strings
    assert "field-brain" not in " ".join(option_strings).lower()
    assert cli.__name__ == "cli"
