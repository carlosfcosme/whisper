import importlib.util
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest
import torch

import whisper
from whisper.model import ModelDimensions, Whisper
from whisper.runtime import (
    BIND_HOST,
    DEFAULT_DEVICE,
    guard_weight_download,
    is_hub_url,
    offline_enabled,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _toy_checkpoint(path: Path) -> Path:
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=16,
        n_audio_state=32,
        n_audio_head=4,
        n_audio_layer=1,
        n_vocab=50,
        n_text_ctx=16,
        n_text_state=32,
        n_text_head=4,
        n_text_layer=1,
    )
    model = Whisper(dims)
    torch.save({"dims": dims.__dict__, "model_state_dict": model.state_dict()}, path)
    return path


def test_default_device_is_cpu():
    assert DEFAULT_DEVICE == "cpu"
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_load_model_defaults_to_cpu(tmp_path):
    ckpt = _toy_checkpoint(tmp_path / "toy.pt")
    loaded = whisper.load_model(str(ckpt))
    assert loaded.device.type == "cpu"


def test_offline_and_hub_env_is_set():
    assert os.environ.get("WHISPER_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert offline_enabled()


def test_is_hub_url_detects_huggingface():
    assert is_hub_url("https://huggingface.co/openai/whisper-tiny")
    assert is_hub_url("https://hf.co/openai/whisper-tiny")
    assert not is_hub_url(
        "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
    )


def test_refuse_hub_download():
    with pytest.raises(RuntimeError, match="no Hub"):
        guard_weight_download(
            "https://huggingface.co/foo/bar/resolve/model.pt", "/tmp/model.pt"
        )


def test_refuse_offline_weight_pull(tmp_path):
    dest = str(tmp_path / "missing.pt")
    with pytest.raises(RuntimeError, match="no weight pulls"):
        guard_weight_download(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt", dest
        )


def test_download_named_model_stays_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="offline|no weight pulls|forbidden"):
        whisper.load_model("tiny")


def test_remote_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen("https://huggingface.co")


def test_azure_weight_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="forbidden"):
        urllib.request.urlopen(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
        )


def test_whisper_package_does_not_import_huggingface_hub():
    for path in (REPO_ROOT / "whisper").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "huggingface_hub" not in text, path
        assert "from_pretrained" not in text, path


def test_bind_host_is_loopback():
    assert BIND_HOST == "127.0.0.1"


def _load_check_no_weights():
    path = REPO_ROOT / "scripts" / "check_no_weights.py"
    spec = importlib.util.spec_from_file_location("check_no_weights", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_check_no_weights_classifies_extensions():
    check = _load_check_no_weights()
    assert check.classify("models/tiny.pt", 100) is not None
    assert check.classify("weights/model.safetensors", 10) is not None
    assert check.classify("export/model.onnx", 10) is not None
    assert check.classify("libfoo.so", 100) is not None
    assert check.classify("whisper/assets/mel_filters.npz", 4271) is None
    assert check.classify("tests/jfk.flac", 1_152_693) is None
    assert check.classify("README.md", 800) is None
    assert check.classify("README.md", check.MAX_FILE_BYTES + 1) is not None


def test_check_no_weights_passes_on_this_repo():
    script = REPO_ROOT / "scripts" / "check_no_weights.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout


def test_check_no_weights_fails_on_planted_checkpoint(tmp_path):
    check = _load_check_no_weights()
    planted = tmp_path / "leaked.pt"
    planted.write_bytes(b"not-a-real-checkpoint")
    violations = check.find_violations(tmp_path, ["leaked.pt"])
    assert violations
    assert violations[0][0] == "leaked.pt"
