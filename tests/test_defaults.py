import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

import whisper
from whisper.model import ModelDimensions, Whisper

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_toy_checkpoint(path: Path) -> None:
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=16,
        n_audio_state=16,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=100,
        n_text_ctx=16,
        n_text_state=16,
        n_text_head=2,
        n_text_layer=1,
    )
    model = Whisper(dims)
    torch.save(
        {"dims": asdict(dims), "model_state_dict": model.state_dict()},
        path,
    )


def test_default_device_is_cpu():
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_default_device_ignores_cuda_availability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_load_model_defaults_to_cpu(tmp_path):
    ckpt = tmp_path / "toy.pt"
    _write_toy_checkpoint(ckpt)
    model = whisper.load_model(str(ckpt))
    assert model.device.type == "cpu"


def test_cli_device_default_is_cpu():
    source = (REPO_ROOT / "whisper" / "transcribe.py").read_text(encoding="utf-8")
    assert "default=DEFAULT_DEVICE" in source
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_cli_help_defaults_to_cpu_without_weight_download(tmp_path):
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "whisper", "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--device" in result.stdout
    assert "default: cpu" in result.stdout
    assert list(tmp_path.rglob("*.pt")) == []
    assert list(tmp_path.rglob("*.pth")) == []
