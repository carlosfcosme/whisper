import subprocess
from pathlib import Path

import pytest
import torch

import whisper
from whisper.device import DEFAULT_DEVICE, default_device


def test_sample_audio_is_local_file(sample_audio_path):
    path = Path(sample_audio_path)
    assert path.is_file()
    assert path.name == "jfk.flac"
    assert path.suffix == ".flac"
    assert not sample_audio_path.startswith(("http://", "https://"))
    assert "huggingface" not in sample_audio_path.lower()


def test_load_model_from_offline_checkpoint(offline_checkpoint_path):
    model = whisper.load_model(offline_checkpoint_path)
    assert model.device.type == "cpu"
    assert model.dims.n_audio_layer == 1
    assert model.dims.n_vocab == 32


def test_load_model_local_path_does_not_download(offline_checkpoint_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("_download should not run for a local checkpoint")

    monkeypatch.setattr(whisper, "_download", boom)
    model = whisper.load_model(offline_checkpoint_path)
    assert model.dims.n_text_layer == 1


def test_cpu_is_default_device(offline_checkpoint_path):
    assert DEFAULT_DEVICE == "cpu"
    assert default_device() == "cpu"
    model = whisper.load_model(offline_checkpoint_path)
    assert model.device.type == DEFAULT_DEVICE
    if torch.cuda.is_available():
        gpu = whisper.load_model(offline_checkpoint_path, device="cuda")
        assert gpu.device.type == "cuda"


def test_cli_device_default_is_cpu():
    from whisper.transcribe import DEFAULT_DEVICE as CLI_DEVICE

    assert CLI_DEVICE == "cpu"


def test_no_huggingface_hub_dependency():
    root = Path(__file__).resolve().parents[1]
    packaging = (root / "pyproject.toml").read_text(encoding="utf-8").lower()
    requirements = (root / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "huggingface" not in packaging
    assert "huggingface" not in requirements
    assert "hf_hub" not in packaging


def test_no_committed_weight_files():
    root = Path(__file__).resolve().parents[1]
    listed = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--",
            "*.pt",
            "*.pth",
            "*.ckpt",
            "*.safetensors",
        ],
        cwd=str(root),
        text=True,
    ).strip()
    assert listed == ""
    script = root / ".github" / "scripts" / "fail-committed-weights.sh"
    subprocess.check_call(["bash", str(script)], cwd=str(root))


@pytest.mark.parametrize(
    "name",
    ["tiny", "tiny.en"],
)
def test_named_model_stays_offline_without_cache(name, tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    def boom(*args, **kwargs):
        raise AssertionError("urlopen must not run in offline tests")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(RuntimeError, match="WHISPER_OFFLINE"):
        whisper.load_model(name)
