import urllib.request

import pytest

import whisper
from whisper.runtime import WeightDownloadError, weight_auto_download_allowed

HF_HUB_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"


def test_weight_auto_download_disabled_in_tests():
    assert weight_auto_download_allowed() is False


def test_load_model_does_not_download(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise AssertionError("tests must not pull weights")

    monkeypatch.setattr(urllib.request, "urlopen", boom)

    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))
    with pytest.raises(WeightDownloadError, match="Hugging Face Hub"):
        whisper._download(HF_HUB_URL, str(tmp_path), in_memory=False)

    assert list(tmp_path.iterdir()) == []


def test_check_downloads_blocked_script():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "check_downloads_blocked.py"
    )
    spec = importlib.util.spec_from_file_location("check_downloads_blocked", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 0
