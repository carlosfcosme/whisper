from pathlib import Path

import pytest

import whisper
from whisper.runtime import is_hub_url, refuse_remote_download


def test_hub_urls_are_detected():
    assert is_hub_url("https://huggingface.co/openai/whisper-tiny")
    assert is_hub_url("https://hf.co/openai/whisper-tiny")
    assert not is_hub_url(
        "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
    )


def test_refuse_hub_download_even_when_online(monkeypatch):
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with pytest.raises(RuntimeError, match="no Hub"):
        refuse_remote_download(
            "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors",
            "/tmp/model.safetensors",
        )


def test_refuse_remote_when_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    dest = str(tmp_path / "tiny.pt")
    with pytest.raises(RuntimeError, match="offline"):
        refuse_remote_download(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt",
            dest,
        )


def test_download_refuses_hub(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with pytest.raises(RuntimeError, match="no Hub"):
        whisper._download(
            "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt",
            str(tmp_path),
            in_memory=False,
        )


def test_huggingface_hub_is_not_a_dependency():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text()
    requirements = (root / "requirements.txt").read_text()
    assert "huggingface" not in pyproject.lower()
    assert "huggingface" not in requirements.lower()
    assert "hf_hub" not in pyproject.lower()
