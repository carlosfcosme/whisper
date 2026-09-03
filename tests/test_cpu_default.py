"""CPU is the default device. Loads policy without requiring CUDA or Hub."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_policy():
    if "_offline_policy" in sys.modules:
        return sys.modules["_offline_policy"]
    spec = importlib.util.spec_from_file_location(
        "_offline_policy", ROOT / "whisper" / "policy.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_offline_policy"] = mod
    spec.loader.exec_module(mod)
    return mod


policy = _load_policy()


def test_default_device_is_cpu():
    assert policy.DEFAULT_DEVICE == "cpu"


def test_bind_host_is_loopback():
    assert policy.BIND_HOST == "127.0.0.1"


def test_hub_urls_are_detected():
    assert policy.is_hub_url("https://huggingface.co/openai/whisper")
    assert policy.is_hub_url("https://hf.co/models/openai/whisper")
    assert not policy.is_hub_url(
        "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
    )


def test_refuse_hub_download(tmp_path):
    dest = str(tmp_path / "tiny.pt")
    try:
        policy.refuse_remote_download("https://huggingface.co/openai/whisper", dest)
    except RuntimeError as exc:
        assert "Hub" in str(exc)
    else:
        raise AssertionError("expected Hub refusal")


def test_refuse_offline_download(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    dest = str(tmp_path / "tiny.pt")
    try:
        policy.refuse_remote_download(
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt", dest
        )
    except RuntimeError as exc:
        assert "offline" in str(exc)
    else:
        raise AssertionError("expected offline refusal")
