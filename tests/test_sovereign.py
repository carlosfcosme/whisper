"""Sovereign tickets: fail if bind is not 127.0.0.1, Hub is contacted, or weights are pulled."""

import inspect
import re
import urllib.request
from pathlib import Path

import pytest
import torch

import whisper
from whisper.offline import (
    LOCALHOST,
    BindNotLoopback,
    DownloadHelperCalled,
    assert_loopback,
    default_device,
    git_root,
    listen_localhost,
    offline_enabled,
)

_BIND_ALL_RE = re.compile(r"""\.bind\(\(\s*['"]0\.0\.0\.0['"]""")
_HUB_IMPORT_RE = re.compile(
    r"^\s*(import huggingface_hub|from huggingface_hub)\b", re.MULTILINE
)
_CUDA_AUTO_RE = re.compile(
    r"""device\s*=\s*["']cuda["']\s*if\s*torch\.cuda\.is_available\(\)"""
)


def _package_py_files():
    root = Path(whisper.__file__).resolve().parent
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def test_ticket1_listen_only_127_0_0_1():
    sock = listen_localhost()
    try:
        host, port = sock.getsockname()[:2]
        assert host == LOCALHOST
        assert host != "0.0.0.0"
        assert port > 0
    finally:
        sock.close()


def test_ticket1_fails_if_bind_is_not_loopback():
    with pytest.raises(BindNotLoopback, match="127.0.0.1"):
        assert_loopback("0.0.0.0")
    with pytest.raises(BindNotLoopback, match="127.0.0.1"):
        assert_loopback("::")
    with pytest.raises(BindNotLoopback, match="127.0.0.1"):
        assert_loopback("1.2.3.4")


def test_ticket1_package_does_not_bind_all_interfaces():
    for path in _package_py_files():
        text = path.read_text(encoding="utf-8")
        match = _BIND_ALL_RE.search(text)
        assert match is None, f"{path} binds 0.0.0.0"


def test_ticket2_inference_device_is_cpu_even_if_cuda_available(monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cpu"
    assert whisper.default_device() == "cpu"


def test_ticket2_load_model_does_not_auto_select_cuda():
    src = Path(whisper.__file__).read_text(encoding="utf-8")
    assert _CUDA_AUTO_RE.search(src) is None
    cli_src = Path(inspect.getfile(whisper.transcribe)).read_text(encoding="utf-8")
    assert _CUDA_AUTO_RE.search(cli_src) is None
    assert "default_device()" in cli_src


def test_ticket3_fails_if_hub_is_contacted():
    with pytest.raises(DownloadHelperCalled, match="download helper"):
        urllib.request.urlopen("https://huggingface.co/openai/whisper-tiny")
    with pytest.raises(DownloadHelperCalled, match="download helper"):
        whisper.urllib.request.urlopen(
            "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt"
        )


def test_ticket3_fails_if_weights_are_pulled(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    monkeypatch.delenv("WHISPER_ALLOW_DOWNLOAD", raising=False)
    assert offline_enabled() is True
    with pytest.raises(RuntimeError, match="offline mode"):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert list(tmp_path.glob("*.pt")) == []


def test_ticket3_package_does_not_import_huggingface_hub():
    for path in _package_py_files():
        text = path.read_text(encoding="utf-8")
        assert _HUB_IMPORT_RE.search(text) is None, f"{path} imports huggingface_hub"
        assert "hf_hub_download" not in text, path
        assert "from_pretrained" not in text, path


def test_ticket3_ci_does_not_pull_weights_or_hit_hub():
    root = git_root()
    assert root
    yml = Path(root, ".github/workflows/test.yml").read_text(encoding="utf-8")
    assert "test_transcribe[tiny]" not in yml
    assert "test_transcribe[tiny.en]" not in yml
    assert "huggingface.co" not in yml
    assert "hf.co" not in yml
    assert "WHISPER_OFFLINE" in yml
    assert "not requires_download" in yml
    assert "not test_transcribe" in yml
