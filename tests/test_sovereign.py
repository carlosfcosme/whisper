"""Tickets 1-3: bind 127.0.0.1, CPU-only, no Hub / no weight pull."""

import importlib
import socket
import urllib.request
from pathlib import Path

import pytest
import torch

import whisper
from whisper.bind import ALL_INTERFACES, LOOPBACK_HOST, BindError, require_loopback_host
from whisper.serve import serve

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"
HUB_URL = "https://huggingface.co/openai/whisper-tiny/resolve/main/pytorch_model.bin"
WEIGHT_URL = "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt"
FORBIDDEN_NAMES = {
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
    "spark.yaml",
    "spark.yml",
}


def test_ticket1_bind_default_is_127_0_0_1():
    assert LOOPBACK_HOST == "127.0.0.1"
    assert require_loopback_host() == "127.0.0.1"
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_ticket1_socket_fails_if_bind_is_not_127_0_0_1():
    httpd = serve(port=0)
    try:
        host = httpd.socket.getsockname()[0]
        assert host == "127.0.0.1"
        assert host != ALL_INTERFACES
    finally:
        httpd.server_close()


def test_ticket1_all_interfaces_bind_is_rejected():
    with pytest.raises(BindError):
        require_loopback_host(ALL_INTERFACES)


def test_ticket2_cpu_only_even_if_cuda_reports_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert torch.cuda.is_available() is True
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_ticket2_cli_device_does_not_auto_select_cuda():
    transcribe_mod = importlib.import_module("whisper.transcribe")
    source = Path(transcribe_mod.__file__).read_text(encoding="utf-8")
    device_lines = [line for line in source.splitlines() if '"--device"' in line]
    assert device_lines
    assert "DEFAULT_DEVICE" in device_lines[0]
    assert "cuda.is_available" not in device_lines[0]


def test_ticket3_hub_contact_fails():
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        urllib.request.urlopen(HUB_URL)


def test_ticket3_weight_pull_fails():
    with pytest.raises(RuntimeError, match="weights|remote hosts|offline|no-store"):
        urllib.request.urlopen(WEIGHT_URL)
    with pytest.raises(RuntimeError, match="offline|no-store|no Hub"):
        whisper.load_model("tiny", device="cpu")


def test_ticket3_non_loopback_connect_fails():
    with pytest.raises(RuntimeError, match="non-loopback"):
        socket.create_connection(("huggingface.co", 443), timeout=1)
    with pytest.raises(RuntimeError, match="non-loopback"):
        socket.create_connection(("openaipublic.azureedge.net", 443), timeout=1)


def test_ticket3_ci_skips_hub_and_weight_pulls():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "HF_HUB_OFFLINE" in workflow
    assert "WHISPER_OFFLINE" in workflow
    assert "WHISPER_NO_STORE" in workflow
    assert "CUDA_VISIBLE_DEVICES" in workflow
    assert "-k 'not test_transcribe'" in workflow
    assert "assert_no_weight_download.py" in workflow
    assert "+cpu" in workflow
    assert "not requires_cuda" in workflow
    assert "bind-guard" in workflow
    assert "check_loopback_bind.py" in workflow
    assert "check_no_all_interfaces.sh" in workflow


def test_no_compose_spark_live_field_brain_or_keys():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", ".venv", "venv", "__pycache__"} for part in path.parts):
            continue
        assert path.name not in FORBIDDEN_NAMES
    for rel in ("whisper", ".cursor"):
        tree = REPO_ROOT / rel
        if not tree.exists():
            continue
        for path in tree.rglob("*"):
            if not path.is_file() or path.suffix in {".pyc"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            assert "Field-Brain" not in text
            assert "FIELD_BRAIN" not in text
            assert "OPENAI_API_KEY" not in text
            assert "--live" not in text
            assert "live=True" not in text
            assert "live = True" not in text
