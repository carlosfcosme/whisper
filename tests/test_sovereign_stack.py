import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import torch

import whisper
from whisper.device import DEFAULT_DEVICE, default_device
from whisper.localhost import (
    LOOPBACK_BIND,
    bind_host,
    check_download_url,
    is_huggingface_hub_url,
    listen,
)

HUB_URLS = (
    "https://huggingface.co/openai/whisper-tiny/resolve/main/tiny.pt",
    "https://hf.co/openai/whisper-tiny/resolve/main/tiny.pt",
    "https://cdn-lfs.huggingface.co/repos/tiny.pt",
)

FORBIDDEN_TREE_NAMES = frozenset(
    {
        "docker-compose.yml",
        "docker-compose.yaml",
        "compose.yaml",
        "compose.yml",
        "spark.yaml",
        "spark.yml",
    }
)
ROOT = Path(__file__).resolve().parents[1]


def _boom_urlopen(*args, **kwargs):
    raise AssertionError("urlopen must not run (Hub / weight pull)")


def test_bind_host_is_127_0_0_1():
    assert LOOPBACK_BIND == "127.0.0.1"
    assert bind_host() == "127.0.0.1"
    host, port = listen()
    assert host == "127.0.0.1"
    assert port == 0


@pytest.mark.parametrize("bad", ("0.0.0.0", "::", "[::]", "192.168.1.10", "localhost"))
def test_listen_rejects_non_loopback(bad):
    with pytest.raises(ValueError, match="127.0.0.1"):
        listen(bad, 8080)


def test_http_server_listens_on_127_0_0_1():
    host, port = listen(port=0)
    server = HTTPServer((host, port), BaseHTTPRequestHandler)
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert server.server_address[0] != "0.0.0.0"
    finally:
        server.server_close()


def test_cpu_is_default_even_when_cuda_reports_available(
    offline_checkpoint_path, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert DEFAULT_DEVICE == "cpu"
    assert default_device() == "cpu"
    model = whisper.load_model(offline_checkpoint_path)
    assert model.device.type == "cpu"


def test_cli_device_default_is_cpu():
    from whisper.transcribe import DEFAULT_DEVICE as CLI_DEVICE

    assert CLI_DEVICE == "cpu"


def test_load_model_local_path_does_not_download(offline_checkpoint_path, monkeypatch):
    monkeypatch.setattr(whisper, "_download", _boom_urlopen)
    model = whisper.load_model(offline_checkpoint_path)
    assert model.device.type == "cpu"
    assert model.dims.n_audio_layer == 1


@pytest.mark.parametrize("url", HUB_URLS)
def test_hub_contact_is_refused_without_urlopen(url, tmp_path, monkeypatch):
    monkeypatch.delenv("WHISPER_OFFLINE", raising=False)
    monkeypatch.setattr("urllib.request.urlopen", _boom_urlopen)
    assert is_huggingface_hub_url(url)
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        whisper._download(url, str(tmp_path), False)
    with pytest.raises(RuntimeError, match="Hugging Face Hub"):
        check_download_url(url, cache_hit=True)


@pytest.mark.parametrize("name", ["tiny", "tiny.en", "turbo"])
def test_named_model_does_not_pull_weights(name, tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr("urllib.request.urlopen", _boom_urlopen)
    with pytest.raises(RuntimeError, match="WHISPER_OFFLINE"):
        whisper.load_model(name)


def test_no_huggingface_hub_dependency():
    packaging = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "huggingface" not in packaging
    assert "huggingface" not in requirements


def test_no_committed_weight_files():
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
        cwd=str(ROOT),
        text=True,
    ).strip()
    assert listed == ""
    script = ROOT / ".github" / "scripts" / "fail-committed-weights.sh"
    subprocess.check_call(["bash", str(script)], cwd=str(ROOT))


def test_ci_does_not_hit_hub_or_pull_weights():
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    lowered = workflow.lower()
    assert "huggingface" not in lowered
    assert "test_transcribe[tiny]" not in workflow
    assert "WHISPER_OFFLINE" in workflow
    assert "not test_transcribe" in workflow or "not requires_weights" in workflow
    assert "fail-committed-weights.sh" in workflow
    assert "fail-remote-fixture-urls.sh" in workflow


def test_no_compose_spark_field_brain_or_keys():
    banned_tokens = (b"field-brain", b"field_brain", b"--live true")
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=str(ROOT), text=True
    ).splitlines()
    for rel in tracked:
        name = Path(rel).name.lower()
        assert name not in FORBIDDEN_TREE_NAMES, rel
        assert "field-brain" not in name
        if Path(rel).suffix.lower() in {".py", ".yml", ".yaml", ".sh", ".toml"}:
            data = (ROOT / rel).read_bytes().lower()
            for token in banned_tokens:
                assert token not in data, (rel, token)
