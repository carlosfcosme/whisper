"""CPU default, localhost bind, no Hub, CI weight gate. Does not download weights."""

import ast
import json
import subprocess
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import whisper
from whisper.defaults import (
    DEFAULT_DEVICE,
    DEFAULT_HOST,
    committed_weight_paths,
    is_huggingface_hub_source,
    reject_huggingface_hub,
    require_loopback_host,
)
from whisper.local_server import create_server

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_SCRIPT = ROOT / ".github" / "scripts" / "fail_if_weights_committed.sh"


@pytest.fixture
def loopback_server():
    server = create_server(DEFAULT_HOST, 0, device=DEFAULT_DEVICE)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _json_request(host, port, path, method="GET", payload=None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(
        f"http://{host}:{port}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=2) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8")
        return exc.code, json.loads(body) if body else {}


def test_default_device_is_cpu():
    assert DEFAULT_DEVICE == "cpu"
    assert whisper.DEFAULT_DEVICE == "cpu"


def test_load_model_and_cli_default_to_cpu():
    source = (ROOT / "whisper" / "transcribe.py").read_text(encoding="utf-8")
    assert "default=DEFAULT_DEVICE" in source
    assert 'default="cuda" if torch.cuda.is_available()' not in source


def test_reject_huggingface_hub_urls():
    hub = "https://huggingface.co/openai/whisper-tiny"
    assert is_huggingface_hub_source(hub)
    with pytest.raises(ValueError, match="Hugging Face Hub"):
        reject_huggingface_hub(hub)
    with pytest.raises(ValueError, match="Hugging Face Hub"):
        whisper.resolve_local_checkpoint(hub)
    with pytest.raises(ValueError, match="Hugging Face Hub"):
        whisper.load_model(hub, local_only=True)


def test_resolve_local_checkpoint_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="local checkpoint not found"):
        whisper.resolve_local_checkpoint("tiny")
    assert list(tmp_path.rglob("*.pt")) == []


def test_require_loopback_host():
    assert require_loopback_host("127.0.0.1") == "127.0.0.1"
    with pytest.raises(ValueError, match="127.0.0.1"):
        require_loopback_host("0.0.0.0")
    with pytest.raises(ValueError, match="127.0.0.1"):
        create_server("0.0.0.0", 0)


def test_server_binds_127_and_health_is_cpu(loopback_server):
    host, port = loopback_server.server_address
    assert host == "127.0.0.1"
    assert loopback_server.device == "cpu"
    status, body = _json_request(host, port, "/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["host"] == "127.0.0.1"
    assert body["device"] == "cpu"
    assert body["hub"] is False


def test_server_rejects_hub_and_missing_checkpoint(loopback_server):
    host, port = loopback_server.server_address
    status, body = _json_request(
        host,
        port,
        "/transcribe",
        method="POST",
        payload={
            "model": "https://huggingface.co/openai/whisper-tiny",
            "audio": str(ROOT / "tests" / "jfk.flac"),
        },
    )
    assert status == 400
    assert "Hub" in body["error"]

    status, body = _json_request(
        host,
        port,
        "/transcribe",
        method="POST",
        payload={"model": "/tmp/missing-whisper-checkpoint.pt", "audio": "x"},
    )
    assert status == 404


def test_committed_weight_paths_and_ci_script(tmp_path):
    assert committed_weight_paths(["whisper/audio.py", "README.md"]) == []
    assert committed_weight_paths(["models/tiny.pt"]) == ["models/tiny.pt"]
    assert committed_weight_paths(["foo.safetensors"]) == ["foo.safetensors"]

    subprocess.check_call(["bash", str(WEIGHTS_SCRIPT)], cwd=str(ROOT))

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init"], cwd=str(repo))
    subprocess.check_call(
        ["git", "config", "user.email", "ci@example.com"], cwd=str(repo)
    )
    subprocess.check_call(["git", "config", "user.name", "ci"], cwd=str(repo))
    (repo / "model.pt").write_bytes(b"not-a-real-checkpoint")
    subprocess.check_call(["git", "add", "model.pt"], cwd=str(repo))
    result = subprocess.run(
        ["bash", str(WEIGHTS_SCRIPT)],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "model.pt" in result.stderr


def test_ci_does_not_download_hub_or_weights():
    yml = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    assert "-k 'not test_transcribe'" in yml
    assert "test_transcribe[tiny]" not in yml
    assert "HF_HUB_OFFLINE" in yml
    assert "fail_if_weights_downloaded.sh" in yml
    assert "Hugging Face Hub must not be a project dependency" in yml
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "huggingface" not in pyproject.lower()
    assert "huggingface" not in requirements.lower()


def test_gitignore_covers_cache_and_weights():
    def ignored(path: str) -> bool:
        return (
            subprocess.call(
                ["git", "check-ignore", "-q", path],
                cwd=str(ROOT),
            )
            == 0
        )

    assert ignored(".cache/whisper/tiny.pt")
    assert ignored("cache/whisper/base.pt")
    assert ignored("model.pt")
    assert ignored("weights/model.safetensors")
    assert not ignored("tests/jfk.flac")
    assert not ignored("whisper/assets/multilingual.tiktoken")


def test_package_does_not_import_huggingface_hub():
    for path in (ROOT / "whisper").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "huggingface" not in alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert "huggingface" not in node.module
