"""Loopback / no-weights integration tests with WAN disabled.

Starts a real ``whisper.serve`` process on 127.0.0.1, refuses Hub/weight
pulls, and verifies planted cache/weight artifacts stay gitignored and unused.
"""

import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

import whisper
from whisper.netguard import NetworkDisabled, is_loopback_connect_host
from whisper.offline import WeightDownloadError

REPO_ROOT = Path(__file__).resolve().parents[1]
NETDISABLE_DIR = REPO_ROOT / "tests" / "netdisable"
VERIFY = REPO_ROOT / "scripts" / "verify_ignored_artifacts.py"

PLANTED_ARTIFACTS = (
    "tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "checkpoints/model.safetensors",
    ".cache/whisper/tiny.pt",
)


def _child_env(tmp_cache=None):
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(NETDISABLE_DIR) + (
        os.pathsep + existing if existing else ""
    )
    env["WHISPER_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("WHISPER_ALLOW_WEIGHT_DOWNLOAD", None)
    if tmp_cache is not None:
        env["XDG_CACHE_HOME"] = str(tmp_cache)
    return env


def _free_loopback_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def _wait_health(port, timeout=8.0):
    deadline = time.time() + timeout
    last = None
    url = "http://127.0.0.1:{}/health".format(port)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last = exc
            time.sleep(0.05)
    raise AssertionError("serve did not become ready on {}: {}".format(url, last))


def _start_serve(port, env):
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "whisper.serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _stop_serve(proc):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _run_verify(*args):
    return subprocess.run(
        [sys.executable, str(VERIFY)] + list(args),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_ignored_artifacts", VERIFY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
def test_wan_is_disabled_loopback_is_not():
    assert is_loopback_connect_host("127.0.0.1")
    assert not is_loopback_connect_host("huggingface.co")
    assert not is_loopback_connect_host("1.1.1.1")
    with pytest.raises(NetworkDisabled, match="network disabled"):
        socket.getaddrinfo("huggingface.co", 443)
    with pytest.raises(NetworkDisabled, match="network disabled"):
        socket.create_connection(("1.1.1.1", 443), timeout=1)
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", 9), timeout=0.5)


@pytest.mark.integration
def test_child_process_has_network_disabled(tmp_path):
    env = _child_env(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import socket, sys\n"
                "try:\n"
                "    socket.getaddrinfo('huggingface.co', 443)\n"
                "except OSError as exc:\n"
                "    sys.stderr.write(str(exc))\n"
                "    sys.exit(2)\n"
                "sys.exit(0)\n"
            ),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "network disabled" in result.stderr


@pytest.mark.integration
def test_serve_health_on_loopback_with_network_disabled(tmp_path):
    env = _child_env(tmp_path)
    port = _free_loopback_port()
    proc = _start_serve(port, env)
    try:
        body = _wait_health(port)
        assert body["status"] == "ok"
        assert body["bind"] == "127.0.0.1"
        assert body["device"] == "cpu"
        assert body["hub"] is False
        assert body["weights"] is False
        with pytest.raises(NetworkDisabled, match="network disabled"):
            socket.create_connection(("huggingface.co", 443), timeout=1)
        assert list(tmp_path.rglob("*.pt")) == []
        assert list(tmp_path.rglob("*.safetensors")) == []
    finally:
        _stop_serve(proc)
    assert proc.returncode is not None


@pytest.mark.integration
def test_named_model_does_not_write_weights_when_network_disabled(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    with pytest.raises(WeightDownloadError, match="no Hub|no weight pull"):
        whisper.load_model("tiny")
    assert list(tmp_path.rglob("*.pt")) == []

    env = _child_env(tmp_path)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import whisper, sys\n"
                "from whisper.offline import WeightDownloadError\n"
                "try:\n"
                "    whisper.load_model('tiny')\n"
                "except WeightDownloadError as exc:\n"
                "    sys.stderr.write(str(exc))\n"
                "    sys.exit(2)\n"
                "sys.exit(0)\n"
            ),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 2, child.stdout + child.stderr
    assert "no weight pull" in child.stderr or "no Hub" in child.stderr
    assert list(tmp_path.rglob("*.pt")) == []


@pytest.mark.integration
def test_ignored_artifacts_verified_and_unused_during_serve(tmp_path):
    static = _run_verify()
    assert static.returncode == 0, static.stderr
    planted = _run_verify("--plant")
    assert planted.returncode == 0, planted.stderr + planted.stdout

    verifier = _load_verifier()
    files, dirs = verifier.plant_artifacts(REPO_ROOT, PLANTED_ARTIFACTS)
    snapshots = {path: path.read_bytes() for path in files}
    try:
        errors = verifier.verify_planted_ignored(REPO_ROOT, PLANTED_ARTIFACTS)
        assert errors == [], errors
        assert verifier.tracked_artifact_paths(REPO_ROOT) == []

        env = _child_env(tmp_path)
        port = _free_loopback_port()
        proc = _start_serve(port, env)
        try:
            body = _wait_health(port)
            assert body["bind"] == "127.0.0.1"
            assert body["weights"] is False
            assert body["hub"] is False
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import whisper\n"
                        "from whisper.offline import WeightDownloadError\n"
                        "raised = False\n"
                        "try:\n"
                        "    whisper.load_model('tiny')\n"
                        "except WeightDownloadError:\n"
                        "    raised = True\n"
                        "raise SystemExit(0 if raised else 1)\n"
                    ),
                ],
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            assert child.returncode == 0, child.stderr
        finally:
            _stop_serve(proc)

        for path, payload in snapshots.items():
            assert path.is_file()
            assert path.read_bytes() == payload
        assert list(tmp_path.rglob("*.pt")) == []
        weights_guard = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "check_no_weights.py")],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert weights_guard.returncode == 0, weights_guard.stderr
    finally:
        verifier.remove_planted(files, dirs)
        assert not any(path.exists() for path in files)
