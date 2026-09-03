"""Offline invariant: localhost binds, no bootstrap weight pull, gitignored caches."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import whisper
from whisper.invariant import check_offline_invariant
from whisper.offline import GITIGNORED_WEIGHT_PATTERNS, weights_offline

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_check_offline_invariant_passes():
    assert check_offline_invariant() == []
    assert whisper.check_offline_invariant() == []


def test_invariant_script_exits_zero():
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "whisper" / "invariant.py")],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "offline invariant OK" in proc.stdout


def test_ci_runs_offline_invariant():
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text()
    assert "offline-invariant:" in workflow
    assert "python3 whisper/invariant.py" in workflow


def test_runtime_bootstrap_writes_no_weights(tmp_path):
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(tmp_path)
    env["WHISPER_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    code = (
        "import whisper;"
        "sock = whisper.listen();"
        "assert sock.getsockname()[0] == '127.0.0.1';"
        "sock.close();"
        "whisper.available_models();"
        "assert whisper.bind_host('localhost') == '127.0.0.1';"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert list(tmp_path.rglob("*.pt")) == []
    assert list(tmp_path.rglob("*.pth")) == []
    assert list(tmp_path.iterdir()) == []


def test_load_model_offline_does_not_create_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_OFFLINE", "1")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert weights_offline()
    with pytest.raises(RuntimeError, match="offline"):
        whisper.load_model("tiny", device="cpu")
    assert list(tmp_path.rglob("*.pt")) == []
    assert list(tmp_path.rglob("*.pth")) == []
    assert list(tmp_path.iterdir()) == []


def test_listen_localhost_alias_binds_loopback():
    server = whisper.listen(host="localhost")
    try:
        assert server.getsockname()[0] == "127.0.0.1"
    finally:
        server.close()


def test_gitignore_patterns_match_offline_constant():
    lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    missing = [p for p in GITIGNORED_WEIGHT_PATTERNS if p not in lines]
    assert missing == []
