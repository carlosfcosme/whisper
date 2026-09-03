"""Bind guard must accept 127.0.0.1 and fail on 0.0.0.0."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_bind_guard():
    path = REPO_ROOT / "whisper" / "bind_guard.py"
    spec = importlib.util.spec_from_file_location("whisper_bind_guard", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bind_guard_accepts_127_0_0_1():
    guard = _load_bind_guard()
    assert guard.bind_guard("127.0.0.1") == "127.0.0.1"
    assert guard.bind_guard("localhost") == "127.0.0.1"
    assert guard.bind_guard("::1") == "::1"


def test_bind_guard_fails_on_0_0_0_0():
    guard = _load_bind_guard()
    with pytest.raises(guard.BindError, match="127.0.0.1"):
        guard.bind_guard("0.0.0.0")
    with pytest.raises(guard.BindError, match="127.0.0.1"):
        guard.bind_guard("::")
    with pytest.raises(guard.BindError, match="127.0.0.1"):
        guard.normalize_bind_host("0.0.0.0")


def test_bind_guard_script_passes():
    script = REPO_ROOT / "scripts" / "check_bind_guard.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "0.0.0.0" in result.stdout
    assert "127.0.0.1" in result.stdout
