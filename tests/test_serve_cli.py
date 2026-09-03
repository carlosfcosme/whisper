"""CLI/start integration: fail if bind is not 127.0.0.1 or weights are pulled."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BIND = "127.0.0.1"
WILDCARD = "0.0.0.0"
BOUND_RE = re.compile(r"whisper serve bound to (127\.0\.0\.1):(\d+)")


def _env(tmp_path, extra=None):
    env = os.environ.copy()
    env["WHISPER_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["XDG_CACHE_HOME"] = str(tmp_path)
    env.pop("WHISPER_DEVICE", None)
    env.pop("HF_TOKEN", None)
    env.pop("HUGGING_FACE_HUB_TOKEN", None)
    env.pop("HUGGINGFACE_HUB_TOKEN", None)
    if extra:
        env.update(extra)
    return env


def _stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


def _wait_bound(proc, timeout=8):
    deadline = time.time() + timeout
    buf = []
    while time.time() < deadline:
        if proc.poll() is not None:
            err = ""
            if proc.stderr:
                err += proc.stderr.read()
            if proc.stdout:
                err += proc.stdout.read()
            raise AssertionError(f"serve exited {proc.returncode}: {err}{''.join(buf)}")
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.05)
            continue
        buf.append(line)
        match = BOUND_RE.search(line)
        if match:
            return match.group(1), int(match.group(2))
    _stop(proc)
    raise AssertionError(f"timed out waiting for 127.0.0.1 bind: {''.join(buf)}")


def _no_weights(tmp_path):
    written = list(tmp_path.rglob("*.pt")) + list(tmp_path.rglob("*.pth"))
    assert written == [], f"weight file written: {written}"


def test_module_binds_127_and_refuses_wildcard(tmp_path):
    proc = subprocess.Popen(
        [sys.executable, "-m", "whisper.serve", "--host", BIND, "--port", "0"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_env(tmp_path),
    )
    try:
        host, port = _wait_bound(proc)
        assert host == BIND
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
            body = json.loads(response.read())
        assert body["bind"] == BIND
        assert body["device"] == "cpu"
        assert body["hub"] is False
        assert body["weights"] is False
    finally:
        _stop(proc)
    _no_weights(tmp_path)


def test_module_fails_if_host_is_not_127(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "whisper.serve", "--host", WILDCARD, "--port", "0"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=8,
        env=_env(tmp_path),
    )
    assert proc.returncode == 2
    assert "127.0.0.1" in proc.stderr
    assert "bound to" not in proc.stdout
    _no_weights(tmp_path)


def test_whisper_serve_subcommand_fails_on_wildcard(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "whisper", "serve", "--host", WILDCARD],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=8,
        env=_env(tmp_path),
    )
    assert proc.returncode == 2
    _no_weights(tmp_path)


def test_start_script_binds_127(tmp_path):
    proc = subprocess.Popen(
        ["bash", str(REPO_ROOT / ".cursor" / "start.sh")],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_env(tmp_path, {"WHISPER_SERVE_PORT": "0"}),
    )
    try:
        host, port = _wait_bound(proc)
        assert host == BIND
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
            body = json.loads(response.read())
        assert body["bind"] == BIND
        assert body["weights"] is False
    finally:
        _stop(proc)
    _no_weights(tmp_path)


def test_cli_help_defaults_to_cpu(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "whisper", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=8,
        env=_env(tmp_path),
    )
    assert proc.returncode == 0
    assert "default: cpu" in proc.stdout
    _no_weights(tmp_path)
