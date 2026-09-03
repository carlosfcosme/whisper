"""Offline, localhost-only runtime tests.

Model and network downloads are prohibited. Artifacts stay in tmp_path
and are gitignored when they land in the tree.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

import whisper

REPO_ROOT = Path(__file__).resolve().parents[1]
IGNORED_ARTIFACTS = (
    "artifacts/health.json",
    "outputs/jfk.txt",
    "outputs/jfk.vtt",
    "tests/jfk.txt",
    "tests/jfk.vtt",
    "tests/jfk.srt",
    "tests/jfk.tsv",
    "tiny.pt",
    ".cache/whisper/tiny.pt",
    "coverage.xml",
    "htmlcov/index.html",
)


def _git_check_ignore(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", path],
        cwd=REPO_ROOT,
        check=False,
    )
    return result.returncode == 0


def test_whisper_offline_env_is_set():
    assert os.environ.get("WHISPER_OFFLINE") == "1"
    assert whisper.is_offline()


def test_named_model_download_is_prohibited(tmp_path):
    with pytest.raises(RuntimeError, match="download prohibited"):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert list(tmp_path.glob("*.pt")) == []
    assert list(tmp_path.rglob("*.pt")) == []


def test_non_loopback_network_download_is_prohibited():
    with pytest.raises(RuntimeError, match="network download prohibited"):
        urllib.request.urlopen("https://example.com/", timeout=2)


def test_runtime_serve_offline_localhost(tmp_path):
    env = os.environ.copy()
    env["WHISPER_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "whisper.serve",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
        ],
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    port = None
    lines = []
    deadline = time.time() + 15
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                rest = proc.stdout.read() if proc.stdout else ""
                pytest.fail(
                    "serve exited {}: {}".format(proc.returncode, "".join(lines) + rest)
                )
            line = proc.stdout.readline()
            if not line:
                continue
            lines.append(line)
            match = re.search(r"http://127\.0\.0\.1:(\d+)", line)
            if match:
                port = int(match.group(1))
                break
        assert port, "serve did not print a loopback URL: {}".format("".join(lines))

        with urllib.request.urlopen(
            "http://127.0.0.1:{}/health".format(port), timeout=2
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["status"] == "ok"
        assert body["bind"] == "127.0.0.1"
        assert body["weights"] is False

        artifact = tmp_path / "artifacts" / "health.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(json.dumps(body) + "\n")
        assert artifact.is_file()
        assert not (REPO_ROOT / "artifacts" / "health.json").exists()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_runtime_serve_refuses_all_interfaces(tmp_path):
    env = os.environ.copy()
    env["WHISPER_OFFLINE"] = "1"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "whisper.serve",
            "--host",
            "0.0.0.0",
            "--port",
            "0",
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "127.0.0.1" in proc.stderr


def test_runtime_artifacts_are_gitignored():
    failed = [path for path in IGNORED_ARTIFACTS if not _git_check_ignore(path)]
    assert failed == [], "expected these artifacts to be gitignored: {}".format(failed)


def test_tracked_fixtures_are_not_gitignored():
    for path in (
        "tests/jfk.flac",
        "tests/conftest.py",
        "data/meanwhile.json",
        "README.md",
    ):
        assert not _git_check_ignore(path), path
        assert (REPO_ROOT / path).is_file()
