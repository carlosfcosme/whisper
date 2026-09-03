"""Executable environment contract.

Runs the real bind, download, and gitignore paths. Does not pull weights
and does not listen off 127.0.0.1.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import pytest

from whisper.env_policy import (
    ALLOW_WEIGHT_FETCH_ENV,
    BIND_HOST,
    BindError,
    WeightFetchError,
)
from whisper.serve import make_server

REPO_ROOT = Path(__file__).resolve().parents[1]
WILDCARD = ".".join(["0"] * 4)
DUMMY_ARTIFACTS = (
    "env-exec-dummy.pt",
    "weights/env-exec-dummy.pt",
    ".cache/whisper/env-exec-dummy.pth",
)


@pytest.mark.environment
def test_bind_rejects_wildcard_before_listen():
    with pytest.raises(BindError, match="127.0.0.1"):
        make_server(WILDCARD, 0)


@pytest.mark.environment
def test_bind_serves_health_on_127_only():
    server = make_server(BIND_HOST, 0)
    try:
        host, port = server.server_address
        assert host == BIND_HOST
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        health = f"http://127.0.0.1:{port}/health"
        with urllib.request.urlopen(health) as response:
            body = json.loads(response.read())
        assert body["ok"] is True
        assert body["bind"] == BIND_HOST
        assert body["weights"] is False
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.environment
def test_serve_cli_refuses_wildcard_bind():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "whisper.serve",
            "--host",
            WILDCARD,
            "--port",
            "0",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert proc.returncode == 2
    assert BIND_HOST in proc.stderr
    assert WILDCARD in proc.stderr


@pytest.mark.environment
def test_start_script_forces_loopback():
    start = (REPO_ROOT / ".cursor" / "start.sh").read_text(encoding="utf-8")
    assert "--host 127.0.0.1" in start
    assert WILDCARD not in start
    assert "whisper.serve" in start


@pytest.mark.environment
@pytest.mark.parametrize("model_name", ["tiny", "tiny.en", "turbo"])
def test_named_model_cache_miss_does_not_fetch(monkeypatch, tmp_path, model_name):
    import whisper

    monkeypatch.setenv(ALLOW_WEIGHT_FETCH_ENV, "0")

    def boom(*args, **kwargs):
        raise AssertionError("weight fetch must not open a network URL")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(WeightFetchError):
        whisper.load_model(model_name, download_root=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.environment
def test_download_writes_nothing_on_refused_fetch(monkeypatch, tmp_path):
    import whisper

    monkeypatch.setenv(ALLOW_WEIGHT_FETCH_ENV, "0")
    url = whisper._MODELS["tiny"]
    with pytest.raises(WeightFetchError):
        whisper._download(url, str(tmp_path), in_memory=False)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.environment
def test_model_artifacts_are_gitignored():
    for rel in DUMMY_ARTIFACTS:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", rel],
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, rel


@pytest.mark.environment
def test_git_add_ignores_new_model_artifacts():
    created = []
    created_dirs = []
    try:
        for rel in DUMMY_ARTIFACTS:
            path = REPO_ROOT / rel
            if path.exists():
                continue
            parent = path.parent
            while parent != REPO_ROOT and not parent.exists():
                created_dirs.append(parent)
                parent = parent.parent
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"not-a-real-checkpoint")
            created.append(path)
            dry = subprocess.run(
                ["git", "add", "-n", "--", rel],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            assert dry.stdout.strip() == "", dry.stdout
            assert dry.returncode != 0 or "ignored" in dry.stderr
        if created:
            status = subprocess.run(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--",
                    *[str(path.relative_to(REPO_ROOT)) for path in created],
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            assert status.stdout.strip() == ""
    finally:
        for path in created:
            if path.exists():
                path.unlink()
        for directory in created_dirs:
            if (
                directory.exists()
                and directory != REPO_ROOT
                and not any(directory.iterdir())
            ):
                directory.rmdir()


@pytest.mark.environment
def test_no_weight_artifacts_are_tracked():
    listed = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--",
            "*.pt",
            "*.pth",
            ".cache",
            ".cache/**",
            "cache",
            "cache/**",
            "weights",
            "weights/**",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [path for path in listed.stdout.split("\0") if path]
    assert tracked == []
