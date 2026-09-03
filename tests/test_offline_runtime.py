"""Offline runtime: bind 127.0.0.1, refuse model fetch, verify ignored weights."""

import json
import os
import re
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import pytest
from hub_offline import (
    HUB_OFFLINE_ENV,
    is_huggingface_hub_host,
    is_weight_fetch_host,
    urlopen_without_hub,
)

import whisper
from whisper.serve import ALL_INTERFACES, LOOPBACK_BIND, BindError, main, serve

REPO_ROOT = Path(__file__).resolve().parents[1]
LISTEN_RE = re.compile(r"http://127\.0\.0\.1:(\d+)")
IGNORED_WEIGHT_PATHS = (
    ".cache/whisper/tiny.pt",
    "models/tiny.pt",
    "weights/base.bin",
    "tiny.pt",
    "export.onnx",
)


def _readline_deadline(pipe, timeout):
    box = []

    def reader():
        box.append(pipe.readline())

    thread = threading.Thread(target=reader)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    return box[0] if box else ""


def _git_check_ignore(relpath):
    return subprocess.run(
        ["git", "check-ignore", "-q", "--", relpath],
        cwd=str(REPO_ROOT),
        check=False,
    ).returncode


def _run_cache_weights_check():
    return subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "check_cache_weights.sh")],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.offline_runtime
def test_runtime_hub_offline_env():
    for name in HUB_OFFLINE_ENV:
        assert os.environ.get(name) == "1"


@pytest.mark.offline_runtime
def test_runtime_serve_requires_127_0_0_1():
    httpd = serve(host=LOOPBACK_BIND, port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        with urllib.request.urlopen(
            "http://127.0.0.1:{}/health".format(port), timeout=2
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["status"] == "ok"
        assert body["bind"] == "127.0.0.1"
        assert body["device"] == "cpu"
        assert body["weights"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.mark.offline_runtime
def test_runtime_serve_refuses_all_interfaces():
    with pytest.raises(BindError, match="127.0.0.1"):
        serve(host=ALL_INTERFACES, port=0)
    assert main(["--host", ALL_INTERFACES, "--port", "0"]) == 2


@pytest.mark.offline_runtime
def test_runtime_subprocess_serve_binds_127():
    proc = subprocess.Popen(
        [sys.executable, "-m", "whisper.serve", "--host", "127.0.0.1", "--port", "0"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = _readline_deadline(proc.stdout, 8)
        match = LISTEN_RE.search(line or "")
        assert match, "serve did not announce 127.0.0.1: {}".format(line)
        port = int(match.group(1))
        with urllib.request.urlopen(
            "http://127.0.0.1:{}/health".format(port), timeout=2
        ) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["bind"] == "127.0.0.1"
        assert body["weights"] is False
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)


@pytest.mark.offline_runtime
def test_runtime_subprocess_serve_refuses_0_0_0_0():
    proc = subprocess.run(
        [sys.executable, "-m", "whisper.serve", "--host", "0.0.0.0", "--port", "0"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "127.0.0.1" in proc.stderr
    assert "0.0.0.0" in proc.stderr


@pytest.mark.offline_runtime
def test_runtime_load_model_does_not_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    called = []

    def boom(*args, **kwargs):
        called.append(args[:1])
        raise AssertionError("runtime tests must not fetch models")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(whisper.WeightDownloadError, match="weight pull is disabled"):
        whisper.load_model("tiny")
    assert called == []
    assert list(tmp_path.rglob("*.pt")) == []
    assert list(tmp_path.rglob("*.bin")) == []
    assert list(tmp_path.rglob("*.onnx")) == []


@pytest.mark.offline_runtime
def test_runtime_subprocess_load_model_does_not_fetch(tmp_path):
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(tmp_path)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    code = (
        "import whisper\n"
        "try:\n"
        "    whisper.load_model('tiny')\n"
        "except whisper.WeightDownloadError as exc:\n"
        "    print('REFUSED', exc)\n"
        "else:\n"
        "    raise SystemExit('model fetch succeeded')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "REFUSED" in proc.stdout
    assert list(tmp_path.rglob("*.pt")) == []


@pytest.mark.offline_runtime
@pytest.mark.parametrize(
    "url,kind",
    [
        ("https://huggingface.co/openai/whisper-tiny", "Hub"),
        (
            "https://openaipublic.azureedge.net/main/whisper/models/tiny.pt",
            "weights",
        ),
    ],
)
def test_runtime_urlopen_blocks_hub_and_weight_cdn(url, kind):
    host = url.split("/")[2]
    if kind == "Hub":
        assert is_huggingface_hub_host(host)
    else:
        assert is_weight_fetch_host(host)
    with pytest.raises(RuntimeError, match="must not"):
        urlopen_without_hub(url)


@pytest.mark.offline_runtime
def test_runtime_ignored_weights_after_failed_fetch(tmp_path):
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(tmp_path)
    env["HF_HUB_OFFLINE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import whisper\n"
            "try:\n"
            "    whisper.load_model('tiny')\n"
            "except whisper.WeightDownloadError:\n"
            "    pass\n",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )
    missed = [path for path in IGNORED_WEIGHT_PATHS if _git_check_ignore(path) != 0]
    assert missed == [], "gitignore does not cover: {}".format(missed)
    check = _run_cache_weights_check()
    assert check.returncode == 0, check.stderr or check.stdout
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=str(REPO_ROOT), text=True
    )
    weight_hits = [
        line
        for line in tracked.splitlines()
        if line.endswith((".pt", ".bin", ".onnx", ".safetensors"))
    ]
    assert weight_hits == []
    assert list(tmp_path.rglob("*.pt")) == []
