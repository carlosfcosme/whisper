"""Remaining sovereignty tests: loopback bind, CPU default, CI no-Hub, gitignore."""

import importlib.util
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import whisper
from whisper.offline import weight_pull_allowed
from whisper.serve import (
    ALL_INTERFACES,
    LOOPBACK_BIND,
    BindError,
    require_loopback_bind,
    serve,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"
GITIGNORE = REPO_ROOT / ".gitignore"
INSTALL = REPO_ROOT / ".cursor" / "install.sh"
START = REPO_ROOT / ".cursor" / "start.sh"

CACHE_WEIGHT_IGNORE_PATTERNS = (
    "*.pt",
    "*.pth",
    "*.safetensors",
    "*.onnx",
    "*.bin",
    ".cache/",
    "cache/",
    "weights/",
    "checkpoints/",
    ".huggingface/",
)

IGNORED_SAMPLE_PATHS = (
    "tiny.pt",
    "model.pth",
    "weights/encoder.bin",
    "weights/tiny.pt",
    "checkpoints/model.safetensors",
    "cache/whisper/tiny.pt",
    ".cache/whisper/tiny.pt",
    ".huggingface/hub/models--openai--whisper-tiny/pytorch_model.bin",
)

LS_FILES_PATHSPECS = (
    ".cache",
    ".cache/**",
    "cache",
    "cache/**",
    "weights",
    "weights/**",
    "checkpoints",
    "checkpoints/**",
    "*.pt",
    "*.pth",
    "*.safetensors",
)

TRACKED_ASSETS = (
    "whisper/assets/gpt2.tiktoken",
    "whisper/assets/multilingual.tiktoken",
    "whisper/assets/mel_filters.npz",
    "whisper/normalizers/english.json",
    "tests/jfk.flac",
    "README.md",
)

REMAINING_LOCAL_TESTS = (
    "test_audio.py",
    "test_tokenizer.py",
    "test_timing.py",
    "test_normalizer.py",
)


def _workflow_text():
    return WORKFLOW.read_text(encoding="utf-8")


def _load_script(name):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_script(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / args[0])] + list(args[1:]),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _git(*args):
    return subprocess.run(
        ["git"] + list(args),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_bind_127_0_0_1_is_the_loopback_contract():
    assert LOOPBACK_BIND == "127.0.0.1"
    assert require_loopback_bind() == "127.0.0.1"
    assert require_loopback_bind("127.0.0.1") == "127.0.0.1"
    assert require_loopback_bind("localhost") == "127.0.0.1"
    with pytest.raises(BindError, match="127.0.0.1"):
        require_loopback_bind(ALL_INTERFACES)


def test_bind_127_0_0_1_socket_roundtrip():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(("127.0.0.1", 0))
        host, port = server.getsockname()
        assert host == "127.0.0.1"
        assert port > 0
        server.listen(1)
        client = socket.create_connection(("127.0.0.1", port), timeout=1)
        try:
            conn, addr = server.accept()
            try:
                assert addr[0] == "127.0.0.1"
                client.sendall(b"ok")
                assert conn.recv(2) == b"ok"
            finally:
                conn.close()
        finally:
            client.close()
    finally:
        server.close()


def test_bind_127_0_0_1_is_asserted_by_live_serve():
    httpd = serve(port=0)
    try:
        host, port = httpd.server_address[:2]
        assert host == "127.0.0.1"
        assert port > 0
    finally:
        httpd.server_close()


def test_wildcard_socket_bind_is_blocked_in_unit_tests():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(OSError, match="127.0.0.1"):
            sock.bind((ALL_INTERFACES, 0))
    finally:
        sock.close()


def test_start_and_ci_do_not_bind_all_interfaces():
    start = START.read_text(encoding="utf-8")
    assert "127.0.0.1" in start
    assert "--host 127.0.0.1" in start
    assert ALL_INTERFACES not in start
    assert ALL_INTERFACES not in _workflow_text()
    env = (REPO_ROOT / ".cursor" / "environment.json").read_text(encoding="utf-8")
    assert ALL_INTERFACES not in env


def test_cpu_only_default():
    assert whisper.DEFAULT_DEVICE == "cpu"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert torch.device(whisper.DEFAULT_DEVICE).type == "cpu"


def test_cpu_only_default_ignores_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert whisper.DEFAULT_DEVICE == "cpu"
    source = (REPO_ROOT / "whisper" / "__init__.py").read_text(encoding="utf-8")
    assert 'DEFAULT_DEVICE = "cpu"' in source
    assert 'device = "cuda" if torch.cuda.is_available() else "cpu"' not in source


def test_cli_help_is_cpu_only_and_does_not_pull_weights(tmp_path):
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "whisper", "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--device" in result.stdout
    assert "default: cpu" in result.stdout
    weights = list(tmp_path.rglob("*.pt")) + list(tmp_path.rglob("*.pth"))
    assert weights == [], weights


def test_ci_and_install_are_cpu_wheels():
    workflow = _workflow_text()
    assert "torch==${{ matrix.pytorch-version }}+cpu" in workflow
    assert "download.pytorch.org/whl/cpu" in workflow
    install = INSTALL.read_text(encoding="utf-8")
    assert "torch==2.5.1+cpu" in install
    assert "download.pytorch.org/whl/cpu" in install


def test_ci_does_not_download_hub_or_weights():
    workflow = _workflow_text()
    assert 'HF_HUB_OFFLINE: "1"' in workflow
    assert 'WHISPER_OFFLINE: "1"' in workflow
    assert "scripts/assert_no_weight_cache.py" in workflow
    assert "scripts/check_ci_skips_hub.py" in workflow
    assert "scripts/check_ci_cache_no_weights.py" in workflow
    assert "scripts/verify_ignored_artifacts.py" in workflow
    assert "-k 'not test_transcribe'" in workflow
    assert "huggingface.co" not in workflow
    assert "hf_hub_download" not in workflow
    assert "snapshot_download" not in workflow
    assert "from_pretrained" not in workflow
    assert weight_pull_allowed() is False

    install = INSTALL.read_text(encoding="utf-8")
    assert "huggingface.co" not in install
    assert "load_model" not in install
    assert "openaipublic.azureedge.net" not in install

    skip = _run_script("check_ci_skips_hub.py")
    assert skip.returncode == 0, skip.stderr
    cache = _run_script("check_ci_cache_no_weights.py")
    assert cache.returncode == 0, cache.stderr


def test_ci_cache_paths_are_pip_and_precommit_only():
    checker = _load_script("check_ci_cache_no_weights.py")
    text = _workflow_text()
    paths = checker._cache_paths_from_text(text)
    assert paths, "expected at least one actions/cache path"
    assert all(not checker._is_forbidden_cache_path(path) for path in paths)
    joined = " ".join(paths)
    assert "pre-commit" in joined
    assert "whisper" not in joined.lower()
    assert "huggingface" not in joined.lower()


def test_forbidden_weight_cache_paths_are_detected():
    checker = _load_script("check_ci_cache_no_weights.py")
    forbidden = [
        "~/.cache/whisper",
        "~/.cache/huggingface",
        "weights/",
        "*.pt",
        "*.safetensors",
        "~/.cache",
    ]
    allowed = [
        "${{ steps.pip-cache.outputs.dir }}",
        "~/.cache/pre-commit",
    ]
    for path in forbidden:
        assert checker._is_forbidden_cache_path(path), path
    for path in allowed:
        assert not checker._is_forbidden_cache_path(path), path


def test_gitignore_covers_cache_and_weights():
    text = GITIGNORE.read_text(encoding="utf-8")
    for pattern in CACHE_WEIGHT_IGNORE_PATTERNS:
        assert pattern in text, "missing gitignore pattern: {}".format(pattern)


@pytest.mark.parametrize("relpath", IGNORED_SAMPLE_PATHS)
def test_git_ignores_cache_and_weight_paths(relpath):
    result = _git("check-ignore", "-q", "--", relpath)
    assert result.returncode == 0, "expected git to ignore {}".format(relpath)


def test_cache_and_weight_paths_are_untracked():
    listed = _git("ls-files", "-z", "--", *LS_FILES_PATHSPECS)
    assert listed.returncode == 0, listed.stderr
    tracked = [path for path in listed.stdout.split("\0") if path]
    assert tracked == [], "cache/weight paths must stay untracked: {}".format(tracked)


def test_tracked_assets_are_not_gitignored():
    for relpath in TRACKED_ASSETS:
        result = _git("check-ignore", "-q", "--", relpath)
        assert result.returncode == 1, "did not expect {} to be gitignored".format(
            relpath
        )
        assert (REPO_ROOT / relpath).is_file()


def test_remaining_local_tests_do_not_pull_hub_or_weights():
    forbidden = (
        "huggingface",
        "hf_hub",
        "from_pretrained",
        "openaipublic.azureedge.net",
        "load_model",
    )
    for name in REMAINING_LOCAL_TESTS:
        text = (REPO_ROOT / "tests" / name).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, "{} must not reference {}".format(name, token)
