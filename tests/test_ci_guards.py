"""CI guards: committed weights, HF Hub fetches, CPU default. No weight pull."""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relpath: str):
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


device = _load("whisper_device_isolated", "whisper/device.py")
check_no_weights = _load("check_no_weights", "scripts/check_no_weights.py")
check_no_hf_hub = _load("check_no_hf_hub", "scripts/check_no_hf_hub.py")
assert_no_weight_cache = _load(
    "assert_no_weight_cache", "scripts/assert_no_weight_cache.py"
)


def test_default_device_is_cpu(monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    assert device.default_device() == "cpu"


def test_default_device_honors_override(monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    assert device.default_device() == "cuda"


def test_hf_hub_urlopen_is_blocked():
    host = "huggingface" + ".co"
    with pytest.raises(RuntimeError, match="HuggingFace Hub"):
        urllib.request.urlopen("https://{}/openai/whisper-tiny".format(host))


def test_hf_hub_socket_is_blocked():
    host = "huggingface" + ".co"
    with pytest.raises(RuntimeError, match="HuggingFace Hub"):
        socket.create_connection((host, 443), timeout=1)


def test_check_no_weights_passes_on_this_repo():
    assert check_no_weights.find_violations(REPO_ROOT) == []
    assert check_no_weights.assert_gitignore(REPO_ROOT) == []
    assert check_no_weights.main() == 0


def test_check_no_weights_fails_on_tracked_checkpoint():
    hits = check_no_weights.find_violations(
        REPO_ROOT, relative_paths=["tiny.pt", "weights/model.bin", ".cache/whisper/x"]
    )
    assert [path for path, _ in hits] == [
        "tiny.pt",
        "weights/model.bin",
        ".cache/whisper/x",
    ]


def test_check_no_hf_hub_passes_on_this_repo():
    assert check_no_hf_hub.main() == 0


def test_check_no_hf_hub_fails_on_hub_download(tmp_path):
    rogue = tmp_path / "tests"
    rogue.mkdir()
    api = "from_" + "pretrained"
    (rogue / "bad.py").write_text("{}('openai/whisper-tiny')\n".format(api))
    hits = check_no_hf_hub.scan([rogue / "bad.py"], check_no_hf_hub.DOWNLOAD_PATTERNS)
    assert hits, hits


def test_gitignore_covers_weight_and_cache_paths():
    for path in (
        "tiny.pt",
        "weights/tiny.pt",
        ".cache/whisper/tiny.pt",
        "cache/whisper/tiny.pt",
        "model.pth",
        ".env",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", path],
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, path


def test_offline_env_defaults():
    assert os.environ.get("WHISPER_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("WHISPER_DEVICE") == "cpu"


def test_assert_no_weight_cache_empty(tmp_path):
    assert assert_no_weight_cache.find_cached_weights([tmp_path]) == []


def test_assert_no_weight_cache_finds_checkpoint(tmp_path):
    weight = tmp_path / "tiny.pt"
    weight.write_bytes(b"not-a-real-checkpoint")
    found = assert_no_weight_cache.find_cached_weights([tmp_path])
    assert found == [weight]
