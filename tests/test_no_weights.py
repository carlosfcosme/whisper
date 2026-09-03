import importlib.util
import urllib.request
from pathlib import Path

import pytest

import whisper

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repo_has_no_committed_weights_or_secrets():
    check = _load_script("check_no_weights", "scripts/check_no_weights.py")
    assert check.find_violations(REPO_ROOT) == []
    assert check.main() == 0


def test_planted_checkpoint_is_a_violation(tmp_path):
    check = _load_script("check_no_weights", "scripts/check_no_weights.py")
    planted = tmp_path / "tiny.pt"
    planted.write_bytes(b"not-a-real-checkpoint")
    violations = check.find_violations(tmp_path, ["tiny.pt"])
    assert violations
    assert violations[0][0] == "tiny.pt"


def test_planted_cache_path_is_a_violation():
    check = _load_script("check_no_weights", "scripts/check_no_weights.py")
    reason = check.classify(".cache/whisper/tiny.pt", 128)
    assert reason is not None
    assert "cache" in reason or "weight" in reason or "checkpoint" in reason


def test_planted_secret_is_a_violation():
    check = _load_script("check_no_weights", "scripts/check_no_weights.py")
    assert check.classify(".env", 12) is not None
    assert check.classify("id_rsa", 12) is not None
    assert check.classify("server.pem", 12) is not None


def test_load_model_does_not_pull_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    called = []

    def boom(*args, **kwargs):
        called.append(True)
        raise AssertionError("urlopen must not be called")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(whisper.WeightDownloadError, match="weight pull is disabled"):
        whisper.load_model("tiny")
    assert called == []
    assert list(tmp_path.rglob("*.pt")) == []


def test_bind_localhost_policy_passes():
    check = _load_script("check_bind_localhost", "scripts/check_bind_localhost.py")
    assert check.main() == 0
