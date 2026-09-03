import importlib.util
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_check_no_weights():
    path = REPO_ROOT / "scripts" / "check_no_weights.py"
    spec = importlib.util.spec_from_file_location("check_no_weights", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


check_no_weights = _load_check_no_weights()


def test_current_tree_has_no_committed_weights():
    assert check_no_weights.find_violations(REPO_ROOT) == []
    assert check_no_weights.main() == 0


@pytest.mark.parametrize(
    "relpath",
    [
        "models/tiny.pt",
        "cache/model.safetensors",
        "export/whisper.onnx",
        "weights/encoder.bin",
    ],
)
def test_weight_suffixes_are_rejected(relpath):
    reason = check_no_weights.classify(relpath, size=128)
    assert reason is not None
    assert "weight" in reason or "checkpoint" in reason


def test_mel_filters_npz_is_accepted():
    assert (
        check_no_weights.classify("whisper/assets/mel_filters.npz", size=4271) is None
    )


def test_oversized_file_is_rejected():
    reason = check_no_weights.classify(
        "blob.dat", size=check_no_weights.MAX_FILE_BYTES + 1
    )
    assert reason is not None
    assert "large file" in reason


def test_find_violations_detects_planted_checkpoint(tmp_path):
    planted = tmp_path / "tiny.pt"
    planted.write_bytes(b"not-a-real-checkpoint")
    violations = check_no_weights.find_violations(tmp_path, ["tiny.pt"])
    assert violations == [("tiny.pt", "model weight or checkpoint (.pt)")]


def test_script_fails_when_weight_is_tracked(tmp_path):
    subprocess.check_call(["git", "init"], cwd=str(tmp_path))
    subprocess.check_call(
        ["git", "config", "user.email", "ci@example.com"], cwd=str(tmp_path)
    )
    subprocess.check_call(["git", "config", "user.name", "ci"], cwd=str(tmp_path))
    planted = tmp_path / "leaked.pt"
    planted.write_bytes(b"weights")
    subprocess.check_call(["git", "add", "leaked.pt"], cwd=str(tmp_path))
    subprocess.check_call(["git", "commit", "-m", "plant"], cwd=str(tmp_path))

    violations = check_no_weights.find_violations(tmp_path)
    assert any(path == "leaked.pt" for path, _reason in violations)


def test_ci_workflow_runs_the_weight_guard():
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    assert "no-committed-weights" in workflow
    assert "scripts/check_no_weights.py" in workflow


def test_script_is_executable():
    path = REPO_ROOT / "scripts" / "check_no_weights.py"
    assert path.is_file()
    if os.name != "nt":
        assert path.stat().st_mode & stat.S_IXUSR
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "OK:" in result.stdout
