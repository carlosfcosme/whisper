import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_check_no_weights():
    path = REPO_ROOT / "scripts" / "check_no_weights.py"
    spec = importlib.util.spec_from_file_location("check_no_weights", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_check_no_weights_classifies_extensions():
    check = _load_check_no_weights()
    assert check.classify("models/tiny.pt", 100) is not None
    assert check.classify("weights/model.safetensors", 10) is not None
    assert check.classify("export/model.onnx", 10) is not None
    assert check.classify("libfoo.so", 100) is not None
    assert check.classify("whisper/assets/mel_filters.npz", 4271) is None
    assert check.classify("tests/jfk.flac", 1_152_693) is None
    assert check.classify("README.md", 800) is None
    assert check.classify("README.md", check.MAX_FILE_BYTES + 1) is not None


def test_check_no_weights_fails_on_planted_checkpoint(tmp_path):
    check = _load_check_no_weights()
    relpath = "models/tiny.pt"
    planted = tmp_path / relpath
    planted.parent.mkdir()
    planted.write_bytes(b"not-a-real-checkpoint")
    hits = check.find_violations(tmp_path, relative_paths=[relpath])
    assert hits
    assert hits[0][0] == relpath
    assert check.probe_negative() == 0


def test_check_no_weights_passes_on_this_repo():
    script = REPO_ROOT / "scripts" / "check_no_weights.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout


def test_check_no_weights_probe_negative_exits_zero():
    script = REPO_ROOT / "scripts" / "check_no_weights.py"
    result = subprocess.run(
        [sys.executable, str(script), "--probe-negative"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "planted checkpoint" in result.stdout
