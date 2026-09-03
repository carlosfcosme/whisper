import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_check_no_weights():
    spec = importlib.util.spec_from_file_location(
        "check_no_weights", ROOT / "scripts" / "check_no_weights.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


check_no_weights = _load_check_no_weights()


def test_repo_has_no_committed_weights():
    violations = check_no_weights.find_violations(ROOT)
    assert violations == []
    assert check_no_weights.main() == 0


def test_check_fails_on_committed_pt(tmp_path):
    weight = tmp_path / "tiny.pt"
    weight.write_bytes(b"not-a-real-checkpoint")
    violations = check_no_weights.find_violations(tmp_path, ["tiny.pt"])
    assert violations == [("tiny.pt", "model weight or checkpoint (.pt)")]


def test_allowlisted_mel_filters_are_not_weights():
    path = "whisper/assets/mel_filters.npz"
    size = (ROOT / path).stat().st_size
    assert check_no_weights.classify(path, size) is None


def test_cli_exits_zero_on_this_tree():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_no_weights.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK:" in result.stdout


@pytest.mark.parametrize(
    "name",
    ["model.pth", "weights.safetensors", "export.onnx", "foo.gguf"],
)
def test_weight_suffixes_are_rejected(tmp_path, name):
    (tmp_path / name).write_bytes(b"x")
    violations = check_no_weights.find_violations(tmp_path, [name])
    assert len(violations) == 1
    assert violations[0][0] == name
