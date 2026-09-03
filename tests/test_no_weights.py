import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_checker():
    path = REPO / "scripts" / "check_no_weights.py"
    spec = importlib.util.spec_from_file_location("check_no_weights", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def test_current_tree_has_no_committed_weights():
    assert checker.find_violations(REPO) == []
    assert checker.main() == 0


def test_planted_checkpoint_is_a_violation(tmp_path):
    planted = tmp_path / "tiny.pt"
    planted.write_bytes(b"not-a-real-checkpoint")
    violations = checker.find_violations(tmp_path, ["tiny.pt"])
    assert violations == [("tiny.pt", "model weight or checkpoint (.pt)")]


def test_classify_weight_suffixes():
    assert checker.classify("models/tiny.pth", 10) is not None
    assert checker.classify("models/model.safetensors", 10) is not None
    assert checker.classify("models/weights.bin", 10) is not None
    assert checker.classify("whisper/transcribe.py", 10) is None
    assert checker.classify("whisper/assets/mel_filters.npz", 4271) is None


def test_check_script_exits_zero_on_clean_repo():
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check_no_weights.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "OK:" in result.stdout


def test_repo_root_matches_checkout():
    assert checker.repo_root() == REPO
