import importlib.util
import subprocess
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_no_weights.py"
_SPEC = importlib.util.spec_from_file_location("check_no_weights", _SCRIPT)
_CHECK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHECK)


def test_classify_weight_suffixes():
    assert _CHECK.classify("tiny.pt", 100) is not None
    assert _CHECK.classify("model.safetensors", 100) is not None
    assert _CHECK.classify("weights.bin", 100) is not None
    assert _CHECK.classify("whisper/__init__.py", 1000) is None
    assert _CHECK.classify("tests/jfk.flac", 1_152_693) is None


def test_find_violations_flags_pt(tmp_path):
    weight = tmp_path / "tiny.pt"
    weight.write_bytes(b"not-a-real-checkpoint")
    hits = _CHECK.find_violations(tmp_path, ["tiny.pt"])
    assert hits == [("tiny.pt", "model weight or checkpoint (.pt)")]


def test_repo_has_no_committed_weights():
    root = _CHECK.repo_root()
    assert _CHECK.find_violations(root) == []
    assert _CHECK.main() == 0


def test_check_script_exits_zero_on_clean_tree():
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "no model weights committed" in proc.stdout
