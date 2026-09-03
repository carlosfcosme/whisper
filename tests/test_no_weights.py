import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_no_weights", ROOT / "scripts" / "check_no_weights.py"
)
check_no_weights = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_no_weights)


def test_checker_passes_on_this_tree():
    assert check_no_weights.violations() == []
    assert check_no_weights.main() == 0


def test_checker_classifies_weight_suffixes():
    assert check_no_weights.is_committed_weight("models/tiny.pt")
    assert check_no_weights.is_committed_weight("foo.safetensors")
    assert check_no_weights.is_committed_weight("weights/model.bin")
    assert check_no_weights.is_committed_weight("cache/whisper/tiny.pt")
    assert not check_no_weights.is_committed_weight("whisper/assets/mel_filters.npz")
    assert not check_no_weights.is_committed_weight("tests/jfk.flac")
    assert not check_no_weights.is_committed_weight("whisper/transcribe.py")


def test_checker_rejects_large_tracked_blob(tmp_path):
    blob = tmp_path / "big.dat"
    blob.write_bytes(b"x" * (check_no_weights.MAX_BYTES + 1))
    assert check_no_weights.is_committed_weight(str(blob), blob.stat().st_size)


def test_planted_pt_is_a_violation(tmp_path, monkeypatch):
    planted = tmp_path / "leaked.pt"
    planted.write_bytes(b"not-a-real-checkpoint")
    monkeypatch.chdir(tmp_path)
    assert check_no_weights.violations([str(planted)])


def test_no_pt_tracked_in_repo():
    tracked = check_no_weights.tracked_files()
    pts = [p for p in tracked if p.endswith(".pt") or p.endswith(".safetensors")]
    assert pts == []
    assert not os.path.isdir(os.path.expanduser("~/.cache/whisper"))
