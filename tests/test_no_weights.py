import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "check_no_weights", REPO_ROOT / "scripts" / "check_no_weights.py"
)
check_no_weights = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(check_no_weights)


def test_check_no_weights_exits_clean():
    assert check_no_weights.main([]) == 0
    assert check_no_weights.violations() == []


def test_weight_suffixes_are_rejected():
    assert check_no_weights.is_committed_weight("tiny.pt")
    assert check_no_weights.is_committed_weight("weights/model.safetensors")
    assert check_no_weights.is_committed_weight(".cache/whisper/tiny.pt")
    assert not check_no_weights.is_committed_weight("whisper/assets/mel_filters.npz")
    assert not check_no_weights.is_committed_weight("tests/jfk.flac")
    assert (REPO_ROOT / "whisper/assets/mel_filters.npz").is_file()
