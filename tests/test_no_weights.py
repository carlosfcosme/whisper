import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_check_no_weights():
    path = ROOT / "scripts" / "check_no_weights.py"
    spec = importlib.util.spec_from_file_location("check_no_weights", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tracked_tree_has_no_committed_weights():
    check = _load_check_no_weights()
    assert check.find_violations(ROOT) == []
    assert check.main() == 0


@pytest.mark.parametrize("name", ["tiny.pt", "model.safetensors", "weights.bin"])
def test_planted_weight_is_a_violation(tmp_path, name):
    check = _load_check_no_weights()
    planted = tmp_path / name
    planted.write_bytes(b"not-a-real-checkpoint")
    reason = check.classify(name, planted.stat().st_size)
    assert reason is not None
    assert "weight" in reason or "checkpoint" in reason
    violations = check.find_violations(tmp_path, relative_paths=[name])
    assert violations == [(name, reason)]


def test_existing_assets_are_not_classified_as_weights():
    check = _load_check_no_weights()
    assert check.classify("whisper/assets/mel_filters.npz", 4271) is None
    assert check.classify("tests/jfk.flac", 1_152_693) is None
    assert check.classify("whisper/assets/gpt2.tiktoken", 835_554) is None
