import importlib.util
from pathlib import Path

import pytest

import whisper
from whisper.offline import WeightDownloadError, weight_auto_download_allowed

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_weight_auto_download_is_disabled_in_tests():
    assert weight_auto_download_allowed() is False


def test_named_checkpoint_cache_miss_does_not_pull(tmp_path):
    with pytest.raises(WeightDownloadError, match="Auto-download"):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_check_no_weights_passes_this_tree():
    check = _load("check_no_weights.py")
    assert check.find_violations(ROOT) == []
    assert check.main() == 0


@pytest.mark.parametrize("name", ["tiny.pt", "model.safetensors", "weights.bin"])
def test_planted_weight_is_a_violation(tmp_path, name):
    check = _load("check_no_weights.py")
    planted = tmp_path / name
    planted.write_bytes(b"not-a-real-checkpoint")
    reason = check.classify(name, planted.stat().st_size)
    assert reason is not None
    assert check.find_violations(tmp_path, relative_paths=[name]) == [(name, reason)]


def test_check_no_hub_and_cpu_offline_scripts_pass():
    assert _load("check_no_hub.py").main() == 0
    assert _load("check_cpu_offline.py").main() == 0
