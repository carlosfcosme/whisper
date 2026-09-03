import importlib.util
from pathlib import Path

import pytest

import whisper
from whisper.runtime import WeightDownloadError

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gitignore_covers_cache_and_weights():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for token in (".cache/", "cache/", "weights/", "*.pt"):
        assert token in text


def test_no_tracked_weight_files():
    checker = _load_script("check_no_weights.py")
    assert checker.main() == 0
    assert checker.find_tracked_weights(checker.tracked_files()) == []


def test_no_weight_cache_after_unit_tests():
    checker = _load_script("assert_no_weight_cache.py")
    assert checker.main() == 0


def test_load_model_does_not_pull_weights(tmp_path):
    with pytest.raises(WeightDownloadError):
        whisper.load_model("tiny", download_root=str(tmp_path))
    assert list(tmp_path.iterdir()) == []
