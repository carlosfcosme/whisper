"""Coverage for the CI tracked-file guard (cache/weight dirs and blobs)."""

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_no_weights.py"
_SPEC = importlib.util.spec_from_file_location("check_no_weights", _SCRIPT)
_CHECK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHECK)


def test_classify_cache_and_weight_dirs():
    assert _CHECK.classify("weights/tiny.pt", 10) is not None
    assert _CHECK.classify("cache/whisper/tiny.en.pt", 10) is not None
    assert _CHECK.classify(".cache/whisper/tiny.pt", 10) is not None
    assert _CHECK.classify("checkpoints/model.pth", 10) is not None
    assert _CHECK.classify("whisper/__init__.py", 100) is None


def test_find_violations_flags_dir_blob(tmp_path):
    blob = tmp_path / "weights" / "tiny.pt"
    blob.parent.mkdir()
    blob.write_bytes(b"not-a-real-checkpoint")
    hits = _CHECK.find_violations(tmp_path, ["weights/tiny.pt"])
    assert len(hits) == 1
    assert hits[0][0] == "weights/tiny.pt"


def test_guard_passes_clean_tree():
    assert _CHECK.missing_gitignore_patterns(_CHECK.repo_root()) == []
    assert _CHECK.tracked_weight_paths(_CHECK.repo_root()) == []
    assert _CHECK.unignored_examples(_CHECK.repo_root()) == []
    assert _CHECK.main() == 0
