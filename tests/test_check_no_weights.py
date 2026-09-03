from pathlib import Path


def _load():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "check_no_weights.py"
    spec = importlib.util.spec_from_file_location("check_no_weights", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_classify_rejects_weight_suffixes():
    check = _load()
    assert check.classify("models/tiny.pt", 100) is not None
    assert check.classify("cache/model.safetensors", 100) is not None
    assert check.classify("weights/foo.bin", 10) is not None


def test_classify_allows_source_and_fixtures():
    check = _load()
    assert check.classify("whisper/__init__.py", 2000) is None
    assert check.classify("tests/jfk.flac", 400_000) is None
    assert check.classify("whisper/assets/mel_filters.npz", 200_000) is None


def test_repo_has_no_committed_weights():
    check = _load()
    assert check.find_violations(check.repo_root()) == []
    assert check.main() == 0


def test_find_violations_detects_temp_weight(tmp_path):
    check = _load()
    rel = "fake_tiny.pt"
    (tmp_path / rel).write_bytes(b"not-a-real-checkpoint")
    found = check.find_violations(tmp_path, [rel])
    assert found == [(rel, "model weight or checkpoint (.pt)")]
