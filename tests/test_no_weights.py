import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    path = ROOT / "scripts" / "check_no_weights.py"
    spec = importlib.util.spec_from_file_location("check_no_weights", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repo_has_no_committed_weights():
    checker = _load_checker()
    assert checker.find_violations(ROOT) == []
    assert checker.main() == 0


def test_checker_flags_checkpoint_suffix():
    checker = _load_checker()
    assert checker.classify("models/tiny.pt", 100) is not None
    assert checker.classify("tests/jfk.flac", 1152693) is None
