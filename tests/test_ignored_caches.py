import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    path = ROOT / "scripts" / "check_ignored_caches.py"
    spec = importlib.util.spec_from_file_location("check_ignored_caches", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weight_and_cache_paths_are_ignored():
    checker = _load_checker()
    assert checker.find_violations(ROOT) == []
    assert checker.main() == 0


def test_example_checkpoint_paths_are_ignored():
    checker = _load_checker()
    assert checker.is_ignored(ROOT, ".cache/whisper/tiny.pt")
    assert checker.is_ignored(ROOT, "weights/base.pt")
    assert checker.is_ignored(ROOT, "tiny.pt")
    assert not checker.is_ignored(ROOT, "tests/jfk.flac")


def test_offline_ci_job_covers_downloads_bind_and_ignores():
    yml = (ROOT / ".github/workflows/test.yml").read_text(encoding="utf-8")
    assert "offline-ci:" in yml
    assert "check_no_weights.py" in yml
    assert "check_loopback_listen.py" in yml
    assert "check_ignored_caches.py" in yml
    assert "-k 'not test_transcribe'" in yml
