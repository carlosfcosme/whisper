import importlib.util
import sys
from pathlib import Path


def _load_checker():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "check_no_weights.py"
    spec = importlib.util.spec_from_file_location("check_no_weights", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def test_repo_has_no_committed_weights():
    issues = checker.findings(checker.repo_root())
    assert issues == []
    assert checker.main() == 0


def test_mel_filters_npz_is_allowlisted():
    assert checker.classify("whisper/assets/mel_filters.npz", 4271) is None


def test_classify_rejects_checkpoint_suffixes():
    assert checker.classify("models/tiny.pt", 100) is not None
    assert checker.classify("weights/model.safetensors", 100) is not None
    assert checker.classify("libfoo.so", 100) is not None


def test_findings_flags_injected_weight(tmp_path):
    weight = tmp_path / "evil.pt"
    weight.write_bytes(b"not-a-real-checkpoint")
    issues = checker.findings(tmp_path, paths=["evil.pt"])
    assert issues
    assert "evil.pt" in issues[0]
