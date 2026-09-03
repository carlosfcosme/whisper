import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    path = ROOT / "scripts" / "check_loopback_bind.py"
    spec = importlib.util.spec_from_file_location("check_loopback_bind", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loopback_ci_script_passes():
    checker = _load_checker()
    assert checker.scan_app_sources(ROOT) == []
    assert checker.check_policy(ROOT) == []
    assert checker.main() == 0


def test_loopback_ci_flags_all_interfaces_literal(tmp_path):
    checker = _load_checker()
    fake_root = tmp_path
    (fake_root / "whisper").mkdir()
    (fake_root / "whisper" / "bad.py").write_text(
        "host = {}\n".format(repr(".".join(("0",) * 4)))
    )
    hits = checker.scan_app_sources(fake_root)
    assert hits
    assert any(token == checker.FORBIDDEN_SUBSTRINGS[0] for _, token in hits)
