import importlib.util
import subprocess
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
    (tmp_path / "whisper").mkdir()
    (tmp_path / "whisper" / "bad.py").write_text(
        "host = {}\n".format(repr(".".join(("0",) * 4)))
    )
    hits = checker.scan_app_sources(tmp_path)
    assert hits
    assert any(token == checker.FORBIDDEN_SUBSTRINGS[0] for _, token in hits)


def test_loopback_ci_flags_empty_host_bind(tmp_path):
    checker = _load_checker()
    (tmp_path / "whisper").mkdir()
    (tmp_path / "whisper" / "bad.py").write_text('sock.bind(("", 8080))\n')
    hits = checker.scan_app_sources(tmp_path)
    assert any(token == "empty-host bind()" for _, token in hits)


def test_check_no_all_interfaces_sh_passes_this_tree():
    script = ROOT / "scripts" / "check_no_all_interfaces.sh"
    result = subprocess.run(
        ["bash", str(script), str(ROOT)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_check_no_all_interfaces_sh_fails_on_planted_token(tmp_path):
    script = ROOT / "scripts" / "check_no_all_interfaces.sh"
    whisper = tmp_path / "whisper"
    whisper.mkdir()
    (whisper / "evil.py").write_text(
        "HTTPServer(({}, 80), handler)\n".format(repr(".".join(("0",) * 4)))
    )
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(
        ["git", "add", "whisper/evil.py"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        ["bash", str(script), str(tmp_path)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "all-interface" in result.stderr
