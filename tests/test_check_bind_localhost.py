import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_bind_localhost.py"
ALL_INTERFACES = ".".join(("0",) * 4)


def _load_check():
    spec = importlib.util.spec_from_file_location("check_bind_localhost", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_checker_subprocess_passes_on_this_tree():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "127.0.0.1" in proc.stdout


def test_checker_finds_unspecified_in_planted_tree(tmp_path, monkeypatch):
    check = _load_check()
    planted = tmp_path / "whisper"
    planted.mkdir()
    (planted / "bad.py").write_text("host = %r\n" % ALL_INTERFACES)
    monkeypatch.setattr(check, "ROOT", tmp_path)
    monkeypatch.setattr(check, "SCAN_DIRS", ("whisper",))
    hits = [Path(h).as_posix() for h in check.find_unspecified_hits()]
    assert "whisper/bad.py" in hits
