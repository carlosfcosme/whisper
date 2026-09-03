import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_no_download.py"


def _load_check():
    spec = importlib.util.spec_from_file_location("check_no_download", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_download_checker_subprocess_passes():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no-download" in proc.stdout


def test_no_download_checker_flags_urlopen_in_download(tmp_path, monkeypatch):
    check = _load_check()
    planted = tmp_path / "whisper"
    planted.mkdir()
    (planted / "__init__.py").write_text(
        "def _download(url, root, in_memory):\n"
        "    import urllib.request\n"
        "    return urllib.request.urlopen(url)\n"
    )
    monkeypatch.setattr(check, "ROOT", tmp_path)
    assert check.find_urlopen_in_download() == ["whisper/__init__.py:_download"]


def test_workflow_runs_no_download_checker():
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    assert "scripts/check_no_download.py" in workflow
