import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_check_offline_default_passes():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_offline_default.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "offline" in proc.stdout.lower()


def test_check_no_wildcard_bind_passes():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_no_wildcard_bind.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
