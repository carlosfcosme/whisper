"""CI bind checker fails on all-interface tokens and accepts this tree."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _checker():
    path = ROOT / "scripts" / "check_bind_localhost.py"
    spec = importlib.util.spec_from_file_location("check_bind_localhost", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


bind_check = _checker()
UNSPECIFIED = ".".join(("0",) * 4)


def test_current_tree_has_no_all_interface_token():
    assert bind_check.find_unspecified_hits() == []


def test_planted_all_interface_token_is_detected(tmp_path):
    whisper = tmp_path / "whisper"
    whisper.mkdir()
    (whisper / "evil.py").write_text("host = %r\n" % UNSPECIFIED)
    hits = bind_check.find_unspecified_hits(tmp_path)
    assert any(hit.endswith("evil.py") for hit in hits)


def test_script_passes_on_this_tree():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_bind_localhost.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok:" in result.stdout
