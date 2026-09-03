"""Package sources must not ship a Hub client or Hub credentials."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _checker():
    path = ROOT / "scripts" / "check_no_hub.py"
    spec = importlib.util.spec_from_file_location("check_no_hub", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


hub_check = _checker()


def test_package_has_no_hub_client():
    assert hub_check.reasons_hub_in_tree() == []


def test_script_passes_on_this_tree():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_no_hub.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK:" in result.stdout


def test_pyproject_has_no_hub_dependency():
    text = (ROOT / "pyproject.toml").read_text()
    assert "huggingface" not in text.lower()
    assert "transformers" not in text.lower()
