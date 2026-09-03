"""Fail if Hub is contacted or weights are pulled."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INIT = os.path.join(ROOT, "whisper", "__init__.py")
WORKFLOW = os.path.join(ROOT, ".github", "workflows", "test.yml")


def test_download_source_refuses_hub_and_offline_pull():
    text = open(INIT, encoding="utf-8").read()
    assert "is_hub_url" in text
    assert "offline_requested" in text
    assert "refusing Hugging Face Hub download" in text
    assert "WHISPER_OFFLINE=1: refusing weight download" in text


def test_ci_workflow_stays_offline():
    text = open(WORKFLOW, encoding="utf-8").read()
    assert "WHISPER_OFFLINE" in text
    assert "HF_HUB_OFFLINE" in text
    assert "not test_transcribe" in text
    assert "test_transcribe[tiny]" not in text


def test_check_scripts_refuse_hub_and_weights():
    for script in (
        "scripts/check_no_remote_fixtures.py",
        "scripts/check_no_weights.py",
        "scripts/check_cpu_default.py",
        "scripts/check_bind_localhost.py",
    ):
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, script)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (script, result.stderr)
