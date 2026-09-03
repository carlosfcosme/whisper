"""Assert the default path leaves weight download unused."""

from pathlib import Path

import whisper
from whisper.offline import (
    assert_download_unused,
    network_download_calls,
    reset_download_usage,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "assert_download_unused.py"
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"


def test_assert_download_unused_passes_when_counter_is_zero():
    reset_download_usage()
    assert network_download_calls() == 0
    assert_download_unused("fresh counter")


def test_assert_download_unused_fails_when_counter_is_nonzero():
    import whisper.offline as offline

    reset_download_usage()
    offline._network_download_calls = 1
    try:
        try:
            assert_download_unused("forced")
        except AssertionError as exc:
            assert "weight download used" in str(exc)
        else:
            raise AssertionError("expected assert_download_unused to fail")
    finally:
        reset_download_usage()


def test_whisper_exports_assert_download_unused():
    reset_download_usage()
    whisper.assert_download_unused("exported")


def test_install_is_offline_and_does_not_download():
    text = (ROOT / ".cursor" / "install.sh").read_text()
    assert "WHISPER_OFFLINE" in text
    assert "HF_HUB_OFFLINE" in text
    assert "load_model" not in text
    assert "huggingface.co" not in text
    assert "HF_TOKEN" not in text
    assert "assert_download_unused.py --probe" in text


def test_workflow_is_offline_and_does_not_select_tiny():
    text = WORKFLOW.read_text()
    assert "WHISPER_OFFLINE" in text
    assert "HF_HUB_OFFLINE" in text
    assert "test_transcribe[tiny]" not in text
    assert "test_transcribe[tiny.en]" not in text
    assert "HF_TOKEN" not in text
    assert "HUGGING_FACE_HUB_TOKEN" not in text


def test_assert_script_passes_on_this_tree():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--check-workflow"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "OK" in proc.stdout
