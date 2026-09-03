"""CPU is the default inference device. Does not load weights."""

import os
import subprocess
import sys
from pathlib import Path

import whisper
from whisper.device import DEFAULT_DEVICE, default_device
from whisper.transcribe import cli

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_device_is_cpu():
    assert DEFAULT_DEVICE == "cpu"
    assert default_device() == "cpu"
    assert whisper.default_device() == "cpu"


def test_load_model_uses_cpu_when_device_is_omitted():
    source = (REPO_ROOT / "whisper" / "__init__.py").read_text()
    assert "device = default_device()" in source
    assert 'device = "cuda" if torch.cuda.is_available() else "cpu"' not in source


def test_cli_device_default_is_cpu(capsys):
    argv = sys.argv
    try:
        sys.argv = ["whisper", "--help"]
        try:
            cli()
        except SystemExit as exc:
            assert exc.code == 0
    finally:
        sys.argv = argv
    help_text = capsys.readouterr().out
    assert "--device" in help_text
    assert "default: cpu" in help_text


def test_cli_help_does_not_download_weights(tmp_path):
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "whisper", "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "default: cpu" in result.stdout
    weights = list(tmp_path.rglob("*.pt")) + list(tmp_path.rglob("*.pth"))
    assert weights == [], weights
