"""CPU is the default inference path. No GPU/CUDA requirement."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INIT = os.path.join(ROOT, "whisper", "__init__.py")
TRANSCRIBE = os.path.join(ROOT, "whisper", "transcribe.py")
CHECK = os.path.join(ROOT, "scripts", "check_cpu_default.py")
CUDA_AUTO = 'device = "cuda" if torch.cuda.is_available()'


def test_default_device_literal_is_cpu():
    init_text = open(INIT, encoding="utf-8").read()
    assert (
        'DEFAULT_DEVICE = "cpu"' in init_text or "DEFAULT_DEVICE = 'cpu'" in init_text
    )
    assert CUDA_AUTO not in init_text


def test_cli_does_not_auto_select_cuda():
    cli_text = open(TRANSCRIBE, encoding="utf-8").read()
    assert "DEFAULT_DEVICE" in cli_text
    assert CUDA_AUTO not in cli_text


def test_check_cpu_default_script_passes():
    result = subprocess.run(
        [sys.executable, CHECK],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "cpu-default: ok" in result.stdout
