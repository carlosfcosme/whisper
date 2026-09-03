#!/usr/bin/env python3
"""CI: fail if the default inference device is not CPU.

Standalone. No torch. No Hub. No weight pull. No secrets.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INIT = ROOT / "whisper" / "__init__.py"
TRANSCRIBE = ROOT / "whisper" / "transcribe.py"
CUDA_AUTO = 'device = "cuda" if torch.cuda.is_available()'


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    if not INIT.is_file():
        fail("missing whisper/__init__.py")
    init_text = INIT.read_text(encoding="utf-8")
    if (
        'DEFAULT_DEVICE = "cpu"' not in init_text
        and "DEFAULT_DEVICE = 'cpu'" not in init_text
    ):
        fail("whisper.DEFAULT_DEVICE must be CPU")
    if CUDA_AUTO in init_text:
        fail("load_model must not auto-select CUDA")

    if not TRANSCRIBE.is_file():
        fail("missing whisper/transcribe.py")
    cli_text = TRANSCRIBE.read_text(encoding="utf-8")
    if "DEFAULT_DEVICE" not in cli_text:
        fail("CLI --device must default to DEFAULT_DEVICE")
    if CUDA_AUTO in cli_text:
        fail("CLI must not auto-select CUDA")

    print("cpu-default: ok (DEFAULT_DEVICE is cpu; no CUDA auto-select)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
