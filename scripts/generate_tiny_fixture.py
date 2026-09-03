#!/usr/bin/env python3
"""Write tests/fixtures/tone.wav locally. No WAN. No weights. No keys."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    path = ROOT / "whisper" / "fixtures.py"
    spec = importlib.util.spec_from_file_location("whisper_fixtures_ci", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    written = module.write_tiny_wav()
    print(f"wrote {written} ({written.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
