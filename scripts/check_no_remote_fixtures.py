#!/usr/bin/env python3
"""CI: fail if test fixtures use HTTP(S) or Hugging Face asset URLs.

Standalone. No torch. No Hub clients. No secrets. No weight pull.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("WHISPER_FIXTURE_ROOT", Path(__file__).resolve().parents[1]))
CONFTEST = ROOT / "tests" / "conftest.py"
SAMPLE_AUDIO = ROOT / "tests" / "jfk.flac"
TINY_AUDIO = ROOT / "tests" / "tiny.wav"
FIXTURES_MOD = ROOT / "whisper" / "fixtures.py"
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"
TESTS_DIR = ROOT / "tests"

# conftest is the fixture source of truth and must stay URL-free.
CONFTEST_FORBIDDEN = ("http://", "https://", "huggingface", "hf.co")

# Tests that prove the guard by passing remote URLs into it.
SCAN_SKIP = frozenset(
    {
        "test_no_hub.py",
        "test_local_fixtures.py",
        "test_download_offline.py",
    }
)

ASSET_URL = re.compile(
    r"""['\"]https?://[^'\"]+\.(?:flac|wav|mp3|ogg|m4a|npz|tiktoken|pt|pth|safetensors)['\"]""",
    re.IGNORECASE,
)
HUB_ASSET = re.compile(
    r"""['\"]https?://(?:huggingface\.co|hf\.co)/[^'\"]+['\"]""",
    re.IGNORECASE,
)


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def tracked(relpath: str) -> str:
    output = subprocess.check_output(
        ["git", "ls-files", "--", relpath],
        cwd=str(ROOT),
        universal_newlines=True,
    ).strip()
    return output


def check_conftest() -> None:
    if not CONFTEST.is_file():
        fail("missing tests/conftest.py")
    text = CONFTEST.read_text(encoding="utf-8").lower()
    hits = [token for token in CONFTEST_FORBIDDEN if token in text]
    if hits:
        fail("tests/conftest.py must not declare remote fixture URLs: {}".format(hits))


def check_local_files() -> None:
    if not SAMPLE_AUDIO.is_file():
        fail("missing in-repo sample audio tests/jfk.flac")
    if tracked("tests/jfk.flac") != "tests/jfk.flac":
        fail("tests/jfk.flac must be tracked in git")
    if SAMPLE_AUDIO.stat().st_size <= 0:
        fail("tests/jfk.flac is empty")

    if not TINY_AUDIO.is_file():
        fail("missing tiny local fixture tests/tiny.wav")
    if tracked("tests/tiny.wav") != "tests/tiny.wav":
        fail("tests/tiny.wav must be tracked in git")
    if TINY_AUDIO.stat().st_size <= 0:
        fail("tests/tiny.wav is empty")

    if not FIXTURES_MOD.is_file():
        fail("missing whisper/fixtures.py")


def check_workflow() -> None:
    if not WORKFLOW.is_file():
        fail("missing .github/workflows/test.yml")
    text = WORKFLOW.read_text(encoding="utf-8")
    if "test_transcribe[tiny]" in text or "test_transcribe[tiny.en]" in text:
        fail("CI must not select test_transcribe[tiny] / tiny.en (that pulls weights)")
    if "not test_transcribe" not in text:
        fail("CI pytest command must exclude test_transcribe (no weight pull)")


def check_test_sources() -> None:
    hits = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        if path.name in SCAN_SKIP:
            continue
        text = path.read_text(encoding="utf-8")
        if ASSET_URL.search(text) or HUB_ASSET.search(text):
            hits.append(str(path.relative_to(ROOT)))
    if hits:
        fail("http(s)/huggingface asset URLs are not allowed in tests: {}".format(hits))


def main() -> int:
    check_conftest()
    check_local_files()
    check_workflow()
    check_test_sources()
    print("OK: fixtures are local; CI does not pull weights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
