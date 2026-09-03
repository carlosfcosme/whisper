#!/usr/bin/env python3
"""Fail if test fixtures use HTTP(S) or Hugging Face URLs.

Standalone: no torch, no Hub clients, no secrets.
"""

from __future__ import print_function

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFTEST = os.path.join(REPO_ROOT, "tests", "conftest.py")
SAMPLE_AUDIO = os.path.join(REPO_ROOT, "tests", "jfk.flac")
WORKFLOW = os.path.join(REPO_ROOT, ".github", "workflows", "test.yml")

FORBIDDEN = ("http://", "https://", "huggingface", "hf.co")


def _fail(message):
    print(message, file=sys.stderr)
    return 1


def main():
    if not os.path.isfile(CONFTEST):
        return _fail("missing tests/conftest.py")
    conftest = open(CONFTEST, "r").read().lower()
    hits = [token for token in FORBIDDEN if token in conftest]
    if hits:
        return _fail(
            "tests/conftest.py must not declare remote fixture URLs: {0}".format(
                ", ".join(hits)
            )
        )

    if not os.path.isfile(SAMPLE_AUDIO):
        return _fail("missing in-repo sample audio tests/jfk.flac")

    import subprocess

    listed = subprocess.check_output(
        ["git", "ls-files", "--", "tests/jfk.flac"],
        cwd=REPO_ROOT,
        universal_newlines=True,
    ).strip()
    if listed != "tests/jfk.flac":
        return _fail("tests/jfk.flac must be tracked in git (got {0!r})".format(listed))

    workflow = open(WORKFLOW, "r").read()
    if "test_transcribe[tiny]" in workflow or "test_transcribe[tiny.en]" in workflow:
        return _fail(
            "CI must not select test_transcribe[tiny] / tiny.en (that pulls weights)"
        )
    if "not test_transcribe" not in workflow:
        return _fail("CI pytest command must exclude test_transcribe (no weight pull)")

    print("OK: fixtures are local; CI does not pull weights")
    return 0


if __name__ == "__main__":
    sys.exit(main())
