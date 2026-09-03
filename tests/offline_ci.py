#!/usr/bin/env python3
"""Checkout-only CI checks: no Hub APIs, no committed weights/caches.

Does not import whisper, torch, or huggingface_hub and does not open sockets.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from hub_guard import (  # noqa: E402
    FORBIDDEN_HUB_APIS,
    REPO_ROOT,
    forbidden_hub_api_hits,
)

GITIGNORE_PATTERNS = (".cache/", "cache/", "weights/", "*.pt", "*.pth")
LS_FILES_PATHSPECS = (
    ".cache",
    ".cache/**",
    "cache",
    "cache/**",
    "weights",
    "weights/**",
    "*.pt",
    "*.pth",
)
IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/tiny.pt",
    "weights/tiny.pt",
    "tiny.pt",
    "model.pth",
)
WILDCARD = ".".join(["0"] * 4)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def check_gitignore() -> None:
    lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    missing = [pattern for pattern in GITIGNORE_PATTERNS if pattern not in lines]
    if missing:
        raise SystemExit("gitignore missing weight/cache patterns: %s" % missing)


def check_untracked_weights() -> None:
    listed = _git("ls-files", "-z", "--", *LS_FILES_PATHSPECS)
    if listed.returncode != 0:
        raise SystemExit(listed.stderr or listed.stdout)
    tracked = [path for path in listed.stdout.split("\0") if path]
    if tracked:
        raise SystemExit("weight/cache paths must not be committed: %s" % tracked)


def check_gitignore_examples() -> None:
    failed = [
        path
        for path in IGNORE_EXAMPLES
        if _git("check-ignore", "-q", "--", path).returncode != 0
    ]
    if failed:
        raise SystemExit("expected these paths to be gitignored: %s" % failed)


def check_no_hub_apis_in_tests() -> None:
    hits = forbidden_hub_api_hits()
    if hits:
        raise SystemExit(
            "tests must not import/call %s; found in: %s"
            % (", ".join(FORBIDDEN_HUB_APIS), hits)
        )


def check_localhost_bind() -> None:
    start = REPO_ROOT / ".cursor" / "start.sh"
    if start.is_file():
        text = start.read_text(encoding="utf-8")
        if "127.0.0.1" not in text:
            raise SystemExit("start.sh must bind 127.0.0.1")
        if WILDCARD in text:
            raise SystemExit("start.sh must not bind %s" % WILDCARD)
    env_path = REPO_ROOT / ".cursor" / "environment.json"
    if env_path.is_file():
        env = json.loads(env_path.read_text(encoding="utf-8"))
        if "ports" in env:
            raise SystemExit("environment.json must not publish ports")


def check_ci_has_no_default_weight_pull() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    if 'WHISPER_ALLOW_WEIGHT_FETCH: "1"' in workflow:
        raise SystemExit(
            "CI must not enable WHISPER_ALLOW_WEIGHT_FETCH=1 (no weight pull)"
        )
    if "test_transcribe[tiny]" in workflow:
        raise SystemExit("CI must not run test_transcribe (offline fixtures only)")
    if 'WHISPER_DEVICE: "cpu"' not in workflow:
        raise SystemExit("CI must force WHISPER_DEVICE=cpu")
    env_sh = (REPO_ROOT / ".cursor" / "env.sh").read_text(encoding="utf-8")
    if "WHISPER_DEVICE=cpu" not in env_sh:
        raise SystemExit("env.sh must force WHISPER_DEVICE=cpu")


def main() -> int:
    check_gitignore()
    check_untracked_weights()
    check_gitignore_examples()
    check_no_hub_apis_in_tests()
    check_localhost_bind()
    check_ci_has_no_default_weight_pull()
    print(
        "offline-ci ok: no Hub APIs, no committed weights, localhost bind, no weight pull"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
