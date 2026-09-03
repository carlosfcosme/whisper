#!/usr/bin/env python3
"""Fail if the default install or CI path would fetch model weights.

Default install (``.cursor/install.sh``) and default CI (``whisper-test``)
must stay offline: no ``load_model``, no ``test_transcribe[tiny]`` selector,
no API keys, localhost only (no ``0.0.0.0``).

This script is the check that fails when fetch-of-weights is the default path.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"
INSTALL = REPO_ROOT / ".cursor" / "install.sh"
ENVIRONMENT = REPO_ROOT / ".cursor" / "environment.json"

FETCH_RE = re.compile(
    r"load_model|_download|openaipublic|azureedge\.net|huggingface\.co|"
    r"WHISPER_PRECACHE|pre-?cache",
    re.I,
)
SECRET_ASSIGN_RE = re.compile(
    r"(API_KEY|SECRET_KEY|ACCESS_TOKEN|HF_TOKEN|OPENAI_API_KEY|"
    r"AWS_SECRET|PRIVATE_KEY)\s*[:=]",
    re.I,
)
SECRET_LITERAL_RE = re.compile(r"\b(sk-|hf_|ghp_)[A-Za-z0-9_\-]{8,}")
WILDCARD_BIND = "0.0.0.0"


def strip_hash_comments(text: str) -> str:
    """Drop ``# ...`` comments so docs can mention forbidden words."""
    lines = []
    for line in text.splitlines():
        lines.append(line.split("#", 1)[0])
    return "\n".join(lines)


def job_block(workflow_text: str, job_name: str) -> str:
    header = re.search(rf"^  {re.escape(job_name)}:\s*$", workflow_text, flags=re.M)
    if header is None:
        raise ValueError(f"missing CI job {job_name!r} in {WORKFLOW}")
    rest = workflow_text[header.end() :]
    nxt = re.search(r"^  [A-Za-z0-9_-]+:\s*$", rest, flags=re.M)
    end = header.end() + nxt.start() if nxt is not None else None
    return workflow_text[header.start() : end]


def _pytest_lines(block: str) -> List[str]:
    return [ln.strip() for ln in block.splitlines() if re.search(r"\bpytest\b", ln)]


def reasons_default_path_fetches_weights(
    *,
    workflow_text: Optional[str] = None,
    install_text: Optional[str] = None,
    environment_text: Optional[str] = None,
) -> List[str]:
    """Return human-readable reasons the default path would fetch weights.

    Empty list means the default install/CI path is offline.
    """
    reasons: List[str] = []

    if workflow_text is None:
        workflow_text = WORKFLOW.read_text()
    if install_text is None:
        install_text = INSTALL.read_text()
    if environment_text is None:
        environment_text = ENVIRONMENT.read_text() if ENVIRONMENT.is_file() else "{}"

    try:
        block = job_block(workflow_text, "whisper-test")
    except ValueError as exc:
        reasons.append(str(exc))
        block = ""

    pytest_lines = _pytest_lines(block)
    if block and not pytest_lines:
        reasons.append("whisper-test job does not run pytest")
    for line in pytest_lines:
        if "test_transcribe[" in line:
            reasons.append(
                "CI pytest selects a weight-fetching transcribe case "
                f"(fetch-of-weights is the default path): {line}"
            )
        if "not test_transcribe" not in line and "not requires_weights" not in line:
            reasons.append(
                "whisper-test pytest does not exclude test_transcribe "
                f"(default path would fetch tiny/tiny.en weights): {line}"
            )

    install_code = strip_hash_comments(install_text)
    if FETCH_RE.search(install_code):
        reasons.append(
            "install.sh would fetch or precache model weights "
            "(load_model / CDN / Hub is not allowed on the default path)"
        )
    if WILDCARD_BIND in install_code:
        reasons.append("install.sh mentions 0.0.0.0 (must be localhost only)")
    reasons.extend(_secret_reasons("install.sh", install_code))

    env_code = strip_hash_comments(environment_text)
    if WILDCARD_BIND in env_code:
        reasons.append("environment.json mentions 0.0.0.0 (must be localhost only)")
    try:
        env = json.loads(environment_text)
    except json.JSONDecodeError as exc:
        reasons.append(f"environment.json is not valid JSON: {exc}")
    else:
        if "ports" in env:
            reasons.append(
                "environment.json publishes ports; localhost-only default has none"
            )
        if any(
            key.lower() in {"secret", "token", "api_key", "hf_token"}
            for key in env
            if isinstance(key, str)
        ):
            reasons.append("environment.json must not store keys")

    workflow_code = strip_hash_comments(workflow_text)
    reasons.extend(_secret_reasons("test.yml", workflow_code))
    if WILDCARD_BIND in workflow_code:
        reasons.append("CI workflow mentions 0.0.0.0 (must be localhost only)")

    return reasons


def _secret_reasons(label: str, text: str) -> Iterable[str]:
    if SECRET_ASSIGN_RE.search(text):
        yield f"{label} looks like it embeds a secret/key assignment"
    if SECRET_LITERAL_RE.search(text):
        yield f"{label} looks like it embeds a key literal"


def main() -> int:
    reasons = reasons_default_path_fetches_weights()
    if reasons:
        print("FAIL: fetch-of-weights is the default install/CI path:")
        for reason in reasons:
            print(f"  - {reason}")
        return 1
    print(
        "OK: default install/CI is offline (no weight download, localhost only, no keys)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
