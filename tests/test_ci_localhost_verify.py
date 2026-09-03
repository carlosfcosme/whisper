"""CI wiring for localhost-only verify. No whisper import, no weights, no WAN."""

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.localhost_only

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"
VERIFY = REPO_ROOT / ".cursor" / "verify.sh"
RUNNER = REPO_ROOT / "tests" / "run_localhost_only.sh"


def _job_block(workflow_text: str, job_name: str) -> str:
    header = re.search(rf"^  {re.escape(job_name)}:\s*$", workflow_text, flags=re.M)
    assert header is not None, f"missing CI job {job_name!r} in {WORKFLOW}"
    rest = workflow_text[header.end() :]
    nxt = re.search(r"^  [A-Za-z0-9_-]+:\s*$", rest, flags=re.M)
    end = header.end() + nxt.start() if nxt is not None else None
    return workflow_text[header.start() : end]


def test_verify_script_is_ci_safe():
    text = VERIFY.read_text()
    assert VERIFY.is_file()
    assert "WHISPER_LOCALHOST_ONLY=1" in text
    assert "tests/run_localhost_only.sh" in text
    assert "load_model" not in text
    assert "test_transcribe" not in text


def test_localhost_only_runner_is_executable():
    assert RUNNER.is_file()
    assert (
        RUNNER.stat().st_mode & 0o111
    ), "tests/run_localhost_only.sh must be executable"
    text = RUNNER.read_text()
    assert "WHISPER_LOCALHOST_ONLY=1" in text
    assert "-m 'localhost_only and not requires_cuda'" in text
    assert "load_model" not in text
    assert "test_transcribe" not in text


def test_generated_artifacts_are_gitignored():
    ignored = (
        ".pytest_cache/v/cache/nodeids",
        ".coverage",
        "htmlcov/index.html",
        "coverage.xml",
        "junit.xml",
        "test-results/out.xml",
        "artifacts/run.bin",
        "whisper.log",
        "tests/out.txt",
        "tests/out.vtt",
        "tests/out.srt",
        "tests/out.tsv",
        "tests/out.json",
        "tiny.pt",
        "model.pth",
        "weights.onnx",
        "weights.h5",
    )
    kept = (
        "tests/run_localhost_only.sh",
        "tests/conftest.py",
        ".cursor/verify.sh",
        "whisper/localhost.py",
    )
    for rel in ignored:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "--", rel],
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 0, f"{rel} must be gitignored"
    for rel in kept:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "--", rel],
            cwd=str(REPO_ROOT),
        )
        assert proc.returncode == 1, f"{rel} must remain tracked"


def test_ci_runs_localhost_only_verify_without_weight_download():
    text = WORKFLOW.read_text()
    block = _job_block(text, "localhost-only-verify")
    assert "bash .cursor/verify.sh" in block
    assert "WHISPER_LOCALHOST_ONLY" in block
    assert "test_transcribe" not in block
    assert "load_model" not in block
    assert "openaipublic" not in block
