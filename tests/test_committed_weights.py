"""CI must fail when model weights are committed."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"
CHECK = REPO_ROOT / "scripts" / "check_no_committed_weights.sh"
WEIGHT_SUFFIXES = (".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".onnx")


def _job_block(workflow_text: str, job_name: str) -> str:
    header = re.search(rf"^  {re.escape(job_name)}:\s*$", workflow_text, flags=re.M)
    assert header is not None, f"missing CI job {job_name!r} in {WORKFLOW}"
    rest = workflow_text[header.end() :]
    nxt = re.search(r"^  [A-Za-z0-9_-]+:\s*$", rest, flags=re.M)
    end = header.end() + nxt.start() if nxt is not None else None
    return workflow_text[header.start() : end]


def test_repo_has_no_committed_weight_files():
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=REPO_ROOT, text=True
    ).splitlines()
    hits = [path for path in tracked if path.endswith(WEIGHT_SUFFIXES)]
    assert hits == [], f"committed weight files are not allowed: {hits}"


def test_check_script_passes_on_this_repo():
    subprocess.run(["bash", str(CHECK)], cwd=REPO_ROOT, check=True)


def test_ci_job_fails_on_committed_weights():
    text = WORKFLOW.read_text()
    block = _job_block(text, "no-committed-weights")
    assert "check_no_committed_weights.sh" in block
    assert "load_model" not in block
    assert "huggingface" not in block.lower()
