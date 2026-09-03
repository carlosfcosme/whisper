"""Coverage that bind and no-weight-pull CI scripts are wired and pass."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"
BIND_SCRIPT = REPO_ROOT / "scripts" / "check_bind_localhost.py"
NO_PULL_SCRIPT = REPO_ROOT / "scripts" / "check_ci_no_weight_pull.py"


def _run(script):
    return subprocess.run(
        ["python3", str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_bind_and_no_pull_scripts_are_wired():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "scripts/check_bind_localhost.py" in workflow
    assert "scripts/check_ci_no_weight_pull.py" in workflow
    precommit = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "scripts/check_bind_localhost.py" in precommit
    assert "scripts/check_ci_no_weight_pull.py" in precommit
    for script in (BIND_SCRIPT, NO_PULL_SCRIPT):
        text = script.read_text(encoding="utf-8")
        assert "sk-" not in text
        assert "FIELD_BRAIN" not in text.upper()
        assert "Field-Brain" not in text


def test_bind_localhost_script_passes():
    result = _run(BIND_SCRIPT)
    assert result.returncode == 0, result.stderr
    assert "127.0.0.1" in result.stdout


def test_ci_no_weight_pull_script_passes():
    result = _run(NO_PULL_SCRIPT)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
