"""The check script must fail CI when cache or weight files are tracked."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_no_tracked_weights.sh"
BIND_CHECK = REPO_ROOT / "scripts" / "check_bind_localhost.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"

GOOD_GITIGNORE = ".cache/\ncache/\nweights/\n*.pt\n*.pth\n"


def _run(args, cwd):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True)


def _init_repo(tmp_path, gitignore_text):
    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / ".gitignore").write_text(gitignore_text)
    subprocess.run(
        ["git", "add", "--", ".gitignore"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_ci_invokes_weight_and_bind_checks():
    workflow = WORKFLOW.read_text()
    assert "scripts/check_no_tracked_weights.sh" in workflow
    assert "scripts/check_bind_localhost.py" in workflow
    assert "scripts/check_ci_no_weight_pull.py" in workflow
    assert "not test_transcribe" in workflow
    assert "test_transcribe[tiny]" not in workflow
    assert BIND_CHECK.is_file()
    assert SCRIPT.is_file()


def test_check_script_passes_on_this_repo():
    result = _run(["bash", str(SCRIPT)], REPO_ROOT)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_bind_check_passes_on_this_repo():
    result = _run(["python3", str(BIND_CHECK)], REPO_ROOT)
    assert result.returncode == 0, result.stderr + result.stdout
    assert "127.0.0.1" in result.stdout


def test_check_script_fails_when_pt_is_tracked(tmp_path):
    _init_repo(tmp_path, GOOD_GITIGNORE)
    (tmp_path / "tiny.pt").write_bytes(b"not-a-real-weight")
    subprocess.run(
        ["git", "add", "-f", "--", "tiny.pt"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    result = _run(["bash", str(SCRIPT)], tmp_path)
    assert result.returncode != 0
    assert "tiny.pt" in result.stderr
