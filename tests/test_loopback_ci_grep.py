"""Ticket 1: bind 127.0.0.1 only; CI/test fails if the host is not loopback."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

from whisper.serve import BindError, create_server, serve, serve_bind_host
from whisper.sovereign import ALL_INTERFACES, BIND_HOST

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_check_loopback_bind():
    path = REPO_ROOT / "scripts" / "check_loopback_bind.py"
    spec = importlib.util.spec_from_file_location("check_loopback_bind", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


check_loopback_bind = _load_check_loopback_bind()


def test_bind_default_is_loopback():
    assert BIND_HOST == "127.0.0.1"
    assert serve_bind_host() == "127.0.0.1"


def test_bind_guard_fails_when_host_is_not_loopback():
    with pytest.raises(BindError, match="127.0.0.1"):
        serve_bind_host(ALL_INTERFACES)
    with pytest.raises(BindError, match="127.0.0.1"):
        create_server(host=ALL_INTERFACES, port=0)
    with pytest.raises(BindError, match="127.0.0.1"):
        serve(host=ALL_INTERFACES, port=0)


def test_git_grep_bind_paths_have_no_all_interfaces():
    hits = check_loopback_bind.grep_all_interfaces(REPO_ROOT)
    assert hits == []
    check_loopback_bind.assert_bind_paths_loopback_only(REPO_ROOT)


def test_ci_workflow_greps_0_0_0_0():
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    assert "loopback-bind" in workflow
    assert "git grep -nF '0.0.0.0'" in workflow
    assert "git grep -nF '0.0.0.0' -- .cursor" in workflow
    assert "scripts/check_loopback_bind.py" in workflow


def test_ci_git_grep_command_exits_nonzero_on_clean_tree():
    """git grep returns 1 when 0.0.0.0 is absent from .cursor/ (the pass case)."""
    proc = subprocess.run(
        ["git", "grep", "-nF", ALL_INTERFACES, "--", ".cursor"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert ALL_INTERFACES not in proc.stdout


def test_git_grep_fails_when_all_interfaces_planted(tmp_path):
    subprocess.check_call(["git", "init"], cwd=tmp_path)
    cursor = tmp_path / ".cursor"
    cursor.mkdir()
    planted = cursor / "start.sh"
    planted.write_text("python3 -m http.server --bind {}\n".format(ALL_INTERFACES))
    subprocess.check_call(["git", "add", "-f", ".cursor/start.sh"], cwd=tmp_path)
    hits = check_loopback_bind.grep_all_interfaces(tmp_path)
    assert hits, "planted 0.0.0.0 must be visible to git grep"
    assert any(ALL_INTERFACES in line for line in hits)
    with pytest.raises(AssertionError, match="127.0.0.1"):
        check_loopback_bind.assert_bind_paths_loopback_only(tmp_path)
