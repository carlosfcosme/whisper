"""Guards that fail if bind is not 127.0.0.1, Hub is used, or weights are pulled."""

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "assert_sovereign.py"
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"


def _load_guard():
    spec = importlib.util.spec_from_file_location("assert_sovereign", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assert_sovereign_passes_on_this_tree():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--all"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout


def test_assert_sovereign_fails_if_workflow_selects_tiny(tmp_path):
    poisoned = tmp_path / "test.yml"
    poisoned.write_text(
        WORKFLOW.read_text().replace(
            "-k 'not test_transcribe'",
            "-k 'not test_transcribe or test_transcribe[tiny]'",
        )
    )
    guard = _load_guard()
    original = guard.WORKFLOW
    guard.WORKFLOW = poisoned
    try:
        try:
            guard.check_workflow()
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise AssertionError("workflow check must fail when tiny is selected")
    finally:
        guard.WORKFLOW = original


def test_repo_has_no_compose_spark_or_live_flags():
    text = WORKFLOW.read_text().lower()
    for token in ("docker-compose", "spark.yml", "spark.yaml", "--live"):
        assert token not in text
    assert not list(ROOT.glob("docker-compose*.yml"))
    assert not list(ROOT.glob("compose.y*ml"))
    assert not list(ROOT.glob("**/spark*.yml"))
    assert not list(ROOT.glob("**/spark*.yaml"))
