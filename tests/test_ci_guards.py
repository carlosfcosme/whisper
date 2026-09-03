import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_loopback_bind_guard_passes_on_this_repo():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_loopback_bind.py")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "127.0.0.1" in result.stdout


def test_no_download_guard_passes_on_this_repo():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_no_download.py")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "blocked" in result.stdout


def test_loopback_bind_scan_flags_all_interfaces(tmp_path):
    check = _load("check_loopback_bind.py")
    app = tmp_path / "whisper"
    app.mkdir()
    (app / "bad.py").write_text(
        "ThreadingHTTPServer(({}, 80), None)\n".format(check.ALL_INTERFACES)
    )
    hits = check.scan_app_sources(tmp_path)
    assert hits


def test_loopback_bind_policy_rejects_all_interfaces():
    check = _load("check_loopback_bind.py")
    serve = check.load_serve(REPO_ROOT)
    errors = check.check_policy(serve)
    assert errors == []
    with pytest.raises(serve.BindError):
        serve.normalize_bind_host(check.ALL_INTERFACES)


def test_no_download_scan_flags_hub_import(tmp_path):
    check = _load("check_no_download.py")
    whisper = tmp_path / "whisper"
    whisper.mkdir()
    (whisper / "bad.py").write_text("import huggingface_hub\n")
    hits = check.scan_hub_imports(tmp_path)
    assert hits
    assert hits[0][1] == "huggingface_hub"


def test_workflow_lint_test_job_is_offline_and_loopback():
    text = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text()
    assert "lint-test:" in text
    assert 'WHISPER_OFFLINE: "1"' in text or "WHISPER_OFFLINE: '1'" in text
    assert "HF_HUB_OFFLINE" in text
    assert "scripts/ci_lint_test.sh" in text
    assert "check_no_weights.py" in text
    assert "check_loopback_bind.py" in text
    assert "needs: [pre-commit, lint-test]" in text
    assert "-k 'not test_transcribe'" in text
    assert "test_transcribe[tiny]" not in text


def test_ci_lint_test_script_exists_and_is_offline():
    script = (REPO_ROOT / "scripts" / "ci_lint_test.sh").read_text()
    assert "black --check" in script
    assert "check_loopback_bind.py" in script
    assert "check_no_download.py" in script
    assert "WHISPER_OFFLINE" in script
    assert "127.0.0.1" in script
