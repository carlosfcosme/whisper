import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    path = REPO_ROOT / "scripts" / "check_no_remote_fixtures.py"
    spec = importlib.util.spec_from_file_location("check_no_remote_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def test_current_tests_have_no_remote_fixture_urls():
    assert checker.find_violations(REPO_ROOT) == []
    assert checker.main() == 0


def test_guard_flags_remote_audio_literal():
    line = "audio_path = " + '"http' + 's://example.invalid/sample.wav"'
    reasons = checker.classify_line("tests/conftest.py", line)
    assert reasons


def test_guard_flags_hub_asset_literal():
    line = (
        "load_audio("
        + '"http'
        + "s://"
        + "huggingface"
        + ".co"
        + '/datasets/x/a.flac")'
    )
    reasons = checker.classify_line("tests/test_audio.py", line)
    assert reasons


def test_guard_allows_local_fixture_assignment():
    reasons = checker.classify_line(
        "tests/conftest.py",
        'audio_path = "/workspace/tests/fixtures/tiny.wav"',
    )
    assert reasons == []


def test_ci_workflow_runs_the_fixture_guard():
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    assert "no-remote-fixtures" in workflow
    assert "scripts/check_no_remote_fixtures.py" in workflow
    assert "not test_transcribe" in workflow
