"""CI must stay Hub-offline and must not pull weights."""

import importlib.util
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GITIGNORE = REPO_ROOT / ".gitignore"

HUB_CACHE_IGNORE = (
    ".huggingface/",
    "hf_cache/",
)
HUB_CACHE_EXAMPLES = (
    ".huggingface/hub/models--openai--whisper-tiny/blobs/x",
    "hf_cache/whisper-tiny.pt",
)


def _load_checker():
    path = REPO_ROOT / "scripts" / "check_no_hub.py"
    spec = importlib.util.spec_from_file_location("check_no_hub", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_check_no_hub_passes_this_repo():
    checker = _load_checker()
    assert checker.find_violations(REPO_ROOT) == []
    assert checker.main() == 0


def test_check_no_hub_fails_when_hub_offline_dropped(tmp_path):
    checker = _load_checker()
    workflow = tmp_path / ".github" / "workflows"
    workflow.mkdir(parents=True)
    (workflow / "test.yml").write_text(
        "jobs:\n  whisper-test:\n    env:\n      WHISPER_OFFLINE: '1'\n"
        "    steps:\n      - run: pytest -k 'not test_transcribe'\n"
    )
    (tmp_path / "whisper").mkdir()
    (tmp_path / "whisper" / "__init__.py").write_text("x = 1\n")
    errors = checker.find_violations(tmp_path)
    assert any("HF_HUB_OFFLINE" in message for message in errors)
    assert checker.main(tmp_path) == 1


def test_check_no_hub_fails_when_package_imports_hub(tmp_path):
    checker = _load_checker()
    workflow = tmp_path / ".github" / "workflows"
    workflow.mkdir(parents=True)
    (workflow / "test.yml").write_text(
        (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text()
    )
    (tmp_path / "whisper").mkdir()
    (tmp_path / "whisper" / "evil.py").write_text("import huggingface_hub\n")
    errors = checker.find_violations(tmp_path)
    assert any("huggingface_hub" in message for message in errors)


def test_check_no_hub_fails_when_ci_runs_tiny_transcribe(tmp_path):
    checker = _load_checker()
    workflow = tmp_path / ".github" / "workflows"
    workflow.mkdir(parents=True)
    text = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text()
    text = text.replace(
        "-k 'not test_transcribe'",
        "-k 'not test_transcribe or test_transcribe[tiny]'",
    )
    (workflow / "test.yml").write_text(text)
    (tmp_path / "whisper").mkdir()
    (tmp_path / "whisper" / "__init__.py").write_text("x = 1\n")
    errors = checker.find_violations(tmp_path)
    assert any("test_transcribe[tiny]" in message for message in errors)


def test_gitignore_covers_hub_cache_and_weights():
    text = GITIGNORE.read_text(encoding="utf-8")
    for pattern in HUB_CACHE_IGNORE + (".cache/", "cache/", "weights/", "*.pt"):
        assert pattern in text
    for example in HUB_CACHE_EXAMPLES + (".cache/whisper/tiny.pt", "weights/tiny.pt"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--", example],
            cwd=REPO_ROOT,
            check=False,
        )
        assert result.returncode == 0, "expected gitignore to match {}".format(example)


def test_script_is_invoked_from_ci_and_precommit():
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text()
    precommit = (REPO_ROOT / ".pre-commit-config.yaml").read_text()
    assert "scripts/check_no_hub.py" in workflow
    assert "no-hub" in workflow
    assert "check_no_hub.py" in precommit
