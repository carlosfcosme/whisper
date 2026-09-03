"""Guard: default install/CI must not fetch model weights.

These tests fail if fetch-of-weights is the default path (the historical
CI selector that included tiny / tiny.en, or load_model during install).
No whisper import. No WAN. No keys.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _checker():
    path = ROOT / "scripts" / "check_default_offline.py"
    spec = importlib.util.spec_from_file_location("check_default_offline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


offline_check = _checker()

pytestmark = pytest.mark.localhost_only


def test_current_default_path_is_offline():
    reasons = offline_check.reasons_default_path_fetches_weights()
    assert reasons == [], reasons


def test_check_fails_when_ci_selects_tiny_weights():
    poisoned = """
jobs:
  whisper-test:
    runs-on: ubuntu-latest
    steps:
      - run: pytest --durations=0 -vv -k 'not test_transcribe or test_transcribe[tiny] or test_transcribe[tiny.en]' -m 'not requires_cuda'
"""
    reasons = offline_check.reasons_default_path_fetches_weights(
        workflow_text=poisoned,
        install_text="pip install -e '.[dev]'\n",
        environment_text=(
            '{"name": "openai-whisper", "install": "bash .cursor/install.sh"}'
        ),
    )
    assert reasons, "checker must fail when CI fetches tiny/tiny.en weights"
    assert any("test_transcribe[" in r for r in reasons)


def test_check_fails_when_pytest_does_not_exclude_transcribe():
    poisoned = """
jobs:
  whisper-test:
    steps:
      - run: pytest --durations=0 -vv
"""
    reasons = offline_check.reasons_default_path_fetches_weights(
        workflow_text=poisoned,
        install_text="pip install -e '.[dev]'\n",
        environment_text="{}",
    )
    assert any("does not exclude test_transcribe" in r for r in reasons)


def test_check_fails_when_install_calls_load_model():
    reasons = offline_check.reasons_default_path_fetches_weights(
        workflow_text="""
jobs:
  whisper-test:
    steps:
      - run: pytest -k 'not test_transcribe'
""",
        install_text="python -c 'import whisper; whisper.load_model(\"tiny\")'\n",
        environment_text="{}",
    )
    assert any("install.sh would fetch" in r for r in reasons)


def test_check_fails_when_ci_calls_load_model():
    reasons = offline_check.reasons_default_path_fetches_weights(
        workflow_text="""
jobs:
  whisper-test:
    steps:
      - run: pytest -k 'not test_transcribe'
      - run: python -c 'import whisper; whisper.load_model("tiny")'
""",
        install_text="pip install -e '.[dev]'\n",
        environment_text="{}",
    )
    assert any("CI workflow would fetch" in r for r in reasons)


def test_check_fails_when_ci_hits_cdn():
    reasons = offline_check.reasons_default_path_fetches_weights(
        workflow_text="""
jobs:
  whisper-test:
    steps:
      - run: pytest -k 'not test_transcribe'
      - run: curl -O https://openaipublic.azureedge.net/main/whisper/models/x/tiny.pt
""",
        install_text="pip install -e '.[dev]'\n",
        environment_text="{}",
    )
    assert any("CI workflow would fetch" in r for r in reasons)


def test_workflow_comment_may_mention_load_model():
    reasons = offline_check.reasons_default_path_fetches_weights(
        workflow_text="""
jobs:
  whisper-test:
    steps:
      # do not call load_model or hit azureedge.net
      - run: pytest -k 'not test_transcribe'
""",
        install_text="pip install -e '.[dev]'\n",
        environment_text="{}",
    )
    assert reasons == [], reasons


def test_check_fails_when_install_hits_cdn():
    reasons = offline_check.reasons_default_path_fetches_weights(
        workflow_text="""
jobs:
  whisper-test:
    steps:
      - run: pytest -k 'not test_transcribe'
""",
        install_text=(
            "curl -O https://openaipublic.azureedge.net/main/whisper/models/x/tiny.pt\n"
        ),
        environment_text="{}",
    )
    assert any("install.sh would fetch" in r for r in reasons)


def test_check_fails_on_wildcard_bind():
    reasons = offline_check.reasons_default_path_fetches_weights(
        workflow_text="""
jobs:
  whisper-test:
    steps:
      - run: pytest -k 'not test_transcribe'
""",
        install_text="nc -l 0.0.0.0 8080\n",
        environment_text="{}",
    )
    assert any("0.0.0.0" in r for r in reasons)


def test_check_fails_on_embedded_key():
    reasons = offline_check.reasons_default_path_fetches_weights(
        workflow_text="""
jobs:
  whisper-test:
    steps:
      - run: pytest -k 'not test_transcribe'
""",
        install_text="export OPENAI_API_KEY=sk-notarealkeyvalue\n",
        environment_text="{}",
    )
    assert reasons, "checker must fail when a key is present"
    assert any("secret" in r.lower() or "key" in r.lower() for r in reasons)


def test_install_comment_may_mention_load_model():
    reasons = offline_check.reasons_default_path_fetches_weights(
        workflow_text="""
jobs:
  whisper-test:
    steps:
      - run: pytest -k 'not test_transcribe'
""",
        install_text=(
            "# Offline default: do not call load_model.\n" "pip install -e '.[dev]'\n"
        ),
        environment_text="{}",
    )
    assert reasons == [], reasons


def test_standalone_script_passes_on_this_tree():
    script = ROOT / "scripts" / "check_default_offline.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK:" in result.stdout


def test_ci_workflow_has_offline_default_job():
    text = (ROOT / ".github" / "workflows" / "test.yml").read_text()
    code = offline_check.strip_hash_comments(text)
    assert "offline-default:" in text
    assert "scripts/check_default_offline.py" in text
    assert "test_transcribe[tiny]" not in code
    assert "test_transcribe[tiny.en]" not in code


def test_environment_is_localhost_only_and_keyless():
    env_path = ROOT / ".cursor" / "environment.json"
    text = env_path.read_text()
    assert "0.0.0.0" not in text
    assert "sk-" not in text
    assert "HF_TOKEN" not in text
    data = json.loads(text)
    assert "ports" not in data
    assert "install" in data
