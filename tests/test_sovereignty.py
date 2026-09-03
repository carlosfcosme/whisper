"""Recycle CPU / no-weights / loopback rules onto remaining sovereignty tests."""

import importlib.util
from pathlib import Path

import pytest

from whisper.sovereign import ALL_INTERFACES, BIND_HOST, DEFAULT_DEVICE

REPO_ROOT = Path(__file__).resolve().parents[1]

# Existing unit tests that must stay Hub-free and weight-pull-free.
REMAINING_TEST_MODULES = (
    "tests/test_audio.py",
    "tests/test_tokenizer.py",
    "tests/test_timing.py",
    "tests/test_normalizer.py",
)

FORBIDDEN_IN_REMAINING_TESTS = (
    "huggingface",
    "hf.co",
    "from_pretrained",
    "hf_hub",
    "load_model",
    "urlopen",
    ALL_INTERFACES,
)

GITIGNORE_REQUIRED = (
    "*.pt",
    "*.pth",
    "*.bin",
    "*.ckpt",
    "*.safetensors",
    "*.onnx",
    ".cache/",
    ".huggingface/",
    "huggingface/",
    "hf_cache/",
    ".hub/",
    ".torch/",
)


def _load_check_loopback_bind():
    path = REPO_ROOT / "scripts" / "check_loopback_bind.py"
    spec = importlib.util.spec_from_file_location("check_loopback_bind", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


check_loopback_bind = _load_check_loopback_bind()


def test_cpu_default_stays_in_force():
    assert DEFAULT_DEVICE == "cpu"
    assert BIND_HOST == "127.0.0.1"
    assert ALL_INTERFACES == "0.0.0.0"


@pytest.mark.parametrize("relpath", REMAINING_TEST_MODULES)
def test_remaining_tests_have_no_hub_or_weight_pull(relpath):
    text = (REPO_ROOT / relpath).read_text(encoding="utf-8")
    for snippet in FORBIDDEN_IN_REMAINING_TESTS:
        assert snippet not in text, "{} must not {} ({})".format(
            relpath, "pull weights / Hub / bind all interfaces", snippet
        )


def test_remaining_test_modules_exist():
    for relpath in REMAINING_TEST_MODULES:
        assert (REPO_ROOT / relpath).is_file()


def test_gitignore_covers_cache_and_weights():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    missing = [item for item in GITIGNORE_REQUIRED if item not in gitignore]
    assert missing == [], "gitignore missing cache/weight entries: {}".format(missing)


def test_ci_keeps_cpu_and_no_weights_and_skips_transcribe():
    workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(
        encoding="utf-8"
    )
    assert "no-committed-weights" in workflow
    assert "scripts/check_no_weights.py" in workflow
    assert "scripts/check_loopback_bind.py" in workflow
    assert "HF_HUB_OFFLINE" in workflow
    assert "CUDA_VISIBLE_DEVICES" in workflow
    assert "not test_transcribe" in workflow
    assert "torch==${{ matrix.pytorch-version }}+cpu" in workflow


def test_start_scripts_exist_and_use_loopback():
    scripts = check_loopback_bind.discover_start_scripts(REPO_ROOT)
    assert any(path.name == "start.sh" for path in scripts)
    check_loopback_bind.assert_start_scripts_localhost_only(scripts)
    start = REPO_ROOT / ".cursor" / "start.sh"
    text = start.read_text(encoding="utf-8")
    assert "127.0.0.1" in text
    assert ALL_INTERFACES not in text


def test_environment_json_does_not_bind_all_interfaces():
    env = (REPO_ROOT / ".cursor" / "environment.json").read_text(encoding="utf-8")
    assert ALL_INTERFACES not in env
    assert "start.sh" in env


def test_scan_fails_when_all_interfaces_in_start_script(tmp_path):
    script = tmp_path / "start.sh"
    script.write_text("python3 -m http.server --bind {}\n".format(ALL_INTERFACES))
    with pytest.raises(AssertionError, match="not allowed in start scripts"):
        check_loopback_bind.assert_start_scripts_localhost_only([script])


def test_loopback_guard_script_passes_on_this_tree():
    assert check_loopback_bind.main() == 0
