"""Coverage for gitignore + CI tracked-weight guard.

Does not import whisper, download Hub/CDN weights, or touch secrets.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_tracked_weights.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"

REQUIRED_GITIGNORE = (
    ".cache/",
    "cache/",
    ".huggingface/",
    ".torch/",
    "weights/",
    "checkpoints/",
    "*.pt",
    "*.pth",
    "*.pth.tar",
    "*.safetensors",
    "*.ckpt",
    "*.onnx",
    "*.gguf",
    "*.ggml",
)

IGNORE_EXAMPLES = (
    ".cache/whisper/tiny.pt",
    "cache/whisper/base.pt",
    "weights/model.pth",
    "checkpoints/epoch.ckpt",
    ".huggingface/hub/config.json",
    ".torch/hub/checkpoint.pth",
    "tiny.pt",
    "model.pth",
    "model.pth.tar",
    "model.safetensors",
    "model.ckpt",
    "model.onnx",
    "model.gguf",
    "model.ggml",
)

TRACKED_ASSETS = (
    "whisper/assets/gpt2.tiktoken",
    "whisper/assets/multilingual.tiktoken",
    "whisper/normalizers/english.json",
    "tests/jfk.flac",
    "README.md",
    ".gitignore",
)

GOOD_GITIGNORE = "\n".join(REQUIRED_GITIGNORE) + "\n"

FAIL_CASES = (
    ("tiny.pt", b"not-a-real-weight"),
    ("model.pth", b"not-a-real-weight"),
    ("model.pth.tar", b"not-a-real-weight"),
    ("model.safetensors", b"not-a-real-weight"),
    ("model.ckpt", b"not-a-real-weight"),
    ("model.onnx", b"not-a-real-weight"),
    ("model.gguf", b"not-a-real-weight"),
    (".cache/whisper/tiny.pt", b"not-a-real-weight"),
    ("cache/whisper/base.pt", b"not-a-real-weight"),
    ("weights/model.pth", b"not-a-real-weight"),
    ("checkpoints/epoch.ckpt", b"not-a-real-weight"),
    (".huggingface/hub/x.safetensors", b"not-a-real-weight"),
)


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


def test_script_and_ci_are_wired():
    assert SCRIPT.is_file()
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "scripts/check_tracked_weights.py" in workflow
    precommit = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "scripts/check_tracked_weights.py" in precommit
    text = SCRIPT.read_text(encoding="utf-8")
    assert "huggingface.co" not in text
    assert "sk-" not in text
    assert "FIELD_BRAIN" not in text.upper()
    assert "Field-Brain" not in text


def test_gitignore_declares_hardened_cache_and_weight_rules():
    lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    missing = [pattern for pattern in REQUIRED_GITIGNORE if pattern not in lines]
    assert missing == [], missing


def test_gitignore_covers_weight_examples():
    failed = []
    for path in IGNORE_EXAMPLES:
        result = _run(["git", "check-ignore", "-q", "--", path], REPO_ROOT)
        if result.returncode != 0:
            failed.append(path)
    assert failed == [], failed


def test_tracked_source_assets_are_not_ignored():
    for path in TRACKED_ASSETS:
        result = _run(["git", "check-ignore", "-q", "--", path], REPO_ROOT)
        assert result.returncode == 1, path
        assert (REPO_ROOT / path).is_file()


def test_guard_passes_on_this_repository():
    result = _run(["python3", str(SCRIPT)], REPO_ROOT)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
    listed = _run(
        [
            "git",
            "ls-files",
            "--",
            "*.pt",
            "*.pth",
            "*.safetensors",
            "*.ckpt",
            ".cache/**",
            "weights/**",
        ],
        REPO_ROOT,
    )
    assert listed.stdout.strip() == ""


def test_guard_fails_when_gitignore_is_incomplete(tmp_path):
    _init_repo(tmp_path, "# empty of required weight rules\n")
    result = _run(["python3", str(SCRIPT)], tmp_path)
    assert result.returncode != 0
    assert "gitignore missing" in result.stderr


@pytest.mark.parametrize("relpath,payload", FAIL_CASES)
def test_guard_fails_when_weight_artifact_is_tracked(tmp_path, relpath, payload):
    _init_repo(tmp_path, GOOD_GITIGNORE)
    dest = tmp_path / relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    subprocess.run(
        ["git", "add", "-f", "--", relpath],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    result = _run(["python3", str(SCRIPT)], tmp_path)
    assert result.returncode != 0, result.stdout
    assert relpath in result.stderr
