import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECK = REPO / "scripts" / "check_no_committed_weights.py"


def test_gitignore_lists_cache_and_weight_names():
    text = (REPO / ".gitignore").read_text(encoding="utf-8")
    for pattern in (
        ".cache/",
        "cache/",
        "weights/",
        "*.pt",
        "*.pth",
        "*.safetensors",
    ):
        assert pattern in text


def test_git_ignores_project_local_caches_and_weights():
    samples = (
        ".cache/whisper/tiny.pt",
        "cache/whisper/tiny.pt",
        "weights/tiny.pt",
        "tiny.pt",
        "model.pth",
        "model.safetensors",
    )
    out = subprocess.check_output(
        ["git", "check-ignore", "-v", *samples], cwd=REPO, text=True
    )
    for sample in samples:
        assert sample in out


def test_no_tracked_weights_or_caches():
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=REPO, text=True
    ).splitlines()
    forbidden_suffix = (".pt", ".pth", ".safetensors", ".ckpt", ".onnx")
    forbidden_prefix = (".cache/", "cache/", "weights/")
    for path in tracked:
        lower = path.lower()
        assert not lower.endswith(forbidden_suffix), path
        assert not any(lower.startswith(p) for p in forbidden_prefix), path


def test_check_script_passes_on_this_repo():
    subprocess.check_call([sys.executable, str(CHECK)], cwd=REPO)


def test_check_script_fails_when_weight_is_tracked(tmp_path):
    subprocess.check_call(["git", "init"], cwd=tmp_path)
    subprocess.check_call(
        ["git", "config", "user.email", "ci@example.com"], cwd=tmp_path
    )
    subprocess.check_call(["git", "config", "user.name", "ci"], cwd=tmp_path)
    (tmp_path / ".gitignore").write_text(
        "*.pt\n*.pth\n*.safetensors\n.cache/\ncache/\nweights/\n"
    )
    weight = tmp_path / "leaked.pt"
    # Force-add a tracked weight even if gitignore would skip it.
    weight.write_bytes(b"not-a-real-checkpoint")
    subprocess.check_call(["git", "add", "-f", "leaked.pt", ".gitignore"], cwd=tmp_path)
    subprocess.check_call(["git", "commit", "-m", "leak"], cwd=tmp_path)
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, str(CHECK)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "leaked.pt" in proc.stderr
