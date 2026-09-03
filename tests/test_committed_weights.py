import importlib.util
import os
import subprocess
import sys

SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "check_committed_weights.py"
)


def _load_guard():
    spec = importlib.util.spec_from_file_location("check_committed_weights", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_repo_has_no_committed_weights():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    proc = subprocess.run(
        [sys.executable, SCRIPT, repo_root],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "no committed model weight files" in proc.stdout


def test_guard_detects_weight_suffixes():
    guard = _load_guard()
    assert guard.is_weight_path("models/tiny.pt")
    assert guard.is_weight_path("pytorch_model.bin")
    assert guard.is_weight_path("model.safetensors")
    assert not guard.is_weight_path("whisper/assets/gpt2.tiktoken")
    assert not guard.is_weight_path("whisper/assets/mel_filters.npz")
    assert not guard.is_weight_path("README.md")


def test_guard_fails_when_weights_are_committed(tmp_path):
    guard = _load_guard()
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    subprocess.run(
        ["git", "config", "user.email", "ci@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "CI"],
        cwd=tmp_path,
        check=True,
    )
    weight = tmp_path / "tiny.pt"
    weight.write_bytes(b"fake-weights")
    subprocess.run(["git", "add", "tiny.pt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "add weights"],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
    )
    assert guard.list_committed_weights(str(tmp_path)) == ["tiny.pt"]
    assert guard.main([str(tmp_path)]) == 1
