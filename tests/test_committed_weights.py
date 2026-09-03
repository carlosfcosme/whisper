import os
import subprocess
import sys

from whisper.committed_weights import (
    find_weight_files,
    git_tracked_files,
    is_weight_path,
    main,
    repo_root,
)

SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "whisper", "committed_weights.py"
)


def test_is_weight_path():
    assert is_weight_path("tiny.pt")
    assert is_weight_path("dir/model.safetensors")
    assert is_weight_path("weights/pytorch_model.bin")
    assert not is_weight_path("whisper/assets/mel_filters.npz")
    assert not is_weight_path("tests/jfk.flac")
    assert not is_weight_path("README.md")


def test_find_weight_files():
    hits = find_weight_files(
        [
            "README.md",
            "whisper/tiny.pt",
            "whisper/assets/mel_filters.npz",
            "models/model.safetensors",
        ]
    )
    assert hits == ["whisper/tiny.pt", "models/model.safetensors"]


def test_repo_has_no_committed_weights():
    root = repo_root()
    assert find_weight_files(git_tracked_files(root)) == []


def test_main_passes_on_clean_paths():
    assert main(["README.md", "whisper/assets/mel_filters.npz"]) == 0


def test_main_fails_on_checkpoint():
    assert main(["whisper/tiny.pt"]) == 1


def test_cli_script_passes_on_this_repo():
    result = subprocess.run(
        [sys.executable, SCRIPT],
        cwd=repo_root(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_script_fails_when_given_a_weight():
    result = subprocess.run(
        [sys.executable, SCRIPT, "checkpoints/model.pth"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "model.pth" in result.stderr
