import subprocess

from check_committed_weights import (
    committed_weight_paths,
    is_weight_path,
    main,
)


def test_is_weight_path_flags_checkpoints():
    assert is_weight_path("tiny.pt")
    assert is_weight_path("models/tiny.en.pt")
    assert is_weight_path("pytorch_model.bin")
    assert is_weight_path("model.safetensors")
    assert is_weight_path("foo.ckpt")
    assert is_weight_path(r"dir\weights.pth")
    assert is_weight_path("llama.gguf")


def test_is_weight_path_allows_repo_assets():
    assert not is_weight_path("whisper/assets/mel_filters.npz")
    assert not is_weight_path("whisper/assets/multilingual.tiktoken")
    assert not is_weight_path("tests/jfk.flac")
    assert not is_weight_path("README.md")


def test_this_repo_has_no_committed_weights():
    assert committed_weight_paths() == []
    assert main() == 0


def test_ci_fails_when_weight_is_committed(tmp_path):
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    weight = tmp_path / "tiny.pt"
    weight.write_bytes(b"not-a-real-checkpoint")
    subprocess.run(
        ["git", "add", "tiny.pt"], cwd=str(tmp_path), check=True, capture_output=True
    )
    assert committed_weight_paths(str(tmp_path)) == ["tiny.pt"]
    assert main(str(tmp_path)) == 1
