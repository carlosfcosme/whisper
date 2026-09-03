from pathlib import Path


def test_gitignore_covers_cache_and_weights():
    text = Path(__file__).resolve().parents[1].joinpath(".gitignore").read_text()
    for needle in (
        ".cache/",
        "cache/",
        "weights/",
        "*.pt",
        "*.pth",
        "*.safetensors",
        "*.onnx",
        "*.bin",
        "*.weights",
    ):
        assert needle in text, "missing gitignore pattern: {}".format(needle)
