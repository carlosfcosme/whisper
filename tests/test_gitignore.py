import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.safetensors",
    "*.ckpt",
    "*.gguf",
)


def test_gitignore_ignores_weight_suffixes():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = {line.strip() for line in text.splitlines()}
    missing = [pattern for pattern in REQUIRED if pattern not in lines]
    assert missing == []


def test_gitignore_ci_script_passes():
    path = ROOT / "scripts" / "check_gitignore.py"
    spec = importlib.util.spec_from_file_location("check_gitignore", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.missing_patterns(ROOT) == []
    assert module.main() == 0
