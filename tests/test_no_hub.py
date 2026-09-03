from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("whisper", "scripts", ".cursor")
HUB_NEEDLES = (
    "hugging" + "face.co",
    "hugging" + "face_hub",
    "hf_" + "hub",
)


def test_huggingface_hub_is_not_a_dependency():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "huggingface" not in text
    req = REPO_ROOT / "requirements.txt"
    if req.is_file():
        assert "huggingface" not in req.read_text(encoding="utf-8").lower()


def test_huggingface_hub_import_is_blocked():
    with pytest.raises(ImportError, match="huggingface_hub"):
        import huggingface_hub  # noqa: F401


def test_application_sources_do_not_reference_the_hub():
    hits = []
    for dirname in SCAN_DIRS:
        root = REPO_ROOT / dirname
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".sh", ".json", ".md"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            lowered = text.lower()
            if any(needle in lowered for needle in HUB_NEEDLES):
                hits.append(str(path.relative_to(REPO_ROOT)))
    assert hits == [], "Hugging Face Hub references in application sources: %s" % hits
