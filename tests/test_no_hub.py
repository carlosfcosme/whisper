import os
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HUB_TOKENS = (
    "huggingface.co",
    "huggingface_hub",
    "hf_hub_download",
    "from_pretrained",
    "hf.co/",
)


def test_hub_offline_env_is_set():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_DATASETS_OFFLINE") == "1"


def test_hub_urlopen_is_blocked():
    with pytest.raises(RuntimeError, match="Hugging Face Hub is forbidden"):
        urllib.request.urlopen("https://huggingface.co/openai/whisper-tiny")


def test_unit_tests_do_not_reference_hub():
    tests = ROOT / "tests"
    for path in tests.glob("test_*.py"):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        for token in HUB_TOKENS:
            assert token not in text, "{} references Hub ({})".format(path, token)
