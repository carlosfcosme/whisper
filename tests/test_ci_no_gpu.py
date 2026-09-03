from pathlib import Path

import torch

import whisper

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = (REPO / ".github" / "workflows" / "test.yml").read_text()


def test_ci_hides_gpu_and_installs_cpu_torch():
    assert 'CUDA_VISIBLE_DEVICES: ""' in WORKFLOW
    assert "torch==${{ matrix.pytorch-version }}+cpu" in WORKFLOW
    assert "--index-url https://download.pytorch.org/whl/cpu" in WORKFLOW
    assert "-m 'not requires_cuda'" in WORKFLOW
    assert "CI must not see a GPU" in WORKFLOW
    assert 'HF_HUB_OFFLINE: "1"' in WORKFLOW
    assert "scripts/check_no_wildcard_bind.py --probe-negative" in WORKFLOW
    assert "scripts/check_no_weights.py --probe-negative" in WORKFLOW


def test_runtime_has_no_visible_cuda():
    assert whisper.DEFAULT_DEVICE == "cpu"
    assert not torch.cuda.is_available()
