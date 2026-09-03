from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / ".github" / "workflows" / "test.yml").read_text()


def test_ci_skips_transcribe_and_disables_hub():
    assert "-k 'not test_transcribe'" in WORKFLOW
    assert "HF_HUB_OFFLINE" in WORKFLOW
    assert "WHISPER_NO_WEIGHT_DOWNLOAD" in WORKFLOW
    assert "WHISPER_CPU_ONLY" in WORKFLOW
    assert "CUDA_VISIBLE_DEVICES" in WORKFLOW
    assert "test_transcribe[tiny]" not in WORKFLOW
    assert "huggingface.co" not in WORKFLOW


def test_ci_fails_on_committed_weights():
    assert "scripts/check_no_weights.py" in WORKFLOW
    assert "no-committed-weights" in WORKFLOW
