"""Fail if CI workflows would pull Hub or model weights.

This module is stdlib-only so the checkout-only CI job can run it as
``python3 tests/test_ci_no_hub_pull.py`` without installing the package.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Tokens that mean a step would fetch Hub or named checkpoints.
_FETCH_INVOCATIONS = (
    "hf_hub_download",
    "snapshot_download",
    "huggingface-cli",
    "from huggingface_hub",
    "import huggingface_hub",
    "https://huggingface.co",
    "https://hf.co/",
    "hf-mirror.com",
    "from_pretrained(",
    "load_model(",
    "whisper.load_model",
    "azureedge.net",
    "test_transcribe[tiny]",
    "test_transcribe[tiny.en]",
)

_DENYLIST_HINTS = (
    "git grep",
    "git check-ignore",
    "git ls-files",
)


def _workflow_paths():
    return sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))


def _iter_run_blocks(text):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if stripped.startswith("run:") or stripped.startswith("- run:"):
            rest = stripped.split("run:", 1)[1].strip()
            indent = len(raw) - len(raw.lstrip())
            if rest in ("|", ">"):
                i += 1
                block = []
                while i < len(lines):
                    cur = lines[i]
                    if cur.strip() and (len(cur) - len(cur.lstrip())) <= indent:
                        break
                    block.append(cur)
                    i += 1
                yield "\n".join(block)
                continue
            if rest:
                yield rest
        i += 1


def _executable_lines(block):
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if any(hint in stripped for hint in _DENYLIST_HINTS):
            continue
        yield stripped


def test_ci_run_steps_do_not_pull_hub_or_weights():
    offenders = []
    for path in _workflow_paths():
        text = path.read_text()
        for block in _iter_run_blocks(text):
            for line in _executable_lines(block):
                for token in _FETCH_INVOCATIONS:
                    if token in line:
                        offenders.append("%s: %s" % (path.name, token))
    assert offenders == [], "CI must not fetch Hub/model weights: %s" % offenders


def test_ci_does_not_use_huggingface_actions():
    offenders = []
    for path in _workflow_paths():
        for line in path.read_text().splitlines():
            if "uses:" in line and "huggingface" in line.lower():
                offenders.append("%s: %s" % (path.name, line.strip()))
    assert offenders == [], "CI must not use Hugging Face actions: %s" % offenders


def test_whisper_test_job_cannot_pull_weights():
    text = (WORKFLOWS / "test.yml").read_text()
    assert "WHISPER_OFFLINE" in text
    assert "not test_transcribe" in text
    assert "not requires_weights" in text
    assert "test_transcribe[tiny]" not in text
    assert "test_transcribe[tiny.en]" not in text
    assert "+cpu" in text


def test_detector_fails_when_a_run_step_would_pull():
    fake = (
        "jobs:\n"
        "  x:\n"
        "    steps:\n"
        "      - run: huggingface-cli download openai/whisper-tiny\n"
        "      - run: python3 -c 'import whisper; whisper.load_model(\"tiny\")'\n"
    )
    found = []
    for block in _iter_run_blocks(fake):
        for line in _executable_lines(block):
            for token in _FETCH_INVOCATIONS:
                if token in line:
                    found.append(token)
    assert "huggingface-cli" in found
    assert "load_model(" in found


def test_gitignore_still_covers_cache_and_weights():
    lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    required = {".cache/", "cache/", "weights/", "*.pt", "*.pth"}
    missing = sorted(required - lines)
    assert missing == [], "cache/weight gitignore missing: %s" % missing


def main():
    test_ci_run_steps_do_not_pull_hub_or_weights()
    test_ci_does_not_use_huggingface_actions()
    test_whisper_test_job_cannot_pull_weights()
    test_detector_fails_when_a_run_step_would_pull()
    test_gitignore_still_covers_cache_and_weights()
    print("ci_no_hub_pull: ok")


if __name__ == "__main__":
    main()
