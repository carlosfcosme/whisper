# Agent notes

## Cursor Cloud specific instructions

Cloud Agent VMs for this repository are **CPU-only by default**. There is no
CUDA GPU, no `nvidia-smi`, and no CUDA toolkit.

Whisper does not take an explicit env var for CUDA vs CPU. The library and CLI
pick a device implicitly:

```python
"cuda" if torch.cuda.is_available() else "cpu"
```

That check is in `whisper.load_model()` (`whisper/__init__.py`) and in the
CLI `--device` default (`whisper/transcribe.py`). On these VMs
`torch.cuda.is_available()` is `False`, so the resolved default is **`cpu`**.

`.cursor/install.sh` installs that contract: `torch==2.5.1+cpu` from the
PyTorch CPU wheel index (same pin as the Python 3.12 / 3.13 CI rows). Do not
replace it with a CUDA wheel. Passing `--device cuda` or `device="cuda"`
fails here.

### Confirm (no model weights)

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect `2.5.1+cpu False`.

### Tests

Skip CUDA-only cases (`tests/test_timing.py` is marked `requires_cuda`).
Skip `test_transcribe` unless checkpoints are already cached — it calls
`load_model` and will download weights on a cache miss.

```bash
pytest --durations=0 -vv -k 'not test_transcribe' -m 'not requires_cuda'
```

### Constraints

- **CPU-only.** Do not assume a GPU. Leave the implicit device default, or
  pass `--device cpu` / `device="cpu"` explicitly.
- **No weights.** Do not commit `.pt` files. Do not run `whisper.load_model()`
  or the `whisper` CLI (default model `turbo`) just to smoke-test; both fetch
  from `openaipublic.azureedge.net` on a cache miss.
- **No secrets.** Do not put tokens, passwords, or credentials in
  `AGENTS.md`, `.cursor/environment.json`, install scripts, or the git tree.
