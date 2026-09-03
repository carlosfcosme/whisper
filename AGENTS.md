# Agent notes

## Cursor Cloud specific instructions

Cloud Agent VMs for this repository are **CPU-only by default**. There is no
CUDA GPU. The Cloud Agent / CI path is implemented in `whisper/runtime.py`,
not implied by `torch.cuda.is_available()`.

| Env var | Default on this path | Effect |
| --- | --- | --- |
| `WHISPER_CPU_ONLY` | `1` | `whisper.default_device()` returns `cpu` |
| `WHISPER_NO_WEIGHT_DOWNLOAD` | `1` | Cache-miss `_download` / `load_model(name)` is refused |
| `CI` | set in GitHub Actions | Same as the two flags above |
| `WHISPER_DEVICE` | unset | Optional override (`cpu` / `cuda`) |
| `WHISPER_ALLOW_WEIGHT_DOWNLOAD` | unset | Escape hatch to allow a cache-miss pull |

`.cursor/install.sh`, `tests/conftest.py`, and `.github/workflows/test.yml`
all set `WHISPER_CPU_ONLY=1` and `WHISPER_NO_WEIGHT_DOWNLOAD=1`. Hugging Face
Hub URLs (`huggingface.co`, `hf.co`) are refused even if a pull is otherwise
allowed.

`load_model()` and the CLI `--device` default call `default_device()`. Passing
`--device cuda` or `device="cuda"` fails on these VMs. Do not install a CUDA
PyTorch wheel; install.sh pins `torch==2.5.1+cpu`.

### Confirm (no model weights)

```bash
python3 -c "import whisper, torch; print(torch.__version__, torch.cuda.is_available(), whisper.default_device())"
```

Expect `2.5.1+cpu False cpu`.

### Tests

```bash
pytest --durations=0 -vv -k 'not test_transcribe' -m 'not requires_cuda'
```

`tests/test_cpu_default.py` asserts the default device is `cpu` and that a
cache miss (official CDN or Hugging Face Hub) is refused with no network.
`test_transcribe` is skipped while auto-download is disabled.

### Constraints

- **CPU-only.** Do not assume a GPU.
- **No weights.** Do not commit `.pt` files. Do not run `whisper.load_model()`
  or the `whisper` CLI (default model `turbo`) to smoke-test; both would
  auto-download and are now refused on this path.
- **No secrets.** Do not put tokens, passwords, or credentials in
  `AGENTS.md`, `.cursor/environment.json`, install scripts, or the git tree.
