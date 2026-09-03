# CPU-only default

`whisper.DEFAULT_DEVICE` is **`cpu`**. That is a sovereign code default, not
a CUDA-availability fallback.

## Device in code

When you omit `--device` / `device=`, the library and CLI use
`DEFAULT_DEVICE` (`"cpu"`):

- `whisper.load_model()` in `whisper/__init__.py`
- the CLI `--device` default in `whisper/transcribe.py`

Pass `--device cuda` or `device="cuda"` to use a GPU. There is no environment
variable that switches the default to CUDA.

`.cursor/install.sh` and `.github/workflows/test.yml` install a CPU wheel
(`torch==…+cpu` from the PyTorch CPU index). On Cloud Agent VMs there is no
CUDA GPU, no `nvidia-smi`, and no CUDA toolkit.

On CPU, `transcribe()` does not use FP16 (it warns and falls back to FP32).

## Confirm (no model weights, no Hub)

These checks do not call `whisper.load_model()` with an official name and do
not download checkpoints from Azure or the Hugging Face Hub:

```bash
python3 -c "import whisper, torch; print(whisper.DEFAULT_DEVICE, torch.__version__, torch.cuda.is_available())"
python3 -m whisper --help
```

Expect `cpu`, a `+cpu` torch version, and `--device` showing `(default: cpu)`.
`--help` exits in argparse before `load_model()`.

Do **not** run `whisper audio.mp3` or `whisper.load_model("turbo")` just to
smoke-test the device. Both fetch weights from `openaipublic.azureedge.net`
on a cache miss.

## Tests

Tests must not hit the Hugging Face Hub. `tests/conftest.py` sets
`HF_HUB_OFFLINE=1` (and related offline flags) and blocks remote `urlopen`
(including `huggingface.co` and the Azure weight CDN). Loopback stays allowed.

Skip CUDA-only cases (`tests/test_timing.py` is marked `requires_cuda`).
`test_transcribe` is marked `requires_local_weights` and skips unless a
checkpoint is already on disk — it does not download from the Hub.

```bash
pytest --durations=0 -vv -k 'not test_transcribe' -m 'not requires_cuda'
```

## Constraints

- **CPU-only default.** Do not restore `"cuda" if torch.cuda.is_available()`.
- **No Hub.** Tests must not download from Hugging Face Hub or the Azure
  weight CDN.
- **No weights.** Do not commit `.pt` / `.pth` files.
- **No secrets.** Do not put tokens, passwords, or credentials in `CPU.md`,
  `.cursor/environment.json`, install scripts, or the git tree.
