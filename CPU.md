# CPU-only default

This checkout's Cloud Agent, CI, and `.cursor/install.sh` environments are
**CPU-only by default**. There is no CUDA GPU, no `nvidia-smi`, and no CUDA
toolkit on those machines.

## Implicit device

Whisper does not read an environment variable for CUDA vs CPU. When you omit
`--device` / `device=`, the library and CLI pick a device with:

```python
"cuda" if torch.cuda.is_available() else "cpu"
```

That check is in:

- `whisper.load_model()` in `whisper/__init__.py`
- the CLI `--device` default in `whisper/transcribe.py`

`.cursor/install.sh` and `.github/workflows/test.yml` install a CPU wheel
(`torch==…+cpu` from the PyTorch CPU index). On those installs
`torch.cuda.is_available()` is `False`, so the resolved default is **`cpu`**.

`--device cuda` or `device="cuda"` fails here. Leave the implicit default, or
pass `--device cpu` / `device="cpu"` explicitly.

On CPU, `transcribe()` does not use FP16 (it warns and falls back to FP32).

## Confirm (no model weights)

These checks do not call `whisper.load_model()` and do not download
checkpoints:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 -m whisper --help
```

Expect a `+cpu` torch version, `False`, and `--device` showing
`(default: cpu)`. `--help` exits in argparse before `load_model()`.

Do **not** run `whisper audio.mp3` or `whisper.load_model("turbo")` just to
smoke-test the device. Both fetch weights from `openaipublic.azureedge.net`
on a cache miss.

## Tests

Skip CUDA-only cases (`tests/test_timing.py` is marked `requires_cuda`).
Skip `test_transcribe` unless checkpoints are already cached — it calls
`load_model` and will download weights on a cache miss.

```bash
pytest --durations=0 -vv -k 'not test_transcribe' -m 'not requires_cuda'
```

## Constraints

- **CPU-only.** Do not assume a GPU. Do not replace the CPU torch wheel with
  a CUDA build in Cloud Agent setup.
- **No weights.** Do not commit `.pt` / `.pth` files. Do not download
  checkpoints to document this default.
- **No secrets.** Do not put tokens, passwords, or credentials in `CPU.md`,
  `.cursor/environment.json`, install scripts, or the git tree.
