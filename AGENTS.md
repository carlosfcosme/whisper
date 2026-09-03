# Agent notes

## Cursor Cloud specific instructions

Sovereign Cloud Agent / CI path, implemented in `whisper/runtime.py`:

1. **CPU default.** `whisper.default_device()` returns `cpu`.
2. **No Hub download in tests.** Cache-miss `_download` / `load_model(name)` is refused, including Hugging Face Hub.
3. **Bind 127.0.0.1.** Helper listeners use `whisper.default_bind_host()` / `bind_localhost()`. Never `0.0.0.0`.

| Env var | Default on this path | Effect |
| --- | --- | --- |
| `WHISPER_CPU_ONLY` | `1` | `default_device()` returns `cpu` |
| `WHISPER_NO_WEIGHT_DOWNLOAD` | `1` | Cache-miss weight pull is refused |
| `WHISPER_LOCALHOST_ONLY` | `1` | Bind policy is loopback-only |
| `CI` | set in GitHub Actions | Same CPU / no-download path |
| `WHISPER_DEVICE` | unset | Optional device override |
| `WHISPER_BIND_HOST` | unset (`127.0.0.1`) | Loopback-only override; `0.0.0.0` is refused |
| `WHISPER_ALLOW_WEIGHT_DOWNLOAD` | unset | Escape hatch for a non-Hub cache-miss pull |

`.cursor/install.sh`, `tests/conftest.py`, and `.github/workflows/test.yml`
set the three flags. Hugging Face Hub URLs (`huggingface.co`, `hf.co`) are
refused even if a pull is otherwise allowed. Unit tests autouse-block Hub
helpers and wildcard `socket.bind`.

`load_model()` and the CLI `--device` default call `default_device()`. Passing
`--device cuda` or `device="cuda"` fails on these VMs. Do not install a CUDA
PyTorch wheel; install.sh pins `torch==2.5.1+cpu`. `.cursor/environment.json`
publishes no ports.

### Confirm (no model weights)

```bash
python3 -c "import whisper, torch; print(torch.__version__, torch.cuda.is_available(), whisper.default_device(), whisper.default_bind_host())"
```

Expect `2.5.1+cpu False cpu 127.0.0.1`.

### Tests

```bash
pytest --durations=0 -vv -k 'not test_transcribe' -m 'not requires_cuda'
```

`tests/test_cpu_default.py` asserts CPU default, no Hub/CDN pull, and bind
host `127.0.0.1`. `tests/test_localhost_bind.py` binds a real socket on
`127.0.0.1` and refuses `0.0.0.0`. `test_transcribe` is skipped while
auto-download is disabled.

### Constraints

- **CPU-only.** Do not assume a GPU.
- **No weights.** Do not commit `.pt` files. Do not run `whisper.load_model()`
  or the `whisper` CLI (default model `turbo`) to smoke-test; both would
  auto-download and are refused on this path.
- **Bind 127.0.0.1.** Do not listen on `0.0.0.0`.
- **No secrets.** Do not put tokens, passwords, or credentials in
  `AGENTS.md`, `.cursor/environment.json`, install scripts, or the git tree.
