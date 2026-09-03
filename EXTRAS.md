# Pip extras

The Cloud Agent extra environment (`.cursor/install.sh`) and CI install
`".[dev]"` implicitly. This file is the inventory of **declared pip extras**
so that extra is not only buried in the install script.

There is **one** extra. Do not invent `cuda`, `gpu`, `docs`, `test`, or
similar extras; they are not declared.

No model weights and no secrets belong in this file or in git.

## Inventory

Source of truth: `[project.optional-dependencies]` in
[`pyproject.toml`](pyproject.toml).

| Extra | Packages | Purpose |
|-------|----------|---------|
| `dev` | `black`, `flake8`, `isort`, `pytest`, `scipy` | Tests and linting |

`scipy` is used by the CPU timing tests (`tests/test_timing.py`), not by the
runtime package. `pre-commit` is **not** in `dev`; CI installs that runner
separately (see [`.pre-commit-config.yaml`](.pre-commit-config.yaml)).

Core runtime dependencies (`more-itertools`, `numba`, `numpy`, `tiktoken`,
`torch`, `tqdm`, and Linux-x86_64 `triton`) are required, not extras.
`ffmpeg` is an OS package, not a pip extra.

## Install

From a checkout (same extra the extra env uses):

```bash
pip install -e ".[dev]"
```

From a released wheel:

```bash
pip install "openai-whisper[dev]"
```

On this Cloud Agent VM, `.cursor/install.sh` uses
`pip install --break-system-packages -e ".[dev]"` after a CPU-only
`torch==2.5.1+cpu` pin. CI (`.github/workflows/test.yml`) uses
`pip3 install .["dev"]`.

## Weights and secrets

Installing extras does not download Whisper checkpoints. Skip
`test_transcribe` unless you intend to fetch weights:

```bash
pytest --durations=0 -vv -k 'not test_transcribe' -m 'not requires_cuda'
```

Do not commit `.pt` files, API keys, or credentials.
