# Python and virtualenv

The Whisper checkout does not define a project virtualenv. Interpreter
requirements are scattered across packaging, CI, and the Cloud Agent install
script. This file records those values **from the tree** so the default
Python is not implicit.

No model weights and no secrets belong in this file or in git.

## Virtualenv

There is no venv in the tree:

- no `pyvenv.cfg`, `venv/`, or `.venv/`
- no `Pipfile` or Conda `environment.yml`
- `.cursor/install.sh` does not run `python3 -m venv`

The Cloud Agent install uses the system `python3` / `pip` on `PATH` and
writes packages to the user site with `--break-system-packages`. GitHub
Actions creates an ephemeral Conda env named `test` in
`.github/workflows/test.yml`; that env is not checked in.

## Versions recorded in the tree

| Role | Version | Source |
| --- | --- | --- |
| Packaging floor | `>=3.8` | `requires-python` in `pyproject.toml` |
| Declared CPython series | 3.8–3.13 | Trove classifiers in `pyproject.toml` |
| Historical train/test mention | 3.9.9 + PyTorch 1.10.1 | `README.md` Setup (compatibility line there still says 3.8–3.11) |
| Pre-commit job | 3.9 | `.github/workflows/test.yml` (`actions/setup-python`) |
| Publish job | 3.8 | `.github/workflows/python-publish.yml` |
| Test matrix | 3.8, 3.9, 3.10, 3.11, 3.12, 3.13 | `.github/workflows/test.yml` |
| Cloud Agent / install default | 3.12 | `.cursor/install.sh` (torch pin “CI uses for Python 3.12”) and `.python-version` |

`.python-version` is `3.12`. That matches the install script’s pairing with
the CI cell `python-version: '3.12'` + `pytorch-version: 2.5.1` (CPU wheel
`torch==2.5.1+cpu`). It is the default interpreter for this environment, not
a new `requires-python` bound.

## CI Python × PyTorch cells

From `.github/workflows/test.yml`:

| Python | PyTorch | NumPy constraint |
| --- | --- | --- |
| 3.8 | 1.10.1 | `numpy<2` |
| 3.8 | 1.13.1 | `numpy<2` |
| 3.8 | 2.0.1 | `numpy<2` |
| 3.9 | 2.1.2 | `numpy<2` |
| 3.10 | 2.2.2 | `numpy<2` |
| 3.11 | 2.3.1 | unconstrained `numpy` |
| 3.12 | 2.4.1 | unconstrained `numpy` |
| 3.12 | 2.5.1 | unconstrained `numpy` (install default) |
| 3.13 | 2.5.1 | unconstrained `numpy` |

## Out of scope

- Do not download or commit checkpoints (`.pt`).
- Do not add API keys, tokens, or credentials.
