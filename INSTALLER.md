# Installer: pip, not uv

Python packages in this tree are installed with **pip**. **uv** is not the
installer.

This file records what Setup, Cloud Agent bootstrap, and CI already run. It
does not download model weights and does not require secrets.

## What the tree uses

| Surface | Command |
| --- | --- |
| Cloud Agent ([`.cursor/install.sh`](.cursor/install.sh)) | `pip install --break-system-packages` (CPU torch, then `-e ".[dev]"`) |
| Tests ([`.github/workflows/test.yml`](.github/workflows/test.yml)) | `pip install --upgrade pre-commit` and `pip3 install .["dev"]` |
| Release ([`.github/workflows/python-publish.yml`](.github/workflows/python-publish.yml)) | `python -m pip install --upgrade pip` |
| README Setup | `pip install -U openai-whisper` |

`.cursor/environment.json` runs `bash .cursor/install.sh`. That script calls
system `pip`. There is no `uv` step.

## What is not present

| Signal | Present? |
| --- | --- |
| `uv` on `PATH` (Cloud Agent image) | No |
| `uv.lock` | No |
| `[tool.uv]` in [`pyproject.toml`](pyproject.toml) | No |
| `uv` as a dependency | No |
| `Pipfile` / `poetry.lock` | No |

Do not run `uv pip install` or `uv sync` against this checkout unless those
files change. Those commands are not part of Setup, CI, or the Cloud Agent
environment.

## Commands the tree actually runs

Cloud Agent ([`.cursor/install.sh`](.cursor/install.sh)):

```bash
pip install --break-system-packages \
  "numpy" torch==2.5.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple

pip install --break-system-packages -e ".[dev]"
```

README (PyPI release):

```bash
pip install -U openai-whisper
```

## Confirm on a Cloud Agent VM

```bash
command -v pip
pip --version
command -v uv   # expected: not found
```

On this environment those resolve to system pip 24.0
(`/usr/lib/python3/dist-packages/pip`, Python 3.12). `uv` is absent.

## Out of scope

- Do not download or commit checkpoints (`.pt`).
- Do not add API keys, tokens, or credentials.
