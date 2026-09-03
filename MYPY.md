# Mypy

Status: **none** (not present).

The Cloud Agent / developer environment treated mypy as implicit. This
checkout does not install or configure it. Do not run `mypy` unless those
files change.

## Inventory

| Item | Present? |
|------|----------|
| `mypy` on `PATH` | No |
| `mypy` Python package | No (`ModuleNotFoundError`) |
| `mypy` in `[project]` / `.[dev]` | No ([`pyproject.toml`](pyproject.toml)) |
| `[tool.mypy]` in [`pyproject.toml`](pyproject.toml) | No |
| `mypy.ini` / `.mypy.ini` | No |
| pre-commit mypy hook | No ([`.pre-commit-config.yaml`](.pre-commit-config.yaml)) |
| CI mypy job | No ([`.github/workflows/test.yml`](.github/workflows/test.yml)) |

`.cursor/install.sh` installs `.[dev]` (`pytest`, `black`, `isort`, `flake8`,
`scipy`). That is not a type checker. Lint is Black / isort / flake8 (and
the CI `pre-commit` job). None of those invoke mypy.

## Weights and secrets

This document does not download Whisper checkpoints and does not need API
keys or credentials. Do not commit `.pt` files.
