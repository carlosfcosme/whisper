# Formatter

The Cloud Agent environment installs **Black**, not Ruff.

Black is present because `.cursor/install.sh` runs `pip install -e ".[dev]"`,
and `black` is listed in `pyproject.toml` `[project.optional-dependencies] dev`.
`ruff` is not a project dependency and is not on `PATH`.

Do not run `ruff format` or add a Ruff config unless those files change.

## Format

```bash
black .
isort --profile black -l 88 --trailing-comma --multi-line 3 .
```

Line length is 88 (`[tool.isort]` and the flake8 / pre-commit args).
`[tool.black]` is present and uses Black defaults (also 88).

## Check

```bash
black --check .
isort --check-only --profile black -l 88 --trailing-comma --multi-line 3 .
flake8 --max-line-length 88 --ignore E203,E501,W503,W504
```

GitHub Actions runs the same stack through pre-commit:

```bash
pre-commit run --all-files
```

Config lives in `pyproject.toml`, `.pre-commit-config.yaml`, and `.flake8`.

This document does not download model weights and does not require secrets.
