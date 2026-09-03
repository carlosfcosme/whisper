# OS dependency names

`pip` does not install these. This file lists **package and CLI names only**. It is not an install script, does not download model weights, and does not contain secrets.

Python packages (`torch`, `tiktoken`, `numpy`, and the rest of `pyproject.toml`) are not OS packages. Whisper checkpoints (`.pt` files under `~/.cache/whisper`) are not OS packages either.

## Required at runtime: `ffmpeg`

`whisper.audio.load_audio` runs the `ffmpeg` CLI from `PATH` (not a Python binding; `ffmpeg-python` was removed). CI installs the conda package `ffmpeg`. `.cursor/install.sh` installs the apt package `ffmpeg` when the CLI is missing.

| OS / manager | Package name | CLI on PATH |
| --- | --- | --- |
| Debian, Ubuntu (`apt` / `apt-get`) | `ffmpeg` | `ffmpeg` |
| Arch Linux (`pacman`) | `ffmpeg` | `ffmpeg` |
| macOS (Homebrew) | `ffmpeg` | `ffmpeg` |
| Windows (Chocolatey) | `ffmpeg` | `ffmpeg` |
| Windows (Scoop) | `ffmpeg` | `ffmpeg` |
| conda (GitHub Actions) | `ffmpeg` | `ffmpeg` |

## Optional at install time: Rust (tiktoken wheels)

Needed only if [tiktoken](https://github.com/openai/tiktoken) has no pre-built wheel. README points at [rustup](https://www.rust-lang.org/learn/get-started). Binaries are `rustc` and `cargo`.

| OS / manager | Package names | CLIs on PATH |
| --- | --- | --- |
| rustup (README) | rustup toolchain | `rustc`, `cargo` |
| Debian, Ubuntu (`apt`) | `rustc`, `cargo` | `rustc`, `cargo` |
| Arch Linux (`pacman`) | `rust` | `rustc`, `cargo` |
| macOS (Homebrew) | `rust` | `rustc`, `cargo` |

`setuptools-rust` is a pip package (`pip install setuptools-rust`), not an OS package.
