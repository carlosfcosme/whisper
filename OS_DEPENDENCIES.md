# OS dependencies

`pip` does not install these. This file lists **package names only**. It is not an install script, does not download model weights, and does not contain secrets.

## Required (runtime)

`whisper.audio.load_audio` runs the `ffmpeg` CLI (must be on `PATH`).

| OS / manager | Package name |
| --- | --- |
| Debian, Ubuntu (`apt`) | `ffmpeg` |
| Arch Linux (`pacman`) | `ffmpeg` |
| macOS (Homebrew) | `ffmpeg` |
| Windows (Chocolatey) | `ffmpeg` |
| Windows (Scoop) | `ffmpeg` |
| conda (CI) | `ffmpeg` |

## Optional (install-time)

Needed only if [tiktoken](https://github.com/openai/tiktoken) has no wheel for the platform.

| OS / manager | Package names |
| --- | --- |
| Debian, Ubuntu (`apt`) | `rustc`, `cargo` |
| Arch Linux (`pacman`) | `rust` |
| macOS (Homebrew) | `rust` |
| rustup | `rustc`, `cargo` |
