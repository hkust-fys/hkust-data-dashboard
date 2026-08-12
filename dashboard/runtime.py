"""Runtime dependency checks with no import-time side effects."""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from pathlib import Path

from dashboard.config import ConfigError

log = logging.getLogger(__name__)


def resolve_ffmpeg_executable(*, verify: bool = True) -> str:
    """Return imageio-ffmpeg's bundled executable or raise a useful error."""
    install_command = f'"{sys.executable}" -m pip install imageio-ffmpeg'
    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
    except ImportError as exc:
        raise ConfigError(
            "Camera support requires imageio-ffmpeg. Install it with: "
            f"{install_command}"
        ) from exc

    try:
        executable = str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(
            "imageio-ffmpeg could not locate its bundled ffmpeg executable. "
            f"Reinstall it with: {install_command}"
        ) from exc

    if not executable or not Path(executable).is_file():
        raise ConfigError(
            f"imageio-ffmpeg reported a missing ffmpeg executable at {executable!r}. "
            f"Reinstall it with: {install_command}"
        )

    if verify:
        try:
            completed = subprocess.run(
                [executable, "-version"],
                capture_output=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ConfigError(
                f"Bundled ffmpeg could not be started at {executable!r}: {exc}. "
                f"Reinstall imageio-ffmpeg with: {install_command}"
            ) from exc
        if completed.returncode != 0:
            raise ConfigError(
                f"Bundled ffmpeg failed its startup check at {executable!r} "
                f"(exit {completed.returncode}). Reinstall imageio-ffmpeg with: "
                f"{install_command}"
            )
    return executable


def startup_preflight() -> str:
    """Report the interpreter and validate camera decoding dependencies."""
    log.info("Python executable: %s", sys.executable)
    executable = resolve_ffmpeg_executable()
    log.info("Bundled ffmpeg executable: %s", executable)
    return executable
