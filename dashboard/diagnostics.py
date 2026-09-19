"""Bounded deployment logs; configured explicitly by the entry point."""

from __future__ import annotations

import copy
import hashlib
import logging
import re
import secrets
import time
from importlib.metadata import PackageNotFoundError, version
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dashboard.http import safe_endpoint

LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3


def redact_log_text(text: str, secret_values: tuple[str, ...] = ()) -> str:
    """Exclude credentials, request queries, and complete sensitive headers."""
    for value in sorted(set(secret_values), key=len, reverse=True):
        if value:
            text = text.replace(value, "<redacted>")
    text = re.sub(
        r"(?im)\b(authorization|proxy-authorization|cookie|set-cookie)\s*[:=][^\r\n]*",
        r"\1=<redacted>", text,
    )
    text = re.sub(r"https?://[^\s\"'<>]+", lambda match: safe_endpoint(match[0]), text)
    text = re.sub(
        r"(?i)\b((?:[a-z0-9]+[_-])*(?:token|key|secret|password))"
        r"[\"']?\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
        r"\1=<redacted>", text,
    )
    # One physical line per event prevents arbitrary exception text from
    # masquerading as a separate log entry.
    return text.replace("\r", r"\r").replace("\n", r"\n")[:4096]


class DiagnosticFormatter(logging.Formatter):
    converter = time.gmtime

    def __init__(self, run_id: str, secret_values: tuple[str, ...] = ()) -> None:
        super().__init__(
            f"%(asctime)sZ %(levelname)s %(name)s pid=%(process)d run={run_id}: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        self.secret_values = secret_values

    def formatException(self, exc_info) -> str:  # noqa: N802
        """Keep cause types and call sites, without locals or source snippets."""
        error = exc_info[1]
        causes = []
        seen: set[int] = set()
        while error is not None and id(error) not in seen and len(causes) < 4:
            seen.add(id(error))
            trace = error.__traceback__
            frames = []
            while trace is not None:
                code = trace.tb_frame.f_code
                frames.append(f"{Path(code.co_filename).name}:{trace.tb_lineno}:{code.co_name}")
                trace = trace.tb_next
            causes.append(f"{type(error).__name__} at {' -> '.join(frames[-8:]) or 'unknown'}")
            error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
        return " caused_by=".join(causes)

    def format(self, record: logging.LogRecord) -> str:
        # Do not reuse another handler's unsanitized cached exception text.
        record = copy.copy(record)
        record.exc_text = None
        return redact_log_text(super().format(record), self.secret_values)


def configure_logging(
    level: str, cache_dir: str, *, secret_values: tuple[str, ...] = (),
) -> Path | None:
    """Install console and size-rotated logs, retaining stderr if disk fails."""
    formatter = DiagnosticFormatter(secrets.token_hex(4), secret_values)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    handlers: list[logging.Handler] = [console]
    path = Path(cache_dir) / "logs" / "dashboard.log"
    file_error = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    except OSError as exc:
        file_error = exc
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, handlers=handlers, force=True)
    # discord.py DEBUG includes full REST bodies and gateway messages. Keep
    # those out of diagnostic files even when application diagnostics use DEBUG.
    logging.getLogger("discord").setLevel(max(logging.INFO, log_level))
    logger = logging.getLogger(__name__)
    if file_error is not None:
        logger.warning("log file unavailable path=%s error=%s; using stderr", path, file_error)
        return None
    logger.info(
        "logging ready file=%s max_bytes=%d backups=%d timestamps=UTC",
        path, LOG_MAX_BYTES, LOG_BACKUP_COUNT,
    )
    return path


def source_fingerprint(root: Path) -> str:
    """Identify the actual deployed Python source, including uncommitted fixes."""
    digest = hashlib.sha256()
    try:
        paths = [root / "bot.py", *sorted((root / "dashboard").rglob("*.py"))]
        for path in paths:
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    except OSError:
        return "unavailable"
    return digest.hexdigest()[:12]


def dependency_versions() -> str:
    installed = []
    for name in ("discord.py", "aiohttp", "playwright", "Pillow"):
        try:
            installed.append(f"{name}={version(name)}")
        except PackageNotFoundError:
            installed.append(f"{name}=missing")
    return ",".join(installed)
