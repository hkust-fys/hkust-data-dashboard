"""Validated configuration for the dashboard.

All values come from environment variables (loaded from ``.env`` by the entry
point). The dataclass performs strict validation at startup so misconfiguration
fails fast instead of crashing inside the update loop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Keys that are required in production but may be absent in dev/dry-run modes.
_REQUIRED_ENV = ("DISCORD_TOKEN", "ANNOUNCE_CHANNEL_ID")


class ConfigError(ValueError):
    """Raised when the environment configuration is invalid."""


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings."""

    discord_token: str
    announce_channel_id: int
    dashboard_message_id: int | None = None
    dev_webhook: str | None = None
    update_interval_seconds: int = 30
    http_timeout_seconds: float = 10.0
    log_level: str = "INFO"
    cache_dir: str = field(default=".cache")
    alert_role_id: int | None = None

    @classmethod
    def from_env(cls, require_keys: bool = True) -> Settings:
        missing = [name for name in _REQUIRED_ENV if not os.getenv(name)]
        if require_keys and missing:
            raise ConfigError(
                "Missing required environment variables: " + ", ".join(missing)
            )

        announce_raw = os.getenv("ANNOUNCE_CHANNEL_ID", "").strip()
        if require_keys:
            try:
                announce_channel_id = int(announce_raw)
            except ValueError as exc:
                raise ConfigError(
                    f"ANNOUNCE_CHANNEL_ID must be an integer, got {announce_raw!r}"
                ) from exc
        else:
            announce_channel_id = int(announce_raw) if announce_raw else 0

        message_raw = os.getenv("DASHBOARD_MESSAGE_ID", "").strip()
        dashboard_message_id = int(message_raw) if message_raw else None

        alert_raw = os.getenv("ALERT_ROLE_ID", "").strip()
        alert_role_id = int(alert_raw) if alert_raw else None

        log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError(f"LOG_LEVEL must be a standard level, got {log_level!r}")

        return cls(
            discord_token=os.getenv("DISCORD_TOKEN", "").strip(),
            announce_channel_id=announce_channel_id,
            dashboard_message_id=dashboard_message_id,
            dev_webhook=os.getenv("DEV_WEBHOOK", "").strip() or None,
            update_interval_seconds=_env_int(
                "UPDATE_INTERVAL_SECONDS", 30, minimum=10
            ),
            http_timeout_seconds=_env_float("HTTP_TIMEOUT_SECONDS", 10.0),
            log_level=log_level,
            cache_dir=os.getenv("CACHE_DIR", ".cache").strip() or ".cache",
            alert_role_id=alert_role_id,
        )
