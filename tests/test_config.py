"""Config tests: validation of env vars, defaults, and failure modes."""

import pytest

from dashboard.config import ConfigError, Settings


@pytest.fixture(autouse=True)
def _google_maps_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "test-maps-key")


def test_settings_from_env_valid(monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "abc")
    monkeypatch.setenv("ANNOUNCE_CHANNEL_ID", "12345")
    monkeypatch.delenv("DASHBOARD_MESSAGE_ID", raising=False)
    settings = Settings.from_env(require_keys=True)
    assert settings.discord_token == "abc"
    assert settings.announce_channel_id == 12345
    assert settings.dashboard_message_id is None
    assert settings.update_interval_seconds == 15


def test_settings_missing_required_keys(monkeypatch):
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.delenv("ANNOUNCE_CHANNEL_ID", raising=False)
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        Settings.from_env(require_keys=True)


def test_settings_dry_run_without_keys(monkeypatch):
    monkeypatch.delenv("DISCORD_TOKEN", raising=False)
    monkeypatch.delenv("ANNOUNCE_CHANNEL_ID", raising=False)
    settings = Settings.from_env(require_keys=False)
    assert settings.announce_channel_id == 0


def test_settings_message_id_parsed(monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "t")
    monkeypatch.setenv("ANNOUNCE_CHANNEL_ID", "1")
    monkeypatch.setenv("DASHBOARD_MESSAGE_ID", "987654321")
    settings = Settings.from_env()
    assert settings.dashboard_message_id == 987654321


def test_settings_alert_role_id_parsed(monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "t")
    monkeypatch.setenv("ANNOUNCE_CHANNEL_ID", "1")
    monkeypatch.setenv("ALERT_ROLE_ID", "555666777")
    settings = Settings.from_env()
    assert settings.alert_role_id == 555666777


def test_settings_invalid_channel_id(monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "t")
    monkeypatch.setenv("ANNOUNCE_CHANNEL_ID", "not-a-number")
    with pytest.raises(ConfigError):
        Settings.from_env()


def test_settings_invalid_interval(monkeypatch):
    monkeypatch.setenv("DISCORD_TOKEN", "t")
    monkeypatch.setenv("ANNOUNCE_CHANNEL_ID", "1")
    monkeypatch.setenv("UPDATE_INTERVAL_SECONDS", "14")  # below minimum 15
    with pytest.raises(ConfigError):
        Settings.from_env()
