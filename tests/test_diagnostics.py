"""Deployment evidence remains useful, bounded, and free of credentials."""

import logging
import sys

from dashboard import diagnostics


def test_formatter_redacts_secrets_queries_headers_and_keeps_exception_locations():
    formatter = diagnostics.DiagnosticFormatter("test-run", ("configured-discord-secret",))
    try:
        try:
            raise ValueError("response content must not appear in the traceback")
        except ValueError as exc:
            raise TimeoutError from exc
    except TimeoutError:
        record = logging.LogRecord(
            "bot", logging.WARNING, __file__, 12,
            "failed token=configured-discord-secret "
            "endpoint=https://example.test/tiles?session=private-query "
            "api_key=private-key\nAuthorization: Bearer private-authorization\n"
            "Cookie: session=private-cookie\n"
            "endpoint=https://discord.com/api/webhooks/123/private-webhook-token",
            (), sys.exc_info(),
        )
    formatted = formatter.format(record)
    for secret in (
        "configured-discord-secret", "private-query", "private-key", "private-authorization",
        "private-cookie", "private-webhook-token", "response content",
    ):
        assert secret not in formatted
    assert "endpoint=https://example.test/tiles" in formatted
    assert "TimeoutError at test_diagnostics.py:" in formatted
    assert "caused_by=ValueError at test_diagnostics.py:" in formatted
    assert "run=test-run" in formatted
    assert "Z WARNING bot pid=" in formatted
    assert "\n" not in formatted


def test_rotating_logs_preserve_startup_evidence_and_redact_both_handlers(monkeypatch, tmp_path):
    installed = {}
    monkeypatch.setattr(diagnostics.logging, "basicConfig", lambda **kwargs: installed.update(kwargs))
    monkeypatch.setattr(diagnostics, "LOG_MAX_BYTES", 512)
    monkeypatch.setattr(diagnostics, "LOG_BACKUP_COUNT", 2)
    discord_logger = logging.getLogger("discord")
    monkeypatch.setattr(discord_logger, "level", logging.NOTSET)
    path = diagnostics.configure_logging("DEBUG", str(tmp_path), secret_values=("private-value",))
    handlers = installed["handlers"]
    try:
        for index in range(30):
            record = logging.LogRecord(
                "bot", logging.INFO, __file__, 50,
                "event=%d credential=private-value endpoint=https://example.test/data?key=hidden",
                (index,), None,
            )
            for handler in handlers:
                handler.handle(record)
        logs = list(path.parent.glob("dashboard.log*"))
        assert len(logs) == 3
        assert all(file.stat().st_size <= 512 for file in logs)
        output = "\n".join(file.read_text(encoding="utf-8") for file in logs)
        assert "private-value" not in output
        assert "hidden" not in output
        assert "pid=" in output and "run=" in output
        assert "event=29" in output
        assert handlers[0].formatter is handlers[1].formatter
        assert not logging.getLogger("discord.http").isEnabledFor(logging.DEBUG)
        assert not logging.getLogger("discord.gateway").isEnabledFor(logging.DEBUG)
    finally:
        for handler in handlers:
            handler.close()


def test_unwritable_log_directory_keeps_console_logging(monkeypatch, tmp_path, caplog):
    blocked = tmp_path / "file-instead-of-directory"
    blocked.write_text("occupied", encoding="utf-8")
    installed = {}
    monkeypatch.setattr(diagnostics.logging, "basicConfig", lambda **kwargs: installed.update(kwargs))
    path = diagnostics.configure_logging("INFO", str(blocked))
    assert path is None
    assert len(installed["handlers"]) == 1
    assert "log file unavailable" in caplog.text
    assert "using stderr" in caplog.text
    installed["handlers"][0].close()


def test_source_fingerprint_identifies_uncommitted_code_without_reading_env(tmp_path):
    (tmp_path / "dashboard").mkdir()
    source = tmp_path / "dashboard" / "module.py"
    source.write_bytes(b"value = 1\n")
    (tmp_path / "bot.py").write_bytes(b"pass\n")
    first = diagnostics.source_fingerprint(tmp_path)
    (tmp_path / ".env").write_text("DISCORD_TOKEN=private", encoding="utf-8")
    source.write_bytes(b"value = 1\r\n")
    assert diagnostics.source_fingerprint(tmp_path) == first
    source.write_bytes(b"value = 2\n")
    assert diagnostics.source_fingerprint(tmp_path) != first
