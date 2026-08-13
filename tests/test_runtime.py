"""Runtime/lifecycle tests with mocked Discord objects: message selection,
no duplicate updater on reconnect, no arbitrary-message edits, continued
operation after a failed provider."""


import pytest

from bot import (
    DASHBOARD_MESSAGE_MARKER,
    DashboardUpdater,
    _find_dashboard_message,
    _resolve_dashboard_message,
    _to_payload,
)
from tests.fixtures import sample_data as s


class _FakeMessage:
    def __init__(self, author, content, id=1):
        self.author = author
        self.content = content
        self.id = id
        self.edits = 0
        self.embeds = []
        self.attachments = []

    async def edit(self, **kwargs):
        self.edits += 1
        self.embeds = kwargs.get("embeds", [])
        self.attachments = kwargs.get("attachments", [])


class _FakeAuthor:
    def __init__(self, bot=False, id=None):
        self.bot = bot
        self.id = id if id is not None else (42 if bot else 7)


class _FakeChannel:
    def __init__(self, messages):
        self.messages = messages
        self.sent = []
        self.guild = type("Guild", (), {"me": _FakeAuthor(bot=True, id=42)})()

    async def history(self, limit=50):
        for m in reversed(self.messages):
            yield m

    async def send(self, **kwargs):
        msg = _FakeMessage(_FakeAuthor(bot=True), kwargs.get("content", ""), id=999)
        self.sent.append(msg)
        return msg

    async def fetch_message(self, message_id):
        return next(message for message in self.messages if message.id == message_id)


class _FakeThread:
    def __init__(self):
        self.sent = []

    async def send(self, **kwargs):
        self.sent.append(kwargs)


@pytest.mark.asyncio
async def test_find_dashboard_message_picks_latest_bot_message():
    channel = _FakeChannel(
        [
            _FakeMessage(_FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=1),
            _FakeMessage(_FakeAuthor(bot=False), "user content", id=2),  # ignored
            _FakeMessage(_FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=3),
        ]
    )
    found = await _find_dashboard_message(channel)
    assert found.id == 3


@pytest.mark.asyncio
async def test_find_dashboard_message_none():
    channel = _FakeChannel([_FakeMessage(_FakeAuthor(bot=False), "user only", id=1)])
    assert await _find_dashboard_message(channel) is None


@pytest.mark.asyncio
async def test_find_dashboard_message_ignores_other_bots():
    channel = _FakeChannel(
        [
            _FakeMessage(_FakeAuthor(bot=True, id=42), DASHBOARD_MESSAGE_MARKER, id=1),
            _FakeMessage(_FakeAuthor(bot=True, id=99), DASHBOARD_MESSAGE_MARKER, id=2),
        ]
    )
    found = await _find_dashboard_message(channel)
    assert found.id == 1


@pytest.mark.asyncio
async def test_find_dashboard_message_ignores_this_bots_other_messages():
    channel = _FakeChannel(
        [
            _FakeMessage(_FakeAuthor(bot=True, id=42), DASHBOARD_MESSAGE_MARKER, id=1),
            _FakeMessage(_FakeAuthor(bot=True, id=42), "command response", id=2),
        ]
    )
    found = await _find_dashboard_message(channel)
    assert found.id == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured",
    [
        _FakeMessage(_FakeAuthor(bot=True, id=99), DASHBOARD_MESSAGE_MARKER, id=9),
        _FakeMessage(_FakeAuthor(bot=True, id=42), "not the exact marker", id=9),
    ],
)
async def test_configured_message_must_be_owned_exact_marker_or_scan(
    configured, caplog
):
    scanned = _FakeMessage(
        _FakeAuthor(bot=True, id=42), DASHBOARD_MESSAGE_MARKER, id=1
    )
    channel = _FakeChannel([scanned, configured])

    found = await _resolve_dashboard_message(channel, 9, channel.guild.me)

    assert found is scanned
    assert "not this bot's exact dashboard marker; scanning" in caplog.text


@pytest.mark.asyncio
async def test_to_payload_isolates_failed_provider():
    results = {
        "transit": (s.route_groups(), s.utc(), []),
        "weather": (None, [], None),
        "traffic": ([], [], [], None),
    }
    payload = _to_payload(results)
    # transit groups render into an embed
    assert len(payload.embeds) >= 1


@pytest.mark.asyncio
async def test_to_payload_surfaces_provider_errors():
    results = {
        "transit": ValueError("KMB down"),
        "weather": ValueError("HKO down"),
        "traffic": ([], [], [], None),
    }
    payload = _to_payload(results)
    # errors rendered into a visible source-status embed
    assert any(
        e.fields and "Source status" in e.fields[0].name for e in payload.embeds
    )


def test_to_payload_uses_news_and_roadwork_times_without_detector_legend():
    from datetime import timedelta

    from dashboard.models import Roadwork

    base = s.utc()
    detector_time = base - timedelta(minutes=30)
    news_time = base - timedelta(minutes=20)
    roadworks_time = base - timedelta(minutes=10)
    payload = _to_payload(
        {
            "traffic": (
                s.traffic_statuses(),
                s.traffic_incidents(),
                [Roadwork("rw", "Roadworks", "CWB")],
                detector_time,
                [],
                {
                    "detectors": detector_time,
                    "traffic_news": news_time,
                    "roadworks": roadworks_time,
                },
            )
        }
    )
    summary = next(embed for embed in payload.embeds if embed.title == "🚦 Traffic summary")

    assert "TD detectors" not in summary.description
    assert f"TD traffic news <t:{int(news_time.timestamp())}:t>" in summary.description
    assert f"TD roadworks <t:{int(roadworks_time.timestamp())}:t>" in summary.description
    assert summary.timestamp == roadworks_time


@pytest.mark.asyncio
async def test_updater_edits_same_message_and_no_duplicate(monkeypatch):
    channel = _FakeChannel([])

    async def fake_collect(client, settings):
        return {
            "transit": (s.route_groups(), s.utc(), []),
            "weather": (s.weather_snapshot(), [], s.utc()),
            "traffic": ([], [], [], None),
        }

    import bot as bot_module

    monkeypatch.setattr(bot_module, "collect_all", fake_collect)

    settings = _fake_settings()
    updater = DashboardUpdater(settings)
    # create the session/client without starting the background loop
    await updater.start(channel)  # noqa: SLF001 - loop runs; stop() cancels it
    first_task = updater._loop_task  # noqa: SLF001
    await updater.start(channel)
    assert updater._loop_task is first_task  # reconnect/start is idempotent
    await updater._tick(channel)  # noqa: SLF001
    first_id = updater._message.id
    await updater._tick(channel)  # noqa: SLF001
    assert updater._message.id == first_id
    assert len(channel.sent) == 1  # exactly one created
    assert updater._message.edits >= 1
    await updater.stop()


@pytest.mark.asyncio
async def test_updater_continues_after_provider_failure(monkeypatch):
    channel = _FakeChannel([])

    async def fake_collect(client, settings):
        return {
            "transit": ValueError("KMB down"),
            "weather": (None, [], None),
            "traffic": ([], [], [], None),
            "cctv": [],
        }

    import bot as bot_module

    monkeypatch.setattr(bot_module, "collect_all", fake_collect)

    settings = _fake_settings()
    updater = DashboardUpdater(settings)
    await updater.start(channel)
    await updater._tick(channel)  # noqa: SLF001
    # still created one message
    assert updater._message is not None
    await updater.stop()


@pytest.mark.asyncio
async def test_updater_posts_new_and_cleared_roadworks_to_dashboard_thread():
    from dashboard.models import Roadwork

    updater = DashboardUpdater(_fake_settings())
    updater._thread = _FakeThread()  # noqa: SLF001
    baseline = {"weather": (None, [], None), "traffic": ([], [], [], None)}
    roadwork = Roadwork(
        "rw-1", "Lane closure near HKUST", "Clear Water Bay Road"
    )

    await updater._post_alert_events(baseline)  # noqa: SLF001 - seed without flood
    assert updater._thread.sent == []  # noqa: SLF001

    active = {
        "weather": (None, [], None),
        "traffic": ([], [], [roadwork], None),
    }
    await updater._post_alert_events(active)  # noqa: SLF001
    await updater._post_alert_events(active)  # noqa: SLF001 - deduplicated
    await updater._post_alert_events(baseline)  # noqa: SLF001 - cleared

    messages = [item["content"] for item in updater._thread.sent]  # noqa: SLF001
    assert len(messages) == 2
    assert "Lane closure near HKUST" in messages[0]
    assert "TD roadworks cleared" in messages[1]


def _fake_settings():
    from dashboard.config import Settings

    return Settings(
        discord_token="",
        announce_channel_id=1,
        update_interval_seconds=15,
        cache_dir=".cache",
    )


def test_camera_playlist_source_time_is_preserved():
    from dashboard.providers.cameras import _latest_segment

    segment, source_time = _latest_segment(
        "#EXTM3U\n"
        "#EXT-X-PROGRAM-DATE-TIME:2026-08-11T12:34:56+08:00\n"
        "media_1.ts\n"
        "#EXT-X-PROGRAM-DATE-TIME:2026-08-11T12:35:02+08:00\n"
        "media_2.ts\n"
    )
    assert segment == "media_2.ts"
    assert source_time is not None
    assert source_time.isoformat() == "2026-08-11T12:35:02+08:00"


def test_runtime_preflight_missing_imageio_is_actionable(monkeypatch):
    from dashboard import runtime
    from dashboard.config import ConfigError

    def missing(_name):
        raise ModuleNotFoundError("imageio_ffmpeg")

    monkeypatch.setattr(runtime.importlib, "import_module", missing)
    with pytest.raises(ConfigError, match=r"python(?:\.exe)?.*-m pip install imageio-ffmpeg"):
        runtime.resolve_ffmpeg_executable()


def test_missing_camera_dependency_warns_but_dashboard_continues(monkeypatch, capsys):
    import bot as bot_module
    from dashboard.config import ConfigError

    def failed_preflight():
        raise ConfigError("use the active interpreter")

    observed = {}

    async def fake_dry_run(settings):
        observed["ffmpeg"] = settings.ffmpeg_executable

    monkeypatch.setattr(bot_module, "startup_preflight", failed_preflight)
    monkeypatch.setattr(bot_module, "run_dry_run", fake_dry_run)
    assert bot_module.main(["--dry-run", "--no-keys"]) == 0
    assert observed["ffmpeg"] is None
    assert "Cameras are disabled; the dashboard will continue" in capsys.readouterr().err
