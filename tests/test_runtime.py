"""Runtime/lifecycle tests with mocked Discord objects: message selection,
no duplicate updater on reconnect, no arbitrary-message edits, continued
operation after a failed provider."""


import pytest

from bot import (
    MESSAGE_MARKER,
    DashboardUpdater,
    _find_dashboard_message,
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
    def __init__(self, bot=False):
        self.bot = bot


class _FakeChannel:
    def __init__(self, messages):
        self.messages = messages
        self.sent = []

    async def history(self, limit=50):
        for m in reversed(self.messages):
            yield m

    async def send(self, **kwargs):
        msg = _FakeMessage(_FakeAuthor(bot=True), kwargs.get("content", ""), id=999)
        self.sent.append(msg)
        return msg


@pytest.mark.asyncio
async def test_find_dashboard_message_only_bot_and_marker():
    channel = _FakeChannel(
        [
            _FakeMessage(_FakeAuthor(bot=True), "other bot content", id=1),
            _FakeMessage(_FakeAuthor(bot=False), MESSAGE_MARKER, id=2),  # user, ignored
            _FakeMessage(_FakeAuthor(bot=True), MESSAGE_MARKER, id=3),
        ]
    )
    found = await _find_dashboard_message(channel)
    assert found.id == 3


@pytest.mark.asyncio
async def test_find_dashboard_message_none():
    channel = _FakeChannel([_FakeMessage(_FakeAuthor(bot=True), "no marker", id=1)])
    assert await _find_dashboard_message(channel) is None


@pytest.mark.asyncio
async def test_to_payload_isolates_failed_provider():
    results = {
        "transit": (s.route_groups(), s.utc(), []),
        "weather": (None, [], None),
        "traffic": ([], [], [], None),
        "cctv": [],
    }
    payload = _to_payload(results, None)
    # transit groups render into an embed
    assert len(payload.embeds) >= 1


@pytest.mark.asyncio
async def test_to_payload_surfaces_provider_errors():
    results = {
        "transit": ValueError("KMB down"),
        "weather": ValueError("HKO down"),
        "traffic": ([], [], [], None),
        "cctv": ValueError("CCTV down"),
    }
    payload = _to_payload(results, None)
    # errors rendered into a visible source-status embed
    assert any("Source status" in e.fields[0].name for e in payload.embeds)


@pytest.mark.asyncio
async def test_updater_edits_same_message_and_no_duplicate(monkeypatch):
    channel = _FakeChannel([])

    async def fake_collect(client, settings):
        return {
            "transit": (s.route_groups(), s.utc(), []),
            "weather": (s.weather_snapshot(), [], s.utc()),
            "traffic": ([], [], [], None),
            "cctv": [],
        }

    def fake_render(*args, **kwargs):
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 16, None

    import bot as bot_module

    monkeypatch.setattr(bot_module, "collect_all", fake_collect)
    monkeypatch.setattr(bot_module, "render_traffic_map", fake_render)

    settings = _fake_settings()
    updater = DashboardUpdater(settings)
    # create the session/client without starting the background loop
    await updater.start(channel)  # noqa: SLF001 - loop runs; stop() cancels it
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

    def fake_render(*args, **kwargs):
        return None, None

    import bot as bot_module

    monkeypatch.setattr(bot_module, "collect_all", fake_collect)
    monkeypatch.setattr(bot_module, "render_traffic_map", fake_render)

    settings = _fake_settings()
    updater = DashboardUpdater(settings)
    await updater.start(channel)
    await updater._tick(channel)  # noqa: SLF001
    # still created one message
    assert updater._message is not None
    await updater.stop()


def _fake_settings():
    from dashboard.config import Settings

    return Settings(
        discord_token="",
        announce_channel_id=1,
        update_interval_seconds=30,
        cache_dir=".cache",
    )
