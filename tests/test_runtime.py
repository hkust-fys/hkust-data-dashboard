"""Runtime/lifecycle tests with mocked Discord objects: message selection,
no duplicate updater on reconnect, no arbitrary-message edits, continued
operation after a failed provider."""


import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from bot import (
    DASHBOARD_MESSAGE_MARKER,
    DashboardUpdater,
    _find_dashboard_message,
    _payload_fingerprint,
    _resolve_dashboard_message,
    _to_payload,
)
from dashboard.models import DashboardPayload, ImageAsset
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
    def __init__(self, *, archived=False):
        self.sent = []
        self.archived = archived
        self.edits = []

    async def send(self, **kwargs):
        self.sent.append(kwargs)

    async def edit(self, **kwargs):
        self.edits.append(kwargs)
        self.archived = kwargs.get("archived", self.archived)


class _FailOnceThread(_FakeThread):
    def __init__(self):
        super().__init__()
        self.failures = 0

    async def send(self, **kwargs):
        if self.failures == 0:
            self.failures += 1
            raise RuntimeError("archived between resolve and send")
        await super().send(**kwargs)


class _ThreadedMessage(_FakeMessage):
    def __init__(self, thread):
        super().__init__(_FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER)
        self.thread = None
        self._fetched_thread = thread
        self.created_threads = 0

    async def fetch_thread(self):
        return self._fetched_thread

    async def create_thread(self, **_kwargs):
        self.created_threads += 1
        raise AssertionError("must not recreate an existing archived thread")


class _LegacyThreadedMessage(_FakeMessage):
    """discord.py 2.3-shaped message without Message.fetch_thread."""

    def __init__(self, thread):
        super().__init__(_FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER)
        self.thread = None
        self.created_threads = 0
        self.fetched_ids = []

        async def fetch_channel(channel_id):
            self.fetched_ids.append(channel_id)
            return thread

        self.guild = type("Guild", (), {"fetch_channel": staticmethod(fetch_channel)})()

    async def create_thread(self, **_kwargs):
        self.created_threads += 1
        raise AssertionError("must not recreate an existing archived thread")


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


def test_to_payload_distinguishes_missing_map_from_present_failure():
    initializing = _to_payload({"traffic": ([], [], [], None)})
    assert initializing.embeds[0].title == "Traffic map initializing"
    assert not any(
        e.fields and "traffic map unavailable" in e.fields[0].value
        for e in initializing.embeds
    )

    failed = _to_payload(
        {"traffic": ([], [], [], None), "traffic_map": ValueError("capture failed")}
    )
    assert failed.embeds[0].title == "🚦 Traffic news"
    assert any(
        e.fields and "traffic map unavailable" in e.fields[0].value
        for e in failed.embeds
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
    summary = next(embed for embed in payload.embeds if embed.title == "🚦 Traffic news")

    assert "TD detectors" not in summary.description
    # The news timestamp lives only in the footer now (no duplicated line).
    assert "TD traffic news updated" not in summary.description
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

    loop_release = asyncio.Event()

    async def dormant_update_loop(_self, _channel=None):
        await loop_release.wait()

    # Keep the real loop task alive for the idempotent-start assertion, but
    # prevent it from racing the two explicit presenter ticks below.
    monkeypatch.setattr(bot_module.DashboardUpdater, "_update_loop", dormant_update_loop)

    settings = _fake_settings()
    updater = DashboardUpdater(settings)
    # create the session/client and retain the background loop task
    await updater.start(channel)  # noqa: SLF001 - loop runs; stop() cancels it
    first_task = updater._loop_task  # noqa: SLF001
    await updater.start(channel)
    assert updater._loop_task is first_task  # reconnect/start is idempotent
    await updater._tick(channel)  # noqa: SLF001
    first_id = updater._message.id
    await updater._tick(channel)  # noqa: SLF001
    assert updater._message.id == first_id
    assert len(channel.sent) == 1  # exactly one created
    assert updater._message.edits == 1
    loop_release.set()
    await updater.stop()


@pytest.mark.asyncio
async def test_edit_log_keeps_payload_generation_when_new_snapshot_publishes(monkeypatch, caplog):
    import bot as bot_module

    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001 - exercise one presenter tick
    updater._message = _FakeMessage(_FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER)
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    payload = DashboardPayload(files=[ImageAsset("map.png", b"map")])
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))

    edit_started = asyncio.Event()
    release_edit = asyncio.Event()

    async def blocked_edit(_message, _payload, view=None):
        edit_started.set()
        await release_edit.wait()

    monkeypatch.setattr(bot_module, "_apply_payload", blocked_edit)
    tick = asyncio.create_task(updater._tick(object()))  # noqa: SLF001
    await edit_started.wait()
    updater._snapshot = bot_module.CollectionSnapshot({}, 2, 0.0)
    release_edit.set()
    with caplog.at_level("INFO", logger="bot"):
        await tick

    record = next(r.message for r in caplog.records if "dashboard edit succeeded" in r.message)
    assert "collection_generation=1" in record
    assert "collection_generation=2" not in record


@pytest.mark.asyncio
async def test_pending_retained_map_waits_for_settlement_then_updates(monkeypatch, caplog):
    import discord

    import bot as bot_module
    from dashboard.render import traffic_map_filename

    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._message = _FakeMessage(_FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))
    old = b"old-map"
    fresh = b"fresh-map"
    old_payload = DashboardPayload(files=[ImageAsset(traffic_map_filename(old), old)])
    fresh_payload = DashboardPayload(files=[ImageAsset(traffic_map_filename(fresh), fresh)])
    current_payload = old_payload
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: current_payload)
    edits: list[DashboardPayload] = []

    async def record_edit(_message, payload, view=None):
        edits.append(payload)

    monkeypatch.setattr(bot_module, "_apply_payload", record_edit)
    updater._snapshot = bot_module.CollectionSnapshot(
        {"traffic_map": (old, [])}, 1, 0.0,
        stale_providers=frozenset({"traffic_map"}),
    )
    await updater._tick(object())  # noqa: SLF001
    assert not edits

    current_payload = fresh_payload
    updater._snapshot = bot_module.CollectionSnapshot(
        {"traffic_map": (fresh, [])}, 2, 0.0,
        settled_providers=frozenset({"traffic_map"}),
    )
    with caplog.at_level("INFO", logger="bot"):
        await updater._tick(object())  # noqa: SLF001
    assert len(edits) == 1
    assert "collection_generation=2" in caplog.text
    assert traffic_map_filename(fresh) in caplog.text

    # A settled exception retains the old map but cannot suppress later edits.
    current_payload = DashboardPayload(
        files=[ImageAsset(traffic_map_filename(fresh), fresh)],
        embeds=[discord.Embed(title="new source")],
    )
    updater._snapshot = bot_module.CollectionSnapshot(
        {"traffic_map": RuntimeError("capture failed")}, 3, 0.0,
        stale_providers=frozenset({"traffic_map"}),
        settled_providers=frozenset({"traffic_map"}),
    )
    await updater._tick(object())  # noqa: SLF001
    assert len(edits) == 2


@pytest.mark.asyncio
async def test_map_gate_uses_snapshot_paired_with_payload_during_message_resolution(monkeypatch):
    import bot as bot_module
    from dashboard.render import traffic_map_filename

    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    old = b"old-map"
    old_payload = DashboardPayload(files=[ImageAsset(traffic_map_filename(old), old)])
    updater._snapshot = bot_module.CollectionSnapshot(
        {"traffic_map": (old, [])}, 1, 0.0,
        stale_providers=frozenset({"traffic_map"}),
    )
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: old_payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))
    resolving = asyncio.Event()
    release = asyncio.Event()

    async def resolve(_channel, _payload, view=None):
        resolving.set()
        await release.wait()
        return _FakeMessage(_FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER)

    monkeypatch.setattr(bot_module, "_ensure_dashboard_message", resolve)
    edits = []
    monkeypatch.setattr(bot_module, "_apply_payload", lambda *args, **kwargs: edits.append(args[1]))
    tick = asyncio.create_task(updater._tick(object()))  # noqa: SLF001
    await resolving.wait()
    updater._snapshot = bot_module.CollectionSnapshot(
        {"traffic_map": (b"new-map", [])}, 2, 0.0,
        settled_providers=frozenset({"traffic_map"}),
    )
    release.set()
    await tick
    assert not edits


@pytest.mark.asyncio
async def test_updaters_own_distinct_trackers_and_collection_receives_its_tracker(monkeypatch):
    import bot as bot_module

    first = DashboardUpdater(_fake_settings())
    second = DashboardUpdater(_fake_settings())
    assert first.marker_tracker is not second.marker_tracker
    seen = []

    async def collect(client, settings, *, tracker=None):
        seen.append(tracker)
        return {}

    monkeypatch.setattr(bot_module, "collect_all", collect)
    first._running = True  # noqa: SLF001
    first.client = object()
    first._start_collection_if_idle()  # noqa: SLF001
    await asyncio.sleep(0)
    assert seen == [first.marker_tracker]
    await first.stop()
    await second.stop()


@pytest.mark.asyncio
async def test_cold_start_initializing_payload_is_not_suppressed(monkeypatch):
    import discord

    import bot as bot_module

    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._message = _FakeMessage(_FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER)
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    payload = DashboardPayload(embeds=[discord.Embed(title="Traffic map initializing")])
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))
    edits = []

    async def record_edit(_message, edited_payload, view=None):
        edits.append(edited_payload)

    monkeypatch.setattr(bot_module, "_apply_payload", record_edit)
    await updater._tick(object())  # noqa: SLF001
    assert len(edits) == 1


@pytest.mark.asyncio
async def test_updater_without_ffmpeg_does_not_start_live_frame_loop():
    updater = DashboardUpdater(_fake_settings())
    await updater.start()
    assert updater.live_frames._task is None  # noqa: SLF001
    await updater.stop()


def test_payload_fingerprint_changes_when_map_bytes_change():
    from dashboard.models import DashboardPayload, ImageAsset

    first = DashboardPayload(files=[ImageAsset("map.png", b"first")])
    changed = DashboardPayload(files=[ImageAsset("map.png", b"second")])
    assert _payload_fingerprint(first) != _payload_fingerprint(changed)


@pytest.mark.asyncio
async def test_apply_payload_retains_unchanged_content_addressed_attachment():
    import discord

    from bot import _apply_payload

    class Attachment:
        def __init__(self, attachment_id, filename, size):
            self.id = attachment_id
            self.filename = filename
            self.size = size

    class Message:
        def __init__(self, attachments):
            self.attachments = attachments
            self.kwargs = None

        async def edit(self, **kwargs):
            self.kwargs = kwargs
            return self

    old_map = Attachment(1, "traffic-map-old.webp", 3)
    warning = Attachment(2, "hko-warnings-stable.png", len(b"warning"))
    message = Message([old_map, warning])
    payload = DashboardPayload(files=[
        ImageAsset("traffic-map-new.webp", b"new"),
        ImageAsset(warning.filename, b"warning"),
    ])

    edited = await _apply_payload(message, payload)
    attachments = message.kwargs["attachments"]
    assert edited is message
    assert old_map not in attachments
    assert attachments[1] is warning
    assert isinstance(attachments[0], discord.File)
    assert attachments[0].filename == "traffic-map-new.webp"
    attachments[0].close()


def test_dry_run_recognizes_content_addressed_traffic_map_filename():
    import bot as bot_module
    from dashboard.render import traffic_map_filename

    assert bot_module._is_traffic_map_filename(traffic_map_filename(b"map"))
    assert not bot_module._is_traffic_map_filename("traffic-map.webp")
    assert not bot_module._is_traffic_map_filename("traffic-map-not-a-hash.webp")


def test_payload_fingerprint_ignores_render_timestamp_but_keeps_content():
    from datetime import UTC, datetime, timedelta

    import discord

    from dashboard.models import DashboardPayload

    first_embed = discord.Embed(title="Traffic")
    first_embed.timestamp = datetime.now(UTC)
    later_embed = discord.Embed(title="Traffic")
    later_embed.timestamp = first_embed.timestamp + timedelta(seconds=10)
    changed_embed = discord.Embed(title="Traffic changed")
    changed_embed.timestamp = later_embed.timestamp

    assert _payload_fingerprint(DashboardPayload(embeds=[first_embed])) == _payload_fingerprint(
        DashboardPayload(embeds=[later_embed])
    )
    assert _payload_fingerprint(DashboardPayload(embeds=[later_embed])) != _payload_fingerprint(
        DashboardPayload(embeds=[changed_embed])
    )


@pytest.mark.asyncio
async def test_updater_stops_provider_refreshes_before_session_close(monkeypatch):
    import bot as bot_module

    events: list[str] = []

    async def stop_browser():
        events.append("browser")

    async def stop_geometry():
        events.append("geometry")

    async def stop_roads():
        events.append("roads")

    async def stop_transit():
        events.append("transit")

    class Session:
        async def close(self):
            events.append("session")

    monkeypatch.setattr(bot_module.route_geometry_provider, "shutdown_background_refreshes", stop_geometry)
    monkeypatch.setattr(bot_module.tracked_roads_provider, "shutdown_background_refreshes", stop_roads)
    monkeypatch.setattr(bot_module.transit, "shutdown_background_refreshes", stop_transit)
    monkeypatch.setattr(bot_module.maps, "shutdown_gmaps_browser", stop_browser)
    updater = DashboardUpdater(_fake_settings())
    updater.session = Session()

    await updater.stop()

    assert events == ["browser", "geometry", "roads", "transit", "session"]


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
@pytest.mark.parametrize("runner", ["run_dry_run", "run_dev_webhook"])
async def test_one_shot_runners_cleanup_background_resources_on_failure(monkeypatch, runner):
    import bot as bot_module

    events: list[str] = []

    async def fail_collect(*_args, **_kwargs):
        raise RuntimeError("collection failed")

    async def stop_browser():
        events.append("browser")

    async def stop_geometry():
        events.append("geometry")

    async def stop_roads():
        events.append("roads")

    async def stop_transit():
        events.append("transit")

    monkeypatch.setattr(bot_module, "collect_all", fail_collect)
    monkeypatch.setattr(bot_module.maps, "shutdown_gmaps_browser", stop_browser)
    monkeypatch.setattr(bot_module.route_geometry_provider, "shutdown_background_refreshes", stop_geometry)
    monkeypatch.setattr(bot_module.tracked_roads_provider, "shutdown_background_refreshes", stop_roads)
    monkeypatch.setattr(bot_module.transit, "shutdown_background_refreshes", stop_transit)
    settings = replace(
        _fake_settings(),
        dev_webhook="https://discord.com/api/webhooks/placeholder/token",
    )

    with pytest.raises(RuntimeError, match="collection failed"):
        await getattr(bot_module, runner)(settings)
    assert events == ["browser", "geometry", "roads", "transit"]


@pytest.mark.asyncio
async def test_presenter_does_not_wait_for_slow_background_collection(monkeypatch):
    """A presentation tick must not inherit provider latency."""
    import asyncio
    import time

    import bot as bot_module

    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def slow_collect(client, settings):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {"weather": (None, [], None), "traffic": ([], [], [], None)}

    monkeypatch.setattr(bot_module, "collect_all", slow_collect)
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001 - avoid a real HTTP session
    updater.client = object()

    began = time.monotonic()
    await updater._tick()  # noqa: SLF001
    assert time.monotonic() - began < 0.1
    await started.wait()
    assert updater._snapshot is None  # noqa: SLF001

    # A second presentation while the map/provider work is slow must not start
    # a competing Playwright capture/collection.
    await updater._tick()  # noqa: SLF001
    assert calls == 1

    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert updater._snapshot is not None  # noqa: SLF001
    await updater.stop()


@pytest.mark.asyncio
async def test_independent_map_restarts_while_ordinary_collection_is_still_pending(
    monkeypatch,
):
    import bot as bot_module

    collection_started = asyncio.Event()
    collection_release = asyncio.Event()
    map_releases = [asyncio.Event(), asyncio.Event()]
    map_started = [asyncio.Event(), asyncio.Event()]
    collection_calls = 0
    map_calls = 0
    active_maps = 0
    max_active_maps = 0

    async def slow_collect(
        _client, _settings, on_result=None, tracker=None,
        include_traffic_map=True,
    ):
        nonlocal collection_calls
        collection_calls += 1
        assert include_traffic_map is False
        collection_started.set()
        await collection_release.wait()
        return {}

    async def map_from_results(_client, _settings, _results, _tracker):
        nonlocal map_calls, active_maps, max_active_maps
        index = map_calls
        map_calls += 1
        active_maps += 1
        max_active_maps = max(max_active_maps, active_maps)
        map_started[index].set()
        try:
            await map_releases[index].wait()
            return (f"map-{index}".encode(), [])
        finally:
            active_maps -= 1

    monkeypatch.setattr(bot_module, "collect_all", slow_collect)
    monkeypatch.setattr(bot_module, "_fetch_traffic_map_from_results", map_from_results)
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater.client = object()

    await updater._tick()  # noqa: SLF001
    await asyncio.wait_for(collection_started.wait(), timeout=1)
    await asyncio.wait_for(map_started[0].wait(), timeout=1)
    map_releases[0].set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    await updater._tick()  # noqa: SLF001
    await asyncio.wait_for(map_started[1].wait(), timeout=1)
    assert collection_calls == 1
    assert map_calls == 2
    assert max_active_maps == 1

    map_releases[1].set()
    collection_release.set()
    await updater.stop()


@pytest.mark.asyncio
async def test_independent_map_keeps_retained_traffic_and_important_road_overlays(
    monkeypatch,
):
    import bot as bot_module

    affected = [(22.33, 114.22), (22.34, 114.23)]
    important = [[(22.35, 114.24), (22.36, 114.25)]]
    incident = SimpleNamespace(
        latitude=22.33, longitude=114.22,
        near_landmark=None, between_landmark=None,
    )

    class Roads:
        def segments_near(self, keys, latitude=None, longitude=None):
            assert keys == ["road"]
            assert (latitude, longitude) == (22.33, 114.22)
            return [affected]

    captured = {}

    async def fetch_map(_client, **kwargs):
        captured.update(kwargs)
        return (b"map", [])

    groups = s.route_groups()
    monkeypatch.setattr(bot_module.maps, "fetch_traffic_map", fetch_map)
    monkeypatch.setattr(
        bot_module.traffic_provider, "resolve_incident_road_keys",
        lambda *_args, **_kwargs: ["road"],
    )
    monkeypatch.setattr(
        bot_module.road_policy, "important_road_paths", lambda _roads: important
    )

    await bot_module._fetch_traffic_map_from_results(  # noqa: SLF001
        object(), _fake_settings(), {
            "transit": (groups, s.utc(), []),
            "traffic": ([], [incident], [], None),
            "tracked_roads": Roads(),
        }, object(),
    )

    assert captured["groups"] == groups
    assert captured["affected_road_paths"] == [affected]
    assert captured["important_road_paths"] == important


@pytest.mark.asyncio
async def test_old_independent_map_cannot_publish_as_a_newer_generation(monkeypatch):
    import bot as bot_module

    first_collection_done = asyncio.Event()
    second_collection_started = asyncio.Event()
    second_collection_release = asyncio.Event()
    map_started = asyncio.Event()
    map_release = asyncio.Event()
    collection_calls = 0

    async def collect(
        _client, _settings, on_result=None, tracker=None,
        include_traffic_map=True,
    ):
        nonlocal collection_calls
        collection_calls += 1
        assert on_result is not None
        if collection_calls == 1:
            on_result("weather", (None, [], None))
            first_collection_done.set()
            return {"weather": (None, [], None)}
        on_result("traffic", ([], [], [], None))
        second_collection_started.set()
        await second_collection_release.wait()
        return {"traffic": ([], [], [], None)}

    async def old_map(_client, _settings, _results, _tracker):
        map_started.set()
        await map_release.wait()
        return (b"old-generation-map", [])

    monkeypatch.setattr(bot_module, "collect_all", collect)
    monkeypatch.setattr(bot_module, "_fetch_traffic_map_from_results", old_map)
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater.client = object()

    await updater._tick()  # noqa: SLF001
    await asyncio.wait_for(first_collection_done.wait(), timeout=1)
    await asyncio.wait_for(map_started.wait(), timeout=1)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await updater._tick()  # noqa: SLF001
    await asyncio.wait_for(second_collection_started.wait(), timeout=1)
    assert updater._snapshot.generation == 2  # noqa: SLF001

    map_release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert "traffic_map" not in updater._snapshot.results  # noqa: SLF001

    second_collection_release.set()
    await updater.stop()


@pytest.mark.asyncio
async def test_stop_cancels_and_drains_independent_map(monkeypatch):
    import bot as bot_module

    map_started = asyncio.Event()
    map_cancelled = asyncio.Event()

    async def collect(
        _client, _settings, on_result=None, tracker=None,
        include_traffic_map=True,
    ):
        await asyncio.Event().wait()

    async def blocked_map(_client, _settings, _results, _tracker):
        map_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            map_cancelled.set()
            raise

    monkeypatch.setattr(bot_module, "collect_all", collect)
    monkeypatch.setattr(bot_module, "_fetch_traffic_map_from_results", blocked_map)
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater.client = object()

    await updater._tick()  # noqa: SLF001
    await asyncio.wait_for(map_started.wait(), timeout=1)
    await updater.stop()

    assert map_cancelled.is_set()
    assert updater._map_task is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_update_loop_skips_missed_deadlines_instead_of_bursting(monkeypatch):
    import bot as bot_module

    clock = [0.0]
    sleeps: list[float] = []
    updater = DashboardUpdater(replace(_fake_settings(), update_interval_seconds=10))
    updater._running = True  # noqa: SLF001

    async def slow_tick(_channel=None):
        clock[0] += 25
        updater._running = False  # noqa: SLF001

    async def fake_sleep(delay):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(updater, "_tick", slow_tick)
    monkeypatch.setattr(bot_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(bot_module.asyncio, "sleep", fake_sleep)

    await updater._update_loop()  # noqa: SLF001

    assert sleeps == [10]


@pytest.mark.asyncio
async def test_provider_snapshot_publishes_before_slow_map_finishes(monkeypatch):
    """Fast transit is presentable while the single map capture is still running."""
    import asyncio

    import bot as bot_module

    map_release = asyncio.Event()

    async def incremental_collect(client, settings, on_result=None):
        transit_result = (s.route_groups(), s.utc(), [])
        assert on_result is not None
        on_result("transit", transit_result)
        await map_release.wait()
        on_result("traffic_map", (b"png", []))
        return {"transit": transit_result, "traffic_map": (b"png", [])}

    monkeypatch.setattr(bot_module, "collect_all", incremental_collect)
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater.client = object()
    await updater._tick()  # noqa: SLF001
    await asyncio.sleep(0)
    assert updater._snapshot is not None  # noqa: SLF001
    assert "transit" in updater._snapshot.results  # noqa: SLF001
    assert "traffic_map" not in updater._snapshot.results  # noqa: SLF001
    map_release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert "traffic_map" in updater._snapshot.results  # noqa: SLF001
    await updater.stop()


@pytest.mark.asyncio
async def test_stop_cancels_inflight_collection(monkeypatch):
    import asyncio

    import bot as bot_module

    cancelled = asyncio.Event()

    async def slow_collect(client, settings):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(bot_module, "collect_all", slow_collect)
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater.client = object()
    await updater._tick()  # noqa: SLF001
    assert updater._collection_task is not None  # noqa: SLF001
    await updater.stop()
    assert cancelled.is_set()
    assert updater._collection_task is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_collect_all_cancellation_awaits_provider_children(monkeypatch):
    import asyncio

    import bot as bot_module
    from dashboard.providers import tracked_roads as tracked_roads_provider

    cancelled: list[str] = []
    ready = {name: asyncio.Event() for name in ("weather", "map")}

    def blocking(name):
        async def run(*_args, **_kwargs):
            try:
                ready[name].set()
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(name)
                raise

        return run

    class Roads:
        def routes_for_text(self, _text):
            return []

    async def roads(*_args, **_kwargs):
        return Roads()

    async def transit(*_args, **_kwargs):
        return ([], None, [])

    async def traffic(*_args, **_kwargs):
        return ([], [], [], None)

    monkeypatch.setattr(tracked_roads_provider, "fetch_tracked_roads", roads)
    monkeypatch.setattr(bot_module.transit, "fetch_transit_etas", transit)
    monkeypatch.setattr(bot_module.weather_provider, "fetch_weather_conditions", blocking("weather"))
    monkeypatch.setattr(bot_module.traffic_provider, "fetch_traffic_data", traffic)
    monkeypatch.setattr(bot_module.maps, "fetch_traffic_map", blocking("map"))

    operation = asyncio.create_task(bot_module.collect_all(object(), _fake_settings()))
    await asyncio.wait_for(
        asyncio.gather(*(event.wait() for event in ready.values())),
        timeout=1.0,
    )
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert set(cancelled) == {"weather", "map"}


@pytest.mark.asyncio
async def test_collect_all_reuses_tracked_roads_after_timeout_for_important_paths(
    monkeypatch,
):
    """One consumer timing out must not cancel geometry needed by the map."""
    import bot as bot_module
    from dashboard.providers import tracked_roads as tracked_roads_provider
    from dashboard.providers.tracked_roads import TrackedRoads

    cwb_path = ((22.33, 114.22), (22.34, 114.23))
    new_cwb_path = ((22.32, 114.21), (22.33, 114.22))
    unrelated_path = ((22.31, 114.20), (22.32, 114.21))
    roads_table = TrackedRoads(
        paths={
            "clear water bay road": (cwb_path,),
            "new clear water bay road": (new_cwb_path,),
            "lung cheung road": (unrelated_path,),
        }
    )
    roads_started = asyncio.Event()
    release_roads = asyncio.Event()
    captured: dict[str, object] = {}

    async def roads(*_args, **_kwargs):
        roads_started.set()
        await release_roads.wait()
        return roads_table

    async def transit(*_args, **_kwargs):
        return ([], None, [])

    async def weather(*_args, **_kwargs):
        return (None, [], None)

    async def traffic(_client, matched_roads):
        assert matched_roads is not roads_table
        release_roads.set()
        await asyncio.sleep(0)
        return ([], [], [], None)

    async def traffic_map(*_args, **kwargs):
        captured.update(kwargs)
        return (b"map", [])

    monkeypatch.setattr(bot_module, "TRACKED_ROADS_WAIT_SECONDS", 0.01)
    monkeypatch.setattr(tracked_roads_provider, "fetch_tracked_roads", roads)
    monkeypatch.setattr(bot_module.transit, "fetch_transit_etas", transit)
    monkeypatch.setattr(bot_module.weather_provider, "fetch_weather_conditions", weather)
    monkeypatch.setattr(bot_module.traffic_provider, "fetch_traffic_data", traffic)
    monkeypatch.setattr(bot_module.maps, "fetch_traffic_map", traffic_map)

    await asyncio.wait_for(bot_module.collect_all(object(), _fake_settings()), timeout=1)

    assert roads_started.is_set()
    assert captured["important_road_paths"] == [list(cwb_path), list(new_cwb_path)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("latitude", "longitude", "near_landmark", "between_landmark", "expected_anchor"),
    [
        (22.335, 114.26, "", "", (22.335, 114.26)),
        (None, None, "HKUST North Gate", "", "skip"),
        (None, None, "", "", (None, None)),
        (22.335, None, "", "", "skip"),
        (25.0, 114.26, "", "", "skip"),
    ],
)
async def test_collect_all_passes_only_anchored_affected_road_segments(
    monkeypatch,
    latitude,
    longitude,
    near_landmark,
    between_landmark,
    expected_anchor,
):
    """Traffic overlays use coordinates, or the provider's guarded fallback."""
    import bot as bot_module
    from dashboard.models import TrafficIncident

    incident = TrafficIncident(
        "notice-1", "Road closure", "works", "Clear Water Bay Road", "HKUST",
        "outbound", "active", latitude=latitude, longitude=longitude,
        near_landmark=near_landmark, between_landmark=between_landmark,
    )
    calls: list[tuple[list[str], float | None, float | None]] = []
    captured: dict[str, object] = {}

    class Roads:
        def match(self, _text):
            return ["clear water bay road"]

        def segments_near(self, keys, lat, lon):
            calls.append((keys, lat, lon))
            return [[(22.335, 114.26), (22.336, 114.261)]]

    async def roads(*_args, **_kwargs):
        return Roads()

    async def transit(*_args, **_kwargs):
        return ([], None, [])

    async def weather(*_args, **_kwargs):
        return (None, [], None)

    async def traffic(*_args, **_kwargs):
        return ([], [incident], [], None)

    async def traffic_map(*_args, **kwargs):
        captured.update(kwargs)
        return (b"map", [])

    from dashboard.providers import tracked_roads as tracked_roads_provider

    monkeypatch.setattr(tracked_roads_provider, "fetch_tracked_roads", roads)
    monkeypatch.setattr(bot_module.transit, "fetch_transit_etas", transit)
    monkeypatch.setattr(bot_module.weather_provider, "fetch_weather_conditions", weather)
    monkeypatch.setattr(bot_module.traffic_provider, "fetch_traffic_data", traffic)
    monkeypatch.setattr(bot_module.maps, "fetch_traffic_map", traffic_map)

    await bot_module.collect_all(object(), _fake_settings())

    assert captured["important_road_paths"] == []
    if expected_anchor == "skip":
        assert calls == []
        assert captured["affected_road_paths"] == []
    else:
        assert calls == [(["clear water bay road"], *expected_anchor)]
        assert captured["affected_road_paths"] == [[(22.335, 114.26), (22.336, 114.261)]]


@pytest.mark.asyncio
async def test_collect_all_does_not_map_direction_only_tracked_road(monkeypatch):
    import bot as bot_module
    from dashboard.models import TrafficIncident
    from dashboard.providers.tracked_roads import TrackedRoads

    names = {
        "tseung kwan o tunnel": "Tseung Kwan O Tunnel",
        "tseung kwan o tunnel road": "Tseung Kwan O Tunnel Road",
    }
    roads_table = TrackedRoads(
        display_names=names,
        aliases={key: key for key in names},
        road_routes={key: ("12",) for key in names},
        paths={
            key: (((22.33, 114.24), (22.33, 114.245)),) for key in names
        },
    )
    incident = TrafficIncident(
        identifier="tko-road-reopened",
        title="Road Incident",
        description=(
            "The fast lane of Tseung Kwan O Road (Tseung Kwan O Tunnel bound) "
            "near Hing Tin Estate which was closed due to traffic accident is re-opened "
            "to all traffic."
        ),
        road="Tseung Kwan O Road",
        location="Tseung Kwan O Road",
        direction="",
        status="active",
    )
    captured: dict[str, object] = {}

    async def roads(*_args, **_kwargs):
        return roads_table

    async def transit(*_args, **_kwargs):
        return ([], None, [])

    async def weather(*_args, **_kwargs):
        return (None, [], None)

    async def traffic(*_args, **_kwargs):
        return ([], [incident], [], None)

    async def traffic_map(*_args, **kwargs):
        captured.update(kwargs)
        return (b"map", [])

    from dashboard.providers import tracked_roads as tracked_roads_provider

    monkeypatch.setattr(tracked_roads_provider, "fetch_tracked_roads", roads)
    monkeypatch.setattr(bot_module.transit, "fetch_transit_etas", transit)
    monkeypatch.setattr(bot_module.weather_provider, "fetch_weather_conditions", weather)
    monkeypatch.setattr(bot_module.traffic_provider, "fetch_traffic_data", traffic)
    monkeypatch.setattr(bot_module.maps, "fetch_traffic_map", traffic_map)

    await bot_module.collect_all(object(), _fake_settings())

    assert captured["affected_road_paths"] == []


@pytest.mark.asyncio
async def test_collect_all_allows_explicit_short_subroad_for_landmark_notice(monkeypatch):
    """A landmark-only notice may select a named short sub-road, not its parent road."""
    import bot as bot_module
    from dashboard.models import TrafficIncident

    incident = TrafficIncident(
        "IN-26-06242",
        "Traffic incident",
        "Lung Cheung Road flyover is closed",
        "Lung Cheung Road",
        "Choi Hung Estate",
        "Mong Kok-bound",
        "active",
        near_landmark="Choi Hung Estate",
    )
    calls: list[tuple[list[str], float | None, float | None]] = []
    captured: dict[str, object] = {}

    class Roads:
        def match(self, _text):
            return ["lung cheung road flyover", "lung cheung road"]

        def segments_near(self, keys, lat, lon):
            calls.append((keys, lat, lon))
            return [[(22.34, 114.20), (22.341, 114.201)]]

    async def roads(*_args, **_kwargs):
        return Roads()

    async def transit(*_args, **_kwargs):
        return ([], None, [])

    async def weather(*_args, **_kwargs):
        return (None, [], None)

    async def traffic(*_args, **_kwargs):
        return ([], [incident], [], None)

    async def traffic_map(*_args, **kwargs):
        captured.update(kwargs)
        return (b"map", [])

    from dashboard.providers import tracked_roads as tracked_roads_provider

    monkeypatch.setattr(tracked_roads_provider, "fetch_tracked_roads", roads)
    monkeypatch.setattr(bot_module.transit, "fetch_transit_etas", transit)
    monkeypatch.setattr(bot_module.weather_provider, "fetch_weather_conditions", weather)
    monkeypatch.setattr(bot_module.traffic_provider, "fetch_traffic_data", traffic)
    monkeypatch.setattr(bot_module.maps, "fetch_traffic_map", traffic_map)

    await bot_module.collect_all(object(), _fake_settings())

    assert calls == [(["lung cheung road flyover"], None, None)]
    assert captured["affected_road_paths"] == [[(22.34, 114.20), (22.341, 114.201)]]


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


@pytest.mark.asyncio
async def test_updater_fetches_and_unarchives_dashboard_message_thread():
    updater = DashboardUpdater(_fake_settings())
    thread = _FakeThread(archived=True)
    message = _ThreadedMessage(thread)
    updater._message = message  # noqa: SLF001

    await updater._ensure_thread()  # noqa: SLF001

    assert updater._thread is thread  # noqa: SLF001
    assert thread.edits == [{"archived": False}]
    assert message.created_threads == 0


@pytest.mark.asyncio
async def test_updater_unarchives_thread_that_archives_after_startup():
    updater = DashboardUpdater(_fake_settings())
    thread = _FakeThread(archived=True)
    updater._message = _ThreadedMessage(thread)  # noqa: SLF001
    updater._thread = thread  # noqa: SLF001

    await updater._ensure_thread()  # noqa: SLF001

    assert updater._thread is thread  # noqa: SLF001
    assert thread.edits == [{"archived": False}]


@pytest.mark.asyncio
async def test_updater_fetches_archived_thread_with_discord_py_23_shape():
    updater = DashboardUpdater(_fake_settings())
    thread = _FakeThread(archived=True)
    message = _LegacyThreadedMessage(thread)
    updater._message = message  # noqa: SLF001

    await updater._ensure_thread()  # noqa: SLF001

    assert updater._thread is thread  # noqa: SLF001
    assert message.fetched_ids == [message.id]
    assert message.created_threads == 0
    assert thread.edits == [{"archived": False}]


@pytest.mark.asyncio
async def test_alert_snapshot_waits_for_both_incremental_inputs():
    import bot as bot_module

    updater = DashboardUpdater(_fake_settings())
    processed = []

    async def record(results):
        processed.append(results)

    updater._post_alert_events = record  # type: ignore[method-assign]
    updater._snapshot = bot_module.CollectionSnapshot(  # noqa: SLF001
        {"weather": (None, [], None)},
        1,
        0.0,
        settled_providers=frozenset({"weather"}),
    )
    await updater._process_alert_snapshot()  # noqa: SLF001
    assert processed == []
    assert updater._last_alert_generation == 0  # noqa: SLF001

    complete = {
        "weather": (None, [], None),
        "traffic": ([], [], [], None),
    }
    updater._snapshot = bot_module.CollectionSnapshot(  # noqa: SLF001
        complete,
        1,
        0.0,
        settled_providers=frozenset({"weather", "traffic"}),
    )
    await updater._process_alert_snapshot()  # noqa: SLF001
    await updater._process_alert_snapshot()  # noqa: SLF001 - no duplicate

    assert processed == [complete]
    assert updater._last_alert_generation == 1  # noqa: SLF001


@pytest.mark.asyncio
async def test_alert_generation_stays_paired_when_new_snapshot_publishes():
    import bot as bot_module

    updater = DashboardUpdater(_fake_settings())
    first = {"weather": (None, [], None), "traffic": ([], [], [], None)}
    second = {"weather": (None, [], None), "traffic": ([], [], [], "new")}
    updater._snapshot = bot_module.CollectionSnapshot(  # noqa: SLF001
        first,
        1,
        0.0,
        settled_providers=frozenset({"weather", "traffic"}),
    )
    processed = []

    async def publish_during_post(results):
        processed.append(results)
        updater._snapshot = bot_module.CollectionSnapshot(  # noqa: SLF001
            second,
            2,
            0.0,
            settled_providers=frozenset({"weather", "traffic"}),
        )

    updater._post_alert_events = publish_during_post  # type: ignore[method-assign]
    await updater._process_alert_snapshot()  # noqa: SLF001

    assert processed == [first]
    assert updater._last_alert_generation == 1  # noqa: SLF001


@pytest.mark.asyncio
async def test_tick_keeps_completed_alert_generation_when_next_collection_starts(
    monkeypatch,
):
    import bot as bot_module
    from dashboard.models import TrafficIncident

    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._message = _FakeMessage(  # noqa: SLF001
        _FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER
    )
    updater._collection_generation = 1  # noqa: SLF001
    incident = TrafficIncident(
        "transient",
        "Transient TD notice",
        "Present for one completed collection only",
        "Clear Water Bay Road",
        "",
        "",
        "active",
    )
    updater._publish_provider_result(  # noqa: SLF001
        1, "weather", (None, [], None)
    )
    updater._publish_provider_result(  # noqa: SLF001
        1, "traffic", ([], [incident], [], None)
    )
    first_results = updater._snapshot.results  # noqa: SLF001
    processed = []

    def start_next_collection():
        updater._collection_generation = 2  # noqa: SLF001
        updater._publish_provider_result(2, "transit", ([], None, []))  # noqa: SLF001

    async def record(results):
        processed.append(results)

    async def no_op(*_args, **_kwargs):
        return None

    monkeypatch.setattr(updater, "_start_collection_if_idle", start_next_collection)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: DashboardPayload())
    monkeypatch.setattr(updater, "_ensure_thread", no_op)
    monkeypatch.setattr(updater, "_post_alert_events", record)
    monkeypatch.setattr(bot_module, "_apply_payload", no_op)

    await updater._tick(object())  # noqa: SLF001

    assert processed == [first_results]
    assert updater._last_alert_generation == 1  # noqa: SLF001
    assert updater._snapshot.generation == 2  # noqa: SLF001


@pytest.mark.asyncio
async def test_failed_alert_send_is_retained_and_retried_without_duplication():
    from dashboard.models import Roadwork

    updater = DashboardUpdater(_fake_settings())
    flaky = _FailOnceThread()
    updater._thread = flaky  # noqa: SLF001
    baseline = {"weather": (None, [], None), "traffic": ([], [], [], None)}
    active = {
        "weather": (None, [], None),
        "traffic": (
            [],
            [],
            [Roadwork("rw-retry", "Lane closure", "Clear Water Bay Road")],
            None,
        ),
    }
    await updater._post_alert_events(baseline)  # noqa: SLF001 - seed
    await updater._post_alert_events(active)  # noqa: SLF001 - first send fails

    assert updater._thread is None  # noqa: SLF001
    assert len(updater._pending_alert_messages) == 1  # noqa: SLF001
    replacement = _FakeThread()
    updater._thread = replacement  # noqa: SLF001
    await updater._flush_alert_messages()  # noqa: SLF001
    await updater._post_alert_events(active)  # noqa: SLF001 - same state, no duplicate

    assert len(replacement.sent) == 1
    assert "Lane closure" in replacement.sent[0]["content"]
    assert not updater._pending_alert_messages  # noqa: SLF001


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


@pytest.mark.asyncio
async def test_map_starts_google_capture_and_geometry_together(monkeypatch):
    """A slow geometry refresh cannot postpone the required browser capture."""
    import asyncio

    from dashboard import maps
    from dashboard.providers.route_geometry import RouteGeometry

    capture_started = asyncio.Event()
    geometry_started = asyncio.Event()
    release = asyncio.Event()

    async def delayed_capture(**_kwargs):
        capture_started.set()
        await release.wait()
        return object()

    async def delayed_geometry(*_args, **_kwargs):
        geometry_started.set()
        await release.wait()
        return RouteGeometry()

    def rendered(*_args):
        return b"map"

    monkeypatch.setattr(maps, "capture_gmaps_base", delayed_capture)
    monkeypatch.setattr(maps, "fetch_route_geometry", delayed_geometry)
    monkeypatch.setattr(maps, "render_map", rendered)
    operation = asyncio.create_task(maps.fetch_traffic_map(object(), cache_dir="unused"))
    await asyncio.wait_for(
        asyncio.gather(capture_started.wait(), geometry_started.wait()), timeout=0.2
    )
    release.set()
    png, _ = await operation
    assert png == b"map"
