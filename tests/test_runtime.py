"""Runtime/lifecycle tests with mocked Discord objects: message selection,
no duplicate updater on reconnect, no arbitrary-message edits, continued
operation after a failed provider."""


import asyncio
from dataclasses import replace

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

    class Session:
        async def close(self):
            events.append("session")

    monkeypatch.setattr(bot_module.route_geometry_provider, "shutdown_background_refreshes", stop_geometry)
    monkeypatch.setattr(bot_module.tracked_roads_provider, "shutdown_background_refreshes", stop_roads)
    monkeypatch.setattr(bot_module.maps, "shutdown_gmaps_browser", stop_browser)
    updater = DashboardUpdater(_fake_settings())
    updater.session = Session()

    await updater.stop()

    assert events == ["browser", "geometry", "roads", "session"]


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

    monkeypatch.setattr(bot_module, "collect_all", fail_collect)
    monkeypatch.setattr(bot_module.maps, "shutdown_gmaps_browser", stop_browser)
    monkeypatch.setattr(bot_module.route_geometry_provider, "shutdown_background_refreshes", stop_geometry)
    monkeypatch.setattr(bot_module.tracked_roads_provider, "shutdown_background_refreshes", stop_roads)
    settings = replace(
        _fake_settings(),
        dev_webhook="https://discord.com/api/webhooks/placeholder/token",
    )

    with pytest.raises(RuntimeError, match="collection failed"):
        await getattr(bot_module, runner)(settings)
    assert events == ["browser", "geometry", "roads"]


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

    if expected_anchor == "skip":
        assert calls == []
        assert captured["affected_road_paths"] == []
    else:
        assert calls == [(["clear water bay road"], *expected_anchor)]
        assert captured["affected_road_paths"] == [[(22.335, 114.26), (22.336, 114.261)]]


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
