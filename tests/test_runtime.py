"""Runtime/lifecycle tests with mocked Discord objects: message selection,
no duplicate updater on reconnect, no arbitrary-message edits, continued
operation after a failed provider."""


import asyncio
from collections import deque
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import discord
import pytest

from bot import (
    DASHBOARD_MESSAGE_MARKER,
    DashboardUpdater,
    _bounded_discord,
    _diagnose_alert_ping_capability,
    _find_dashboard_message,
    _payload_fingerprint,
    _resolve_dashboard_message,
    _to_payload,
)
from dashboard.models import DashboardPayload, ImageAsset
from tests.fixtures import sample_data as s


@pytest.fixture(autouse=True)
def _isolate_dashboard_runtime_state(monkeypatch, tmp_path):
    import bot as bot_module

    state_path = tmp_path / "dashboard-runtime-state.json"
    monkeypatch.setattr(
        bot_module,
        "_dashboard_runtime_state_path",
        lambda _cache_dir: state_path,
    )


class _FakeMessage:
    def __init__(self, author, content, id=1, *, created_at=None, delete_error=None):
        self.author = author
        self.content = content
        self.id = id
        self.created_at = created_at
        self.delete_error = delete_error
        self.deleted = False
        self.edits = 0
        self.embeds = []
        self.attachments = []

    async def edit(self, **kwargs):
        self.edits += 1
        self.embeds = kwargs.get("embeds", [])
        self.attachments = kwargs.get("attachments", [])

    async def delete(self):
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted = True


class _FakeAuthor:
    def __init__(self, bot=False, id=None):
        self.bot = bot
        self.id = id if id is not None else (42 if bot else 7)


class _FakeChannel:
    def __init__(self, messages):
        self.messages = messages
        self.sent = []
        self.send_kwargs = []
        self.guild = type("Guild", (), {"me": _FakeAuthor(bot=True, id=42)})()

    async def history(self, limit=50, **_kwargs):
        for m in reversed(self.messages):
            yield m

    async def send(self, **kwargs):
        msg = _FakeMessage(_FakeAuthor(bot=True), kwargs.get("content", ""), id=999)
        msg.nonce = kwargs.get("nonce")
        msg.embeds = kwargs.get("embeds", [])
        self.sent.append(msg)
        self.send_kwargs.append(kwargs)
        return msg

    async def fetch_message(self, message_id):
        return next(message for message in self.messages if message.id == message_id)


class _FakeHTTPResponse:
    def __init__(self, status, payload, *, entered=None, bucket_headers=False):
        self.status = status
        self.reason = "test"
        self.headers = {
            "content-type": "application/json",
            "Via": "1.1 test",
        }
        if bucket_headers:
            self.headers.update(
                {
                    "X-Ratelimit-Bucket": "shared-test-bucket",
                    "X-Ratelimit-Remaining": "0",
                    "X-Ratelimit-Reset-After": "1.0",
                }
            )
        self._payload = payload
        self._entered = entered

    async def __aenter__(self):
        if self._entered is not None:
            await self._entered.wait()
        return self

    async def __aexit__(self, *_args):
        return None

    async def text(self, encoding="utf-8"):
        import json

        return json.dumps(self._payload)


class _FakeHTTPSession:
    def __init__(self, responses, *, repeat_last=False):
        self.responses = deque(responses)
        self.repeat_last = repeat_last
        self._last = None
        self.calls = 0

    def request(self, *_args, **_kwargs):
        self.calls += 1
        if self.responses:
            self._last = self.responses.popleft()
        elif not self.repeat_last or self._last is None:
            raise AssertionError("unexpected fake HTTP request")
        if len(self._last) == 2:
            status, payload = self._last
            entered = None
        else:
            status, payload, entered = self._last
        return _FakeHTTPResponse(status, payload, entered=entered)


class _WindowedHTTPSession:
    def __init__(self):
        self.calls = 0
        self._window_until = 0.0

    def request(self, *_args, **_kwargs):
        import time

        self.calls += 1
        if time.monotonic() < self._window_until:
            return _FakeHTTPResponse(
                429,
                {"retry_after": self._window_until - time.monotonic(), "global": False},
            )
        self._window_until = time.monotonic() + 1.0
        return _FakeHTTPResponse(200, {"ok": True}, bucket_headers=True)


class _HTTPBackedMessage(_FakeMessage):
    def __init__(self, http, channel_id=1):
        super().__init__(_FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=321)
        self._http = http
        self._channel_id = channel_id

    async def edit(self, **kwargs):
        from discord.http import Route

        await self._http.request(
            Route("GET", "/channels/{channel_id}/messages", channel_id=self._channel_id)
        )
        await super().edit(**kwargs)


class _HTTPBackedChannel:
    def __init__(self, http, message):
        self._http = http
        self._message = message
        self.sent = []
        self.guild = type("Guild", (), {"me": _FakeAuthor(bot=True, id=42)})()

    async def fetch_message(self, _message_id):
        from discord.http import Route

        await self._http.request(
            Route("GET", "/channels/{channel_id}/messages", channel_id=1)
        )
        return self._message

    async def history(self, limit=50, **_kwargs):
        await self.fetch_message(self._message.id)
        yield self._message

    async def send(self, **_kwargs):
        raise AssertionError("startup reconciliation must not send")


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
    def __init__(self, thread, *, id=1):
        super().__init__(_FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=id)
        thread.id = id
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

    def __init__(self, thread, *, id=1):
        super().__init__(_FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=id)
        thread.id = id
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


class _CreatableThreadMessage(_FakeMessage):
    def __init__(self, *, id):
        super().__init__(_FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=id)
        self.thread = None
        self.created_threads = 0

    async def create_thread(self, **_kwargs):
        self.created_threads += 1
        self.thread = _FakeThread()
        self.thread.id = self.id
        return self.thread


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
    assert failed.embeds[0].title == "Traffic map unavailable"
    assert failed.embeds[1].title == "🚦 Traffic news"
    assert not any(
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
    assert summary.timestamp is None


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
@pytest.mark.parametrize("age_minutes", [56, 61, 180, 2880])
async def test_updater_keeps_old_dashboard_and_thread(monkeypatch, age_minutes):
    import bot as bot_module

    old_thread = _FakeThread()
    old = _FakeMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        created_at=datetime.now(UTC) - timedelta(minutes=age_minutes),
    )
    channel = _FakeChannel([old])
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._message = old  # noqa: SLF001
    updater._thread = old_thread  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))

    await updater._tick(channel)  # noqa: SLF001

    assert not old.deleted
    assert old.edits == 1
    assert updater._message is old
    assert updater._thread is old_thread
    assert not channel.sent
    assert updater._last_payload_fingerprint == _payload_fingerprint(payload)  # noqa: SLF001


@pytest.mark.asyncio
async def test_dashboard_rollover_tracks_old_message_if_delete_fails(monkeypatch):
    import bot as bot_module

    old = _FakeMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        created_at=datetime.now(UTC) - timedelta(minutes=56),
        delete_error=RuntimeError("cannot delete old dashboard"),
    )
    channel = _FakeChannel([old])
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._message = old  # noqa: SLF001
    updater._thread = _FakeThread()  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))

    # Seed recovery state written by the former rollover implementation.
    updater._dashboard_messages_reconciled = True
    await updater._rollover_dashboard(channel, payload, _payload_fingerprint(payload))

    replacement = channel.sent[0]
    assert updater._message is replacement  # noqa: SLF001
    assert not old.deleted
    assert not replacement.deleted
    assert updater._dashboard_message_key(old) in updater._pending_dashboard_deletes  # noqa: SLF001
    assert updater._last_payload_fingerprint == _payload_fingerprint(payload)  # noqa: SLF001

    # A failed cleanup must not trigger another rollover and accumulate copies.
    updater._snapshot = bot_module.CollectionSnapshot({}, 2, 0.0)
    next_payload = DashboardPayload(files=[ImageAsset("map.png", b"newer-map")])
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: next_payload)
    await updater._tick(channel)  # noqa: SLF001
    assert len(channel.sent) == 1


@pytest.mark.asyncio
async def test_rollover_persists_stale_predecessor_before_clearing_episode(
    monkeypatch, tmp_path
):
    import bot as bot_module

    old = _FakeMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        id=100,
    )
    replacement = _FakeMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        id=200,
    )
    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    updater = DashboardUpdater(settings)
    updater._message = old  # noqa: SLF001
    real_store = bot_module._store_dashboard_runtime_state  # noqa: SLF001
    writes = []

    def fail_first_cleanup_write(store_settings, state):
        writes.append(dict(state))
        if len(writes) == 2:
            return False
        return real_store(store_settings, state)

    monkeypatch.setattr(
        bot_module,
        "_store_dashboard_runtime_state",
        fail_first_cleanup_write,
    )
    assert updater._mark_rollover_send_uncertain()  # noqa: SLF001

    updater._adopt_dashboard_rollover(  # noqa: SLF001
        (replacement, old),
        "replacement-fingerprint",
    )
    persisted = bot_module._load_dashboard_runtime_state(settings)  # noqa: SLF001
    assert persisted["pending_dashboard_message_ids"] == [old.id]
    assert persisted["dashboard_message_id"] == replacement.id
    assert "dashboard_send_nonce" not in persisted
    assert "dashboard_send_predecessor_id" not in persisted

    abandoned_cleanup = updater._dashboard_cleanup_task  # noqa: SLF001
    assert abandoned_cleanup is not None
    abandoned_cleanup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned_cleanup

    class RestartChannel(_FakeChannel):
        async def history(self, limit=50, **_kwargs):
            for message in ():
                yield message

        async def fetch_message(self, message_id):
            return {
                replacement.id: replacement,
                old.id: old,
            }[message_id]

    restarted = DashboardUpdater(settings)
    await restarted._reconcile_dashboard_messages(  # noqa: SLF001
        RestartChannel([replacement])
    )
    cleanup = restarted._dashboard_cleanup_task  # noqa: SLF001
    assert cleanup is not None
    await cleanup
    await asyncio.sleep(0)

    assert old.deleted
    assert restarted._message is replacement  # noqa: SLF001
    assert not restarted._pending_dashboard_delete_ids  # noqa: SLF001


@pytest.mark.asyncio
async def test_dashboard_rollover_treats_missing_old_message_as_success(monkeypatch):
    import bot as bot_module

    class MissingMessage(Exception):
        status = 404

    old = _FakeMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        created_at=datetime.now(UTC) - timedelta(minutes=56),
        delete_error=MissingMessage("already deleted"),
    )
    channel = _FakeChannel([old])
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._message = old  # noqa: SLF001
    updater._thread = _FakeThread()  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))

    # Seed recovery state written by the former rollover implementation.
    updater._dashboard_messages_reconciled = True
    await updater._rollover_dashboard(channel, payload, _payload_fingerprint(payload))

    assert updater._message is channel.sent[0]  # noqa: SLF001
    assert not updater._pending_dashboard_deletes  # noqa: SLF001


@pytest.mark.asyncio
async def test_cancelled_rollover_finishes_single_handoff(monkeypatch):
    import bot as bot_module

    delete_started = asyncio.Event()
    release_delete = asyncio.Event()

    class SlowDeleteMessage(_FakeMessage):
        async def delete(self):
            delete_started.set()
            await release_delete.wait()
            self.deleted = True

    old = SlowDeleteMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        created_at=datetime.now(UTC) - timedelta(minutes=56),
    )
    channel = _FakeChannel([old])
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    fingerprint = _payload_fingerprint(payload)
    updater = DashboardUpdater(_fake_settings())
    updater._message = old  # noqa: SLF001
    monkeypatch.setattr(bot_module, "DASHBOARD_DISCORD_TIMEOUT_SECONDS", 1.0)

    rollover = asyncio.create_task(
        updater._rollover_dashboard(channel, payload, fingerprint)  # noqa: SLF001
    )
    await delete_started.wait()
    rollover.cancel()
    release_delete.set()
    with pytest.raises(asyncio.CancelledError):
        await rollover

    assert old.deleted
    assert len(channel.sent) == 1
    assert updater._message is channel.sent[0]  # noqa: SLF001
    assert updater._last_payload_fingerprint == fingerprint  # noqa: SLF001


@pytest.mark.asyncio
async def test_updater_keeps_dashboard_when_discord_reports_old_edit_cap(monkeypatch):
    import bot as bot_module

    class OldMessageEditCap(Exception):
        code = 30046

    old = _FakeMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        created_at=datetime.now(UTC),
    )
    channel = _FakeChannel([old])
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._message = old  # noqa: SLF001
    updater._thread = _FakeThread()  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))

    async def reject_edit(*_args, **_kwargs):
        raise OldMessageEditCap("old-message edit quota reached")

    monkeypatch.setattr(bot_module, "_apply_payload", reject_edit)

    await updater._tick(channel)  # noqa: SLF001

    assert not old.deleted
    assert not channel.sent
    assert updater._message is old
    assert updater._last_payload_fingerprint is None
    assert updater._dashboard_edit_retry_at > bot_module.time.monotonic()  # noqa: SLF001


@pytest.mark.asyncio
async def test_updater_bounds_discord_edit_retry_wait(monkeypatch):
    import bot as bot_module

    old = _FakeMessage(_FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER)
    channel = _FakeChannel([old])
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._message = old  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))
    monkeypatch.setattr(bot_module, "DASHBOARD_DISCORD_TIMEOUT_SECONDS", 0.01)

    async def blocked_edit(*_args, **_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(bot_module, "_apply_payload", blocked_edit)

    await asyncio.wait_for(updater._tick(channel), timeout=0.25)  # noqa: SLF001

    assert updater._last_payload_fingerprint is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_edit_cooldown_keeps_thread_updates_and_retries_same_message(monkeypatch):
    import bot as bot_module

    class LimitedOnce(_FakeMessage):
        attempts = 0

        async def edit(self, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                response = SimpleNamespace(status=429, reason="Rate limited", headers={
                    "Retry-After": "120",
                })
                raise discord.HTTPException(response, {"code": 30046, "message": "Try later"})
            await super().edit(**kwargs)

    old = LimitedOnce(_FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER)
    channel = _FakeChannel([old])
    updater = DashboardUpdater(_fake_settings())
    updater._running = True
    updater._message = old
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    payload = DashboardPayload(files=[ImageAsset("map.png", b"map")])
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    status_cycles = []

    async def status_updates():
        status_cycles.append(True)

    monkeypatch.setattr(updater, "_process_alert_snapshot", status_updates)
    await updater._tick(channel)
    assert updater._dashboard_edit_retry_at - bot_module.time.monotonic() > 119
    await updater._tick(channel)
    assert old.attempts == 1
    assert len(status_cycles) == 2
    updater._dashboard_edit_retry_at = bot_module.time.monotonic() - 1
    await updater._tick(channel)
    assert old.attempts == 2
    assert old.edits == 1
    assert updater._message is old
    assert not channel.sent and not old.deleted
    assert updater._last_payload_fingerprint == _payload_fingerprint(payload)


@pytest.mark.asyncio
async def test_confirmed_missing_dashboard_recovers_once_without_deleting(monkeypatch):
    import bot as bot_module

    def missing():
        return discord.NotFound(
            SimpleNamespace(status=404, reason="Not found"), {"code": 10008}
        )

    class DeletedMessage(_FakeMessage):
        async def edit(self, **_kwargs):
            raise missing()

        async def delete(self):
            raise AssertionError("recovery must not delete the dashboard")

    class Channel(_FakeChannel):
        async def fetch_message(self, _message_id):
            raise missing()

    old = DeletedMessage(_FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=321)
    channel = Channel([])
    updater = DashboardUpdater(_fake_settings())
    updater._running = True
    updater._message = old
    updater._dashboard_messages_reconciled = True
    updater._persisted_dashboard_message_id = old.id
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    payload = DashboardPayload(embeds=[discord.Embed(title="Current dashboard")])
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_process_alert_snapshot", lambda: asyncio.sleep(0))

    await updater._tick(channel)
    assert updater._message is None
    assert not channel.sent
    await updater._tick(channel)
    await updater._tick(channel)
    assert len(channel.sent) == 1
    assert updater._message is channel.sent[0]
    assert updater._message.id != old.id


@pytest.mark.asyncio
async def test_thread_creation_race_fetches_the_attached_thread():
    thread = _FakeThread()
    thread.id = 321

    class Message(_CreatableThreadMessage):
        fetches = 0

        async def fetch_thread(self):
            self.fetches += 1
            if self.fetches == 1:
                raise discord.NotFound(
                    SimpleNamespace(status=404, reason="Not found"), {"code": 10003}
                )
            return thread

        async def create_thread(self, **kwargs):
            self.created_threads += 1
            assert kwargs == {"name": "status updates"}
            raise discord.HTTPException(
                SimpleNamespace(status=400, reason="Bad request"), {"code": 160004}
            )

    message = Message(id=321)
    updater = DashboardUpdater(_fake_settings())
    updater._message = message
    await updater._ensure_thread()
    assert updater._thread is thread
    assert updater._status_thread_id == message.id
    assert message.fetches == 2
    assert message.created_threads == 1


@pytest.mark.asyncio
async def test_thread_unarchive_retains_discord_returned_object():
    unarchived = _FakeThread()
    unarchived.id = 321

    class Archived(_FakeThread):
        async def edit(self, **kwargs):
            self.edits.append(kwargs)
            return unarchived

    archived = Archived(archived=True)
    message = _ThreadedMessage(archived, id=321)
    message.thread = archived
    updater = DashboardUpdater(_fake_settings())
    updater._message = message
    await updater._ensure_thread()
    await updater._ensure_thread()
    assert updater._thread is unarchived
    assert archived.edits == [{"archived": False}]
    updater._pending_alert_messages.append(("weather update", False))
    await updater._flush_alert_messages()
    assert unarchived.sent[0]["content"] == "weather update"


@pytest.mark.asyncio
async def test_timed_out_accepted_rollover_is_reconciled_before_retry(
    monkeypatch, tmp_path
):
    import bot as bot_module

    class AcceptedThenLostChannel(_FakeChannel):
        async def history(self, limit=50, **_kwargs):
            for message in reversed([*self.messages, *self.sent]):
                yield message

        async def send(self, **kwargs):
            message = await super().send(**kwargs)
            if len(self.sent) == 1:
                await asyncio.Event().wait()
            return message

    old = _FakeMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        id=100,
        created_at=datetime.now(UTC) - timedelta(minutes=56),
    )
    channel = AcceptedThenLostChannel([old])
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    updater = DashboardUpdater(settings)
    updater._running = True  # noqa: SLF001
    updater._message = old  # noqa: SLF001
    updater._thread = _FakeThread()  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))
    monkeypatch.setattr(bot_module, "DASHBOARD_DISCORD_TIMEOUT_SECONDS", 0.01)

    # Seed recovery state written by the former rollover implementation.
    updater._dashboard_messages_reconciled = True
    with pytest.raises(TimeoutError):
        await updater._rollover_dashboard(channel, payload, _payload_fingerprint(payload))
    assert len(channel.sent) == 1
    assert updater._rollover_uncertain_since is not None  # noqa: SLF001

    await updater._tick(channel)  # noqa: SLF001
    cleanup = updater._dashboard_cleanup_task  # noqa: SLF001
    if cleanup is not None:
        await cleanup
        await asyncio.sleep(0)

    assert len(channel.sent) == 1
    assert updater._message is channel.sent[0]  # noqa: SLF001
    assert updater._rollover_uncertain_since is None  # noqa: SLF001
    assert old.deleted


@pytest.mark.asyncio
async def test_reconciliation_never_downgrades_to_a_pending_stale_message():
    old = _FakeMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        id=100,
    )
    replacement = _FakeMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        id=200,
    )
    channel = _FakeChannel([old])
    updater = DashboardUpdater(_fake_settings())
    updater._message = replacement  # noqa: SLF001
    updater._pending_dashboard_deletes[old.id] = old  # noqa: SLF001
    updater._pending_dashboard_delete_ids.add(old.id)  # noqa: SLF001

    await updater._reconcile_dashboard_messages(channel)  # noqa: SLF001
    cleanup = updater._dashboard_cleanup_task  # noqa: SLF001
    if cleanup is not None:
        await cleanup
        await asyncio.sleep(0)

    assert updater._message is replacement  # noqa: SLF001
    assert not replacement.deleted
    assert old.deleted
    assert replacement.id not in updater._pending_dashboard_deletes  # noqa: SLF001
    assert replacement.id not in updater._pending_dashboard_delete_ids  # noqa: SLF001


@pytest.mark.asyncio
async def test_timed_out_accepted_initial_send_is_reconciled_before_retry(
    monkeypatch, tmp_path
):
    import bot as bot_module

    class AcceptedThenLostInitialChannel(_FakeChannel):
        def __init__(self):
            super().__init__([])
            self.history_calls = 0

        async def history(self, limit=50, **_kwargs):
            self.history_calls += 1
            if self.history_calls == 3:
                raise RuntimeError("temporary history failure")
            if self.history_calls >= 4:
                for message in reversed([*self.messages, *self.sent]):
                    yield message

        async def send(self, **kwargs):
            message = await super().send(**kwargs)
            if len(self.sent) == 1:
                await asyncio.Event().wait()
            return message

    channel = AcceptedThenLostInitialChannel()
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    monkeypatch.setattr(bot_module, "DASHBOARD_DISCORD_TIMEOUT_SECONDS", 0.01)

    def prepare(updater):
        updater._running = True  # noqa: SLF001
        updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)  # noqa: SLF001
        monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
        monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
        monkeypatch.setattr(
            updater, "_post_alert_events", lambda _results: asyncio.sleep(0)
        )

    first = DashboardUpdater(settings)
    prepare(first)
    with pytest.raises(TimeoutError):
        await first._tick(channel)  # noqa: SLF001
    assert len(channel.sent) == 1
    assert first._rollover_uncertain_since is not None  # noqa: SLF001
    assert first._dashboard_send_nonce is not None  # noqa: SLF001

    restarted = DashboardUpdater(settings)
    prepare(restarted)
    assert restarted._rollover_uncertain_since is not None  # noqa: SLF001
    assert (  # noqa: SLF001
        restarted._dashboard_send_nonce == first._dashboard_send_nonce
    )

    # A failed strict scan must keep the send blocked, not fall through to the
    # best-effort ensure path and create a second dashboard.
    await restarted._tick(channel)  # noqa: SLF001
    assert len(channel.sent) == 1
    assert restarted._message is None  # noqa: SLF001
    assert restarted._rollover_uncertain_since is not None  # noqa: SLF001

    await restarted._tick(channel)  # noqa: SLF001
    assert len(channel.sent) == 1
    assert restarted._message is channel.sent[0]  # noqa: SLF001
    assert restarted._rollover_uncertain_since is None  # noqa: SLF001
    assert restarted._dashboard_send_nonce is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_initial_send_nonce_deduplicates_before_history_is_visible(
    monkeypatch, tmp_path
):
    import bot as bot_module

    class DelayedHistoryChannel(_FakeChannel):
        def __init__(self):
            super().__init__([])
            self.by_nonce = {}
            self.attempt_nonces = []

        async def history(self, limit=50, **_kwargs):
            for message in self.messages:
                yield message

        async def send(self, **kwargs):
            nonce = kwargs["nonce"]
            self.attempt_nonces.append(nonce)
            if nonce in self.by_nonce:
                return self.by_nonce[nonce]
            message = await super().send(**kwargs)
            self.by_nonce[nonce] = message
            await asyncio.Event().wait()

    channel = DelayedHistoryChannel()
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    monkeypatch.setattr(bot_module, "DASHBOARD_DISCORD_TIMEOUT_SECONDS", 0.01)

    def prepare(updater):
        updater._running = True  # noqa: SLF001
        updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)  # noqa: SLF001
        monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
        monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
        monkeypatch.setattr(
            updater, "_post_alert_events", lambda _results: asyncio.sleep(0)
        )

    first = DashboardUpdater(settings)
    prepare(first)
    with pytest.raises(TimeoutError):
        await first._tick(channel)  # noqa: SLF001

    nonce = first._dashboard_send_nonce  # noqa: SLF001
    assert nonce is not None
    assert channel.attempt_nonces == [nonce]
    assert len(channel.sent) == 1

    restarted = DashboardUpdater(settings)
    prepare(restarted)
    assert restarted._dashboard_send_nonce == nonce  # noqa: SLF001

    await restarted._tick(channel)  # noqa: SLF001

    assert channel.attempt_nonces == [nonce, nonce]
    assert len(channel.sent) == 1
    assert restarted._message is channel.sent[0]  # noqa: SLF001
    assert restarted._rollover_uncertain_since is None  # noqa: SLF001
    assert restarted._dashboard_send_nonce is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_rollover_nonce_deduplicates_before_history_is_visible(
    monkeypatch, tmp_path
):
    import bot as bot_module

    class DelayedHistoryChannel(_FakeChannel):
        def __init__(self, messages):
            super().__init__(messages)
            self.by_nonce = {}
            self.attempt_nonces = []

        async def send(self, **kwargs):
            nonce = kwargs["nonce"]
            self.attempt_nonces.append(nonce)
            if nonce in self.by_nonce:
                return self.by_nonce[nonce]
            message = await super().send(**kwargs)
            self.by_nonce[nonce] = message
            await asyncio.Event().wait()

    old = _FakeMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        id=100,
        created_at=datetime.now(UTC) - timedelta(minutes=56),
    )
    channel = DelayedHistoryChannel([old])
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    updater = DashboardUpdater(settings)
    updater._running = True  # noqa: SLF001
    updater._message = old  # noqa: SLF001
    updater._thread = _FakeThread()  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))
    monkeypatch.setattr(bot_module, "DASHBOARD_DISCORD_TIMEOUT_SECONDS", 0.01)

    # Seed recovery state written by the former rollover implementation.
    updater._dashboard_messages_reconciled = True
    with pytest.raises(TimeoutError):
        await updater._rollover_dashboard(channel, payload, _payload_fingerprint(payload))

    nonce = updater._dashboard_send_nonce  # noqa: SLF001
    assert nonce is not None
    assert channel.attempt_nonces == [nonce]
    assert len(channel.sent) == 1
    assert not old.deleted

    await updater._tick(channel)  # noqa: SLF001

    assert channel.attempt_nonces == [nonce, nonce]
    assert len(channel.sent) == 1
    assert updater._message is channel.sent[0]  # noqa: SLF001
    assert updater._rollover_uncertain_since is None  # noqa: SLF001
    assert updater._dashboard_send_nonce is None  # noqa: SLF001
    assert old.deleted


@pytest.mark.asyncio
async def test_send_nonce_keeps_original_window_and_stops_after_retry_horizon(
    monkeypatch, tmp_path
):
    import bot as bot_module

    class DelayedHistoryChannel(_FakeChannel):
        def __init__(self):
            super().__init__([])
            self.by_nonce = {}
            self.attempt_nonces = []
            self.visible = False

        async def history(self, limit=50, **_kwargs):
            if self.visible:
                for message in self.sent:
                    yield message

        async def send(self, **kwargs):
            nonce = kwargs["nonce"]
            self.attempt_nonces.append(nonce)
            if nonce not in self.by_nonce:
                self.by_nonce[nonce] = await super().send(**kwargs)
            await asyncio.Event().wait()

    now = [1_000.0]
    monkeypatch.setattr(bot_module.time, "time", lambda: now[0])
    monkeypatch.setattr(bot_module, "DASHBOARD_DISCORD_TIMEOUT_SECONDS", 0.01)
    channel = DelayedHistoryChannel()
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    updater = DashboardUpdater(settings)
    updater._running = True  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))

    with pytest.raises(TimeoutError):
        await updater._tick(channel)  # noqa: SLF001
    nonce = updater._dashboard_send_nonce  # noqa: SLF001
    started_at = updater._rollover_uncertain_since  # noqa: SLF001
    assert started_at == 1_000.0

    for retry_at in (1_030.0, 1_060.0):
        now[0] = retry_at
        with pytest.raises(TimeoutError):
            await updater._tick(channel)  # noqa: SLF001
        assert updater._rollover_uncertain_since == started_at  # noqa: SLF001

    now[0] = 1_121.0
    await updater._tick(channel)  # noqa: SLF001
    assert channel.attempt_nonces == [nonce, nonce, nonce]
    assert updater._dashboard_send_nonce == nonce  # noqa: SLF001
    assert updater._rollover_uncertain_since == started_at  # noqa: SLF001

    channel.visible = True
    now[0] = 1_130.0
    await updater._tick(channel)  # noqa: SLF001

    assert len(channel.sent) == 1
    assert updater._message is channel.sent[0]  # noqa: SLF001
    assert updater._dashboard_send_nonce is None  # noqa: SLF001
    assert updater._rollover_uncertain_since is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_dashboard_send_is_never_attempted_without_persisted_recovery_state(
    monkeypatch,
):
    import bot as bot_module

    channel = _FakeChannel([])
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))
    monkeypatch.setattr(
        bot_module,
        "_store_dashboard_runtime_state",
        lambda _settings, _state: False,
    )

    with pytest.raises(RuntimeError, match="create deferred"):
        await updater._tick(channel)  # noqa: SLF001
    assert not channel.sent

    with pytest.raises(RuntimeError, match="retry deferred"):
        await updater._tick(channel)  # noqa: SLF001
    assert not channel.sent


@pytest.mark.asyncio
async def test_first_explicit_send_rejection_starts_a_fresh_episode(monkeypatch):
    import bot as bot_module

    class ExplicitRejection(Exception):
        status = 403

    class RejectOnceChannel(_FakeChannel):
        def __init__(self):
            super().__init__([])
            self.attempts = 0

        async def send(self, **kwargs):
            self.attempts += 1
            bot_module._note_dashboard_send_wire_attempt(  # noqa: SLF001
                "POST",
                SimpleNamespace(path="/api/v10/channels/123/messages"),
            )
            if self.attempts == 1:
                raise ExplicitRejection("missing permissions")
            return await super().send(**kwargs)

    channel = RejectOnceChannel()
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))

    with pytest.raises(ExplicitRejection):
        await updater._tick(channel)  # noqa: SLF001
    assert updater._dashboard_send_nonce is None  # noqa: SLF001
    assert updater._rollover_uncertain_since is None  # noqa: SLF001

    await updater._tick(channel)  # noqa: SLF001
    assert channel.attempts == 2
    assert len(channel.sent) == 1
    assert updater._message is channel.sent[0]  # noqa: SLF001


@pytest.mark.asyncio
async def test_internal_retry_then_rejection_retains_original_send_nonce(monkeypatch):
    import bot as bot_module

    class ExplicitRejection(Exception):
        status = 403

    class CommitThenRetryRejectedChannel(_FakeChannel):
        def __init__(self):
            super().__init__([])
            self.application_nonces = []
            self.wire_nonces = []
            self.committed = None

        async def history(self, limit=50, **_kwargs):
            for message in ():
                yield message

        async def send(self, **kwargs):
            nonce = kwargs["nonce"]
            self.application_nonces.append(nonce)
            if self.committed is None:
                self.committed = _FakeMessage(
                    _FakeAuthor(bot=True),
                    kwargs.get("content", ""),
                    id=1001,
                )
                self.committed.nonce = nonce
                self.sent.append(self.committed)
                for _ in range(2):
                    self.wire_nonces.append(nonce)
                    bot_module._note_dashboard_send_wire_attempt(  # noqa: SLF001
                        "POST",
                        SimpleNamespace(path="/api/v10/channels/123/messages"),
                    )
                raise ExplicitRejection("retry lost permissions")

            self.wire_nonces.append(nonce)
            bot_module._note_dashboard_send_wire_attempt(  # noqa: SLF001
                "POST",
                SimpleNamespace(path="/api/v10/channels/123/messages"),
            )
            assert nonce == self.committed.nonce
            return self.committed

    channel = CommitThenRetryRejectedChannel()
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))

    with pytest.raises(ExplicitRejection):
        await updater._tick(channel)  # noqa: SLF001
    nonce = updater._dashboard_send_nonce  # noqa: SLF001
    assert nonce is not None
    assert channel.wire_nonces == [nonce, nonce]

    await updater._tick(channel)  # noqa: SLF001
    assert channel.application_nonces == [nonce, nonce]
    assert channel.wire_nonces == [nonce, nonce, nonce]
    assert len(channel.sent) == 1
    assert updater._dashboard_send_nonce is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_restart_fetches_confirmed_initial_message_hidden_from_history(
    monkeypatch,
    tmp_path,
):
    import bot as bot_module

    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    first_channel = _FakeChannel([])
    first = DashboardUpdater(settings)
    first._running = True  # noqa: SLF001
    first._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(first, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(first, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(first, "_post_alert_events", lambda _results: asyncio.sleep(0))

    await first._tick(first_channel)  # noqa: SLF001
    confirmed = first._message  # noqa: SLF001
    persisted = bot_module._load_dashboard_runtime_state(settings)  # noqa: SLF001
    assert persisted["dashboard_message_id"] == confirmed.id

    class HiddenConfirmedChannel(_FakeChannel):
        async def history(self, limit=50, **_kwargs):
            for message in ():
                yield message

        async def fetch_message(self, message_id):
            assert message_id == confirmed.id
            return confirmed

    restart_channel = HiddenConfirmedChannel([])
    restarted = DashboardUpdater(settings)
    restarted._running = True  # noqa: SLF001
    restarted._snapshot = bot_module.CollectionSnapshot({}, 2, 0.0)
    monkeypatch.setattr(restarted, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(restarted, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(
        restarted,
        "_post_alert_events",
        lambda _results: asyncio.sleep(0),
    )

    await restarted._tick(restart_channel)  # noqa: SLF001

    assert restarted._message is confirmed  # noqa: SLF001
    assert not restart_channel.sent


@pytest.mark.asyncio
async def test_ambiguous_confirmed_message_fetch_blocks_new_create(
    monkeypatch,
    tmp_path,
):
    import bot as bot_module

    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    bot_module._store_dashboard_runtime_state(  # noqa: SLF001
        settings,
        {
            "announce_channel_id": settings.announce_channel_id,
            "pending_dashboard_message_ids": [],
            "dashboard_message_id": 1001,
        },
    )

    class FailingFetchChannel(_FakeChannel):
        async def history(self, limit=50, **_kwargs):
            for message in ():
                yield message

        async def fetch_message(self, _message_id):
            raise RuntimeError("temporary canonical fetch failure")

    channel = FailingFetchChannel([])
    updater = DashboardUpdater(settings)
    updater._running = True  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))

    await updater._tick(channel)  # noqa: SLF001

    assert not updater._dashboard_messages_reconciled  # noqa: SLF001
    assert updater._persisted_dashboard_message_id == 1001  # noqa: SLF001
    assert not channel.sent


@pytest.mark.asyncio
async def test_unreconciled_fallback_cannot_replace_confirmed_ownership(
    monkeypatch,
    tmp_path,
):
    import bot as bot_module

    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    stale = _FakeMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        id=50,
    )
    confirmed = _FakeMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        id=100,
    )
    bot_module._store_dashboard_runtime_state(  # noqa: SLF001
        settings,
        {
            "announce_channel_id": settings.announce_channel_id,
            "pending_dashboard_message_ids": [stale.id],
            "dashboard_message_id": confirmed.id,
        },
    )

    class DelayedCanonicalChannel(_FakeChannel):
        def __init__(self):
            super().__init__([stale])
            self.canonical_fetches = 0

        async def fetch_message(self, message_id):
            if message_id == confirmed.id:
                self.canonical_fetches += 1
                if self.canonical_fetches == 1:
                    raise RuntimeError("temporary canonical fetch failure")
                return confirmed
            if message_id == stale.id:
                return stale
            return await super().fetch_message(message_id)

    channel = DelayedCanonicalChannel()
    updater = DashboardUpdater(settings)
    updater._running = True  # noqa: SLF001
    updater._message = stale  # noqa: SLF001 - provisional startup fallback
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    thread_calls = []
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(
        updater,
        "_ensure_thread",
        lambda: thread_calls.append(True) or asyncio.sleep(0),
    )
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))

    await updater._tick(channel)  # noqa: SLF001

    persisted = bot_module._load_dashboard_runtime_state(settings)  # noqa: SLF001
    assert persisted["dashboard_message_id"] == confirmed.id
    assert persisted["pending_dashboard_message_ids"] == [stale.id]
    assert not thread_calls
    assert not channel.sent

    await updater._tick(channel)  # noqa: SLF001
    cleanup = updater._dashboard_cleanup_task  # noqa: SLF001
    if cleanup is not None:
        await cleanup
        await asyncio.sleep(0)

    assert channel.canonical_fetches == 2
    assert updater._message is confirmed  # noqa: SLF001
    assert stale.deleted
    assert not channel.sent


@pytest.mark.asyncio
async def test_missing_confirmed_message_allows_fresh_create(monkeypatch, tmp_path):
    import bot as bot_module

    class MissingMessage(Exception):
        status = 404

    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    bot_module._store_dashboard_runtime_state(  # noqa: SLF001
        settings,
        {
            "announce_channel_id": settings.announce_channel_id,
            "pending_dashboard_message_ids": [],
            "dashboard_message_id": 1001,
        },
    )

    class MissingConfirmedChannel(_FakeChannel):
        async def history(self, limit=50, **_kwargs):
            for message in ():
                yield message

        async def fetch_message(self, _message_id):
            raise MissingMessage("confirmed dashboard was deleted")

    channel = MissingConfirmedChannel([])
    updater = DashboardUpdater(settings)
    updater._running = True  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))

    await updater._tick(channel)  # noqa: SLF001

    assert len(channel.sent) == 1
    assert updater._message is channel.sent[0]  # noqa: SLF001
    persisted = bot_module._load_dashboard_runtime_state(settings)  # noqa: SLF001
    assert persisted["dashboard_message_id"] == channel.sent[0].id


@pytest.mark.asyncio
async def test_unobserved_rejection_conservatively_retains_send_nonce(monkeypatch):
    import bot as bot_module

    class ExplicitRejection(Exception):
        status = 403

    class UntracedRejectChannel(_FakeChannel):
        async def send(self, **_kwargs):
            raise ExplicitRejection("transport provenance unavailable")

    channel = UntracedRejectChannel([])
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))

    with pytest.raises(ExplicitRejection):
        await updater._tick(channel)  # noqa: SLF001

    assert updater._dashboard_send_nonce is not None  # noqa: SLF001
    assert updater._rollover_uncertain_since is not None  # noqa: SLF001


@pytest.mark.asyncio
async def test_rejected_legacy_rollover_resumes_editing_original(monkeypatch):
    import bot as bot_module

    class ExplicitRejection(Exception):
        status = 403

    class RejectOnceChannel(_FakeChannel):
        def __init__(self, messages):
            super().__init__(messages)
            self.attempts = 0

        async def send(self, **kwargs):
            self.attempts += 1
            bot_module._note_dashboard_send_wire_attempt(  # noqa: SLF001
                "POST",
                SimpleNamespace(path="/api/v10/channels/123/messages"),
            )
            if self.attempts == 1:
                raise ExplicitRejection("missing permissions")
            return await super().send(**kwargs)

    old = _FakeMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        id=100,
        created_at=datetime.now(UTC) - timedelta(minutes=56),
    )
    channel = RejectOnceChannel([old])
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._message = old  # noqa: SLF001
    updater._thread = _FakeThread()  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))

    # Seed recovery state written by the former rollover implementation.
    updater._dashboard_messages_reconciled = True
    with pytest.raises(ExplicitRejection):
        await updater._rollover_dashboard(channel, payload, _payload_fingerprint(payload))
    assert channel.attempts == 1
    assert updater._dashboard_send_nonce is None  # noqa: SLF001
    assert updater._message is old  # noqa: SLF001
    assert not old.deleted

    await updater._tick(channel)  # noqa: SLF001
    assert channel.attempts == 1
    assert not channel.sent
    assert updater._message is old
    assert old.edits == 1
    assert not old.deleted


@pytest.mark.asyncio
async def test_replay_rejection_cannot_clear_a_prior_ambiguous_send(monkeypatch):
    import bot as bot_module

    class ExplicitRejection(Exception):
        status = 403

    class LostThenRejectedChannel(_FakeChannel):
        def __init__(self):
            super().__init__([])
            self.attempts = 0

        async def history(self, limit=50, **_kwargs):
            for message in self.messages:
                yield message

        async def send(self, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                await super().send(**kwargs)
                await asyncio.Event().wait()
            raise ExplicitRejection("permissions changed after ambiguous send")

    channel = LostThenRejectedChannel()
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))
    monkeypatch.setattr(bot_module, "DASHBOARD_DISCORD_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(TimeoutError):
        await updater._tick(channel)  # noqa: SLF001
    nonce = updater._dashboard_send_nonce  # noqa: SLF001
    started_at = updater._rollover_uncertain_since  # noqa: SLF001

    with pytest.raises(ExplicitRejection):
        await updater._tick(channel)  # noqa: SLF001
    assert updater._dashboard_send_nonce == nonce  # noqa: SLF001
    assert updater._rollover_uncertain_since == started_at  # noqa: SLF001

    monkeypatch.setattr(bot_module, "DASHBOARD_SEND_NONCE_RETRY_SECONDS", 0)
    await updater._tick(channel)  # noqa: SLF001
    assert channel.attempts == 2
    assert len(channel.sent) == 1


@pytest.mark.asyncio
async def test_restarted_hidden_rollover_retires_predecessor_and_edits_current_payload(
    monkeypatch, tmp_path
):
    import bot as bot_module

    class HiddenRolloverChannel(_FakeChannel):
        def __init__(self, messages):
            super().__init__(messages)
            self.by_nonce = {}
            self.attempt_nonces = []

        async def history(self, limit=50, **_kwargs):
            for message in ():
                yield message

        async def send(self, **kwargs):
            nonce = kwargs["nonce"]
            self.attempt_nonces.append(nonce)
            if nonce in self.by_nonce:
                return self.by_nonce[nonce]
            message = await super().send(**kwargs)
            self.by_nonce[nonce] = message
            await asyncio.Event().wait()

    old = _FakeMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        id=100,
        created_at=datetime.now(UTC) - timedelta(minutes=56),
    )
    channel = HiddenRolloverChannel([old])
    first_payload = DashboardPayload(
        embeds=[bot_module.discord.Embed(title="first")]
    )
    current_payload = DashboardPayload(
        embeds=[bot_module.discord.Embed(title="current")]
    )
    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    monkeypatch.setattr(bot_module, "DASHBOARD_DISCORD_TIMEOUT_SECONDS", 0.01)

    first = DashboardUpdater(settings)
    first._running = True  # noqa: SLF001
    first._message = old  # noqa: SLF001
    first._thread = _FakeThread()  # noqa: SLF001
    first._dashboard_messages_reconciled = True  # noqa: SLF001
    first._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(first, "_snapshot_payload", lambda: first_payload)
    monkeypatch.setattr(first, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(first, "_post_alert_events", lambda _results: asyncio.sleep(0))

    # Seed recovery state written by the former rollover implementation.
    first._dashboard_messages_reconciled = True
    with pytest.raises(TimeoutError):
        await first._rollover_dashboard(channel, first_payload, _payload_fingerprint(first_payload))
    nonce = first._dashboard_send_nonce  # noqa: SLF001
    assert nonce is not None
    assert first._dashboard_send_predecessor_id == old.id  # noqa: SLF001
    assert not old.deleted

    restarted = DashboardUpdater(settings)
    restarted._running = True  # noqa: SLF001
    restarted._snapshot = bot_module.CollectionSnapshot({}, 2, 0.0)
    monkeypatch.setattr(restarted, "_snapshot_payload", lambda: current_payload)
    monkeypatch.setattr(restarted, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(
        restarted, "_post_alert_events", lambda _results: asyncio.sleep(0)
    )

    await restarted._tick(channel)  # noqa: SLF001
    cleanup = restarted._dashboard_cleanup_task  # noqa: SLF001
    if cleanup is not None:
        await cleanup
        await asyncio.sleep(0)

    assert channel.attempt_nonces == [nonce, nonce]
    assert len(channel.sent) == 1
    assert restarted._message is channel.sent[0]  # noqa: SLF001
    assert restarted._message.embeds[0].title == "current"  # noqa: SLF001
    assert restarted._last_payload_fingerprint == _payload_fingerprint(  # noqa: SLF001
        current_payload
    )
    assert restarted._dashboard_send_nonce is None  # noqa: SLF001
    assert restarted._dashboard_send_predecessor_id is None  # noqa: SLF001
    assert old.deleted


@pytest.mark.asyncio
async def test_nonce_replay_cannot_replace_a_newer_canonical_dashboard(monkeypatch):
    import bot as bot_module

    nonce = 123456
    replay = _FakeMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        id=1001,
    )
    replay.nonce = nonce
    newer = _FakeMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        id=2000,
    )

    class ReplayChannel(_FakeChannel):
        async def send(self, **kwargs):
            self.send_kwargs.append(kwargs)
            return replay

    channel = ReplayChannel([newer])
    payload = DashboardPayload(embeds=[bot_module.discord.Embed(title="current")])
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._message = newer  # noqa: SLF001
    updater._dashboard_messages_reconciled = True  # noqa: SLF001
    updater._dashboard_send_retry_ready = True  # noqa: SLF001
    updater._dashboard_send_nonce = nonce  # noqa: SLF001
    updater._rollover_uncertain_since = bot_module.time.time()  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))

    await updater._tick(channel)  # noqa: SLF001
    cleanup = updater._dashboard_cleanup_task  # noqa: SLF001
    if cleanup is not None:
        await cleanup
        await asyncio.sleep(0)

    assert updater._message is newer  # noqa: SLF001
    assert not newer.deleted
    assert newer.edits == 1
    assert replay.deleted
    assert updater._last_payload_fingerprint == _payload_fingerprint(payload)  # noqa: SLF001


@pytest.mark.asyncio
async def test_restart_reconciles_uncertain_send_before_creating(monkeypatch, tmp_path):
    import bot as bot_module

    class WindowOnlyChannel(_FakeChannel):
        async def history(self, limit=50, **kwargs):
            if kwargs.get("after") is not None:
                yield replacement

    replacement = _FakeMessage(
        _FakeAuthor(bot=True),
        DASHBOARD_MESSAGE_MARKER,
        id=200,
        created_at=datetime.now(UTC),
    )
    channel = WindowOnlyChannel([])
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    bot_module._store_dashboard_runtime_state(  # noqa: SLF001
        settings,
        {
            "announce_channel_id": settings.announce_channel_id,
            "pending_dashboard_message_ids": [],
            "rollover_uncertain_since": datetime.now(UTC).timestamp(),
        },
    )
    updater = DashboardUpdater(settings)
    updater._running = True  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_ensure_thread", lambda: asyncio.sleep(0))
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))

    await updater._tick(channel)  # noqa: SLF001

    assert updater._message is replacement  # noqa: SLF001
    assert updater._rollover_uncertain_since is None  # noqa: SLF001
    assert replacement.edits == 1
    assert not channel.sent


@pytest.mark.asyncio
async def test_discord_global_gate_self_recovers_if_retry_is_cancelled(
    monkeypatch,
):
    import bot as bot_module

    event = asyncio.Event()
    event.set()
    http = SimpleNamespace(max_ratelimit_timeout=None, _global_over=event)
    monkeypatch.setattr(bot_module, "DISCORD_MAX_RATELIMIT_RETRY_SECONDS", 0.01)

    # The first configuration precedes login; discord.py's static_login then
    # replaces the Event, and on_ready must wrap that replacement again.
    bot_module._configure_discord_http_deadlines(http)  # noqa: SLF001
    login_event = asyncio.Event()
    login_event.set()
    http._global_over = login_event
    bot_module._configure_discord_http_deadlines(http)  # noqa: SLF001

    assert isinstance(
        http._global_over, bot_module._CancellationSafeDiscordGlobalGate  # noqa: SLF001
    )
    assert http._global_over._event is login_event  # noqa: SLF001
    http._global_over.clear()
    assert not http._global_over.is_set()
    await asyncio.sleep(0.02)

    assert http.max_ratelimit_timeout == 0.01
    assert http._global_over.is_set()


@pytest.mark.asyncio
async def test_discord_bucket_admission_allows_ordinary_reset_but_rejects_long_wait():
    """The installed discord.py applies our ceiling before HTTP I/O."""
    from discord.http import Ratelimit

    import bot as bot_module

    loop = asyncio.get_running_loop()
    allowed = Ratelimit(bot_module.DISCORD_MAX_RATELIMIT_RETRY_SECONDS)
    allowed.reset_after = 1.00
    allowed.expires = loop.time() + allowed.reset_after
    started = loop.time()
    async with allowed:
        pass
    assert loop.time() - started >= 0.85

    rejected = Ratelimit(bot_module.DISCORD_MAX_RATELIMIT_RETRY_SECONDS)
    rejected.reset_after = 1.50
    rejected.expires = loop.time() + rejected.reset_after
    with pytest.raises(discord.errors.RateLimited):
        async with rejected:
            pass


@pytest.mark.asyncio
async def test_discord_http_allows_one_second_429_retry_and_rejects_long_retry():
    from discord.http import HTTPClient, Route

    import bot as bot_module

    loop = asyncio.get_running_loop()
    http = HTTPClient(loop)
    http._global_over = asyncio.Event()  # noqa: SLF001
    http._global_over.set()  # noqa: SLF001
    bot_module._configure_discord_http_deadlines(http)  # noqa: SLF001
    route = Route("GET", "/channels/{channel_id}/messages", channel_id=1)

    session = _FakeHTTPSession([
        (429, {"retry_after": 1.0, "global": False}),
        (200, {"ok": True}),
    ])
    http._HTTPClient__session = session  # noqa: SLF001
    assert await asyncio.wait_for(http.request(route), timeout=3.0) == {"ok": True}
    assert session.calls == 2

    rejected_session = _FakeHTTPSession([
        (429, {"retry_after": 1.5, "global": True}),
    ])
    http._HTTPClient__session = rejected_session  # noqa: SLF001
    with pytest.raises(discord.errors.RateLimited):
        await http.request(route)
    assert rejected_session.calls == 1
    assert http._global_over.is_set()  # noqa: SLF001


@pytest.mark.asyncio
async def test_discord_http_global_retry_cancellation_reopens_gate():
    from discord.http import HTTPClient, Route

    import bot as bot_module

    http = HTTPClient(asyncio.get_running_loop())
    http._global_over = asyncio.Event()  # noqa: SLF001
    http._global_over.set()  # noqa: SLF001
    bot_module._configure_discord_http_deadlines(http)  # noqa: SLF001
    http._HTTPClient__session = _FakeHTTPSession(
        [(429, {"retry_after": 0.04, "global": True})],
        repeat_last=True,
    )  # noqa: SLF001
    route = Route("GET", "/channels/{channel_id}/messages", channel_id=1)
    clear_calls = 0
    gate = http._global_over  # noqa: SLF001
    original_clear = gate.clear

    def record_clear():
        nonlocal clear_calls
        clear_calls += 1
        original_clear()
        if clear_calls >= 2:
            second_clear.set()

    gate.clear = record_clear
    second_clear = asyncio.Event()

    request = asyncio.create_task(http.request(route))
    await asyncio.wait_for(second_clear.wait(), timeout=2.0)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert http._HTTPClient__session.calls >= 2  # noqa: SLF001
    await asyncio.wait_for(http._global_over.wait(), timeout=2.0)  # noqa: SLF001


@pytest.mark.asyncio
async def test_discord_bucket_queued_acquisition_cancellation_is_reusable():
    from discord.http import Ratelimit

    bucket = Ratelimit(1.25)
    release_holder = asyncio.Event()

    async def hold_bucket():
        async with bucket:
            await release_holder.wait()

    holder = asyncio.create_task(hold_bucket())
    for _ in range(100):
        if bucket.outgoing:
            break
        await asyncio.sleep(0)
    assert bucket.outgoing == 1
    waiter = asyncio.create_task(bucket.acquire())
    for _ in range(100):
        if bucket._pending_requests:  # noqa: SLF001
            break
        await asyncio.sleep(0)
    assert bucket._pending_requests  # noqa: SLF001
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    release_holder.set()
    await asyncio.wait_for(holder, timeout=1.0)
    async with bucket:
        pass
    assert bucket.outgoing == 0


@pytest.mark.asyncio
async def test_discord_http_request_cancellation_during_response_entry_is_reusable(
    monkeypatch,
):
    from discord.http import HTTPClient, Route

    import bot as bot_module

    monkeypatch.setattr(bot_module, "DASHBOARD_DISCORD_TIMEOUT_SECONDS", 0.05)
    http = HTTPClient(asyncio.get_running_loop())
    http._global_over = asyncio.Event()  # noqa: SLF001
    http._global_over.set()  # noqa: SLF001
    bot_module._configure_discord_http_deadlines(http)  # noqa: SLF001
    entered = asyncio.Event()
    http._HTTPClient__session = _FakeHTTPSession(  # noqa: SLF001
        [(200, {"ok": True}, entered)]
    )
    route = Route("GET", "/channels/{channel_id}/messages", channel_id=1)

    with pytest.raises(asyncio.TimeoutError):
        await _bounded_discord(http.request(route))

    http._HTTPClient__session = _FakeHTTPSession(  # noqa: SLF001
        [(200, {"ok": True})]
    )
    assert await asyncio.wait_for(http.request(route), timeout=1.0) == {"ok": True}


@pytest.mark.asyncio
async def test_discord_cancellation_during_bucket_exit_is_reusable():
    """Cancellation during bucket release does not strand that bucket."""
    from discord.http import Ratelimit

    bucket = Ratelimit(1.25)
    bucket.reset_after = 0.2
    bucket.expires = asyncio.get_running_loop().time() + bucket.reset_after

    async def consume_bucket():
        async with bucket:
            pass

    operation = asyncio.create_task(consume_bucket())
    for _ in range(1000):
        if bucket._sleeping.locked():  # noqa: SLF001
            break
        await asyncio.sleep(0)
    assert bucket._sleeping.locked()  # noqa: SLF001
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    await asyncio.sleep(0.25)
    async with bucket:
        pass
    assert bucket.outgoing == 0


@pytest.mark.asyncio
async def test_persisted_dashboard_startup_fetch_history_then_edits_once(tmp_path):
    """One shared Discord bucket permits persisted fetch, history, then edit."""
    from discord.http import HTTPClient

    import bot as bot_module

    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    http = HTTPClient(asyncio.get_running_loop())
    http._global_over = asyncio.Event()  # noqa: SLF001
    http._global_over.set()  # noqa: SLF001
    bot_module._configure_discord_http_deadlines(http)  # noqa: SLF001
    http._HTTPClient__session = _WindowedHTTPSession()  # noqa: SLF001
    message = _HTTPBackedMessage(http)
    channel = _HTTPBackedChannel(http, message)
    bot_module._store_dashboard_runtime_state(  # noqa: SLF001
        settings,
        {
            "announce_channel_id": settings.announce_channel_id,
            "dashboard_message_id": message.id,
        },
    )
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    updater = DashboardUpdater(settings)
    updater._running = True  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    updater._persisted_dashboard_message_id = message.id  # noqa: SLF001
    updater._message = None  # noqa: SLF001
    updater._ensure_thread = lambda: asyncio.sleep(0)
    updater._post_alert_events = lambda _results: asyncio.sleep(0)
    updater._snapshot_payload = lambda: payload

    await asyncio.wait_for(updater._tick(channel), timeout=7.0)  # noqa: SLF001

    assert updater._message is message  # noqa: SLF001
    assert message.edits == 1
    assert not channel.sent
    assert updater._dashboard_messages_reconciled  # noqa: SLF001


@pytest.mark.asyncio
async def test_expired_nonce_known_canonical_allows_edit_only_and_deduplicates(
    tmp_path,
):
    import bot as bot_module

    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    message = _FakeMessage(
        _FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=321
    )
    channel = _FakeChannel([message])
    bot_module._store_dashboard_runtime_state(  # noqa: SLF001
        settings,
        {
            "announce_channel_id": settings.announce_channel_id,
            "dashboard_message_id": message.id,
            "dashboard_send_nonce": 123,
            "dashboard_send_predecessor_id": 111,
            "rollover_uncertain_since": 1.0,
        },
    )
    first = DashboardPayload(files=[ImageAsset("map.png", b"first")])
    second = DashboardPayload(files=[ImageAsset("map.png", b"second")])
    payloads = iter((first, second, second))
    updater = DashboardUpdater(settings)
    updater._running = True  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    updater._snapshot_payload = lambda: next(payloads)

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("unresolved nonce must remain edit-only")

    updater._ensure_thread = forbidden
    updater._post_alert_events = forbidden
    updater._process_alert_snapshot = forbidden

    await updater._tick(channel)  # noqa: SLF001
    await updater._tick(channel)  # noqa: SLF001
    await updater._tick(channel)  # noqa: SLF001

    assert message.edits == 2
    assert not channel.sent
    assert updater._dashboard_send_nonce == 123  # noqa: SLF001
    assert updater._dashboard_send_predecessor_id == 111  # noqa: SLF001
    assert updater._rollover_uncertain_since == 1.0  # noqa: SLF001
    assert not updater._dashboard_messages_reconciled  # noqa: SLF001
    assert not updater._pending_dashboard_delete_ids  # noqa: SLF001
    persisted = bot_module._load_dashboard_runtime_state(settings)  # noqa: SLF001
    assert persisted["dashboard_message_id"] == message.id
    assert persisted["dashboard_send_nonce"] == 123
    assert persisted["dashboard_send_predecessor_id"] == 111


@pytest.mark.asyncio
async def test_expired_nonce_recovers_proven_successor_and_creates_attached_thread(
    tmp_path,
):
    import time

    import bot as bot_module

    class MissingMessage(Exception):
        status = 404

    started_at = time.time() - 10 * 60
    predecessor_id = 111
    message = _CreatableThreadMessage(id=321)
    message.created_at = datetime.fromtimestamp(started_at + 2, UTC)

    class ProvenSuccessorChannel(_FakeChannel):
        def __init__(self):
            super().__init__([message])
            self.fetched_ids = []

        async def fetch_message(self, message_id):
            self.fetched_ids.append(message_id)
            if message_id == message.id:
                return message
            raise MissingMessage()

    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    bot_module._store_dashboard_runtime_state(  # noqa: SLF001
        settings,
        {
            "announce_channel_id": settings.announce_channel_id,
            "dashboard_message_id": message.id,
            "dashboard_send_nonce": 123,
            "dashboard_send_predecessor_id": predecessor_id,
            "rollover_uncertain_since": started_at,
        },
    )
    updater = DashboardUpdater(settings)
    updater._running = True  # noqa: SLF001
    updater._set_dashboard_message(message)  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    updater._snapshot_payload = lambda: DashboardPayload(
        files=[ImageAsset("map.png", b"current")]
    )
    channel = ProvenSuccessorChannel()

    await updater._tick(channel)  # noqa: SLF001

    assert predecessor_id in channel.fetched_ids
    assert not channel.sent
    assert updater._dashboard_send_nonce is None  # noqa: SLF001
    assert updater._rollover_uncertain_since is None  # noqa: SLF001
    assert updater._dashboard_messages_reconciled  # noqa: SLF001
    assert updater._thread is message.thread  # noqa: SLF001
    assert updater._status_thread_id == message.id  # noqa: SLF001
    assert message.created_threads == 1
    persisted = bot_module._load_dashboard_runtime_state(settings)  # noqa: SLF001
    assert persisted["dashboard_message_id"] == message.id
    assert persisted["status_thread_id"] == message.id
    assert "dashboard_send_nonce" not in persisted
    assert "rollover_uncertain_since" not in persisted


@pytest.mark.asyncio
async def test_expired_nonce_failed_history_edits_known_fallback_only(tmp_path):
    import bot as bot_module

    class FailedHistoryChannel(_FakeChannel):
        async def history(self, **_kwargs):
            raise RuntimeError("history temporarily unavailable")
            yield  # pragma: no cover

    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    message = _FakeMessage(
        _FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=321
    )
    bot_module._store_dashboard_runtime_state(  # noqa: SLF001
        settings,
        {
            "announce_channel_id": settings.announce_channel_id,
            "dashboard_message_id": message.id,
            "dashboard_send_nonce": 123,
            "dashboard_send_predecessor_id": 111,
            "rollover_uncertain_since": 1.0,
        },
    )
    updater = DashboardUpdater(settings)
    updater._running = True  # noqa: SLF001
    updater._message = message  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    updater._snapshot_payload = lambda: DashboardPayload(
        files=[ImageAsset("map.png", b"fallback")]
    )
    updater._ensure_thread = lambda: (_ for _ in ()).throw(
        AssertionError("thread must remain suppressed")
    )
    updater._process_alert_snapshot = lambda: (_ for _ in ()).throw(
        AssertionError("alerts must remain suppressed")
    )
    channel = FailedHistoryChannel([message])

    await updater._tick(channel)  # noqa: SLF001

    assert updater._message is message  # noqa: SLF001
    assert message.edits == 1
    assert not channel.sent
    assert updater._dashboard_send_nonce == 123  # noqa: SLF001
    assert not updater._dashboard_messages_reconciled  # noqa: SLF001


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("edit failed"),
        type("Missing", (Exception,), {"status": 404})(),
        type("Unknown", (Exception,), {"code": 30046})(),
    ],
)
async def test_expired_nonce_edit_failure_cannot_rollover_or_send(tmp_path, failure):
    import bot as bot_module

    class FailingMessage(_FakeMessage):
        async def edit(self, **_kwargs):
            raise failure

    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    message = FailingMessage(
        _FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=321
    )
    bot_module._store_dashboard_runtime_state(  # noqa: SLF001
        settings,
        {
            "announce_channel_id": settings.announce_channel_id,
            "dashboard_message_id": message.id,
            "dashboard_send_nonce": 123,
            "dashboard_send_predecessor_id": 111,
            "rollover_uncertain_since": 1.0,
        },
    )
    updater = DashboardUpdater(settings)
    updater._running = True  # noqa: SLF001
    updater._message = message  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    updater._last_payload_fingerprint = "old-fingerprint"  # noqa: SLF001
    updater._snapshot_payload = lambda: DashboardPayload(
        files=[ImageAsset("map.png", b"new")]
    )
    channel = _FakeChannel([message])

    await updater._tick(channel)  # noqa: SLF001

    assert not channel.sent
    expected = None if getattr(failure, "status", None) == 404 else "old-fingerprint"
    assert updater._last_payload_fingerprint == expected
    assert updater._dashboard_send_nonce == 123  # noqa: SLF001
    assert updater._rollover_uncertain_since == 1.0  # noqa: SLF001
    assert not updater._dashboard_messages_reconciled  # noqa: SLF001


@pytest.mark.asyncio
async def test_old_dashboard_reuses_its_attached_thread(monkeypatch, tmp_path):
    import bot as bot_module

    thread = _FakeThread()
    thread.id = 100
    old = _ThreadedMessage(thread, id=100)
    old.created_at = datetime.now(UTC) - timedelta(minutes=56)

    class ThreadedRolloverChannel(_FakeChannel):
        async def send(self, **kwargs):
            message = _CreatableThreadMessage(id=999)
            message.nonce = kwargs.get("nonce")
            self.sent.append(message)
            self.send_kwargs.append(kwargs)
            return message

    channel = ThreadedRolloverChannel([old])
    payload = DashboardPayload(files=[ImageAsset("map.png", b"fresh-map")])
    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    updater = DashboardUpdater(settings)
    updater._running = True  # noqa: SLF001
    updater._message = old  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot({}, 1, 0.0)
    monkeypatch.setattr(updater, "_snapshot_payload", lambda: payload)
    monkeypatch.setattr(updater, "_post_alert_events", lambda _results: asyncio.sleep(0))

    await updater._tick(channel)  # noqa: SLF001

    assert not old.deleted
    assert not channel.sent
    assert updater._message is old
    assert old.edits == 1
    assert updater._thread is thread
    assert updater._status_thread_id == old.id
    assert old.created_threads == 0


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
async def test_pending_retained_map_does_not_suppress_changed_dashboard(monkeypatch, caplog):
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
    old_payload = DashboardPayload(
        files=[ImageAsset(traffic_map_filename(old), old)],
        embeds=[discord.Embed(title="old source")],
    )
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
    assert len(edits) == 1

    # A replacement capture can be pending for several ticks.  Changed
    # non-map data must still be presented with the retained last-good map.
    current_payload = DashboardPayload(
        files=[ImageAsset(traffic_map_filename(old), old)],
        embeds=[discord.Embed(title="changed source")],
    )
    updater._snapshot = bot_module.CollectionSnapshot(
        {"traffic_map": (old, [])}, 2, 0.0,
        stale_providers=frozenset({"traffic_map"}),
    )
    await updater._tick(object())  # noqa: SLF001
    assert len(edits) == 2
    assert edits[-1].embeds[0].title == "changed source"

    current_payload = fresh_payload
    updater._snapshot = bot_module.CollectionSnapshot(
        {"traffic_map": (fresh, [])}, 2, 0.0,
        settled_providers=frozenset({"traffic_map"}),
    )
    with caplog.at_level("INFO", logger="bot"):
        await updater._tick(object())  # noqa: SLF001
    assert len(edits) == 3
    # The log is paired with the snapshot that supplied this payload (gen 2),
    # even though the next ordinary collection has not yet settled.
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
    assert len(edits) == 4


@pytest.mark.asyncio
async def test_snapshot_payload_pairing_survives_message_resolution(monkeypatch):
    import bot as bot_module
    from dashboard.render import traffic_map_filename

    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._dashboard_messages_reconciled = True  # noqa: SLF001
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

    async def resolve(_channel, _payload, view=None, **_kwargs):
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
    assert len(edits) == 1
    # The payload was captured before the await and must remain the payload
    # presented by this tick, even though a newer snapshot became available.
    assert edits[0] is old_payload


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


@pytest.mark.asyncio
async def test_apply_payload_reuploads_warning_when_only_cdn_thumbnail_remains():
    import time

    from bot import _apply_payload

    filename = "hko-warnings-0123456789ab.png"
    expires = f"{int(time.time()) + 600:x}"
    live_url = (
        f"https://cdn.discordapp.com/attachments/123/456/{filename}"
        f"?ex={expires}&is=0&hm=test"
    )
    current = discord.Embed()
    current.set_thumbnail(url=live_url)
    desired = discord.Embed()
    desired.set_thumbnail(url=f"attachment://{filename}")

    class Message:
        channel = SimpleNamespace(id=123)
        attachments = []
        embeds = [current]

        async def edit(self, **kwargs):
            self.kwargs = kwargs
            return self

    message = Message()
    payload = DashboardPayload(
        embeds=[desired], files=[ImageAsset(filename, b"same-warning-strip")]
    )

    await _apply_payload(message, payload)

    uploaded = message.kwargs["attachments"]
    assert len(uploaded) == 1
    assert isinstance(uploaded[0], discord.File)
    assert uploaded[0].filename == filename
    uploaded[0].close()
    assert message.kwargs["embeds"][0].thumbnail.url == f"attachment://{filename}"
    assert payload.embeds[0].thumbnail.url == f"attachment://{filename}"


@pytest.mark.asyncio
async def test_apply_payload_reuploads_warning_for_expired_cdn_thumbnail():
    from bot import _apply_payload

    filename = "hko-warnings-0123456789ab.png"
    expired_url = (
        f"https://cdn.discordapp.com/attachments/123/456/{filename}"
        "?ex=1&is=0&hm=test"
    )
    current = discord.Embed()
    current.set_thumbnail(url=expired_url)
    desired = discord.Embed()
    desired.set_thumbnail(url=f"attachment://{filename}")

    class Message:
        channel = SimpleNamespace(id=123)
        attachments = []
        embeds = [current]

        async def edit(self, **kwargs):
            self.kwargs = kwargs
            return self

    message = Message()
    await _apply_payload(
        message,
        DashboardPayload(
            embeds=[desired], files=[ImageAsset(filename, b"warning-strip")]
        ),
    )

    uploaded = message.kwargs["attachments"]
    assert len(uploaded) == 1
    assert isinstance(uploaded[0], discord.File)
    uploaded[0].close()





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
async def test_real_collection_wrapper_keeps_incremental_updates_and_independent_map(
    monkeypatch,
):
    """The production wrapper must expose the capabilities the updater inspects."""
    import bot as bot_module

    collection_started = asyncio.Event()
    map_started = asyncio.Event()
    release = asyncio.Event()
    received = {}
    weather = (None, [], None)
    pipeline_timeout = bot_module.pipeline.TRACKED_ROADS_WAIT_SECONDS

    async def collect(_client, _settings, **kwargs):
        received.update(kwargs)
        callback = kwargs.get("on_result")
        if callback is not None:
            callback("weather", weather)
        collection_started.set()
        await release.wait()
        return {"weather": weather}

    async def fetch_map(_client, _settings, _results, _tracker):
        map_started.set()
        await release.wait()
        return (None, [])

    # Patch the pipeline, not bot.collect_all: the real compatibility wrapper
    # and the updater's signature-based capability discovery remain exercised.
    monkeypatch.setattr(bot_module.pipeline, "collect_all", collect)
    monkeypatch.setattr(bot_module, "_fetch_traffic_map_from_results", fetch_map)
    monkeypatch.setattr(bot_module, "TRACKED_ROADS_WAIT_SECONDS", 0.25)
    updater = DashboardUpdater(_fake_settings())
    updater._running = True
    updater.client = object()

    try:
        await updater._tick()
        await asyncio.wait_for(collection_started.wait(), timeout=1)
        assert received["include_traffic_map"] is False
        assert received["tracker"] is updater.marker_tracker
        assert callable(received["on_result"])
        assert received["tracked_roads_wait_seconds"] == 0.25
        assert pipeline_timeout == bot_module.pipeline.TRACKED_ROADS_WAIT_SECONDS
        assert updater._snapshot.results["weather"] == weather
        assert not updater._collection_task.done()
        await asyncio.wait_for(map_started.wait(), timeout=1)
        assert not updater._map_task.done()
    finally:
        release.set()
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
        source="TD", related_reports=(),
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
async def test_late_independent_map_merges_into_newest_collection(monkeypatch):
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
    assert updater._snapshot.results["traffic_map"] == (  # noqa: SLF001
        b"old-generation-map", []
    )

    second_collection_release.set()
    await updater.stop()


@pytest.mark.asyncio
async def test_replaced_independent_map_completion_cannot_overwrite_current_task():
    import bot as bot_module

    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._collection_generation = 2  # noqa: SLF001
    updater._snapshot = bot_module.CollectionSnapshot(
        {"weather": (None, [], None)}, 2, 0.0
    )
    stale = asyncio.create_task(asyncio.sleep(0, result=(b"stale-map", [])))
    current = asyncio.create_task(asyncio.sleep(0, result=(b"current-map", [])))
    await asyncio.gather(stale, current)
    updater._map_task = current  # noqa: SLF001
    updater._map_generation = 1  # noqa: SLF001

    updater._map_finished(stale, 1)  # noqa: SLF001
    assert updater._snapshot.results.get("traffic_map") is None  # noqa: SLF001
    assert updater._map_task is current  # noqa: SLF001

    updater._map_finished(current, 1)  # noqa: SLF001
    assert updater._snapshot.results["traffic_map"] == (  # noqa: SLF001
        b"current-map", []
    )


@pytest.mark.asyncio
async def test_collection_fallback_keeps_returned_non_map_after_independent_map():
    updater = DashboardUpdater(_fake_settings())
    updater._running = True  # noqa: SLF001
    updater._collection_generation = 1  # noqa: SLF001

    # The independent map completion may publish before an alternate
    # collector returns its aggregate dictionary.  The fallback must retain
    # that map while still publishing providers absent from settled_providers.
    updater._publish_provider_result(  # noqa: SLF001
        0, "traffic_map", (b"independent-map", []), independent_map=True
    )
    collection = asyncio.create_task(
        asyncio.sleep(
            0,
            result={
                "traffic_map": (b"returned-map", []),
                "weather": (None, [], None),
            },
        )
    )
    await collection
    updater._collection_task = collection  # noqa: SLF001
    updater._collection_finished(collection)  # noqa: SLF001

    assert updater._snapshot.results["traffic_map"] == (  # noqa: SLF001
        b"independent-map", []
    )
    assert updater._snapshot.results["weather"] == (None, [], None)  # noqa: SLF001
    assert updater._snapshot.settled_providers == frozenset({  # noqa: SLF001
        "traffic_map", "weather"
    })


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

    async def traffic(_client, matched_roads, **_kwargs):
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
        (None, None, "", "", "skip"),
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
    """Traffic overlays consume resolved sections and never re-expand ambiguous news."""
    from dataclasses import replace

    import bot as bot_module
    from dashboard.models import TrafficIncident
    from dashboard.providers import traffic_location

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

    async def enriched(_client, items, _roads, _lines, _cache):
        assert items == [incident]
        paths = () if expected_anchor == "skip" else (((22.335, 114.26), (22.336, 114.261)),)
        return [replace(incident, affected_paths=paths, location_resolution="section" if paths else "unresolved")]

    monkeypatch.setattr(traffic_location, "enrich_incident_locations", enriched)
    await bot_module.collect_all(object(), _fake_settings())

    assert captured["important_road_paths"] == []
    assert calls == []
    if expected_anchor == "skip":
        assert calls == []
        assert captured["affected_road_paths"] == []
    else:
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
async def test_collect_all_unresolved_landmark_does_not_guess_short_subroad(monkeypatch):
    """An unresolved landmark never expands to an entire guessed sub-road."""
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

    assert calls == []
    assert captured["affected_road_paths"] == []


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
async def test_weather_thread_updates_disable_all_mentions():
    from dashboard.models import WeatherWarning

    updater = DashboardUpdater(replace(_fake_settings(), alert_role_id=123456))
    updater._thread = _FakeThread()  # noqa: SLF001
    baseline = {"weather": (None, [], None), "traffic": ([], [], [], None)}
    warning = WeatherWarning(code="WRAINB", name="Black Rainstorm @everyone")
    active = {
        "weather": (None, [warning], s.utc()),
        "traffic": ([], [], [], None),
    }

    await updater._post_alert_events(baseline)  # noqa: SLF001
    await updater._post_alert_events(active)  # noqa: SLF001

    sent = updater._thread.sent  # noqa: SLF001
    assert len(sent) == 1
    assert "<@&123456>" not in sent[0]["content"]
    allowed = sent[0]["allowed_mentions"]
    assert allowed.everyone is False
    assert allowed.users is False
    assert allowed.roles is False
    assert allowed.replied_user is False


@pytest.mark.asyncio
async def test_traffic_thread_allows_only_intended_configured_role():
    from dashboard.models import TrafficIncident
    from dashboard.providers.tracked_roads import fallback_roads

    role_id = 123456
    updater = DashboardUpdater(replace(_fake_settings(), alert_role_id=role_id))
    updater._thread = _FakeThread()  # noqa: SLF001
    baseline = {
        "weather": (None, [], None),
        "traffic": ([], [], [], None),
        "tracked_roads": fallback_roads(),
    }
    important = TrafficIncident(
        "cwb-alert",
        "CWB incident",
        "source markup @everyone <@999> <@&888>",
        "Clear Water Bay Road",
        "",
        "Sai Kung bound",
        "ACTIVE",
    )
    ordinary = TrafficIncident(
        "lung-cheung-update",
        "Lung Cheung incident",
        "another @everyone <@999> <@&888>",
        "Lung Cheung Road",
        "",
        "eastbound",
        "ACTIVE",
    )
    active = {
        **baseline,
        "traffic": ([], [important, ordinary], [], None),
    }

    await updater._post_alert_events(baseline)  # noqa: SLF001
    await updater._post_alert_events(active)  # noqa: SLF001

    sent = updater._thread.sent  # noqa: SLF001
    assert len(sent) == 2
    important_send, ordinary_send = sent
    assert f"<@&{role_id}>" in important_send["content"]
    important_allowed = important_send["allowed_mentions"]
    assert important_allowed.everyone is False
    assert important_allowed.users is False
    assert [role.id for role in important_allowed.roles] == [role_id]
    assert ordinary_send["allowed_mentions"].roles is False
    assert ordinary_send["allowed_mentions"].everyone is False
    assert ordinary_send["allowed_mentions"].users is False


@pytest.mark.asyncio
async def test_updater_fetches_and_unarchives_dashboard_message_thread(caplog):
    updater = DashboardUpdater(_fake_settings())
    thread = _FakeThread(archived=True)
    message = _ThreadedMessage(thread)
    updater._message = message  # noqa: SLF001

    with caplog.at_level("INFO", logger="bot"):
        await updater._ensure_thread()  # noqa: SLF001

    assert updater._thread is thread  # noqa: SLF001
    assert thread.edits == [{"archived": False}]
    assert message.created_threads == 0
    assert "message_id=1 thread_id=1" in caplog.text
    assert "via=starter_id_fetch unarchived=True" in caplog.text


@pytest.mark.asyncio
async def test_thread_failure_diagnostics_identify_operation_and_retained_queue(caplog):
    class Forbidden(Exception):
        status = 403
        code = 50013

    message = _CreatableThreadMessage(id=123)

    async def forbidden(**_kwargs):
        raise Forbidden("missing permissions")

    message.create_thread = forbidden
    updater = DashboardUpdater(replace(_fake_settings(), announce_channel_id=123))
    updater._message = message
    updater._pending_alert_messages.append(("private queued message text", False))
    await updater._ensure_thread()
    assert "POST /channels/123/messages/123/threads" in caplog.text
    assert "via=created error=Forbidden status=403 code=50013" in caplog.text
    assert "timeout_s=4.0 pending_alerts=1" in caplog.text
    assert "private queued message text" not in caplog.text
    assert updater._thread is None
    assert len(updater._pending_alert_messages) == 1


@pytest.mark.asyncio
async def test_thread_send_logs_retry_and_delivery_without_message_text(caplog):
    updater = DashboardUpdater(replace(_fake_settings(), announce_channel_id=123))
    thread = _FailOnceThread()
    updater._message = _ThreadedMessage(thread, id=321)
    updater._thread = thread
    updater._pending_alert_messages.append(("private queued message text", False))
    with caplog.at_level("INFO", logger="bot"):
        await updater._flush_alert_messages()
        await updater._ensure_thread()
        await updater._flush_alert_messages()
    assert "POST /channels/321/messages failed" in caplog.text
    assert "pending_alerts=1" in caplog.text
    assert "status update delivered channel_id=123 message_id=321 thread_id=321" in caplog.text
    assert "remaining_alerts=0" in caplog.text
    assert "private queued message text" not in caplog.text
    assert not updater._pending_alert_messages


def test_dashboard_health_logs_expiry_missing_icons_recovery_and_bounded_heartbeat(monkeypatch, caplog):
    import bot as bot_module
    from dashboard.models import TrafficMapResult, WeatherWarning

    now = datetime.now(UTC)
    monotonic_time = [100.0]
    monkeypatch.setattr(bot_module.time, "monotonic", lambda: monotonic_time[0])
    updater = DashboardUpdater(_fake_settings())
    results = {
        "traffic_map": TrafficMapResult(
            b"map", now - timedelta(seconds=5), base_updated_at=now - timedelta(seconds=75),
        ),
        "weather": (None, [WeatherWarning("WHOT", "Very Hot Weather Warning")], now),
    }
    updater._snapshot = bot_module.CollectionSnapshot(results, 1, 0.0)
    with caplog.at_level("INFO", logger="bot"):
        updater._log_dashboard_health(updater._snapshot_payload())
        updater._log_dashboard_health(updater._snapshot_payload())
        assert len(caplog.records) == 1
        assert "map_state=base_expired" in caplog.text
        assert "missing_warning_icons=('WHOT',) warning_strip=False" in caplog.text
        results["traffic_map"] = TrafficMapResult(b"map", now, base_updated_at=now)
        updater._log_dashboard_health(updater._snapshot_payload())
        assert len(caplog.records) == 2
        assert "map_state=available" in caplog.records[-1].message
        monotonic_time[0] += 301
        updater._log_dashboard_health(updater._snapshot_payload())
        assert len(caplog.records) == 3


def test_dashboard_permissions_log_only_missing_effective_permissions(caplog):
    from bot import _diagnose_dashboard_permissions

    permissions = SimpleNamespace(
        view_channel=True, send_messages=True, embed_links=True, attach_files=False,
        read_message_history=True, create_public_threads=False, send_messages_in_threads=True,
    )
    channel = SimpleNamespace(
        id=123, guild=SimpleNamespace(id=456, me=SimpleNamespace(id=789)),
        permissions_for=lambda _member: permissions,
    )
    _diagnose_dashboard_permissions(channel)
    assert "channel_id=123 guild_id=456 bot_id=789" in caplog.text
    assert "missing=['attach_files', 'create_public_threads']" in caplog.text


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
async def test_updater_rejects_persisted_sibling_thread_after_rollover(tmp_path):
    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    thread = _FakeThread()
    thread.id = 321
    first = DashboardUpdater(settings)
    first._message = _ThreadedMessage(thread, id=321)  # noqa: SLF001

    await first._ensure_thread()  # noqa: SLF001

    fetched_ids = []

    async def fetch_channel(channel_id):
        fetched_ids.append(channel_id)
        return thread

    replacement = _CreatableThreadMessage(id=999)
    replacement.guild = SimpleNamespace(fetch_channel=fetch_channel)
    restarted = DashboardUpdater(settings)
    restarted._message = replacement  # noqa: SLF001

    await restarted._ensure_thread()  # noqa: SLF001

    assert fetched_ids == [replacement.id]
    assert restarted._thread is replacement.thread  # noqa: SLF001
    assert restarted._thread is not thread  # noqa: SLF001
    assert restarted._status_thread_id == replacement.id  # noqa: SLF001
    assert replacement.created_threads == 1


@pytest.mark.asyncio
async def test_pending_alert_retry_moves_to_replacement_dashboard_thread():
    updater = DashboardUpdater(_fake_settings())
    old_message = _CreatableThreadMessage(id=100)
    old_thread = _FailOnceThread()
    old_thread.id = old_message.id
    old_message.thread = old_thread
    updater._set_dashboard_message(old_message)  # noqa: SLF001
    updater._thread = old_thread  # noqa: SLF001
    updater._status_thread_id = old_thread.id  # noqa: SLF001
    updater._pending_alert_messages.append(("queued status", False))  # noqa: SLF001

    await updater._flush_alert_messages()  # noqa: SLF001

    assert updater._thread is None  # noqa: SLF001
    assert len(updater._pending_alert_messages) == 1  # noqa: SLF001

    replacement = _CreatableThreadMessage(id=200)
    updater._set_dashboard_message(replacement)  # noqa: SLF001
    await updater._ensure_thread()  # noqa: SLF001
    await updater._flush_alert_messages()  # noqa: SLF001

    assert updater._thread is replacement.thread  # noqa: SLF001
    assert replacement.thread.sent[0]["content"] == "queued status"
    assert not updater._pending_alert_messages  # noqa: SLF001


def test_updater_ignores_non_object_runtime_state(tmp_path):
    import bot as bot_module

    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    bot_module._dashboard_runtime_state_path(settings.cache_dir).write_text(  # noqa: SLF001
        "[]", encoding="utf-8"
    )

    updater = DashboardUpdater(settings)

    assert updater._status_thread_id is None  # noqa: SLF001
    assert not updater._pending_dashboard_delete_ids  # noqa: SLF001


def test_failed_thread_state_write_is_retried(monkeypatch, tmp_path):
    import bot as bot_module

    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    updater = DashboardUpdater(settings)
    thread = _FakeThread()
    thread.id = 321
    outcomes = iter([False, True])
    writes = []

    def store(_settings, state):
        writes.append(state)
        return next(outcomes)

    monkeypatch.setattr(bot_module, "_store_dashboard_runtime_state", store)

    updater._remember_status_thread(thread)  # noqa: SLF001
    assert updater._persisted_status_thread_id is None  # noqa: SLF001
    updater._remember_status_thread(thread)  # noqa: SLF001

    assert len(writes) == 2
    assert updater._persisted_status_thread_id == thread.id  # noqa: SLF001


@pytest.mark.asyncio
async def test_restart_keeps_persisted_thread_for_same_dashboard(tmp_path):
    import bot as bot_module

    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    thread = _FakeThread()
    message = _ThreadedMessage(thread, id=321)
    bot_module._store_dashboard_runtime_state(settings, {  # noqa: SLF001
        "announce_channel_id": settings.announce_channel_id,
        "dashboard_message_id": message.id, "status_thread_id": message.id,
    })
    updater = DashboardUpdater(settings)
    updater._set_dashboard_message(message)  # noqa: SLF001
    await updater._reconcile_dashboard_messages(_FakeChannel([message]))  # noqa: SLF001
    await updater._ensure_thread()  # noqa: SLF001
    stored = bot_module._load_dashboard_runtime_state(settings)  # noqa: SLF001
    assert stored["status_thread_id"] == message.id
    assert updater._thread is thread  # noqa: SLF001
    assert message.created_threads == 0


@pytest.mark.asyncio
async def test_updater_reconciles_interrupted_rollover_without_new_copy():
    old = _FakeMessage(
        _FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=100
    )
    replacement = _FakeMessage(
        _FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=200
    )
    channel = _FakeChannel([old, replacement])
    updater = DashboardUpdater(_fake_settings())
    updater._message = old  # noqa: SLF001
    updater._last_payload_fingerprint = "belongs-to-old"  # noqa: SLF001

    await updater._reconcile_dashboard_messages(channel)  # noqa: SLF001
    cleanup = updater._dashboard_cleanup_task  # noqa: SLF001
    assert cleanup is not None
    await cleanup
    await asyncio.sleep(0)

    assert updater._message is replacement  # noqa: SLF001
    assert updater._last_payload_fingerprint is None  # noqa: SLF001
    assert old.deleted
    assert len(channel.sent) == 0
    assert not updater._pending_dashboard_deletes  # noqa: SLF001


@pytest.mark.asyncio
async def test_updater_recovers_persisted_stale_message_outside_history(tmp_path):
    import bot as bot_module

    class HiddenStaleChannel(_FakeChannel):
        def __init__(self, messages, hidden):
            super().__init__(messages)
            self.hidden = hidden

        async def fetch_message(self, message_id):
            if message_id in self.hidden:
                return self.hidden[message_id]
            return await super().fetch_message(message_id)

    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    old = _FakeMessage(
        _FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=100
    )
    current = _FakeMessage(
        _FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=200
    )
    bot_module._store_dashboard_runtime_state(  # noqa: SLF001
        settings,
        {
            "announce_channel_id": settings.announce_channel_id,
            "pending_dashboard_message_ids": [old.id],
        },
    )
    channel = HiddenStaleChannel([current], {old.id: old})
    updater = DashboardUpdater(settings)
    updater._message = current  # noqa: SLF001

    await updater._reconcile_dashboard_messages(channel)  # noqa: SLF001
    cleanup = updater._dashboard_cleanup_task  # noqa: SLF001
    assert cleanup is not None
    await cleanup
    await asyncio.sleep(0)

    assert old.deleted
    assert not updater._pending_dashboard_delete_ids  # noqa: SLF001


@pytest.mark.asyncio
async def test_persisted_stale_fetch_failure_retries_next_reconciliation(tmp_path):
    import bot as bot_module

    class FailOnceStaleChannel(_FakeChannel):
        def __init__(self, messages, stale):
            super().__init__(messages)
            self.stale = stale
            self.fetch_attempts = 0

        async def fetch_message(self, message_id):
            if message_id == self.stale.id:
                self.fetch_attempts += 1
                if self.fetch_attempts == 1:
                    raise RuntimeError("temporary fetch failure")
                return self.stale
            return await super().fetch_message(message_id)

    settings = replace(_fake_settings(), cache_dir=str(tmp_path))
    stale = _FakeMessage(
        _FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=100
    )
    current = _FakeMessage(
        _FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=200
    )
    bot_module._store_dashboard_runtime_state(  # noqa: SLF001
        settings,
        {
            "announce_channel_id": settings.announce_channel_id,
            "pending_dashboard_message_ids": [stale.id],
        },
    )
    channel = FailOnceStaleChannel([current], stale)
    updater = DashboardUpdater(settings)
    updater._message = current  # noqa: SLF001

    await updater._reconcile_dashboard_messages(channel)  # noqa: SLF001
    assert not updater._dashboard_messages_reconciled  # noqa: SLF001
    assert stale.id in updater._pending_dashboard_delete_ids  # noqa: SLF001

    await updater._reconcile_dashboard_messages(channel)  # noqa: SLF001
    cleanup = updater._dashboard_cleanup_task  # noqa: SLF001
    assert cleanup is not None
    await cleanup
    await asyncio.sleep(0)

    assert channel.fetch_attempts == 2
    assert stale.deleted
    assert not updater._pending_dashboard_delete_ids  # noqa: SLF001


@pytest.mark.asyncio
async def test_reconciliation_does_not_delete_same_message_fetched_twice():
    resolved = _FakeMessage(
        _FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=200
    )
    history_copy = _FakeMessage(
        _FakeAuthor(bot=True), DASHBOARD_MESSAGE_MARKER, id=200
    )
    channel = _FakeChannel([history_copy])
    updater = DashboardUpdater(_fake_settings())
    updater._message = resolved  # noqa: SLF001

    await updater._reconcile_dashboard_messages(channel)  # noqa: SLF001
    await asyncio.sleep(0)

    assert updater._message is history_copy  # noqa: SLF001
    assert not resolved.deleted
    assert not history_copy.deleted
    assert updater._dashboard_cleanup_task is None  # noqa: SLF001
    assert not updater._pending_dashboard_deletes  # noqa: SLF001


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
    assert replacement.sent[0]["allowed_mentions"].roles is False
    assert not updater._pending_alert_messages  # noqa: SLF001


@pytest.mark.asyncio
async def test_failed_ping_send_retains_role_permission_for_retry():
    from dashboard.models import TrafficIncident
    from dashboard.providers.tracked_roads import fallback_roads

    role_id = 654321
    updater = DashboardUpdater(replace(_fake_settings(), alert_role_id=role_id))
    updater._thread = _FailOnceThread()  # noqa: SLF001
    baseline = {
        "weather": (None, [], None),
        "traffic": ([], [], [], None),
        "tracked_roads": fallback_roads(),
    }
    incident = TrafficIncident(
        "cwb-retry",
        "CWB incident",
        "lane closed",
        "Clear Water Bay Road",
        "",
        "Sai Kung bound",
        "ACTIVE",
    )
    active = {**baseline, "traffic": ([], [incident], [], None)}

    await updater._post_alert_events(baseline)  # noqa: SLF001
    await updater._post_alert_events(active)  # noqa: SLF001
    assert updater._thread is None  # noqa: SLF001
    assert len(updater._pending_alert_messages) == 1  # noqa: SLF001

    replacement = _FakeThread()
    updater._thread = replacement  # noqa: SLF001
    await updater._flush_alert_messages()  # noqa: SLF001
    await updater._post_alert_events(active)  # noqa: SLF001

    assert len(replacement.sent) == 1
    sent = replacement.sent[0]
    assert f"<@&{role_id}>" in sent["content"]
    assert [role.id for role in sent["allowed_mentions"].roles] == [role_id]
    assert not updater._pending_alert_messages  # noqa: SLF001


def test_alert_ping_diagnostic_warns_without_stopping_service(caplog):
    role_id = 654321
    role = SimpleNamespace(mentionable=False)
    member = object()
    guild = SimpleNamespace(
        me=member,
        get_role=lambda requested: role if requested == role_id else None,
    )

    class Channel:
        def __init__(self):
            self.guild = guild

        def permissions_for(self, requested_member):
            assert requested_member is member
            return SimpleNamespace(mention_everyone=False)

    settings = replace(_fake_settings(), alert_role_id=role_id)

    assert not _diagnose_alert_ping_capability(Channel(), settings)
    assert "traffic role pings will not notify" in caplog.text
    assert "server administrator" in caplog.text
    assert str(role_id) not in caplog.text


@pytest.mark.parametrize(
    ("mentionable", "mention_everyone"),
    [(True, False), (False, True)],
)
def test_alert_ping_diagnostic_accepts_either_discord_permission(
    mentionable, mention_everyone
):
    role_id = 654321
    member = object()
    guild = SimpleNamespace(
        me=member,
        get_role=lambda requested: (
            SimpleNamespace(mentionable=mentionable) if requested == role_id else None
        ),
    )
    channel = SimpleNamespace(
        guild=guild,
        permissions_for=lambda requested: (
            SimpleNamespace(mention_everyone=mention_everyone)
            if requested is member
            else None
        ),
    )
    settings = replace(_fake_settings(), alert_role_id=role_id)

    assert _diagnose_alert_ping_capability(channel, settings)


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


def test_missing_camera_dependency_warns_but_dashboard_continues(monkeypatch, caplog):
    import bot as bot_module
    from dashboard.config import ConfigError

    def failed_preflight():
        raise ConfigError("use the active interpreter")

    observed = {}

    async def fake_dry_run(settings):
        observed["ffmpeg"] = settings.ffmpeg_executable

    monkeypatch.setattr(bot_module, "startup_preflight", failed_preflight)
    monkeypatch.setattr(bot_module, "run_dry_run", fake_dry_run)
    monkeypatch.setattr(bot_module, "_setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot_module, "load_dotenv", lambda: None)
    assert bot_module.main(["--dry-run", "--no-keys"]) == 0
    assert observed["ffmpeg"] is None
    assert "Cameras are disabled; the dashboard will continue" in caplog.text


def test_main_failure_uses_configured_logging_and_returns_failure(monkeypatch, caplog):
    import bot as bot_module

    async def failed_run(_settings):
        raise RuntimeError("failed during startup")

    monkeypatch.setattr(bot_module, "_setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bot_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(bot_module, "startup_preflight", lambda: None)
    monkeypatch.setattr(bot_module, "run_dry_run", failed_run)
    assert bot_module.main(["--dry-run", "--no-keys"]) == 1
    assert "dashboard stopped reason=unhandled_error mode=dry_run error=RuntimeError" in caplog.text
    assert caplog.records[-1].exc_info is not None


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
    png = (await operation).webp
    assert png == b"map"
