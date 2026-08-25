"""Live-view button tests: instant ephemeral snapshot, countdown, refresh."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

import pytest

import bot as bot_module
from dashboard.models import CameraFrame
from tests.fixtures.sample_data import jpeg_bytes


class FakeResponse:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.deferred = False
        self.ephemeral = False

    async def send_message(self, content=None, **kwargs):
        self.sent.append({"content": content, **kwargs})

    async def defer(self, *, ephemeral=False, thinking=False):
        self.deferred = True
        self.ephemeral = ephemeral


class FakeFollowup:
    def __init__(self, interaction) -> None:
        self.interaction = interaction
        self.sent: list[dict] = []

    async def send(self, content=None, wait=False, **kwargs):
        message = FakeMessage()
        self.sent.append({"content": content, "message": message, **kwargs})
        self.interaction.sent_messages.append(self.sent[-1])
        return message


class FakeMessage:
    def __init__(self) -> None:
        self.id = 12345
        self.deleted = False
        self.edits = 0
        self.last_edit: dict | None = None
        self.fetches = 0

    async def delete(self):
        self.deleted = True

    async def edit(self, **kwargs):
        self.edits += 1
        self.last_edit = kwargs


class FakeInteraction:
    def __init__(self) -> None:
        self.response = FakeResponse()
        self.followup = FakeFollowup(self)
        self.sent_messages: list[dict] = []
        self.channel = object()


def _updater(monkeypatch, ffmpeg="/fake/ffmpeg"):
    from dashboard.config import Settings

    settings = Settings(
        discord_token="x", announce_channel_id=1, ffmpeg_executable=ffmpeg
    )
    updater = bot_module.DashboardUpdater(settings)
    updater.client = object()
    return updater


def _frames():
    return [
        CameraFrame(jpeg_bytes(), "North Gate bus stop", datetime.now(UTC)),
        CameraFrame(jpeg_bytes(), "South Gate bus stop", datetime.now(UTC)),
    ]


def _fresh_cache(frames=None) -> bot_module.LiveFrameCache:
    cache = bot_module.LiveFrameCache()
    cache.frames = frames if frames is not None else _frames()
    cache.updated_monotonic = time.monotonic()
    return cache


@pytest.mark.asyncio
async def test_button_defers_then_answers_with_countdown(monkeypatch):
    updater = _updater(monkeypatch)
    updater.live_frames = _fresh_cache()
    monkeypatch.setattr(bot_module, "LIVE_VIEW_SNAPSHOT_SECONDS", 0)
    interaction = FakeInteraction()
    view = bot_module.LiveViewSnapshotView(updater)

    await view.children[0].callback(interaction)
    await updater.live_snapshot_task

    assert interaction.response.deferred is True
    sent = interaction.sent_messages[-1]
    assert sent["ephemeral"] is True
    assert len(sent["files"]) == 2 and len(sent["embeds"]) == 2
    assert updater.last_live_snapshot != float("-inf")
    assert "disappears in" in sent["embeds"][0].description
    assert updater.live_snapshot_message_id is None  # deleted after window


@pytest.mark.asyncio
async def test_press_while_showing_refreshes_instead_of_erroring(monkeypatch):
    updater = _updater(monkeypatch)
    updater.live_frames = _fresh_cache()
    running = asyncio.ensure_future(asyncio.sleep(5))
    updater.live_snapshot_task = running
    updater.live_snapshot_message = FakeMessage()
    interaction = FakeInteraction()
    view = bot_module.LiveViewSnapshotView(updater)

    await view.children[0].callback(interaction)

    running.cancel()
    sent = interaction.sent_messages[-1]
    assert sent.get("ephemeral")
    assert "Refreshed the snapshot" in sent.get("content", "")
    assert updater.live_snapshot_task is not running
    updater.live_snapshot_task.cancel()


@pytest.mark.asyncio
async def test_refresh_without_fresher_frames_hints(monkeypatch):
    updater = _updater(monkeypatch)
    stale = _fresh_cache()
    stale.updated_monotonic = time.monotonic() - 999
    updater.live_frames = stale
    running = asyncio.ensure_future(asyncio.sleep(5))
    updater.live_snapshot_task = running
    updater.live_snapshot_message = FakeMessage()
    interaction = FakeInteraction()
    view = bot_module.LiveViewSnapshotView(updater)

    await view.children[0].callback(interaction)

    running.cancel()
    sent = interaction.sent_messages[-1]
    assert sent.get("ephemeral") and "No fresher frames" in sent.get("content", "")


@pytest.mark.asyncio
async def test_rapid_second_press_does_not_cancel_initial_send(monkeypatch):
    updater = _updater(monkeypatch)
    updater.live_frames = _fresh_cache()
    initial_send = asyncio.ensure_future(asyncio.sleep(5))
    updater.live_snapshot_task = initial_send
    assert updater.live_snapshot_message is None
    interaction = FakeInteraction()
    view = bot_module.LiveViewSnapshotView(updater)

    await view.children[0].callback(interaction)

    assert updater.live_snapshot_task is initial_send
    assert not initial_send.cancelled()
    sent = interaction.sent_messages[-1]
    assert sent.get("ephemeral") and "snapshot is opening" in sent.get("content", "")
    initial_send.cancel()
    with pytest.raises(asyncio.CancelledError):
        await initial_send


@pytest.mark.asyncio
async def test_snapshot_response_is_deleted_after_window(monkeypatch):
    updater = _updater(monkeypatch)
    interaction = FakeInteraction()
    monkeypatch.setattr(bot_module, "LIVE_VIEW_SNAPSHOT_SECONDS", 0)

    parts = bot_module._SnapshotParts(((b"jpeg", "busstop-0.jpg"),), ())
    await bot_module._send_ephemeral_snapshot(interaction, parts, updater)
    assert interaction.sent_messages, "snapshot should be sent via followup"
    message = interaction.sent_messages[-1]["message"]
    assert message.deleted is True
    assert updater.live_snapshot_message_id is None


@pytest.mark.asyncio
async def test_countdown_edits_keep_attachments(monkeypatch):
    """Countdown edits must re-pass attachments or Discord detaches images."""
    updater = _updater(monkeypatch)
    interaction = FakeInteraction()
    parts = bot_module._SnapshotParts(
        ((b"file-a", "busstop-0.jpg"), (b"file-b", "busstop-1.jpg")), ()
    )
    # One countdown tick, then the window ends: sleep returns immediately on
    # first call and a long real sleep is avoided by the zero window.
    calls = {"n": 0}

    real_sleep = bot_module.asyncio.sleep

    async def fake_sleep(seconds):
        calls["n"] += 1
        if calls["n"] <= 2:
            return await real_sleep(0)
        await real_sleep(0)

    monkeypatch.setattr(bot_module.asyncio, "sleep", fake_sleep)
    await bot_module._send_ephemeral_snapshot(interaction, parts, updater)
    message = interaction.sent_messages[-1]["message"]
    assert message.edits >= 1
    attachments = message.last_edit["attachments"]
    assert len(attachments) == 2
    assert all(getattr(file, "fp", None) and not file.fp.closed for file in attachments)


@pytest.mark.asyncio
async def test_refresh_reuses_sent_message_and_recreates_uploads(monkeypatch):
    """Refresh edits the webhook handle; it must not fetch or reuse closed files."""
    updater = _updater(monkeypatch)
    updater.live_frames = _fresh_cache()
    interaction = FakeInteraction()
    monkeypatch.setattr(bot_module, "LIVE_VIEW_SNAPSHOT_SECONDS", 0)
    initial = bot_module._snapshot_parts_from_frames(updater.live_frames.frames)

    # Keep the initial window alive long enough to press the button.
    monkeypatch.setattr(bot_module, "LIVE_VIEW_SNAPSHOT_SECONDS", 60)
    updater.live_snapshot_task = asyncio.create_task(
        bot_module._send_ephemeral_snapshot(interaction, initial, updater)
    )
    await asyncio.sleep(0)
    message = updater.live_snapshot_message
    assert message is not None
    old_files = interaction.sent_messages[0]["files"]
    for file in old_files:
        file.fp.close()

    refreshed = bot_module._SnapshotParts(((b"new-jpeg", "busstop-0.jpg"),), ())
    task = asyncio.create_task(bot_module._refresh_ephemeral_snapshot(
        updater.live_snapshot_task, refreshed, updater
    ))
    updater.live_snapshot_task = task
    for _ in range(20):
        await asyncio.sleep(0)
        if message.edits:
            break
    assert updater.live_snapshot_message is message
    assert message.edits >= 1
    assert message.deleted is False
    assert interaction.channel is not getattr(updater, "live_snapshot_channel", None)
    assert message.last_edit["attachments"][0].fp is not old_files[0].fp
    # A second handoff must invalidate the first refresh owner without
    # deleting the message it is about to reuse.
    second = bot_module._SnapshotParts(((b"third-jpeg", "busstop-0.jpg"),), ())
    generation = updater.live_snapshot_generation + 1
    updater.live_snapshot_generation = generation
    second_task = asyncio.create_task(bot_module._refresh_ephemeral_snapshot(
        task, second, updater, generation
    ))
    updater.live_snapshot_task = second_task
    for _ in range(20):
        await asyncio.sleep(0)
        if message.edits >= 2:
            break
    assert message.deleted is False
    second_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second_task
    assert message.deleted is True


@pytest.mark.asyncio
async def test_frame_cache_stop_awaits_refresh_task():
    cache = bot_module.LiveFrameCache()
    finished = asyncio.Event()

    async def worker():
        try:
            await asyncio.sleep(60)
        finally:
            finished.set()

    cache._task = asyncio.create_task(worker())
    await cache.stop()
    assert finished.is_set()
    assert cache._task is None


@pytest.mark.asyncio
async def test_cooldown_rejects_rapid_press(monkeypatch):
    updater = _updater(monkeypatch)
    updater.live_frames = _fresh_cache()
    updater.last_live_snapshot = time.monotonic()
    interaction = FakeInteraction()
    view = bot_module.LiveViewSnapshotView(updater)

    await view.children[0].callback(interaction)

    sent = interaction.sent_messages[-1]
    assert sent.get("ephemeral") and "try again shortly" in sent.get("content", "")


def test_frame_cache_freshness_window():
    cache = bot_module.LiveFrameCache()
    assert cache.fresh is False  # empty
    cache.frames = _frames()
    cache.updated_monotonic = time.monotonic()
    assert cache.fresh is True
    cache.updated_monotonic = time.monotonic() - bot_module.LIVE_FRAME_MAX_AGE_SECONDS - 1
    assert cache.fresh is False  # expired


@pytest.mark.asyncio
async def test_frame_refresh_loop_updates_cache(monkeypatch):
    updater = _updater(monkeypatch)

    async def fake_fetch(client, ffmpeg_executable=None):
        return _frames()

    monkeypatch.setattr(bot_module.cameras, "fetch_bus_stop_frames", fake_fetch)
    monkeypatch.setattr(bot_module, "LIVE_FRAME_REFRESH_SECONDS", 0)
    cache = bot_module.LiveFrameCache()

    task = asyncio.ensure_future(bot_module._frame_refresh_loop(cache, updater))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cache.fresh is True
    assert len(cache.frames) == 2


def test_countdown_note_mentions_refresh():
    embeds = [bot_module.discord.Embed(title="📷 x")]
    ends = time.monotonic() + 60
    stamped = bot_module._snapshot_embeds_with_countdown(embeds, ends, 1)
    assert "Bus stops live" in stamped[0].description
    assert "disappears in" in stamped[0].description
