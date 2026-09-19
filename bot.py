"""HKUST Campus Dashboard bot entry point.

Executable lifecycle: shared session + providers, concurrent fetches, one live
dashboard message edited in place, dev-webhook and dry-run modes.

Imports must have no filesystem/network/package-installation/bot-launch side
effects; all side effects live under ``main()``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import contextvars
import copy
import hashlib
import inspect
import io
import json
import logging
import math
import os
import re
import secrets
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import discord
from dotenv import load_dotenv

from dashboard import maps, pipeline, road_policy  # noqa: F401
from dashboard.config import ConfigError, Settings
from dashboard.diagnostics import configure_logging, dependency_versions, source_fingerprint
from dashboard.http import HttpClient
from dashboard.models import (
    MAP_CAPTURE_MAX_AGE_SECONDS,
    MAP_CAPTURE_STALE_AFTER_SECONDS,
    CameraFrame,
    DashboardPayload,
    ImageAsset,
    TrafficMapResult,
    WeatherConditions,
)

# Historical provider names remain importable for diagnostics and monkeypatches.
from dashboard.providers import (
    cameras,
    transit,  # noqa: F401
)
from dashboard.providers import route_geometry as route_geometry_provider  # noqa: F401
from dashboard.providers import tracked_roads as tracked_roads_provider  # noqa: F401
from dashboard.providers import traffic as traffic_provider  # noqa: F401
from dashboard.providers import weather as weather_provider  # noqa: F401
from dashboard.providers.traffic_news import filter_expired_rthk_incidents
from dashboard.render import traffic_map_filename
from dashboard.runtime import startup_preflight

log = logging.getLogger(__name__)
DASHBOARD_MESSAGE_MARKER = "HKUST Campus Dashboard"
DASHBOARD_EDIT_BACKOFF_SECONDS = 60.0
DASHBOARD_DISCORD_TIMEOUT_SECONDS = 4.0
DASHBOARD_SEND_NONCE_RETRY_SECONDS = 2 * 60
DISCORD_MAX_RATELIMIT_RETRY_SECONDS = 1.25
DASHBOARD_RUNTIME_STATE_FILENAME = "dashboard-runtime-state.json"
TRACKED_ROADS_WAIT_SECONDS = 5.0


@dataclass
class _DashboardSendAttemptEvidence:
    """Wire-level evidence for one application-level dashboard create."""

    wire_attempts: int = 0


_active_dashboard_send_evidence: contextvars.ContextVar[_DashboardSendAttemptEvidence | None] = (
    contextvars.ContextVar("active_dashboard_send_evidence", default=None)
)


def _note_dashboard_send_wire_attempt(method: object, url: object) -> None:
    """Count one Discord create-message request in the active send episode."""
    evidence = _active_dashboard_send_evidence.get()
    if evidence is None or str(method).upper() != "POST":
        return
    path = getattr(url, "path", None)
    if isinstance(path, str) and re.fullmatch(
        r"(?:/api/v\d+)?/channels/\d+/messages",
        path,
    ):
        evidence.wire_attempts += 1


def _dashboard_http_trace_config():
    """Build the aiohttp trace that exposes discord.py's internal retries."""
    import aiohttp

    trace = aiohttp.TraceConfig()

    async def request_started(_session, _trace_context, params) -> None:
        _note_dashboard_send_wire_attempt(params.method, params.url)

    trace.on_request_start.append(request_started)
    return trace


async def _observe_dashboard_send(
    operation,
    evidence: _DashboardSendAttemptEvidence,
):
    token = _active_dashboard_send_evidence.set(evidence)
    try:
        return await operation
    finally:
        _active_dashboard_send_evidence.reset(token)


class _CancellationSafeDiscordGlobalGate:
    """Ensure discord.py cannot strand its global 429 gate on cancellation."""

    def __init__(self, event: asyncio.Event, maximum_retry: float) -> None:
        self._event = event
        self._maximum_retry = maximum_retry
        self._reset_handle: asyncio.TimerHandle | None = None

    def is_set(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> bool:
        return await self._event.wait()

    def clear(self) -> None:
        self._event.clear()
        if self._reset_handle is not None:
            self._reset_handle.cancel()
        # HTTPClient rejects retry_after values above this maximum before it
        # clears the gate.  If cancellation interrupts an allowed sleep, this
        # fallback opens the gate once that maximum has elapsed.
        self._reset_handle = asyncio.get_running_loop().call_later(
            self._maximum_retry,
            self.set,
        )

    def set(self) -> None:
        if self._reset_handle is not None:
            self._reset_handle.cancel()
            self._reset_handle = None
        self._event.set()


def _configure_discord_http_deadlines(http) -> None:
    """Reject long 429 sleeps and make the library's global gate recoverable."""
    http.max_ratelimit_timeout = DISCORD_MAX_RATELIMIT_RETRY_SECONDS
    global_gate = getattr(http, "_global_over", None)
    if global_gate is not None and not isinstance(global_gate, _CancellationSafeDiscordGlobalGate):
        http._global_over = _CancellationSafeDiscordGlobalGate(  # noqa: SLF001
            global_gate,
            DISCORD_MAX_RATELIMIT_RETRY_SECONDS,
        )


def _setup_logging(
    level: str, cache_dir: str = ".cache", *, secret_values: tuple[str, ...] = (),
) -> None:
    configure_logging(level, cache_dir, secret_values=secret_values)


# --------------------------------------------------------------------------
# Data collection
# --------------------------------------------------------------------------


# Compatibility forwarding names. Collection and payload adaptation live in
# dashboard.pipeline; these names keep diagnostic scripts and older tests
# importing bot's historical API working.
_map_road_paths_from_results = pipeline.map_road_paths_from_results
_fetch_traffic_map_from_results = pipeline.fetch_traffic_map_from_results
_to_payload = pipeline.to_payload


async def collect_all(
    client: HttpClient,
    settings: Settings,
    on_result: pipeline.ResultCallback | None = None,
    tracker=None,
    include_traffic_map: bool = True,
) -> pipeline.ProviderResults:
    """Expose the collection capabilities used by updater feature detection."""
    return await pipeline.collect_all(
        client,
        settings,
        on_result=on_result,
        tracker=tracker,
        include_traffic_map=include_traffic_map,
        tracked_roads_wait_seconds=TRACKED_ROADS_WAIT_SECONDS,
    )


# --------------------------------------------------------------------------
# Message lifecycle
# --------------------------------------------------------------------------

LIVE_VIEW_BUTTON_ID = "busstop:live"
LIVE_VIEW_SNAPSHOT_SECONDS = 60
LIVE_VIEW_COOLDOWN_SECONDS = 30
LIVE_FRAME_MAX_AGE_SECONDS = 45.0
LIVE_FRAME_REFRESH_SECONDS = 20.0
LIVE_COUNTDOWN_REFRESH_SECONDS = 15


class LiveFrameCache:
    """Rolling latest camera frames, refreshed in the background.

    HLS segment fetch + ffmpeg decode takes seconds; a button press must be
    answered within three. So the cache refreshes continuously and presses
    answer instantly from the latest decoded frames.
    """

    def __init__(self) -> None:
        self.frames: list[CameraFrame] = []
        self.updated_monotonic: float = 0.0
        self._task: asyncio.Task | None = None

    @property
    def fresh(self) -> bool:
        return (
            bool(self.frames)
            and time.monotonic() - self.updated_monotonic <= LIVE_FRAME_MAX_AGE_SECONDS
        )

    def start(self, updater: DashboardUpdater) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(_frame_refresh_loop(self, updater))

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            # Give a freshly-created task one turn to enter its coroutine so
            # its cancellation cleanup is guaranteed to run.
            await asyncio.sleep(0)
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
        self._task = None


async def _frame_refresh_loop(cache: LiveFrameCache, updater: DashboardUpdater) -> None:
    """Decode camera frames continuously so button presses are instant."""
    while True:
        try:
            assert updater.client is not None and updater.settings.ffmpeg_executable
            frames = await cameras.fetch_bus_stop_frames(
                updater.client,
                ffmpeg_executable=updater.settings.ffmpeg_executable,
            )
            good = [f for f in frames if isinstance(f, CameraFrame) and f.data]
            if good:
                cache.frames = good
                cache.updated_monotonic = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("live-frame refresh failed: %s", type(exc).__name__)
        await asyncio.sleep(LIVE_FRAME_REFRESH_SECONDS)


@dataclass(frozen=True)
class _SnapshotParts:
    """Immutable snapshot assets; upload objects are created per Discord call."""

    assets: tuple[tuple[bytes, str], ...]
    embeds: tuple[discord.Embed, ...]

    def files(self) -> list[discord.File]:
        return [discord.File(io.BytesIO(data), filename=name) for data, name in self.assets]


def _snapshot_parts_from_frames(frames: list[CameraFrame]) -> _SnapshotParts:
    assets: list[tuple[bytes, str]] = []
    embeds: list[discord.Embed] = []
    for index, frame in enumerate(frames):
        filename = f"busstop-{index}.jpg"
        assets.append((bytes(frame.data), filename))
        embed = discord.Embed(title=f"📷 {frame.label} — live snapshot", color=0x0F766E)
        embed.set_image(url=f"attachment://{filename}")
        stamp = frame.source_time
        if stamp is not None:
            if isinstance(stamp, (int, float)):
                from datetime import UTC as _UTC
                from datetime import datetime as _dt

                stamp = _dt.fromtimestamp(float(stamp), tz=_UTC)
            if stamp.tzinfo is None:
                from datetime import UTC as _UTC

                stamp = stamp.replace(tzinfo=_UTC)
            embed.timestamp = stamp
            embed.set_footer(text="HKUST live view")
        embeds.append(embed)
    return _SnapshotParts(tuple(assets), tuple(embeds))


class LiveViewSnapshotView(discord.ui.View):
    """Persistent button answering instantly with the cached live frames.

    A press while a snapshot is showing REFRESHES it: newest frames plus a
    restarted disappearance countdown, instead of an error.
    """

    def __init__(self, updater: DashboardUpdater) -> None:
        super().__init__(timeout=None)
        self.updater = updater

    @discord.ui.button(
        label="Bus stops live",
        style=discord.ButtonStyle.primary,
        custom_id=LIVE_VIEW_BUTTON_ID,
        emoji="📷",
    )
    async def snapshot(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        # Acknowledge within the 3-second window no matter how busy the loop
        # is; every outcome is delivered as an ephemeral followup.
        await interaction.response.defer(ephemeral=True, thinking=False)
        updater = self.updater
        cache: LiveFrameCache | None = getattr(updater, "live_frames", None)
        now = time.monotonic()

        # Refresh-in-place: if a snapshot window is live, swap in the newest
        # frames and restart its countdown rather than erroring.
        running = updater.live_snapshot_task
        if running is not None and not running.done():
            # The initial followup send yields before it publishes its webhook
            # message handle. Do not cancel that send if another press lands
            # in this small window: there is no message to refresh yet.
            if updater.live_snapshot_message is None:
                with contextlib.suppress(Exception):
                    await interaction.followup.send(
                        "The snapshot is opening; try again in a moment to refresh it.",
                        ephemeral=True,
                    )
                return
            if cache is None or not cache.fresh:
                with contextlib.suppress(Exception):
                    await interaction.followup.send(
                        "No fresher frames yet; try again in a few seconds.",
                        ephemeral=True,
                    )
                return
            parts = _snapshot_parts_from_frames(cache.frames)
            generation = updater.live_snapshot_generation + 1
            updater.live_snapshot_generation = generation
            updater.live_snapshot_task = asyncio.create_task(
                _refresh_ephemeral_snapshot(running, parts, updater, generation)
            )
            return

        if now - updater.last_live_snapshot < LIVE_VIEW_COOLDOWN_SECONDS:
            with contextlib.suppress(Exception):
                await interaction.followup.send(
                    "The last snapshot just ended; try again shortly.",
                    ephemeral=True,
                )
            return
        if (
            cache is None
            or not cache.fresh
            or updater.client is None
            or (not updater.settings.ffmpeg_executable)
        ):
            with contextlib.suppress(Exception):
                await interaction.followup.send(
                    "Live frames are still loading; try again in a few seconds.",
                    ephemeral=True,
                )
            return

        parts = _snapshot_parts_from_frames(cache.frames)
        generation = updater.live_snapshot_generation + 1
        updater.live_snapshot_generation = generation
        updater.live_snapshot_task = asyncio.create_task(
            _send_ephemeral_snapshot(interaction, parts, updater, generation)
        )


def _snapshot_embeds_with_countdown(
    embeds: list[discord.Embed] | tuple[discord.Embed, ...],
    ends_monotonic: float,
    frame_count: int,
) -> list[discord.Embed]:
    """Stamp each embed with a visible disappearance countdown line."""
    remaining = max(0, int(round(ends_monotonic - time.monotonic())))
    note = (
        f"⏳ disappears in <t:{int(time.time()) + remaining}:R> — "
        "press **Bus stops live** again to refresh"
    )
    out: list[discord.Embed] = []
    for embed in embeds:
        embed = copy.deepcopy(embed)
        embed.description = note
        if frame_count > 1:
            pass
        out.append(embed)
    return out


async def _send_ephemeral_snapshot(
    interaction: discord.Interaction,
    parts: _SnapshotParts,
    updater: DashboardUpdater,
    generation: int | None = None,
) -> None:
    """Answer the deferred button with the snapshot, delete it after a minute."""
    message = None
    if generation is None:
        generation = updater.live_snapshot_generation + 1
        updater.live_snapshot_generation = generation
    try:
        ends = time.monotonic() + LIVE_VIEW_SNAPSHOT_SECONDS
        message = await interaction.followup.send(
            embeds=_snapshot_embeds_with_countdown(parts.embeds, ends, len(parts.assets)),
            files=parts.files(),
            ephemeral=True,
            wait=True,
        )
        updater.live_snapshot_message_id = message.id
        updater.live_snapshot_message = message
        updater.last_live_snapshot = time.monotonic()
        # Refresh the countdown line periodically so it visibly counts down.
        # Attachments must be re-passed on every edit or Discord detaches the
        # images the embeds still reference.
        ticks = int(LIVE_VIEW_SNAPSHOT_SECONDS // LIVE_COUNTDOWN_REFRESH_SECONDS)
        for _ in range(ticks):
            await asyncio.sleep(LIVE_COUNTDOWN_REFRESH_SECONDS)
            remaining = max(0, int(round(ends - time.monotonic())))
            if remaining <= 0 or message is None:
                break
            with contextlib.suppress(Exception):
                await message.edit(
                    embeds=_snapshot_embeds_with_countdown(parts.embeds, ends, len(parts.assets)),
                    attachments=parts.files(),
                )
        await asyncio.sleep(max(0.0, ends - time.monotonic()))
    except Exception as exc:  # noqa: BLE001
        import traceback

        log.warning(
            "ephemeral snapshot delivery failed: %s\n%s",
            exc,
            traceback.format_exc(limit=5),
        )
        with contextlib.suppress(Exception):
            await interaction.followup.send(
                "Snapshot delivery failed; please try again.", ephemeral=True
            )
    finally:
        if (
            message is not None
            and updater.live_snapshot_message is message
            and updater.live_snapshot_generation == generation
        ):
            updater.live_snapshot_message_id = None
            updater.live_snapshot_message = None
            with contextlib.suppress(Exception):
                await message.delete()


async def _refresh_ephemeral_snapshot(
    old_task: asyncio.Task,
    parts: _SnapshotParts,
    updater: DashboardUpdater,
    generation: int | None = None,
) -> None:
    """Replace the running snapshot's frames and restart its countdown."""
    # Stop the old window WITHOUT deleting its message, then reuse it.
    message = updater.live_snapshot_message
    if generation is None:
        generation = updater.live_snapshot_generation + 1
        updater.live_snapshot_generation = generation
    old_task.cancel()
    updater.last_live_snapshot = time.monotonic()
    with contextlib.suppress(asyncio.CancelledError):
        await old_task
    if message is None:
        return
    updater.live_snapshot_message = message
    try:
        ends = time.monotonic() + LIVE_VIEW_SNAPSHOT_SECONDS
        await message.edit(
            embeds=_snapshot_embeds_with_countdown(parts.embeds, ends, len(parts.assets)),
            attachments=parts.files(),
        )
        ticks = int(LIVE_VIEW_SNAPSHOT_SECONDS // LIVE_COUNTDOWN_REFRESH_SECONDS)
        for _ in range(ticks):
            await asyncio.sleep(LIVE_COUNTDOWN_REFRESH_SECONDS)
            remaining = max(0, int(round(ends - time.monotonic())))
            if remaining <= 0:
                break
            with contextlib.suppress(Exception):
                await message.edit(
                    embeds=_snapshot_embeds_with_countdown(parts.embeds, ends, len(parts.assets)),
                    attachments=parts.files(),
                )
        await asyncio.sleep(max(0.0, ends - time.monotonic()))
    except Exception as exc:  # noqa: BLE001
        log.warning("ephemeral snapshot refresh failed: %s", exc)
    finally:
        if message is not None and (
            updater.live_snapshot_message is message
            and updater.live_snapshot_generation == generation
        ):
            with contextlib.suppress(Exception):
                await message.delete()
            updater.live_snapshot_message = None
            updater.live_snapshot_message_id = None


def _is_dashboard_message(message, expected_author) -> bool:
    """Return whether ``message`` is this bot's exact dashboard marker."""
    expected_author_id = getattr(expected_author, "id", None)
    author = getattr(message, "author", None)
    return bool(
        expected_author_id is not None
        and getattr(author, "bot", False)
        and getattr(author, "id", None) == expected_author_id
        and getattr(message, "content", "") == DASHBOARD_MESSAGE_MARKER
    )


async def _scan_dashboard_messages(
    channel,
    expected_author=None,
    *,
    after: datetime | None = None,
    before: datetime | None = None,
) -> list[object]:
    """Strictly scan history for this bot's dashboard messages."""
    expected_author = expected_author or getattr(getattr(channel, "guild", None), "me", None)
    history_options = {"limit": None, "after": after, "before": before}
    if after is None and before is None:
        history_options = {"limit": 50}
    return [
        message
        async for message in channel.history(**history_options)
        if _is_dashboard_message(message, expected_author)
    ]


async def _find_dashboard_messages(channel, expected_author=None) -> list[object]:
    """Best-effort scan for this bot's dashboard messages, newest first.

    Author ID plus the stable marker avoid taking over another message from
    this bot (for example an alert or command response) in the same channel.
    """
    try:
        return await _scan_dashboard_messages(channel, expected_author)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "GET /channels/%s/messages history scan failed error=%s status=%s code=%s: %s",
            getattr(channel, "id", None), type(exc).__name__,
            getattr(exc, "status", None), getattr(exc, "code", None), exc,
            exc_info=True,
        )
    return []


async def _find_dashboard_message(channel, expected_author=None) -> object | None:
    """Return this bot's latest exact dashboard marker, if present."""
    messages = await _find_dashboard_messages(channel, expected_author)
    return messages[0] if messages else None


async def _resolve_dashboard_message(
    channel,
    configured_message_id: int | None,
    expected_author,
) -> object | None:
    """Resolve the newest live dashboard, validating any configured fallback."""
    configured = None
    if configured_message_id:
        try:
            configured = await channel.fetch_message(configured_message_id)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "GET /channels/%s/messages/%s configured message lookup failed; scanning "
                "error=%s status=%s code=%s: %s",
                getattr(channel, "id", None), configured_message_id, type(exc).__name__,
                getattr(exc, "status", None), getattr(exc, "code", None), exc,
            )
        else:
            if not _is_dashboard_message(configured, expected_author):
                log.warning(
                    "configured message %s is not this bot's exact dashboard marker; scanning",
                    configured_message_id,
                )
                configured = None
    messages = await _find_dashboard_messages(channel, expected_author)
    if messages:
        # A rollover can complete immediately before a restart.  History order
        # is authoritative here so a stale configured starter cannot win over
        # the newer, fully populated dashboard.
        log.info(
            "dashboard discovery channel_id=%s configured_message_id=%s selected_message_id=%s "
            "candidates=%s source=history",
            getattr(channel, "id", None), configured_message_id, messages[0].id,
            [message.id for message in messages],
        )
        return messages[0]
    log.info(
        "dashboard discovery channel_id=%s configured_message_id=%s selected_message_id=%s "
        "source=configured_fallback",
        getattr(channel, "id", None), configured_message_id, getattr(configured, "id", None),
    )
    return configured


async def _ensure_dashboard_message(
    channel,
    payload: DashboardPayload,
    view=None,
    *,
    nonce: int | None = None,
    attempt_evidence: _DashboardSendAttemptEvidence | None = None,
) -> object:
    """Reuse the known message or find/create exactly one."""
    if nonce is None:
        message = await _find_dashboard_message(channel)
        if message is not None:
            return message
    # create exactly one
    evidence = attempt_evidence or _DashboardSendAttemptEvidence()
    message = await _observe_dashboard_send(
        channel.send(
            content=DASHBOARD_MESSAGE_MARKER,
            view=view,
            nonce=nonce,
        ),
        evidence,
    )
    log.info("created dashboard message %s in %s", message.id, getattr(channel, "id", "?"))
    return message


def discord_file(asset: ImageAsset) -> discord.File:
    return discord.File(io.BytesIO(asset.data), filename=asset.filename)


def _is_traffic_map_filename(filename: str) -> bool:
    """Recognize renderer-generated traffic-map attachments for dry-run output."""
    return bool(re.fullmatch(r"traffic-map-[0-9a-f]{12}\.webp", filename))


def _payload_fingerprint(payload: DashboardPayload) -> str:
    """Hash sent content, excluding render-time embed timestamps."""
    embeds = []
    for embed in payload.embeds:
        if embed is None:
            continue
        as_dict = embed.to_dict() if hasattr(embed, "to_dict") else embed
        if isinstance(as_dict, dict):
            # ``build_payload`` stamps each render with ``checked_at``.  The
            # timestamp is useful to display, but should not defeat dedup when
            # all source content and attachments are unchanged.
            as_dict = dict(as_dict)
            as_dict.pop("timestamp", None)
        embeds.append(as_dict)
    files = [
        {
            "filename": asset.filename,
            "content_type": asset.content_type,
            "data_sha256": hashlib.sha256(asset.data).hexdigest(),
        }
        for asset in payload.files
    ]
    serialized = json.dumps(
        {"embeds": embeds, "files": files},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _dashboard_runtime_state_path(cache_dir: str) -> Path:
    return Path(cache_dir) / DASHBOARD_RUNTIME_STATE_FILENAME


def _load_dashboard_runtime_state(settings: Settings) -> dict[str, object]:
    """Load rollover state only when it belongs to this announce channel."""
    try:
        raw = json.loads(
            _dashboard_runtime_state_path(settings.cache_dir).read_text(encoding="utf-8")
        )
        if not isinstance(raw, dict):
            return {}
        if raw.get("announce_channel_id") != settings.announce_channel_id:
            return {}
        return raw
    except FileNotFoundError:
        log.debug("dashboard runtime state absent channel_id=%s", settings.announce_channel_id)
        return {}
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        log.warning(
            "dashboard runtime state unreadable path=%s channel_id=%s error=%s",
            _dashboard_runtime_state_path(settings.cache_dir), settings.announce_channel_id,
            type(exc).__name__, exc_info=True,
        )
        return {}


def _store_dashboard_runtime_state(
    settings: Settings,
    state: dict[str, object],
) -> bool:
    """Atomically retain thread and incomplete-rollover recovery state."""
    path = _dashboard_runtime_state_path(settings.cache_dir)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(state, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            temporary.unlink()
        log.warning("could not persist dashboard runtime state: %s", exc)
        return False
    return True


async def _bounded_discord(operation):
    """Keep one Discord request/retry chain within a presentation period."""
    return await asyncio.wait_for(
        operation,
        timeout=DASHBOARD_DISCORD_TIMEOUT_SECONDS,
    )


async def _apply_payload(message, payload: DashboardPayload, view=None):
    """Edit atomically, retaining actual attachments or uploading missing images.

    A CDN thumbnail URL is not an attachment to retain. Omitting its file from
    the edit removes it even while the signed URL temporarily still resolves.
    """
    embeds = [e.copy() for e in payload.embeds if e is not None]
    existing_by_filename = {
        attachment.filename: attachment
        for attachment in getattr(message, "attachments", ())
        if getattr(attachment, "filename", None)
    }
    attachments = []
    retained = []
    uploaded = []
    for asset in payload.files:
        existing = existing_by_filename.get(asset.filename)
        existing_size = getattr(existing, "size", None)
        if (
            existing is not None
            and getattr(existing, "id", None) is not None
            and existing_size == len(asset.data)
        ):
            attachments.append(existing)
            retained.append(f"{asset.filename}:{existing.id}:{existing_size}")
            # Keep both the attachment ID and the thumbnail URL stable.
            # Discord accepts unsigned CDN URLs in embeds and renews their
            # signatures itself; rotating query tokens need not change this
            # unchanged warning image on every map/ETA edit.
            thumbnail_url = _retained_thumbnail_url(existing)
            if thumbnail_url is not None:
                for embed in embeds:
                    if embed.thumbnail.url == f"attachment://{asset.filename}":
                        embed.set_thumbnail(url=thumbnail_url)
        else:
            attachments.append(discord_file(asset))
            uploaded.append(f"{asset.filename}:{len(asset.data)}")
    log.debug(
        "dashboard attachments message_id=%s embed_count=%d retained=%s uploaded=%s",
        getattr(message, "id", None), len(embeds), retained, uploaded,
    )
    return await message.edit(
        content=DASHBOARD_MESSAGE_MARKER,
        embeds=embeds,
        attachments=attachments,
        view=view,
    )


def _discord_edit_retry_delay(exc: BaseException) -> float | None:
    """Respect Discord's edit cooldown without replacing the starter message."""
    if not (
        getattr(exc, "code", None) == 30046
        or getattr(exc, "status", None) == 429
        or isinstance(exc, discord.RateLimited)
    ):
        return None
    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
    for value in (
        getattr(exc, "retry_after", None),
        headers.get("Retry-After"),
        headers.get("X-RateLimit-Reset-After"),
    ):
        try:
            delay = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(delay) and delay > 0:
            return delay
    return DASHBOARD_EDIT_BACKOFF_SECONDS


def _is_discord_not_found(exc: BaseException) -> bool:
    return isinstance(exc, discord.NotFound) or getattr(exc, "status", None) == 404


def _is_definitive_discord_send_rejection(exc: BaseException) -> bool:
    """Return whether Discord explicitly rejected this particular POST."""
    status = getattr(exc, "status", None)
    return bool(
        isinstance(status, int)
        and not isinstance(status, bool)
        and 400 <= status < 500
        and status != 408
    )


async def _send_payload(
    channel,
    payload: DashboardPayload,
    view=None,
    *,
    nonce: int | None = None,
    attempt_evidence: _DashboardSendAttemptEvidence | None = None,
):
    """Create a complete dashboard message without exposing a blank frame."""
    evidence = attempt_evidence or _DashboardSendAttemptEvidence()
    return await _observe_dashboard_send(
        channel.send(
            content=DASHBOARD_MESSAGE_MARKER,
            embeds=[embed for embed in payload.embeds if embed is not None],
            files=[discord_file(asset) for asset in payload.files],
            view=view,
            nonce=nonce,
        ),
        evidence,
    )


async def _rollover_dashboard_message(
    channel,
    message,
    payload: DashboardPayload,
    view=None,
    *,
    nonce: int | None = None,
    attempt_evidence: _DashboardSendAttemptEvidence | None = None,
):
    """Create the successor, then retire the old dashboard when possible."""
    replacement = await _bounded_discord(
        _send_payload(
            channel,
            payload,
            view=view,
            nonce=nonce,
            attempt_evidence=attempt_evidence,
        )
    )
    stale_message = None
    log.info(
        "dashboard deletion requested channel_id=%s message_id=%s replacement_message_id=%s "
        "reason=legacy_rollover_recovery",
        getattr(channel, "id", None), message.id, replacement.id,
    )
    try:
        await _bounded_discord(message.delete())
    except Exception as exc:  # noqa: BLE001
        if not _is_discord_not_found(exc):
            # The complete replacement is now canonical.  Retain the old
            # object for bounded background cleanup instead of rolling back an
            # uncertain delete (which could leave the channel blank) or
            # creating another successor on the next tick.
            stale_message = message
            log.warning("old dashboard cleanup deferred: %s", exc)
    log.info(
        "legacy dashboard rollover resolved previous_message_id=%s message_id=%s cleanup_pending=%s",
        message.id, replacement.id, stale_message is not None,
    )
    return replacement, stale_message


# --------------------------------------------------------------------------
# Updater loop
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CollectionSnapshot:
    """The atomically published, last-good provider results.

    Provider fetches may take much longer than a Discord presentation period.
    The updater therefore publishes a completed collection as one immutable
    snapshot.  A failed provider retains its previous value where possible,
    while the error is retained so the renderer can surface stale data.
    """

    results: dict[str, object]
    generation: int
    completed_monotonic: float
    stale_providers: frozenset[str] = frozenset()
    # Providers that have published (including an exception) for this
    # generation.  The default keeps older test/dev constructors compatible.
    settled_providers: frozenset[str] = frozenset()


class DashboardUpdater:
    """Owns the session, providers, caches, and the single update loop."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = None
        self.client: HttpClient | None = None
        self._last_good_payload: DashboardPayload | None = None
        self._last_payload_fingerprint: str | None = None
        self._snapshot: CollectionSnapshot | None = None
        self._collection_task: asyncio.Task | None = None
        self._map_task: asyncio.Task | None = None
        self._map_generation: int | None = None
        self._independent_map_enabled = False
        self._collection_generation = 0
        self._last_alert_generation = 0
        self._last_queued_alert_generation = 0
        self._pending_alert_snapshots: deque[CollectionSnapshot] = deque()
        # (content, permit configured alert-role mention).  Keeping the mention
        # policy beside the queued content preserves it across send retries.
        self._pending_alert_messages: deque[tuple[str, bool]] = deque()
        self._message = None
        self._thread = None
        self._dashboard_edit_retry_at = 0.0
        self._last_health_signature = None
        self._last_health_log_at = float("-inf")
        runtime_state = _load_dashboard_runtime_state(settings)
        loaded_message_id = runtime_state.get("dashboard_message_id")
        self._persisted_dashboard_message_id = (
            loaded_message_id
            if isinstance(loaded_message_id, int)
            and not isinstance(loaded_message_id, bool)
            and loaded_message_id > 0
            else None
        )
        loaded_thread_id = runtime_state.get("status_thread_id")
        if not isinstance(loaded_thread_id, int) or loaded_thread_id <= 0:
            loaded_thread_id = None
        self._status_thread_id = loaded_thread_id or settings.dashboard_message_id
        self._persisted_status_thread_id = loaded_thread_id
        pending_ids = runtime_state.get("pending_dashboard_message_ids", [])
        self._pending_dashboard_delete_ids = (
            {
                item
                for item in pending_ids
                if isinstance(item, int) and not isinstance(item, bool) and item > 0
            }
            if isinstance(pending_ids, list)
            else set()
        )
        uncertain_since = runtime_state.get("rollover_uncertain_since")
        self._rollover_uncertain_since = (
            float(uncertain_since)
            if isinstance(uncertain_since, (int, float))
            and not isinstance(uncertain_since, bool)
            and uncertain_since > 0
            else None
        )
        send_nonce = runtime_state.get("dashboard_send_nonce")
        self._dashboard_send_nonce = (
            send_nonce
            if isinstance(send_nonce, int)
            and not isinstance(send_nonce, bool)
            and 0 < send_nonce < 2**63
            and self._rollover_uncertain_since is not None
            else None
        )
        predecessor_id = runtime_state.get("dashboard_send_predecessor_id")
        self._dashboard_send_predecessor_id = (
            predecessor_id
            if self._dashboard_send_nonce is not None
            and isinstance(predecessor_id, int)
            and not isinstance(predecessor_id, bool)
            and predecessor_id > 0
            else None
        )
        # Any episode surviving a process boundary has an attempt whose
        # server-side outcome was not durably resolved.
        self._dashboard_send_had_ambiguous_attempt = self._dashboard_send_nonce is not None
        log.info(
            "dashboard recovery state channel_id=%s persisted_message_id=%s persisted_thread_id=%s "
            "pending_cleanup_count=%d unresolved_send=%s",
            settings.announce_channel_id, self._persisted_dashboard_message_id, loaded_thread_id,
            len(self._pending_dashboard_delete_ids), self._dashboard_send_had_ambiguous_attempt,
        )
        self._dashboard_send_retry_ready = False
        self._dashboard_messages_reconciled = False
        self._pending_dashboard_deletes: dict[int, object] = {}
        self._dashboard_cleanup_task: asyncio.Task | None = None
        self._rollover_task: asyncio.Task | None = None
        self._loop_task: asyncio.Task | None = None
        self._running = False
        self._start_lock = asyncio.Lock()
        self.live_snapshot_task: asyncio.Task | None = None
        self.last_live_snapshot = float("-inf")
        self.live_snapshot_message_id: int | None = None
        self.live_snapshot_message = None
        self.live_snapshot_generation = 0
        self.live_view = LiveViewSnapshotView(self)
        self.marker_tracker = maps.MarkerTracker()
        self.live_frames = LiveFrameCache()
        from dashboard.alerts import AlertMonitor

        self.alerts = AlertMonitor()

    async def start(self, channel=None) -> None:
        import aiohttp

        async with self._start_lock:
            if self.is_running:
                log.info("dashboard update loop already running; ignoring duplicate start")
                return
            self.session = aiohttp.ClientSession()
            self.client = HttpClient(
                self.session, timeout_seconds=self.settings.http_timeout_seconds
            )
            # Camera decoding is unavailable in dry-run/test configurations
            # without the preflight-resolved ffmpeg executable.  Do not start
            # a retry loop that can only emit repeated assertion warnings.
            if self.settings.ffmpeg_executable:
                self.live_frames.start(self)
            self._running = True
            self._loop_task = asyncio.create_task(self._update_loop(channel))

    @property
    def is_running(self) -> bool:
        return bool(self._running and self._loop_task is not None and not self._loop_task.done())

    def _persist_dashboard_runtime_state(self) -> bool:
        state: dict[str, object] = {
            "announce_channel_id": self.settings.announce_channel_id,
            "pending_dashboard_message_ids": sorted(self._pending_dashboard_delete_ids),
        }
        if self._status_thread_id is not None:
            state["status_thread_id"] = self._status_thread_id
        if self._persisted_dashboard_message_id is not None:
            state["dashboard_message_id"] = self._persisted_dashboard_message_id
        if self._rollover_uncertain_since is not None:
            state["rollover_uncertain_since"] = self._rollover_uncertain_since
        if self._dashboard_send_nonce is not None:
            state["dashboard_send_nonce"] = self._dashboard_send_nonce
        if self._dashboard_send_predecessor_id is not None:
            state["dashboard_send_predecessor_id"] = self._dashboard_send_predecessor_id
        return _store_dashboard_runtime_state(self.settings, state)

    def _set_dashboard_message(self, message) -> None:
        """Adopt a dashboard message and discard any predecessor's thread."""
        previous_id = self._dashboard_message_id(self._message)
        message_id = self._dashboard_message_id(message)
        self._message = message
        if message_id != previous_id:
            log.info(
                "dashboard identity changed channel_id=%s previous_message_id=%s message_id=%s "
                "previous_thread_id=%s",
                self.settings.announce_channel_id, previous_id, message_id,
                getattr(self._thread, "id", None),
            )
            self._thread = None
            self._dashboard_edit_retry_at = 0.0
            # On startup the resolved message may be the persisted starter.
            # Keep its valid thread ID; dropping it here writes an empty state
            # before _remember_status_thread sees the already-persisted ID.
            if self._status_thread_id != message_id:
                self._status_thread_id = None
                self._persisted_status_thread_id = None

    def _mark_rollover_send_uncertain(self) -> bool:
        if self._dashboard_send_nonce is None:
            if self._dashboard_messages_reconciled:
                current_message_id = self._dashboard_message_id(self._message)
                if current_message_id is not None:
                    self._persisted_dashboard_message_id = current_message_id
            self._dashboard_send_nonce = secrets.randbits(63) or 1
            self._rollover_uncertain_since = time.time()
            predecessor_id = getattr(self._message, "id", None)
            self._dashboard_send_predecessor_id = (
                predecessor_id
                if isinstance(predecessor_id, int)
                and not isinstance(predecessor_id, bool)
                and predecessor_id > 0
                else None
            )
            self._dashboard_send_had_ambiguous_attempt = False
        self._dashboard_send_retry_ready = False
        self._dashboard_messages_reconciled = False
        return self._persist_dashboard_runtime_state()

    def _clear_rollover_send_uncertainty(self) -> None:
        if (
            self._rollover_uncertain_since is None
            and self._dashboard_send_nonce is None
            and self._dashboard_send_predecessor_id is None
        ):
            return
        self._rollover_uncertain_since = None
        self._dashboard_send_nonce = None
        self._dashboard_send_predecessor_id = None
        self._dashboard_send_had_ambiguous_attempt = False
        self._dashboard_send_retry_ready = False
        self._persist_dashboard_runtime_state()

    def _record_dashboard_send_failure(
        self,
        exc: BaseException,
        attempt_evidence: _DashboardSendAttemptEvidence | None = None,
    ) -> None:
        """Resolve proven single-wire rejections; retain ambiguous episodes."""
        if (
            _is_definitive_discord_send_rejection(exc)
            and attempt_evidence is not None
            and attempt_evidence.wire_attempts == 1
            and not self._dashboard_send_had_ambiguous_attempt
        ):
            self._clear_rollover_send_uncertainty()
            self._dashboard_messages_reconciled = False
            return
        self._dashboard_send_had_ambiguous_attempt = True
        self._dashboard_send_retry_ready = False
        self._dashboard_messages_reconciled = False
        self._persist_dashboard_runtime_state()

    @staticmethod
    def _dashboard_message_id(message) -> int | None:
        message_id = getattr(message, "id", None)
        return (
            message_id
            if isinstance(message_id, int) and not isinstance(message_id, bool) and message_id > 0
            else None
        )

    @staticmethod
    def _dashboard_message_key(message) -> int:
        message_id = DashboardUpdater._dashboard_message_id(message)
        return message_id if message_id is not None else id(message)

    @staticmethod
    def _message_matches_uncertain_send_window(message, started_at: float) -> bool:
        created_at = getattr(message, "created_at", None)
        try:
            created_timestamp = float(created_at.timestamp())
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False
        return (
            started_at - 60
            <= created_timestamp
            <= started_at
            + DASHBOARD_SEND_NONCE_RETRY_SECONDS
            + DASHBOARD_DISCORD_TIMEOUT_SECONDS
        )

    def _queue_dashboard_delete(self, message) -> None:
        if message is None:
            return
        key = self._dashboard_message_key(message)
        if self._message is not None and key == self._dashboard_message_key(self._message):
            return
        self._pending_dashboard_deletes[key] = message
        if isinstance(getattr(message, "id", None), int):
            self._pending_dashboard_delete_ids.add(key)
            self._persist_dashboard_runtime_state()

    async def _reconcile_dashboard_messages(self, channel) -> None:
        """Adopt the newest dashboard and queue crash leftovers for cleanup."""
        if self._dashboard_messages_reconciled:
            return
        self._dashboard_send_retry_ready = False
        expected_author = getattr(getattr(channel, "guild", None), "me", None)
        persisted_message = None
        persisted_message_id = self._persisted_dashboard_message_id
        pending_delete_keys = {
            *self._pending_dashboard_delete_ids,
            *self._pending_dashboard_deletes,
        }
        if persisted_message_id in pending_delete_keys:
            if self._dashboard_message_id(self._message) == persisted_message_id:
                self._set_dashboard_message(None)
            self._persisted_dashboard_message_id = None
            persisted_message_id = None
            self._persist_dashboard_runtime_state()
        if persisted_message_id is not None:
            current_message_id = self._dashboard_message_id(self._message)
            if current_message_id == persisted_message_id and _is_dashboard_message(
                self._message, expected_author
            ):
                persisted_message = self._message
            else:
                try:
                    persisted_message = await _bounded_discord(
                        channel.fetch_message(persisted_message_id)
                    )
                except Exception as exc:  # noqa: BLE001
                    if not _is_discord_not_found(exc):
                        log.warning(
                            "persisted canonical dashboard fetch deferred: %s",
                            exc,
                        )
                        return
                    self._persisted_dashboard_message_id = None
                    if current_message_id == persisted_message_id:
                        self._set_dashboard_message(None)
                    self._persist_dashboard_runtime_state()
                else:
                    if not _is_dashboard_message(
                        persisted_message,
                        expected_author,
                    ):
                        self._persisted_dashboard_message_id = None
                        persisted_message = None
                        if current_message_id == persisted_message_id:
                            self._set_dashboard_message(None)
                        self._persist_dashboard_runtime_state()
        try:
            messages = await _bounded_discord(_scan_dashboard_messages(channel, expected_author))
            if self._rollover_uncertain_since is not None:
                uncertain_at = datetime.fromtimestamp(self._rollover_uncertain_since, UTC)
                messages.extend(
                    await _bounded_discord(
                        _scan_dashboard_messages(
                            channel,
                            expected_author,
                            after=uncertain_at - timedelta(minutes=1),
                            before=uncertain_at
                            + timedelta(
                                seconds=DASHBOARD_SEND_NONCE_RETRY_SECONDS
                                + DASHBOARD_DISCORD_TIMEOUT_SECONDS
                            ),
                        )
                    )
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("dashboard reconciliation deferred: %s", exc)
            return
        if persisted_message is not None:
            messages.append(persisted_message)
        unique_messages: dict[int, object] = {}
        for message in messages:
            unique_messages.setdefault(self._dashboard_message_key(message), message)
        history_messages = sorted(
            unique_messages.values(),
            key=self._dashboard_message_key,
            reverse=True,
        )
        previous = self._message
        previous_key = self._dashboard_message_key(previous) if previous is not None else None
        pending_delete_keys = {
            *self._pending_dashboard_delete_ids,
            *self._pending_dashboard_deletes,
        }
        canonical_candidates = [
            message
            for message in history_messages
            if self._dashboard_message_key(message) not in pending_delete_keys
        ]
        if previous is not None and previous_key not in pending_delete_keys:
            canonical_candidates.append(previous)
        if canonical_candidates:
            self._set_dashboard_message(
                max(canonical_candidates, key=self._dashboard_message_key)
            )
        elif previous_key in pending_delete_keys:
            self._set_dashboard_message(None)
        canonical_key = (
            self._dashboard_message_key(self._message) if self._message is not None else None
        )
        self._persisted_dashboard_message_id = self._dashboard_message_id(self._message)
        if canonical_key != previous_key:
            self._last_payload_fingerprint = None
        if canonical_key is not None:
            self._pending_dashboard_deletes.pop(canonical_key, None)
            self._pending_dashboard_delete_ids.discard(canonical_key)
        for stale in [*history_messages, previous]:
            if stale is not None:
                self._queue_dashboard_delete(stale)

        pending_nonce = self._dashboard_send_nonce
        nonce_resolved = pending_nonce is not None and any(
            str(getattr(message, "nonce", "")) == str(pending_nonce)
            for message in [*history_messages, previous]
            if message is not None
        )
        predecessor_id = self._dashboard_send_predecessor_id
        if nonce_resolved and predecessor_id is not None and predecessor_id != canonical_key:
            self._pending_dashboard_delete_ids.add(predecessor_id)

        unresolved_pending_id = False
        for message_id in tuple(self._pending_dashboard_delete_ids):
            if message_id == canonical_key:
                self._pending_dashboard_delete_ids.discard(message_id)
                continue
            if message_id in self._pending_dashboard_deletes:
                continue
            try:
                stale = await _bounded_discord(channel.fetch_message(message_id))
            except Exception as exc:  # noqa: BLE001
                if _is_discord_not_found(exc):
                    self._pending_dashboard_delete_ids.discard(message_id)
                else:
                    log.warning("persisted stale dashboard fetch deferred: %s", exc)
                    unresolved_pending_id = True
                continue
            if _is_dashboard_message(stale, expected_author):
                self._queue_dashboard_delete(stale)
            else:
                self._pending_dashboard_delete_ids.discard(message_id)

        # Discord omits ``nonce`` from fetched messages in some responses.  An
        # expired episode can still be resolved without another create when the
        # persisted canonical is the sole dashboard, was created in the exact
        # uncertain-send window, and the recorded predecessor is confirmed gone.
        inferred_successor = False
        started_at = self._rollover_uncertain_since or 0.0
        nonce_age = max(0.0, time.time() - started_at)
        dashboard_keys = {
            self._dashboard_message_key(message)
            for message in history_messages
            if _is_dashboard_message(message, expected_author)
        }
        if (
            pending_nonce is not None
            and nonce_age >= DASHBOARD_SEND_NONCE_RETRY_SECONDS
            and predecessor_id is not None
            and canonical_key is not None
            and canonical_key == persisted_message_id
            and canonical_key != predecessor_id
            and dashboard_keys == {canonical_key}
            and _is_dashboard_message(self._message, expected_author)
            and self._message_matches_uncertain_send_window(self._message, started_at)
        ):
            try:
                await _bounded_discord(channel.fetch_message(predecessor_id))
            except Exception as exc:  # noqa: BLE001
                if _is_discord_not_found(exc):
                    inferred_successor = True
                else:
                    log.warning("rollover predecessor verification deferred: %s", exc)
        if inferred_successor:
            log.info(
                "resolved expired dashboard send from canonical creation window "
                "and missing predecessor"
            )

        if pending_nonce is None:
            # Backward compatibility for timestamp-only recovery state written
            # before dashboard sends acquired an idempotency nonce.
            self._rollover_uncertain_since = None
            self._dashboard_send_predecessor_id = None
        elif nonce_resolved or inferred_successor:
            self._rollover_uncertain_since = None
            self._dashboard_send_nonce = None
            self._dashboard_send_predecessor_id = None
            self._dashboard_send_had_ambiguous_attempt = False
        else:
            self._dashboard_send_retry_ready = (
                not unresolved_pending_id and nonce_age < DASHBOARD_SEND_NONCE_RETRY_SECONDS
            )
        # Derive readiness from the state after the resolution branch.  This
        # keeps clearing a proven nonce and enabling thread/alert side effects
        # one atomic transition even when tests or callbacks interleave ticks.
        self._dashboard_messages_reconciled = not unresolved_pending_id and (
            self._dashboard_send_nonce is None or self._dashboard_send_retry_ready
        )
        self._persist_dashboard_runtime_state()
        self._start_dashboard_cleanup_if_idle()

    async def _retry_uncertain_dashboard_send(
        self,
        channel,
        payload: DashboardPayload,
    ) -> object:
        """Resolve one recent ambiguous create with its original nonce."""
        nonce = self._dashboard_send_nonce
        if nonce is None or not self._dashboard_send_retry_ready:
            raise RuntimeError("dashboard send retry is not reconciled")
        if not self._persist_dashboard_runtime_state():
            self._dashboard_send_retry_ready = False
            self._dashboard_messages_reconciled = False
            raise RuntimeError("cannot persist dashboard-send recovery state; retry deferred")
        previous = self._message
        predecessor_id = self._dashboard_send_predecessor_id
        self._dashboard_send_retry_ready = False
        self._dashboard_messages_reconciled = False
        # On failure, retain the immutable first-attempt timestamp and nonce;
        # the next tick must scan the same history window before retrying.
        replacement = await _bounded_discord(
            _send_payload(
                channel,
                payload,
                view=self.live_view,
                nonce=nonce,
            )
        )

        candidates = [message for message in (previous, replacement) if message is not None]
        self._set_dashboard_message(max(candidates, key=self._dashboard_message_key))
        self._persisted_dashboard_message_id = self._dashboard_message_id(self._message)
        # A nonce replay returns the payload accepted by the original POST,
        # which may predate this tick. Let the normal edit path apply and hash
        # the current payload before claiming its fingerprint.
        self._last_payload_fingerprint = None
        for stale in candidates:
            self._queue_dashboard_delete(stale)

        canonical_key = self._dashboard_message_key(self._message)
        known_keys = {self._dashboard_message_key(message) for message in candidates}
        if predecessor_id is not None and predecessor_id != canonical_key:
            if predecessor_id not in known_keys:
                try:
                    predecessor = await _bounded_discord(channel.fetch_message(predecessor_id))
                except Exception as exc:  # noqa: BLE001
                    if not _is_discord_not_found(exc):
                        log.warning(
                            "rollover predecessor fetch deferred: %s",
                            exc,
                        )
                        self._pending_dashboard_delete_ids.add(predecessor_id)
                else:
                    expected_author = getattr(getattr(channel, "guild", None), "me", None)
                    if _is_dashboard_message(predecessor, expected_author):
                        self._queue_dashboard_delete(predecessor)
            elif predecessor_id not in self._pending_dashboard_deletes:
                self._pending_dashboard_delete_ids.add(predecessor_id)
        unresolved_predecessor = (
            predecessor_id is not None
            and predecessor_id != canonical_key
            and predecessor_id not in self._pending_dashboard_deletes
            and predecessor_id in self._pending_dashboard_delete_ids
        )
        self._clear_rollover_send_uncertainty()
        self._dashboard_messages_reconciled = not unresolved_predecessor
        self._persist_dashboard_runtime_state()
        self._start_dashboard_cleanup_if_idle()
        return self._message

    async def _delete_stale_dashboard(self, key: int, message) -> int:
        if key == self._persisted_dashboard_message_id:
            return key
        log.info(
            "dashboard deletion requested channel_id=%s message_id=%s canonical_message_id=%s "
            "reason=duplicate_recovery",
            self.settings.announce_channel_id, key, self._persisted_dashboard_message_id,
        )
        try:
            await _bounded_discord(message.delete())
        except Exception as exc:  # noqa: BLE001
            if not _is_discord_not_found(exc):
                raise
        return key

    def _dashboard_cleanup_finished(self, task: asyncio.Task) -> None:
        if self._dashboard_cleanup_task is task:
            self._dashboard_cleanup_task = None
        if task.cancelled():
            return
        try:
            key = task.result()
        except Exception as exc:  # noqa: BLE001
            log.warning("stale dashboard cleanup failed; will retry: %s", exc)
            return
        self._pending_dashboard_deletes.pop(key, None)
        self._pending_dashboard_delete_ids.discard(key)
        self._persist_dashboard_runtime_state()

    def _start_dashboard_cleanup_if_idle(self) -> None:
        task = self._dashboard_cleanup_task
        if task is not None and not task.done():
            return
        if task is not None:
            self._dashboard_cleanup_finished(task)
        if self._persisted_dashboard_message_id is not None:
            canonical_key = self._persisted_dashboard_message_id
            self._pending_dashboard_deletes.pop(canonical_key, None)
            self._pending_dashboard_delete_ids.discard(canonical_key)
        if not self._pending_dashboard_deletes:
            return
        self._persist_dashboard_runtime_state()
        key, message = next(iter(self._pending_dashboard_deletes.items()))
        task = asyncio.create_task(self._delete_stale_dashboard(key, message))
        self._dashboard_cleanup_task = task
        task.add_done_callback(self._dashboard_cleanup_finished)

    def _adopt_dashboard_rollover(
        self,
        result: tuple[object, object | None],
        fingerprint: str,
    ) -> object:
        replacement, stale = result
        self._set_dashboard_message(replacement)
        self._persisted_dashboard_message_id = self._dashboard_message_id(replacement)
        self._last_payload_fingerprint = fingerprint
        if stale is not None:
            # Persist cleanup ownership before clearing the send episode. If
            # either write fails, a restart retains at least one recovery path
            # to the predecessor instead of orphaning it outside history.
            self._queue_dashboard_delete(stale)
        self._clear_rollover_send_uncertainty()
        # The replacement response itself proves the canonical message.  Keep
        # status-thread creation in this tick instead of waiting for another
        # history reconciliation pass.
        self._dashboard_messages_reconciled = True
        if stale is not None:
            self._start_dashboard_cleanup_if_idle()
        return replacement

    async def _rollover_dashboard(
        self,
        channel,
        payload: DashboardPayload,
        fingerprint: str,
    ) -> object:
        """Finish and adopt one bounded rollover even if the loop is stopped."""
        if self._dashboard_send_nonce is not None:
            raise RuntimeError("uncertain dashboard send must reconcile before rollover")
        if not self._mark_rollover_send_uncertain():
            raise RuntimeError("cannot persist rollover recovery state; replacement deferred")
        attempt_evidence = _DashboardSendAttemptEvidence()
        task = asyncio.create_task(
            _rollover_dashboard_message(
                channel,
                self._message,
                payload,
                view=self.live_view,
                nonce=self._dashboard_send_nonce,
                attempt_evidence=attempt_evidence,
            )
        )
        self._rollover_task = task
        try:
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError as cancelled:
                # Do not abandon a fully-sent successor during shutdown.  Each
                # underlying request has its own short deadline, so completing
                # the ownership handoff is bounded.
                try:
                    result = await asyncio.shield(task)
                except Exception as exc:  # noqa: BLE001
                    self._record_dashboard_send_failure(exc, attempt_evidence)
                    log.warning("dashboard rollover did not complete during stop: %s", exc)
                else:
                    self._adopt_dashboard_rollover(result, fingerprint)
                raise cancelled
            return self._adopt_dashboard_rollover(result, fingerprint)
        except Exception as exc:
            # The POST may have committed before its response was lost.  A
            # strict time-window history reconciliation must succeed before
            # another replacement can be attempted.
            self._record_dashboard_send_failure(exc, attempt_evidence)
            raise
        finally:
            if task.done() and self._rollover_task is task:
                self._rollover_task = None

    async def _stored_status_thread(self):
        """Resolve a persisted thread only when it belongs to this message."""
        thread_id = self._status_thread_id
        message_id = self._dashboard_message_id(self._message)
        if thread_id is None or thread_id != message_id or self._message is None:
            return None
        fetch_channel = getattr(getattr(self._message, "guild", None), "fetch_channel", None)
        if fetch_channel is None:
            return None
        try:
            thread = await _bounded_discord(fetch_channel(thread_id))
        except Exception as exc:  # noqa: BLE001
            if _is_discord_not_found(exc):
                return None
            raise
        if not self._thread_belongs_to_dashboard(thread):
            log.warning("persisted status thread is not attached to dashboard; ignoring")
            return None
        return thread

    def _thread_belongs_to_dashboard(self, thread) -> bool:
        message_id = self._dashboard_message_id(self._message)
        thread_id = getattr(thread, "id", None)
        if message_id is None or thread_id != message_id:
            return False
        parent_id = getattr(thread, "parent_id", None)
        return parent_id is None or parent_id == self.settings.announce_channel_id

    def _remember_status_thread(self, thread) -> None:
        thread_id = getattr(thread, "id", None)
        if not isinstance(thread_id, int) or thread_id <= 0:
            return
        self._status_thread_id = thread_id
        if (
            thread_id != self._persisted_status_thread_id
            and self._persist_dashboard_runtime_state()
        ):
            self._persisted_status_thread_id = thread_id

    async def _fetch_attached_status_thread(self):
        """Resolve by starter ID, including archived threads absent from cache."""
        fetch_thread = getattr(self._message, "fetch_thread", None)
        fetch_channel = getattr(getattr(self._message, "guild", None), "fetch_channel", None)
        try:
            if fetch_thread is not None:
                thread = await _bounded_discord(fetch_thread())
            elif fetch_channel is not None:
                thread = await _bounded_discord(fetch_channel(self._message.id))
            else:
                return None
        except Exception as exc:
            if _is_discord_not_found(exc):
                return None
            raise
        return thread if self._thread_belongs_to_dashboard(thread) else None

    async def _ensure_thread(self) -> None:
        """Create the updates thread under the dashboard message once."""
        if self._message is None:
            return
        endpoint = f"GET /channels/{self._message.id}"
        started = time.monotonic()
        resolution = "updater_cache"
        try:
            # Public thread IDs equal their starter-message IDs.  Archived
            # threads are absent from ``channel.threads``, so resolve the
            # thread through its dashboard message before trying to create it.
            thread = self._thread
            if not self._thread_belongs_to_dashboard(thread):
                thread = None
            if thread is None:
                resolution = "message_cache"
                thread = getattr(self._message, "thread", None)
                if not self._thread_belongs_to_dashboard(thread):
                    thread = None
            if thread is None:
                resolution = "persisted_id"
                thread = await self._stored_status_thread()
            if thread is None:
                resolution = "channel_cache"
                thread = next(
                    (
                        item
                        for item in getattr(getattr(self._message, "channel", None), "threads", [])
                        if getattr(item, "id", None) == self._message.id
                    ),
                    None,
                )
                if not self._thread_belongs_to_dashboard(thread):
                    thread = None
            if thread is None:
                resolution = "starter_id_fetch"
                thread = await self._fetch_attached_status_thread()
            if thread is None:
                resolution = "created"
                endpoint = (
                    f"POST /channels/{self.settings.announce_channel_id}"
                    f"/messages/{self._message.id}/threads"
                )
                try:
                    thread = await _bounded_discord(
                        self._message.create_thread(name="status updates")
                    )
                except Exception as exc:
                    if getattr(exc, "code", None) != 160004:
                        raise
                    # A previous creation may have committed after its request
                    # timed out, or the gateway cache may lag behind REST.
                    resolution = "already_created_fetch"
                    endpoint = f"GET /channels/{self._message.id}"
                    thread = await self._fetch_attached_status_thread()
                if not self._thread_belongs_to_dashboard(thread):
                    raise RuntimeError("status thread is not attached to dashboard")
            was_archived = getattr(thread, "archived", False)
            if was_archived:
                endpoint = f"PATCH /channels/{thread.id}"
                updated = await _bounded_discord(thread.edit(archived=False))
                if updated is not None:
                    thread = updated
            if self._thread is not thread or was_archived:
                log.info(
                    "resolved status thread channel_id=%s message_id=%s thread_id=%s "
                    "parent_id=%s via=%s unarchived=%s archived=%s locked=%s "
                    "pending_alerts=%d elapsed_s=%.3f",
                    self.settings.announce_channel_id, self._message.id, thread.id,
                    getattr(thread, "parent_id", None), resolution, was_archived,
                    getattr(thread, "archived", None), getattr(thread, "locked", None),
                    len(self._pending_alert_messages), time.monotonic() - started,
                )
            self._thread = thread
            self._remember_status_thread(thread)
        except Exception as exc:  # noqa: BLE001
            self._thread = None
            log.warning(
                "%s could not resolve status thread channel_id=%s message_id=%s "
                "persisted_thread_id=%s via=%s error=%s status=%s code=%s "
                "elapsed_s=%.3f timeout_s=%.1f pending_alerts=%d: %s",
                endpoint, self.settings.announce_channel_id, self._message.id,
                self._persisted_status_thread_id, resolution, type(exc).__name__,
                getattr(exc, "status", None), getattr(exc, "code", None),
                time.monotonic() - started, DASHBOARD_DISCORD_TIMEOUT_SECONDS,
                len(self._pending_alert_messages), exc, exc_info=True,
            )
            if getattr(exc, "status", None) == 403:
                _diagnose_dashboard_permissions(getattr(self._message, "channel", None))

    async def _flush_alert_messages(self) -> None:
        """Deliver queued thread content in order, retaining failures for retry."""
        while self._pending_alert_messages and self._thread is not None:
            text, permit_alert_role = self._pending_alert_messages[0]
            role_id = self.settings.alert_role_id
            allowed_roles: bool | list[discord.Object] = False
            if permit_alert_role and role_id is not None:
                allowed_roles = [discord.Object(id=role_id)]
            allowed_mentions = discord.AllowedMentions(
                everyone=False,
                users=False,
                roles=allowed_roles,
                replied_user=False,
            )
            started = time.monotonic()
            try:
                sent = await _bounded_discord(
                    self._thread.send(
                        content=text,
                        allowed_mentions=allowed_mentions,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                # Force a fresh resolve/unarchive on the next presentation tick.
                thread_id = getattr(self._thread, "id", None)
                self._thread = None
                log.warning(
                    "POST /channels/%s/messages failed (retaining alert for retry) "
                    "channel_id=%s message_id=%s error=%s status=%s code=%s "
                    "pending_alerts=%d elapsed_s=%.3f timeout_s=%.1f: %s",
                    thread_id, self.settings.announce_channel_id,
                    self._dashboard_message_id(self._message), type(exc).__name__,
                    getattr(exc, "status", None), getattr(exc, "code", None),
                    len(self._pending_alert_messages), time.monotonic() - started,
                    DASHBOARD_DISCORD_TIMEOUT_SECONDS, exc, exc_info=True,
                )
                return
            self._pending_alert_messages.popleft()
            log.info(
                "status update delivered channel_id=%s message_id=%s thread_id=%s "
                "update_message_id=%s remaining_alerts=%d elapsed_s=%.3f",
                self.settings.announce_channel_id, self._dashboard_message_id(self._message),
                getattr(self._thread, "id", None), getattr(sent, "id", None),
                len(self._pending_alert_messages), time.monotonic() - started,
            )

    async def _post_alert_events(self, results: dict[str, object]) -> None:
        """Feed the alert monitor and post any thread messages."""

        weather_result = results.get("weather")
        warnings = []
        if isinstance(weather_result, tuple) and len(weather_result) == 3:
            warnings = weather_result[1]
        traffic_result = results.get("traffic")
        statuses = []
        incidents = []
        roadworks = []
        if isinstance(traffic_result, tuple):
            statuses = traffic_result[0]
            if len(traffic_result) > 1:
                incidents = traffic_result[1]
            if len(traffic_result) > 2:
                roadworks = traffic_result[2]
        incidents = filter_expired_rthk_incidents(
            incidents, now=datetime.now(UTC),
            max_age_hours=self.settings.rthk_news_max_age_hours,
        )
        roads = results.get("tracked_roads")
        if roads is not None and not isinstance(roads, Exception):
            self.alerts.roads = roads
        events = self.alerts.update(warnings, statuses, incidents, roadworks)
        for event in events:
            messages = self.alerts.messages_for(event, self.settings.alert_role_id)
            self._pending_alert_messages.extend(
                (text, event.ping and index == 0)
                for index, text in enumerate(messages)
            )
        await self._flush_alert_messages()

    def _queue_alert_snapshot_if_ready(self, snapshot: CollectionSnapshot | None) -> None:
        """Retain each completed alert generation until the presenter consumes it."""
        if snapshot is None or snapshot.generation <= self._last_queued_alert_generation:
            return
        settled = snapshot.settled_providers
        if settled and not {"weather", "traffic"}.issubset(settled):
            return
        self._pending_alert_snapshots.append(snapshot)
        self._last_queued_alert_generation = snapshot.generation

    async def _process_alert_snapshot(self) -> None:
        """Process complete alert snapshots in order and retry queued messages."""
        await self._flush_alert_messages()
        # Directly injected test/dev snapshots do not pass through the normal
        # publisher, so offer the current value to the same queue here.
        self._queue_alert_snapshot_if_ready(self._snapshot)
        while self._pending_alert_snapshots:
            snapshot = self._pending_alert_snapshots[0]
            if snapshot.generation <= self._last_alert_generation:
                self._pending_alert_snapshots.popleft()
                continue
            await self._post_alert_events(snapshot.results)
            # Keep the generation paired with the exact queued results above;
            # Discord I/O can allow a newer snapshot to publish while we await.
            self._last_alert_generation = snapshot.generation
            self._pending_alert_snapshots.popleft()

    async def _update_loop(self, channel=None) -> None:
        # Start collection promptly, but never make rendering wait for it.
        self._start_collection_if_idle()
        next_tick = time.monotonic()
        while self._running:
            try:
                await self._tick(channel)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "update tick failed channel_id=%s message_id=%s thread_id=%s "
                    "collection_generation=%s error=%s: %s",
                    self.settings.announce_channel_id, self._dashboard_message_id(self._message),
                    getattr(self._thread, "id", None), self._collection_generation,
                    type(exc).__name__, exc, exc_info=True,
                )
            next_tick += self.settings.update_interval_seconds
            # A slow provider or Discord edit can miss several presentation
            # windows.  Skip those deadlines instead of replaying them as a
            # zero-sleep catch-up burst; the next tick is the next real window.
            now = time.monotonic()
            if next_tick <= now:
                interval = max(0.001, self.settings.update_interval_seconds)
                next_tick = now + interval
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))

    def _start_collection_if_idle(self) -> None:
        """Launch at most one async collection; map capture stays single-flight.

        ``collect_all`` starts the Google Maps capture only once per
        collection.  Keeping the enclosing task single-flight therefore also
        prevents overlapping Playwright captures and cache-file races.
        """
        if not self._running or self.client is None:
            return
        if self._collection_task is not None and not self._collection_task.done():
            return
        self._collection_generation += 1
        generation = self._collection_generation
        publish = lambda name, value: self._publish_provider_result(  # noqa: E731
            generation, name, value
        )
        # Keeping this small compatibility branch makes injected test/dev
        # collectors from before snapshots continue to work; production's
        # ``collect_all`` always has the incremental callback.
        parameters = inspect.signature(collect_all).parameters
        self._independent_map_enabled = "include_traffic_map" in parameters
        kwargs = {"on_result": publish} if "on_result" in parameters else {}
        if "tracker" in parameters:
            kwargs["tracker"] = self.marker_tracker
        if self._independent_map_enabled:
            kwargs["include_traffic_map"] = False
        if kwargs:
            collection = collect_all(self.client, self.settings, **kwargs)
        else:
            collection = collect_all(self.client, self.settings)
        task = asyncio.create_task(collection)
        self._collection_task = task
        task.add_done_callback(self._collection_finished)

    def _start_map_if_idle(self) -> None:
        if not self._running or self.client is None or not self._independent_map_enabled:
            return
        if self._map_task is not None:
            if not self._map_task.done():
                return
            self._map_finished(self._map_task, self._map_generation)
        results = dict(self._snapshot.results) if self._snapshot else {}
        generation = self._collection_generation
        self._map_generation = generation
        self._map_task = asyncio.create_task(
            _fetch_traffic_map_from_results(
                self.client, self.settings, results, self.marker_tracker
            )
        )
        self._map_task.add_done_callback(
            lambda done, generation=generation: self._map_finished(done, generation)
        )

    def _map_finished(self, task: asyncio.Task, generation: int | None) -> None:
        if self._map_task is not task:
            return
        self._map_task = None
        self._map_generation = None
        if task.cancelled():
            return
        try:
            value = task.result()
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "background map refresh failed collection_generation=%s error=%s: %s",
                generation, type(exc).__name__, exc, exc_info=True,
            )
            value = exc
        if generation is not None:
            # A map capture is an independent single-flight stream.  Its task
            # identity was checked above, so an ordinary collection starting
            # while it was running does not make this newest map obsolete.
            self._publish_provider_result(generation, "traffic_map", value, independent_map=True)

    def _collection_finished(self, task: asyncio.Task) -> None:
        """Publish a completed collection without blocking the presenter."""
        if self._collection_task is not task:
            # A cancelled/replaced task can still run its done callback.
            return
        self._collection_task = None
        if task.cancelled():
            return
        try:
            fresh = task.result()
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "background collection failed collection_generation=%s error=%s: %s",
                self._collection_generation, type(exc).__name__, exc, exc_info=True,
            )
            return
        if not isinstance(fresh, dict):
            log.warning("background collection returned invalid result")
            return

        # ``collect_all`` publishes providers individually.  This fallback is
        # retained for alternate collectors that do not invoke the callback.
        # An independent map may have advanced the snapshot while this
        # collector was finishing, so use settled-provider identity instead
        # of comparing only snapshot generations.
        snapshot = self._snapshot
        settled = (
            snapshot.settled_providers
            if snapshot is not None and snapshot.generation == self._collection_generation
            else frozenset()
        )
        for name, value in fresh.items():
            if name not in settled:
                self._publish_provider_result(self._collection_generation, name, value)

    def _publish_provider_result(
        self,
        generation: int,
        name: str,
        value: object,
        *,
        independent_map: bool = False,
    ) -> None:
        """Atomically merge one provider into the current last-good snapshot."""
        if not self._running:
            return
        if independent_map:
            if name != "traffic_map":
                return
            # The single-flight task identity in ``_map_finished`` is the
            # freshness fence.  Publish into the newest ordinary snapshot,
            # retaining every provider value that arrived since map start.
            generation = self._collection_generation
        elif generation != self._collection_generation:
            # Ordinary callbacks from a cancelled/replaced collection must not
            # overwrite a newer generation.
            return
        previous = self._snapshot
        merged: dict[str, object] = {}
        stale: set[str] = set()
        settled: set[str] = set()
        if previous is not None:
            merged.update(previous.results)
            stale.update(previous.stale_providers)
            if previous.generation == generation:
                settled.update(previous.settled_providers)
            # At the beginning of a new collection, all retained values are
            # stale until their owning provider has supplied this generation.
            if previous.generation != generation:
                stale.update(merged)
        if (
            isinstance(value, Exception)
            and name in merged
            and not isinstance(merged[name], Exception)
        ):
            stale.add(name)
        else:
            merged[name] = value
            if isinstance(value, Exception):
                stale.add(name)
            else:
                stale.discard(name)
        settled.add(name)
        self._snapshot = CollectionSnapshot(
            results=merged,
            generation=generation,
            completed_monotonic=time.monotonic(),
            stale_providers=frozenset(stale),
            settled_providers=frozenset(settled),
        )
        self._queue_alert_snapshot_if_ready(self._snapshot)

    def _snapshot_payload(self) -> DashboardPayload | None:
        """Build from the latest completed snapshot, never from a live fetch."""
        if self._snapshot is None:
            return None
        return _to_payload(
            self._snapshot.results,
            rthk_news_max_age_hours=self.settings.rthk_news_max_age_hours,
        )

    def _log_dashboard_health(self, payload: DashboardPayload) -> None:
        """Report displayed availability on transitions and every five minutes."""
        snapshot = self._snapshot
        if snapshot is None:
            return
        mr = snapshot.results.get("traffic_map")
        now = datetime.now(UTC)
        capture_age = base_age = None
        if isinstance(mr, TrafficMapResult):
            if mr.captured_at is not None:
                capture_age = round((now - mr.captured_at).total_seconds(), 1)
            base_time = mr.base_updated_at or mr.captured_at
            if base_time is not None:
                base_age = round((now - base_time).total_seconds(), 1)
        has_map = any(_is_traffic_map_filename(asset.filename) for asset in payload.files)
        if "traffic_map" not in snapshot.results:
            map_state = "initializing"
        elif has_map:
            map_state = "cached" if (
                isinstance(mr, TrafficMapResult) and (
                    mr.stale or (capture_age is not None and capture_age > MAP_CAPTURE_STALE_AFTER_SECONDS)
                )
            ) else "available"
        elif isinstance(mr, Exception):
            map_state = f"failed_{type(mr).__name__}"
        elif base_age is not None and base_age > MAP_CAPTURE_MAX_AGE_SECONDS:
            map_state = "base_expired"
        else:
            map_state = "capture_unavailable"
        weather = snapshot.results.get("weather")
        warnings = weather.warnings if isinstance(weather, WeatherConditions) else (
            weather[1] if isinstance(weather, tuple) and len(weather) == 3 else []
        )
        warning_codes = tuple(warning.code for warning in warnings)
        missing_icons = tuple(warning.code for warning in warnings if not warning.icon_data)
        has_strip = any(asset.label == "HKO warning icons" for asset in payload.files)
        signature = (
            map_state, tuple(sorted(snapshot.stale_providers)), warning_codes, missing_icons,
            has_strip, self._dashboard_message_id(self._message), getattr(self._thread, "id", None),
            len(self._pending_alert_messages),
        )
        monotonic_now = time.monotonic()
        if signature == self._last_health_signature and monotonic_now - self._last_health_log_at < 300:
            return
        self._last_health_signature = signature
        self._last_health_log_at = monotonic_now
        log.info(
            "dashboard health channel_id=%s message_id=%s thread_id=%s collection_generation=%s "
            "map_state=%s map_capture_age_s=%s map_base_age_s=%s map_task_running=%s "
            "stale_providers=%s warning_codes=%s missing_warning_icons=%s warning_strip=%s "
            "pending_alerts=%d pending_alert_snapshots=%d edit_retry_remaining_s=%.1f",
            self.settings.announce_channel_id, self._dashboard_message_id(self._message),
            getattr(self._thread, "id", None), snapshot.generation, map_state,
            capture_age, base_age, self._map_task is not None and not self._map_task.done(),
            sorted(snapshot.stale_providers), warning_codes, missing_icons, has_strip,
            len(self._pending_alert_messages), len(self._pending_alert_snapshots),
            max(0.0, self._dashboard_edit_retry_at - monotonic_now),
        )

    async def _tick(self, channel=None) -> None:
        # The presenter is deliberately independent of provider latency.  It
        # starts the next refresh when idle then reads only a completed snapshot.
        self._start_collection_if_idle()
        if self._independent_map_enabled:
            self._start_map_if_idle()
        # Let an already-ready task publish its callback, without awaiting a
        # slow provider.  This also makes fast local/dry-run providers visible
        # on the first presentation.
        await asyncio.sleep(0)
        if self._collection_task is not None and self._collection_task.done():
            self._collection_finished(self._collection_task)
        payload_snapshot = self._snapshot
        payload = self._snapshot_payload()
        if payload is None:
            return
        self._log_dashboard_health(payload)
        # Keep the generation paired with this payload.  The edit awaits
        # Discord I/O, during which a newer provider snapshot may publish.
        payload_generation = payload_snapshot.generation if payload_snapshot else 0

        if channel is None:
            # dry-run / dev: just keep the last payload for inspection
            self._last_good_payload = payload
            return

        # Reconcile persisted uncertain sends before any create.  This strict
        # path can recover a committed replacement outside recent history.
        await self._reconcile_dashboard_messages(channel)
        self._start_dashboard_cleanup_if_idle()
        if self._dashboard_send_nonce is not None:
            if not self._dashboard_send_retry_ready:
                # A failed scan, unresolved cleanup, or expired nonce window
                # cannot authorize another possibly creating request.
                if self._message is None:
                    return
            else:
                self._set_dashboard_message(
                    await self._retry_uncertain_dashboard_send(channel, payload)
                )
        elif self._rollover_uncertain_since is not None:
            # Legacy timestamp-only state still requires one successful scan.
            return
        if self._message is None and not self._dashboard_messages_reconciled:
            return
        # Ensure the message only after reconciliation proves that no prior
        # uncertain create needs adoption.
        if self._message is None:
            if not self._mark_rollover_send_uncertain():
                raise RuntimeError("cannot persist dashboard-send recovery state; create deferred")
            attempt_evidence = _DashboardSendAttemptEvidence()
            try:
                self._set_dashboard_message(
                    await _bounded_discord(
                        _ensure_dashboard_message(
                            channel,
                            payload,
                            view=self.live_view,
                            nonce=self._dashboard_send_nonce,
                            attempt_evidence=attempt_evidence,
                        )
                    )
                )
            except asyncio.CancelledError as exc:
                self._record_dashboard_send_failure(exc, attempt_evidence)
                raise
            except Exception as exc:
                # The initial POST has the same accepted-but-response-lost
                # ambiguity as a rollover. Keep the persisted timestamp and
                # require a strict history reconciliation before another send.
                self._record_dashboard_send_failure(exc, attempt_evidence)
                raise
            self._persisted_dashboard_message_id = self._dashboard_message_id(self._message)
            self._clear_rollover_send_uncertainty()
        fingerprint = _payload_fingerprint(payload)
        edit_started = time.monotonic()
        try:
            if (
                fingerprint != self._last_payload_fingerprint
                and time.monotonic() >= self._dashboard_edit_retry_at
            ):
                edited_message = await _bounded_discord(
                    _apply_payload(self._message, payload, view=self.live_view)
                )
                if edited_message is not None:
                    # discord.py 2.x returns the edited Message rather than
                    # mutating this object in place. Retain it so its
                    # Attachment objects can be passed through next time.
                    self._set_dashboard_message(edited_message)
                self._last_payload_fingerprint = fingerprint
                self._dashboard_edit_retry_at = 0.0
                map_asset = next(
                    (
                        asset
                        for asset in payload.files
                        if asset.filename == traffic_map_filename(asset.data)
                    ),
                    None,
                )
                if map_asset is not None:
                    map_hash = hashlib.sha256(map_asset.data).hexdigest()[:12]
                    log.info(
                        "dashboard edit succeeded collection_generation=%s "
                        "payload_fingerprint=%s traffic_map_sha256=%s "
                        "traffic_map_filename=%s channel_id=%s message_id=%s elapsed_s=%.3f",
                        payload_generation,
                        fingerprint[:12],
                        map_hash,
                        map_asset.filename,
                        self.settings.announce_channel_id, self._dashboard_message_id(self._message),
                        time.monotonic() - edit_started,
                    )
                else:
                    log.info(
                        "dashboard edit succeeded collection_generation=%s "
                        "payload_fingerprint=%s traffic_map_sha256=none "
                        "traffic_map_filename=none channel_id=%s message_id=%s elapsed_s=%.3f",
                        payload_generation,
                        fingerprint[:12],
                        self.settings.announce_channel_id, self._dashboard_message_id(self._message),
                        time.monotonic() - edit_started,
                    )
                self._last_good_payload = payload
        except Exception as exc:  # noqa: BLE001
            retry_delay = _discord_edit_retry_delay(exc)
            message_id = self._dashboard_message_id(self._message)
            endpoint = (
                f"PATCH /channels/{self.settings.announce_channel_id}/messages/{message_id}"
            )
            if retry_delay is not None:
                self._dashboard_edit_retry_at = time.monotonic() + retry_delay
                log.warning(
                    "%s rate limited (%s, code=%s); keeping message/thread, retry in %.1fs "
                    "status=%s thread_id=%s elapsed_s=%.3f",
                    endpoint, type(exc).__name__, getattr(exc, "code", None), retry_delay,
                    getattr(exc, "status", None), getattr(self._thread, "id", None),
                    time.monotonic() - edit_started,
                )
            elif _is_discord_not_found(exc):
                # Only confirmed deletion starts discovery/recovery. An edit
                # quota or transient failure must never create a successor.
                self._set_dashboard_message(None)
                self._last_payload_fingerprint = None
                self._dashboard_messages_reconciled = False
                log.warning("%s: message missing; rediscovering dashboard", endpoint)
            else:
                log.warning(
                    "%s failed (keeping last good) error=%s status=%s code=%s "
                    "thread_id=%s elapsed_s=%.3f timeout_s=%.1f: %s",
                    endpoint, type(exc).__name__, getattr(exc, "status", None),
                    getattr(exc, "code", None), getattr(self._thread, "id", None),
                    time.monotonic() - edit_started, DASHBOARD_DISCORD_TIMEOUT_SECONDS, exc,
                    exc_info=True,
                )
                if getattr(exc, "status", None) == 403:
                    _diagnose_dashboard_permissions(channel)

        # An unresolved send may reuse a known canonical for ordinary edits,
        # but cannot run status/alert side effects until reconciliation proves
        # which wire attempt owns the dashboard.
        if self._dashboard_send_nonce is not None and not self._dashboard_messages_reconciled:
            return

        # status thread + alerts (create the thread after the message exists)
        if self._dashboard_messages_reconciled:
            await self._ensure_thread()
        await self._process_alert_snapshot()

    async def stop(self) -> None:
        self._running = False
        await self.live_frames.stop()
        if self.live_snapshot_task is not None and not self.live_snapshot_task.done():
            self.live_snapshot_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.live_snapshot_task
        self.live_snapshot_task = None
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._loop_task
            self._loop_task = None
        rollover_task = self._rollover_task
        if rollover_task is not None and not rollover_task.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.shield(rollover_task)
        self._rollover_task = None
        cleanup_task = self._dashboard_cleanup_task
        if cleanup_task is not None and not cleanup_task.done():
            cleanup_task.cancel()
        if cleanup_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await cleanup_task
        self._dashboard_cleanup_task = None
        collection_task = self._collection_task
        if collection_task is not None and not collection_task.done():
            collection_task.cancel()
        if collection_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await collection_task
        self._collection_task = None
        map_task = self._map_task
        if map_task is not None and not map_task.done():
            map_task.cancel()
        if map_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await map_task
        self._map_task = None
        self._map_generation = None
        self.marker_tracker.clear()
        await pipeline.shutdown_background_resources()
        if self.session is not None:
            await self.session.close()
            self.session = None
        self.client = None


# --------------------------------------------------------------------------
# Discord bot wiring
# --------------------------------------------------------------------------


def _diagnose_dashboard_permissions(channel) -> None:
    """Inspect cached effective permissions without making another request."""
    guild = getattr(channel, "guild", None)
    member = getattr(guild, "me", None)
    try:
        permissions = channel.permissions_for(member)
    except Exception as exc:
        log.warning(
            "dashboard permissions unavailable channel_id=%s error=%s",
            getattr(channel, "id", None), type(exc).__name__,
        )
        return
    required = (
        "view_channel", "send_messages", "embed_links", "attach_files", "read_message_history",
        "create_public_threads", "send_messages_in_threads",
    )
    missing = [name for name in required if not getattr(permissions, name, False)]
    log.log(
        logging.WARNING if missing else logging.INFO,
        "dashboard permissions channel_id=%s guild_id=%s bot_id=%s missing=%s",
        getattr(channel, "id", None), getattr(guild, "id", None), getattr(member, "id", None), missing,
    )


def _retained_thumbnail_url(attachment) -> str | None:
    """Return a stable CDN URL; Discord renews embed URL signatures itself."""
    url = getattr(attachment, "url", None)
    if not isinstance(url, str):
        return None
    try:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in {
            "cdn.discordapp.com", "media.discordapp.net",
        }:
            return None
        if parsed.path.split("/")[-2:] != [str(attachment.id), attachment.filename]:
            return None
    except (ValueError, TypeError, AttributeError):
        return None
    return parsed._replace(query="", fragment="").geturl()


def _diagnose_alert_ping_capability(channel, settings: Settings) -> bool:
    """Log once whether the configured traffic-alert role can be mentioned.

    This uses the resolved channel's cached guild/member/role objects and makes
    no additional Discord request.  Failure is diagnostic only: dashboard and
    no-ping thread updates continue normally.
    """
    role_id = settings.alert_role_id
    if role_id is None:
        log.info("traffic role pings are disabled because ALERT_ROLE_ID is not configured")
        return False
    guild = getattr(channel, "guild", None)
    member = getattr(guild, "me", None)
    get_role = getattr(guild, "get_role", None)
    role = get_role(role_id) if callable(get_role) else None
    if guild is None or member is None or role is None:
        log.warning(
            "traffic role pings are configured but unavailable: the bot could not "
            "resolve its guild member or configured role"
        )
        return False
    try:
        permissions = channel.permissions_for(member)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not verify traffic role ping permission: %s", exc)
        return False
    if getattr(role, "mentionable", False) or getattr(
        permissions, "mention_everyone", False
    ):
        log.info("traffic role ping permission verified")
        return True
    log.warning(
        "traffic role pings will not notify: the configured role is not mentionable "
        "and the bot lacks 'Mention @everyone, @here, and All Roles' in the dashboard "
        "channel; a server administrator must make the role mentionable or grant that "
        "channel permission to the bot"
    )
    return False


async def run_discord_bot(settings: Settings) -> None:
    from discord.ext import commands

    # Default intents only: the marker scan matches the bot's OWN message
    # content, which is always visible without the privileged Message Content
    # intent. This avoids requiring the portal toggle.
    intents = discord.Intents.default()
    bot = commands.Bot(
        command_prefix="d.",
        intents=intents,
        help_command=None,
        http_trace=_dashboard_http_trace_config(),
        max_ratelimit_timeout=DISCORD_MAX_RATELIMIT_RETRY_SECONDS,
    )
    # discord.py clamps the constructor option to 30 seconds.  Set the desired
    # bound now for login; ``on_ready`` wraps the Event that ``static_login``
    # creates afterward.
    bot.http.max_ratelimit_timeout = DISCORD_MAX_RATELIMIT_RETRY_SECONDS

    updater = DashboardUpdater(settings)
    # Persistent component: survives restarts via the custom_id registration.
    bot.add_view(updater.live_view)

    @bot.event
    async def on_ready() -> None:
        # ``static_login`` replaces the global-rate-limit Event, so install the
        # cancellation-safe wrapper only after login and before dashboard I/O.
        _configure_discord_http_deadlines(bot.http)
        log.info(
            "Discord ready bot_id=%s channel_id=%s latency_s=%.3f updater_running=%s",
            getattr(bot.user, "id", None), settings.announce_channel_id, bot.latency,
            updater.is_running,
        )
        if updater.is_running:
            log.info("dashboard update loop already active after reconnect")
            return
        try:
            channel = bot.get_channel(settings.announce_channel_id)
            if channel is None:
                channel = await bot.fetch_channel(settings.announce_channel_id)
        except Exception as exc:  # noqa: BLE001
            log.error(
                "cannot resolve announce channel %s: %s",
                settings.announce_channel_id,
                exc,
            )
            await bot.close()
            return
        _diagnose_dashboard_permissions(channel)
        _diagnose_alert_ping_capability(channel, settings)
        # Resolve the message once (configured ID or history scan).
        message = await _resolve_dashboard_message(
            channel,
            updater._persisted_dashboard_message_id  # noqa: SLF001
            or settings.dashboard_message_id,
            bot.user,
        )
        updater._set_dashboard_message(message)  # noqa: SLF001
        await updater.start(channel)

    @bot.event
    async def on_disconnect() -> None:
        log.info("disconnected; updater continues on reconnect")

    @bot.event
    async def on_command_error(ctx, error) -> None:
        if isinstance(error, commands.CommandNotFound):
            return
        raise error

    try:
        await bot.start(settings.discord_token)
    finally:
        await updater.stop()


async def run_dev_webhook(settings: Settings) -> None:
    """One-shot: build the payload and send it to DEV_WEBHOOK."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        try:
            client = HttpClient(session, timeout_seconds=settings.http_timeout_seconds)
            results = await collect_all(client, settings)
            payload = _to_payload(results, rthk_news_max_age_hours=settings.rthk_news_max_age_hours)
            webhook = discord.Webhook.from_url(settings.dev_webhook, session=session)
            files = [discord_file(a) for a in payload.files]
            await webhook.send(
                content="",
                embeds=[e for e in payload.embeds if e is not None],
                files=files or None,
            )
            log.info(
                "dev webhook sent %d embeds, %d files",
                len(payload.embeds),
                len(payload.files),
            )
        finally:
            await pipeline.shutdown_background_resources()


async def run_dry_run(settings: Settings) -> None:
    """Build the payload once and write a preview under .private/ (no Discord)."""
    import aiohttp

    os.makedirs(".private", exist_ok=True)
    async with aiohttp.ClientSession() as session:
        try:
            client = HttpClient(session, timeout_seconds=settings.http_timeout_seconds)
            results = await collect_all(client, settings)
            payload = _to_payload(results, rthk_news_max_age_hours=settings.rthk_news_max_age_hours)

            lines: list[str] = []
            for i, embed in enumerate(payload.embeds):
                if embed is None:
                    continue
                title = embed.title or "(no title)"
                lines.append(f"=== Embed {i + 1}: {title} ===")
                if embed.description:
                    lines.append(f"description: {embed.description}")
                for field in embed.fields:
                    lines.append(f"[{field.name}]")
                    lines.append(field.value)
                if embed.image and embed.image.url:
                    lines.append(f"image: {embed.image.url}")
            lines.append("")
            lines.append(f"files: {[a.filename for a in payload.files]}")

            preview_path = os.path.join(".private", "dashboard-preview.txt")
            with open(preview_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            for asset in payload.files:
                if _is_traffic_map_filename(asset.filename):
                    with open(os.path.join(".private", "traffic-map-preview.webp"), "wb") as f:
                        f.write(asset.data)
            log.info("dry-run preview written to %s", preview_path)
        finally:
            await pipeline.shutdown_background_resources()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="hkust-dashboard",
        description="HKUST campus data dashboard for Discord.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build payload and write preview under .private/ (no Discord)",
    )
    parser.add_argument(
        "--dev-webhook",
        action="store_true",
        help="send one-shot payload to DEV_WEBHOOK",
    )
    parser.add_argument(
        "--no-keys",
        action="store_true",
        help="allow running without DISCORD_TOKEN/ANNOUNCE_CHANNEL_ID (for dry-run)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    load_dotenv()
    try:
        settings = Settings.from_env(require_keys=not args.no_keys and not args.dry_run)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    _setup_logging(
        settings.log_level, settings.cache_dir,
        secret_values=tuple(value for value in (
            settings.discord_token, settings.dev_webhook,
            os.environ.get("BUS_QUEUE_KEY"), os.environ.get("PPL_COUNT_KEY"),
        ) if value),
    )
    mode = "dev_webhook" if args.dev_webhook else "dry_run" if args.dry_run else "discord"
    log.info(
        "dashboard startup mode=%s source_build=%s python=%s dependencies=%s "
        "channel_id=%s configured_message_id=%s interval_s=%.1f http_timeout_s=%.1f "
        "rthk_window_h=%.1f cache_dir=%s",
        mode, source_fingerprint(Path(__file__).resolve().parent), sys.version.split()[0],
        dependency_versions(), settings.announce_channel_id, settings.dashboard_message_id,
        settings.update_interval_seconds, settings.http_timeout_seconds,
        settings.rthk_news_max_age_hours, settings.cache_dir,
    )

    try:
        ffmpeg_executable = startup_preflight()
    except ConfigError as exc:
        ffmpeg_executable = None
        log.warning("Camera warning: %s Cameras are disabled; the dashboard will continue.", exc)
    settings = replace(settings, ffmpeg_executable=ffmpeg_executable)

    try:
        if args.dev_webhook:
            if not settings.dev_webhook:
                print("DEV_WEBHOOK is not set", file=sys.stderr)
                return 2
            asyncio.run(run_dev_webhook(settings))
        elif args.dry_run:
            asyncio.run(run_dry_run(settings))
        else:
            asyncio.run(run_discord_bot(settings))
    except KeyboardInterrupt:
        log.info("dashboard stopped reason=keyboard_interrupt")
    except Exception as exc:
        log.exception(
            "dashboard stopped reason=unhandled_error mode=%s error=%s: %s",
            mode, type(exc).__name__, exc,
        )
        return 1
    else:
        log.info("dashboard stopped reason=normal mode=%s", mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
