"""HKUST Campus Dashboard bot entry point.

Executable lifecycle: shared session + providers, concurrent fetches, one
persistent dashboard message edited in place, dev-webhook and dry-run modes.

Imports must have no filesystem/network/package-installation/bot-launch side
effects; all side effects live under ``main()``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import hashlib
import inspect
import io
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

import discord
from dotenv import load_dotenv

from dashboard import maps
from dashboard.config import ConfigError, Settings
from dashboard.http import HttpClient
from dashboard.models import (
    CameraFrame,
    DashboardPayload,
    ImageAsset,
    WeatherConditions,
)
from dashboard.providers import cameras, transit
from dashboard.providers import route_geometry as route_geometry_provider
from dashboard.providers import tracked_roads as tracked_roads_provider
from dashboard.providers import traffic as traffic_provider
from dashboard.providers import weather as weather_provider
from dashboard.render import (
    build_payload,
)
from dashboard.runtime import startup_preflight

log = logging.getLogger(__name__)
DASHBOARD_MESSAGE_MARKER = "HKUST Campus Dashboard"

def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# --------------------------------------------------------------------------
# Data collection
# --------------------------------------------------------------------------

async def collect_all(
    client: HttpClient,
    settings: Settings,
    on_result: Callable[[str, object], None] | None = None,
) -> dict[str, object]:
    """Fetch all provider groups concurrently; return raw results keyed by name.

    Each entry is either the provider's result object or an exception (callers
    isolate failures).
    """
    from dashboard.providers import tracked_roads as tracked_roads_provider

    tasks: dict[str, asyncio.Task] = {}

    async def _tracked_roads():
        return await tracked_roads_provider.fetch_tracked_roads(
            client, cache_dir=settings.cache_dir
        )

    async def _transit():
        return await transit.fetch_transit_etas(client)

    async def _weather():
        return await weather_provider.fetch_weather_conditions(client)

    async def _traffic():
        # Road matching needs the tracked-roads table. A cold derivation can
        # take a while, so give it a short grace period and fall back to the
        # curated seed rather than delaying TD news.
        try:
            roads = await asyncio.wait_for(
                asyncio.shield(tasks["tracked_roads"]), timeout=5
            )
        except Exception:  # noqa: BLE001
            roads = tracked_roads_provider.fallback_roads()
        return await traffic_provider.fetch_traffic_data(client, roads)

    async def _affected_road_paths() -> list[list[tuple[float, float]]]:
        """Return anchored OSM segments for current TD traffic notices."""
        try:
            traffic_result = await tasks["traffic"]
        except Exception:  # noqa: BLE001
            return []
        if not (isinstance(traffic_result, tuple) and len(traffic_result) >= 3):
            return []
        try:
            roads = await asyncio.wait_for(
                asyncio.shield(tasks["tracked_roads"]), timeout=5
            )
        except Exception:  # noqa: BLE001
            roads = tracked_roads_provider.fallback_roads()
        segments_near = getattr(roads, "segments_near", None)
        if segments_near is None:
            return []
        paths: list[list[tuple[float, float]]] = []
        seen_paths: set[tuple[tuple[float, float], ...]] = set()
        for incident in traffic_result[1] or []:
            latitude = getattr(incident, "latitude", None)
            longitude = getattr(incident, "longitude", None)
            text = f"{incident.title} {incident.description} {incident.location} {incident.road}"
            keys = roads.match(text)
            has_coordinates = (
                isinstance(latitude, (int, float))
                and isinstance(longitude, (int, float))
                and 22.0 <= latitude <= 23.0
                and 113.5 <= longitude <= 114.7
            )
            if not has_coordinates:
                # A malformed, partial, or out-of-range coordinate is an
                # explicit source signal, not permission to guess a whole
                # road. Only a completely coordinate-less notice may use the
                # provider's conservative short-road fallback.
                if latitude is not None or longitude is not None:
                    continue
                near_landmark = str(getattr(incident, "near_landmark", "") or "").strip()
                between_landmark = str(
                    getattr(incident, "between_landmark", "") or ""
                ).strip()
                if near_landmark or between_landmark:
                    # A landmark-only notice may still name a specific short
                    # sub-road (for example, "Lung Cheung Road flyover").
                    # Permit the conservative whole-way fallback only for
                    # that explicit refinement, never for the generic road.
                    def _words(value: object) -> str:
                        return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()

                    road_words = _words(getattr(incident, "road", ""))
                    text_words = _words(text)
                    keys = [
                        key
                        for key in keys
                        if (key_words := _words(key))
                        and road_words
                        and key_words.startswith(f"{road_words} ")
                        and key_words in text_words
                    ]
                    if not keys:
                        continue
                latitude = longitude = None
            for path in segments_near(keys, latitude, longitude) or ():
                normalized = tuple((float(lat), float(lon)) for lat, lon in path)
                if len(normalized) >= 2 and normalized not in seen_paths:
                    seen_paths.add(normalized)
                    paths.append(list(normalized))
        return paths

    async def _traffic_map():
        # Transit ETA groups still drive retained estimated bus markers.
        groups: list = []
        try:
            transit_result = await tasks["transit"]
            if isinstance(transit_result, tuple) and len(transit_result) == 3:
                groups = transit_result[0]
        except Exception as exc:  # noqa: BLE001
            log.warning("traffic map: transit groups unavailable: %s", exc)
        affected_paths = await _affected_road_paths()
        return await maps.fetch_traffic_map(
            client,
            groups=groups,
            cache_dir=settings.cache_dir,
            affected_road_paths=affected_paths,
        )

    for name, coro in (
        ("tracked_roads", _tracked_roads()),
        ("transit", _transit()),
        ("weather", _weather()),
        ("traffic", _traffic()),
    ):
        tasks[name] = asyncio.create_task(coro)

    # The map uses transit ETA groups to estimate bus positions.
    tasks["traffic_map"] = asyncio.create_task(_traffic_map())

    results: dict[str, object] = {}
    task_names = {task: name for name, task in tasks.items()}
    pending = set(tasks.values())
    try:
        # Publish every provider as soon as it settles.  In particular, a slow
        # browser capture no longer prevents weather/traffic/transit from reaching
        # the snapshot used by the fixed-cadence presenter.
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                name = task_names[task]
                try:
                    value = task.result()
                except Exception as exc:  # noqa: BLE001
                    log.warning("provider %s failed: %s", name, exc)
                    value = exc
                results[name] = value
                if on_result is not None:
                    on_result(name, value)
        return results
    finally:
        # A cancelled collection must not leave provider tasks (especially the
        # Playwright map capture) running after the caller has shut down.
        remaining = [task for task in tasks.values() if not task.done()]
        for task in remaining:
            task.cancel()
        if remaining:
            await asyncio.gather(*remaining, return_exceptions=True)


def _to_payload(results: dict[str, object]) -> DashboardPayload:
    """Convert collected results into a renderable payload."""
    errors: list[str] = []

    transit_result = results.get("transit")
    if isinstance(transit_result, Exception):
        errors.append("transit ETA unavailable")
        groups: list = []
        transit_source_time = None
    elif isinstance(transit_result, tuple) and len(transit_result) == 3:
        groups, transit_source_time, failed_ops = transit_result
        for op in failed_ops:
            errors.append(f"{op} ETA unavailable")
    else:
        groups = []
        transit_source_time = None

    weather_result = results.get("weather")
    weather: WeatherConditions | None = None
    if isinstance(weather_result, Exception):
        errors.append("HKO weather unavailable")
    elif isinstance(weather_result, tuple) and len(weather_result) == 3:
        snap, warnings, warn_time = weather_result
        weather = WeatherConditions(
            warnings=warnings, snapshot=snap, warning_time=warn_time
        )
    elif isinstance(weather_result, WeatherConditions):
        weather = weather_result

    traffic_result = results.get("traffic")
    traffic_source_times: dict = {}
    if isinstance(traffic_result, Exception):
        errors.append("TD traffic unavailable")
        statuses, incidents, roadworks, capture_time, traffic_stale = [], [], [], None, []
    elif isinstance(traffic_result, tuple) and len(traffic_result) >= 6:
        (
            statuses,
            incidents,
            roadworks,
            capture_time,
            traffic_stale,
            traffic_source_times,
        ) = traffic_result[:6]
    elif isinstance(traffic_result, tuple) and len(traffic_result) >= 5:
        statuses, incidents, roadworks, capture_time, traffic_stale = traffic_result
    elif isinstance(traffic_result, tuple) and len(traffic_result) == 4:
        statuses, incidents, roadworks, capture_time = traffic_result
        traffic_stale = []
    else:
        statuses, incidents, roadworks, capture_time, traffic_stale = [], [], [], None, []

    # The map provider returns the Google base image and retained markers.
    smap_result = results.get("traffic_map")
    if isinstance(smap_result, Exception):
        map_webp: bytes | None = None
    elif isinstance(smap_result, tuple) and len(smap_result) >= 2:
        map_webp = smap_result[0]
    else:
        map_webp = None
    if map_webp is None:
        errors.append("traffic map unavailable")
    map_source_time = None

    # Bus-stop live view moved behind the dashboard button; the dashboard
    # message itself no longer carries always-on camera embeds.
    # Tracked-roads table (OSM-derived) drives affected-route listings.
    roads_table = results.get("tracked_roads")
    if isinstance(roads_table, Exception):
        roads_table = None

    return build_payload(
        weather=weather,
        groups=groups,
        statuses=statuses,
        incidents=incidents,
        capture_time=capture_time,
        traffic_map_webp=map_webp,
        transit_source_time=transit_source_time,
        map_source_time=map_source_time,
        roadworks=roadworks,
        traffic_stale_sources=traffic_stale,
        traffic_source_times=traffic_source_times,
        traffic_source_time=capture_time,
        errors=errors,
        roads=roads_table,
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

    @discord.ui.button(label="Bus stops live", style=discord.ButtonStyle.primary,
                       custom_id=LIVE_VIEW_BUTTON_ID, emoji="📷")
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
        if cache is None or not cache.fresh or updater.client is None or (
            not updater.settings.ffmpeg_executable
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
            embeds=_snapshot_embeds_with_countdown(
                parts.embeds, ends, len(parts.assets)
            ),
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
                    embeds=_snapshot_embeds_with_countdown(
                        parts.embeds, ends, len(parts.assets)
                    ),
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
                        embeds=_snapshot_embeds_with_countdown(
                        parts.embeds, ends, len(parts.assets)
                    ),
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


async def _find_dashboard_message(channel, expected_author=None) -> object | None:
    """Scan recent history for this bot's latest dashboard message.

    Author ID plus the stable marker avoid taking over another message from
    this bot (for example an alert or command response) in the same channel.
    """
    expected_author = expected_author or getattr(
        getattr(channel, "guild", None), "me", None
    )
    try:
        async for message in channel.history(limit=50):
            if _is_dashboard_message(message, expected_author):
                return message
    except Exception as exc:  # noqa: BLE001
        log.warning("history scan failed: %s", exc)
    return None


async def _resolve_dashboard_message(
    channel,
    configured_message_id: int | None,
    expected_author,
) -> object | None:
    """Validate a configured message, otherwise fall back to history scan."""
    if configured_message_id:
        try:
            configured = await channel.fetch_message(configured_message_id)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "configured message %s not found, scanning: %s",
                configured_message_id,
                exc,
            )
        else:
            if _is_dashboard_message(configured, expected_author):
                return configured
            log.warning(
                "configured message %s is not this bot's exact dashboard marker; scanning",
                configured_message_id,
            )
    return await _find_dashboard_message(channel, expected_author)


async def _ensure_dashboard_message(channel, payload: DashboardPayload, view=None) -> object:
    """Reuse the known message or find/create exactly one."""
    message = await _find_dashboard_message(channel)
    if message is not None:
        return message
    # create exactly one
    message = await channel.send(content=DASHBOARD_MESSAGE_MARKER, view=view)
    log.info("created dashboard message %s in %s", message.id, getattr(channel, "id", "?"))
    return message


def discord_file(asset: ImageAsset) -> discord.File:
    return discord.File(io.BytesIO(asset.data), filename=asset.filename)


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
        {"filename": asset.filename, "content_type": asset.content_type,
         "data_sha256": hashlib.sha256(asset.data).hexdigest()}
        for asset in payload.files
    ]
    serialized = json.dumps(
        {"embeds": embeds, "files": files},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


async def _apply_payload(message, payload: DashboardPayload, view=None) -> None:
    """Edit embeds + attachments atomically; replace images to avoid CDN cache."""
    embeds = [e for e in payload.embeds if e is not None]
    files = [discord_file(asset) for asset in payload.files]
    await message.edit(
        content=DASHBOARD_MESSAGE_MARKER,
        embeds=embeds,
        attachments=files,
        view=view,
    )


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
        self._collection_generation = 0
        self._last_alert_generation = 0
        self._message = None
        self._thread = None
        self._loop_task: asyncio.Task | None = None
        self._running = False
        self._start_lock = asyncio.Lock()
        self.live_snapshot_task: asyncio.Task | None = None
        self.last_live_snapshot = float("-inf")
        self.live_snapshot_message_id: int | None = None
        self.live_snapshot_message = None
        self.live_snapshot_generation = 0
        self.live_view = LiveViewSnapshotView(self)
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
        return bool(
            self._running
            and self._loop_task is not None
            and not self._loop_task.done()
        )

    async def _ensure_thread(self) -> None:
        """Create the updates thread under the dashboard message once."""
        if self._thread is not None or self._message is None:
            return
        try:
            # reuse an existing thread by name if present
            for t in self._message.channel.threads:
                if t.name == "status updates":
                    self._thread = t
                    return
            self._thread = await self._message.create_thread(
                name="status updates", auto_archive_duration=10080
            )
            log.info("created status thread %s", self._thread.id)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not create status thread: %s", exc)

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
        roads = results.get("tracked_roads")
        if roads is not None and not isinstance(roads, Exception):
            self.alerts.roads = roads
        events = self.alerts.update(warnings, statuses, incidents, roadworks)
        if not events or self._thread is None:
            return
        for event in events:
            text = self.alerts.ping_for(event, self.settings.alert_role_id)
            try:
                await self._thread.send(content=text)
            except Exception as exc:  # noqa: BLE001
                log.warning("alert post failed: %s", exc)

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
                log.warning("update tick failed: %s", exc)
            next_tick += self.settings.update_interval_seconds
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
        if "on_result" in inspect.signature(collect_all).parameters:
            collection = collect_all(self.client, self.settings, on_result=publish)
        else:
            collection = collect_all(self.client, self.settings)
        task = asyncio.create_task(collection)
        self._collection_task = task
        task.add_done_callback(self._collection_finished)

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
            log.warning("background collection failed: %s", exc)
            return
        if not isinstance(fresh, dict):
            log.warning("background collection returned invalid result")
            return

        # ``collect_all`` publishes providers individually.  This fallback is
        # retained for alternate collectors that do not invoke the callback.
        if self._snapshot is None or self._snapshot.generation < self._collection_generation:
            for name, value in fresh.items():
                self._publish_provider_result(self._collection_generation, name, value)

    def _publish_provider_result(self, generation: int, name: str, value: object) -> None:
        """Atomically merge one provider into the current last-good snapshot."""
        if not self._running or generation != self._collection_generation:
            return
        previous = self._snapshot
        merged: dict[str, object] = {}
        stale: set[str] = set()
        if previous is not None:
            merged.update(previous.results)
            stale.update(previous.stale_providers)
            # At the beginning of a new collection, all retained values are
            # stale until their owning provider has supplied this generation.
            if previous.generation != generation:
                stale.update(merged)
        if isinstance(value, Exception) and name in merged and not isinstance(merged[name], Exception):
            stale.add(name)
        else:
            merged[name] = value
            if isinstance(value, Exception):
                stale.add(name)
            else:
                stale.discard(name)
        self._snapshot = CollectionSnapshot(
            results=merged,
            generation=generation,
            completed_monotonic=time.monotonic(),
            stale_providers=frozenset(stale),
        )

    def _snapshot_payload(self) -> DashboardPayload | None:
        """Build from the latest completed snapshot, never from a live fetch."""
        if self._snapshot is None:
            return None
        return _to_payload(self._snapshot.results)

    async def _tick(self, channel=None) -> None:
        # The presenter is deliberately independent of provider latency.  It
        # starts the next refresh when idle then reads only a completed snapshot.
        self._start_collection_if_idle()
        # Let an already-ready task publish its callback, without awaiting a
        # slow provider.  This also makes fast local/dry-run providers visible
        # on the first presentation.
        await asyncio.sleep(0)
        if self._collection_task is not None and self._collection_task.done():
            self._collection_finished(self._collection_task)
        payload = self._snapshot_payload()
        if payload is None:
            return

        if channel is None:
            # dry-run / dev: just keep the last payload for inspection
            self._last_good_payload = payload
            return

        # ensure message exists (create once)
        if self._message is None:
            self._message = await _ensure_dashboard_message(
                channel, payload, view=self.live_view
            )
        fingerprint = _payload_fingerprint(payload)
        try:
            if fingerprint != self._last_payload_fingerprint:
                await _apply_payload(self._message, payload, view=self.live_view)
                self._last_payload_fingerprint = fingerprint
            self._last_good_payload = payload
        except Exception as exc:  # noqa: BLE001
            log.warning("edit failed (keeping last good): %s", exc)

        # status thread + alerts (create the thread after the message exists)
        await self._ensure_thread()
        if self._snapshot is not None and self._snapshot.generation != self._last_alert_generation:
            await self._post_alert_events(self._snapshot.results)
            self._last_alert_generation = self._snapshot.generation

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
        collection_task = self._collection_task
        if collection_task is not None and not collection_task.done():
            collection_task.cancel()
        if collection_task is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await collection_task
        self._collection_task = None
        await route_geometry_provider.shutdown_background_refreshes()
        await tracked_roads_provider.shutdown_background_refreshes()
        if self.session is not None:
            await self.session.close()
            self.session = None
        self.client = None


# --------------------------------------------------------------------------
# Discord bot wiring
# --------------------------------------------------------------------------

async def run_discord_bot(settings: Settings) -> None:
    from discord.ext import commands

    # Default intents only: the marker scan matches the bot's OWN message
    # content, which is always visible without the privileged Message Content
    # intent. This avoids requiring the portal toggle.
    intents = discord.Intents.default()
    bot = commands.Bot(command_prefix="d.", intents=intents, help_command=None)

    updater = DashboardUpdater(settings)
    # Persistent component: survives restarts via the custom_id registration.
    bot.add_view(updater.live_view)

    @bot.event
    async def on_ready() -> None:
        log.info("Logged in as %s", bot.user)
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
        # Resolve the message once (configured ID or history scan).
        message = await _resolve_dashboard_message(
            channel,
            settings.dashboard_message_id,
            bot.user,
        )
        updater._message = message  # noqa: SLF001
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
        client = HttpClient(session, timeout_seconds=settings.http_timeout_seconds)
        results = await collect_all(client, settings)
        payload = _to_payload(results)
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


async def run_dry_run(settings: Settings) -> None:
    """Build the payload once and write a preview under .private/ (no Discord)."""
    import aiohttp

    os.makedirs(".private", exist_ok=True)
    async with aiohttp.ClientSession() as session:
        client = HttpClient(session, timeout_seconds=settings.http_timeout_seconds)
        results = await collect_all(client, settings)
        payload = _to_payload(results)

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
            if asset.filename == "traffic-map.webp":
                with open(os.path.join(".private", "traffic-map-preview.webp"), "wb") as f:
                    f.write(asset.data)
        log.info("dry-run preview written to %s", preview_path)


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

    _setup_logging(settings.log_level)

    try:
        ffmpeg_executable = startup_preflight()
    except ConfigError as exc:
        ffmpeg_executable = None
        print(
            f"Camera warning: {exc} Cameras are disabled; the dashboard will continue.",
            file=sys.stderr,
        )
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
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
