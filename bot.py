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
import io
import logging
import os
import sys

import discord
from dotenv import load_dotenv

from dashboard.config import ConfigError, Settings
from dashboard.http import HttpClient
from dashboard.models import (
    DashboardPayload,
    ImageAsset,
    WeatherConditions,
)
from dashboard.providers import traffic as traffic_provider
from dashboard.providers import transit
from dashboard.providers import weather as weather_provider
from dashboard.render import (
    DASHBOARD_FOOTER,
    DASHBOARD_MARKER,
    build_payload,
)
from dashboard.traffic_map import render_traffic_map

log = logging.getLogger(__name__)

# Marker text embedded in the dashboard message so the bot can find its own
# message when scanning history (never edits arbitrary user messages).
MESSAGE_MARKER = f"{DASHBOARD_MARKER} · {DASHBOARD_FOOTER}"


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
) -> dict[str, object]:
    """Fetch all provider groups concurrently; return raw results keyed by name.

    Each entry is either the provider's result object or an exception (callers
    isolate failures).
    """
    tasks: dict[str, asyncio.Task] = {}

    async def _transit():
        return await transit.fetch_transit_etas(client)

    async def _weather():
        return await weather_provider.fetch_weather_conditions(client)

    async def _traffic():
        return await traffic_provider.fetch_traffic_data(client)

    async def _cctv():
        return await traffic_provider.fetch_cctv_images(client)

    for name, coro in (
        ("transit", _transit()),
        ("weather", _weather()),
        ("traffic", _traffic()),
        ("cctv", _cctv()),
    ):
        tasks[name] = asyncio.create_task(coro)

    results: dict[str, object] = {}
    for name, task in tasks.items():
        try:
            results[name] = await task
        except Exception as exc:  # noqa: BLE001
            log.warning("provider %s failed: %s", name, exc)
            results[name] = exc
    return results


def _to_payload(
    results: dict[str, object],
    traffic_map_png: bytes | None,
) -> DashboardPayload:
    """Convert collected results into a renderable payload."""
    errors: list[str] = []

    transit_result = results.get("transit")
    if isinstance(transit_result, Exception):
        errors.append("transit ETA unavailable")
        groups: list = []
    elif isinstance(transit_result, tuple) and len(transit_result) == 3:
        groups, _, failed_ops = transit_result
        for op in failed_ops:
            errors.append(f"{op} ETA unavailable")
    else:
        groups = []

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
    if isinstance(traffic_result, Exception):
        errors.append("TD traffic unavailable")
        statuses, incidents, capture_time = [], [], None
    elif isinstance(traffic_result, tuple):
        statuses, incidents, _, capture_time = traffic_result
    else:
        statuses, incidents, capture_time = [], [], None

    cctv_result = results.get("cctv")
    if isinstance(cctv_result, Exception):
        errors.append("TD CCTV unavailable")
        cctv_images: list[ImageAsset] = []
    elif isinstance(cctv_result, list):
        cctv_images = cctv_result
    else:
        cctv_images = []

    return build_payload(
        weather=weather,
        groups=groups,
        statuses=statuses,
        incidents=incidents,
        capture_time=capture_time,
        traffic_map_png=traffic_map_png,
        cctv_images=cctv_images,
        errors=errors,
    )


# --------------------------------------------------------------------------
# Message lifecycle
# --------------------------------------------------------------------------

async def _find_dashboard_message(channel) -> object | None:
    """Scan recent history for a bot-authored message with our marker."""
    try:
        async for message in channel.history(limit=50):
            if message.author.bot and MESSAGE_MARKER in (message.content or ""):
                return message
    except Exception as exc:  # noqa: BLE001
        log.warning("history scan failed: %s", exc)
    return None


async def _ensure_dashboard_message(channel, payload: DashboardPayload) -> object:
    """Reuse the known message or find/create exactly one."""
    message = await _find_dashboard_message(channel)
    if message is not None:
        return message
    # create exactly one
    message = await channel.send(content=MESSAGE_MARKER)
    log.info("created dashboard message %s in %s", message.id, getattr(channel, "id", "?"))
    return message


def discord_file(asset: ImageAsset) -> discord.File:
    return discord.File(io.BytesIO(asset.data), filename=asset.filename)


async def _apply_payload(message, payload: DashboardPayload) -> None:
    """Edit embeds + attachments atomically; replace images to avoid CDN cache."""
    embeds = [e for e in payload.embeds if e is not None]
    files = [discord_file(asset) for asset in payload.files]
    await message.edit(content=MESSAGE_MARKER, embeds=embeds, attachments=files)


# --------------------------------------------------------------------------
# Updater loop
# --------------------------------------------------------------------------

class DashboardUpdater:
    """Owns the session, providers, caches, and the single update loop."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = None
        self.client: HttpClient | None = None
        self._last_good_payload: DashboardPayload | None = None
        self._message = None
        self._thread = None
        self._loop_task: asyncio.Task | None = None
        self._running = False
        from dashboard.alerts import AlertMonitor

        self.alerts = AlertMonitor()

    async def start(self, channel=None) -> None:
        import aiohttp

        self.session = aiohttp.ClientSession()
        self.client = HttpClient(
            self.session, timeout_seconds=self.settings.http_timeout_seconds
        )
        self._running = True
        self._loop_task = asyncio.create_task(self._update_loop(channel))

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
        if isinstance(traffic_result, tuple):
            statuses = traffic_result[0]
        events = self.alerts.update(warnings, statuses)
        if not events or self._thread is None:
            return
        for event in events:
            text = self.alerts.ping_for(event, self.settings.alert_role_id)
            try:
                await self._thread.send(content=text)
            except Exception as exc:  # noqa: BLE001
                log.warning("alert post failed: %s", exc)

    async def _update_loop(self, channel=None) -> None:
        while self._running:
            try:
                await self._tick(channel)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("update tick failed: %s", exc)
            await asyncio.sleep(self.settings.update_interval_seconds)

    async def _tick(self, channel=None) -> None:
        assert self.client is not None
        results = await collect_all(self.client, self.settings)

        # Traffic map: rebuild only when traffic data changed.
        traffic_result = results.get("traffic")
        map_png: bytes | None = None
        if isinstance(traffic_result, tuple):
            statuses, incidents, roadworks, capture = traffic_result
            map_png, err = render_traffic_map(
                statuses,
                incidents,
                roadworks,
                capture,
                cache_dir=self.settings.cache_dir,
            )
            if err:
                log.warning("traffic map render failed: %s", err)

        payload = _to_payload(results, map_png)

        if channel is None:
            # dry-run / dev: just keep the last payload for inspection
            self._last_good_payload = payload
            return

        # ensure message exists (create once)
        if self._message is None:
            self._message = await _ensure_dashboard_message(channel, payload)
        try:
            await _apply_payload(self._message, payload)
            self._last_good_payload = payload
        except Exception as exc:  # noqa: BLE001
            log.warning("edit failed (keeping last good): %s", exc)

        # status thread + alerts (create the thread after the message exists)
        await self._ensure_thread()
        await self._post_alert_events(results)

    async def stop(self) -> None:
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._loop_task
        if self.session is not None:
            await self.session.close()


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

    @bot.event
    async def on_ready() -> None:
        log.info("Logged in as %s", bot.user)
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
        message = None
        if settings.dashboard_message_id:
            try:
                message = await channel.fetch_message(settings.dashboard_message_id)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "configured message %s not found, scanning: %s",
                    settings.dashboard_message_id,
                    exc,
                )
        if message is None:
            message = await _find_dashboard_message(channel)
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
        traffic_result = results.get("traffic")
        map_png: bytes | None = None
        if isinstance(traffic_result, tuple):
            statuses, incidents, roadworks, capture = traffic_result
            map_png, err = render_traffic_map(
                statuses, incidents, roadworks, capture, cache_dir=settings.cache_dir
            )
            if err:
                log.warning("traffic map render failed: %s", err)
        payload = _to_payload(results, map_png)
        webhook = discord.Webhook.from_url(settings.dev_webhook, session=session)
        files = [discord_file(a) for a in payload.files]
        await webhook.send(
            content=MESSAGE_MARKER,
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
        traffic_result = results.get("traffic")
        map_png: bytes | None = None
        if isinstance(traffic_result, tuple):
            statuses, incidents, roadworks, capture = traffic_result
            map_png, err = render_traffic_map(
                statuses, incidents, roadworks, capture, cache_dir=settings.cache_dir
            )
            if err:
                log.warning("traffic map render failed: %s", err)
        payload = _to_payload(results, map_png)

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
        lines.append(f"marker: {MESSAGE_MARKER}")

        preview_path = os.path.join(".private", "dashboard-preview.txt")
        with open(preview_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        for asset in payload.files:
            if asset.filename == "traffic-map.png":
                with open(os.path.join(".private", "traffic-map-preview.png"), "wb") as f:
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
