"""Limit-aware multi-embed renderer.

Builds a ``DashboardPayload`` (ordered embeds + attachments) from provider
results, enforcing Discord limits: field value 1024, description 4096, 25
fields/embed, 6000 aggregate chars/embed, 10 embeds/files per message.
"""

from __future__ import annotations

import hashlib
import io
from datetime import UTC, datetime

import discord
from PIL import Image, UnidentifiedImageError

from dashboard.models import (
    DashboardPayload,
    EtaKind,
    ImageAsset,
    Operator,
    Roadwork,
    RouteEtaGroup,
    TrafficCorridorStatus,
    TrafficIncident,
    WeatherConditions,
    WeatherWarning,
)

# Discord limits (from the discord.py docs).
FIELD_VALUE_MAX = 1024
DESC_MAX = 4096
FIELDS_PER_EMBED_MAX = 25
CHARS_PER_EMBED_MAX = 6000
EMBEDS_PER_MESSAGE_MAX = 10
TD_TRAFFIC_NEWS_URL = "https://www.td.gov.hk/en/special_news/spnews.htm"

_OPERATOR_ICON = {
    Operator.KMB: "🟥",
    Operator.CITYBUS: "🟨",
    Operator.GMB: "🟩",
}

# Characters that have markdown meaning in Discord; escape them in data text
# (route names, destinations, warnings, incident descriptions) so the
# dashboard never renders unexpected formatting from upstream strings.
_MD_ESCAPE = str.maketrans(
    {"\\": "\\\\", "*": "\\*", "_": "\\_", "`": "\\`", "~": "\\~", "|": "\\|", ">": "\\>"}
)


def _esc(text: str) -> str:
    """Escape markdown metacharacters in untrusted display text."""
    return text.translate(_MD_ESCAPE)


def _fmt_timestamp(dt: datetime | None, fmt: str = "t") -> str:
    """Discord timestamp that renders in each viewer's local timezone.

    `style` is a Discord timestamp letter: t = HH:MM, f = full date + time,
    R = relative ("2 hours ago"). Returns "—" when the datetime is missing.
    """
    if dt is None:
        return "—"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return f"<t:{int(dt.timestamp())}:{fmt}>"


def _set_source_timestamp(
    embed: discord.Embed,
    label: str,
    source_time: datetime | None,
) -> discord.Embed:
    """Apply the same native timestamp/footer treatment to every data pane."""
    embed.set_footer(text=label)
    if source_time is not None:
        if source_time.tzinfo is None:
            source_time = source_time.replace(tzinfo=UTC)
        embed.timestamp = source_time
    return embed


def _discord_rel(ts: datetime | None) -> str:
    """Relative Discord timestamp (renders as "2 hours ago" per viewer)."""
    if ts is None:
        return "—"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return f"<t:{int(ts.timestamp())}:R>"


def _eta_cell(row_minutes: int | None, kind: EtaKind) -> str:
    """ETA cell for use inside a code block: number (2 chars) + symbol (1).

    Code blocks render everything literally (Discord has no table element, and
    markdown — including **bold** and *emphasis* — is not parsed inside them),
    so the markers are plain text: * = scheduled, ! = moving slowly,
    ‼ = delayed. The number is LEFT-padded to 2 chars (right-aligned) and the
    marker sits in the 3rd char (`` 6*``, ``25 ``). ETAs of 100+ minutes cap
    at ``99+`` — nothing larger than 99 is ever shown.
    """
    if row_minutes is None:
        return "  —"
    if row_minutes >= 100:
        return "99+"
    cell = str(row_minutes).rjust(2)
    if kind == EtaKind.SCHEDULED:
        cell += "*"
    elif kind == EtaKind.DELAYED:
        cell += "‼"
    elif kind == EtaKind.MOVING_SLOWLY:
        cell += "!"
    else:
        cell += " "
    return cell


def _route_line(group: RouteEtaGroup, dest_width: int) -> str:
    """Route row for a code block: color box + route + destination + ETA column.

    ``dest_width`` is the widest destination in the block (e.g. "Clear Water
    Bay" = 15), so every row's ETA column aligns. The color emoji (🟥/🟨/🟩)
    is a single-width character, so within the fixed-width code block every
    row lines up the same way. No markdown escaping is needed — code blocks
    render literally.
    """
    icon = _OPERATOR_ICON.get(group.operator, "⬜")
    cells = " ".join(_eta_cell(r.minutes, r.kind) for r in group.rows)
    return f"{icon}{group.route:<6} {group.destination:<{dest_width}} {cells}"


def _gate_block(gate_routes: list[RouteEtaGroup]) -> str:
    """Code block with column-aligned ETA rows (no header row).

    The destination column width is the longest destination in the block, so
    the ETA column aligns even with names like "Clear Water Bay" (15 chars).
    """
    dest_width = max((len(g.destination) for g in gate_routes), default=14)
    rows = ["```"]
    for group in gate_routes:
        rows.append(_route_line(group, dest_width))
    rows.append("```")
    return "\n".join(rows)


def _has_departures(group: RouteEtaGroup) -> bool:
    """Hide routes whose every ETA is unavailable (no current departure)."""
    return any(r.minutes is not None for r in group.rows)


# --------------------------------------------------------------------------
# Embed builders (each returns a list of embed "parts" to respect limits)
# --------------------------------------------------------------------------

def _display_weather_warnings(
    weather: WeatherConditions | None,
) -> list[WeatherWarning]:
    """Return active warnings in the provider's importance order."""
    if weather is None:
        return []
    return list(weather.warnings or [])


def _build_warning_icon_strip(weather: WeatherConditions | None) -> bytes | None:
    """Combine all available official warning icons into one compact thumbnail."""
    tiles: list[Image.Image] = []
    for warning in _display_weather_warnings(weather):
        if not warning.icon_data:
            continue
        try:
            with Image.open(io.BytesIO(warning.icon_data)) as source:
                icon = source.convert("RGBA")
        except (OSError, UnidentifiedImageError):
            continue
        icon.thumbnail((64, 64), Image.Resampling.LANCZOS)
        tile = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        tile.alpha_composite(icon, ((64 - icon.width) // 2, (64 - icon.height) // 2))
        tiles.append(tile)
    if not tiles:
        return None

    strip = Image.new("RGBA", (64 * len(tiles), 64), (0, 0, 0, 0))
    for index, tile in enumerate(tiles):
        strip.alpha_composite(tile, (index * 64, 0))
    output = io.BytesIO()
    strip.save(output, format="PNG", optimize=True)
    return output.getvalue()

def _build_weather_embed(weather: WeatherConditions | None) -> discord.Embed | None:
    if weather is None:
        return None
    warnings = _display_weather_warnings(weather)
    snap = weather.snapshot

    lines: list[str] = []
    if warnings:
        for w in warnings:
            line = _esc(w.name)
            if w.issued_at:
                # Relative timestamps keep the warning line readable across
                # midnight (Discord renders nearby dates as “today”/
                # “yesterday” rather than repeating a calendar date).
                line += f" · issued {_fmt_timestamp(w.issued_at, 'R')}"
            lines.append(line)

    # title = the observation reading (🌡️ temp · 🌧️ rain · 💧 humidity)
    title_parts: list[str] = []
    if snap:
        if snap.temperature_c is not None:
            title_parts.append(f"🌡️ {snap.temperature_c:.0f}°C")
        if snap.rainfall_mm is not None:
            title_parts.append(f"🌧️ {snap.rainfall_mm:.1f}mm")
        if snap.humidity_pct is not None:
            title_parts.append(f"💧 {snap.humidity_pct}%")
    title = " · ".join(title_parts) if title_parts else "🌦️ Weather"

    if not lines and not title_parts:
        return None  # nothing to show

    lines.append(
        "🔗 [HKO warnings](https://www.hko.gov.hk/en/wxinfo/dailywx/wxwarntoday.htm)"
    )
    value = "\n".join(lines)
    if len(value) > DESC_MAX:
        value = value[: DESC_MAX - 1] + "…"
    embed = discord.Embed(color=0xE0AF68, title=title, description=value)
    source_time = snap.source_time if snap and snap.source_time else weather.warning_time
    return _set_source_timestamp(embed, "HKO", source_time)


def _build_transit_embed(
    groups: list[RouteEtaGroup],
    source_time: datetime | None = None,
) -> discord.Embed | None:
    if not groups:
        return None
    lines = ["🟥 KMB · 🟨 Citybus · 🟩 Minibus (non-realtime) · * scheduled · ! slow · ‼ delayed"]
    visible = [g for g in groups if _has_departures(g)]
    if visible:
        # separate by gate with explicit headers, matching the original mockup
        for gate, label in (("N", "⬆ North Gate"), ("S", "⬇ South Gate")):
            gate_routes = [g for g in visible if g.gate == gate]
            if not gate_routes:
                continue
            lines.append(f"**{label}**")
            lines.append(_gate_block(gate_routes))
    else:
        lines.append("No departures at this time")
    lines.append(
        "🔗 [HKUST shuttle](https://cso.ust.hk/tran/stud_sh_b) · "
        "[Bus stops live](http://liveview.ust.hk/busstop/)"
    )

    value = "\n".join(lines)
    if len(value) > DESC_MAX:
        value = value[: DESC_MAX - 1] + "…"
    embed = discord.Embed(color=0xE0AF68, title="🚌 Bus stops", description=value)
    return _set_source_timestamp(embed, "Transport Department · bus ETA", source_time)


def _build_traffic_map_embed(
    webp: bytes,
    source_time: datetime | None = None,
    filename: str | None = None,
) -> discord.Embed | None:
    """Render the Google Maps base screenshot as an image pane."""
    if not webp:
        return None
    description = "[Open territory-wide view in HKeMobility](https://www.hkemobility.gov.hk/)"
    embed = discord.Embed(
        title="🗺️ Traffic map",
        color=0x2563EB,
        description=description,
    )
    embed.set_image(url=f"attachment://{filename or traffic_map_filename(webp)}")
    return _set_source_timestamp(embed, "Google traffic", source_time)


def traffic_map_filename(webp: bytes) -> str:
    """Return the bounded, content-addressed filename for a map image."""
    digest = hashlib.sha256(webp).hexdigest()[:12]
    return f"traffic-map-{digest}.webp"


def _build_traffic_map_initializing_embed(
    source_time: datetime | None = None,
) -> discord.Embed:
    """Show the map's reserved slot while its first capture is still pending."""
    embed = discord.Embed(
        title="Traffic map initializing",
        color=0x2563EB,
        description="The traffic map is being prepared.",
    )
    return _set_source_timestamp(embed, "Google traffic", source_time)


def _delay_text(delay_min: float) -> str:
    if delay_min < 1:
        return "<1 min"
    rounded = round(delay_min)
    return f"{rounded} min" if abs(delay_min - rounded) < 0.05 else f"{delay_min:.1f} min"


def _affected_routes_line(
    text: str, roads: object | None
) -> str:
    """Suffix line naming the tracked routes serving roads matched in ``text``."""
    if roads is None:
        return ""
    routes = roads.routes_for_text(text)
    if not routes:
        return ""
    return f"  ↳ affects: {', '.join(routes)}"


def _matched_road_names(text: str, roads: object | None) -> list[str]:
    """Display names of tracked roads matched in ``text``."""
    if roads is None:
        return []
    names = []
    for key in roads.match(text):
        name = roads.display_name(key)
        if name not in names:
            names.append(name)
    return names


def _build_traffic_summary_embed(
    statuses: list[TrafficCorridorStatus] | None,
    incidents: list[TrafficIncident] | None,
    source_time: datetime | None,
    roadworks: list[Roadwork] | None = None,
    stale_sources: list[str] | None = None,
    td_source_time: datetime | None = None,
    traffic_source_times: dict[str, datetime] | None = None,
    roads: object | None = None,
) -> discord.Embed:
    """List matched TD traffic-news and relevant roadwork evidence.

    Detector statuses and their timestamps remain accepted for compatibility,
    but are intentionally omitted from the user-facing summary.
    """
    lines: list[str] = []
    evidence_times: list[str] = []
    displayed_source_times: list[datetime] = []
    td_times = traffic_source_times or {}
    if incidents:
        evidence_times.append("TD traffic news")
        news_time = td_times.get("traffic_news") or td_source_time
        if news_time is not None:
            displayed_source_times.append(news_time)
    if roadworks:
        roadworks_time = td_times.get("roadworks")
        evidence_times.append(f"TD roadworks {_fmt_timestamp(roadworks_time, 't')}")
        if roadworks_time is not None:
            displayed_source_times.append(roadworks_time)
    if evidence_times:
        lines.append(" · ".join(evidence_times))
    lines.append(f"🔗 [TD traffic news]({TD_TRAFFIC_NEWS_URL})")
    if stale_sources:
        lines.append("⚠️ Stale source cache: " + ", ".join(_esc(name) for name in stale_sources))
    notices = list(incidents or [])
    if notices:
        lines.append("\n**Active traffic notices**")
        for incident in notices:
            text = f"{incident.title} {incident.description} {incident.location} {incident.road}"
            road_names = _matched_road_names(text, roads)
            matched_roads = {name.casefold().strip() for name in road_names}
            # Preserve the notice wording itself as a quote and avoid
            # repeating title/location fragments from TD's feed. A bare road
            # field is rendered by the normalized road annotation below, so
            # omit that duplicate from the quotation while retaining titles,
            # descriptions, and descriptive location wording.
            notice_parts: list[str] = []
            for part in (incident.title, incident.description, incident.location, incident.road):
                cleaned = part.strip()
                if cleaned.casefold() in matched_roads:
                    continue
                if cleaned and cleaned.casefold() not in {
                    existing.casefold() for existing in notice_parts
                }:
                    notice_parts.append(cleaned)
            for part in notice_parts:
                lines.append(f"> {_esc(part)}")
            if road_names:
                lines.append(f"  ↳ road: {', '.join(_esc(name) for name in road_names)}")
            affected = _affected_routes_line(text, roads)
            if affected:
                lines.append(_esc(affected))
    works = list(roadworks or [])
    if works:
        lines.append("\n**Relevant roadworks**")
        for work in works:
            text = f"{work.description} {work.road}"
            lines.append(f"• {_esc(work.description)}")
            road_names = _matched_road_names(text, roads)
            if road_names:
                lines.append(f"  ↳ road: {', '.join(_esc(name) for name in road_names)}")
            affected = _affected_routes_line(text, roads)
            if affected:
                lines.append(_esc(affected))

    description = "\n".join(lines)
    if len(description) > DESC_MAX:
        description = description[: DESC_MAX - 1] + "…"
    embed = discord.Embed(
        title="🚦 Traffic news",
        color=0xF59E0B if notices or works else 0x16A34A,
        description=description,
    )
    normalized_times = [
        time.replace(tzinfo=UTC) if time.tzinfo is None else time.astimezone(UTC)
        for time in displayed_source_times
    ]
    footer_time = max(normalized_times, default=None)
    return _set_source_timestamp(
        embed, "Transport Department · traffic notices", footer_time
    )


def _embed(name: str, value: str, inline: bool = False) -> discord.Embed:
    embed = discord.Embed(color=0xE0AF68)
    embed.add_field(name=name, value=value, inline=inline)
    return embed


# --------------------------------------------------------------------------
# Public
# --------------------------------------------------------------------------

def _build_error_embed(
    errors: list[str], source_time: datetime | None = None
) -> discord.Embed | None:
    """Show unavailable providers visibly instead of silently dropping them."""
    if not errors:
        return None
    value = "\n".join(f"⚠️ {_esc(e)}" for e in errors[:10])
    if len(value) > FIELD_VALUE_MAX:
        value = value[: FIELD_VALUE_MAX - 1] + "…"
    embed = _embed("⚠️ Source status", value, inline=False)
    return _set_source_timestamp(
        embed, "Dashboard check", source_time or datetime.now(UTC)
    )


def build_payload(
    weather: WeatherConditions | None,
    groups: list[RouteEtaGroup],
    statuses: list[TrafficCorridorStatus],
    incidents: list[TrafficIncident],
    capture_time: datetime | None,
    traffic_map_webp: bytes | None,
    transit_source_time: datetime | None = None,
    map_source_time: datetime | None = None,
    roadworks: list[Roadwork] | None = None,
    traffic_stale_sources: list[str] | None = None,
    traffic_source_times: dict[str, datetime] | None = None,
    traffic_source_time: datetime | None = None,
    errors: list[str] | None = None,
    roads: object | None = None,
    now: datetime | None = None,
    traffic_map_initializing: bool = False,
) -> DashboardPayload:
    """Compose the dashboard payload, enforcing every Discord limit.

    Embed order follows the user's reading flow: map → traffic → weather →
    bus ETA (bottom, most actionable).  Bus-stop live frames are no longer
    dashboard embeds; they live behind the "Bus stops live" button.
    """
    payload = DashboardPayload()
    checked_at = now or datetime.now(UTC)

    # 1. Image-first traffic map. During incremental startup, reserve this
    # slot even before the map provider has returned; a present-but-failed
    # result is handled as a genuine source error below instead.
    if traffic_map_webp:
        map_filename = traffic_map_filename(traffic_map_webp)
        map_embed = _build_traffic_map_embed(
            traffic_map_webp,
            map_source_time or checked_at,
            map_filename,
        )
        if map_embed is not None:
            payload.embeds.append(map_embed)
            payload.files.append(
                ImageAsset(
                    filename=map_filename,
                    data=traffic_map_webp,
                    content_type="image/webp",
                    label="Traffic map",
                    source_time=map_source_time,
                )
            )
    elif traffic_map_initializing:
        payload.embeds.append(
            _build_traffic_map_initializing_embed(map_source_time or checked_at)
        )

    # 3. Traffic text is a separate pane so the map remains legible.
    payload.embeds.append(
        _build_traffic_summary_embed(
            statuses=statuses,
            incidents=incidents,
            source_time=map_source_time,
            roadworks=roadworks,
            stale_sources=traffic_stale_sources,
            td_source_time=traffic_source_time or capture_time,
            traffic_source_times=traffic_source_times,
            roads=roads,
        )
    )

    # 4. Weather
    weather_embed = _build_weather_embed(weather)
    if weather_embed is not None:
        warning_icon_strip = _build_warning_icon_strip(weather)
        if warning_icon_strip:
            weather_embed.set_thumbnail(url="attachment://hko-warnings.png")
            payload.files.append(
                ImageAsset(
                    filename="hko-warnings.png",
                    data=warning_icon_strip,
                    content_type="image/png",
                    label="HKO warning icons",
                    source_time=weather.warning_time if weather else None,
                )
            )
        payload.embeds.append(weather_embed)

    # 5. Source errors — visible so an unavailable provider is never silent.
    error_embed = _build_error_embed(errors or [], checked_at)
    if error_embed is not None:
        payload.embeds.append(error_embed)

    # 6. Bus ETA — most actionable, sits at the bottom where Discord users'
    #    eyes land.
    transit_embed = _build_transit_embed(groups, transit_source_time)
    if transit_embed is not None:
        payload.embeds.append(transit_embed)

    # Enforce global caps.
    payload.embeds = payload.embeds[:EMBEDS_PER_MESSAGE_MAX]
    payload.files = payload.files[:EMBEDS_PER_MESSAGE_MAX]

    return payload


def finalize_embed(
    embed: discord.Embed,
    max_fields: int = FIELDS_PER_EMBED_MAX,
    max_chars: int = CHARS_PER_EMBED_MAX,
) -> discord.Embed:
    """Trim an embed's fields to hard limits (used for safety in tests)."""
    if len(embed.fields) > max_fields:
        fields = list(embed.fields[:max_fields])
        embed.clear_fields()
        for f in fields:
            embed.add_field(name=f.name, value=f.value, inline=f.inline)
    total = sum(len(f.value) for f in embed.fields) + len(embed.description or "")
    if total > max_chars:
        # Reduce each field to fit (naive but safe).
        overflow = total - max_chars
        for f in list(embed.fields):
            if overflow <= 0:
                break
            cut = min(overflow, len(f.value) - 10)
            if cut > 0:
                f.value = f.value[: len(f.value) - cut - 1] + "…"
                overflow -= cut
    return embed
