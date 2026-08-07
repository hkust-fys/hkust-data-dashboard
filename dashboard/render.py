"""Limit-aware multi-embed renderer.

Builds a ``DashboardPayload`` (ordered embeds + attachments) from provider
results, enforcing Discord limits: field value 1024, description 4096, 25
fields/embed, 6000 aggregate chars/embed, 10 embeds/files per message.
"""

from __future__ import annotations

from datetime import UTC, datetime

import discord

from dashboard.models import (
    DashboardPayload,
    EtaKind,
    ImageAsset,
    Operator,
    RouteEtaGroup,
    SpeedBand,
    TrafficCorridorStatus,
    TrafficIncident,
    WeatherConditions,
)

# Discord limits (from the discord.py docs).
FIELD_VALUE_MAX = 1024
DESC_MAX = 4096
FIELDS_PER_EMBED_MAX = 25
CHARS_PER_EMBED_MAX = 6000
EMBEDS_PER_MESSAGE_MAX = 10

DASHBOARD_MARKER = "HKUST Campus Dashboard"
DASHBOARD_FOOTER = "hkust-data-dashboard · updated"

RESOURCE_LINE = (
    "[HKUST shuttle](https://cso.ust.hk/tran/stud_sh_b) · "
    "[Bus stops live](http://liveview.ust.hk/busstop/) · "
    "[HKO warnings](https://www.hko.gov.hk/en/wxinfo/currwx/warning.htm) · "
    "[HKeMobility](https://www.hkemobility.gov.hk/)"
)

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


def _fmt_timestamp(dt: datetime | None, fmt: str = "%H:%M") -> str:
    """Discord timestamp that renders in each viewer's local timezone.

    `style` is a Discord timestamp letter: t = HH:MM, f = full date + time,
    R = relative ("2 hours ago"). Returns "—" when the datetime is missing.
    """
    if dt is None:
        return "—"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return f"<t:{int(dt.timestamp())}:{fmt}>"


def _discord_rel(ts: datetime | None) -> str:
    """Relative Discord timestamp (renders as "2 hours ago" per viewer)."""
    if ts is None:
        return "—"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return f"<t:{int(ts.timestamp())}:R>"


def _eta_cell(row_minutes: int | None, kind: EtaKind) -> str:
    """ETA cell for use inside a code block.

    Code blocks render everything literally (Discord has no table element, and
    markdown — including **bold** and *emphasis* — is not parsed inside them),
    so the markers are plain text: * = scheduled, ! = moving slowly,
    ‼ = delayed. There is no ≤5-min marker: without bold it would just be
    noise, and the minutes themselves are already the clearest signal.
    """
    if row_minutes is None:
        return "—"
    cell = str(row_minutes)
    if kind == EtaKind.SCHEDULED:
        cell += "*"
    elif kind == EtaKind.DELAYED:
        cell += "‼"
    elif kind == EtaKind.MOVING_SLOWLY:
        cell += "!"
    return cell


def _route_line(group: RouteEtaGroup) -> str:
    """Route row for a code block: color box + route + destination + ETA column.

    The color emoji (🟥/🟨/🟩) is a single-width character, so within the
    fixed-width code block every row lines up the same way. No markdown
    escaping is needed — code blocks render literally.
    """
    icon = _OPERATOR_ICON.get(group.operator, "⬜")
    cells = ", ".join(_eta_cell(r.minutes, r.kind) for r in group.rows)
    return f"{icon}{group.route:<6} {group.destination:<14} {cells}"


def _gate_block(gate_routes: list[RouteEtaGroup]) -> str:
    """Code block with a fixed header and column-aligned ETA rows."""
    rows = ["```", f"{'Route':<21}ETA (mins)", "-" * 30]
    for group in gate_routes:
        rows.append(_route_line(group))
    rows.append("```")
    return "\n".join(rows)


def _has_departures(group: RouteEtaGroup) -> bool:
    """Hide routes whose every ETA is unavailable (no current departure)."""
    return any(r.minutes is not None for r in group.rows)


# --------------------------------------------------------------------------
# Embed builders (each returns a list of embed "parts" to respect limits)
# --------------------------------------------------------------------------

def _build_weather_embed(weather: WeatherConditions | None) -> discord.Embed | None:
    if weather is None:
        return None
    warnings = weather.warnings or []
    snap = weather.snapshot

    lines: list[str] = []
    thumbnail: str | None = None
    if warnings:
        # show every active warning icon inline (an embed has only one
        # thumbnail slot, so icons are embedded in the text)
        icons = [w.icon_url for w in warnings if w.icon_url]
        if len(icons) == 1:
            thumbnail = icons[0]
        elif icons:
            lines.append(" ".join(f"[!]({u})" for u in icons))
        for w in warnings:
            line = f"**{_esc(w.name)}**"
            if w.summary:
                line += f" — {_esc(w.summary)}"
            lines.append(line)
        if weather.warning_time:
            lines.append(f"_Issued {_discord_rel(weather.warning_time)}_")
    if snap:
        parts = []
        if snap.temperature_c is not None:
            parts.append(f"🌡️ {snap.temperature_c:.0f}°C")
        if snap.rainfall_mm is not None:
            parts.append(f"🌧️ {snap.rainfall_mm:.1f}mm")
        if snap.humidity_pct is not None:
            parts.append(f"💧 {snap.humidity_pct}% humidity")
        if parts:
            lines.append(" · ".join(parts))
        lines.append(f"_Sai Kung station, HKO {_fmt_timestamp(snap.source_time, 't')}_")
    elif not warnings:
        return None  # nothing to show

    if not lines:
        return None
    value = "\n".join(lines)
    if len(value) > FIELD_VALUE_MAX:
        value = value[: FIELD_VALUE_MAX - 1] + "…"
    embed = _embed("🌦️ Weather", value, inline=False)
    if thumbnail:
        embed.set_thumbnail(url=thumbnail)
    embed.add_field(
        name="🔗",
        value="[HKO warnings](https://www.hko.gov.hk/en/wxinfo/currwx/warning.htm)",
        inline=False,
    )
    return embed


def _build_transit_embed(groups: list[RouteEtaGroup]) -> discord.Embed | None:
    if not groups:
        return None
    lines: list[str] = []
    visible = [g for g in groups if _has_departures(g)]
    lines.append("🟥 KMB · 🟨 Citybus · 🟩 Minibus (non-realtime)")
    # legend written so it does not start with a markdown bullet character
    lines.append("scheduled = * · moving slowly = ! · delayed = ‼")
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

    value = "\n".join(lines)
    if len(value) > FIELD_VALUE_MAX:
        # Split into two fields if needed.
        truncated = value[: FIELD_VALUE_MAX - 1] + "…"
        return _embed("🚌 Bus stops", truncated, inline=False)

    embed = _embed("🚌 Bus stops", value, inline=False)
    embed.add_field(
        name="🔗",
        value="[HKUST shuttle](https://cso.ust.hk/tran/stud_sh_b) · "
              "[Bus stops live](http://liveview.ust.hk/busstop/)",
        inline=False,
    )
    return embed


def _traffic_summary(statuses: list[TrafficCorridorStatus]) -> str | None:
    """Compact one-liner for the map embed description.

    The map itself shows every detector; this just flags corridors that are
    moving slowly or congested so the important info is visible at a glance.
    """
    flags = []
    for status in statuses:
        bands = {o.band for o in status.observations}
        if SpeedBand.RED in bands:
            flags.append(f"🔴 {_esc(status.name)} slow")
        elif SpeedBand.AMBER in bands:
            flags.append(f"🟠 {_esc(status.name)} heavy")
    if not flags:
        return None
    return " · ".join(flags[:3])


def _build_traffic_map_embed(
    png: bytes,
    statuses: list[TrafficCorridorStatus] | None = None,
    incidents: list[TrafficIncident] | None = None,
    capture_time: datetime | None = None,
) -> discord.Embed | None:
    if not png:
        return None
    lines = []
    summary = _traffic_summary(statuses or [])
    if summary:
        lines.append(summary)
    for inc in (incidents or [])[:2]:
        lines.append(f"⚠️ **{_esc(inc.title)}**")
    if capture_time:
        lines.append(f"_TD detectors {_fmt_timestamp(capture_time, 't')}_")
    if not lines:
        lines.append("_Detector speeds at monitored points; gray = no fresh observation_")
    embed = discord.Embed(
        title="🗺️ Traffic map — HKUST approaches",
        color=0x2563EB,
        description="\n".join(lines),
    )
    embed.set_image(url="attachment://traffic-map.png")
    embed.add_field(
        name="🔗",
        value="[HKeMobility](https://www.hkemobility.gov.hk/) — territory-wide live map",
        inline=False,
    )
    return embed


def _build_cctv_embeds(images: list[ImageAsset]) -> list[discord.Embed]:
    embeds = []
    for asset in images:
        embed = discord.Embed(title=f"📷 {asset.label}", color=0x0F766E)
        embed.set_image(url=f"attachment://{asset.filename}")
        if asset.source_time:
            # embed.timestamp is the native auto-localizing timestamp field;
            # <t:...> markdown does NOT parse inside footer text.
            embed.timestamp = asset.source_time
        embeds.append(embed)
    return embeds


def _embed(name: str, value: str, inline: bool = False) -> discord.Embed:
    embed = discord.Embed(color=0xE0AF68)
    embed.add_field(name=name, value=value, inline=inline)
    return embed


# --------------------------------------------------------------------------
# Public
# --------------------------------------------------------------------------

def _build_error_embed(errors: list[str]) -> discord.Embed | None:
    """Show unavailable providers visibly instead of silently dropping them."""
    if not errors:
        return None
    value = "\n".join(f"⚠️ {_esc(e)}" for e in errors[:10])
    if len(value) > FIELD_VALUE_MAX:
        value = value[: FIELD_VALUE_MAX - 1] + "…"
    return _embed("⚠️ Source status", value, inline=False)


def build_payload(
    weather: WeatherConditions | None,
    groups: list[RouteEtaGroup],
    statuses: list[TrafficCorridorStatus],
    incidents: list[TrafficIncident],
    capture_time: datetime | None,
    traffic_map_png: bytes | None,
    cctv_images: list[ImageAsset],
    errors: list[str] | None = None,
    now: datetime | None = None,
) -> DashboardPayload:
    """Compose the dashboard payload, enforcing every Discord limit.

    Embed order follows Discord reading flow: users scroll bottom-up, so the
    most important/actionable pane (bus stops) is placed last (bottom), and
    the least important (CCTV) comes first (top).
    """
    payload = DashboardPayload(footer_text=DASHBOARD_FOOTER)

    # 1. CCTV (least important — nice-to-have point views)
    cctv_embeds = _build_cctv_embeds(cctv_images)
    for embed, asset in zip(cctv_embeds, cctv_images, strict=True):
        payload.embeds.append(embed)
        payload.files.append(asset)

    # 2. Source errors — visible so an unavailable provider is never silent.
    error_embed = _build_error_embed(errors or [])
    if error_embed is not None:
        payload.embeds.append(error_embed)

    # 3. Traffic map — the map embed carries the summary, incidents, capture
    #    time and HKeMobility link, so no separate text pane is needed.
    if traffic_map_png:
        map_embed = _build_traffic_map_embed(traffic_map_png, statuses, incidents, capture_time)
        if map_embed is not None:
            payload.embeds.append(map_embed)
            payload.files.append(
                ImageAsset(
                    filename="traffic-map.png",
                    data=traffic_map_png,
                    content_type="image/png",
                    label="Traffic map",
                )
            )

    # 4. Weather
    weather_embed = _build_weather_embed(weather)
    if weather_embed is not None:
        payload.embeds.append(weather_embed)

    # 5. Transit — most important, so it sits at the bottom where Discord
    #    users' eyes land.
    transit_embed = _build_transit_embed(groups)
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
        embed.clear_fields()
        for f in embed.fields[:max_fields]:
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
