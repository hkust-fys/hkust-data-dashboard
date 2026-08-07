"""Renderer tests: payload composition, Discord limit enforcement, and the
explicit display states (Unavailable / Stale / No matching notice)."""

import discord

from dashboard.models import (
    EtaKind,
    Operator,
)
from dashboard.render import (
    CHARS_PER_EMBED_MAX,
    DESC_MAX,
    EMBEDS_PER_MESSAGE_MAX,
    FIELD_VALUE_MAX,
    FIELDS_PER_EMBED_MAX,
    RESOURCE_LINE,
    _build_transit_embed,
    _eta_cell,
    _route_line,
    build_payload,
    finalize_embed,
)
from tests.fixtures import sample_data as s


def test_eta_cell_highlight_and_markers():
    # cells are plain text for code blocks: * = scheduled, no bold/◀ marker
    assert _eta_cell(5, EtaKind.REALTIME) == "5"
    assert _eta_cell(4, EtaKind.SCHEDULED) == "4*"
    assert _eta_cell(6, EtaKind.DELAYED) == "6‼"
    assert _eta_cell(6, EtaKind.MOVING_SLOWLY) == "6!"
    assert _eta_cell(None, EtaKind.REALTIME) == "—"


def test_route_line_plain_text_for_code_block():
    """Route rows live in code blocks, so no markdown escaping is applied —
    asterisks stay literal and there is no ≤5-min symbol."""
    from dashboard.models import RouteEtaGroup

    group = RouteEtaGroup(
        route="91**x**", destination="Cho*i Hung", gate="S", operator=Operator.KMB,
        rows=[s.eta_row("91**x**", "Cho*i Hung", "S", 5, EtaKind.SCHEDULED, Operator.KMB)],
    )
    line = _route_line(group)
    assert "\\*" not in line  # no escaping inside code blocks
    assert "*" in line  # destination asterisk literal
    assert "5*" in line  # scheduled marker plain text, no ◀ prefix


def test_transit_embed_marks_nonrealtime_and_scheduled():
    groups = [
        {
            "route": "11B", "destination": "Choi Hung", "gate": "S", "operator": Operator.GMB,
            "rows": [s.eta_row("11B", "Choi Hung", "S", 5, EtaKind.SCHEDULED, Operator.GMB)],
        }
    ]
    from dashboard.models import RouteEtaGroup

    embed = _build_transit_embed([RouteEtaGroup(**g) for g in groups])
    assert embed is not None
    value = embed.fields[0].value
    assert "scheduled" in value.lower()
    assert "◀" not in value
    # color boxes used consistently in both legend and table rows
    assert "🟩 Minibus" in value
    assert "🟩11B" in value or "🟩 11B" in value
    # legend written so it does not start with a markdown bullet character
    assert "scheduled = *" in value
    # the route rows live in a code block with column headers
    assert "```" in value
    assert "ETA (mins)" in value


def test_transit_embed_gate_blocks_align_columns():
    from dashboard.models import RouteEtaGroup

    groups = [
        RouteEtaGroup(
            route="91", destination="Diamond Hill", gate="S", operator=Operator.KMB,
            rows=[s.eta_row("91", "Diamond Hill", "S", 3)],
        ),
        RouteEtaGroup(
            route="11S", destination="Po Lam", gate="N", operator=Operator.GMB,
            rows=[s.eta_row("11S", "Po Lam", "N", 5, EtaKind.REALTIME, Operator.GMB)],
        ),
    ]
    embed = _build_transit_embed(groups)
    value = embed.fields[0].value
    # both gate headers present
    assert "North Gate" in value and "South Gate" in value
    # each gate's rows are inside its own code block
    north_start = value.index("North Gate")
    south_start = value.index("South Gate")
    assert value.index("```", north_start) < south_start


def test_transit_embed_hides_routes_without_departures():
    from dashboard.models import RouteEtaGroup

    groups = [
        RouteEtaGroup(
            route="91", destination="Diamond Hill", gate="S", operator=Operator.KMB,
            rows=[s.eta_row("91", "Diamond Hill", "S", None)],
        ),
        RouteEtaGroup(
            route="11S", destination="Po Lam", gate="N", operator=Operator.GMB,
            rows=[s.eta_row("11S", "Po Lam", "N", 5, EtaKind.REALTIME, Operator.GMB)],
        ),
    ]
    embed = _build_transit_embed(groups)
    assert embed is not None
    value = embed.fields[0].value
    assert "11S" in value
    assert "91 Diamond Hill" not in value  # no departures -> hidden


def test_transit_embed_no_departures_shows_message():
    """Routes without departures render a clear state."""
    from dashboard.models import RouteEtaGroup

    groups = [
        RouteEtaGroup(
            route="91", destination="Diamond Hill", gate="S", operator=Operator.KMB,
            rows=[s.eta_row("91", "Diamond Hill", "S", None)],
        ),
    ]
    embed = _build_transit_embed(groups)
    assert embed is not None
    assert "No departures" in embed.fields[0].value


def test_error_embed_shows_provider_failures():
    from dashboard.render import _build_error_embed

    assert _build_error_embed([]) is None
    embed = _build_error_embed(["KMB ETA unavailable", "TD CCTV unavailable"])
    assert embed is not None
    value = embed.fields[0].value
    assert "KMB ETA unavailable" in value
    assert "TD CCTV unavailable" in value


def test_traffic_map_embed_is_the_only_traffic_pane():
    """The map embed carries the summary; there is no separate text pane."""
    from dashboard.render import _build_traffic_map_embed

    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    embed = _build_traffic_map_embed(png, s.traffic_statuses(), [], s.utc())
    assert embed is not None
    desc = embed.description
    # red corridor flagged compactly, no per-detector speed dump
    assert "Clear Water Bay Road" in desc
    assert "km/h" not in desc
    # HKeMobility link is inlined on the map embed
    assert "HKeMobility" in embed.fields[-1].value
    assert embed.image.url == "attachment://traffic-map.png"


def test_traffic_map_embed_omitted_without_png():
    from dashboard.render import _build_traffic_map_embed

    assert _build_traffic_map_embed(None, [], [], None) is None


def test_traffic_map_embed_fallback_description_when_clear():
    """All-clear still renders the map with a neutral description."""
    from dashboard.models import SpeedBand, TrafficCorridorStatus, TrafficObservation
    from dashboard.render import _build_traffic_map_embed

    clear = TrafficCorridorStatus(
        name="Clear Water Bay Road",
        direction="",
        observations=[
            TrafficObservation(
                corridor="Clear Water Bay Road", direction="", description="x",
                latitude=22.33, longitude=114.22, speed_kmh=60, volume=1,
                occupancy_pct=0.1, capture_time=s.utc(), band=SpeedBand.GREEN,
            )
        ],
        capture_time=s.utc(),
    )
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    embed = _build_traffic_map_embed(png, [clear], [], None)
    assert embed is not None
    assert "monitored points" in embed.description


def test_weather_embed_multiple_warning_icons():
    """With several active warnings, each icon renders inline in the field
    (an embed has only one thumbnail slot, so the first icon is used only
    when there is a single warning)."""
    from dashboard.models import WeatherConditions, WeatherWarning
    from dashboard.render import _build_weather_embed

    warnings = [
        WeatherWarning(code="TC8NE", name="Gale Signal No. 8 NE",
                       icon_url="https://hko.example/tc8ne.issuing.gif"),
        WeatherWarning(code="WRAINB", name="Black Rainstorm",
                       icon_url="https://hko.example/rainb.issuing.gif"),
    ]
    embed = _build_weather_embed(WeatherConditions(warnings=warnings))
    assert embed is not None
    value = embed.fields[0].value
    assert "tc8ne.issuing.gif" in value
    assert "rainb.issuing.gif" in value
    assert not embed.thumbnail.url  # multiple icons -> no single thumbnail

    single = _build_weather_embed(WeatherConditions(warnings=warnings[:1]))
    assert single.thumbnail is not None
    assert single.thumbnail.url.endswith("tc8ne.issuing.gif")


def test_build_payload_respects_embed_caps():
    from dashboard.models import WeatherConditions

    payload = build_payload(
        weather=WeatherConditions(
            warnings=s.weather_warnings(), snapshot=s.weather_snapshot()
        ),
        groups=[],
        statuses=s.traffic_statuses(),
        incidents=s.traffic_incidents(),
        capture_time=s.utc(),
        traffic_map_png=b"\x89PNG\r\n" + b"\x00" * 100,
        cctv_images=s.cctv_assets(),
    )
    assert len(payload.embeds) <= EMBEDS_PER_MESSAGE_MAX
    assert len(payload.files) <= EMBEDS_PER_MESSAGE_MAX
    for embed in payload.embeds:
        assert len(embed.fields) <= FIELDS_PER_EMBED_MAX
        for field in embed.fields:
            assert len(field.value) <= FIELD_VALUE_MAX
        assert len(embed.description or "") <= DESC_MAX
        total = sum(len(f.value) for f in embed.fields) + len(embed.description or "")
        assert total <= CHARS_PER_EMBED_MAX


def test_finalize_embed_enforces_field_cap():
    embed = discord.Embed()
    for i in range(30):
        embed.add_field(name=f"f{i}", value="x")
    embed = finalize_embed(embed)
    assert len(embed.fields) <= FIELDS_PER_EMBED_MAX


def test_resources_line_links():
    for link in ("cso.ust.hk", "liveview.ust.hk", "hko.gov.hk", "hkemobility.gov.hk"):
        assert link in RESOURCE_LINE
