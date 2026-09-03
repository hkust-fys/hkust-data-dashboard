"""Renderer tests: payload composition, Discord limit enforcement, and the
explicit display states (Unavailable / Stale / No matching notice)."""

import io
from datetime import timedelta

import discord
from PIL import Image

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
    _build_transit_embed,
    _eta_cell,
    _route_line,
    build_payload,
    finalize_embed,
)
from tests.fixtures import sample_data as s


def _icon_png(color: tuple[int, int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", (32, 32), color).save(output, format="PNG")
    return output.getvalue()


def test_eta_cell_highlight_and_markers():
    # cells are plain text for code blocks, number LEFT-padded to 2 chars
    # (right-aligned) + symbol in the 3rd char: * = scheduled, no bold/◀.
    assert _eta_cell(5, EtaKind.REALTIME) == " 5 "
    assert _eta_cell(4, EtaKind.SCHEDULED) == " 4*"
    assert _eta_cell(6, EtaKind.DELAYED) == " 6‼"
    assert _eta_cell(6, EtaKind.MOVING_SLOWLY) == " 6!"
    assert _eta_cell(25, EtaKind.REALTIME) == "25 "
    assert _eta_cell(100, EtaKind.REALTIME) == "99+"
    assert _eta_cell(125, EtaKind.REALTIME) == "99+"
    assert _eta_cell(None, EtaKind.REALTIME) == "  —"


def test_route_line_plain_text_for_code_block():
    """Route rows live in code blocks, so no markdown escaping is applied —
    asterisks stay literal and there is no ≤5-min symbol."""
    from dashboard.models import RouteEtaGroup

    group = RouteEtaGroup(
        route="91**x**", destination="Cho*i Hung", gate="S", operator=Operator.KMB,
        rows=[s.eta_row("91**x**", "Cho*i Hung", "S", 5, EtaKind.SCHEDULED, Operator.KMB)],
    )
    line = _route_line(group, 14)
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
    assert embed.title == "🚌 Bus stops"
    value = embed.description
    assert "scheduled" in value.lower()
    assert "◀" not in value
    # color boxes used consistently in both legend and table rows
    assert "🟩 Minibus" in value
    assert "🟩11B" in value or "🟩 11B" in value
    # legend is the first line under the title
    assert value.startswith("🟥 KMB · 🟨 Citybus · 🟩 Minibus (non-realtime)")
    assert "* scheduled" in value
    assert "! slow" in value
    assert "‼ delayed" in value
    # the route rows live in a code block; the header row was removed
    assert "```" in value
    assert "ETA (mins)" not in value
    assert "------" not in value
    # links inline in the description, not a separate 🔗 field
    assert "🔗 [HKUST shuttle]" in value
    assert "[Bus stops live]" in value
    assert not embed.fields
    assert embed.footer.text == "Transport Department · bus ETA"


def test_transit_embed_route_line_no_commas():
    """ETA minutes join with a space, not a comma."""
    from dashboard.models import RouteEtaGroup

    group = RouteEtaGroup(
        route="91", destination="Diamond Hill", gate="S", operator=Operator.KMB,
        rows=[s.eta_row("91", "Diamond Hill", "S", 2), s.eta_row("91", "Diamond Hill", "S", 20)],
    )
    line = _route_line(group, 14)
    assert " 2  20 " in line  # left-padded numbers + marker column
    assert "," not in line


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
    value = embed.description
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
    value = embed.description
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
    assert "No departures" in embed.description


def test_error_embed_shows_provider_failures():
    from dashboard.render import _build_error_embed

    assert _build_error_embed([]) is None
    embed = _build_error_embed(["KMB ETA unavailable", "bus-stop cameras unavailable"])
    assert embed is not None
    value = embed.fields[0].value
    assert "KMB ETA unavailable" in value
    assert "bus-stop cameras unavailable" in value


def test_traffic_map_embed_is_image_onlyish_and_timestamped():
    from dashboard.render import _build_traffic_map_embed, traffic_map_filename

    webp = b"RIFF" + b"\x00" * 64
    embed = _build_traffic_map_embed(webp, s.utc())
    assert embed is not None
    assert embed.title == "🗺️ Traffic map"
    desc = embed.description
    assert "min" not in desc
    assert "delay" not in desc
    assert "HKeMobility" in desc
    assert not embed.fields
    assert embed.image.url == f"attachment://{traffic_map_filename(webp)}"
    assert embed.timestamp == s.utc()
    assert embed.footer.text == "Google traffic"


def test_traffic_map_embed_omitted_without_png():
    from dashboard.render import _build_traffic_map_embed

    assert _build_traffic_map_embed(None, s.utc()) is None


def test_traffic_map_filename_is_stable_and_content_addressed():
    from dashboard.render import traffic_map_filename

    assert traffic_map_filename(b"same") == traffic_map_filename(b"same")
    assert traffic_map_filename(b"same") != traffic_map_filename(b"changed")
    assert traffic_map_filename(b"same").startswith("traffic-map-")
    assert traffic_map_filename(b"same").endswith(".webp")


def test_build_payload_omits_map_legend_without_map():
    payload = build_payload(
        weather=None,
        groups=[],
        statuses=[],
        incidents=[],
        capture_time=s.utc(),
        traffic_map_webp=None,
    )
    assert all(embed.title != "🗺️ Map legend" for embed in payload.embeds)
    assert all(file.filename != "traffic-map-legend.png" for file in payload.files)


def test_build_payload_shows_initializing_map_in_reserved_first_slot():
    now = s.utc()
    payload = build_payload(
        weather=None,
        groups=[],
        statuses=[],
        incidents=[],
        capture_time=now,
        traffic_map_webp=None,
        traffic_map_initializing=True,
        now=now,
    )
    assert payload.embeds[0].title == "Traffic map initializing"
    assert payload.embeds[1].title == "🚦 Traffic news"
    assert payload.embeds[0].timestamp == now
    assert not payload.embeds[0].image.url
    assert not payload.files


def test_build_payload_present_failed_map_keeps_source_error():
    payload = build_payload(
        weather=None,
        groups=[],
        statuses=[],
        incidents=[],
        capture_time=s.utc(),
        traffic_map_webp=None,
        errors=["traffic map unavailable"],
    )
    assert payload.embeds[0].title == "🚦 Traffic news"
    assert "traffic map unavailable" in payload.embeds[-1].fields[0].value


def test_payload_keeps_one_map_attachment_before_traffic_news():
    source = io.BytesIO()
    Image.new("RGB", (960, 540), (80, 120, 160)).save(source, format="WEBP")
    payload = build_payload(
        weather=None, groups=[], statuses=[], incidents=[], capture_time=s.utc(),
        traffic_map_webp=source.getvalue(), map_source_time=s.utc(),
    )
    assert [embed.title for embed in payload.embeds[:2]] == [
        "🗺️ Traffic map", "🚦 Traffic news"
    ]
    assert len(payload.files) == 1
    assert payload.embeds[0].image.url == f"attachment://{payload.files[0].filename}"
    map_image = Image.open(io.BytesIO(payload.files[0].data))
    assert map_image.size == (960, 540)


def test_traffic_map_and_summary_label_stale_or_unavailable_data():
    from dashboard.render import _build_traffic_map_embed, _build_traffic_summary_embed

    map_embed = _build_traffic_map_embed(b"png", s.utc())
    assert "Stale last-good" not in map_embed.description

    stale_summary = _build_traffic_summary_embed(
        [], [], s.utc(), roadworks=[], stale_sources=["TD roadworks"]
    )
    assert "Stale source cache: TD roadworks" in stale_summary.description

    unavailable = _build_traffic_summary_embed([], [], s.utc())
    assert "Traffic route estimate unavailable" not in unavailable.description


def test_traffic_summary_includes_relevant_roadworks():
    from dashboard.models import Roadwork
    from dashboard.render import _build_traffic_summary_embed

    embed = _build_traffic_summary_embed(
        [],
        [],
        None,
        roadworks=[Roadwork("rw", "Lane closure on Clear Water Bay Road", "CWB")],
        traffic_source_times={"roadworks": s.utc()},
    )
    assert "Relevant roadworks" in embed.description
    assert "Lane closure" in embed.description
    assert "TD roadworks" in embed.description
    assert embed.timestamp == s.utc()


def test_traffic_summary_lists_affected_routes_for_news_and_roadworks():
    from dashboard.models import Roadwork, TrafficIncident
    from dashboard.providers.route_geometry import RouteLine
    from dashboard.providers.tracked_roads import build_tracked_roads
    from dashboard.render import _build_traffic_summary_embed

    lines = [
        RouteLine("91M", "KMB", "outbound"),
        RouteLine("792M", "CTB", "outbound"),
    ]
    roads = build_tracked_roads(lines, [["Hang Hau Road"], ["Hang Hau Road"]])
    incident = TrafficIncident(
        identifier="tn",
        title="Lane closure on Hang Hau Road",
        description="One lane closed",
        road="Hang Hau Road",
        location="Hang Hau",
        direction="",
        status="active",
    )
    embed = _build_traffic_summary_embed(
        [],
        [incident],
        None,
        roadworks=[Roadwork("rw", "Resurfacing on Hang Hau Road", "Hang Hau Road")],
        roads=roads,
    )
    assert "affects: 91M, 792M" in embed.description
    assert embed.description.count("affects:") == 2
    assert "> Lane closure on Hang Hau Road" in embed.description
    assert "> One lane closed" in embed.description

    # without a road table, no suffix renders
    plain = _build_traffic_summary_embed([], [incident], None)
    assert "affects:" not in plain.description


def test_traffic_summary_names_affected_road_per_item():
    from dashboard.models import Roadwork, TrafficIncident
    from dashboard.providers.route_geometry import RouteLine
    from dashboard.providers.tracked_roads import build_tracked_roads
    from dashboard.render import _build_traffic_summary_embed

    lines = [RouteLine("91M", "KMB", "outbound"), RouteLine("792M", "CTB", "outbound")]
    roads = build_tracked_roads(lines, [["Hang Hau Road"], ["Hang Hau Road"]])
    incident = TrafficIncident(
        identifier="tn",
        title="Lane closure on Hang Hau Road",
        description="One lane closed",
        road="Hang Hau Road",
        location="Hang Hau",
        direction="",
        status="active",
    )
    embed = _build_traffic_summary_embed(
        [], [incident], None,
        roadworks=[Roadwork("rw", "Resurfacing near Hang Hau Road", "")],
        roads=roads,
    )
    # each item carries its matched road display name
    assert embed.description.count("↳ road: Hang Hau Road") == 2
    # longest-match-wins still applies for overlapping names
    cwb_roads = build_tracked_roads(
        [RouteLine("91", "KMB", "outbound")],
        [["New Clear Water Bay Road"]],
    )
    cwb_incident = TrafficIncident(
        identifier="tn2", title="Works on New Clear Water Bay Road",
        description="", road="", location="", direction="", status="active",
    )
    cwb_embed = _build_traffic_summary_embed([], [cwb_incident], None, roads=cwb_roads)
    assert "↳ road: New Clear Water Bay Road" in cwb_embed.description
    assert "Clear Water Bay Road\n" not in cwb_embed.description.replace(
        "New Clear Water Bay Road", ""
    )


def test_traffic_summary_omits_bare_road_from_quote_but_keeps_annotation():
    from dashboard.models import TrafficIncident
    from dashboard.providers.route_geometry import RouteLine
    from dashboard.providers.tracked_roads import build_tracked_roads
    from dashboard.render import _build_traffic_summary_embed

    roads = build_tracked_roads(
        [RouteLine("91", "KMB", "outbound")], [["Waterloo Road"]]
    )
    incident = TrafficIncident(
        identifier="waterloo",
        title="Traffic congestion",
        description="Waterloo Road is slow near the junction",
        road="Waterloo Road",
        location="Waterloo Road",
        direction="",
        status="active",
    )
    embed = _build_traffic_summary_embed([], [incident], None, roads=roads)
    assert "\n> Waterloo Road\n" not in embed.description
    assert "> Traffic congestion" in embed.description
    assert "> Waterloo Road is slow near the junction" in embed.description
    assert "↳ road: Waterloo Road" in embed.description


def test_traffic_summary_does_not_annotate_direction_only_tracked_road():
    from dashboard.models import TrafficIncident
    from dashboard.providers.tracked_roads import TrackedRoads
    from dashboard.render import _build_traffic_summary_embed

    names = {
        "tseung kwan o tunnel": "Tseung Kwan O Tunnel",
        "tseung kwan o tunnel road": "Tseung Kwan O Tunnel Road",
    }
    roads = TrackedRoads(
        display_names=names,
        aliases={key: key for key in names},
        road_routes={key: ("12",) for key in names},
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

    embed = _build_traffic_summary_embed([], [incident], None, roads=roads)

    assert "Tseung Kwan O Tunnel bound" in embed.description
    assert "↳ road:" not in embed.description
    assert "↳ affects:" not in embed.description


def test_traffic_summary_links_to_official_td_traffic_news():
    from dashboard.render import TD_TRAFFIC_NEWS_URL, _build_traffic_summary_embed

    embed = _build_traffic_summary_embed([], [], None)

    assert f"[TD traffic news]({TD_TRAFFIC_NEWS_URL})" in embed.description
    assert embed.footer.text == "Transport Department · traffic notices"


def test_traffic_summary_keeps_each_displayed_source_time_separate():
    from dashboard.models import Roadwork
    from dashboard.render import _build_traffic_summary_embed

    base = s.utc()
    google_time = base - timedelta(minutes=40)
    detector_time = base - timedelta(minutes=30)
    news_time = base - timedelta(minutes=20)
    roadworks_time = base - timedelta(minutes=10)
    embed = _build_traffic_summary_embed(
        s.traffic_statuses(),
        s.traffic_incidents(),
        google_time,
        roadworks=[Roadwork("rw", "Roadworks", "CWB")],
        traffic_source_times={
            "detectors": detector_time,
            "traffic_news": news_time,
            "roadworks": roadworks_time,
        },
    )

    assert "TD detectors" not in embed.description
    assert "TD monitored slow points" not in embed.description
    # The news timestamp lives only in the footer now (no duplicated line).
    assert "TD traffic news updated" not in embed.description
    assert (
        f"TD roadworks <t:{int(roadworks_time.timestamp())}:t>" in embed.description
    )
    assert embed.timestamp == roadworks_time
    assert embed.footer.text == "Transport Department · traffic notices"


def test_traffic_summary_ignores_detector_statuses_and_color():
    from dashboard.render import _build_traffic_summary_embed

    embed = _build_traffic_summary_embed(
        s.traffic_statuses(),
        [],
        s.utc(),
        traffic_source_times={"detectors": s.utc()},
    )
    assert "Traffic route estimate unavailable" not in embed.description
    assert "TD detectors" not in embed.description
    assert "TD monitored slow points" not in embed.description
    assert embed.color.value == 0x16A34A
    assert embed.timestamp is None
    assert embed.footer.text == "Transport Department · traffic notices"


def test_weather_embed_multiple_warning_icons():
    """Direct weather embeds never expose animated source GIF URLs."""
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
    value = embed.description
    assert "tc8ne.issuing.gif" not in value
    assert "rainb.issuing.gif" not in value
    assert "Gale Signal No. 8 NE" in value  # warning name is inline text, not a title
    assert embed.thumbnail.url is None
    assert "thumbnail" not in embed.to_dict()

    single = _build_weather_embed(WeatherConditions(warnings=warnings[:1]))
    assert single.thumbnail.url is None
    assert "thumbnail" not in single.to_dict()


def test_weather_embed_omits_thumbnail_when_static_icons_are_unavailable():
    from dashboard.models import WeatherConditions, WeatherWarning
    from dashboard.render import _build_weather_embed

    embed = _build_weather_embed(WeatherConditions(warnings=[WeatherWarning(
        code="WRAINA", name="Amber Rainstorm Warning Signal",
        icon_url="https://www.hko.gov.hk/images_e/raina.gif",
    )]))

    assert embed is not None
    assert embed.thumbnail.url is None
    assert "thumbnail" not in embed.to_dict()


def test_weather_embed_omits_long_warning_statement():
    from dashboard.models import WeatherConditions, WeatherWarning
    from dashboard.render import _build_weather_embed

    embed = _build_weather_embed(WeatherConditions(warnings=[WeatherWarning(
        code="WRAINA",
        name="Amber Rainstorm Warning Signal",
        summary="A deliberately long HKO warning statement that belongs on the source page.",
    )]))

    assert embed is not None
    assert "Amber Rainstorm Warning Signal" in embed.description
    assert "deliberately long" not in embed.description


def test_weather_embed_shows_all_warnings_once_in_one_icon_strip():
    from dashboard.models import WeatherConditions, WeatherWarning
    from dashboard.render import build_payload

    weather = WeatherConditions(warnings=[
        WeatherWarning("WRAINA", "Amber Rainstorm Warning Signal",
                       icon_url="https://www.hko.gov.hk/images_e/raina.gif",
                       icon_data=_icon_png((255, 191, 0, 255))),
        WeatherWarning("WTS", "Thunderstorm Warning",
                       icon_url="https://www.hko.gov.hk/images_e/ts.gif",
                       icon_data=_icon_png((0, 0, 255, 255))),
        WeatherWarning("TC3", "Strong Wind Signal No. 3",
                       icon_url="https://www.hko.gov.hk/images_e/tc3.gif",
                       icon_data=_icon_png((255, 0, 0, 255))),
    ])

    payload = build_payload(weather, [], [], [], None, None)
    descriptions = "\n".join(embed.description or "" for embed in payload.embeds)

    assert descriptions.count("Strong Wind Signal No. 3") == 1
    assert descriptions.count("Amber Rainstorm Warning Signal") == 1
    assert descriptions.count("Thunderstorm Warning") == 1
    assert [embed.thumbnail.url for embed in payload.embeds if embed.thumbnail] == [
        "attachment://hko-warnings.png",
    ]
    assert [asset.filename for asset in payload.files] == ["hko-warnings.png"]
    strip = Image.open(io.BytesIO(payload.files[0].data))
    assert strip.size == (192, 64)


def test_build_payload_combines_warning_icons_in_one_thumbnail():
    from dashboard.models import WeatherConditions, WeatherWarning
    from dashboard.render import build_payload

    weather = WeatherConditions(warnings=[
        WeatherWarning("TC8NE", "Gale Signal No. 8 NE",
                       icon_url="https://hko.example/tc8.gif",
                       icon_data=_icon_png((255, 0, 0, 255))),
        WeatherWarning("WRAINA", "Amber Rainstorm Warning Signal",
                       icon_url="https://hko.example/rain.gif",
                       icon_data=_icon_png((255, 191, 0, 255))),
    ])
    payload = build_payload(weather, [], [], [], None, None)
    weather_embeds = [e for e in payload.embeds if e.thumbnail]
    assert [e.thumbnail.url for e in weather_embeds] == [
        "attachment://hko-warnings.png"
    ]
    assert len(payload.files) == 1
    strip = Image.open(io.BytesIO(payload.files[0].data))
    assert strip.size == (128, 64)
    assert "[!]" not in "\n".join(e.description or "" for e in payload.embeds)


def test_warning_icons_use_one_meaningful_embed_without_displacing_transit():
    from dashboard.models import WeatherConditions, WeatherWarning
    from dashboard.render import build_payload

    weather = WeatherConditions(warnings=[
        WeatherWarning(
            f"TC{i}", f"Warning {i}", icon_url=f"https://hko.example/{i}.gif",
            icon_data=_icon_png((i * 20, 0, 0, 255)),
        )
        for i in range(8)
    ])
    payload = build_payload(weather, s.route_groups(), [], [], None, None)
    thumbnails = [e.thumbnail.url for e in payload.embeds if e.thumbnail]
    assert thumbnails == ["attachment://hko-warnings.png"]
    assert sum("Warning " in (e.description or "") for e in payload.embeds) == 1
    assert any(e.title == "🚌 Bus stops" for e in payload.embeds)


def test_weather_embed_title_reading_and_issued_timestamps():
    """The reading (🌡️/🌧️/💧) is the embed title; each warning carries an
    'issued <t:>' timestamp; the provider timestamp remains native metadata."""
    from dashboard.models import WeatherConditions, WeatherWarning
    from dashboard.render import _build_weather_embed

    now = s.utc()
    warnings = [
        WeatherWarning(code="TC8NE", name="Gale Signal No. 8 NE", issued_at=now),
        WeatherWarning(code="WRAINB", name="Black Rainstorm", issued_at=now),
    ]
    embed = _build_weather_embed(
        WeatherConditions(warnings=warnings, snapshot=s.weather_snapshot())
    )
    assert embed is not None
    assert embed.title == "🌡️ 28°C · 🌧️ 0.0mm · 💧 71%"
    assert "Sai Kung" not in embed.description  # source line moved to the footer
    # each warning has an issued timestamp (Discord <t:> renders locally)
    assert "issued <t:" in embed.description
    assert embed.description.count("issued <t:") == 2
    assert embed.description.count(":R>") == 2
    assert ":f>" not in embed.description
    assert ":t>" not in embed.description
    assert (
        "🔗 [HKO warnings](https://www.hko.gov.hk/en/wxinfo/dailywx/wxwarntoday.htm)"
        in embed.description
    )
    assert embed.timestamp is not None  # native footer timestamp field
    assert embed.footer.text == "HKO"


def test_weather_embed_title_fallback_without_snapshot():
    """No snapshot -> generic weather title."""
    from dashboard.models import WeatherConditions, WeatherWarning
    from dashboard.render import _build_weather_embed

    embed = _build_weather_embed(
        WeatherConditions(warnings=[WeatherWarning(code="TC", name="Typhoon")])
    )
    assert embed is not None
    assert embed.title == "🌦️ Weather"


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
        traffic_map_webp=b"RIFF" + b"\x00" * 100,
        transit_source_time=s.utc(),
        map_source_time=s.utc(),
    )
    assert len(payload.embeds) <= EMBEDS_PER_MESSAGE_MAX
    assert len(payload.files) <= EMBEDS_PER_MESSAGE_MAX
    for embed in payload.embeds:
        assert len(embed.fields) <= FIELDS_PER_EMBED_MAX
        for field in embed.fields:
            assert len(field.value) <= FIELD_VALUE_MAX
        assert len(embed.description or "") <= DESC_MAX
        assert len(embed) <= CHARS_PER_EMBED_MAX
        assert embed.timestamp is not None
        assert embed.footer.text
        assert "source time" not in embed.footer.text


def test_finalize_embed_enforces_field_cap():
    embed = discord.Embed()
    for i in range(30):
        embed.add_field(name=f"f{i}", value="x")
    embed = finalize_embed(embed)
    assert len(embed.fields) <= FIELDS_PER_EMBED_MAX
