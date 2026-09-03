"""Tracked-roads provider tests: OSM parsing, table building, fallback."""

import asyncio
import json
import math
from collections import namedtuple

import aiohttp
import pytest

from dashboard.maps.positions import BusEstimate  # noqa: F401  (import sanity)
from dashboard.providers import tracked_roads
from dashboard.providers.route_geometry import RouteLine, Stop
from dashboard.providers.tracked_roads import (
    FALLBACK_ROADS,
    TrackedRoads,
    build_overpass_query,
    build_tracked_roads,
    collect_way_roads,
    fallback_roads,
    parse_overpass_roads,
    replace_fetched_at,
    roads_for_line,
)


def test_parse_overpass_roads_prefers_english_names_and_skips_paths():
    raw = {
        "elements": [
            {"tags": {"highway": "primary", "name": "龍翔道", "name:en": "Lung Cheung Road"}},
            {"tags": {"highway": "footway", "name": "Garden Path"}},
            {"tags": {"highway": "secondary", "name": "Pik Sha Road"}},
            {"tags": {"highway": "secondary", "name": "pik sha road"}},  # dup
            {"tags": {"highway": "service"}},
        ]
    }
    assert parse_overpass_roads(raw) == ["Lung Cheung Road", "Pik Sha Road"]


def test_build_overpass_query_unions_all_samples():
    query = build_overpass_query([(22.3, 114.2), (22.31, 114.21)])
    assert query.count("way(around:30,") == 2
    assert query.startswith("[out:json]")
    assert ");\nout geom;" in query


def _line(route, roads):
    stops = [Stop(f"{route}-{i}", name, 22.33 + i * 0.001, 114.26) for i, name in enumerate(roads)]
    return RouteLine(route, "KMB", "outbound", stops)


def test_build_tracked_roads_inverts_to_route_lists():
    lines = [_line("91", ["Clear Water Bay Road", "Lung Cheung Road"]), _line("11", ["Clear Water Bay Road"])]
    roads = build_tracked_roads(
        lines,
        [
            ["Clear Water Bay Road", "Lung Cheung Road"],
            ["Clear Water Bay Road"],
        ],
    )
    assert roads.source == "osm"
    assert set(roads.display_names) == {
        "clear water bay road",
        "lung cheung road",
    }
    assert roads.road_routes["clear water bay road"] == ("11", "91")
    assert roads.road_routes["lung cheung road"] == ("91",)
    # numeric-then-alpha route ordering
    assert list(roads.road_routes["clear water bay road"]) == sorted(
        roads.road_routes["clear water bay road"]
    )


def test_build_tracked_roads_keeps_only_matched_way_paths_and_crops_near_anchor():
    lines = [_line("91", ["Clear Water Bay Road"])]
    roads = build_tracked_roads(
        lines,
        [["Clear Water Bay Road"]],
        [
            {"name": "Clear Water Bay Road", "points": [(22.3270, 114.2300), (22.3270, 114.2400)]},
            {"name": "Untracked Road", "points": [(22.3270, 114.2300), (22.3270, 114.2400)]},
        ],
    )
    assert set(roads.paths) == {"clear water bay road"}
    assert len(roads.paths["clear water bay road"]) == 1
    segments = roads.segments_near(["clear water bay road"], 22.3270, 114.2350)
    assert len(segments) == 1
    assert len(segments[0]) >= 2
    assert roads.segments_near(["clear water bay road"], 22.3400, 114.2350) == []


def test_coordinate_less_segments_allow_short_roads_and_log_info(caplog):
    roads = TrackedRoads(
        display_names={"short road": "Short Road"}, aliases={"short road": "short road"},
        paths={"short road": (((22.3, 114.2), (22.3, 114.205)),)},
    )
    with caplog.at_level("INFO"):
        segments = roads.segments_near(["short road"], None, None)
    assert segments == [[(22.3, 114.2), (22.3, 114.205)]]
    assert "Short Road" in caplog.text
    assert "coordinate-less" in caplog.text
    assert any(record.levelname == "INFO" for record in caplog.records)


def test_coordinate_less_segments_reject_long_roads():
    roads = TrackedRoads(
        display_names={"long road": "Long Road"}, aliases={"long road": "long road"},
        paths={"long road": (((22.3, 114.2), (22.3, 114.22)),)},
    )
    assert roads.segments_near(["long road"], None, None) == []


def test_match_is_case_insensitive_and_apostrophe_tolerant():
    roads = fallback_roads()
    assert roads.match("Accident on CLEAR WATER BAY ROAD") == ["clear water bay road"]
    assert roads.match("Hiram\u2019s Highway closure") == ["hiram's highway"]
    assert roads.match("Nathan Road works") == []


def test_routes_for_text_unions_matched_roads():
    lines = [_line("91M", ["Hang Hau Road"]), _line("792M", ["Hang Hau Road"])]
    roads = build_tracked_roads(lines, [["hang hau road"], ["hang hau road"]])
    # numeric-then-alpha ordering: 91M sorts before 792M
    assert roads.routes_for_text("Roadworks on Hang Hau Road near TKO") == ["91M", "792M"]
    assert roads.routes_for_text("unrelated incident") == []


def test_fallback_seed_contains_the_core_corridors():
    names = {name.lower() for name in FALLBACK_ROADS}
    for expected in (
        "clear water bay road",
        "new clear water bay road",
        "lung cheung road",
        "hiram's highway",
        "sai kung road",
        "tai po tsai road",
        "university road",
        "ngan ying road",
        "ying yip road",
        "hang hau road",
        "wan po road",
        "po ning road",
        "po shun road",
        "tseung kwan o tunnel road",
        "po lam road",
        "chun ying street",
    ):
        assert expected in names
    seed = fallback_roads()
    assert isinstance(seed, TrackedRoads)
    assert seed.source == "fallback"


def test_collect_way_roads_keeps_named_highways_with_geometry():
    raw = {
        "elements": [
            {
                "type": "way",
                "tags": {"highway": "primary", "name": "龍翔道", "name:en": "Lung Cheung Road"},
                "geometry": [
                    {"lat": 22.33, "lon": 114.20},
                    {"lat": 22.33, "lon": 114.21},
                ],
            },
            {
                "type": "way",
                "tags": {"highway": "footway", "name": "Garden Path"},
                "geometry": [
                    {"lat": 22.33, "lon": 114.20},
                    {"lat": 22.33, "lon": 114.21},
                ],
            },
            {
                "type": "way",
                "tags": {"highway": "secondary", "name": "No Geometry"},
            },
            {"type": "node", "tags": {"name": "Just A Node"}},
        ]
    }
    ways = collect_way_roads(raw)
    assert len(ways) == 1
    assert ways[0]["name_en"] == "Lung Cheung Road"
    assert len(ways[0]["points"]) == 2


def test_roads_for_line_matches_nearby_geometry_only():
    ways = [
        {
            "name": "Clear Water Bay Road",
            "name_en": "Clear Water Bay Road",
            "points": [(22.3330, 114.2600), (22.3330, 114.2700)],
        },
        {
            "name": "Po Lam Road",
            "name_en": "Po Lam Road",
            "points": [(22.3100, 114.2500), (22.3100, 114.2600)],
        },
    ]
    line_points = [(22.3330, 114.2620), (22.3330, 114.2680)]
    assert roads_for_line(line_points, ways) == ["Clear Water Bay Road"]


def test_roads_for_line_requires_sustained_heading_aligned_overlap():
    line_points = [(22.3330, 114.2600), (22.3330, 114.2700)]
    ways = [
        {
            "name": "Aligned One-way Road",
            "name_en": "Aligned One-way Road",
            # Reverse coordinate order proves heading comparison is undirected.
            "points": [(22.3331, 114.2680), (22.3331, 114.2620)],
        },
        {
            "name": "Perpendicular Crossing",
            "name_en": "Perpendicular Crossing",
            "points": [(22.3300, 114.2650), (22.3360, 114.2650)],
        },
        {
            "name": "Single-hit Parallel Fragment",
            "name_en": "Single-hit Parallel Fragment",
            "points": [(22.3331, 114.2649), (22.3331, 114.2651)],
        },
    ]

    assert roads_for_line(line_points, ways) == ["Aligned One-way Road"]


def test_roads_for_line_accepts_only_complete_tightly_aligned_short_way():
    origin = (22.3330, 114.2600)

    def point(east_metres: float, north_metres: float = 0.0) -> tuple[float, float]:
        return (
            origin[0] + north_metres / 111_320.0,
            origin[1]
            + east_metres / (111_320.0 * math.cos(math.radians(origin[0]))),
        )

    line_points = [point(0), point(100)]
    shallow_east = 22.5 * math.cos(math.radians(15))
    shallow_north = 22.5 * math.sin(math.radians(15))
    ways = [
        {
            "name": "Complete Short Road",
            "name_en": "Complete Short Road",
            "points": [point(20, 2), point(65, 2)],  # complete 45 m alignment
        },
        {
            "name": "Nearby Parallel Road",
            "name_en": "Nearby Parallel Road",
            # Within the broad 30 m radius, but outside the strict short-way radius.
            "points": [point(20, 12), point(65, 12)],
        },
        {
            "name": "Tiny Incidental Fragment",
            "name_en": "Tiny Incidental Fragment",
            "points": [point(40), point(60)],
        },
        {
            "name": "Short Crossing",
            "name_en": "Short Crossing",
            "points": [point(50, -22.5), point(50, 22.5)],
        },
        {
            "name": "Shallow Short Crossing",
            "name_en": "Shallow Short Crossing",
            # Entirely within 8 m, but 15 degrees off the route heading.
            "points": [
                point(50 - shallow_east, -shallow_north),
                point(50 + shallow_east, shallow_north),
            ],
        },
        {
            "name": "Partial Short Road",
            "name_en": "Partial Short Road",
            "points": [point(80, 2), point(125, 2)],
        },
    ]

    assert roads_for_line(line_points, ways) == ["Complete Short Road"]


def test_replace_fetched_at_returns_stamped_copy():
    roads = fallback_roads()
    stamped = replace_fetched_at(roads, 1234.5)
    assert stamped.fetched_at == 1234.5
    assert roads.fetched_at == 0.0  # original untouched
    assert stamped.display_names == roads.display_names


def test_replace_fetched_at_preserves_paths():
    roads = TrackedRoads(
        display_names={"road": "Road"}, aliases={"road": "road"},
        paths={"road": (((22.3, 114.2), (22.31, 114.2)),)},
    )
    assert replace_fetched_at(roads, 1234.5).paths == roads.paths


def test_disk_cache_roundtrip_and_legacy_missing_paths(tmp_path):
    roads = TrackedRoads(
        display_names={"road": "Road"}, aliases={"road": "road"},
        road_routes={"road": ("91",)},
        paths={"road": (((22.3, 114.2), (22.31, 114.2)),)},
        source="osm", fetched_at=10.0,
    )
    tracked_roads._save_disk_cache(roads, str(tmp_path))
    loaded = tracked_roads._load_disk_cache(str(tmp_path))
    assert loaded is not None
    assert loaded.paths == roads.paths

    cache_path = tmp_path / "maps" / tracked_roads.ROADS_CACHE_NAME
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    raw.pop("paths")
    cache_path.write_text(json.dumps(raw), encoding="utf-8")
    legacy = tracked_roads._load_disk_cache(str(tmp_path))
    assert legacy is not None
    assert legacy.display_names == roads.display_names
    assert legacy.paths == {}

    raw["version"] = tracked_roads.ROADS_CACHE_VERSION - 1
    cache_path.write_text(json.dumps(raw), encoding="utf-8")
    assert tracked_roads._load_disk_cache(str(tmp_path)) is None


def test_build_tracked_roads_retains_all_distinct_fragments_and_anchors_nearest():
    lines = [_line("91", ["Clear Water Bay Road"])]
    first = ((22.3270, 114.2300), (22.3270, 114.2310))
    second = ((22.3270, 114.2400), (22.3270, 114.2410))
    roads = build_tracked_roads(
        lines, [["Clear Water Bay Road"]],
        [
            {"name": "Clear Water Bay Road", "points": list(first)},
            {"name": "Clear Water Bay Road", "points": list(second)},
            {"name": "Clear Water Bay Road", "points": list(first)},
        ],
    )
    assert roads.paths["clear water bay road"] == (first, second)
    segment = roads.segments_near(["clear water bay road"], 22.3270, 114.2405)
    assert segment == [list(second)]


def test_coordinate_less_segments_sum_all_fragments_before_short_fallback():
    roads = TrackedRoads(
        display_names={"road": "Road"}, aliases={"road": "road"},
        paths={
            "road": (
                ((22.3, 114.2), (22.3, 114.205)),
                ((22.3, 114.21), (22.3, 114.215)),
                ((22.3, 114.22), (22.3, 114.225)),
            )
        },
    )
    # Each fragment is short, but together exceed the conservative threshold.
    assert roads.segments_near(["road"]) == []


def test_partial_coordinate_does_not_use_coordinate_less_fallback():
    roads = TrackedRoads(
        display_names={"road": "Road"}, aliases={"road": "road"},
        paths={"road": (((22.3, 114.2), (22.3, 114.205)),)},
    )
    assert roads.segments_near(["road"], latitude=22.3) == []
    assert roads.segments_near(["road"], longitude=114.2) == []


_Key = namedtuple("_Key", "host port")


def _cert_error():
    return aiohttp.ClientConnectorCertificateError(
        _Key("overpass-api.de", 443), __import__("ssl").SSLError("cert verify failed")
    )


@pytest.mark.asyncio
async def test_overpass_skips_plain_http_when_no_tls_failure():
    requested: list[str] = []

    class FakeClient:
        async def post_form_json(self, url, data, timeout_seconds=None, attempts=None):
            requested.append(url)
            if url.startswith("https://"):
                raise aiohttp.ClientConnectionError("connection refused")
            raise AssertionError("plain-HTTP fallback must not be reached")

    with pytest.raises(aiohttp.ClientConnectionError):
        await tracked_roads._fetch_overpass(FakeClient(), "query")

    assert requested == [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]


@pytest.mark.asyncio
async def test_overpass_uses_plain_http_after_tls_failure_and_retries_it():
    calls: list[tuple[str, int]] = []

    class FakeClient:
        async def post_form_json(self, url, data, timeout_seconds=None, attempts=None):
            calls.append((url, attempts))
            if url.startswith("https://"):
                raise _cert_error()
            return {"n": len(calls)}

    result = await tracked_roads._fetch_overpass(FakeClient(), "query")

    assert result == {"n": 3}
    # Both HTTPS mirrors fail with a TLS certificate error, authorising the
    # plain-HTTP escape hatch, which requests a single retry (attempts=2).
    assert calls == [
        ("https://overpass-api.de/api/interpreter", 1),
        ("https://overpass.kumi.systems/api/interpreter", 1),
        ("http://overpass-api.de/api/interpreter", 2),
    ]


@pytest.mark.asyncio
async def test_cancelled_background_refresh_is_normal_shutdown():
    cache_key = "cancel-test"
    tracked_roads._refresh_retry_after.pop(cache_key, None)
    task = asyncio.create_task(asyncio.sleep(60))
    tracked_roads._refresh_tasks[cache_key] = task
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    tracked_roads._finish_refresh(task, cache_key)

    assert cache_key not in tracked_roads._refresh_tasks
    assert cache_key not in tracked_roads._refresh_retry_after


@pytest.mark.asyncio
async def test_shutdown_background_refreshes_drains_registry():
    cache_key = "shutdown-test"
    tracked_roads._refresh_shutdown = False
    tracked_roads._refresh_retry_after[cache_key] = 123.0
    task = asyncio.create_task(asyncio.sleep(60))
    tracked_roads._refresh_tasks[cache_key] = task

    await tracked_roads.shutdown_background_refreshes()

    assert task.cancelled()
    assert tracked_roads._refresh_tasks == {}
    assert tracked_roads._refresh_retry_after == {}
