from dataclasses import replace

import pytest

from dashboard.models import TrafficIncident
from dashboard.providers import traffic_location as location
from dashboard.providers.route_geometry import RouteLine
from dashboard.providers.tracked_roads import TrackedRoads


def incident(description="Traffic is busy on Lung Cheung Road."):
    return TrafficIncident("one", "", description, "Lung Cheung Road", "", "", "NEW")


def line(route, start, end):
    return RouteLine(route, "KMB", "outbound", path=[(22.34, start), (22.34, end)])


ROAD = [[(22.34, 114.10), (22.34, 114.12)]]


@pytest.mark.parametrize("near", ["", "Unknown Junction"])
def test_unenriched_news_cannot_expand_into_whole_road_rail(near):
    from dashboard.pipeline import map_road_paths_from_results

    roads = TrackedRoads(
        display_names={"lung cheung road": "Lung Cheung Road"},
        paths={"lung cheung road": tuple(tuple(path) for path in ROAD)},
    )
    news = replace(incident(), near_landmark=near)
    assert map_road_paths_from_results(([], [news], []), roads)[0] == []


def test_coverage_is_fraction_of_whole_road_not_bus_route_length():
    assert location.route_coverage(ROAD, line("91", 114.11, 114.12)) == pytest.approx(0.5, abs=0.01)
    assert location.route_coverage(ROAD, line("91M", 114.119, 114.15)) < 0.1
    assert location.route_coverage(ROAD, line("291P", 114.09, 114.13)) == pytest.approx(1)


def test_crossing_a_road_is_not_travelling_along_it():
    crossing = RouteLine("91", "KMB", "outbound", path=[(22.339, 114.11), (22.341, 114.11)])
    assert location.route_coverage(ROAD, crossing) == 0


def test_direction_rejects_opposite_bus_even_on_same_geometry():
    assert location.route_coverage(ROAD, line("91", 114.10, 114.12), "east") == 1
    assert location.route_coverage(ROAD, line("91M", 114.12, 114.10), "east") == 0
    assert location.route_coverage(ROAD, line("91", 114.10, 114.12), "unknown place") == 0


def test_near_section_does_not_include_distant_part_of_same_road():
    section = location.locate_section(ROAD, ((22.34, 114.101),))
    assert section
    assert location.route_coverage(section, line("91", 114.11, 114.12)) == 0
    assert location.route_coverage(section, line("291P", 114.10, 114.11)) == 1


def test_between_anchors_must_be_on_same_continuous_road():
    assert location.locate_section(ROAD, ((22.34, 114.105), (22.34, 114.115)))
    assert not location.locate_section(ROAD, ((22.34, 114.105), (22.4, 114.115)))


def test_landmark_extraction_english_and_chinese():
    en = incident("The slow lane of Lung Cheung Road (Tsuen Wan bound) near Wong Tai Sin MTR Station which was closed is re-opened.")
    zh = incident("龍翔道往荃灣方向，近黃大仙港鐵站慢線的交通意外已清理。")
    assert location.location_terms(en) == (("Wong Tai Sin MTR Station",), "Tsuen Wan")
    assert location.location_terms(zh) == (("黃大仙港鐵站",), "荃灣")
    assert location.location_terms(incident("龍翔道西行交通繁忙。")) == ((), "西行")


def test_place_search_requires_exact_name_and_rejects_distant_namesakes():
    row = {"nameEN": "MTR Wong Tai Sin Station", "nameZH": "黃大仙站", "latitude": 22.34, "longitude": 114.19}
    assert location.exact_place_point("Wong Tai Sin MTR Station", [row]) == (22.34, 114.19)
    assert location.exact_place_point("Wong Tai Sin", [row]) is None
    assert location.exact_place_point("Wong Tai Sin MTR Station", [row, {**row, "latitude": 22.5}]) is None


@pytest.mark.asyncio
async def test_road_only_keeps_news_and_lists_only_representative_bus_paths(monkeypatch):
    async def whole(*args):
        return ROAD
    monkeypatch.setattr(location, "_whole_road", whole)
    roads = TrackedRoads(display_names={"lung cheung road": "Lung Cheung Road"})
    news = incident()
    result = await location.enrich_incident_locations(None, [news], roads, [
        line("91", 114.1099, 114.12), line("91M", 114.119, 114.12),
    ])
    assert len(result) == 1
    assert result[0].description == news.description
    assert result[0].affected_routes == ("91",)
    assert result[0].location_resolution == "road_coverage"
    assert not result[0].affected_paths  # an unspecified jam is not a whole-road rail


@pytest.mark.asyncio
async def test_specific_location_overrides_representativeness(monkeypatch):
    async def whole(*args):
        return ROAD
    async def place(*args):
        return (22.34, 114.101)
    monkeypatch.setattr(location, "_whole_road", whole)
    monkeypatch.setattr(location, "_place", place)
    roads = TrackedRoads(display_names={"lung cheung road": "Lung Cheung Road"})
    news = incident("Traffic is busy on Lung Cheung Road near Named Station.")
    result = await location.enrich_incident_locations(None, [news], roads, [
        line("91", 114.105, 114.12), line("291P", 114.10, 114.105),
    ])
    assert result[0].affected_routes == ("291P",)
    assert result[0].location_resolution == "section"


@pytest.mark.asyncio
async def test_unresolved_explicit_landmark_does_not_fall_back_to_whole_road(monkeypatch):
    async def missing(*args):
        return None
    monkeypatch.setattr(location, "_place", missing)
    roads = TrackedRoads(display_names={"lung cheung road": "Lung Cheung Road"})
    news = incident("Traffic is busy on Lung Cheung Road near Ambiguous Place.")
    result = await location.enrich_incident_locations(None, [news], roads, [line("91", 114.10, 114.12)])
    assert result[0] == replace(news, location_resolution="unresolved")


@pytest.mark.asyncio
async def test_coordinate_location_survives_unresolvable_place_name(monkeypatch):
    async def whole(*args):
        return ROAD
    monkeypatch.setattr(location, "_whole_road", whole)
    roads = TrackedRoads(display_names={"lung cheung road": "Lung Cheung Road"})
    news = replace(incident("Traffic near Ambiguous Place."), latitude=22.34, longitude=114.101)
    result = await location.enrich_incident_locations(None, [news], roads, [line("91", 114.10, 114.12)])
    assert result[0].affected_routes == ("91",)
    assert result[0].location_resolution == "section"


@pytest.mark.asyncio
async def test_news_bus_destinations_share_map_shorthand(monkeypatch):
    from dashboard.maps import _destination_map

    async def whole(*args):
        return ROAD
    monkeypatch.setattr(location, "_whole_road", whole)
    roads = TrackedRoads(display_names={"lung cheung road": "Lung Cheung Road"})
    bus = line("91M", 114.10, 114.12)
    bus.destination = "DIAMOND HILL STATION BUS TERMINUS"
    result = await location.enrich_incident_locations(None, [incident()], roads, [bus])
    destination = _destination_map([], [bus])[("KMB", "91M", "outbound")]
    assert destination == "Diamond Hill"
    assert result[0].affected_routes == (f"91M → {destination}",)


@pytest.mark.asyncio
async def test_disk_cached_geometry_still_reaches_the_map(monkeypatch):
    import json

    from dashboard.pipeline import map_road_paths_from_results

    async def whole(*args):
        # The disk JSON round trip changes coordinate tuples into lists.
        return json.loads(json.dumps([[(22.34, 114.10), (22.34, 114.101), (22.34, 114.102)]]))
    monkeypatch.setattr(location, "_whole_road", whole)
    roads = TrackedRoads(display_names={"lung cheung road": "Lung Cheung Road"})
    news = replace(incident(), latitude=22.34, longitude=114.101)
    reports = await location.enrich_incident_locations(None, [news], roads, [line("91", 114.10, 114.12)])
    result, _important = map_road_paths_from_results(([], reports, []), roads)
    assert result and any((22.34, 114.101) in path for path in result)
    assert all(isinstance(point, tuple) for path in reports[0].affected_paths for point in path)
