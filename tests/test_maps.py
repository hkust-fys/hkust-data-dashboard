"""Google base-map capture and retained bus/stop marker tests."""

from __future__ import annotations

import base64
import io
import math
import sys
import types

from PIL import Image

from dashboard.maps import renderer, tiles
from dashboard.models import EtaKind, EtaRow, Operator, RouteEtaGroup
from dashboard.providers.route_geometry import RouteLine, Stop


def _png_data_url(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def _detailed_image(size: tuple[int, int], alpha: int = 255) -> Image.Image:
    image = Image.new("RGBA", size)
    pixels = image.load()
    for y in range(size[1]):
        for x in range(size[0]):
            pixels[x, y] = ((x * 17) % 256, (y * 29) % 256, ((x + y) * 13) % 256, alpha)
    return image


def test_canvas_export_is_normalized_to_projection_viewport():
    source = _detailed_image((40, 20)).convert("RGB")
    exported = tiles._decode_canvas_export(_png_data_url(source), (40, 20))
    assert exported.size == (40, 20)
    assert exported.mode == "RGB"
    assert exported.getpixel((20, 10)) == source.getpixel((20, 10))


def test_canvas_alpha_is_composited_over_neutral_and_incomplete_is_rejected():
    translucent = _detailed_image((40, 20), alpha=242)
    exported = tiles._decode_canvas_export(_png_data_url(translucent), (40, 20))
    assert exported.getpixel((10, 5)) != translucent.getpixel((10, 5))[:3]

    incomplete = Image.new("RGBA", (20, 10), (10, 20, 30, 0))
    try:
        tiles._decode_canvas_export(_png_data_url(incomplete), (20, 10))
    except ValueError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("materially incomplete canvas export was accepted")


def test_canvas_export_rejects_non_png_data_url():
    try:
        tiles._decode_canvas_export("data:text/plain;base64,Zm9v", (40, 20))
    except ValueError as exc:
        assert "PNG data URL" in str(exc)
    else:
        raise AssertionError("non-PNG canvas export was accepted")


def test_canvas_capture_contract_uses_geometry_without_dom_class_selectors():
    script = tiles.CANVAS_EXPORT_SCRIPT
    assert "normalized.toDataURL('image/png')" in script
    assert "document.querySelectorAll('canvas')" in script
    assert "getBoundingClientRect" in script
    assert "sourceScaleX = canvas.width / rect.width" in script
    assert "normalized.width = viewportWidth" in script
    assert "context.drawImage" in script
    assert "gm-style" not in script


def test_largest_successful_png_wins_among_multiple_canvas_candidates():
    small_png_large_canvas = tiles._canvas_candidate_rank(2_000, 2_000_000, 4_000_000)
    large_png_smaller_canvas = tiles._canvas_candidate_rank(8_000, 1_900_000, 2_000_000)
    assert max(small_png_large_canvas, large_png_smaller_canvas) == large_png_smaller_canvas

    script = tiles.CANVAS_EXPORT_SCRIPT
    assert "right.exportLength - left.exportLength" in script
    assert script.index("right.exportLength - left.exportLength") < script.index(
        "right.visibleArea - left.visibleArea"
    )
    assert "candidates.map(candidate => candidate.dataUrl)" in script


def test_first_valid_canvas_skips_larger_invalid_candidate():
    invalid = _png_data_url(Image.new("RGBA", (40, 20), (0, 0, 0, 0)))
    detailed = _detailed_image((40, 20)).convert("RGB")
    valid = _png_data_url(detailed)
    selected = tiles._decode_first_valid_canvas([invalid, valid], (40, 20))
    assert selected.getpixel((20, 10)) == detailed.getpixel((20, 10))


def test_canvas_export_rejects_opaque_tile_loading_grid():
    grid = Image.new("RGB", (512, 256), (244, 243, 240))
    draw = renderer.ImageDraw.Draw(grid)
    for x in range(0, grid.width, 128):
        draw.line((x, 0, x, grid.height), fill=(255, 255, 255), width=1)
    for y in range(0, grid.height, 128):
        draw.line((0, y, grid.width, y), fill=(255, 255, 255), width=1)
    try:
        tiles._decode_canvas_export(_png_data_url(grid), grid.size)
    except ValueError as exc:
        assert "loading placeholder" in str(exc)
    else:
        raise AssertionError("opaque tile loading grid was accepted")


async def test_invalid_black_cache_is_not_reused(tmp_path, monkeypatch):
    cache_path = tmp_path / "gmaps_base.png"
    Image.new("RGB", (40, 20), (0, 0, 0)).save(cache_path)

    fake_api = types.ModuleType("playwright.async_api")

    def unavailable():
        raise RuntimeError("browser unavailable")

    fake_api.async_playwright = unavailable
    fake_package = types.ModuleType("playwright")
    fake_package.async_api = fake_api
    monkeypatch.setitem(sys.modules, "playwright", fake_package)
    monkeypatch.setitem(sys.modules, "playwright.async_api", fake_api)

    result = await tiles.capture_gmaps_base(str(tmp_path), viewport=(40, 20))
    assert result.size == (40, 20)
    assert result.getpixel((20, 10)) == (240, 242, 245)


def test_fit_view_returns_fixed_base_map_bounds():
    center_lat, center_lon, zoom = renderer.fit_view()
    assert (center_lat, center_lon, zoom) == (
        renderer.BASE_MAP_LAT,
        renderer.BASE_MAP_LON,
        renderer.BASE_MAP_ZOOM,
    )


def test_bus_prediction_interpolates_on_matching_official_upstream_geometry():
    farther_upstream = Stop("F", "Farther upstream", 22.333360, 114.242881)
    upstream = Stop("U", "Upstream", 22.333360, 114.252881)
    gate = Stop("G", "H.K.U.S.T. SOUTH", 22.333360, 114.262881)
    destination = Stop("D", "Destination", 22.333360, 114.272881)
    # The road bends north between the upstream TD stops.  The estimate must
    # follow this routed shape, not the straight stop-to-stop chord.
    path = [
        (farther_upstream.lat, farther_upstream.lon),
        (22.343360, 114.242881),
        (upstream.lat, upstream.lon),
        (gate.lat, gate.lon),
        (destination.lat, destination.lon),
    ]
    lengths = [
        renderer._path_segment_length(a, b) for a, b in zip(path, path[1:], strict=False)
    ]
    offsets = [0.0, lengths[0] + lengths[1]]
    offsets.extend([offsets[-1] + lengths[2], offsets[-1] + lengths[2] + lengths[3]])
    lines = [
        RouteLine(
            "X", "KMB", "outbound", [farther_upstream, upstream, gate, destination], path, offsets
        )
    ]
    group = RouteEtaGroup(
        route="X",
        destination="Destination",
        gate="S",
        operator=Operator.KMB,
        rows=[EtaRow("X", "Destination", "S", Operator.KMB, 3, EtaKind.REALTIME)],
    )
    bus = renderer.predict_buses([group], lines)[0]
    # Three minutes at two minutes per official stop is halfway from the
    # previous TD stop to the one before it, rather than a snapped stop.
    assert bus[1] > upstream.lat  # on the northward road bend
    assert not math.isclose(bus[2], (farther_upstream.lon + upstream.lon) / 2, abs_tol=2e-5)
    assert not math.isclose(bus[-1], 0.0, abs_tol=0.05)


def test_bus_prediction_has_no_straight_chord_fallback_without_osm_path():
    upstream = Stop("U", "Upstream", 22.333360, 114.252881)
    gate = Stop("G", "H.K.U.S.T. SOUTH", 22.333360, 114.262881)
    destination = Stop("D", "Destination", 22.333360, 114.272881)
    line = RouteLine("X", "KMB", "outbound", [upstream, gate, destination])
    group = RouteEtaGroup(
        route="X", destination="Destination", gate="S", operator=Operator.KMB,
        rows=[EtaRow("X", "Destination", "S", Operator.KMB, 1, EtaKind.REALTIME)],
    )
    assert renderer.predict_buses([group], [line]) == []


def test_interpolated_bus_arrow_uses_the_local_curved_road_tangent():
    # Eastbound road turns north: a bus after the bend must not keep the
    # route-wide eastbound heading.
    path = [(22.3300, 114.2600), (22.3300, 114.2610), (22.3310, 114.2610)]
    first = renderer._path_segment_length(path[0], path[1])
    located = renderer._point_at_path_offset(path, first + 10)
    assert located is not None
    _lat, _lon, heading = located
    assert math.isclose(heading, math.pi / 2, abs_tol=0.01)
    arrow = renderer._bus_direction_arrow_triangle((100, 100), heading)
    assert arrow[0][1] < 100  # screen-space arrow points north after the bend

    reverse_path = list(reversed(path))
    reverse = renderer._point_at_path_offset(reverse_path, 10)
    assert reverse is not None
    assert math.isclose(reverse[2], -math.pi / 2, abs_tol=0.01)
    reverse_arrow = renderer._bus_direction_arrow_triangle((100, 100), reverse[2])
    assert reverse_arrow[0][1] > 100


def test_bus_prediction_never_renders_at_official_route_termini():
    upstream = Stop("U", "Upstream", 22.333360, 114.252881)
    gate = Stop("G", "H.K.U.S.T. SOUTH", 22.333360, 114.262881)
    destination = Stop("D", "Destination", 22.333360, 114.272881)
    path = [(s.lat, s.lon) for s in (upstream, gate, destination)]
    first = renderer._path_segment_length(path[0], path[1])
    lines = [RouteLine("X", "KMB", "outbound", [upstream, gate, destination], path, [0, first, first * 2])]
    group = RouteEtaGroup(
        route="X", destination="Destination", gate="S", operator=Operator.KMB,
        rows=[EtaRow("X", "Destination", "S", Operator.KMB, 2, EtaKind.REALTIME)],
    )
    # Two minutes would put this estimate precisely on the upstream terminus.
    assert renderer.predict_buses([group], lines) == []


def test_public_stop_markers_are_offset_on_opposite_road_sides(tmp_path, monkeypatch):
    lines = [
        RouteLine(
            "X", "KMB", "outbound",
            [Stop("W", "West", 22.3340, 114.2290), Stop("C", "Centre", 22.3340, 114.2300)],
        ),
        RouteLine(
            "X", "KMB", "inbound",
            [Stop("C", "Centre", 22.3340, 114.2300), Stop("W", "West", 22.3340, 114.2290)],
        ),
    ]
    center = lines[0].stops[1]
    captured: list[tuple[float, float, float]] = []
    original = renderer._draw_marker_on_left

    def tracking(draw, x, y, heading, color, square):
        result = original(draw, x, y, heading, color, square)
        if square:
            captured.append((result[0], result[1], heading))
        return result

    monkeypatch.setattr(renderer, "_draw_marker_on_left", tracking)
    png = renderer.render_map([], str(tmp_path), [center], lines)
    image = Image.open(io.BytesIO(png))
    assert image.size == (renderer.MAP_WIDTH, renderer.MAP_HEIGHT)
    assert len(captured) == 2
    assert math.isclose(captured[0][0], captured[1][0], abs_tol=0.1)
    assert abs(captured[0][1] - captured[1][1]) >= 13.5
    assert math.isclose(renderer._angular_distance(captured[0][2], captured[1][2]), math.pi)


def test_stop_glyphs_are_eight_pixels():
    for square in (False, True):
        canvas = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
        renderer._draw_marker_on_left(
            renderer.ImageDraw.Draw(canvas), 20, 20, 0, renderer.PUBLIC_STOP_COLOR, square
        )
        left, top, right, bottom = canvas.getbbox()
        assert (right - left, bottom - top) == (8, 8)


def test_bus_estimate_is_a_colored_route_pill_with_embedded_white_direction_triangle():
    import inspect

    canvas = Image.new("RGBA", (80, 80), (120, 120, 120, 255))
    renderer._draw_bus_route_marker(
        renderer.ImageDraw.Draw(canvas),
        renderer.LabelPlacement("91", (30, 20, 55, 41), (20, 30), Operator.KMB, 0),
        renderer.OPERATOR_COLORS[Operator.KMB],
        renderer._font(13),
    )
    colors = {pixel[:3] for _count, pixel in canvas.getcolors(maxcolors=1_000_000)}
    assert (20, 20, 20) in colors  # compact dark label outline
    assert renderer.OPERATOR_COLORS[Operator.KMB] in colors
    assert (255, 255, 255) in colors  # route text and embedded direction triangle
    assert "draw.line" not in inspect.getsource(renderer._draw_bus_route_marker)
    assert not hasattr(renderer, "_draw_bus_marker")
    assert not hasattr(renderer, "_draw_bus_label_pointer")


def test_bus_direction_arrow_is_centred_in_the_pill_at_the_road_anchor():
    marker = renderer.BusMarker(("91",), 40, 40, Operator.KMB, 0.0)
    canvas = Image.new("RGBA", (120, 90), (120, 120, 120, 255))
    draw = renderer.ImageDraw.Draw(canvas)
    placement = renderer._layout_bus_labels([marker], draw, renderer._font(13), canvas.size)[0]
    triangle = renderer._bus_direction_arrow_triangle(placement.marker, placement.heading)
    centroid = (
        sum(point[0] for point in triangle) / 3,
        sum(point[1] for point in triangle) / 3,
    )
    assert centroid == placement.marker == (marker.x, marker.y)
    assert placement.rect[0] <= marker.x <= placement.rect[2]
    assert placement.rect[1] <= marker.y <= placement.rect[3]
    assert (placement.rect[1] + placement.rect[3]) / 2 == marker.y
    assert placement.rect[0] < marker.x < placement.rect[2]


def test_bus_label_layout_never_overlaps_is_in_bounds_and_deterministic():
    canvas = Image.new("RGBA", (360, 180), (255, 255, 255, 255))
    draw = renderer.ImageDraw.Draw(canvas)
    font = renderer._font(13)
    markers = [
        renderer.BusMarker((str(index),), 150 + index * 4, 80, Operator.KMB, 0)
        for index in range(5)
    ]
    first = renderer._layout_bus_labels(markers, draw, font, canvas.size)
    second = renderer._layout_bus_labels(markers, draw, font, canvas.size)
    assert first == second
    assert len(first) == 1
    assert first[0].text == "0/1/2/3/4"
    for index, placement in enumerate(first):
        left, top, right, bottom = placement.rect
        assert 0 <= left < right <= canvas.width
        assert 0 <= top < bottom <= canvas.height
        assert all(
            not renderer._rects_overlap(placement.rect, other.rect)
            for other in first[index + 1 :]
        )
        assert (top + bottom) / 2 == placement.marker[1]


def test_public_stops_merge_by_place_and_direction_but_keep_opposite_direction():
    west = Stop("W", "West", 22.3340, 114.2290)
    same_a = Stop("A", "Same A", 22.3340, 114.230000)
    same_b = Stop("B", "Same B", 22.3340, 114.230001)
    opposite = Stop("C", "Opposite", 22.3340, 114.230002)
    east = Stop("E", "East", 22.3340, 114.2310)
    lines = [
        RouteLine("1", "KMB", "out", [west, same_a, east]),
        RouteLine("2", "KMB", "out", [west, same_b, east]),
        RouteLine("3", "KMB", "in", [east, opposite, west]),
    ]
    paths = [[(stop.lat, stop.lon) for stop in line.stops] for line in lines]
    markers = renderer._merged_public_stop_markers(
        [same_a, same_b, opposite], lines, paths
    )
    assert len(markers) == 2
    assert math.isclose(renderer._angular_distance(markers[0][2], markers[1][2]), math.pi)


def test_792m_and_kmb_same_direction_stops_merge_across_operator_offsets():
    west = Stop("W", "West", 22.3340, 114.2290)
    kmb = Stop("KMB", "Ngan Ying Road", 22.3340, 114.2300)
    ctb = Stop("CTB", "Ngan Ying Road", 22.33435, 114.2301)
    east = Stop("E", "East", 22.3340, 114.2310)
    lines = [
        RouteLine("91M", "KMB", "out", [west, kmb, east]),
        RouteLine("792M", "CTB", "out", [west, ctb, east]),
    ]
    paths = [[(stop.lat, stop.lon) for stop in line.stops] for line in lines]
    markers = renderer._merged_public_stop_markers([kmb, ctb], lines, paths)
    assert len(markers) == 1


def test_hang_hau_same_direction_kmb_ctb_gmb_merge_but_opposite_stays():
    west = Stop("W", "Hang Hau Road West", 22.31780, 114.26400)
    east = Stop("E", "Hang Hau Road East", 22.31780, 114.26800)
    kmb = Stop("K", "Hang Hau Road", 22.317800, 114.266000)
    ctb = Stop("C", "Hang Hau Rd Bus Stop", 22.317825, 114.266020)
    gmb = Stop("G", "Hang Hau Road", 22.317780, 114.265985)
    opposite = Stop("O", "Hang Hau Road", 22.317840, 114.266010)
    lines = [
        RouteLine("91M", "KMB", "out", [west, kmb, east]),
        RouteLine("792M", "CTB", "out", [west, ctb, east]),
        RouteLine("11", "GMB", "seq-2", [west, gmb, east]),
        RouteLine("91M", "KMB", "in", [east, opposite, west]),
    ]
    paths = [[(s.lat, s.lon) for s in line.stops] for line in lines]
    markers = renderer._merged_public_stop_markers([kmb, ctb, gmb, opposite], lines, paths)
    assert len(markers) == 2
    assert math.isclose(renderer._angular_distance(markers[0][2], markers[1][2]), math.pi)


def test_gmb_stops_do_not_emit_public_glyphs_but_keep_geometry_for_eta():
    upstream = Stop("G-U", "Minibus upstream", 22.333360, 114.252881)
    gate = Stop("G-G", "H.K.U.S.T. SOUTH", 22.333360, 114.262881)
    destination = Stop("G-D", "Minibus destination", 22.333360, 114.272881)
    path = [(stop.lat, stop.lon) for stop in (upstream, gate, destination)]
    first = renderer._path_segment_length(path[0], path[1])
    line = RouteLine("11", "GMB", "seq-1", [upstream, gate, destination], path, [0, first, first * 2])

    # The provider's GMB geometry remains available to ETA interpolation.
    group = RouteEtaGroup(
        route="11", destination="Minibus destination", gate="S", operator=Operator.GMB,
        rows=[EtaRow("11", "Minibus destination", "S", Operator.GMB, 1, EtaKind.REALTIME)],
    )
    assert renderer.predict_buses([group], [line])
    assert renderer._merged_public_stop_markers([gate], [line], [path]) == []


def test_bus_markers_merge_matching_route_operator_at_same_position():
    predictions = [
        ("91", 22.334, 114.230, Operator.KMB, 0.0),
        ("91", 22.334, 114.230, Operator.KMB, 0.0),
        ("91M", 22.334, 114.230, Operator.KMB, 0.0),
    ]
    markers = renderer._merge_bus_markers(
        predictions, renderer.BASE_MAP_LAT, renderer.BASE_MAP_LON,
        renderer.BASE_MAP_ZOOM, (renderer.MAP_WIDTH, renderer.MAP_HEIGHT)
    )
    assert len(markers) == 2
    assert [marker.routes for marker in markers] == [("91",), ("91M",)]


def test_off_map_bus_prediction_has_no_marker_or_label():
    markers = renderer._merge_bus_markers(
        [("91", 90.0, 0.0, Operator.KMB, 0.0)],
        renderer.BASE_MAP_LAT,
        renderer.BASE_MAP_LON,
        renderer.BASE_MAP_ZOOM,
        (renderer.MAP_WIDTH, renderer.MAP_HEIGHT),
    )
    assert markers == []


def test_legend_has_no_obsolete_google_traffic_or_direction_explanation():
    import inspect

    source = inspect.getsource(renderer._draw_legend)
    assert "Estimated buses (not GPS)" in source
    assert "Live traffic speed" not in source
    assert "traffic jam" not in source
    assert "Arrows show both travel directions" not in source
