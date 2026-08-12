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


def test_canvas_export_is_normalized_to_projection_viewport():
    source = Image.new("RGB", (20, 10), (12, 34, 56))
    exported = tiles._decode_canvas_export(_png_data_url(source), (40, 20))
    assert exported.size == (40, 20)
    assert exported.mode == "RGB"
    assert exported.getpixel((20, 10)) == (12, 34, 56)


def test_canvas_alpha_is_composited_over_neutral_and_incomplete_is_rejected():
    translucent = Image.new("RGBA", (20, 10), (0, 0, 0, 242))
    exported = tiles._decode_canvas_export(_png_data_url(translucent), (20, 10))
    assert exported.getpixel((10, 5)) != (0, 0, 0)

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
    valid = _png_data_url(Image.new("RGB", (40, 20), (12, 34, 56)))
    selected = tiles._decode_first_valid_canvas([invalid, valid], (40, 20))
    assert selected.getpixel((20, 10)) == (12, 34, 56)


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


def test_bus_prediction_uses_matching_official_upstream_stop_and_heading():
    upstream = Stop("U", "Upstream", 22.333360, 114.252881)
    gate = Stop("G", "H.K.U.S.T. SOUTH", 22.333360, 114.262881)
    destination = Stop("D", "Destination", 22.333360, 114.272881)
    lines = [RouteLine("X", "KMB", "outbound", [upstream, gate, destination])]
    group = RouteEtaGroup(
        route="X",
        destination="Destination",
        gate="S",
        operator=Operator.KMB,
        rows=[EtaRow("X", "Destination", "S", Operator.KMB, 3, EtaKind.REALTIME)],
    )
    bus = renderer.predict_buses([group], lines)[0]
    assert (bus[1], bus[2]) == (upstream.lat, upstream.lon)
    assert math.isclose(bus[-1], 0.0, abs_tol=1e-9)


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


def test_bus_marker_has_high_contrast_bordered_effect():
    canvas = Image.new("RGBA", (80, 80), (120, 120, 120, 255))
    renderer._draw_bus_marker(
        renderer.ImageDraw.Draw(canvas), 40, 40, 0, renderer.OPERATOR_COLORS[Operator.KMB]
    )
    colors = {pixel[:3] for _count, pixel in canvas.getcolors(maxcolors=1_000_000)}
    assert (255, 255, 255) in colors
    assert any(max(color) < 30 for color in colors)
    assert (210, 239, 250) in colors  # windows
    assert renderer.OPERATOR_COLORS[Operator.KMB] in colors


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
    assert len(first) == len(markers)
    for index, placement in enumerate(first):
        left, top, right, bottom = placement.rect
        assert 0 <= left < right <= canvas.width
        assert 0 <= top < bottom <= canvas.height
        assert all(
            not renderer._rects_overlap(
                placement.rect,
                (marker.x - 17, marker.y - 13, marker.x + 17, marker.y + 13),
            )
            for marker in markers
        )
        assert all(
            not renderer._rects_overlap(placement.rect, other.rect)
            for other in first[index + 1 :]
        )


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


def test_legend_has_no_obsolete_google_traffic_or_direction_explanation():
    import inspect

    source = inspect.getsource(renderer._draw_legend)
    assert "Estimated buses (not GPS)" in source
    assert "Live traffic speed" not in source
    assert "traffic jam" not in source
    assert "Arrows show both travel directions" not in source
