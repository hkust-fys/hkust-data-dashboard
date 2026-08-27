"""Google base-map capture and retained bus/stop marker tests."""

from __future__ import annotations

import asyncio
import base64
import io
import math
import sys
import types

import pytest
from PIL import Image

from dashboard.maps import renderer, tiles
from dashboard.models import EtaRow, Operator, RouteEtaGroup
from dashboard.providers.route_geometry import RouteLine, Stop


def test_map_capture_and_projection_contract_stays_synchronized():
    assert tiles.VIEWPORT_WIDTH == renderer.MAP_WIDTH == 960
    assert tiles.VIEWPORT_HEIGHT == renderer.MAP_HEIGHT == 540
    assert renderer.BASE_MAP_ZOOM == 14.0
    assert ",14z/" in tiles.GMAPS_BASE_URL
    assert "!5m1!1e1" in tiles.GMAPS_BASE_URL


def test_custom_capture_cache_identity_includes_url_and_viewport():
    default = tiles.cache_filename(tiles.GMAPS_BASE_URL, (tiles.VIEWPORT_WIDTH, tiles.VIEWPORT_HEIGHT))
    custom_url = tiles.cache_filename("https://example.test/maps", (960, 540))
    custom_size = tiles.cache_filename(tiles.GMAPS_BASE_URL, (640, 360))
    assert default == tiles.BASE_CACHE_FILENAME
    assert custom_url != default and custom_size != default
    assert custom_url != custom_size
    assert custom_url.endswith(".png") and "960x540" in custom_url


def test_authoritative_gate_mapping_uses_verified_direction_variants():
    import dashboard.maps as maps

    def line(operator, route, bound, stop_id):
        return RouteLine(
            route, operator, bound,
            [Stop(stop_id, "gate", 22.33, 114.26), Stop("x", "x", 22.331, 114.261)],
        )

    groups = [
        RouteEtaGroup("91", "Diamond Hill", "S", Operator.KMB,
                      [EtaRow("91", "Diamond Hill", "S", Operator.KMB, 14)],
                      bound="outbound"),
        RouteEtaGroup("792M", "TKO", "N", Operator.CITYBUS,
                      [EtaRow("792M", "TKO", "N", Operator.CITYBUS, 8)],
                      bound="inbound"),
        RouteEtaGroup("11B", "Choi Hung", "S", Operator.GMB,
                      [EtaRow("11B", "Choi Hung", "S", Operator.GMB, 5)],
                      bound="seq-1"),
    ]
    rows = maps._authoritative_etas(
        groups,
        [
            line("KMB", "91", "outbound", "B002CEF0DBC568F5"),
            line("CTB", "792M", "inbound", "003130"),
            line("GMB", "11B", "seq-1", "20013011"),
        ],
    )
    assert [(row.operator, row.route, row.bound, row.index) for row in rows] == [
        ("KMB", "91", "outbound", 0),
        ("CTB", "792M", "inbound", 0),
        ("GMB", "11B", "seq-1", 0),
    ]


@pytest.mark.asyncio
async def test_probe_failure_still_renders_google_base(monkeypatch):
    import dashboard.maps as maps
    from dashboard.providers.route_geometry import RouteGeometry
    line = RouteLine(
        "91", "KMB", "outbound",
        [Stop("a", "A", 22.33, 114.26), Stop("b", "B", 22.331, 114.261)],
    )
    base = b"google-base"

    async def capture(*_args, **_kwargs):
        return base

    async def geometry(*_args, **_kwargs):
        return RouteGeometry(routes=[line])

    async def failed_probes(*_args, **_kwargs):
        raise RuntimeError("probe unavailable")

    captured: dict[str, object] = {}

    def render(*args, **kwargs):
        captured["base"] = args[4]
        return b"rendered"

    monkeypatch.setattr(maps, "capture_gmaps_base", capture)
    monkeypatch.setattr(maps, "fetch_route_geometry", geometry)
    monkeypatch.setattr(maps, "fetch_probe_etas", failed_probes)
    monkeypatch.setattr(maps, "render_map", render)

    png, _ = await maps.fetch_traffic_map(object())
    assert png == b"rendered"
    assert captured["base"] == base


@pytest.mark.asyncio
async def test_map_cancellation_cleans_up_capture_and_geometry(monkeypatch):
    import asyncio

    import dashboard.maps as maps
    capture_cancelled = asyncio.Event()
    geometry_cancelled = asyncio.Event()

    async def blocked_capture(**_kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            capture_cancelled.set()
            raise

    async def blocked_geometry(*_args, **_kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            geometry_cancelled.set()
            raise

    monkeypatch.setattr(maps, "capture_gmaps_base", blocked_capture)
    monkeypatch.setattr(maps, "fetch_route_geometry", blocked_geometry)

    operation = asyncio.create_task(maps.fetch_traffic_map(object()))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert capture_cancelled.is_set()
    assert geometry_cancelled.is_set()


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
    cache_path = tmp_path / "gmaps_base_z14_960x540.png"
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


@pytest.mark.asyncio
async def test_shared_browser_is_reused_and_shutdown_allows_reinit(monkeypatch):
    launches = []

    class Browser:
        def __init__(self):
            self.closed = False
        def is_connected(self):
            return not self.closed
        async def close(self):
            self.closed = True

    class Chromium:
        async def launch(self, **_kwargs):
            browser = Browser()
            launches.append(browser)
            return browser

    class Manager:
        chromium = Chromium()
        async def start(self):
            return self
        async def stop(self):
            pass

    fake_api = types.ModuleType("playwright.async_api")
    fake_api.async_playwright = lambda: Manager()
    monkeypatch.setitem(sys.modules, "playwright.async_api", fake_api)
    tiles._shared_browser = None
    tiles._playwright_manager = None

    assert await tiles._get_shared_browser() is await tiles._get_shared_browser()
    assert len(launches) == 1
    await tiles.shutdown_gmaps_browser()
    await tiles.shutdown_gmaps_browser()
    assert await tiles._get_shared_browser() is launches[1]
    assert len(launches) == 2
    await tiles.shutdown_gmaps_browser()


@pytest.mark.asyncio
async def test_disconnected_capture_failure_closes_shared_browser_without_deadlock(tmp_path):
    class Browser:
        def __init__(self):
            self.closed = False
        def is_connected(self):
            return not self.closed
        async def new_context(self, **_kwargs):
            self.closed = True
            raise RuntimeError("browser disconnected")
        async def close(self):
            self.closed = True

    class Manager:
        async def stop(self):
            pass

    browser = Browser()
    tiles._shared_browser = browser
    tiles._playwright_manager = Manager()
    tiles._browser_loop = asyncio.get_running_loop()
    tiles._capture_lock_loop = asyncio.get_running_loop()
    tiles._capture_retry_after = 0.0

    result = await asyncio.wait_for(
        tiles.capture_gmaps_base(str(tmp_path), viewport=(40, 20)), timeout=2.0
    )
    assert result.size == (40, 20)
    assert tiles._shared_browser is None
    assert tiles._playwright_manager is None
    await tiles.shutdown_gmaps_browser()


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


def test_public_stop_markers_are_offset_on_opposite_road_sides(tmp_path, monkeypatch):
    lines = [
        RouteLine(
            "X",
            "KMB",
            "outbound",
            [Stop("W", "West", 22.3340, 114.2290), Stop("C", "Centre", 22.3340, 114.2300)],
        ),
        RouteLine(
            "X",
            "KMB",
            "inbound",
            [Stop("C", "Centre", 22.3340, 114.2300), Stop("W", "West", 22.3340, 114.2290)],
        ),
    ]
    center = lines[0].stops[1]
    captured: list[tuple[float, float, float]] = []
    original = renderer._draw_marker_on_left

    def tracking(draw, x, y, heading, color, square, metrics=renderer.DEFAULT_METRICS):
        result = original(draw, x, y, heading, color, square, metrics)
        if square:
            captured.append((result[0], result[1], heading))
        return result

    monkeypatch.setattr(renderer, "_draw_marker_on_left", tracking)
    png = renderer.render_map([], str(tmp_path), [center], lines)
    image = Image.open(io.BytesIO(png))
    assert image.size == (renderer.MAP_WIDTH, renderer.MAP_HEIGHT + renderer.LEGEND_BAND_HEIGHT)
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


def test_bus_estimate_has_operator_colored_arrow_separate_from_label():
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
    assert canvas.getpixel((20, 30))[:3] == renderer.OPERATOR_COLORS[Operator.KMB]
    assert not hasattr(renderer, "_draw_bus_marker")
    assert not hasattr(renderer, "_draw_bus_label_pointer")


def test_mixed_operator_arrow_contains_each_operator_color():
    canvas = Image.new("RGBA", (80, 80), (120, 120, 120, 255))
    draw = renderer.ImageDraw.Draw(canvas)
    renderer._draw_colored_bus_arrow(draw, (40, 40), 0, [
        renderer.OPERATOR_COLORS[Operator.KMB], renderer.OPERATOR_COLORS[Operator.GMB]
    ])
    colors = {canvas.getpixel((x, y))[:3] for x in range(34, 47) for y in range(34, 47)}
    assert renderer.OPERATOR_COLORS[Operator.KMB] in colors
    assert renderer.OPERATOR_COLORS[Operator.GMB] in colors
    assert (25, 25, 25) in colors


def test_displaced_label_connector_runs_from_anchor_to_nearest_edge():
    class RecordingDraw:
        def __init__(self, wrapped):
            self.wrapped, self.line_calls = wrapped, []
        def line(self, xy, *args, **kwargs):
            self.line_calls.append((xy, kwargs.get("fill")))
            return self.wrapped.line(xy, *args, **kwargs)
        def __getattr__(self, name):
            return getattr(self.wrapped, name)
    canvas = Image.new("RGBA", (120, 80), (120, 120, 120, 255))
    draw = RecordingDraw(renderer.ImageDraw.Draw(canvas))
    placement = renderer.LabelPlacement("792M", (70, 20, 110, 38), (50, 30), Operator.GMB, 0)
    renderer._draw_bus_route_marker(draw, placement, renderer.OPERATOR_COLORS[Operator.GMB], renderer._font(13))
    assert draw.line_calls[0] == ((50.0, 30.0, 70.0, 30.0), renderer.OPERATOR_COLORS[Operator.GMB] + (220,))


def test_mixed_group_connector_contains_each_operator_color():
    class RecordingDraw:
        def __init__(self, wrapped):
            self.wrapped, self.line_calls = wrapped, []
        def line(self, xy, *args, **kwargs):
            self.line_calls.append(kwargs.get("fill"))
            return self.wrapped.line(xy, *args, **kwargs)
        def __getattr__(self, name):
            return getattr(self.wrapped, name)
    canvas = Image.new("RGBA", (180, 100), (120, 120, 120, 255))
    draw = RecordingDraw(renderer.ImageDraw.Draw(canvas))
    placement = renderer.LabelPlacement("91/792M", (90, 20, 170, 56), (50, 38), Operator.KMB, 0, False,
        (("91", Operator.KMB, False), ("792M", Operator.GMB, False)))
    renderer._draw_bus_route_marker(draw, placement, renderer.OPERATOR_COLORS[Operator.KMB], renderer._font(13), phase="connector")
    assert renderer.OPERATOR_COLORS[Operator.KMB] + (220,) in draw.line_calls
    assert renderer.OPERATOR_COLORS[Operator.GMB] + (220,) in draw.line_calls


def test_bus_direction_arrow_is_centred_at_the_road_anchor():
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
    assert not renderer._rects_overlap(placement.rect, renderer._arrow_footprint(placement.marker), padding=0)


def test_bus_label_layout_stacks_collisions_without_overlap_and_deterministic():
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
    # Every marker keeps a pill (nothing dropped) and every pill keeps its
    # Every marker retains a visible label and immutable arrow anchor.
    assert len(first) == 2
    for placement in first:
        left, top, right, bottom = placement.rect
        assert 0 <= left < right <= canvas.width
        assert 0 <= top < bottom <= canvas.height
        assert right - left >= 8  # full-size label, never squashed
        pointer_x, pointer_y = placement.marker
        assert 0 <= pointer_x <= canvas.width
        assert 0 <= pointer_y <= canvas.height
    # Stacked pills never overlap each other.
    for index, placement in enumerate(first):
        assert all(
            not renderer._rects_overlap(placement.rect, other.rect)
            for other in first[index + 1 :]
        )


def test_bus_label_layout_merges_convoy_on_the_road_row():
    canvas = Image.new("RGBA", (360, 180), (255, 255, 255, 255))
    draw = renderer.ImageDraw.Draw(canvas)
    font = renderer._font(13)
    markers = [
        renderer.BusMarker(("91", "91M"), 150, 80, Operator.KMB, 0),
    ]
    placed = renderer._layout_bus_labels(markers, draw, font, canvas.size)
    assert len(placed) == 1
    placement = placed[0]
    assert placement.marker == (150, 80)
    assert not renderer._rects_overlap(placement.rect, renderer._arrow_footprint(placement.marker), padding=0)


def test_bus_label_layout_uses_left_side_when_right_is_blocked():
    canvas = Image.new("RGBA", (180, 100), (255, 255, 255, 255))
    draw = renderer.ImageDraw.Draw(canvas)
    marker = renderer.BusMarker(("792M TKO",), 140, 50, Operator.GMB, 0)
    placed = renderer._layout_bus_labels(
        [marker], draw, renderer._font(13), canvas.size,
        placed_rects=[(120, 35, 178, 65)],
    )
    assert placed[0].rect[2] <= 120
    assert placed[0].marker == (140, 50)


def test_bus_label_layout_clusters_same_anchor_with_per_row_operator_colors():
    canvas = Image.new("RGBA", (240, 120), (255, 255, 255, 255))
    draw = renderer.ImageDraw.Draw(canvas)
    markers = [
        renderer.BusMarker(("792M TKO",), 100, 60, Operator.GMB, 0),
        renderer.BusMarker(("91 Diamond Hill",), 106, 61, Operator.KMB, 0),
    ]
    placed = renderer._layout_bus_labels(markers, draw, renderer._font(13), canvas.size)
    assert len(placed) == 1
    assert [row[1] for row in placed[0].rows] == [Operator.GMB, Operator.KMB]
    assert placed[0].marker == (100, 60)


def test_bus_label_layout_keeps_opposite_direction_arrows_separate():
    canvas = Image.new("RGBA", (240, 120), (255, 255, 255, 255))
    draw = renderer.ImageDraw.Draw(canvas)
    markers = [
        renderer.BusMarker(("91 east",), 100, 60, Operator.KMB, 0),
        renderer.BusMarker(("91 west",), 101, 60, Operator.KMB, math.pi),
    ]
    placed = renderer._layout_bus_labels(markers, draw, renderer._font(13), canvas.size)
    assert len(placed) == 2
    assert {placement.marker for placement in placed} == {(100, 58.0), (101, 62.0)}


def test_bus_label_layout_offsets_grouped_direction_against_opposite_group():
    canvas = Image.new("RGBA", (240, 120), (255, 255, 255, 255))
    draw = renderer.ImageDraw.Draw(canvas)
    markers = [
        renderer.BusMarker(("91",), 100, 60, Operator.KMB, 0),
        renderer.BusMarker(("91M",), 104, 60, Operator.KMB, 0),
        renderer.BusMarker(("792M",), 101, 60, Operator.GMB, math.pi),
    ]
    placed = renderer._layout_bus_labels(markers, draw, renderer._font(13), canvas.size)
    assert len(placed) == 2
    assert {placement.marker for placement in placed} == {(100.0, 58.0), (101.0, 62.0)}


def test_grouped_label_spiral_avoids_initially_occupied_slots():
    canvas = Image.new("RGBA", (240, 180), (255, 255, 255, 255))
    draw = renderer.ImageDraw.Draw(canvas)
    markers = [
        renderer.BusMarker(("792M TKO",), 120, 90, Operator.GMB, 0),
        renderer.BusMarker(("91 Diamond Hill",), 124, 90, Operator.KMB, 0),
    ]
    occupied = [(0, 0, 240, 60), (0, 70, 110, 110), (130, 70, 240, 110)]
    extra_arrow = (112, 148, 124, 160)
    occupied.append(extra_arrow)
    placed = renderer._layout_bus_labels(markers, draw, renderer._font(13), canvas.size, occupied)
    assert len(placed) == 1
    rect = placed[0].rect
    assert 0 <= rect[0] < rect[2] <= canvas.width
    assert 0 <= rect[1] < rect[3] <= canvas.height
    assert not renderer._rects_overlap(rect, occupied[0])
    assert not renderer._rects_overlap(rect, occupied[1])
    assert not renderer._rects_overlap(rect, occupied[2])
    assert not renderer._rects_overlap(rect, extra_arrow, padding=0)


def test_singleton_left_label_reserves_arrow_only_on_left_half():
    canvas = Image.new("RGBA", (180, 80), (255, 255, 255, 255))
    class RecordingDraw:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.text_calls = []

        def text(self, xy, text, *args, **kwargs):
            self.text_calls.append((xy, text))
            return self.wrapped.text(xy, text, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    draw = RecordingDraw(renderer.ImageDraw.Draw(canvas))
    placement = renderer.LabelPlacement("792M TKO", (20, 25, 105, 43), (100, 34), Operator.GMB, 0)
    renderer._draw_bus_route_marker(draw, placement, renderer.OPERATOR_COLORS[Operator.GMB], renderer._font(13))
    text_x, _ = draw.text_calls[-1][0]
    width = draw.textlength("792M TKO", font=renderer._font(13))
    assert text_x == 24
    assert text_x + width <= 101


def test_displaced_singleton_text_stays_inside_selected_box():
    canvas = Image.new("RGBA", (180, 80), (255, 255, 255, 255))
    class RecordingDraw:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.text_calls = []
        def text(self, xy, text, *args, **kwargs):
            self.text_calls.append((xy, text))
            return self.wrapped.text(xy, text, *args, **kwargs)
        def __getattr__(self, name):
            return getattr(self.wrapped, name)
    draw = RecordingDraw(renderer.ImageDraw.Draw(canvas))
    placement = renderer.LabelPlacement("792M TKO", (120, 25, 210, 43), (95, 34), Operator.GMB, 0)
    renderer._draw_bus_route_marker(draw, placement, renderer.OPERATOR_COLORS[Operator.GMB], renderer._font(13))
    text_x, _ = draw.text_calls[-1][0]
    width = draw.textlength("792M TKO", font=renderer._font(13))
    assert text_x == 124
    assert text_x + width <= 206


def test_grouped_mixed_reliability_draws_pale_dashed_row():
    canvas = Image.new("RGBA", (240, 120), (255, 255, 255, 255))
    draw = renderer.ImageDraw.Draw(canvas)
    placement = renderer.LabelPlacement(
        "11/792M", (70, 40, 180, 76), (77, 58), Operator.GMB, 0,
        False, (("11", Operator.GMB, False), ("792M", Operator.GMB, True)),
    )
    renderer._draw_bus_route_marker(draw, placement, renderer.OPERATOR_COLORS[Operator.GMB], renderer._font(13))
    reliable = canvas.getpixel((130, 49))
    unreliable = canvas.getpixel((130, 67))
    assert reliable != unreliable
    assert reliable[1] > reliable[0]
    assert unreliable[0] > reliable[0]
    assert any(
        70 <= x <= 180 and 58 <= y <= 76 and 70 <= canvas.getpixel((x, y))[0] <= 120
        and abs(canvas.getpixel((x, y))[0] - canvas.getpixel((x, y))[1]) <= 15
        for y in range(58, 77) for x in range(70, 181)
    )


def test_grouped_right_of_anchor_rows_keep_text_inside_box():
    canvas = Image.new("RGBA", (240, 120), (255, 255, 255, 255))
    class RecordingDraw:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.text_calls = []
        def text(self, xy, text, *args, **kwargs):
            self.text_calls.append((xy, text))
            return self.wrapped.text(xy, text, *args, **kwargs)
        def __getattr__(self, name):
            return getattr(self.wrapped, name)
    draw = RecordingDraw(renderer.ImageDraw.Draw(canvas))
    placement = renderer.LabelPlacement(
        "11/792M", (120, 40, 220, 76), (100, 58), Operator.GMB, 0,
        False, (("11", Operator.GMB, False), ("792M", Operator.GMB, True)),
    )
    renderer._draw_bus_route_marker(draw, placement, renderer.OPERATOR_COLORS[Operator.GMB], renderer._font(13))
    for ((text_x, _y), _text), (text, _operator, _unreliable) in zip(
        draw.text_calls, placement.rows, strict=True
    ):
        assert text_x == 124
        assert text_x + draw.textlength(text, font=renderer._font(13)) <= 216


def test_traffic_mask_ignores_pale_green_land_and_keeps_google_traffic_colors():
    base = Image.new("RGB", (80, 40), (202, 239, 211))
    draw = renderer.ImageDraw.Draw(base)
    captured_palette = (
        (22, 224, 152), (255, 224, 104), (250, 198, 49),
        (247, 74, 85), (169, 39, 39),
    )
    assert captured_palette == renderer.GOOGLE_TRAFFIC_COLORS
    for index, color in enumerate(captured_palette):
        draw.rectangle((5 + index * 14, 17, 15 + index * 14, 21), fill=color)
    occupancy = renderer._traffic_occupancy(base)
    assert occupancy.overlap((0, 0, 4, 8))[0] == 0
    for index in range(5):
        count, _area = occupancy.overlap((5 + index * 14, 15, 16 + index * 14, 24))
        assert count > 0


def test_traffic_colored_pixels_steer_bus_label_to_clear_candidate():
    canvas = Image.new("RGBA", (240, 120), "white")
    draw = renderer.ImageDraw.Draw(canvas)
    mask = Image.new("L", canvas.size, 0)
    renderer.ImageDraw.Draw(mask).rectangle((0, 0, 99, 119), fill=255)
    traffic = renderer.TrafficOccupancy(mask)
    marker = renderer.BusMarker(("91 Diamond Hill",), 110, 60, Operator.KMB, 0)
    placement = renderer._layout_bus_labels(
        [marker], draw, renderer._font(13), canvas.size, traffic=traffic
    )[0]
    assert placement.marker == (110, 60)
    assert placement.rect[0] >= 120
    assert traffic.overlap(placement.rect)[0] == 0


def test_grouped_label_also_avoids_traffic_colored_candidate():
    canvas = Image.new("RGBA", (280, 160), "white")
    draw = renderer.ImageDraw.Draw(canvas)
    mask = Image.new("L", canvas.size, 0)
    renderer.ImageDraw.Draw(mask).rectangle((0, 0, 129, 159), fill=255)
    traffic = renderer.TrafficOccupancy(mask)
    markers = [
        renderer.BusMarker(("91 Diamond Hill",), 140, 80, Operator.KMB, 0),
        renderer.BusMarker(("11 Choi Hung",), 143, 80, Operator.GMB, 0),
    ]
    placement = renderer._layout_bus_labels(
        markers, draw, renderer._font(13), canvas.size, traffic=traffic
    )[0]
    assert placement.rect[0] >= 150
    assert traffic.overlap(placement.rect)[0] == 0


def test_unavoidable_traffic_keeps_bus_label_local_and_bounded():
    canvas = Image.new("RGBA", (180, 90), "white")
    traffic = renderer.TrafficOccupancy(Image.new("L", canvas.size, 255))
    marker = renderer.BusMarker(("792M TKO",), 90, 45, Operator.GMB, 0)
    placement = renderer._layout_bus_labels(
        [marker], renderer.ImageDraw.Draw(canvas), renderer._font(13), canvas.size,
        traffic=traffic,
    )[0]
    left, top, right, bottom = placement.rect
    assert 0 <= left < right <= canvas.width
    assert 0 <= top < bottom <= canvas.height
    assert (top + bottom) / 2 == pytest.approx(marker.y)
    assert min(abs(right - marker.x), abs(left - marker.x)) <= 11
    assert traffic.overlap(placement.rect)[0] > 0


def test_bus_anchor_snaps_perpendicularly_to_traffic_band():
    base = Image.new("RGB", (200, 120), (232, 238, 233))
    renderer.ImageDraw.Draw(base).line((70, 50, 130, 50), fill=(22, 224, 152), width=5)
    occupancy = renderer._traffic_occupancy(base)
    snapped = occupancy.snap_anchor((100, 60), 0.0)
    assert snapped[0] == pytest.approx(100)
    assert 48 <= snapped[1] <= 52
    assert math.hypot(snapped[0] - 100, snapped[1] - 60) <= 10


def test_bus_anchor_without_traffic_retains_exact_route_anchor():
    occupancy = renderer.TrafficOccupancy(Image.new("L", (200, 120), 0))
    assert occupancy.snap_anchor((100, 60), 0.75) == (100, 60)


def test_bus_anchor_does_not_jump_forward_to_along_route_traffic():
    mask = Image.new("L", (200, 120), 0)
    renderer.ImageDraw.Draw(mask).rectangle((107, 58, 145, 62), fill=255)
    occupancy = renderer.TrafficOccupancy(mask)
    assert occupancy.snap_anchor((100, 60), 0.0) == (100, 60)


def test_saturated_circular_map_icon_does_not_attract_bus_anchor():
    base = Image.new("RGB", (200, 120), (232, 238, 233))
    renderer.ImageDraw.Draw(base).ellipse((96, 48, 104, 56), fill=(247, 74, 85))
    occupancy = renderer._traffic_occupancy(base)
    assert occupancy.snap_anchor((100, 60), 0.0) == (100, 60)


def test_strong_right_road_band_beats_weak_left_speck():
    mask = Image.new("L", (200, 120), 0)
    draw = renderer.ImageDraw.Draw(mask)
    draw.line((92, 52, 108, 52), fill=255, width=1)  # credible but weak left trace
    draw.line((70, 69, 130, 69), fill=255, width=5)  # dense right road band
    occupancy = renderer.TrafficOccupancy(mask)
    snapped = occupancy.snap_anchor((100, 60), 0.0)
    assert snapped[1] > 60
    assert snapped[0] == pytest.approx(100)


def test_strong_left_road_band_beats_weak_center_trace():
    mask = Image.new("L", (200, 120), 0)
    draw = renderer.ImageDraw.Draw(mask)
    draw.line((92, 60, 108, 60), fill=255, width=1)
    draw.line((70, 50, 130, 50), fill=255, width=5)
    occupancy = renderer.TrafficOccupancy(mask)
    snapped = occupancy.snap_anchor((100, 60), 0.0)
    assert snapped[1] < 60
    assert snapped != (100, 60)


def test_left_side_anchor_snap_reverses_with_opposing_heading():
    mask = Image.new("L", (200, 120), 0)
    draw = renderer.ImageDraw.Draw(mask)
    draw.rectangle((75, 49, 125, 52), fill=255)
    draw.rectangle((75, 68, 125, 71), fill=255)
    occupancy = renderer.TrafficOccupancy(mask)
    eastbound = occupancy.snap_anchor((100, 60), 0.0)
    westbound = occupancy.snap_anchor((100, 60), math.pi)
    assert eastbound[1] < 60  # screen-up is left for eastbound traffic
    assert westbound[1] > 60  # screen-down is left after heading reversal
    assert eastbound[0] == pytest.approx(westbound[0], abs=0.01)


def test_traffic_snap_then_opposing_separation_keeps_each_arrow_on_its_left():
    mask = Image.new("L", (240, 120), 0)
    renderer.ImageDraw.Draw(mask).line((70, 60, 170, 60), fill=255, width=5)
    traffic = renderer.TrafficOccupancy(mask)
    original = (120.0, 60.0)
    east_anchor = traffic.snap_anchor(original, 0.0)
    west_anchor = traffic.snap_anchor(original, math.pi)
    markers = [
        renderer.BusMarker(("91 east",), *east_anchor, Operator.KMB, 0.0),
        renderer.BusMarker(("91 west",), *west_anchor, Operator.KMB, math.pi),
    ]
    canvas = Image.new("RGBA", mask.size, "white")
    placements = renderer._layout_bus_labels(
        markers, renderer.ImageDraw.Draw(canvas), renderer._font(13), canvas.size,
        traffic=traffic,
    )
    by_heading = {round(placement.heading): placement.marker for placement in placements}
    east = by_heading[0]
    west = by_heading[3]
    # Screen travel left normals: east=(0,-1), west=(0,+1).
    assert east[1] < east_anchor[1]
    assert west[1] > west_anchor[1]
    assert east[0] == pytest.approx(original[0])
    assert west[0] == pytest.approx(original[0])


def test_bus_anchor_snap_radius_scales_with_native_candidate():
    mask = Image.new("L", (150, 90), 0)
    draw = renderer.ImageDraw.Draw(mask)
    draw.rectangle((55, 32, 95, 34), fill=255)  # just beyond the 0.75x radius
    occupancy = renderer.TrafficOccupancy(mask)
    anchor = (75, 45)
    assert occupancy.snap_anchor(anchor, 0.0, renderer.RenderMetrics(0.75)) == anchor
    assert occupancy.snap_anchor(anchor, 0.0, renderer.DEFAULT_METRICS)[1] < anchor[1]


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
    markers = renderer._merged_public_stop_markers([same_a, same_b, opposite], lines, paths)
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
    extra = Stop("G-X", "Minibus extra", 22.333360, 114.282881)
    path = [
        (stop.lat, stop.lon)
        for stop in (upstream, gate, destination, extra)
    ]
    first = renderer._path_segment_length(path[0], path[1])
    second = renderer._path_segment_length(path[1], path[2])
    third = renderer._path_segment_length(path[2], path[3])
    line = RouteLine(
        "11",
        "GMB",
        "seq-1",
        [upstream, gate, destination, extra],
        path,
        [0, first, first + second, first + second + third],
    )

    # The provider's GMB geometry remains available to ETA interpolation.
    from dashboard.maps.positions import estimate_bus_positions
    from tests.test_positions import Probe

    probes = [
        Probe("GMB", "11", "seq-1", 1, 1),
        Probe("GMB", "11", "seq-1", 2, 0),
    ]
    assert estimate_bus_positions(probes, [line])
    assert renderer._merged_public_stop_markers([gate], [line], [path]) == []


def test_alerted_parallel_carriageways_merge_into_one_transparent_rectangle(monkeypatch):
    monkeypatch.setattr(renderer, "project", lambda lat, lon, *args: (lon, lat))
    base = Image.new("RGBA", (100, 100), (37, 92, 168, 255))
    count = renderer._draw_alerted_road_rectangles(
        base,
        [[(10, 10), (10, 30)], [(14, 10), (14, 30)]],
        renderer.BASE_MAP_LAT, renderer.BASE_MAP_LON, renderer.BASE_MAP_ZOOM, base.size,
    )
    assert count == 1
    # The unfilled centre preserves Google's underlying traffic pixel.
    assert base.getpixel((20, 10)) == (37, 92, 168, 255)
    # The incident indicator is magenta-only; its transparent centre preserves
    # the underlying traffic layer without a white halo.
    assert base.getpixel((20, 4)) == renderer.ALERT_RECT_COLOR
    assert base.getpixel((20, 7)) == (37, 92, 168, 255)
    # An unaffected route remainder remains completely untouched.
    assert base.getpixel((70, 10)) == (37, 92, 168, 255)


def test_alerted_distant_sections_remain_two_rectangles_and_clip_to_bounds(monkeypatch):
    monkeypatch.setattr(renderer, "project", lambda lat, lon, *args: (lon, lat))
    base = Image.new("RGBA", (100, 100), (41, 96, 166, 255))
    count = renderer._draw_alerted_road_rectangles(
        base,
        [[(10, -20), (10, 10)], [(70, 70), (70, 120)]],
        renderer.BASE_MAP_LAT, renderer.BASE_MAP_LON, renderer.BASE_MAP_ZOOM, base.size,
    )
    assert count == 2
    assert base.getpixel((99, 70)) == renderer.ALERT_RECT_COLOR
    assert base.getpixel((50, 50)) == (41, 96, 166, 255)


def test_legend_uses_native_grey_band_without_inset_panel():
    legend = Image.open(io.BytesIO(renderer.render_legend())).convert("RGB")
    background = (246, 247, 249)
    assert legend.getpixel((0, 0)) == background
    assert legend.getpixel((legend.width - 1, legend.height - 1)) == background
    # The rectangle indicator is represented in the legend as well.
    assert renderer.ALERT_RECT_COLOR[:3] in {
        pixel for _count, pixel in legend.getcolors(maxcolors=1_000_000)
    }
    swatch = legend.crop((10, 62, 31, 73))
    assert (255, 255, 255) not in {
        pixel for _count, pixel in swatch.getcolors(maxcolors=1_000_000)
    }
    # Attribution is consolidated and minimized at the bottom of the band.
    source = renderer._draw_legend
    import inspect
    text = inspect.getsource(source)
    assert "Map data © Google · Route geometry © Transport Department HKeMobility" in text


def test_render_map_keeps_bus_and_minibus_markers_with_short_destinations(tmp_path, monkeypatch):
    from dashboard.maps.positions import BusEstimate

    def route_line(route: str, operator: str, latitude: float) -> RouteLine:
        stops = [
            Stop(f"{route}-{index}", f"Stop {index}", latitude, longitude)
            for index, longitude in enumerate((114.242881, 114.252881, 114.262881, 114.272881))
        ]
        path = [(stop.lat, stop.lon) for stop in stops]
        offsets = [0.0]
        for first, second in zip(path, path[1:], strict=False):
            offsets.append(offsets[-1] + renderer._path_segment_length(first, second))
        return RouteLine(route, operator, "outbound", stops, path, offsets)

    lines = [route_line("91", "KMB", 22.333360), route_line("11", "GMB", 22.333361)]
    estimates = [
        BusEstimate("91 Diamond Hill", 22.333360, 114.252881, Operator.KMB, 0.0),
        BusEstimate("11 Choi Hung", 22.333361, 114.252881, Operator.GMB, 0.0),
    ]
    drawn: list[renderer.LabelPlacement] = []
    original = renderer._draw_bus_route_marker

    def tracking(
        draw, placement, color, font, unreliable=False, phase="all",
        metrics=renderer.DEFAULT_METRICS,
    ):
        if phase == "label":
            drawn.append(placement)
        return original(
            draw, placement, color, font, unreliable=unreliable, phase=phase,
            metrics=metrics,
        )

    monkeypatch.setattr(renderer, "_draw_bus_route_marker", tracking)
    renderer.render_map(
        estimates,
        str(tmp_path),
        route_lines=lines,
        base_image=Image.new("RGB", (renderer.MAP_WIDTH, renderer.MAP_HEIGHT), "white"),
    )
    assert len(drawn) == 1
    assert {operator for _text, operator, _unreliable in drawn[0].rows} == {
        Operator.KMB,
        Operator.GMB,
    }
    assert drawn[0].text == "11 Choi Hung/91 Diamond Hill"


def test_render_map_draws_connectors_labels_then_arrows_in_global_passes(tmp_path, monkeypatch):
    from dashboard.maps.positions import BusEstimate

    calls = []
    original = renderer._draw_bus_route_marker
    def tracking(
        draw, placement, color, font, unreliable=False, phase="all",
        metrics=renderer.DEFAULT_METRICS,
    ):
        calls.append(phase)
        return original(
            draw, placement, color, font, unreliable=unreliable, phase=phase,
            metrics=metrics,
        )
    monkeypatch.setattr(renderer, "_draw_bus_route_marker", tracking)
    renderer.render_map(
        [
            BusEstimate("91 Diamond Hill", 22.333360, 114.252881, Operator.KMB, 0.0),
            BusEstimate("792M TKO", 22.333400, 114.272881, Operator.GMB, 0.0),
        ], str(tmp_path), base_image=Image.new("RGB", (renderer.MAP_WIDTH, renderer.MAP_HEIGHT), "white")
    )
    assert calls == ["connector", "connector", "label", "label", "arrow", "arrow"]


def test_render_map_returns_bounded_webp_with_legend_band(tmp_path):
    encoded = renderer.render_map(
        [], str(tmp_path), base_image=Image.new("RGB", (renderer.MAP_WIDTH, renderer.MAP_HEIGHT), (238, 241, 245))
    )
    decoded = Image.open(io.BytesIO(encoded))
    assert decoded.format == "WEBP"
    assert decoded.size == (renderer.MAP_WIDTH, renderer.MAP_HEIGHT + renderer.LEGEND_BAND_HEIGHT)
    assert len(encoded) <= 100_000
    assert all(
        abs(channel - target) <= 3
        for channel, target in zip(decoded.getpixel((500, 200)), (238, 241, 245), strict=True)
    )
    assert decoded.getpixel((10, renderer.MAP_HEIGHT)) != (238, 241, 245)


def test_render_legend_is_standalone_opaque_png_with_all_key_labels():
    encoded = renderer.render_legend()
    image = Image.open(io.BytesIO(encoded))
    image.load()
    assert image.format == "PNG"
    assert image.size == (renderer.LEGEND_WIDTH, renderer.LEGEND_HEIGHT)
    assert image.mode == "RGB"
    assert len(encoded) < 25_000
    # The artwork contains the stable explanatory labels and more than one
    # fill colour, rather than being a blank/transparent attachment.
    assert len(image.getcolors(maxcolors=1_000_000)) > 10
    palette = {pixel for _count, pixel in image.getcolors(maxcolors=1_000_000)}
    captured_traffic_cores = {
        (22, 224, 152), (255, 224, 104), (250, 198, 49),
        (247, 74, 85), (169, 39, 39),
    }
    assert captured_traffic_cores <= palette


def test_attribution_remains_dark_and_legible_after_lossy_webp_at_native_sizes():
    for width, height in ((960, 540), (720, 405)):
        scale = width / renderer.MAP_WIDTH
        metrics = renderer.RenderMetrics(2 * scale)
        composite = renderer._append_legend_band(
            Image.new("RGB", (width, height), (238, 241, 245))
        )
        buffer = io.BytesIO()
        composite.save(buffer, format="WEBP", quality=60, method=6)
        assert len(buffer.getvalue()) <= 100_000
        decoded = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
        mask = Image.new("L", decoded.size, 0)
        attribution = "Map data © Google · Route geometry © Transport Department HKeMobility"
        origin = (60 * scale, height + 20 * scale)
        text_xy = (origin[0] + metrics.px(10), origin[1] + metrics.px(94))
        renderer.ImageDraw.Draw(mask).text(
            text_xy, attribution, fill=255,
            font=renderer._font(metrics.font_size(9, minimum=7)),
        )
        bbox = mask.getbbox()
        assert bbox is not None
        glyph_luma = []
        nearby_background_luma = []
        pixels = decoded.load()
        mask_pixels = mask.load()
        for y in range(max(0, bbox[1] - 2), min(decoded.height, bbox[3] + 2)):
            for x in range(max(0, bbox[0] - 2), min(decoded.width, bbox[2] + 2)):
                luma = sum(pixels[x, y]) / 3
                if mask_pixels[x, y] >= 128:
                    glyph_luma.append(luma)
                elif mask_pixels[x, y] == 0:
                    nearby_background_luma.append(luma)
        assert glyph_luma and nearby_background_luma
        assert sum(glyph_luma) / len(glyph_luma) < 145
        assert (
            sum(nearby_background_luma) / len(nearby_background_luma)
            - sum(glyph_luma) / len(glyph_luma)
            >= 70
        )


def test_render_map_does_not_paint_legend_over_lower_left(tmp_path):
    base = Image.new("RGB", (renderer.MAP_WIDTH, renderer.MAP_HEIGHT), (211, 212, 213))
    encoded = renderer.render_map([], str(tmp_path), base_image=base)
    image = Image.open(io.BytesIO(encoded)).convert("RGB")
    # This was previously inside the opaque legend exclusion panel. It should
    # now retain the traffic base so labels can occupy the area.
    rendered = image.getpixel((100, 500))
    expected = base.getpixel((100, 500))
    assert all(
        abs(channel - target) <= 8
        for channel, target in zip(rendered, expected, strict=True)
    )


def test_render_map_textured_traffic_base_stays_readable_and_bounded(tmp_path):
    # Deterministic map-like texture plus thin traffic strokes exercises the
    # dimension fallback without permitting an unreadably tiny image.
    width, height = renderer.MAP_WIDTH, renderer.MAP_HEIGHT
    textured = Image.new("RGB", (width, height), (232, 235, 238))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(textured)
    for index in range(120):
        y = 8 + index * 4
        draw.line((0, y, width, y + (index % 7) * 3), fill=(150 + index % 40,) * 3, width=1)
    traffic = ((100, (20, 190, 50)), (180, (235, 210, 20)), (260, (245, 130, 10)), (340, (220, 30, 40)))
    for y, color in traffic:
        draw.line((180, y, 780, y + 30), fill=color, width=3)
    encoded = renderer.render_map([], str(tmp_path), base_image=textured)
    decoded = Image.open(io.BytesIO(encoded))
    decoded.load()
    assert decoded.format == "WEBP"
    assert len(encoded) <= 100_000
    assert decoded.width >= renderer.MIN_MAP_WIDTH and decoded.height >= renderer.MIN_MAP_HEIGHT
    for index, (y, _) in enumerate(traffic):
        red, green, blue = decoded.getpixel((480, y + 15))
        assert max(red, green, blue) > 100
        if index == 0:
            assert green > red and green > blue
        elif index == 1:
            assert red > 150 and green > 150 and blue < 150
        elif index == 2:
            assert red > green > blue
        else:
            assert red > green + 50 and red > blue + 40


def test_render_map_webp_preserves_traffic_color_relationships(tmp_path):
    base = Image.new("RGB", (renderer.MAP_WIDTH, renderer.MAP_HEIGHT), "white")
    bands = ((120, (20, 190, 50)), (180, (235, 210, 20)),
             (240, (245, 130, 10)), (300, (220, 30, 40)))
    for y, color in bands:
        for row in range(y, y + 24):
            for x in range(220, 740):
                base.putpixel((x, row), color)
    encoded = renderer.render_map([], str(tmp_path), base_image=base)
    decoded = Image.open(io.BytesIO(encoded))
    decoded.load()
    assert len(encoded) <= 100_000
    green, yellow, orange, red = [decoded.getpixel((480, y + 12)) for y, _ in bands]
    assert green[1] > green[0] and green[1] > green[2]
    assert yellow[0] > 150 and yellow[1] > 150 and yellow[2] < yellow[1] - 60
    assert orange[0] > orange[1] > orange[2]
    assert red[0] > red[1] + 80 and red[0] > red[2] + 60


def test_bus_markers_merge_matching_route_operator_at_same_position():
    from dashboard.maps.positions import BusEstimate

    estimates = [
        BusEstimate("91 Diamond Hill", 22.334, 114.230, Operator.KMB, 0.0),
        BusEstimate("91 Diamond Hill", 22.334, 114.230, Operator.KMB, 0.0),
        BusEstimate("91M Po Lam", 22.334, 114.230, Operator.KMB, 0.0),
    ]
    markers = renderer._merge_bus_markers(
        estimates,
        renderer.BASE_MAP_LAT,
        renderer.BASE_MAP_LON,
        renderer.BASE_MAP_ZOOM,
        (renderer.MAP_WIDTH, renderer.MAP_HEIGHT),
    )
    assert len(markers) == 2
    assert [marker.routes for marker in markers] == [("91 Diamond Hill",), ("91M Po Lam",)]


def test_off_map_bus_prediction_has_no_marker_or_label():
    from dashboard.maps.positions import BusEstimate

    markers = renderer._merge_bus_markers(
        [BusEstimate("91 Diamond Hill", 90.0, 0.0, Operator.KMB, 0.0)],
        renderer.BASE_MAP_LAT,
        renderer.BASE_MAP_LON,
        renderer.BASE_MAP_ZOOM,
        (renderer.MAP_WIDTH, renderer.MAP_HEIGHT),
    )
    assert markers == []


def test_native_fallback_scales_projection_with_resized_base(tmp_path, monkeypatch):
    calls = []

    def fake_once(
        estimates, cache_dir, public_stops=(), route_lines=(), base_image=None,
        affected_road_paths=(),
    ):
        del estimates, cache_dir, public_stops, route_lines, affected_road_paths
        scale = base_image.width / renderer.MAP_WIDTH
        zoom = renderer.BASE_MAP_ZOOM + math.log2(scale)
        calls.append((base_image.size, renderer.project(
            22.33336, 114.252881, renderer.BASE_MAP_LAT, renderer.BASE_MAP_LON,
            zoom, base_image.size,
        )))
        if len(calls) == 1:
            raise renderer._OversizedMapError("map WebP exceeds 100 KB at native resolution")
        return b"native"

    monkeypatch.setattr(renderer, "_render_map_once", fake_once)
    assert renderer.render_map(
        [], str(tmp_path),
        base_image=Image.new("RGB", (renderer.MAP_WIDTH, renderer.MAP_HEIGHT), "white"),
    ) == b"native"
    assert [size for size, _point in calls] == [(960, 540), (864, 486)]
    full = calls[0][1]
    reduced = calls[1][1]
    assert reduced[0] == pytest.approx(full[0] * 0.9)
    assert reduced[1] == pytest.approx(full[1] * 0.9)


def test_native_fallback_rerenders_authored_overlays_without_resizing_composite(
    tmp_path, monkeypatch,
):
    render_sizes = []
    resize_calls = []
    original_resize = Image.Image.resize
    resize_depth = 0

    def tracking_resize(self, size, *args, **kwargs):
        nonlocal resize_depth
        if resize_depth == 0:
            resize_calls.append((self.size, size))
        resize_depth += 1
        try:
            return original_resize(self, size, *args, **kwargs)
        finally:
            resize_depth -= 1

    def fake_once(
        estimates, cache_dir, public_stops=(), route_lines=(), base_image=None,
        affected_road_paths=(),
    ):
        del estimates, cache_dir, public_stops, route_lines, affected_road_paths
        render_sizes.append(base_image.size)
        if len(render_sizes) < 4:
            raise renderer._OversizedMapError("map WebP exceeds 100 KB at native resolution")
        return b"small-enough"

    monkeypatch.setattr(Image.Image, "resize", tracking_resize)
    monkeypatch.setattr(renderer, "_render_map_once", fake_once)
    assert renderer.render_map(
        [], str(tmp_path),
        base_image=Image.new("RGB", (renderer.MAP_WIDTH, renderer.MAP_HEIGHT), "white"),
    ) == b"small-enough"
    assert render_sizes == [(960, 540), (864, 486), (778, 438), (720, 405)]
    assert resize_calls == [
        ((960, 540), (864, 486)),
        ((960, 540), (778, 438)),
        ((960, 540), (720, 405)),
    ]
    assert all(source[1] <= renderer.MAP_HEIGHT for source, _target in resize_calls)
    assert all(source[1] * renderer.MAP_WIDTH == source[0] * renderer.MAP_HEIGHT
               for source, _target in resize_calls)


def test_native_legend_uses_scaled_metrics_without_raster_resize(monkeypatch):
    calls = []

    def tracking_legend(draw, size, metrics=renderer.DEFAULT_METRICS, origin=(0, 0)):
        del draw
        calls.append((size, metrics.scale, origin))

    def forbidden_resize(*args, **kwargs):
        del args, kwargs
        raise AssertionError("legend artwork must be drawn natively")

    monkeypatch.setattr(renderer, "_draw_legend", tracking_legend)
    monkeypatch.setattr(Image.Image, "resize", forbidden_resize)
    composite = renderer._append_legend_band(
        Image.new("RGB", (renderer.MIN_MAP_WIDTH, renderer.MIN_MAP_HEIGHT), "white")
    )
    assert composite.size == (720, 585)
    assert calls == [((720, 180), 1.5, (45.0, 15.0))]


def test_generators_survive_native_candidate_retries(tmp_path, monkeypatch):
    from dashboard.maps.positions import BusEstimate

    estimate = BusEstimate("91 Diamond Hill", 22.33336, 114.252881, Operator.KMB, 0.0)
    stop = Stop("gate", "HKUST", 22.33336, 114.262881)
    line = RouteLine("91", "KMB", "outbound", [stop])
    path = [(22.333, 114.250), (22.334, 114.251)]
    seen = []

    def fake_once(
        estimates, cache_dir, public_stops=(), route_lines=(), base_image=None,
        affected_road_paths=(),
    ):
        del cache_dir
        seen.append((
            list(estimates), list(public_stops), list(route_lines),
            [list(item) for item in affected_road_paths], base_image.size,
        ))
        if len(seen) == 1:
            raise renderer._OversizedMapError("map WebP exceeds 100 KB at native resolution")
        return b"done"

    monkeypatch.setattr(renderer, "_render_map_once", fake_once)
    assert renderer.render_map(
        (item for item in [estimate]), str(tmp_path),
        public_stops=(item for item in [stop]),
        route_lines=(item for item in [line]),
        base_image=Image.new("RGB", (960, 540), "white"),
        affected_road_paths=((point for point in path) for _ in range(1)),
    ) == b"done"
    assert len(seen) == 2
    for estimates, stops, lines, paths, _size in seen:
        assert estimates == [estimate]
        assert stops == [stop]
        assert lines == [line]
        assert paths == [path]


def test_scaled_authored_metrics_cover_fonts_arrows_and_traffic(tmp_path, monkeypatch):
    metrics = renderer.RenderMetrics(0.75)
    assert metrics.font_size(13) == 10
    triangle = renderer._bus_direction_arrow_triangle((50, 50), 0.0, metrics)
    assert triangle[0] == pytest.approx((54.5, 50.0))
    assert renderer._arrow_footprint((50, 50), metrics) == pytest.approx(
        (45.5, 45.5, 54.5, 54.5)
    )

    canvas = Image.new("RGBA", (720, 405), (41, 96, 166, 255))
    zoom = renderer.BASE_MAP_ZOOM + math.log2(metrics.scale)
    path = [(renderer.BASE_MAP_LAT, renderer.BASE_MAP_LON - 0.001),
            (renderer.BASE_MAP_LAT, renderer.BASE_MAP_LON + 0.001)]
    assert renderer._draw_alerted_road_rectangles(
        canvas, [path], renderer.BASE_MAP_LAT, renderer.BASE_MAP_LON,
        zoom, canvas.size, metrics,
    ) == 1
    magenta = [
        (x, y) for y in range(canvas.height) for x in range(canvas.width)
        if canvas.getpixel((x, y)) == renderer.ALERT_RECT_COLOR
    ]
    assert magenta
    xs = [point[0] for point in magenta]
    expected_points = [renderer.project(
        lat, lon, renderer.BASE_MAP_LAT, renderer.BASE_MAP_LON, zoom, canvas.size
    ) for lat, lon in path]
    assert min(xs) == pytest.approx(min(point[0] for point in expected_points) - 5.25, abs=1)
    assert max(xs) == pytest.approx(max(point[0] for point in expected_points) + 5.25, abs=1)
    # A 4 px logical traffic outline becomes three native pixels at 0.75x.
    mid_x = round(sum(point[0] for point in expected_points) / 2)
    vertical_pixels = [y for y in range(canvas.height)
                       if canvas.getpixel((mid_x, y)) == renderer.ALERT_RECT_COLOR]
    assert len(vertical_pixels) == 6

    font_sizes = []
    original_font = renderer._font

    def tracking_font(size):
        font_sizes.append(size)
        return original_font(size)

    monkeypatch.setattr(renderer, "_font", tracking_font)
    encoded = renderer._render_map_once(
        [], str(tmp_path), base_image=Image.new("RGB", (720, 405), (238, 241, 245))
    )
    assert len(encoded) <= 100_000
    decoded = Image.open(io.BytesIO(encoded))
    assert decoded.size == (renderer.MIN_MAP_WIDTH, renderer.MIN_MAP_HEIGHT + 180)
    assert 10 in font_sizes  # map labels/gate labels
    assert 18 in font_sizes  # 12 px legend copy authored directly at 1.5x


def test_legend_has_compact_google_traffic_key_without_obsolete_explanation():
    import inspect

    source = inspect.getsource(renderer._draw_legend)
    assert "Estimated buses (not GPS)" in source
    assert "Google traffic" in source
    assert "Live traffic speed" not in source
    assert "traffic jam" not in source
    assert "Arrows show both travel directions" not in source


def test_google_traffic_legend_is_swatch_first():
    import inspect

    source = inspect.getsource(renderer._draw_legend)
    assert source.index("for color in GOOGLE_TRAFFIC_COLORS") < source.index(
        '"Google traffic"'
    )
    assert "Map data © Google" in source
