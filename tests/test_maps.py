"""Google base-map capture and retained bus/stop marker tests."""

from __future__ import annotations

import base64
import asyncio
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

    def tracking(draw, x, y, heading, color, square):
        result = original(draw, x, y, heading, color, square)
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
    # white pointer inside its own box.
    assert len(first) == len(markers)
    for placement in first:
        left, top, right, bottom = placement.rect
        assert 0 <= left < right <= canvas.width
        assert 0 <= top < bottom <= canvas.height
        assert right - left >= 20  # full-size pill, never squashed
        pointer_x, pointer_y = placement.marker
        assert left + 2 <= pointer_x <= right - 2
        assert top + 2 <= pointer_y <= bottom - 2
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
    # Road row: pill vertically centred on the anchor with the white triangle
    # exactly at the road point.
    assert (placement.rect[1] + placement.rect[3]) / 2 == 80
    assert placement.marker == (150, 80)
    assert placement.rect[0] < 150 < placement.rect[2]


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

    def tracking(draw, placement, color, font, unreliable=False):
        drawn.append(placement)
        return original(draw, placement, color, font, unreliable=unreliable)

    monkeypatch.setattr(renderer, "_draw_bus_route_marker", tracking)
    renderer.render_map(
        estimates,
        str(tmp_path),
        route_lines=lines,
        base_image=Image.new("RGB", (renderer.MAP_WIDTH, renderer.MAP_HEIGHT), "white"),
    )
    assert {placement.operator for placement in drawn} == {Operator.KMB, Operator.GMB}
    assert {placement.text for placement in drawn} == {
        "91 Diamond Hill",
        "11 Choi Hung",
    }


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


def test_legend_has_no_obsolete_google_traffic_or_direction_explanation():
    import inspect

    source = inspect.getsource(renderer._draw_legend)
    assert "Estimated buses (not GPS)" in source
    assert "Live traffic speed" not in source
    assert "traffic jam" not in source
    assert "Arrows show both travel directions" not in source
    assert "Map data © Google" in source
