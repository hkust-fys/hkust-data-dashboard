"""Google base-map capture and retained bus/stop marker tests."""

from __future__ import annotations

import asyncio
import base64
import io
import math
import sys
import types
from types import SimpleNamespace

import pytest
from PIL import Image, ImageChops

from dashboard.maps import renderer, tiles
from dashboard.maps.positions import BusEstimate
from dashboard.models import EtaRow, Operator, RouteEtaGroup
from dashboard.providers.route_geometry import RouteLine, Stop


def _probe_snapshot(rows=()):
    return SimpleNamespace(
        rows=tuple(rows), complete_routes=(), routes=(), collected_at=None
    )


def _candidate(position):
    return BusEstimate("R destination", 22.3, 114.2, Operator.KMB, 0.0,
                       route="R", bound="out", position=position,
                       operator_code="KMB")


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
    monkeypatch.setattr(maps, "fetch_probe_snapshot", failed_probes)
    monkeypatch.setattr(maps, "render_map", render)

    png, _ = await maps.fetch_traffic_map(object())
    assert png == b"rendered"
    assert captured["base"] == base


@pytest.mark.asyncio
async def test_marker_audit_failure_never_blocks_a_rendered_frame(monkeypatch, caplog):
    import dashboard.maps as maps
    from dashboard.providers.route_geometry import RouteGeometry

    line = RouteLine(
        "91", "KMB", "outbound",
        [Stop("a", "A", 22.33, 114.26), Stop("b", "B", 22.331, 114.261)],
    )
    audit_calls = 0

    async def capture(*_args, **_kwargs):
        return b"google-base"

    async def geometry(*_args, **_kwargs):
        return RouteGeometry(routes=[line])

    async def probes(*_args, **_kwargs):
        return _probe_snapshot()

    def broken_audit(*_args, **_kwargs):
        nonlocal audit_calls
        audit_calls += 1
        raise RuntimeError("audit test failure")

    monkeypatch.setattr(maps, "capture_gmaps_base", capture)
    monkeypatch.setattr(maps, "fetch_route_geometry", geometry)
    monkeypatch.setattr(maps, "fetch_probe_snapshot", probes)
    monkeypatch.setattr(maps, "audit_marker_positions", broken_audit)
    monkeypatch.setattr(maps, "render_map", lambda *_args, **_kwargs: b"rendered")

    image, _ = await maps.fetch_traffic_map(object())

    assert image == b"rendered"
    assert audit_calls == 1
    assert "marker audit unavailable" in caplog.text


@pytest.mark.asyncio
async def test_probe_selection_receives_verified_gate_ids_as_mandatory_anchors(monkeypatch):
    import dashboard.maps as maps
    from dashboard.providers.route_geometry import RouteGeometry

    line = RouteLine("91", "KMB", "outbound", [Stop("gate", "gate", 22.33, 114.26)])
    seen = {}

    async def geometry(*_args, **_kwargs):
        return RouteGeometry(routes=[line])

    def select(lines, *, mandatory_stop_ids):
        seen["ids"] = set(mandatory_stop_ids)
        return []

    monkeypatch.setattr(maps, "capture_gmaps_base", lambda **_kwargs: _resolved(b"base"))
    monkeypatch.setattr(maps, "fetch_route_geometry", geometry)
    monkeypatch.setattr(maps, "select_probe_stops", select)
    monkeypatch.setattr(maps, "render_map", lambda *_args, **_kwargs: b"rendered")
    await maps.fetch_traffic_map(object())
    assert {"B002CEF0DBC568F5", "003130", "20013011"} <= seen["ids"]


@pytest.mark.asyncio
async def test_tracker_failure_renders_stateless_candidates(monkeypatch):
    import dashboard.maps as maps
    from dashboard.providers.route_geometry import RouteGeometry

    candidate = _candidate(1.0)
    async def geometry(*_args, **_kwargs):
        return RouteGeometry(routes=[RouteLine("91", "KMB", "outbound", [])])
    async def snapshot(*_args, **_kwargs):
        return _probe_snapshot()
    class BrokenTracker:
        async def update(self, *_args, **_kwargs):
            raise RuntimeError("tracker unavailable")
    seen = {}
    monkeypatch.setattr(maps, "capture_gmaps_base", lambda **_kwargs: _resolved(b"base"))
    monkeypatch.setattr(maps, "fetch_route_geometry", geometry)
    monkeypatch.setattr(maps, "fetch_probe_snapshot", snapshot)
    monkeypatch.setattr(maps, "estimate_bus_positions", lambda *_args, **_kwargs: [candidate])
    def render(estimates, *_args):
        seen["estimates"] = estimates
        return b"rendered"
    monkeypatch.setattr(maps, "render_map", render)
    await maps.fetch_traffic_map(object(), tracker=BrokenTracker())
    assert seen["estimates"] == [candidate]


@pytest.mark.asyncio
async def test_marker_audit_receives_stateless_candidates_when_tracker_changes_output(monkeypatch):
    import dashboard.maps as maps
    from dashboard.providers.route_geometry import RouteGeometry
    candidate = _candidate(1.0)
    changed = _candidate(2.0)
    audited = {}
    async def geometry(*_args, **_kwargs):
        return RouteGeometry(routes=[RouteLine("91", "KMB", "outbound", [])])
    async def snapshot(*_args, **_kwargs):
        return _probe_snapshot()
    class Tracker:
        async def update(self, *_args, **_kwargs):
            return [changed]
    def audit(*args, **kwargs):
        audited["estimates"] = args[2]
        return {"checks": [], "issues": [], "stats": {"markers": 0}, "ok": True,
                "gmb_marker_pairs": ()}
    monkeypatch.setattr(maps, "capture_gmaps_base", lambda **_kwargs: _resolved(b"base"))
    monkeypatch.setattr(maps, "fetch_route_geometry", geometry)
    monkeypatch.setattr(maps, "fetch_probe_snapshot", snapshot)
    monkeypatch.setattr(maps, "estimate_bus_positions", lambda *_args, **_kwargs: [candidate])
    monkeypatch.setattr(maps, "audit_marker_positions", audit)
    monkeypatch.setattr(maps, "render_map", lambda *_args, **_kwargs: b"rendered")
    await maps.fetch_traffic_map(object(), tracker=Tracker())
    assert audited["estimates"] == [candidate]


def test_marker_issue_warning_key_is_deduplicated_and_bounded():
    import dashboard.maps as maps

    maps._logged_marker_issue_keys.clear()
    key = (("GMB", "11", "seq-1"), "checkpoint", 4, "no unique later-stop ETA match")
    assert maps._first_marker_issue(key)
    assert not maps._first_marker_issue(key)
    maps._logged_marker_issue_keys.clear()


def test_marker_issue_warning_key_deduplicates_gmb_pair_details():
    import dashboard.maps as maps

    maps._logged_marker_issue_keys.clear()
    key = (("GMB", "11", "seq-1"), "gmb-marker-pair", 12, "stacked")
    assert maps._first_marker_issue(key)
    assert not maps._first_marker_issue(key)
    maps._logged_marker_issue_keys.clear()


def test_marker_issue_warning_key_evicts_oldest_deterministically(monkeypatch):
    import dashboard.maps as maps

    monkeypatch.setattr(maps, "_MARKER_ISSUE_KEY_LIMIT", 2)
    maps._logged_marker_issue_keys.clear()
    assert maps._first_marker_issue(("oldest",))
    assert maps._first_marker_issue(("middle",))
    assert maps._first_marker_issue(("newest",))
    assert list(maps._logged_marker_issue_keys) == [("middle",), ("newest",)]
    assert maps._first_marker_issue(("oldest",))
    assert not maps._first_marker_issue(("newest",))
    maps._logged_marker_issue_keys.clear()


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


@pytest.mark.asyncio
async def test_capture_reuses_page_and_skips_unchanged_cache_write(tmp_path, monkeypatch):
    class Browser:
        def is_connected(self):
            return True

    class Page:
        async def evaluate(self, _script):
            return ["valid"]

        async def close(self):
            pass

    page = Page()
    create_calls = []
    image = Image.new("RGB", (40, 20), (20, 40, 60))
    async def create(_key):
        create_calls.append(1)
        tiles._shared_page = page
        tiles._shared_context = object()
        tiles._capture_key = _key
        return page

    monkeypatch.setattr(tiles, "_get_shared_browser", lambda: _resolved(Browser()))
    monkeypatch.setattr(tiles, "_create_capture_page", create)
    monkeypatch.setattr(tiles, "_decode_first_valid_canvas", lambda *_args: image.copy())
    tiles._shared_browser = Browser()
    tiles._shared_page = None
    tiles._shared_context = None
    tiles._capture_key = None
    tiles._last_capture_digest = None
    tiles._last_capture_image = None
    tiles._last_capture_identity = None
    tiles._capture_retry_after = 0.0
    first = await tiles.capture_gmaps_base(str(tmp_path), viewport=(40, 20))
    cache_path = tmp_path / tiles.cache_filename(tiles.GMAPS_BASE_URL, (40, 20))
    first_mtime = cache_path.stat().st_mtime_ns
    await asyncio.sleep(0.01)
    second = await tiles.capture_gmaps_base(str(tmp_path), viewport=(40, 20))
    assert first.tobytes() == second.tobytes() and first is not second
    assert len(create_calls) == 1
    assert cache_path.stat().st_mtime_ns == first_mtime
    await tiles.shutdown_gmaps_browser()


async def _resolved(value):
    return value


@pytest.mark.asyncio
async def test_identical_pixels_switching_cache_directory_writes_both_targets(tmp_path, monkeypatch):
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"
    image = Image.new("RGB", (40, 20), (20, 40, 60))
    class Page:
        async def evaluate(self, _script):
            return ["valid"]
        async def close(self):
            pass
    page = Page()
    async def create(key):
        tiles._shared_page = page
        tiles._shared_context = object()
        tiles._capture_key = key
        return page
    monkeypatch.setattr(tiles, "_get_shared_browser", lambda: _resolved(object()))
    monkeypatch.setattr(tiles, "_create_capture_page", create)
    monkeypatch.setattr(tiles, "_decode_first_valid_canvas", lambda *_args: image.copy())
    for name in ("_shared_page", "_shared_context", "_capture_key", "_last_capture_digest",
                 "_last_capture_image", "_last_capture_identity"):
        setattr(tiles, name, None)
    tiles._capture_retry_after = 0.0
    await tiles.capture_gmaps_base(str(first_dir), viewport=(40, 20))
    await tiles.capture_gmaps_base(str(second_dir), viewport=(40, 20))
    assert (first_dir / tiles.cache_filename(tiles.GMAPS_BASE_URL, (40, 20))).exists()
    assert (second_dir / tiles.cache_filename(tiles.GMAPS_BASE_URL, (40, 20))).exists()
    await tiles.shutdown_gmaps_browser()


@pytest.mark.asyncio
async def test_capture_page_settles_through_placeholders_before_valid_export(monkeypatch):
    class Page:
        def __init__(self):
            self.calls = 0
        async def goto(self, *_args, **_kwargs):
            pass
        async def wait_for_selector(self, *_args, **_kwargs):
            pass
        async def evaluate(self, _script):
            self.calls += 1
            return ["candidate"]
    class Context:
        async def new_page(self):
            return page
    class Browser:
        async def new_context(self, **_kwargs):
            return context
    page, context = Page(), Context()
    monkeypatch.setattr(tiles, "_get_shared_browser", lambda: _resolved(Browser()))
    outcomes = iter([ValueError("placeholder"), ValueError("placeholder"), Image.new("RGB", (40, 20))])
    def decode(*_args):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(tiles, "_decode_first_valid_canvas", decode)
    monkeypatch.setattr(tiles.asyncio, "sleep", lambda _delay: _resolved(None))
    tiles._shared_page = tiles._shared_context = tiles._capture_key = None
    await tiles._create_capture_page(("https://example.test", (40, 20)))
    assert page.calls == 3
    await tiles._recycle_capture_page()


@pytest.mark.asyncio
async def test_recycle_cancellation_finishes_page_and_context_cleanup():
    page_started = asyncio.Event()
    page_release = asyncio.Event()
    class Resource:
        def __init__(self, is_page=False):
            self.closed = False
            self.is_page = is_page
        async def close(self):
            if self.is_page:
                page_started.set()
                await page_release.wait()
            self.closed = True
    page, context = Resource(True), Resource()
    tiles._shared_page, tiles._shared_context = page, context
    task = asyncio.create_task(tiles._recycle_capture_page())
    await page_started.wait()
    task.cancel()
    page_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert page.closed and context.closed
    assert tiles._shared_page is None and tiles._shared_context is None


@pytest.mark.asyncio
async def test_shutdown_clears_last_frame_identity():
    class Browser:
        def is_connected(self):
            return True
        async def close(self):
            pass
    tiles._shared_browser = Browser()
    tiles._shared_context = tiles._shared_page = None
    tiles._last_capture_digest = "digest"
    tiles._last_capture_image = Image.new("RGB", (2, 2))
    tiles._last_capture_identity = ("cache.png", ("url", (2, 2)))
    await tiles.shutdown_gmaps_browser()
    assert tiles._last_capture_digest is None
    assert tiles._last_capture_image is None
    assert tiles._last_capture_identity is None


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


def test_citybus_live_text_is_dark_in_singleton_and_grouped_labels():
    class RecordingDraw:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.text_fills = []

        def text(self, *args, **kwargs):
            self.text_fills.append(kwargs.get("fill"))
            return self.wrapped.text(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    canvas = Image.new("RGBA", (220, 100), (120, 120, 120, 255))
    draw = RecordingDraw(renderer.ImageDraw.Draw(canvas))
    font = renderer._font(13)
    renderer._draw_bus_route_marker(
        draw,
        renderer.LabelPlacement(
            "792M TKO", (20, 15, 100, 35), (60, 25),
            Operator.CITYBUS, 0,
        ),
        renderer.OPERATOR_COLORS[Operator.CITYBUS],
        font,
        phase="label",
    )
    renderer._draw_bus_route_marker(
        draw,
        renderer.LabelPlacement(
            "792M/91", (110, 15, 205, 55), (155, 35), Operator.CITYBUS, 0,
            False,
            (("792M TKO", Operator.CITYBUS, False),
             ("91 Diamond Hill", Operator.KMB, False)),
        ),
        renderer.OPERATOR_COLORS[Operator.CITYBUS],
        font,
        phase="label",
    )
    assert draw.text_fills == [
        (25, 25, 25, 255),
        (25, 25, 25, 255),
        (255, 255, 255, 255),
    ]


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
    assert draw.line_calls[0] == ((50.0, 30.0, 70.0, 30.0), (0, 0, 0, 255))


def test_diagonal_connector_terminates_on_visible_rounded_corner():
    class RecordingDraw:
        def __init__(self, wrapped):
            self.wrapped, self.line_calls = wrapped, []

        def line(self, xy, *args, **kwargs):
            self.line_calls.append(xy)
            return self.wrapped.line(xy, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    canvas = Image.new("RGBA", (120, 80), (120, 120, 120, 255))
    draw = RecordingDraw(renderer.ImageDraw.Draw(canvas))
    placement = renderer.LabelPlacement(
        "792M", (70, 20, 110, 38), (50, 0), Operator.GMB, 0
    )
    renderer._draw_bus_route_marker(
        draw,
        placement,
        renderer.OPERATOR_COLORS[Operator.GMB],
        renderer._font(13),
        phase="connector",
    )

    start_x, start_y, end_x, end_y = draw.line_calls[0]
    assert (start_x, start_y) == placement.marker
    assert (end_x, end_y) != (70, 20)  # the transparent bounding-box corner
    assert math.hypot(end_x - 74, end_y - 24) == pytest.approx(4)
    assert end_x == pytest.approx(71.171572875)
    assert end_y == pytest.approx(21.171572875)


def test_mixed_group_connector_is_one_opaque_black_link():
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
    assert draw.line_calls
    assert set(draw.line_calls) == {(0, 0, 0, 255)}


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
    assert {placement.marker for placement in placed} == {(100, 60), (101, 60)}
    assert not renderer._rects_overlap(placed[0].rect, placed[1].rect)


def test_bus_label_layout_keeps_group_anchor_when_opposite_group_is_nearby():
    canvas = Image.new("RGBA", (240, 120), (255, 255, 255, 255))
    draw = renderer.ImageDraw.Draw(canvas)
    markers = [
        renderer.BusMarker(("91",), 100, 60, Operator.KMB, 0),
        renderer.BusMarker(("91M",), 104, 60, Operator.KMB, 0),
        renderer.BusMarker(("792M",), 101, 60, Operator.GMB, math.pi),
    ]
    placed = renderer._layout_bus_labels(markers, draw, renderer._font(13), canvas.size)
    assert len(placed) == 2
    assert {placement.marker for placement in placed} == {(100, 60), (101, 60)}
    assert not renderer._rects_overlap(placed[0].rect, placed[1].rect)


def test_grouped_label_spiral_avoids_initially_occupied_slots():
    canvas = Image.new("RGBA", (240, 180), (255, 255, 255, 255))
    draw = renderer.ImageDraw.Draw(canvas)
    markers = [
        renderer.BusMarker(("792M TKO",), 120, 90, Operator.GMB, 0),
        renderer.BusMarker(("91 Diamond Hill",), 124, 90, Operator.KMB, 0),
    ]
    occupied = [(0, 0, 240, 60), (0, 70, 110, 110), (130, 70, 240, 110)]
    # Leave a genuinely clear lower grid slot across platform font metrics.
    extra_arrow = (112, 160, 124, 172)
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


def test_grouped_grid_fallback_prefers_clear_slot_with_fixed_font_metrics():
    """Exercise the grid branch independently of platform-installed fonts."""
    class FixedWidthDraw:
        def textlength(self, _text, font=None):
            del font
            return 104.078

    markers = [
        renderer.BusMarker(("792M TKO",), 120, 90, Operator.GMB, 0),
        renderer.BusMarker(("91 Diamond Hill",), 124, 90, Operator.KMB, 0),
    ]
    occupied = [(0, 0, 240, 60), (0, 70, 110, 110), (130, 70, 240, 110)]
    # Leave one arrow-safe grid slot below the occupied bands; the old grid
    # fallback incorrectly preferred a closer slot overlapping the right band.
    occupied.append((112, 160, 124, 172))
    placed = renderer._layout_bus_labels(
        markers, FixedWidthDraw(), renderer._font(13), (240, 180), occupied
    )
    assert len(placed) == 1
    rect = placed[0].rect
    assert not any(renderer._rects_overlap(rect, other) for other in occupied[:3])
    assert not renderer._rects_overlap(rect, occupied[3], padding=0)


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
    lower_edge = [canvas.getpixel((70, y)) for y in range(59, 72)]
    assert any(max(pixel[:3]) < 80 for pixel in lower_edge)
    assert any(max(pixel[:3]) > 100 for pixel in lower_edge)


@pytest.mark.parametrize("rows", [
    (("live", Operator.KMB, False), ("scheduled", Operator.GMB, True)),
    (("scheduled", Operator.GMB, True), ("live", Operator.KMB, False)),
])
@pytest.mark.parametrize("scale", [1.0, 0.75])
def test_grouped_rows_share_one_outline_without_horizontal_separators(rows, scale):
    metrics = renderer.RenderMetrics(scale)
    canvas = Image.new("RGBA", (round(240 * scale), round(120 * scale)), (255, 255, 255, 255))
    draw = renderer.ImageDraw.Draw(canvas)
    top, bottom = 40 * scale, 78 * scale
    left, right = 70 * scale, 190 * scale
    placement = renderer.LabelPlacement(
        "KMB/GMB", (left, top, right, bottom),
        (77 * scale, (top + bottom) / 2), Operator.KMB, 0,
        False, rows,
    )
    renderer._draw_bus_route_marker(
        draw, placement, renderer.OPERATOR_COLORS[Operator.KMB],
        renderer._font(metrics.font_size(13)), metrics=metrics,
    )
    midpoint = (top + bottom) / 2
    # The operator fill changes directly at the row boundary; no black/grey
    # horizontal border is inserted through the combined box.
    boundary_pixel = canvas.getpixel((round(right - 12 * scale), round(midpoint)))
    assert max(boundary_pixel[:3]) > 90

    # Each row still controls its own portion of the outside edge: live is a
    # continuous stroke, while scheduled has visible dash gaps.
    radius = metrics.integer(4)
    for index, (_text, _operator, scheduled) in enumerate(rows):
        row_top = top + index * (bottom - top) / len(rows)
        row_bottom = top + (index + 1) * (bottom - top) / len(rows)
        sample_top = math.ceil(max(row_top, top + radius) + 1)
        sample_bottom = math.floor(min(row_bottom, bottom - radius) - 1)
        samples = [
            canvas.getpixel((round(left), y))
            for y in range(sample_top, sample_bottom + 1)
        ]
        dark = sum(max(pixel[:3]) < 80 for pixel in samples)
        assert dark > 0
        if scheduled:
            assert dark < len(samples)
        else:
            assert dark >= len(samples) - 1


@pytest.mark.parametrize("scheduled", [False, True])
def test_same_reliability_group_has_one_rounded_outer_border(scheduled):
    canvas = Image.new("RGBA", (240, 120), (255, 255, 255, 255))
    rows = (
        ("91 Diamond Hill", Operator.KMB, scheduled),
        ("91 Diamond Hill", Operator.KMB, scheduled),
    )
    placement = renderer.LabelPlacement(
        "91 Diamond Hill/91 Diamond Hill", (70, 40, 200, 78),
        (60, 59), Operator.KMB, 0, scheduled, rows,
    )
    renderer._draw_bus_route_marker(
        renderer.ImageDraw.Draw(canvas), placement,
        renderer.OPERATOR_COLORS[Operator.KMB], renderer._font(13),
    )
    # Rounded outside corner and uninterrupted interior confirm that the two
    # vehicle rows are items in one marker, not two bordered mini-boxes.
    assert canvas.getpixel((70, 40)) == (255, 255, 255, 255)
    assert max(canvas.getpixel((190, 59))[:3]) > 90


@pytest.mark.parametrize("scale", [1.0, 0.75])
def test_timetable_label_is_opaque_and_high_contrast_at_native_scales(scale):
    metrics = renderer.RenderMetrics(scale)
    canvas = Image.new("RGBA", (960 if scale == 1 else 720, 120), (20, 20, 20, 255))
    draw = renderer.ImageDraw.Draw(canvas)
    placement = renderer.LabelPlacement(
        "91", (70 * scale, 40 * scale, 150 * scale, 62 * scale),
        (30 * scale, 51 * scale), Operator.KMB, 0, True,
    )
    renderer._draw_bus_route_marker(
        draw, placement, renderer.OPERATOR_COLORS[Operator.KMB], renderer._font(metrics.font_size(13)),
        unreliable=True, metrics=metrics,
    )
    # The fill is opaque and the label includes genuinely dark text pixels.
    fill = canvas.getpixel((round(100 * scale), round(45 * scale)))
    assert fill[3] == 255
    assert any(
        canvas.getpixel((x, y))[0] < 70
        for x in range(round(70 * scale), round(150 * scale) + 1)
        for y in range(round(40 * scale), round(62 * scale) + 1)
    )


@pytest.mark.parametrize("scale", [1.0, 0.75])
def test_bus_arrow_and_connector_use_opaque_native_two_pixel_strokes(scale):
    metrics = renderer.RenderMetrics(scale)

    class RecordingDraw:
        def __init__(self):
            self.lines = []
        def line(self, xy, *args, **kwargs):
            self.lines.append(kwargs)
        def polygon(self, *args, **kwargs):
            pass

    arrow_draw = RecordingDraw()
    renderer._draw_colored_bus_arrow(arrow_draw, (40, 40), 0, [renderer.OPERATOR_COLORS[Operator.KMB]], metrics)
    assert arrow_draw.lines[-1]["width"] == 2
    assert arrow_draw.lines[-1]["fill"] == (25, 25, 25, 255)

    canvas = Image.new("RGBA", (120, 80), (120, 120, 120, 255))
    connector_draw = renderer.ImageDraw.Draw(canvas)
    placement = renderer.LabelPlacement("91", (70, 20, 110, 38), (50, 30), Operator.KMB, 0)
    renderer._draw_bus_route_marker(
        connector_draw, placement, renderer.OPERATOR_COLORS[Operator.KMB], renderer._font(13),
        phase="connector", metrics=metrics,
    )
    # A two-pixel opaque connector changes both rows around its midpoint.
    assert canvas.getpixel((60, 30)) == (0, 0, 0, 255)


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


def _synthetic_route_mask(
    *, scale=1.0, heading=0.0, width=3, anchor=(100.0, 60.0),
    lateral=0.0, length=35.0,
):
    mask = Image.new("L", (round(200 * scale), round(120 * scale)), 0)
    dx, dy = math.cos(heading), -math.sin(heading)
    nx, ny = dy, -dx
    center = (
        anchor[0] * scale + nx * lateral * scale,
        anchor[1] * scale + ny * lateral * scale,
    )
    extent = length * scale
    renderer.ImageDraw.Draw(mask).line(
        (
            round(center[0] - dx * extent), round(center[1] - dy * extent),
            round(center[0] + dx * extent), round(center[1] + dy * extent),
        ),
        fill=255, width=max(1, round(width * scale)),
    )
    return mask


def _merged_synthetic_bus_anchor(
    monkeypatch, traffic_or_mask, heading=0.0, scale=1.0, anchor=(100.0, 60.0),
):
    monkeypatch.setattr(
        renderer, "project", lambda *args: (anchor[0] * scale, anchor[1] * scale)
    )
    estimate = BusEstimate("R destination", 22.3, 114.2, Operator.KMB, heading)
    traffic = (
        traffic_or_mask
        if isinstance(traffic_or_mask, renderer.TrafficOccupancy)
        else renderer.TrafficOccupancy(traffic_or_mask)
    )
    markers = renderer._merge_bus_markers(
        [estimate], 0, 0, 0, (traffic.width, traffic.height),
        renderer.RenderMetrics(scale), traffic,
    )
    return markers[0]


def _expected_left_anchor(anchor, heading, scale, offset=2.5):
    dx, dy = math.cos(heading), -math.sin(heading)
    return (
        anchor[0] * scale + dy * offset * scale,
        anchor[1] * scale - dx * offset * scale,
    )


@pytest.mark.parametrize("width", [2, 3, 4, 5, 6])
def test_route_marker_accepts_realistic_odd_and_even_route_strokes(monkeypatch, width):
    marker = _merged_synthetic_bus_anchor(
        monkeypatch, _synthetic_route_mask(width=width)
    )
    assert (marker.x, marker.y) == pytest.approx(_expected_left_anchor((100, 60), 0, 1))


@pytest.mark.parametrize(
    ("heading", "anchor"),
    [(math.radians(8), (100.0, 60.0)), (math.radians(35), (100.35, 60.4))],
)
def test_route_marker_accepts_matching_diagonal_heading_and_subpixel_anchor(
    monkeypatch, heading, anchor,
):
    marker = _merged_synthetic_bus_anchor(
        monkeypatch,
        _synthetic_route_mask(heading=heading, width=5, anchor=anchor),
        heading=heading,
        anchor=anchor,
    )
    assert (marker.x, marker.y) == pytest.approx(
        _expected_left_anchor(anchor, heading, 1), abs=0.01
    )


@pytest.mark.parametrize("lateral", [-1, 1])
def test_route_marker_accepts_registered_stroke_only_when_core_overlaps_centerline(
    monkeypatch, lateral,
):
    marker = _merged_synthetic_bus_anchor(
        monkeypatch, _synthetic_route_mask(width=3, lateral=lateral)
    )
    assert (marker.x, marker.y) == pytest.approx(_expected_left_anchor((100, 60), 0, 1))


@pytest.mark.parametrize("gap", [1, 2])
def test_route_marker_accepts_short_classification_gap(monkeypatch, gap):
    mask = _synthetic_route_mask(width=4)
    renderer.ImageDraw.Draw(mask).rectangle((88, 54, 87 + gap, 66), fill=0)
    marker = _merged_synthetic_bus_anchor(monkeypatch, mask)
    assert (marker.x, marker.y) == pytest.approx(_expected_left_anchor((100, 60), 0, 1))


@pytest.mark.parametrize("scale", [1.0, 0.9, 0.81, 0.75])
@pytest.mark.parametrize("distance", [1, 2, 3])
def test_route_marker_ignores_nearby_parallel_traffic(monkeypatch, distance, scale):
    base = Image.new("RGB", (round(200 * scale), round(120 * scale)), (232, 238, 233))
    core = _synthetic_route_mask(scale=scale, width=1, lateral=distance)
    base.paste((247, 74, 85), mask=core)
    marker = _merged_synthetic_bus_anchor(
        monkeypatch, renderer._traffic_occupancy(base, renderer.RenderMetrics(scale)),
        scale=scale,
    )
    assert (marker.x, marker.y) == (100 * scale, 60 * scale)


def test_route_marker_without_traffic_is_exact_official_point(monkeypatch):
    monkeypatch.setattr(renderer, "project", lambda *args: (100, 60))
    estimate = BusEstimate("R destination", 22.3, 114.2, Operator.KMB, 0.0)
    without_traffic = renderer._merge_bus_markers([estimate], 0, 0, 0, (200, 120))
    with_empty_traffic = renderer._merge_bus_markers(
        [estimate], 0, 0, 0, (200, 120), renderer.DEFAULT_METRICS,
        renderer.TrafficOccupancy(Image.new("L", (200, 120), 0)),
    )
    assert without_traffic == with_empty_traffic
    assert (without_traffic[0].x, without_traffic[0].y) == (100, 60)


@pytest.mark.parametrize("scale", [1.0, 0.9, 0.81, 0.75])
def test_route_marker_uses_left_offset_when_own_road_has_center_hole(monkeypatch, scale):
    mask = _synthetic_route_mask(scale=scale, width=4)
    draw = renderer.ImageDraw.Draw(mask)
    draw.rectangle(
        tuple(round(value * scale) for value in (96, 58, 104, 62)), fill=0
    )  # map text-shaped hole
    marker = _merged_synthetic_bus_anchor(monkeypatch, mask, scale=scale)
    assert marker.x == pytest.approx(100 * scale)
    assert marker.y == pytest.approx((60 - 2.5) * scale)


@pytest.mark.parametrize("scale", [1.0, 0.9, 0.81, 0.75])
def test_rgb_route_core_survives_native_downscaled_mask(monkeypatch, scale):
    base = Image.new("RGB", (round(200 * scale), round(120 * scale)), (232, 238, 233))
    base.paste((247, 74, 85), mask=_synthetic_route_mask(scale=scale, width=4))
    marker = _merged_synthetic_bus_anchor(
        monkeypatch, renderer._traffic_occupancy(base, renderer.RenderMetrics(scale)),
        scale=scale,
    )
    assert (marker.x, marker.y) == pytest.approx(
        _expected_left_anchor((100, 60), 0, scale)
    )


def test_shallow_angle_crossing_cannot_gate_route_offset(monkeypatch):
    mask = _synthetic_route_mask(heading=math.radians(12), width=4)
    marker = _merged_synthetic_bus_anchor(monkeypatch, mask)
    assert (marker.x, marker.y) == (100, 60)


def test_perpendicular_crossing_cannot_gate_route_offset(monkeypatch):
    mask = _synthetic_route_mask(heading=math.pi / 2, width=4)
    marker = _merged_synthetic_bus_anchor(monkeypatch, mask)
    assert (marker.x, marker.y) == (100, 60)


def test_disconnected_route_aligned_fragments_cannot_gate_offset(monkeypatch):
    mask = Image.new("L", (200, 120), 0)
    draw = renderer.ImageDraw.Draw(mask)
    for segment in ((86, 60, 90, 60), (93, 60, 94, 60),
                    (106, 60, 108, 60), (111, 60, 114, 60)):
        draw.line(segment, fill=255, width=1)
    marker = _merged_synthetic_bus_anchor(monkeypatch, mask)
    assert (marker.x, marker.y) == (100, 60)


@pytest.mark.parametrize("shape", ["blob", "solid"])
def test_blob_or_solid_region_cannot_gate_route_offset(monkeypatch, shape):
    mask = Image.new("L", (200, 120), 0)
    draw = renderer.ImageDraw.Draw(mask)
    bounds = (84, 44, 116, 76)
    if shape == "blob":
        draw.ellipse(bounds, fill=255)
    else:
        draw.rectangle(bounds, fill=255)
    marker = _merged_synthetic_bus_anchor(monkeypatch, mask)
    assert (marker.x, marker.y) == (100, 60)


@pytest.mark.parametrize("scale", [1.0, 0.9, 0.81, 0.75])
def test_route_marker_offset_reverses_with_heading(monkeypatch, scale):
    mask = _synthetic_route_mask(scale=scale, width=4)
    eastbound = _merged_synthetic_bus_anchor(monkeypatch, mask, 0.0, scale)
    westbound = _merged_synthetic_bus_anchor(monkeypatch, mask, math.pi, scale)
    assert eastbound.y == pytest.approx((60 - 2.5) * scale)
    assert westbound.y == pytest.approx((60 + 2.5) * scale)


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
    # Attribution is consolidated and minimized at the bottom of the band.
    source = renderer._draw_legend
    import inspect
    text = inspect.getsource(source)
    assert "Map data © Google · Route geometry © Transport Department HKeMobility" in text


def test_legend_uses_actual_operator_marker_samples():
    canvas = Image.new(
        "RGBA", (renderer.LEGEND_WIDTH, renderer.LEGEND_HEIGHT),
        (246, 247, 249, 255),
    )
    renderer._draw_legend(renderer.ImageDraw.Draw(canvas, "RGBA"), canvas.size)
    assert canvas.getpixel((95, 28)) == renderer.OPERATOR_COLORS[Operator.KMB] + (245,)
    assert canvas.getpixel((220, 28)) == renderer.OPERATOR_COLORS[Operator.CITYBUS] + (245,)
    assert canvas.getpixel((345, 28)) == renderer.OPERATOR_COLORS[Operator.GMB] + (245,)
    assert canvas.getpixel((95, 48))[3] == 255  # scheduled/pale variant


def test_legend_operator_examples_form_an_aligned_table(monkeypatch):
    placements = []
    original = renderer._draw_bus_route_marker

    def tracking(draw, placement, color, font, **kwargs):
        placements.append(placement)
        return original(draw, placement, color, font, **kwargs)

    monkeypatch.setattr(renderer, "_draw_bus_route_marker", tracking)
    canvas = Image.new(
        "RGBA", (renderer.LEGEND_WIDTH, renderer.LEGEND_HEIGHT),
        (246, 247, 249, 255),
    )
    renderer._draw_legend(renderer.ImageDraw.Draw(canvas, "RGBA"), canvas.size)

    assert len(placements) == 6
    live, scheduled = placements[:3], placements[3:]
    assert [placement.operator for placement in live] == [
        Operator.KMB, Operator.CITYBUS, Operator.GMB
    ]
    assert [placement.marker[0] for placement in live] == [75, 200, 325]
    assert [placement.marker[0] for placement in scheduled] == [75, 200, 325]
    assert {placement.marker[1] for placement in live} == {30}
    assert {placement.marker[1] for placement in scheduled} == {50}
    assert [
        second - first
        for first, second in zip(
            renderer.LEGEND_ROW_CENTERS[:-1],
            renderer.LEGEND_ROW_CENTERS[1:],
            strict=True,
        )
    ] == [20, 20, 20, 20, 20]
    for first, second in zip(live, scheduled, strict=True):
        assert first.rect[0] == second.rect[0]
        assert first.rect[2] == second.rect[2]
        assert first.rect[3] < second.rect[1]

    legend = Image.open(io.BytesIO(renderer.render_legend())).convert("RGB")
    background = Image.new("RGB", legend.size, (246, 247, 249))
    content = ImageChops.difference(legend, background).getbbox()
    assert content is not None
    left, top, right, bottom = content
    assert left >= 10 and top >= 4
    assert legend.width - right >= 15
    assert legend.height - bottom >= 5


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
    assert calls[:6] == ["connector", "connector", "label", "label", "arrow", "arrow"]


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
        origin = (60 * scale, height)
        attribution_font = renderer._font(metrics.font_size(9, minimum=7))
        mask_draw = renderer.ImageDraw.Draw(mask)
        bounds = mask_draw.textbbox((0, 0), attribution, font=attribution_font)
        center_y = height + metrics.px(renderer.LEGEND_ROW_CENTERS[-1])
        text_xy = (
            origin[0] + metrics.px(10),
            center_y - (bounds[1] + bounds[3]) / 2,
        )
        mask_draw.text(text_xy, attribution, fill=255, font=attribution_font)
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
    assert len(markers) == 3
    assert [marker.routes for marker in markers] == [
        ("91 Diamond Hill",), ("91 Diamond Hill",), ("91M Po Lam",)
    ]


def test_mixed_route_support_preserves_anchors_through_label_layout(monkeypatch):
    estimates = [
        BusEstimate("R destination", lat, 114.2, Operator.KMB, 0.0)
        for lat in (1.0, 1.0, 2.0, 2.0)
    ]
    monkeypatch.setattr(
        renderer, "project",
        lambda lat, *_args: (100.0, 60.0 if lat == 1.0 else 58.0),
    )
    traffic = renderer.TrafficOccupancy(_synthetic_route_mask(width=1))

    markers = renderer._merge_bus_markers(
        estimates, 0, 0, 0, (200, 120), traffic=traffic
    )

    assert len(markers) == 4
    assert [marker.routes for marker in markers] == [("R destination",)] * 4
    assert [(marker.x, marker.y) for marker in markers] == [
        (100.0, 57.5), (100.0, 57.5),
        (100.0, 58.0), (100.0, 58.0),
    ]
    assert [marker.route_supported for marker in markers] == [True, True, False, False]

    canvas = Image.new("RGBA", (200, 120), "white")
    placements = renderer._layout_bus_labels(
        markers, renderer.ImageDraw.Draw(canvas), renderer._font(13), canvas.size
    )

    assert len(placements) == 2
    assert {placement.marker for placement in placements} == {
        (100.0, 57.5), (100.0, 58.0),
    }
    assert [len(placement.rows) for placement in placements] == [2, 2]


def test_supported_opposing_markers_keep_only_fixed_route_offset(monkeypatch):
    estimates = [
        BusEstimate("R east", 1.0, 114.2, Operator.KMB, 0.0),
        BusEstimate("R west", 1.0, 114.2, Operator.KMB, math.pi),
    ]
    monkeypatch.setattr(renderer, "project", lambda *_args: (100.0, 60.0))
    traffic = renderer.TrafficOccupancy(_synthetic_route_mask(width=4))
    markers = renderer._merge_bus_markers(
        estimates, 0, 0, 0, (200, 120), traffic=traffic
    )
    canvas = Image.new("RGBA", (200, 120), "white")
    placements = renderer._layout_bus_labels(
        markers, renderer.ImageDraw.Draw(canvas), renderer._font(13), canvas.size
    )

    assert all(marker.route_supported for marker in markers)
    assert {placement.marker for placement in placements} == {
        (100.0, 57.5), (100.0, 62.5),
    }


@pytest.mark.parametrize("scale", [1.0, 0.75])
@pytest.mark.parametrize("reliabilities", [(False, False), (False, True)])
def test_duplicate_estimates_stack_rows_but_share_one_anchor_marker(scale, reliabilities):
    from dashboard.maps.positions import BusEstimate

    size = (round(renderer.MAP_WIDTH * scale), round(renderer.MAP_HEIGHT * scale))
    metrics = renderer.RenderMetrics(scale)
    estimates = [
        BusEstimate("91 Diamond Hill", 22.334, 114.230, Operator.KMB, 0.0, unreliable=flag)
        for flag in reliabilities
    ]
    markers = renderer._merge_bus_markers(
        estimates, renderer.BASE_MAP_LAT, renderer.BASE_MAP_LON,
        renderer.BASE_MAP_ZOOM + math.log2(scale), size, metrics,
    )
    canvas = Image.new("RGBA", size, (255, 255, 255, 255))
    placements = renderer._layout_bus_labels(
        markers, renderer.ImageDraw.Draw(canvas), renderer._font(metrics.font_size(13)),
        size, metrics=metrics,
    )
    assert len(markers) == 2
    assert len(placements) == 1
    assert len(placements[0].rows) == 2
    assert [row[0] for row in placements[0].rows] == [
        "91 Diamond Hill", "91 Diamond Hill"
    ]
    assert [row[2] for row in placements[0].rows] == list(reliabilities)

    class RecordingDraw:
        def __init__(self):
            self.lines = 0
            self.polygons = 0

        def line(self, *_args, **_kwargs):
            self.lines += 1

        def polygon(self, *_args, **_kwargs):
            self.polygons += 1

    connector = RecordingDraw()
    renderer._draw_bus_route_marker(
        connector, placements[0], renderer.OPERATOR_COLORS[Operator.KMB],
        renderer._font(metrics.font_size(13)), phase="connector", metrics=metrics,
    )
    arrow = RecordingDraw()
    renderer._draw_bus_route_marker(
        arrow, placements[0], renderer.OPERATOR_COLORS[Operator.KMB],
        renderer._font(metrics.font_size(13)), phase="arrow", metrics=metrics,
    )
    assert connector.lines == 1
    assert arrow.polygons == 1


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
    assert calls == [((720, 180), 1.5, (45.0, 0.0))]


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
    assert decoded.size == (
        renderer.MIN_MAP_WIDTH,
        renderer.MIN_MAP_HEIGHT + round(renderer.LEGEND_BAND_HEIGHT * 0.75),
    )
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
