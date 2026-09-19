"""Fresh browser exports, bounded fallback, and honest displayed capture age."""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from PIL import Image

from dashboard import maps, pipeline
from dashboard.maps import tiles
from dashboard.models import TrafficIncident, TrafficMapResult
from dashboard.providers.route_geometry import RouteGeometry
from dashboard.render import _build_traffic_summary_embed


@pytest.fixture
def capture_state(monkeypatch):
    for name in (
        "_shared_browser", "_shared_page", "_shared_context", "_capture_key",
        "_last_capture_digest", "_last_capture_image", "_last_capture_identity",
        "_capture_lock", "_capture_lock_loop", "_browser_loop", "_playwright_manager",
        "_warming_task", "_warming_key", "_warming_loop",
        "_last_cache_warning", "_last_withhold_warning",
    ):
        monkeypatch.setattr(tiles, name, None)
    monkeypatch.setattr(tiles, "_capture_retry_after", 0.0)
    monkeypatch.setattr(tiles, "_warming_retry_after", 0.0)
    monkeypatch.setattr(tiles, "_page_loaded_at", 0.0)
    monkeypatch.setattr(tiles, "_page_base_updated_at", None)


def cached_map(tmp_path, age=20):
    image = Image.new("RGB", (40, 20))
    image.putdata([
        ((x * 17) % 256, (y * 29) % 256, ((x + y) * 13) % 256)
        for y in range(20) for x in range(40)
    ])
    path = tmp_path / tiles.cache_filename(tiles.GMAPS_BASE_URL, image.size)
    image.save(path)
    stamp = (datetime.now(UTC) - timedelta(seconds=age)).timestamp()
    os.utime(path, (stamp, stamp))
    return path, image


async def _resolved(value):
    return value


@pytest.mark.asyncio
async def test_failed_capture_keeps_original_timestamp_and_retries_after_ten_seconds(
    tmp_path, monkeypatch, capture_state,
):
    path, image = cached_map(tmp_path)
    timestamp = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    clock = [100.0]
    monkeypatch.setattr(tiles, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    attempts = []

    async def browser():
        attempts.append(clock[0])
        if len(attempts) == 1:
            raise RuntimeError("temporary browser outage")
        return Browser()

    class Page:
        async def goto(self, *_args, **_kwargs):
            pass

        async def wait_for_selector(self, *_args, **_kwargs):
            pass

        async def evaluate(self, _script):
            return ["valid"]

    class Context:
        async def new_page(self):
            return Page()

    class Browser:
        async def new_context(self, **_kwargs):
            return Context()

    monkeypatch.setattr(tiles, "_get_shared_browser", browser)
    monkeypatch.setattr(tiles, "_decode_first_valid_canvas", lambda *_args: image.copy())
    monkeypatch.setattr(tiles, "CANVAS_STABILITY_INTERVAL_SECONDS", 0.0)
    first = await tiles.capture_gmaps_base(str(tmp_path), viewport=image.size)
    assert first.image is not None and first.stale
    assert first.captured_at == timestamp
    clock[0] += 9
    retained = await tiles.capture_gmaps_base(str(tmp_path), viewport=image.size)
    assert retained.captured_at == timestamp and retained.stale
    assert len(attempts) == 1
    clock[0] += 1
    recovered = await tiles.capture_gmaps_base(str(tmp_path), viewport=image.size)
    assert len(attempts) == 2
    assert recovered.captured_at > timestamp and not recovered.stale
    assert recovered.base_updated_at is not None
    await tiles.shutdown_gmaps_browser()


@pytest.mark.asyncio
async def test_empty_launch_timeout_log_has_phase_safe_endpoint_and_detail(
    monkeypatch, caplog, capture_state,
):
    async def unavailable():
        raise TimeoutError()

    monkeypatch.setattr(tiles, "_get_shared_browser", unavailable)
    with pytest.raises(TimeoutError):
        await tiles._prepare_capture_page(
            ("https://www.google.com/maps?token=do-not-log", (40, 20))
        )
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "phase=launch" in messages
    assert "endpoint=https://www.google.com/maps" in messages
    assert "type=TimeoutError" in messages
    assert "detail=operation timed out" in messages
    assert "elapsed=" in messages
    assert "do-not-log" not in messages


def test_disk_fallback_expires_instead_of_showing_old_traffic(tmp_path):
    path, image = cached_map(tmp_path, age=301)
    result = tiles._cached_capture(str(path), image.size)
    assert result.image is None and result.captured_at is None
    assert result.stale


def test_expired_cache_warning_has_age_and_is_not_repeated(
    tmp_path, monkeypatch, caplog,
):
    path, image = cached_map(tmp_path, age=61)
    monkeypatch.setattr(tiles, "_last_cache_warning", None)
    first = tiles._cached_capture(str(path), image.size)
    second = tiles._cached_capture(str(path), image.size)
    assert first.image is None and second.image is None
    warnings = [
        record.getMessage()
        for record in caplog.records
        if "Google Maps cache unavailable reason=expired" in record.getMessage()
        and record.levelname == "WARNING"
    ]
    assert len(warnings) == 1
    assert "age_s=" in warnings[0]
    assert f"max_age_s={tiles.MAP_CAPTURE_MAX_AGE_SECONDS:.3f}" in warnings[0]


@pytest.mark.asyncio
async def test_capture_continues_while_replacement_warms_then_swaps_atomically(
    tmp_path, monkeypatch, capture_state,
):
    clock = [100.0]
    monkeypatch.setattr(tiles, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    class Resource:
        def __init__(self, image):
            self.image = image
            self.closed = False
            self.exports = 0

        async def close(self):
            self.closed = True

    active = Resource(Image.new("RGB", (40, 20), (0, 0, 0)))
    active_context = Resource(None)
    replacement = Resource(Image.new("RGB", (40, 20), (247, 74, 85)))
    replacement_context = Resource(None)
    key = (tiles.GMAPS_BASE_URL, active.image.size)
    old_base = datetime.now(UTC) - timedelta(seconds=46)
    new_base = datetime.now(UTC)
    tiles._shared_page = active
    tiles._shared_context = active_context
    tiles._capture_key = key
    tiles._page_loaded_at = clock[0]
    tiles._page_base_updated_at = old_base
    started, release = asyncio.Event(), asyncio.Event()

    async def export(page, _viewport):
        page.exports += 1
        return page.image.copy()

    async def prepare(prepared_key):
        assert prepared_key == key
        started.set()
        await release.wait()
        return tiles._PreparedCapturePage(
            replacement_context, replacement, key, replacement.image.copy(), clock[0], new_base,
        )

    monkeypatch.setattr(tiles, "_export_page_canvas", export)
    monkeypatch.setattr(tiles, "_prepare_capture_page", prepare)
    clock[0] = 146.0
    first = await tiles.capture_gmaps_base(str(tmp_path), viewport=active.image.size)
    await asyncio.wait_for(started.wait(), timeout=0.2)
    clock[0] = 150.0
    during_warmup = await asyncio.wait_for(
        tiles.capture_gmaps_base(str(tmp_path), viewport=active.image.size), timeout=0.2,
    )
    assert first.image.getpixel((5, 5)) == (0, 0, 0)
    assert during_warmup.image.getpixel((5, 5)) == (0, 0, 0)
    assert tiles._shared_page is active
    release.set()
    await asyncio.sleep(0)
    clock[0] = 155.0
    swapped = await tiles.capture_gmaps_base(str(tmp_path), viewport=active.image.size)
    assert swapped.image.getpixel((5, 5)) == (247, 74, 85)
    assert swapped.base_updated_at == new_base
    assert tiles._shared_page is replacement
    assert active.closed and active_context.closed
    assert active.exports == 2 and replacement.exports == 1
    await tiles.shutdown_gmaps_browser()


@pytest.mark.asyncio
async def test_twenty_second_warmup_allows_twenty_five_second_replacement_before_expiry(
    tmp_path, monkeypatch, capture_state,
):
    clock = [100.0]
    monkeypatch.setattr(tiles, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    class Resource:
        def __init__(self, image=None):
            self.image = image
            self.closed = False

        async def close(self):
            self.closed = True

    active = Resource(Image.new("RGB", (40, 20), (10, 20, 30)))
    active_context = Resource()
    replacement = Resource(Image.new("RGB", (40, 20), (210, 220, 230)))
    replacement_context = Resource()
    key = (tiles.GMAPS_BASE_URL, active.image.size)
    tiles._shared_page = active
    tiles._shared_context = active_context
    tiles._capture_key = key
    tiles._page_loaded_at = clock[0]
    tiles._page_base_updated_at = datetime.now(UTC)
    started, release = asyncio.Event(), asyncio.Event()
    attempts = []

    async def export(page, _viewport):
        return page.image.copy()

    async def prepare(prepared_key):
        attempts.append(clock[0])
        assert prepared_key == key
        started.set()
        await release.wait()
        return tiles._PreparedCapturePage(
            replacement_context,
            replacement,
            key,
            replacement.image.copy(),
            clock[0],
            datetime.now(UTC),
        )

    monkeypatch.setattr(tiles, "_export_page_canvas", export)
    monkeypatch.setattr(tiles, "_prepare_capture_page", prepare)

    clock[0] = 119.0
    await tiles.capture_gmaps_base(str(tmp_path), viewport=active.image.size)
    assert tiles._warming_task is None
    clock[0] = 120.0
    during_warmup = await tiles.capture_gmaps_base(
        str(tmp_path), viewport=active.image.size
    )
    await asyncio.wait_for(started.wait(), timeout=0.2)
    assert attempts == [120.0]
    assert during_warmup.image.getpixel((5, 5)) == (10, 20, 30)

    clock[0] = 145.0
    release.set()
    await asyncio.wait_for(asyncio.shield(tiles._warming_task), timeout=0.2)
    clock[0] = 150.0
    swapped = await tiles.capture_gmaps_base(str(tmp_path), viewport=active.image.size)
    assert swapped.image.getpixel((5, 5)) == (210, 220, 230)
    assert not swapped.stale
    assert tiles._shared_page is replacement
    assert active.closed and active_context.closed
    await tiles.shutdown_gmaps_browser()


@pytest.mark.asyncio
async def test_unresponsive_canvas_is_bounded_and_serves_labelled_cache(
    tmp_path, monkeypatch, capture_state,
):
    path, image = cached_map(tmp_path)
    cancelled = []

    class Page:
        async def evaluate(self, _script):
            try:
                await asyncio.Future()
            finally:
                cancelled.append(True)

    async def browser():
        return object()

    async def create(key):
        tiles._shared_page = Page()
        tiles._shared_context = object()
        tiles._capture_key = key

    monkeypatch.setattr(tiles, "_get_shared_browser", browser)
    monkeypatch.setattr(tiles, "_create_capture_page", create)
    monkeypatch.setattr(tiles, "CANVAS_EXPORT_TIMEOUT_SECONDS", 0.01)
    result = await asyncio.wait_for(
        tiles.capture_gmaps_base(str(tmp_path), viewport=image.size), timeout=1,
    )
    assert len(cancelled) == 1
    assert result.stale and result.image is not None
    assert result.captured_at.timestamp() == path.stat().st_mtime
    assert result.base_updated_at == result.captured_at
    await tiles.shutdown_gmaps_browser()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["changing", "invalid"])
async def test_replacement_never_swaps_changing_or_invalid_canvases(
    monkeypatch, capture_state, mode,
):
    class Resource:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    class Page(Resource):
        async def goto(self, *_args, **_kwargs):
            pass

        async def wait_for_selector(self, *_args, **_kwargs):
            pass

    class Context(Resource):
        async def new_page(self):
            return page

    class Browser:
        async def new_context(self, **_kwargs):
            return context

    active, page, context = object(), Page(), Context()
    tiles._shared_page = active
    tiles._shared_context = object()
    key = (tiles.GMAPS_BASE_URL, (40, 20))
    images = [Image.new("RGB", (40, 20), color) for color in ((5, 5, 5), (9, 9, 9))]
    calls = 0

    async def export(_page, _viewport):
        nonlocal calls
        calls += 1
        if mode == "invalid":
            raise ValueError("loading placeholder")
        return images[calls % 2].copy()

    monkeypatch.setattr(tiles, "_get_shared_browser", lambda: _resolved(Browser()))
    monkeypatch.setattr(tiles, "_export_page_canvas", export)
    monkeypatch.setattr(tiles, "PAGE_PREPARATION_TIMEOUT_SECONDS", 0.02)
    monkeypatch.setattr(tiles, "CANVAS_STABILITY_INTERVAL_SECONDS", 0.0)
    with pytest.raises((TimeoutError, ValueError)):
        await tiles._prepare_capture_page(key)
    assert calls > 1
    assert tiles._shared_page is active
    assert page.closed and context.closed


@pytest.mark.asyncio
async def test_replacement_loading_canvas_is_bounded_and_never_publishes(
    monkeypatch, capture_state,
):
    class Resource:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    class Page(Resource):
        async def goto(self, *_args, **_kwargs):
            pass

        async def wait_for_selector(self, *_args, **_kwargs):
            pass

        async def evaluate(self, _script):
            await asyncio.Future()

    class Context(Resource):
        async def new_page(self):
            return page

    class Browser:
        async def new_context(self, **_kwargs):
            return context

    active, page, context = object(), Page(), Context()
    tiles._shared_page = active
    tiles._shared_context = object()
    monkeypatch.setattr(tiles, "_get_shared_browser", lambda: _resolved(Browser()))
    monkeypatch.setattr(tiles, "CANVAS_EXPORT_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(tiles, "PAGE_PREPARATION_TIMEOUT_SECONDS", 0.04)
    with pytest.raises(TimeoutError):
        await tiles._prepare_capture_page((tiles.GMAPS_BASE_URL, (40, 20)))
    assert tiles._shared_page is active
    assert page.closed and context.closed


@pytest.mark.asyncio
async def test_hung_replacement_cannot_keep_expired_active_map_fresh(
    tmp_path, monkeypatch, capture_state,
):
    _path, image = cached_map(tmp_path, age=61)
    clock = [100.0]
    monkeypatch.setattr(tiles, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    active = SimpleNamespace(exports=0)
    key = (tiles.GMAPS_BASE_URL, image.size)
    tiles._shared_page = active
    tiles._shared_context = object()
    tiles._capture_key = key
    tiles._page_loaded_at = clock[0]
    tiles._page_base_updated_at = datetime.now(UTC) - timedelta(seconds=61)

    async def prepare(_key):
        await asyncio.Future()

    async def export(_page, _viewport):
        active.exports += 1
        return image.copy()

    monkeypatch.setattr(tiles, "_prepare_capture_page", prepare)
    monkeypatch.setattr(tiles, "_export_page_canvas", export)
    clock[0] = 161.0
    result = await tiles.capture_gmaps_base(str(tmp_path), viewport=image.size)
    assert result.image is None and result.captured_at is None and result.stale
    assert active.exports == 0
    await asyncio.sleep(0)
    await tiles.shutdown_gmaps_browser()


@pytest.mark.asyncio
async def test_export_crossing_validity_edge_does_not_publish_expired_pixels(
    tmp_path, monkeypatch, capture_state,
):
    _path, image = cached_map(tmp_path, age=61)
    clock = [100.0]
    monkeypatch.setattr(tiles, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    active = object()
    key = (tiles.GMAPS_BASE_URL, image.size)
    tiles._shared_page = active
    tiles._shared_context = object()
    tiles._capture_key = key
    tiles._page_loaded_at = clock[0]
    tiles._page_base_updated_at = datetime.now(UTC) - timedelta(seconds=59)

    async def export(_page, _viewport):
        clock[0] = 164.0
        return image.copy()

    monkeypatch.setattr(tiles, "_export_page_canvas", export)
    monkeypatch.setattr(tiles, "_start_warming_page", lambda *_args, **_kwargs: None)
    clock[0] = 159.0
    result = await tiles.capture_gmaps_base(str(tmp_path), viewport=image.size)
    assert result.image is None and result.captured_at is None and result.stale


@pytest.mark.asyncio
async def test_failed_active_export_keeps_matching_warm_replacement_alive(
    tmp_path, monkeypatch, capture_state,
):
    _path, image = cached_map(tmp_path, age=10)
    clock = [100.0]
    monkeypatch.setattr(tiles, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    class Resource:
        def __init__(self, image=None):
            self.image = image
            self.closed = False

        async def close(self):
            self.closed = True

    active, active_context = Resource(), Resource()
    warm_image = image.copy()
    warm_image.putpixel((5, 5), (250, 20, 20))
    warm, warm_context = Resource(warm_image), Resource()
    key = (tiles.GMAPS_BASE_URL, image.size)
    tiles._shared_page = active
    tiles._shared_context = active_context
    tiles._capture_key = key
    tiles._page_loaded_at = clock[0]
    tiles._page_base_updated_at = datetime.now(UTC) - timedelta(seconds=10)
    started, release = asyncio.Event(), asyncio.Event()

    async def export(page, _viewport):
        if page is active:
            raise RuntimeError("active canvas failed")
        return page.image.copy()

    async def prepare(_key):
        started.set()
        await release.wait()
        return tiles._PreparedCapturePage(
            warm_context, warm, key, warm_image.copy(), clock[0], datetime.now(UTC),
        )

    monkeypatch.setattr(tiles, "_export_page_canvas", export)
    monkeypatch.setattr(tiles, "_prepare_capture_page", prepare)
    first = await tiles.capture_gmaps_base(str(tmp_path), viewport=image.size)
    assert first.stale and first.image is not None
    await asyncio.wait_for(started.wait(), timeout=0.2)
    warm_task = tiles._warming_task
    clock[0] = 101.0
    retained = await tiles.capture_gmaps_base(str(tmp_path), viewport=image.size)
    assert retained.stale and retained.image is not None
    assert tiles._warming_task is warm_task and not warm_task.done()
    release.set()
    await asyncio.sleep(0)
    clock[0] = 102.0
    recovered = await tiles.capture_gmaps_base(str(tmp_path), viewport=image.size)
    assert recovered.image.getpixel((5, 5)) == (250, 20, 20)
    assert not recovered.stale and active.closed and active_context.closed
    await tiles.shutdown_gmaps_browser()


@pytest.mark.asyncio
async def test_failed_replacement_retries_after_ten_seconds(
    tmp_path, monkeypatch, capture_state,
):
    clock = [100.0]
    monkeypatch.setattr(tiles, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    image = Image.new("RGB", (40, 20), (0, 0, 0))
    replacement = Image.new("RGB", (40, 20), (220, 20, 20))
    active = SimpleNamespace(image=image, exports=0)
    key = (tiles.GMAPS_BASE_URL, image.size)
    tiles._shared_page = active
    tiles._shared_context = object()
    tiles._capture_key = key
    tiles._page_loaded_at = clock[0]
    tiles._page_base_updated_at = datetime.now(UTC) - timedelta(seconds=46)
    attempts = []

    async def export(page, _viewport):
        page.exports += 1
        return page.image.copy()

    async def prepare(_key):
        attempts.append(clock[0])
        if len(attempts) == 1:
            raise RuntimeError("replacement unavailable")
        warm_page = SimpleNamespace(image=replacement, exports=0)
        return tiles._PreparedCapturePage(
            object(), warm_page, key, replacement.copy(), clock[0], datetime.now(UTC),
        )

    monkeypatch.setattr(tiles, "_export_page_canvas", export)
    monkeypatch.setattr(tiles, "_prepare_capture_page", prepare)
    clock[0] = 146.0
    await tiles.capture_gmaps_base(str(tmp_path), viewport=image.size)
    await asyncio.sleep(0)
    clock[0] = 147.0
    await tiles.capture_gmaps_base(str(tmp_path), viewport=image.size)
    clock[0] = 156.0
    await tiles.capture_gmaps_base(str(tmp_path), viewport=image.size)
    assert attempts == [146.0]
    clock[0] = 157.0
    await tiles.capture_gmaps_base(str(tmp_path), viewport=image.size)
    await asyncio.sleep(0)
    clock[0] = 158.0
    result = await tiles.capture_gmaps_base(str(tmp_path), viewport=image.size)
    assert attempts == [146.0, 157.0]
    assert result.image.getpixel((5, 5)) == (220, 20, 20)
    await tiles.shutdown_gmaps_browser()


@pytest.mark.asyncio
async def test_shutdown_cancels_and_closes_warming_page_resources(
    tmp_path, monkeypatch, capture_state,
):
    class Resource:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    class WarmPage(Resource):
        async def goto(self, *_args, **_kwargs):
            pass

        async def wait_for_selector(self, *_args, **_kwargs):
            pass

    class WarmContext(Resource):
        async def new_page(self):
            return warm_page

    class Browser:
        async def new_context(self, **_kwargs):
            return warm_context

    clock = [100.0]
    monkeypatch.setattr(tiles, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    active, active_context = Resource(), Resource()
    warm_page, warm_context = WarmPage(), WarmContext()
    image = Image.new("RGB", (40, 20))
    key = (tiles.GMAPS_BASE_URL, image.size)
    tiles._shared_page = active
    tiles._shared_context = active_context
    tiles._capture_key = key
    tiles._page_loaded_at = clock[0]
    tiles._page_base_updated_at = datetime.now(UTC) - timedelta(seconds=46)
    warming_started = asyncio.Event()

    async def export(page, _viewport):
        if page is active:
            return image.copy()
        warming_started.set()
        await asyncio.Future()

    monkeypatch.setattr(tiles, "_get_shared_browser", lambda: _resolved(Browser()))
    monkeypatch.setattr(tiles, "_export_page_canvas", export)
    clock[0] = 146.0
    await tiles.capture_gmaps_base(str(tmp_path), viewport=image.size)
    await asyncio.wait_for(warming_started.wait(), timeout=0.2)
    await tiles.shutdown_gmaps_browser()
    assert warm_page.closed and warm_context.closed
    assert active.closed and active_context.closed
    assert tiles._warming_task is None and tiles._shared_page is None


@pytest.mark.asyncio
async def test_warming_task_remains_registered_during_its_browser_recovery(
    capture_state,
):
    key = (tiles.GMAPS_BASE_URL, (40, 20))

    async def warming_task():
        await tiles._discard_warming_page()

    task = asyncio.create_task(warming_task())
    tiles._warming_task = task
    tiles._warming_key = key
    tiles._warming_loop = asyncio.get_running_loop()
    await task
    assert tiles._warming_task is task
    assert tiles._warming_key == key
    tiles._warming_task = None
    tiles._warming_key = None
    tiles._warming_loop = None


@pytest.mark.asyncio
async def test_browser_is_not_relaunched_for_each_ten_second_export(
    tmp_path, monkeypatch, capture_state,
):
    clock = [100.0]
    monkeypatch.setattr(tiles, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    image = Image.new("RGB", (40, 20), (20, 40, 60))
    launches = []

    class Page:
        exports = 0

        async def goto(self, *_args, **_kwargs):
            pass

        async def wait_for_selector(self, *_args, **_kwargs):
            pass

        async def evaluate(self, _script):
            self.exports += 1
            return ["valid"]

    page = Page()

    class Context:
        async def new_page(self):
            return page

    class Browser:
        async def new_context(self, **_kwargs):
            return Context()

    async def browser():
        launches.append(clock[0])
        return Browser()

    monkeypatch.setattr(tiles, "_get_shared_browser", browser)
    monkeypatch.setattr(tiles, "_decode_first_valid_canvas", lambda *_args: image.copy())
    monkeypatch.setattr(tiles, "CANVAS_STABILITY_INTERVAL_SECONDS", 0.0)
    for moment in (100.0, 110.0, 120.0, 130.0):
        clock[0] = moment
        await tiles.capture_gmaps_base(str(tmp_path), viewport=image.size)
    assert launches == [100.0]
    assert page.exports == 6  # two stability samples plus four presenter exports
    await tiles.shutdown_gmaps_browser()


@pytest.mark.asyncio
async def test_capture_metadata_survives_render_and_payload(monkeypatch):
    timestamp = datetime.now(UTC) - timedelta(seconds=5)
    image = Image.new("RGB", (40, 20))

    async def capture(**_kwargs):
        return tiles.MapCapture(image, timestamp)

    async def geometry(*_args, **_kwargs):
        return RouteGeometry()

    def render(*args):
        assert args[4] is image
        return b"webp"

    monkeypatch.setattr(maps, "capture_gmaps_base", capture)
    monkeypatch.setattr(maps, "fetch_route_geometry", geometry)
    monkeypatch.setattr(maps, "render_map", render)
    before_render = datetime.now(UTC)
    result = await maps.fetch_traffic_map(object())
    assert result.webp == b"webp" and result.captured_at == timestamp
    assert before_render <= result.markers_refreshed_at <= datetime.now(UTC)
    first = pipeline.to_payload({"traffic_map": result})
    second = pipeline.to_payload({"traffic_map": result})
    assert first.embeds[0].timestamp == second.embeds[0].timestamp == timestamp
    assert first.files[0].source_time == timestamp
    assert f"<t:{int(timestamp.timestamp())}:T>" in first.embeds[0].description
    marker_label = f"Markers refreshed <t:{int(result.markers_refreshed_at.timestamp())}:T>"
    assert marker_label in first.embeds[0].description
    assert marker_label in second.embeds[0].description
    assert "Cached map" not in first.embeds[0].description


def test_retained_map_ages_without_being_relabeled_as_a_new_capture():
    timestamp = datetime.now(UTC) - timedelta(seconds=40)
    markers_time = datetime.now(UTC)
    payload = pipeline.to_payload({"traffic_map": TrafficMapResult(
        b"webp", timestamp, markers_refreshed_at=markers_time,
    )})
    assert payload.embeds[0].timestamp == timestamp
    assert "Cached map" in payload.embeds[0].description
    expired = datetime.now(UTC) - timedelta(seconds=301)
    payload = pipeline.to_payload({"traffic_map": TrafficMapResult(
        b"webp", expired, markers_refreshed_at=markers_time,
    )})
    assert not payload.files
    assert payload.embeds[0].title == "Traffic map unavailable"


def test_empty_news_does_not_imply_clear_roads_or_invent_report_time():
    timestamp = datetime.now(UTC)
    embed = _build_traffic_summary_embed(
        [], [], None, traffic_source_times={"traffic_news_checked": timestamp},
    )
    assert embed.timestamp is None
    assert "No current TD notices match" in embed.description
    assert "do not cover every traffic jam" in embed.description
    assert embed.color.value == 0x64748B


def test_unavailable_news_is_not_described_as_retained_data():
    embed = _build_traffic_summary_embed(
        [], [], None, stale_sources=["TD traffic news unavailable"],
    )
    assert embed.timestamp is None
    assert "not been checked successfully" in embed.description
    assert "retained data" not in embed.description
    assert "Stale source cache" not in embed.description


def test_cleared_chinese_report_remains_attributed_without_incident_map_rails():
    from dashboard.providers.tracked_roads import fallback_roads

    timestamp = datetime.now(UTC)
    report = TrafficIncident(
        identifier="rthk-cleared", title="交通消息",
        description="清水灣道往西貢方向的交通意外已清理，行車線重開。",
        road="", location="", direction="", status="CLOSED",
        source="RTHK", announcement_time=timestamp,
    )
    roads = fallback_roads()
    embed = _build_traffic_summary_embed(
        [], [report], None, roads=roads,
        traffic_source_times={"rthk_news_checked": timestamp},
    )
    assert "**RTHK**" in embed.description
    assert report.description in embed.description
    assert "清水灣道 (Clear Water Bay Road)" in embed.description
    assert "Cleared / 重開" in embed.description
    assert embed.color.value == 0x64748B
    assert embed.footer.text == "TD · RTHK traffic reports"
    assert pipeline.map_road_paths_from_results(([], [report], []), roads)[0] == []
