from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image, ImageDraw

from dashboard.maps import renderer
from dashboard.models import Operator


def _base(size=(32, 20), color=(80, 90, 100)):
    return Image.new("RGB", size, color)


def setup_function():
    renderer._clear_renderer_caches()


def test_traffic_cache_hits_identical_final_size_pixels(monkeypatch):
    calls = []
    original = renderer._traffic_occupancy

    def counted(image, metrics=renderer.DEFAULT_METRICS):
        calls.append(image.size)
        return original(image, metrics)

    monkeypatch.setattr(renderer, "_traffic_occupancy", counted)
    first = renderer._cached_traffic_occupancy(_base())
    second = renderer._cached_traffic_occupancy(_base())

    assert first is second
    assert calls == [(32, 20)]
    stats = renderer._renderer_cache_stats()
    assert stats["traffic_hits"] == 1
    assert stats["traffic_misses"] == 1


def test_traffic_cache_misses_changed_pixels_and_size(monkeypatch):
    calls = []
    original = renderer._traffic_occupancy
    monkeypatch.setattr(
        renderer,
        "_traffic_occupancy",
        lambda image, metrics=renderer.DEFAULT_METRICS: (calls.append(image.size) or original(image, metrics)),
    )

    renderer._cached_traffic_occupancy(_base())
    changed = _base()
    changed.putpixel((0, 0), (81, 90, 100))
    renderer._cached_traffic_occupancy(changed)
    renderer._cached_traffic_occupancy(_base((31, 20)))

    assert calls == [(32, 20), (32, 20), (31, 20)]
    assert renderer._renderer_cache_stats()["traffic_misses"] == 3


def test_traffic_cache_is_bounded_and_concurrent():
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(renderer._cached_traffic_occupancy, (_base((24 + i, 20)) for i in range(12))))

    stats = renderer._renderer_cache_stats()
    assert stats["traffic_entries"] <= renderer._TRAFFIC_CACHE_LIMIT
    assert stats["traffic_evictions"] >= 1


def test_known_google_legend_and_canvas_colors_exclude_unrelated_saturated_colors():
    swatches = (*renderer.GOOGLE_TRAFFIC_LEGEND_COLORS, *renderer.GOOGLE_TRAFFIC_COLORS)
    base = _base((len(swatches) * 20 + 80, 30))
    draw = ImageDraw.Draw(base)
    for index, color in enumerate(swatches):
        # Include small anti-aliasing/resize differences from the solid cores.
        shifted = tuple(min(255, channel + 7) for channel in color)
        draw.rectangle((index * 20 + 4, 10, index * 20 + 12, 18), fill=shifted)
    offset = len(swatches) * 20
    for index, color in enumerate(((0, 255, 0), (255, 0, 0), (202, 239, 211))):
        draw.rectangle((offset + index * 20 + 4, 10, offset + index * 20 + 12, 18), fill=color)
    traffic = renderer._cached_traffic_occupancy(base)
    for index in range(len(swatches)):
        assert traffic.overlap((index * 20 + 5, 11, index * 20 + 11, 17))[0] > 0
    for index in range(3):
        assert traffic.overlap((offset + index * 20 + 5, 11, offset + index * 20 + 11, 17))[0] == 0


def test_quality_hint_is_reused_then_periodically_probed_upward():
    size = (960, 540)
    renderer._remember_quality(size, 74, 80_000)
    assert renderer._quality_candidates(size) == (74, 70, 65, 60)
    for _ in range(renderer._QUALITY_PROBE_INTERVAL - 2):
        renderer._quality_candidates(size)
    candidates = renderer._quality_candidates(size)

    assert candidates == (78, 74, 70, 65, 60)
    assert renderer._renderer_cache_stats()["quality_probes"] == 1
    assert renderer._quality_candidates(size) == (74, 70, 65, 60)


def test_quality_hint_does_not_probe_without_headroom():
    size = (864, 486)
    renderer._remember_quality(size, 74, 99_000)
    for _ in range(renderer._QUALITY_PROBE_INTERVAL + 1):
        candidates = renderer._quality_candidates(size)
    assert candidates == (74, 70, 65, 60)
    assert renderer._renderer_cache_stats()["quality_probes"] == 0


def test_no_exact_render_cache_can_reuse_stale_overlays():
    # Traffic occupancy is intentionally the only derived cache. There is no
    # final-render cache whose signature could accidentally omit live ETAs.
    assert not hasattr(renderer, "_render_cache")


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("color", renderer.GOOGLE_TRAFFIC_COLORS)
def test_refreshed_traffic_pixels_move_labels_and_cleared_pixels_release_space(grouped, color):
    class FixedWidthDraw:
        def textlength(self, _text, font=None):
            return 60

    base = _base((320, 200))
    markers = [renderer.BusMarker(("91",), 130, 100, Operator.KMB, 0)]
    if grouped:
        markers.append(renderer.BusMarker(("11",), 130, 100, Operator.GMB, 0))

    def place(image):
        traffic = renderer._cached_traffic_occupancy(image)
        placement = renderer._layout_bus_labels(
            markers, FixedWidthDraw(), renderer._font(13), image.size,
            traffic=traffic,
        )[0]
        return placement, traffic

    original, original_mask = place(base)
    # Simulate a refreshed Google frame that introduces a traffic stroke at
    # yesterday's chosen label location, including mutation of the same image.
    ImageDraw.Draw(base).rectangle(original.rect, fill=color)
    updated, updated_mask = place(base)
    assert updated_mask is not original_mask
    assert updated_mask.overlap(original.rect)[0] > 0
    assert updated.rect != original.rect
    assert updated.marker == original.marker
    assert updated_mask.overlap(updated.rect)[0] == 0

    # When the stroke disappears, no retained traffic mask may keep steering
    # labels away from space that is now clear.
    base.paste((80, 90, 100), (0, 0, *base.size))
    cleared, cleared_mask = place(base)
    assert cleared_mask.overlap(original.rect)[0] == 0
    assert cleared.rect == original.rect
