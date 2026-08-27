from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from dashboard.maps import renderer


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
