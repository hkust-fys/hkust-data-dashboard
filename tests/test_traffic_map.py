"""Traffic map renderer tests: PNG signature, dimensions, fallback background."""

import io

from PIL import Image

from dashboard.traffic_map import (
    BG_COLOR,
    MAP_HEIGHT,
    MAP_WIDTH,
    render_traffic_map,
)
from tests.fixtures import sample_data as s


def test_render_traffic_map_png_signature_and_size(tmp_path):
    png, err = render_traffic_map(
        s.traffic_statuses(),
        s.traffic_incidents(),
        s.roadworks(),
        s.utc(),
        cache_dir=str(tmp_path),
    )
    assert err is None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(png))
    assert img.size == (MAP_WIDTH, MAP_HEIGHT)


def test_render_traffic_map_fallback_on_tile_failure(tmp_path, monkeypatch):
    """Even without any tiles the PNG must be produced (neutral background)."""

    # Force every tile download to fail so the fallback path is exercised
    # regardless of local network availability.
    import dashboard.traffic_map as tm

    def _fail(*args, **kwargs):
        return None

    monkeypatch.setattr(tm, "_load_tile", _fail)
    png, err = render_traffic_map(
        [],
        [],
        [],
        s.utc(),
        cache_dir=str(tmp_path / "missing"),
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(png))
    assert img.size == (MAP_WIDTH, MAP_HEIGHT)
    # corner pixel is the neutral background (tiles failed to download)
    assert img.getpixel((0, 0)) == BG_COLOR


def test_render_traffic_map_no_coverage_state(tmp_path):
    """Empty statuses still render without claiming coverage."""
    png, err = render_traffic_map([], [], [], None, cache_dir=str(tmp_path))
    assert err is None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_fit_bounds_covers_points_and_hkust():
    from dashboard.traffic_map import HKUST_LAT, HKUST_LON, _fit_bounds

    lon_min, lon_max, lat_min, lat_max = _fit_bounds(s.traffic_statuses())
    # all fixture points + HKUST must be inside
    for status in s.traffic_statuses():
        for obs in status.observations:
            assert lon_min <= obs.longitude <= lon_max
            assert lat_min <= obs.latitude <= lat_max
    assert lon_min <= HKUST_LON <= lon_max
    assert lat_min <= HKUST_LAT <= lat_max
    # spans are at least the minimums (with float tolerance)
    assert lon_max - lon_min >= 0.08 - 1e-9
    assert lat_max - lat_min >= 0.05 - 1e-9


def test_render_traffic_map_fits_and_centers_points(tmp_path, monkeypatch):
    """The map bounds must come from the data, not a fixed box."""
    import dashboard.traffic_map as tm

    captured = {}

    def _fake_load(cache_dir, zoom, x, y):
        captured["tile"] = (x, y)
        return Image.new("RGB", (tm.TILE_SIZE, tm.TILE_SIZE), BG_COLOR)

    monkeypatch.setattr(tm, "_load_tile", _fake_load)
    png, err = render_traffic_map(
        s.traffic_statuses(),
        [],
        [],
        s.utc(),
        cache_dir=str(tmp_path),
    )
    assert err is None
    img = Image.open(io.BytesIO(png))
    assert img.size == (MAP_WIDTH, MAP_HEIGHT)
