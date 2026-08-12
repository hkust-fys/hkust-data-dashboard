"""Google Maps browser canvas base-map generation."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
import os
from contextlib import suppress

from PIL import Image

log = logging.getLogger(__name__)

GMAPS_BASE_URL = (
    "https://www.google.com/maps/@22.3274138,114.2331738,15z/data=!5m1!1e1?entry=ttu"
)
VIEWPORT_WIDTH = 1920
VIEWPORT_HEIGHT = 1080
TILE_SIZE = 256

# Google Maps changes its DOM names regularly. Find the map bitmap without
# relying on any of those names: a candidate must be visible, occupy most of
# the viewport, and permit an actual PNG canvas export. Exporting the bitmap
# (rather than screenshotting the element) excludes DOM controls layered over it.
CANVAS_EXPORT_SCRIPT = """
() => {
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;
    const viewportArea = viewportWidth * viewportHeight;
    const candidates = [];

    for (const canvas of document.querySelectorAll('canvas')) {
        const rect = canvas.getBoundingClientRect();
        const style = getComputedStyle(canvas);
        const intersectionWidth = Math.max(
            0, Math.min(rect.right, viewportWidth) - Math.max(rect.left, 0)
        );
        const intersectionHeight = Math.max(
            0, Math.min(rect.bottom, viewportHeight) - Math.max(rect.top, 0)
        );
        const visibleArea = intersectionWidth * intersectionHeight;
        if (
            style.display === 'none' ||
            style.visibility === 'hidden' ||
            Number(style.opacity) === 0 ||
            rect.width <= 0 ||
            rect.height <= 0 ||
            intersectionWidth < viewportWidth * 0.95 ||
            intersectionHeight < viewportHeight * 0.95
        ) {
            continue;
        }

        let dataUrl;
        try {
            // Canvas backing stores may be DPR-scaled and their DOM rectangle
            // may extend outside the viewport. Copy precisely the visible DOM
            // intersection into a projection-sized transparent canvas. This
            // avoids both CSS cropping and black padding without knowing any
            // Google implementation class names.
            const normalized = document.createElement('canvas');
            normalized.width = viewportWidth;
            normalized.height = viewportHeight;
            const context = normalized.getContext('2d');
            const sourceScaleX = canvas.width / rect.width;
            const sourceScaleY = canvas.height / rect.height;
            const intersectionLeft = Math.max(rect.left, 0);
            const intersectionTop = Math.max(rect.top, 0);
            const sourceX = (intersectionLeft - rect.left) * sourceScaleX;
            const sourceY = (intersectionTop - rect.top) * sourceScaleY;
            context.drawImage(
                canvas,
                sourceX, sourceY,
                intersectionWidth * sourceScaleX,
                intersectionHeight * sourceScaleY,
                intersectionLeft, intersectionTop,
                intersectionWidth, intersectionHeight
            );
            dataUrl = normalized.toDataURL('image/png');
        } catch (_error) {
            continue;
        }
        if (!dataUrl.startsWith('data:image/png;base64,') || dataUrl.length < 100) {
            continue;
        }

        // Multiple full-size canvases can coexist. The largest successful PNG
        // is the actual rendered bitmap; geometry and backing-store area only
        // break ties between equally large exports.
        const candidate = {
            dataUrl,
            exportLength: dataUrl.length,
            visibleArea,
            bitmapArea: canvas.width * canvas.height,
        };
        candidates.push(candidate);
    }
    candidates.sort((left, right) =>
        right.exportLength - left.exportLength ||
        right.visibleArea - left.visibleArea ||
        right.bitmapArea - left.bitmapArea
    );
    return candidates.map(candidate => candidate.dataUrl);
}
"""


def _canvas_candidate_rank(
    export_length: int, visible_area: float, bitmap_area: int
) -> tuple[int, float, int]:
    """Mirror the browser's deterministic canvas-candidate ordering."""
    return export_length, visible_area, bitmap_area


def _normalize_canvas_image(source: Image.Image, viewport: tuple[int, int]) -> Image.Image:
    """Normalize and validate a canvas or cached image for projection use."""
    rgba = source.convert("RGBA")
    if rgba.size != viewport:
        rgba = rgba.resize(viewport, Image.Resampling.LANCZOS)

    alpha = rgba.getchannel("A")
    histogram = alpha.histogram()
    opaque_equivalent = sum(level * count for level, count in enumerate(histogram))
    coverage = opaque_equivalent / (255 * rgba.width * rgba.height)
    if coverage < 0.90:
        raise ValueError("map canvas export was materially incomplete")
    neutral = Image.new("RGBA", rgba.size, (240, 242, 245, 255))
    neutral.alpha_composite(rgba)
    image = neutral.convert("RGB")
    sample = image.resize((128, 72), Image.Resampling.BOX)
    black_pixels = sum(1 for rgb in sample.getdata() if max(rgb) < 8)
    if black_pixels > sample.width * sample.height * 0.20:
        raise ValueError("map canvas export contained a materially black region")
    return image


def _decode_canvas_export(data_url: str, viewport: tuple[int, int]) -> Image.Image:
    """Decode a projection-sized PNG, rejecting incomplete map candidates."""
    prefix = "data:image/png;base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        raise ValueError("map canvas did not return a PNG data URL")
    try:
        payload = base64.b64decode(data_url[len(prefix) :], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("map canvas returned invalid base64") from exc

    try:
        with Image.open(io.BytesIO(payload)) as source:
            source.load()
            if source.format != "PNG":
                raise ValueError("map canvas data URL was not a PNG")
            decoded = source.copy()
    except (OSError, ValueError) as exc:
        raise ValueError("map canvas returned an invalid PNG") from exc

    return _normalize_canvas_image(decoded, viewport)


def _decode_first_valid_canvas(
    data_urls: object, viewport: tuple[int, int]
) -> Image.Image:
    """Use the largest browser-ranked candidate that passes image validation."""
    candidates = [data_urls] if isinstance(data_urls, str) else data_urls
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("no visible exportable map canvas found")
    last_error: ValueError | None = None
    for data_url in candidates:
        try:
            return _decode_canvas_export(data_url, viewport)
        except ValueError as exc:
            last_error = exc
    raise ValueError("all map canvas exports were invalid") from last_error


async def capture_gmaps_base(
    cache_dir: str = ".cache",
    url: str = GMAPS_BASE_URL,
    viewport: tuple[int, int] = (VIEWPORT_WIDTH, VIEWPORT_HEIGHT),
) -> Image.Image:
    """Export Google Maps' visible map canvas with its traffic layer."""
    cache_path = os.path.join(cache_dir, "gmaps_base.png")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(
                viewport={"width": viewport[0], "height": viewport[1]}
            )
            await page.goto(url, wait_until="commit", timeout=20000)
            with suppress(Exception):
                await page.wait_for_selector("canvas", timeout=15000)
            await asyncio.sleep(3.5)
            data_urls = await page.evaluate(CANVAS_EXPORT_SCRIPT)
            await browser.close()

        image = _decode_first_valid_canvas(data_urls, viewport)
        image.save(cache_path, format="PNG")
        return image
    except Exception as exc:  # noqa: BLE001
        log.warning("Playwright Google Maps canvas export failed (%s), loading cache", exc)
        if os.path.exists(cache_path):
            try:
                with Image.open(cache_path) as cached:
                    cached.load()
                    return _normalize_canvas_image(cached, viewport)
            except Exception:  # noqa: BLE001
                pass
        # Never fall back to a full-page screenshot: it would reintroduce UI.
        return Image.new("RGB", viewport, (240, 242, 245))


def load_tile(_cache_dir: str, _zoom: int, _x: int, _y: int) -> Image.Image | None:
    """Legacy tile loader kept for compatibility."""
    return Image.new("RGB", (TILE_SIZE, TILE_SIZE), (240, 242, 245))
