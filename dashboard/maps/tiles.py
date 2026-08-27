"""Google Maps browser canvas base-map generation."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hashlib
import io
import logging
import os
import time

from PIL import Image, ImageStat

log = logging.getLogger(__name__)

GMAPS_BASE_URL = (
    "https://www.google.com/maps/@22.3274138,114.2331738,14z/data=!5m1!1e1?entry=ttu"
)
VIEWPORT_WIDTH = 960
VIEWPORT_HEIGHT = 540
TILE_SIZE = 256
BASE_CACHE_FILENAME = "gmaps_base_z14_960x540.png"


def cache_filename(url: str, viewport: tuple[int, int]) -> str:
    """Return a collision-safe base cache name for a URL/viewport pair."""
    if url == GMAPS_BASE_URL and viewport == (VIEWPORT_WIDTH, VIEWPORT_HEIGHT):
        return BASE_CACHE_FILENAME
    digest = hashlib.sha256(f"{url}\0{viewport[0]}x{viewport[1]}".encode()).hexdigest()[:12]
    return f"gmaps_base_custom_{viewport[0]}x{viewport[1]}_{digest}.png"

# Repeated launch failures (dead system proxy/PAC) would otherwise warn on
# every presenter tick. Back off before trying again; the cached base map
# keeps rendering meanwhile.
CAPTURE_FAILURE_BACKOFF_SECONDS = 10 * 60.0
_capture_retry_after = 0.0
_playwright_manager = None
_shared_browser = None
_shared_context = None
_shared_page = None
_capture_key: tuple[str, tuple[int, int]] | None = None
_last_capture_digest: str | None = None
_last_capture_image: Image.Image | None = None
_last_capture_identity: tuple[str, tuple[str, tuple[int, int]]] | None = None
_capture_lock: asyncio.Lock | None = None
_capture_lock_loop = None
_browser_loop = None


async def _close_shared_browser() -> None:
    """Close shared resources, retaining globals until both closes finish."""
    global _playwright_manager, _shared_browser, _browser_loop
    global _shared_context, _shared_page, _capture_key
    global _last_capture_digest, _last_capture_image, _last_capture_identity
    await _recycle_capture_page()
    _last_capture_digest = None
    _last_capture_image = None
    _last_capture_identity = None
    browser, manager = _shared_browser, _playwright_manager
    if browser is not None:
        with contextlib.suppress(Exception):
            await browser.close()
    if manager is not None:
        with contextlib.suppress(Exception):
            await manager.stop()
    _shared_browser = None
    _playwright_manager = None
    _browser_loop = None


async def _get_shared_browser():
    """Start Playwright/Chromium once and reuse it across map captures."""
    global _playwright_manager, _shared_browser, _browser_loop
    current_loop = asyncio.get_running_loop()
    if _browser_loop is not None and _browser_loop is not current_loop:
        await _close_shared_browser()
    if _shared_browser is not None:
        try:
            if _shared_browser.is_connected():
                return _shared_browser
        except Exception:  # noqa: BLE001
            pass
        await _close_shared_browser()
    from playwright.async_api import async_playwright

    manager = async_playwright()
    _playwright_manager = await manager.start()
    try:
        _shared_browser = await _playwright_manager.chromium.launch(
            headless=True,
            args=["--no-proxy-server"],
        )
    except Exception:
        await _playwright_manager.stop()
        _playwright_manager = None
        raise
    _browser_loop = current_loop
    return _shared_browser


async def shutdown_gmaps_browser() -> None:
    """Idempotently close the shared browser and allow later reinitialization."""
    global _capture_lock, _capture_lock_loop
    loop = asyncio.get_running_loop()
    if _capture_lock is None or _capture_lock_loop is not loop:
        if _capture_lock_loop is not None and _capture_lock_loop is not loop:
            await _close_shared_browser()
        _capture_lock = asyncio.Lock()
        _capture_lock_loop = loop
    async with _capture_lock:
        await _close_shared_browser()


async def _bounded_close(resource) -> bool:
    """Close a Playwright resource, returning whether cancellation occurred."""
    if resource is None:
        return False
    close = getattr(resource, "close", None)
    if close is None:
        return False
    task = asyncio.create_task(close())
    cancelled = False
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
    except asyncio.CancelledError:
        cancelled = True
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        except Exception:
            await asyncio.gather(task, return_exceptions=True)
    except Exception:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return cancelled


async def _recycle_capture_page() -> None:
    """Close the persistent page/context, retaining the browser process."""
    global _shared_context, _shared_page, _capture_key
    page, context = _shared_page, _shared_context
    _shared_page = None
    _shared_context = None
    _capture_key = None
    cancelled = await _bounded_close(page)
    context_cancelled = await _bounded_close(context)
    if cancelled or context_cancelled:
        raise asyncio.CancelledError


def _capture_digest(image: Image.Image) -> str:
    """Hash normalized RGB pixels, excluding encoder/file metadata."""
    rgb = image.convert("RGB")
    return hashlib.sha256(rgb.tobytes()).hexdigest()


async def _create_capture_page(key: tuple[str, tuple[int, int]]):
    """Create, navigate, and settle a page, tolerating loading placeholders."""
    global _shared_context, _shared_page, _capture_key
    browser = await _get_shared_browser()
    _shared_context = await browser.new_context(
        viewport={"width": key[1][0], "height": key[1][1]}
    )
    _shared_page = await _shared_context.new_page()
    _capture_key = key
    await _shared_page.goto(key[0], wait_until="domcontentloaded", timeout=30000)
    await _shared_page.wait_for_selector("canvas", timeout=15000)
    deadline = asyncio.get_running_loop().time() + 25.0
    while asyncio.get_running_loop().time() < deadline:
        try:
            data_urls = await _shared_page.evaluate(CANVAS_EXPORT_SCRIPT)
            _decode_first_valid_canvas(data_urls, key[1])
        except ValueError:
            await asyncio.sleep(1.0)
            continue
        await asyncio.sleep(1.5)
        return _shared_page
    raise ValueError("Google Maps canvas did not finish rendering")

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
    quantized_colors = {
        (red // 16, green // 16, blue // 16)
        for red, green, blue in sample.getdata()
    }
    luminance_spread = ImageStat.Stat(sample.convert("L")).stddev[0]
    if len(quantized_colors) < 24 or luminance_spread < 4:
        raise ValueError("map canvas export was a low-information loading placeholder")
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
    global _capture_retry_after, _shared_context, _shared_page, _capture_key
    global _last_capture_digest, _last_capture_image, _last_capture_identity
    # Version the cache so a previously captured 1920x1080/zoom-15 map can
    # never be reused as the new zoom-14 base.
    cache_path = os.path.join(cache_dir, cache_filename(url, viewport))
    os.makedirs(cache_dir, exist_ok=True)
    if time.monotonic() < _capture_retry_after:
        return _cached_or_placeholder(cache_path, viewport)
    global _capture_lock, _capture_lock_loop
    loop = asyncio.get_running_loop()
    if _capture_lock is None or _capture_lock_loop is not loop:
        if _capture_lock_loop is not None and _capture_lock_loop is not loop:
            await _close_shared_browser()
        _capture_lock = asyncio.Lock()
        _capture_lock_loop = loop
    async with _capture_lock:
        key = (url, viewport)
        try:
            await _get_shared_browser()
            if _capture_key != key or _shared_page is None or _shared_context is None:
                await _recycle_capture_page()
                await _create_capture_page(key)

            # A crashed page can be recovered once without throwing away the
            # browser process. This also handles transient evaluation errors.
            for attempt in range(2):
                try:
                    data_urls = await _shared_page.evaluate(CANVAS_EXPORT_SCRIPT)
                    image = _decode_first_valid_canvas(data_urls, viewport)
                    break
                except asyncio.CancelledError:
                    raise
                except Exception:
                    if attempt:
                        raise
                    await _recycle_capture_page()
                    await _create_capture_page(key)
            digest = _capture_digest(image)
            identity = (os.path.abspath(cache_path), key)
            if digest == _last_capture_digest and _last_capture_image is not None and identity == _last_capture_identity:
                return _last_capture_image.copy()
            temporary_path = cache_path + ".tmp"
            image.save(temporary_path, format="PNG")
            os.replace(temporary_path, cache_path)
            _last_capture_digest = digest
            _last_capture_image = image.copy()
            _last_capture_identity = identity
            return image
        except asyncio.CancelledError:
            await _recycle_capture_page()
            raise
        except Exception as exc:  # noqa: BLE001
            await _recycle_capture_page()
            if _shared_browser is not None:
                with contextlib.suppress(Exception):
                    if not _shared_browser.is_connected():
                        await _close_shared_browser()
            _capture_retry_after = time.monotonic() + CAPTURE_FAILURE_BACKOFF_SECONDS
            log.warning(
                "Playwright Google Maps canvas export failed for %s (%s); backing off %d min, "
                "loading cache",
                url, exc, int(CAPTURE_FAILURE_BACKOFF_SECONDS / 60),
            )
            return _cached_or_placeholder(cache_path, viewport)


def _cached_or_placeholder(
    cache_path: str, viewport: tuple[int, int]
) -> Image.Image:
    """Last-good base map, or a neutral placeholder when nothing is cached."""
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
