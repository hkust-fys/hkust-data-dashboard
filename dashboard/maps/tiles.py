"""Google Maps browser canvas base-map generation."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hashlib
import io
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from PIL import Image, ImageStat

from dashboard.models import MAP_CAPTURE_MAX_AGE_SECONDS

log = logging.getLogger(__name__)

GMAPS_BASE_URL = (
    "https://www.google.com/maps/@22.3274138,114.2331738,14z/data=!5m1!1e1?entry=ttu"
)
VIEWPORT_WIDTH = 960
VIEWPORT_HEIGHT = 540
TILE_SIZE = 256
BASE_CACHE_FILENAME = "gmaps_base_z14_960x540.png"
BASE_CACHE_METADATA_SUFFIX = ".meta.json"


def cache_filename(url: str, viewport: tuple[int, int]) -> str:
    """Return a collision-safe base cache name for a URL/viewport pair."""
    if url == GMAPS_BASE_URL and viewport == (VIEWPORT_WIDTH, VIEWPORT_HEIGHT):
        return BASE_CACHE_FILENAME
    digest = hashlib.sha256(f"{url}\0{viewport[0]}x{viewport[1]}".encode()).hexdigest()[:12]
    return f"gmaps_base_custom_{viewport[0]}x{viewport[1]}_{digest}.png"


def _cache_metadata_path(cache_path: str) -> str:
    """Return the small sidecar retaining export time separately from base age."""
    return cache_path + BASE_CACHE_METADATA_SUFFIX

# Retry transient browser failures on the next presentation window. A failed
# capture must not silently freeze the traffic layer for ten minutes.
CAPTURE_FAILURE_BACKOFF_SECONDS = 10.0
CANVAS_EXPORT_TIMEOUT_SECONDS = 5.0
# The consumer Google Maps page has been observed to keep exporting the same
# traffic bitmap while freshly loaded pages show newer traffic. Bound the age
# of that document independently of the ten-second canvas export cadence.
PAGE_REFRESH_SECONDS = 60.0
# Leave room for page loading plus the separate 10-second presentation cycle.
PAGE_WARMUP_SECONDS = 30.0
PAGE_PREPARATION_TIMEOUT_SECONDS = 30.0
CANVAS_STABILITY_INTERVAL_SECONDS = 1.0
_capture_retry_after = 0.0
_playwright_manager = None
_shared_browser = None
_shared_context = None
_shared_page = None
_capture_key: tuple[str, tuple[int, int]] | None = None
_page_loaded_at = 0.0
_page_base_updated_at: datetime | None = None
_last_capture_digest: str | None = None
_last_capture_image: Image.Image | None = None
_last_capture_identity: tuple[str, tuple[str, tuple[int, int]]] | None = None
_capture_lock: asyncio.Lock | None = None
_capture_lock_loop = None
_browser_loop = None


@dataclass(frozen=True)
class MapCapture:
    image: Image.Image | None
    captured_at: datetime | None
    stale: bool = False
    # ``captured_at`` is the timestamp of this browser canvas export.  This
    # separately records when the active Maps document was fully verified, so
    # repeated exports cannot make a stuck document look newly loaded.
    base_updated_at: datetime | None = None


@dataclass(frozen=True)
class _PreparedCapturePage:
    """A validated replacement page that has not yet replaced the active one."""

    context: object
    page: object
    key: tuple[str, tuple[int, int]]
    image: Image.Image
    loaded_at: float
    base_updated_at: datetime


_warming_task: asyncio.Task[_PreparedCapturePage] | None = None
_warming_key: tuple[str, tuple[int, int]] | None = None
_warming_loop = None
_warming_retry_after = 0.0


async def _close_shared_browser() -> None:
    """Close shared resources, retaining globals until both closes finish."""
    global _playwright_manager, _shared_browser, _browser_loop
    global _shared_context, _shared_page, _capture_key
    global _last_capture_digest, _last_capture_image, _last_capture_identity
    global _page_base_updated_at
    await _discard_warming_page()
    await _recycle_capture_page()
    _last_capture_digest = None
    _last_capture_image = None
    _last_capture_identity = None
    _page_base_updated_at = None
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
    global _shared_context, _shared_page, _capture_key, _page_loaded_at
    global _page_base_updated_at
    page, context = _shared_page, _shared_context
    _shared_page = None
    _shared_context = None
    _capture_key = None
    _page_loaded_at = 0.0
    _page_base_updated_at = None
    cancelled = await _bounded_close(page)
    context_cancelled = await _bounded_close(context)
    if cancelled or context_cancelled:
        raise asyncio.CancelledError


def _capture_digest(image: Image.Image) -> str:
    """Hash normalized RGB pixels, excluding encoder/file metadata."""
    rgb = image.convert("RGB")
    return hashlib.sha256(rgb.tobytes()).hexdigest()


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


async def _close_capture_resources(page: object | None, context: object | None) -> None:
    """Close a page/context pair, completing cleanup before cancellation escapes."""
    page_cancelled = await _bounded_close(page)
    context_cancelled = await _bounded_close(context)
    if page_cancelled or context_cancelled:
        raise asyncio.CancelledError


async def _export_page_canvas(page: object, viewport: tuple[int, int]) -> Image.Image:
    """Perform one bounded browser export from a known page."""
    data_urls = await asyncio.wait_for(
        page.evaluate(CANVAS_EXPORT_SCRIPT),  # type: ignore[attr-defined]
        timeout=CANVAS_EXPORT_TIMEOUT_SECONDS,
    )
    return _decode_first_valid_canvas(data_urls, viewport)


async def _wait_for_stable_capture_page(
    page: object, key: tuple[str, tuple[int, int]]
) -> Image.Image:
    """Reject placeholders and changing canvases before a replacement can publish."""
    await page.wait_for_selector("canvas", timeout=15000)  # type: ignore[attr-defined]
    deadline = asyncio.get_running_loop().time() + PAGE_PREPARATION_TIMEOUT_SECONDS
    previous_digest: str | None = None
    while asyncio.get_running_loop().time() < deadline:
        try:
            image = await _export_page_canvas(page, key[1])
        except ValueError:
            await asyncio.sleep(CANVAS_STABILITY_INTERVAL_SECONDS)
            continue
        digest = _capture_digest(image)
        if digest == previous_digest:
            return image
        previous_digest = digest
        await asyncio.sleep(CANVAS_STABILITY_INTERVAL_SECONDS)
    raise ValueError("Google Maps canvas did not finish rendering stably")


async def _prepare_capture_page(
    key: tuple[str, tuple[int, int]]
) -> _PreparedCapturePage:
    """Build a replacement independently, without changing the active page."""
    context: object | None = None
    page: object | None = None
    try:
        async with asyncio.timeout(PAGE_PREPARATION_TIMEOUT_SECONDS):
            browser = await _get_shared_browser()
            context = await browser.new_context(
                viewport={"width": key[1][0], "height": key[1][1]}
            )
            page = await context.new_page()
            await page.goto(key[0], wait_until="domcontentloaded", timeout=30000)
            image = await _wait_for_stable_capture_page(page, key)
            return _PreparedCapturePage(
                context=context,
                page=page,
                key=key,
                image=image,
                loaded_at=time.monotonic(),
                base_updated_at=datetime.now(UTC),
            )
    except BaseException:
        await _close_capture_resources(page, context)
        raise


async def _activate_prepared_page(prepared: _PreparedCapturePage) -> None:
    """Atomically make a validated replacement active, then retire the old page."""
    global _shared_context, _shared_page, _capture_key, _page_loaded_at
    global _page_base_updated_at
    old_page, old_context = _shared_page, _shared_context
    _shared_context = prepared.context
    _shared_page = prepared.page
    _capture_key = prepared.key
    _page_loaded_at = prepared.loaded_at
    _page_base_updated_at = prepared.base_updated_at
    await _close_capture_resources(old_page, old_context)


async def _create_capture_page(key: tuple[str, tuple[int, int]]):
    """Create the initial active page only after it has a stable viewport canvas."""
    prepared = await _prepare_capture_page(key)
    await _activate_prepared_page(prepared)
    return prepared.page


def _observe_warming_task(task: asyncio.Task[_PreparedCapturePage]) -> None:
    """Retrieve background exceptions; the next capture decides retry/swap policy."""
    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.exception()


def _start_warming_page(
    key: tuple[str, tuple[int, int]], *, force: bool = False
) -> None:
    """Begin one replacement preparation without disturbing the current page."""
    global _warming_task, _warming_key, _warming_loop
    if _warming_task is not None:
        return
    if not force and time.monotonic() < _warming_retry_after:
        return
    task = asyncio.create_task(_prepare_capture_page(key), name="gmaps-page-warmup")
    task.add_done_callback(_observe_warming_task)
    _warming_task = task
    _warming_key = key
    _warming_loop = asyncio.get_running_loop()


def _close_unpublished_warming_result(task: asyncio.Task[_PreparedCapturePage]) -> None:
    """Best-effort close for a task that finishes on its original event loop."""
    if task.cancelled():
        return
    try:
        prepared = task.result()
    except Exception:  # noqa: BLE001
        return
    try:
        asyncio.get_running_loop().create_task(
            _close_capture_resources(prepared.page, prepared.context)
        )
    except RuntimeError:
        # A closing loop will be followed by browser shutdown, which releases
        # its child contexts when no further task can safely run there.
        return


async def _discard_warming_page() -> None:
    """Cancel a pending warm page and close a completed, unpublished replacement."""
    global _warming_task, _warming_key, _warming_loop
    task, task_loop = _warming_task, _warming_loop
    if task is None:
        return
    if task is asyncio.current_task():
        # `_get_shared_browser` can recycle a disconnected browser from inside
        # the warming task. Keep this task registered so its valid replacement
        # remains available to swap after that browser recovery finishes.
        return
    _warming_task = None
    _warming_key = None
    _warming_loop = None
    task.cancel()
    current_loop = asyncio.get_running_loop()
    if task_loop is not None and task_loop is not current_loop:
        # A previous event loop owns the task. Cancelling it and closing the
        # browser below is the only safe cross-loop cleanup operation. A
        # completed result can still be closed directly; a pending task closes
        # its own partial resources or schedules this result cleanup on its
        # original loop.
        if task.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                prepared = task.result()
                await _close_capture_resources(prepared.page, prepared.context)
        else:
            task.add_done_callback(_close_unpublished_warming_result)
        return
    result = (await asyncio.gather(task, return_exceptions=True))[0]
    if isinstance(result, _PreparedCapturePage):
        await _close_capture_resources(result.page, result.context)


async def _take_ready_warming_page(
    key: tuple[str, tuple[int, int]]
) -> _PreparedCapturePage | None:
    """Swap only a completed replacement for the matching active-map identity."""
    global _warming_task, _warming_key, _warming_loop, _warming_retry_after
    task = _warming_task
    if task is None:
        return None
    if _warming_key != key:
        await _discard_warming_page()
        return None
    if not task.done():
        return None
    _warming_task = None
    _warming_key = None
    _warming_loop = None
    try:
        prepared = task.result()
    except asyncio.CancelledError:
        _warming_retry_after = time.monotonic() + CAPTURE_FAILURE_BACKOFF_SECONDS
        return None
    except Exception as exc:  # noqa: BLE001
        _warming_retry_after = time.monotonic() + CAPTURE_FAILURE_BACKOFF_SECONDS
        log.warning(
            "Google Maps replacement preparation failed (%s); retrying in %d s",
            type(exc).__name__, int(CAPTURE_FAILURE_BACKOFF_SECONDS),
        )
        return None
    if prepared.key != key:
        await _close_capture_resources(prepared.page, prepared.context)
        _warming_retry_after = time.monotonic() + CAPTURE_FAILURE_BACKOFF_SECONDS
        return None
    await _activate_prepared_page(prepared)
    _warming_retry_after = 0.0
    log.info("Google Maps replacement page is now active")
    return prepared


def _active_document_age() -> float:
    """Return the monotonic age of the active, validated Maps document."""
    if _page_base_updated_at is None:
        return float("inf")
    return max(0.0, time.monotonic() - _page_loaded_at)


def _ensure_active_page_metadata() -> datetime:
    """Support test doubles while keeping page-age and wall-clock metadata paired."""
    global _page_loaded_at, _page_base_updated_at
    if _page_loaded_at <= 0:
        _page_loaded_at = time.monotonic()
    if _page_base_updated_at is None:
        _page_base_updated_at = datetime.now(UTC)
    return _page_base_updated_at


def _persist_capture(
    cache_path: str,
    image: Image.Image,
    key: tuple[str, tuple[int, int]],
    captured_at: datetime,
    base_updated_at: datetime,
) -> tuple[str, bool]:
    """Persist the latest pixels while retaining the document's original age."""
    global _last_capture_digest, _last_capture_image, _last_capture_identity
    digest = _capture_digest(image)
    identity = (os.path.abspath(cache_path), key)
    unchanged = (
        digest == _last_capture_digest
        and _last_capture_image is not None
        and identity == _last_capture_identity
        and os.path.exists(cache_path)
    )
    if not unchanged:
        temporary_path = cache_path + ".tmp"
        image.save(temporary_path, format="PNG")
        os.replace(temporary_path, cache_path)
    # mtime deliberately records the stable document age, never the ten-second
    # export cadence. `_cached_capture` can therefore enforce the hard bound.
    os.utime(cache_path, (base_updated_at.timestamp(), base_updated_at.timestamp()))
    metadata_path = _cache_metadata_path(cache_path)
    temporary_metadata_path = metadata_path + ".tmp"
    try:
        with open(temporary_metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(
                {
                    "base_updated_at": base_updated_at.isoformat(),
                    "captured_at": captured_at.isoformat(),
                    "digest": digest,
                },
                metadata_file,
                separators=(",", ":"),
            )
        os.replace(temporary_metadata_path, metadata_path)
    except OSError as exc:
        # The PNG remains valid and bounded by its mtime. A missing sidecar is
        # conservatively reported using that base timestamp on fallback.
        with contextlib.suppress(OSError):
            os.unlink(temporary_metadata_path)
        log.warning("could not retain Google Maps export timestamp: %s", type(exc).__name__)
    _last_capture_digest = digest
    _last_capture_image = image.copy()
    _last_capture_identity = identity
    return digest, not unchanged


def _cached_export_timestamp(
    cache_path: str, image: Image.Image, base_updated_at: datetime
) -> datetime:
    """Read a matching last-export time, falling back conservatively to base age."""
    try:
        with open(_cache_metadata_path(cache_path), encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        captured_at = datetime.fromisoformat(str(metadata["captured_at"]))
        metadata_base = datetime.fromisoformat(str(metadata["base_updated_at"]))
        if captured_at.tzinfo is None or metadata_base.tzinfo is None:
            return base_updated_at
        captured_at = captured_at.astimezone(UTC)
        metadata_base = metadata_base.astimezone(UTC)
        if abs((metadata_base - base_updated_at).total_seconds()) > 1.0:
            return base_updated_at
        if metadata.get("digest") != _capture_digest(image):
            return base_updated_at
        return captured_at
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return base_updated_at


async def capture_gmaps_base(
    cache_dir: str = ".cache",
    url: str = GMAPS_BASE_URL,
    viewport: tuple[int, int] = (VIEWPORT_WIDTH, VIEWPORT_HEIGHT),
) -> MapCapture:
    """Export Google Maps' visible map canvas with its traffic layer."""
    global _capture_retry_after
    # Version the cache so a previously captured 1920x1080/zoom-15 map can
    # never be reused as the new zoom-14 base.
    cache_path = os.path.join(cache_dir, cache_filename(url, viewport))
    os.makedirs(cache_dir, exist_ok=True)
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
            await _take_ready_warming_page(key)
            if _capture_key != key or _shared_page is None or _shared_context is None:
                if _warming_task is not None and _warming_key == key:
                    # An active export just failed, but its independently
                    # prepared replacement is still useful. Keep it alive and
                    # serve only the bounded disk fallback until it is ready.
                    return _cached_capture(cache_path, viewport)
                await _discard_warming_page()
                await _recycle_capture_page()
                if time.monotonic() < _capture_retry_after:
                    return _cached_capture(cache_path, viewport)
                await _create_capture_page(key)
            base_updated_at = _ensure_active_page_metadata()
            document_age = _active_document_age()
            if document_age >= PAGE_WARMUP_SECONDS:
                _start_warming_page(key)
            if document_age >= PAGE_REFRESH_SECONDS:
                # A warming page may complete on a later presentation tick,
                # but an expired active document must never keep publishing.
                return _cached_capture(cache_path, viewport)
            image = await _export_page_canvas(_shared_page, viewport)
            captured_at = datetime.now(UTC)
            # A five-second export can cross the hard validity edge. Prefer a
            # ready replacement's already-stable sample; otherwise do not
            # publish pixels from the expired document.
            if _active_document_age() >= PAGE_REFRESH_SECONDS:
                prepared = await _take_ready_warming_page(key)
                if prepared is None:
                    return _cached_capture(cache_path, viewport)
                image = prepared.image.copy()
                captured_at = prepared.base_updated_at
                base_updated_at = prepared.base_updated_at
            digest, pixels_changed = _persist_capture(
                cache_path, image, key, captured_at, base_updated_at
            )
            _capture_retry_after = 0.0
            log.info(
                "Google map captured at=%s base_updated_at=%s sha256=%s pixels_changed=%s",
                captured_at.isoformat(), base_updated_at.isoformat(), digest[:12], pixels_changed,
            )
            return MapCapture(image, captured_at, base_updated_at=base_updated_at)
        except asyncio.CancelledError:
            await _recycle_capture_page()
            raise
        except Exception as exc:  # noqa: BLE001
            active_key = _capture_key
            if active_key == key and _shared_page is not None:
                # The replacement is prepared independently so a failed export
                # does not block future recovery behind a second cold launch.
                _start_warming_page(key, force=True)
            await _recycle_capture_page()
            if _shared_browser is not None:
                with contextlib.suppress(Exception):
                    if not _shared_browser.is_connected():
                        await _close_shared_browser()
            _capture_retry_after = time.monotonic() + CAPTURE_FAILURE_BACKOFF_SECONDS
            log.warning(
                "Playwright Google Maps canvas export failed for %s (%s); retrying in %d s, "
                "loading cache",
                url, exc, int(CAPTURE_FAILURE_BACKOFF_SECONDS),
            )
            return _cached_capture(cache_path, viewport)


def _cached_capture(
    cache_path: str, viewport: tuple[int, int]
) -> MapCapture:
    """Serve only a recent last-good base, preserving its document age."""
    if os.path.exists(cache_path):
        try:
            base_updated_at = datetime.fromtimestamp(os.path.getmtime(cache_path), UTC)
            age = (datetime.now(UTC) - base_updated_at).total_seconds()
            if not 0 <= age <= MAP_CAPTURE_MAX_AGE_SECONDS:
                return MapCapture(None, None, stale=True)
            with Image.open(cache_path) as cached:
                cached.load()
                image = _normalize_canvas_image(cached, viewport)
                captured_at = _cached_export_timestamp(
                    cache_path, image, base_updated_at
                )
                return MapCapture(
                    image,
                    captured_at,
                    stale=True,
                    base_updated_at=base_updated_at,
                )
        except Exception:  # noqa: BLE001
            pass
    return MapCapture(None, None, stale=True)


def load_tile(_cache_dir: str, _zoom: int, _x: int, _y: int) -> Image.Image | None:
    """Legacy tile loader kept for compatibility."""
    return Image.new("RGB", (TILE_SIZE, TILE_SIZE), (240, 242, 245))
