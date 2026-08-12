"""HKUST North/South Gate camera stills from the official live-view HLS feeds.

The provider performs no work at import time. Network access uses the injected
``HttpClient`` and ffmpeg is resolved only while fetching a frame.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from datetime import UTC, datetime
from typing import TypedDict
from urllib.parse import urljoin

from dashboard.http import HttpClient, as_datetime
from dashboard.models import CameraFrame
from dashboard.runtime import resolve_ffmpeg_executable

log = logging.getLogger(__name__)


class CameraStream(TypedDict):
    label: str
    playlist: str


BUS_STOP_STREAMS: tuple[CameraStream, ...] = (
    {
        "label": "North Gate bus stop",
        "playlist": (
            "https://5986dd19a5c13.streamlock.net/Northgate/"
            "ngrp:Northgate.stream_all/playlist.m3u8"
        ),
    },
    {
        "label": "South Gate bus stop",
        "playlist": (
            "https://5986dd19a5c13.streamlock.net/Southgate/"
            "ngrp:Southgate.stream_all/playlist.m3u8"
        ),
    },
)


def _pick_lowest_chunklist(playlist_text: str) -> str | None:
    """Pick the last advertised variant (the server orders high to low)."""
    lines = [line.strip() for line in playlist_text.splitlines() if line.strip()]
    lowest: str | None = None
    for index, line in enumerate(lines):
        if (
            line.startswith("#EXT-X-STREAM-INF")
            and index + 1 < len(lines)
            and not lines[index + 1].startswith("#")
        ):
            lowest = lines[index + 1]
    return lowest


def _latest_segment(playlist_text: str) -> tuple[str | None, datetime | None]:
    """Return the newest media segment and its HLS program timestamp."""
    latest: str | None = None
    latest_time: datetime | None = None
    pending_time: datetime | None = None
    for raw_line in playlist_text.splitlines():
        line = raw_line.strip()
        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            pending_time = as_datetime(line.partition(":")[2].strip())
        elif line and not line.startswith("#"):
            latest = line
            latest_time = pending_time
    return latest, latest_time


def _is_jpeg(data: bytes) -> bool:
    return len(data) >= 3 and data[:3] == b"\xff\xd8\xff"


def _decode_jpeg(executable: str, segment: bytes) -> subprocess.CompletedProcess[bytes]:
    """Decode one HLS segment outside the event loop."""
    return subprocess.run(
        [
            executable,
            "-y",
            "-i",
            "-",
            "-frames:v",
            "1",
            "-q:v",
            "4",
            "-f",
            "mjpeg",
            "-",
        ],
        input=segment,
        capture_output=True,
        check=False,
        timeout=30,
    )


async def fetch_bus_stop_frame(
    client: HttpClient,
    stream: CameraStream,
    *,
    ffmpeg_executable: str | None = None,
) -> CameraFrame | None:
    """Fetch and decode the newest HLS segment for one camera."""
    try:
        master = await client.fetch_text(stream["playlist"])
        chunklist_name = _pick_lowest_chunklist(master)
        if not chunklist_name:
            return None
        chunklist_url = urljoin(stream["playlist"], chunklist_name)
        chunklist = await client.fetch_text(chunklist_url)
        segment_name, source_time = _latest_segment(chunklist)
        if not segment_name:
            return None
        segment = await client.fetch_bytes(urljoin(chunklist_url, segment_name))
    except Exception as exc:  # noqa: BLE001
        log.warning("bus-stop camera %s fetch failed: %s", stream["label"], exc)
        return None

    try:
        executable = ffmpeg_executable or resolve_ffmpeg_executable(verify=False)
        completed = await asyncio.to_thread(_decode_jpeg, executable, segment)
    except Exception as exc:  # noqa: BLE001
        log.warning("bus-stop camera %s decode failed: %s", stream["label"], exc)
        return None
    if completed.returncode != 0 or not _is_jpeg(completed.stdout):
        log.warning("bus-stop camera %s ffmpeg returned no JPEG", stream["label"])
        return None

    # EXT-X-PROGRAM-DATE-TIME is the true frame source time when published.
    # The successful decode time is an honest fallback for feeds that omit it.
    return CameraFrame(
        data=completed.stdout,
        label=stream["label"],
        source_time=source_time or datetime.now(UTC),
    )


async def fetch_bus_stop_frames(
    client: HttpClient,
    *,
    ffmpeg_executable: str | None = None,
) -> list[CameraFrame]:
    """Fetch both camera frames concurrently, isolating individual failures."""
    results = await client.gather_any(
        [
            fetch_bus_stop_frame(
                client,
                stream,
                ffmpeg_executable=ffmpeg_executable,
            )
            for stream in BUS_STOP_STREAMS
        ]
    )
    return [result for result in results if isinstance(result, CameraFrame)]
