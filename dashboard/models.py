"""Provider-neutral data models shared across the dashboard.

Providers return these dataclasses; the renderer consumes them. Nothing here
depends on aiohttp or discord.py so the models are cheap to import and test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

# --------------------------------------------------------------------------
# Freshness / errors
# --------------------------------------------------------------------------

@dataclass
class ProviderError:
    """A provider failed; carry a human-readable reason."""

    message: str


@dataclass
class ProviderResult:
    """Wrapper carrying the payload plus freshness/error metadata.

    ``value`` may be ``None`` while ``error`` is set, or vice versa. When
    ``stale`` is True the value is the last successful fetch and ``source_time``
    reflects when that fetch happened.
    """

    value: object | None
    fetched_at: datetime
    source_time: datetime | None = None
    error: ProviderError | None = None
    stale: bool = False

    @property
    def is_ok(self) -> bool:
        return self.error is None and self.value is not None

    def age_seconds(self) -> float:
        return time.time() - self.fetched_at.timestamp()


# --------------------------------------------------------------------------
# Transit
# --------------------------------------------------------------------------

class EtaKind(StrEnum):
    """Semantic classification of an ETA entry."""

    REALTIME = "realtime"
    SCHEDULED = "scheduled"
    DELAYED = "delayed"
    MOVING_SLOWLY = "moving_slowly"
    UNAVAILABLE = "unavailable"


class Operator(StrEnum):
    KMB = "KMB"
    CITYBUS = "Citybus"
    GMB = "GMB"


@dataclass
class EtaRow:
    """One departure estimate at a stop."""

    route: str
    destination: str
    gate: str  # "N" or "S"
    operator: Operator
    minutes: int | None  # None when unavailable
    kind: EtaKind = EtaKind.REALTIME
    eta_time: datetime | None = None
    source_time: datetime | None = None
    stop_seq: int | None = None  # GMB loop stop index; None for KMB/Citybus
    bound: str | None = None  # official direction ("outbound"/"seq-1"...)


@dataclass
class RouteEtaGroup:
    """Stable display group: a route at a gate with its ETAs."""

    route: str
    destination: str
    gate: str
    operator: Operator
    rows: list[EtaRow] = field(default_factory=list)
    stop_seq: int | None = None  # GMB loop stop index; None for KMB/Citybus
    bound: str | None = None  # official direction, when derivable


# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------

@dataclass
class WeatherSnapshot:
    """Observations for the Sai Kung area."""

    temperature_c: float | None = None
    rainfall_mm: float | None = None
    humidity_pct: int | None = None
    station: str = "Sai Kung"
    source_time: datetime | None = None
    stale: bool = False


@dataclass
class WeatherWarning:
    """One active HKO warning/signal."""

    code: str
    name: str
    summary: str = ""
    action: str = ""
    icon_url: str = ""
    issued_at: datetime | None = None


@dataclass
class WeatherConditions:
    """Full weather payload: active warnings plus observations."""

    warnings: list[WeatherWarning] = field(default_factory=list)
    snapshot: WeatherSnapshot | None = None
    warning_time: datetime | None = None
    stale: bool = False


# --------------------------------------------------------------------------
# Traffic
# --------------------------------------------------------------------------

class SpeedBand(StrEnum):
    """Dashboard speed bands (heuristics, not TD classifications)."""

    RED = "red"  # < 20 km/h
    AMBER = "amber"  # 20–40 km/h
    GREEN = "green"  # > 40 km/h
    GRAY = "gray"  # missing/stale observation


@dataclass
class TrafficObservation:
    """One detector observation for a corridor."""

    corridor: str
    direction: str
    description: str
    latitude: float
    longitude: float
    speed_kmh: float | None
    volume: int | None
    occupancy_pct: float | None
    capture_time: datetime | None
    band: SpeedBand = SpeedBand.GRAY
    stale: bool = False


@dataclass
class TrafficCorridorStatus:
    """Summarized status for one matched corridor."""

    name: str
    direction: str
    observations: list[TrafficObservation] = field(default_factory=list)
    capture_time: datetime | None = None
    source: str = "TD detector"


@dataclass
class TrafficIncident:
    """A TD Special Traffic News notice relevant to our corridors."""

    identifier: str
    title: str
    description: str
    road: str
    location: str
    direction: str
    status: str  # raw status string from TD
    start_time: datetime | None = None
    end_time: datetime | None = None
    announcement_time: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    near_landmark: str = ""
    between_landmark: str = ""


@dataclass
class Roadwork:
    """A TD roadworks feature matched to our corridors."""

    identifier: str
    description: str
    road: str
    start_time: datetime | None = None
    end_time: datetime | None = None


# --------------------------------------------------------------------------
# Images / payload
# --------------------------------------------------------------------------

@dataclass
class ImageAsset:
    """An in-memory image ready to attach to a Discord message."""

    filename: str
    data: bytes
    content_type: str = "image/png"
    label: str = ""
    caption: str = ""
    source_time: datetime | None = None


@dataclass
class CameraFrame:
    """A decoded live-view frame before renderer attachment naming."""

    data: bytes
    label: str
    source_time: datetime


@dataclass
class DashboardPayload:
    """Complete dashboard: ordered embeds plus attachments."""

    embeds: list[object] = field(default_factory=list)  # list[discord.Embed]
    files: list[ImageAsset] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
