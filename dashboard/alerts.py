"""Alert monitor: posts thread updates when weather warnings or traffic status
change.

Every meaningful change (a warning hoisted/removed, a corridor turning heavy or
clearing) is posted to the dashboard thread. Critical changes — black rainstorm
hoist/removal, typhoon signal 8+, heavy congestion appearing/disappearing —
also ping the configured alert role.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from dashboard.models import SpeedBand, TrafficCorridorStatus, WeatherWarning

log = logging.getLogger(__name__)

# Critical warning codes that warrant a role ping.
CRITICAL_WARNING_CODES: frozenset[str] = frozenset(
    {
        "WRAINB",  # black rainstorm
        "TC8NE", "TC8NW", "TC8SE", "TC8SW",  # signal 8
        "TC9",  # increasing gale
        "TC10",  # hurricane
    }
)

# Friendly names so alerts read well even after a warning is removed
# (mirrors the HKO code->name map in the weather provider).
WARNING_NAMES: dict[str, str] = {
    "TC1": "Standby Signal No. 1",
    "TC3": "Strong Wind Signal No. 3",
    "TC8NE": "Gale Signal No. 8 NE",
    "TC8NW": "Gale Signal No. 8 NW",
    "TC8SE": "Gale Signal No. 8 SE",
    "TC8SW": "Gale Signal No. 8 SW",
    "TC9": "Gale Signal No. 9",
    "TC10": "Hurricane Signal No. 10",
    "WRAINA": "Amber Rainstorm",
    "WRAINR": "Red Rainstorm",
    "WRAINB": "Black Rainstorm",
    "WTS": "Thunderstorm Warning",
    "WL": "Landslip Warning",
    "WFIRER": "Red Fire Danger",
    "WFIREY": "Yellow Fire Danger",
    "WHOT": "Very Hot Weather",
    "WCOLD": "Cold Weather Warning",
    "WMSGNL": "Strong Monsoon Signal",
    "WFROST": "Frost Warning",
    "WTM": "Tsunami Warning",
}


def _warning_name(code: str) -> str:
    return WARNING_NAMES.get(code, code)


def _family_of(code: str) -> str:
    """Group related warning codes so escalations read as upgrades:
    TC1/TC3/TC8*/TC9/TC10 -> "TC", WRAINA/WRAINR/WRAINB -> "WRAIN"."""
    if code.startswith("TC"):
        return "TC"
    if code.startswith("WRAIN"):
        return "WRAIN"
    if code.startswith("WFIRE"):
        return "WFIRE"
    return code


# Speed below which a corridor counts as "heavy congestion" (dashboard
# heuristic matching the RED band).
HEAVY_SPEED_KMH = 20.0


@dataclass
class AlertState:
    """Last-known signal state, used to detect transitions."""

    warning_codes: frozenset[str] = frozenset()
    heavy_corridors: frozenset[str] = frozenset()


@dataclass
class AlertEvent:
    """A single thread-worthy change."""

    text: str
    critical: bool = False


@dataclass
class AlertMonitor:
    """Tracks state and emits AlertEvents on change."""

    state: AlertState = field(default_factory=AlertState)
    _initialized: bool = False

    def _warning_events(
        self, warnings: list[WeatherWarning], now: datetime
    ) -> list[AlertEvent]:
        new_codes = frozenset(w.code for w in warnings)
        old_codes = self.state.warning_codes
        events: list[AlertEvent] = []

        hoisted = new_codes - old_codes
        removed = old_codes - new_codes
        for code in sorted(hoisted):
            w = next((x for x in warnings if x.code == code), None)
            name = _warning_name(w.name if w else code)
            critical = code in CRITICAL_WARNING_CODES
            # if the same family had a lower level before, this is an upgrade
            family = _family_of(code)
            upgraded = any(_family_of(c) == family for c in removed)
            verb = "upgraded to" if upgraded else "hoisted"
            events.append(
                AlertEvent(
                    text=f"⚠️ **{name}** {verb} by HKO "
                         f"(<t:{int(now.timestamp())}:t>)",
                    critical=critical,
                )
            )
        for code in sorted(removed):
            name = _warning_name(code)
            critical = code in CRITICAL_WARNING_CODES
            family = _family_of(code)
            downgraded = any(_family_of(c) == family for c in hoisted)
            verb = "downgraded from" if downgraded else "cancelled"
            events.append(
                AlertEvent(
                    text=f"✅ **{name}** {verb} by HKO "
                         f"(<t:{int(now.timestamp())}:t>)",
                    critical=critical,
                )
            )
        return events

    def _traffic_events(
        self, statuses: list[TrafficCorridorStatus], now: datetime
    ) -> list[AlertEvent]:
        heavy = frozenset(
            st.name
            for st in statuses
            if any(
                o.band == SpeedBand.RED
                or (o.speed_kmh is not None and o.speed_kmh < HEAVY_SPEED_KMH)
                for o in st.observations
            )
        )
        old_heavy = self.state.heavy_corridors
        events: list[AlertEvent] = []
        for corridor in sorted(heavy - old_heavy):
            events.append(
                AlertEvent(
                    text=f"🚦 **{corridor}** heavy congestion "
                         f"(<t:{int(now.timestamp())}:t>)",
                    critical=True,
                )
            )
        for corridor in sorted(old_heavy - heavy):
            events.append(
                AlertEvent(
                    text=f"✅ **{corridor}** congestion cleared "
                         f"(<t:{int(now.timestamp())}:t>)",
                    critical=True,
                )
            )
        return events

    def update(
        self,
        warnings: list[WeatherWarning],
        statuses: list[TrafficCorridorStatus],
        now: datetime | None = None,
    ) -> list[AlertEvent]:
        """Feed the latest state; returns the events for this tick.

        The first call only seeds the state (no flood of events on startup).
        """
        now = now or datetime.now(UTC)
        events = self._warning_events(warnings, now) + self._traffic_events(
            statuses, now
        )
        self.state.warning_codes = frozenset(w.code for w in warnings)
        self.state.heavy_corridors = frozenset(
            st.name
            for st in statuses
            if any(
                o.band == SpeedBand.RED
                or (o.speed_kmh is not None and o.speed_kmh < HEAVY_SPEED_KMH)
                for o in st.observations
            )
        )
        if not self._initialized:
            self._initialized = True
            return []
        return events

    def ping_for(self, event: AlertEvent, role_id: int | None) -> str:
        """Render a thread message, appending the role ping when critical."""
        if event.critical and role_id is not None:
            return f"{event.text}\n<@&{role_id}>"
        return event.text
