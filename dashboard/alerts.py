"""Alert monitor: posts thread updates when weather warnings or TD notices
change.

Every meaningful change (a warning hoisted/removed or a relevant TD traffic
notice/roadwork appearing/clearing) is posted to the dashboard thread. Critical
weather changes (black rainstorm hoist/removal and typhoon signal 8+) also ping
the configured alert role.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from dashboard.models import Roadwork, TrafficCorridorStatus, TrafficIncident, WeatherWarning

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


@dataclass
class AlertState:
    """Last-known signal state, used to detect transitions."""

    warning_codes: frozenset[str] = frozenset()
    traffic_incidents: frozenset[str] = frozenset()
    roadworks: frozenset[str] = frozenset()
    roadwork_labels: dict[str, str] = field(default_factory=dict)


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

    def _incident_events(
        self, incidents: list[TrafficIncident], now: datetime
    ) -> list[AlertEvent]:
        current = frozenset(incident.identifier or incident.title for incident in incidents)
        events: list[AlertEvent] = []
        for incident in incidents:
            key = incident.identifier or incident.title
            if key not in self.state.traffic_incidents:
                road = incident.road or incident.location
                suffix = f" — {road}" if road else ""
                events.append(
                    AlertEvent(
                        text=f"📢 **{incident.title}**{suffix} "
                        f"(<t:{int(now.timestamp())}:t>)"
                    )
                )
        for key in sorted(self.state.traffic_incidents - current):
            events.append(
                AlertEvent(
                    text=f"✅ TD traffic notice cleared: **{key}** "
                    f"(<t:{int(now.timestamp())}:t>)"
                )
            )
        return events

    @staticmethod
    def _roadwork_key(roadwork: Roadwork) -> str:
        return roadwork.identifier or f"{roadwork.road}:{roadwork.description}"

    @staticmethod
    def _roadwork_label(roadwork: Roadwork) -> str:
        return roadwork.description or roadwork.road or roadwork.identifier

    def _roadwork_events(
        self, roadworks: list[Roadwork], now: datetime
    ) -> list[AlertEvent]:
        current = {self._roadwork_key(item): item for item in roadworks}
        events: list[AlertEvent] = []
        for key in sorted(current.keys() - self.state.roadworks):
            roadwork = current[key]
            suffix = (
                f" — {roadwork.road}"
                if roadwork.road and roadwork.road not in roadwork.description
                else ""
            )
            events.append(
                AlertEvent(
                    text=f"🚧 **TD roadworks:** {roadwork.description}{suffix} "
                    f"(<t:{int(now.timestamp())}:t>)"
                )
            )
        for key in sorted(self.state.roadworks - current.keys()):
            label = self.state.roadwork_labels.get(key, key)
            events.append(
                AlertEvent(
                    text=f"✅ TD roadworks cleared: **{label}** "
                    f"(<t:{int(now.timestamp())}:t>)"
                )
            )
        return events

    def update(
        self,
        warnings: list[WeatherWarning],
        statuses: list[TrafficCorridorStatus],
        incidents: list[TrafficIncident] | None = None,
        roadworks: list[Roadwork] | None = None,
        now: datetime | None = None,
    ) -> list[AlertEvent]:
        """Feed the latest state; returns the events for this tick.

        The first call only seeds the state (no flood of events on startup).
        """
        now = now or datetime.now(UTC)
        current_incidents = incidents or []
        current_roadworks = roadworks or []
        events = (
            self._warning_events(warnings, now)
            + self._incident_events(current_incidents, now)
            + self._roadwork_events(current_roadworks, now)
        )
        self.state.warning_codes = frozenset(w.code for w in warnings)
        self.state.traffic_incidents = frozenset(
            incident.identifier or incident.title for incident in current_incidents
        )
        self.state.roadworks = frozenset(
            self._roadwork_key(roadwork) for roadwork in current_roadworks
        )
        self.state.roadwork_labels = {
            self._roadwork_key(roadwork): self._roadwork_label(roadwork)
            for roadwork in current_roadworks
        }
        if not self._initialized:
            self._initialized = True
            return []
        return events

    def ping_for(self, event: AlertEvent, role_id: int | None) -> str:
        """Render a thread message, appending the role ping when critical."""
        if event.critical and role_id is not None:
            return f"{event.text}\n<@&{role_id}>"
        return event.text
