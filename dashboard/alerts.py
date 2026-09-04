"""Alert monitor: posts thread updates when weather warnings or TD notices
change.

Every meaningful change (a warning hoisted/removed or a relevant TD traffic
notice/roadwork appearing/clearing) is posted to the dashboard thread. New TD
traffic notices on Clear Water Bay Road or New Clear Water Bay Road, and heavy
detector-confirmed congestion, ping the configured alert role. Weather changes
and roadworks do not ping.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from dashboard import road_policy
from dashboard.models import Roadwork, TrafficCorridorStatus, TrafficIncident, WeatherWarning
from dashboard.providers.traffic import resolve_incident_road_keys

log = logging.getLogger(__name__)

# Warning codes that are posted to the thread prominently. They no longer ping:
# weather signals are either pre-announced (Pre-No. 8) or instantaneous (black
# rain), so the role ping is reserved for heavy congestion.
CRITICAL_WARNING_CODES: frozenset[str] = frozenset(
    {
        "WRAINB",  # black rainstorm
        "TC8NE", "TC8NW", "TC8SE", "TC8SW",  # signal 8
        "TC9",  # increasing gale
        "TC10",  # hurricane
        "TC8PRE",  # Pre-No. 8 Special Announcement (~2 h lead time)
    }
)

CONGESTION_RED_TICKS_REQUIRED = 2
ROAD_ALERT_COOLDOWN_SECONDS = 60 * 60.0
DISCORD_MESSAGE_CONTENT_LIMIT = 2_000

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
    warning_names: dict[str, str] = field(default_factory=dict)
    traffic_incidents: frozenset[str] = frozenset()
    roadworks: frozenset[str] = frozenset()
    roadwork_labels: dict[str, str] = field(default_factory=dict)
    red_streaks: dict[str, int] = field(default_factory=dict)
    active_reds: set[str] = field(default_factory=set)
    road_alert_cooldown_until: dict[str, float] = field(default_factory=dict)


@dataclass
class AlertEvent:
    """A single thread-worthy change.

    ``ping`` marks events that mention the alert role;
    ``critical`` marks prominent weather events, which post without pinging.
    """

    text: str
    critical: bool = False
    ping: bool = False
    source_text: str = ""


@dataclass
class AlertMonitor:
    """Tracks state and emits AlertEvents on change."""

    state: AlertState = field(default_factory=AlertState)
    roads: object | None = None  # TrackedRoads table for names/affected routes
    _initialized: bool = False

    def _claim_road_alert(self, road_keys: set[str]) -> bool:
        """Claim one shared one-hour role-alert window for the matched roads."""
        watched = road_policy.IMPORTANT_ROAD_KEYS.intersection(road_keys)
        if not watched:
            return False
        if not self._initialized:
            return True  # startup events are discarded and must not consume cooldown
        now = time.monotonic()
        if all(now < self.state.road_alert_cooldown_until.get(key, 0.0) for key in watched):
            return False
        until = now + ROAD_ALERT_COOLDOWN_SECONDS
        for key in watched:
            self.state.road_alert_cooldown_until[key] = until
        return True

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
            name = w.name if w else code
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
            name = self.state.warning_names.get(code, code)
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
                        f"(<t:{int(now.timestamp())}:t>)",
                        ping=self._claim_road_alert(self._incident_road_keys(incident)),
                        source_text=incident.description,
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

    def _incident_road_keys(self, incident: TrafficIncident) -> set[str]:
        """Resolve a TD notice to the watched canonical road keys."""
        keys = resolve_incident_road_keys(
            incident,
            self.roads,
            explicit_fallback_keys=road_policy.IMPORTANT_ROAD_KEYS,
        )
        return road_policy.IMPORTANT_ROAD_KEYS.intersection(keys)

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

    def _congestion_events(self, statuses, now: datetime) -> list[AlertEvent]:
        """Red-band detector congestion with hysteresis and shared cooldown.

        A road must read RED on consecutive updates before alerting (one
        detector glitch is not a jam), re-alerts at most hourly, and
        posts one cleared note when it leaves the red band.
        """
        events: list[AlertEvent] = []
        state = self.state
        current_reds: dict[str, object] = {}
        for status in statuses:
            bands = [o.band for o in status.observations if o.band is not None]
            if not bands or any(band.value != "red" for band in bands):
                continue
            current_reds[status.name] = status

        for name, status in current_reds.items():
            road_key = name.casefold().strip()
            if road_key not in road_policy.IMPORTANT_ROAD_KEYS:
                continue
            streak = state.red_streaks.get(name, 0) + 1
            state.red_streaks[name] = streak
            if name in state.active_reds:
                continue
            if streak < CONGESTION_RED_TICKS_REQUIRED:
                continue
            state.active_reds.add(name)
            events.append(
                self._congestion_event(
                    name,
                    status,
                    now,
                    cleared=False,
                    ping=self._claim_road_alert({road_key}),
                )
            )

        for name in sorted(set(state.active_reds) - set(current_reds)):
            state.active_reds.discard(name)
            events.append(self._congestion_event(name, None, now, cleared=True))
        # A red reading that does not persist must not bank progress toward
        # the next alert: only consecutive red updates count.
        for name in list(state.red_streaks):
            if name not in current_reds:
                state.red_streaks[name] = 0
        return events

    def _congestion_event(
        self, name: str, status, now: datetime, cleared: bool, ping: bool = False
    ) -> AlertEvent:
        roads = self.roads
        display = roads.display_name(name) if roads is not None else name
        routes: list[str] = []
        if roads is not None:
            routes = roads.routes_for_keys([name])
        suffix = f" — affects: {', '.join(routes)}" if routes else ""
        stamp = f" (<t:{int(now.timestamp())}:t>)"
        if cleared:
            return AlertEvent(text=f"✅ Congestion easing on **{display}**{stamp}")
        speed_text = ""
        observations = getattr(status, "observations", []) if status is not None else []
        speeds = [
            o.speed_kmh for o in observations if getattr(o, "speed_kmh", None) is not None
        ]
        if speeds:
            speed_text = f" ({min(speeds):.0f} km/h)"
        return AlertEvent(
            text=f"🚗 Heavy congestion on **{display}**{speed_text}{suffix}{stamp}",
            ping=ping,
        )

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
        monotonic_now = time.monotonic()
        self.state.road_alert_cooldown_until = {
            key: until
            for key, until in self.state.road_alert_cooldown_until.items()
            if until > monotonic_now
        }
        current_incidents = incidents or []
        current_roadworks = roadworks or []
        events = (
            self._warning_events(warnings, now)
            + self._incident_events(current_incidents, now)
            + self._roadwork_events(current_roadworks, now)
            + self._congestion_events(statuses, now)
        )
        self.state.warning_codes = frozenset(w.code for w in warnings)
        self.state.warning_names = {w.code: w.name for w in warnings}
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
        """Render a thread message, adding the configured role when requested."""
        if event.ping and role_id is not None:
            return f"{event.text}\n<@&{role_id}>"
        return event.text

    def messages_for(self, event: AlertEvent, role_id: int | None) -> list[str]:
        """Render bounded thread messages while retaining the complete TD notice."""
        text = self.ping_for(event, role_id)
        if event.source_text:
            text = f"{text}\n\n**Full TD source text:**\n{event.source_text}"
        return [
            text[start : start + DISCORD_MESSAGE_CONTENT_LIMIT]
            for start in range(0, len(text), DISCORD_MESSAGE_CONTENT_LIMIT)
        ]
