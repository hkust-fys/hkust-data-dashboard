"""Conservative reconciliation of attributed TD and RTHK traffic reports."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any

from dashboard.models import TrafficIncident

HKT = timezone(timedelta(hours=8))
MAX_REPORT_LAG = timedelta(hours=3)


@dataclass(frozen=True)
class IncidentDetails:
    road_keys: tuple[str, ...]
    direction: str
    landmark: str
    cause: str
    cleared: bool


@dataclass(frozen=True)
class _TdPageSnapshot:
    updated_at: datetime | None
    items: tuple[str, ...]


class _TdListParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.page_text: list[str] = []
        self.items: list[str] = []
        self._list_depth = 0
        self._parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.casefold()
        if tag == "ol":
            self._list_depth += 1
        elif tag == "li" and self._list_depth:
            self._parts = []
        elif tag == "br" and self._parts is not None:
            self._parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag == "li" and self._parts is not None:
            text = _clean(" ".join(self._parts))
            if text:
                self.items.append(text)
            self._parts = None
        elif tag == "ol" and self._list_depth:
            self._list_depth -= 1

    def handle_data(self, data: str) -> None:
        self.page_text.append(data)
        if self._parts is not None:
            self._parts.append(data)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _td_page_snapshot(html_text: str) -> _TdPageSnapshot:
    parser = _TdListParser()
    parser.feed(html_text)
    parser.close()
    page_text = _clean(" ".join(parser.page_text))
    match = re.search(
        r"(\d{4})(?:/|\u5e74)(\d{1,2})(?:/|\u6708)(\d{1,2})(?:\u65e5)?\s+"
        r"(\d{1,2}:\d{2}:\d{2})\s+([AP]M)",
        page_text,
        re.IGNORECASE,
    )
    updated_at = None
    if match is not None:
        normalized = "/".join(match.group(1, 2, 3)) + " " + " ".join(match.group(4, 5))
        with suppress(ValueError):
            updated_at = datetime.strptime(normalized.upper(), "%Y/%m/%d %I:%M:%S %p").replace(
                tzinfo=HKT
            )
    return _TdPageSnapshot(updated_at, tuple(parser.items))


_CLEARED = re.compile(
    r"re[\s-]?opened\s+to\s+all\s+traffic|"
    r"\u73fe\u5df2\u89e3\u5c01|\u5df2\u6e05\u7406|\u5df2\u6e05\u5834|"
    r"\u5df2\u62d6\u8d70|\u4ea4\u901a\u56de\u5fa9\u6b63\u5e38",
    re.IGNORECASE,
)

_CAUSE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "traffic-accident",
        re.compile(r"traffic accident|\u4ea4\u901a\u610f\u5916", re.IGNORECASE),
    ),
    (
        "vehicle-breakdown",
        re.compile(
            r"vehicle breakdown|\u8eca\u8f1b\u6545\u969c|\u58de\u8eca",
            re.IGNORECASE,
        ),
    ),
    (
        "watermain-works",
        re.compile(
            r"watermain (?:emergency )?works|\u6c34\u7ba1\u7dca\u6025\u7dad\u4fee",
            re.IGNORECASE,
        ),
    ),
    (
        "road-works",
        re.compile(r"road ?works|\u9053\u8def\u5de5\u7a0b", re.IGNORECASE),
    ),
)


def _cause(text: str) -> str:
    return next((name for name, pattern in _CAUSE_PATTERNS if pattern.search(text)), "")


def _road_keys(text: str, roads: Any) -> tuple[str, ...]:
    match = getattr(roads, "match", None)
    return tuple(dict.fromkeys(match(text))) if callable(match) else ()


def _chinese_direction(text: str) -> str:
    match = re.search(r"\u5f80\s*([^\uff0c\u3002()\uff08\uff09]{1,16}?)\s*\u65b9\u5411", text)
    return _clean(match.group(1)) if match else ""


def _english_direction(text: str) -> str:
    match = re.search(r"\(([^()]*(?:bound|bounds))\)", text, re.IGNORECASE)
    return _clean(match.group(1).casefold()) if match else ""


_CHINESE_LANDMARK = re.compile(
    r"\u8fd1\s*(.+?)(?="
    r"(?:\u7684)?(?:\u5feb\u7dda|\u6162\u7dda|\u4e2d\u7dda|"
    r"\u90e8\u5206\u884c\u8eca\u7dda|\u90e8\u4efd\u884c\u8eca\u7dda|"
    r"\u6240\u6709\u884c\u8eca\u7dda)(?:\u7684)?|"
    r"(?:\u7684|\u6709)?(?:\u4ea4\u901a\u610f\u5916|\u58de\u8eca|"
    r"\u8eca\u8f1b\u6545\u969c|\u6c34\u7ba1\u7dca\u6025\u7dad\u4fee)|"
    r"[\uff0c\u3002;\uff1b]|$)"
)

# TD's official Chinese page calls this Clear Water Bay Road locality 「井欄樹」,
# while RTHK can shorten the same village name to 「井欄村」.  These are a
# deliberately finite, evidence-backed set of variants; do not generalize by
# stripping place suffixes, which would weaken the exact-landmark safeguard.
_LANDMARK_IDENTITIES = {
    "井欄樹": "tseng lan shue",
    "井欄樹村": "tseng lan shue",
    "井欄村": "tseng lan shue",
}


def _landmark(text: str) -> str:
    if match := _CHINESE_LANDMARK.search(text):
        return _clean(match.group(1))
    match = re.search(
        r"\bnear\s+(.+?)(?=\s+(?:which|is|was|has|with)\b|[.;,]|$)",
        text,
        re.IGNORECASE,
    )
    return _clean(match.group(1).casefold()) if match else ""


def _landmark_identity(landmark: str) -> str:
    normalized = _clean(landmark).casefold()
    return _LANDMARK_IDENTITIES.get(normalized, normalized)


def _is_cleared(report: TrafficIncident, text: str) -> bool:
    return report.is_cleared or bool(_CLEARED.search(text))


def incident_details(report: TrafficIncident, roads: Any) -> IncidentDetails:
    """Extract a strict, source-neutral signature without fuzzy place matching."""
    translated = getattr(report, "translated_description", "")
    text = translated or report.description
    road_text = " ".join((report.road, report.location, text))
    direction = (
        _chinese_direction(text) or _english_direction(text) or _clean(report.direction.casefold())
    )
    return IncidentDetails(
        road_keys=_road_keys(road_text, roads),
        direction=direction,
        landmark=_landmark_identity(_landmark(text)),
        cause=_cause(text),
        cleared=_is_cleared(report, text),
    )


def pair_td_bilingual_pages(
    english_html: str,
    chinese_html: str,
    td_reports: list[TrafficIncident],
    roads: Any,
) -> dict[str, str]:
    """Return proven TD English-ID to official Chinese-description sidecars."""
    english = _td_page_snapshot(english_html)
    chinese = _td_page_snapshot(chinese_html)
    if (
        english.updated_at is None
        or english.updated_at != chinese.updated_at
        or len(english.items) != len(chinese.items)
        or len(td_reports) != len(english.items)
    ):
        return {}

    def signature(text: str) -> tuple[tuple[str, ...], str, bool] | None:
        keys = _road_keys(text, roads)
        cause = _cause(text)
        return (keys, cause, bool(_CLEARED.search(text))) if keys and cause else None

    english_signatures = [signature(text) for text in english.items]
    chinese_signatures = [signature(text) for text in chinese.items]
    english_counts = Counter(item for item in english_signatures if item is not None)
    chinese_counts = Counter(item for item in chinese_signatures if item is not None)

    sidecars: dict[str, str] = {}
    for report, chinese_text, english_signature, chinese_signature in zip(
        td_reports,
        chinese.items,
        english_signatures,
        chinese_signatures,
        strict=True,
    ):
        # Untracked administrative notices do not invalidate independently
        # provable traffic-item pairs elsewhere on the page.
        if english_signature is None and chinese_signature is None:
            continue
        if english_signature is None or english_signature != chinese_signature:
            continue
        # Position alone is insufficient when a page has two same-road events
        # with the same cause/state: the locales can publish them in a different
        # order.  Such items receive no translation sidecar.
        if (
            english_counts[english_signature] != 1
            or chinese_counts[chinese_signature] != 1
        ):
            continue
        sidecars[report.identifier] = chinese_text
    return sidecars


def _semantic_key(details: IncidentDetails) -> str:
    identity = "\n".join(
        ("|".join(details.road_keys), details.direction, details.landmark, details.cause)
    )
    return "traffic-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _within_report_window(td: TrafficIncident, rthk: TrafficIncident) -> bool:
    page_time = getattr(td, "page_updated_at", None)
    report_time = rthk.announcement_time
    if page_time is None or report_time is None:
        return False
    return page_time - MAX_REPORT_LAG <= report_time <= page_time


def reconcile_incident_reports(reports: list[TrafficIncident], roads: Any) -> list[TrafficIncident]:
    """Group only unique TD/RTHK reports with identical strict signatures."""
    details = {id(report): incident_details(report, roads) for report in reports}
    candidates: dict[IncidentDetails, dict[str, list[TrafficIncident]]] = {}
    for report in reports:
        item = details[id(report)]
        if len(item.road_keys) != 1 or not item.direction or not item.landmark or not item.cause:
            continue
        candidates.setdefault(item, {}).setdefault(report.source, []).append(report)

    matched_rthk: set[int] = set()
    replacements: dict[int, TrafficIncident] = {}
    for item, by_source in candidates.items():
        td_items = by_source.get("TD", [])
        rthk_items = by_source.get("RTHK", [])
        if len(td_items) != 1 or len(rthk_items) != 1:
            continue
        td, rthk = td_items[0], rthk_items[0]
        if not _within_report_window(td, rthk):
            continue
        key = _semantic_key(item)
        related = replace(rthk, reconciliation_key=key)
        replacements[id(td)] = replace(
            td,
            related_reports=(related,),
            reconciliation_key=key,
        )
        matched_rthk.add(id(rthk))

    reconciled: list[TrafficIncident] = []
    for report in reports:
        if id(report) in matched_rthk:
            continue
        replacement = replacements.get(id(report))
        if replacement is not None:
            reconciled.append(replacement)
            continue
        item = details[id(report)]
        if len(item.road_keys) == 1 and item.direction and item.landmark and item.cause:
            report = replace(report, reconciliation_key=_semantic_key(item))
        reconciled.append(report)
    return reconciled


__all__ = [
    "IncidentDetails",
    "incident_details",
    "pair_td_bilingual_pages",
    "reconcile_incident_reports",
]
