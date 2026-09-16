"""Parsers for unstructured current traffic-news pages.

These helpers perform no import-time I/O.  Callers provide the fetched page and
the current tracked-road table.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any

from dashboard.http import FetchError
from dashboard.models import TrafficIncident

HKT = timezone(timedelta(hours=8))
RTHK_TRAFFIC_NEWS_URL = "https://programme.rthk.hk/channel/radio/trafficnews/index.php"

_CLEARED = re.compile(
    r"(?:已清理|已清場|已拖走|交通回復正常|已解封|重新開放|"
    r"re[\s-]?opened\s+to\s+all\s+traffic)",
    re.IGNORECASE,
)


def is_cleared_notice(incident: TrafficIncident) -> bool:
    return incident.status.casefold() in {"closed", "reopened", "cleared"} or bool(
        _CLEARED.search(" ".join((incident.title, incident.description)))
    )


def _sanitize(text: str) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _classes(attrs: list[tuple[str, str | None]]) -> set[str]:
    for name, value in attrs:
        if name.casefold() == "class":
            return {part.casefold() for part in (value or "").split()}
    return set()


class _RthkPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found_articles = False
        self.page_text: list[str] = []
        self.items: list[tuple[str, str]] = []
        self._in_news_list = 0
        self._item_parts: list[str] | None = None
        self._date_parts: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        lowered = tag.casefold()
        classes = _classes(attrs)
        if lowered == "div" and "articles" in classes:
            self.found_articles = True
        elif lowered == "ul" and "dec" in classes:
            self._in_news_list += 1
        elif lowered == "li" and self._in_news_list and "inner" in classes:
            self._item_parts = []
        elif lowered == "div" and self._item_parts is not None and "date" in classes:
            self._date_parts = []
        elif lowered == "br" and self._item_parts is not None:
            self._item_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered == "div" and self._date_parts is not None:
            return
        if lowered == "li" and self._item_parts is not None:
            text = _sanitize(" ".join(self._item_parts))
            date_text = _sanitize(" ".join(self._date_parts or []))
            if text:
                self.items.append((text, date_text))
            self._item_parts = None
            self._date_parts = None
        elif lowered == "ul" and self._in_news_list:
            self._in_news_list -= 1

    def handle_data(self, data: str) -> None:
        self.page_text.append(data)
        if self._date_parts is not None:
            self._date_parts.append(data)
        elif self._item_parts is not None:
            self._item_parts.append(data)


def _parse_rthk_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d HKT %H:%M").replace(tzinfo=HKT)
    except ValueError:
        return None


def _road_keys(text: str, roads: Any) -> list[str]:
    if roads is None:
        return []
    match = getattr(roads, "match", None)
    return list(match(text)) if callable(match) else []


def rthk_latest_report_time(html_text: str) -> datetime | None:
    """Newest published report on the page, including roads outside our routes."""
    parser = _RthkPageParser()
    parser.feed(html_text)
    return max(
        (stamp for _text, date in parser.items if (stamp := _parse_rthk_time(date)) is not None),
        default=None,
    )


def parse_rthk_traffic_news(html_text: str, roads: Any) -> list[TrafficIncident]:
    """Parse relevant reports from RTHK's current (unparameterized) page."""
    parser = _RthkPageParser()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception as exc:
        raise FetchError("Invalid RTHK traffic news HTML") from exc

    page_text = _sanitize(" ".join(parser.page_text))
    if "交通消息" not in page_text or not parser.found_articles:
        raise FetchError("Unrecognized RTHK traffic news page")

    incidents: list[TrafficIncident] = []
    for text, date_text in parser.items:
        keys = _road_keys(text, roads)
        if not keys:
            continue
        announcement_time = _parse_rthk_time(date_text)
        digest_input = f"{date_text}\n{text}".casefold().encode("utf-8")
        digest = hashlib.sha256(digest_input).hexdigest()[:16]
        direction_match = re.search(r"往(.{1,12}?)方向", text)
        display_name = "; ".join(roads.display_name(key) for key in keys)
        incidents.append(
            TrafficIncident(
                identifier=f"RTHK-WEB-{digest}",
                title="RTHK traffic update",
                description=text,
                road=display_name,
                location=display_name,
                direction=(direction_match.group(1).strip() if direction_match else ""),
                status="CLOSED" if _CLEARED.search(text) else "ACTIVE",
                announcement_time=announcement_time,
                source="RTHK",
                source_url=RTHK_TRAFFIC_NEWS_URL,
            )
        )
    return incidents


__all__ = [
    "RTHK_TRAFFIC_NEWS_URL",
    "is_cleared_notice",
    "parse_rthk_traffic_news",
]
