"""Bounded LandsD road-centreline lookup for route-road derivation.

Dataset catalogue: https://data.gov.hk/en-data/dataset/hk-landsd-openmap-road-centreline
Data supplied by the Lands Department, HKSAR Government, via CSDI.
"""

from __future__ import annotations

import asyncio
import math
from urllib.parse import urlencode

from dashboard.http import FetchError, HttpClient

ROAD_CENTRELINE_QUERY_URL = (
    "https://portal.csdi.gov.hk/server/rest/services/common/"
    "landsd_rcd_1637310758814_80061/FeatureServer/0/query"
)
PAGE_SIZE = 1000
MAX_PAGES = 24


async def fetch_official_road_ways(
    client: HttpClient, route_paths: list[list[tuple[float, float]]]
) -> list[dict]:
    """Fetch bilingual named road geometry inside the tracked route envelope.

    Pagination must finish successfully before callers replace their cache.
    One-metre simplification bounds download size without affecting the 30m
    sustained-overlap test. Coordinates returned by ArcGIS are lon, lat.
    """
    points = [point for path in route_paths for point in path]
    if not points:
        raise FetchError("No route extent for official road lookup")
    latitudes, longitudes = zip(*points, strict=True)
    if not all(math.isfinite(value) for point in points for value in point):
        raise FetchError("Invalid route extent for official road lookup")
    envelope = ",".join(str(value) for value in (
        min(longitudes) - 0.0005, min(latitudes) - 0.0005,
        max(longitudes) + 0.0005, max(latitudes) + 0.0005,
    ))
    ways: list[dict] = []
    async with asyncio.timeout(90):
        for page in range(MAX_PAGES):
            params = {
                "f": "json", "where": "ENGLISHSTREETNAME IS NOT NULL",
                "geometry": envelope, "geometryType": "esriGeometryEnvelope",
                "inSR": 4326, "outSR": 4326,
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "ENGLISHSTREETNAME,CHINESESTREETNAME",
                "returnGeometry": "true", "geometryPrecision": 6,
                "maxAllowableOffset": 0.00001,
                "resultRecordCount": PAGE_SIZE, "resultOffset": page * PAGE_SIZE,
                "orderByFields": "OBJECTID",
            }
            raw = await client.fetch_json(f"{ROAD_CENTRELINE_QUERY_URL}?{urlencode(params)}")
            if not isinstance(raw, dict) or "error" in raw or not isinstance(raw.get("features"), list):
                raise FetchError("Invalid official road geometry response")
            for feature in raw["features"]:
                attributes = feature.get("attributes") or {}
                english = str(attributes.get("ENGLISHSTREETNAME") or "").strip()
                chinese = str(attributes.get("CHINESESTREETNAME") or "").strip()
                if not english or english.casefold() in {"null", "n/a", "unknown"}:
                    continue
                for path in (feature.get("geometry") or {}).get("paths", []):
                    try:
                        coordinates = [(float(point[1]), float(point[0])) for point in path]
                    except (IndexError, TypeError, ValueError) as exc:
                        raise FetchError("Invalid official road coordinates") from exc
                    if len(coordinates) >= 2 and all(
                        math.isfinite(value) for point in coordinates for value in point
                    ):
                        ways.append({
                            "name": english, "name_en": english, "name_zh": chinese,
                            "points": coordinates,
                        })
            if not raw.get("exceededTransferLimit"):
                if not ways:
                    raise FetchError("Official road geometry was empty")
                return ways
    raise FetchError("Official road geometry exceeded pagination limit")
