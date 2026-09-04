"""Shared policy for roads which receive dashboard priority treatment."""

from __future__ import annotations

import math
from collections.abc import Mapping

# Canonical keys from ``TrackedRoads``.  The same set controls role-alert
# eligibility and the OSM corridors which bus/minibus labels should avoid.
IMPORTANT_ROAD_KEYS: frozenset[str] = frozenset(
    {"clear water bay road", "new clear water bay road"}
)


def important_road_paths(roads: object | None) -> list[list[tuple[float, float]]]:
    """Return valid named OSM paths for the centrally important roads only.

    Curated fallback road tables intentionally have no geometry.  Missing or
    malformed path data therefore degrades to an empty list, preserving the
    map while disabling only the extra label-avoidance priority for that frame.
    """
    paths_by_key = getattr(roads, "paths", None)
    if not isinstance(paths_by_key, Mapping):
        return []

    paths: list[list[tuple[float, float]]] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for key in sorted(IMPORTANT_ROAD_KEYS):
        try:
            raw_paths = iter(paths_by_key.get(key, ()) or ())
        except TypeError:
            continue
        for raw_path in raw_paths:
            normalized: list[tuple[float, float]] = []
            try:
                for raw_latitude, raw_longitude in raw_path:
                    latitude = float(raw_latitude)
                    longitude = float(raw_longitude)
                    if not (
                        math.isfinite(latitude)
                        and math.isfinite(longitude)
                        and -90.0 <= latitude <= 90.0
                        and -180.0 <= longitude <= 180.0
                    ):
                        normalized = []
                        break
                    normalized.append((latitude, longitude))
            except (TypeError, ValueError):
                continue
            frozen = tuple(normalized)
            if len(frozen) >= 2 and frozen not in seen:
                seen.add(frozen)
                paths.append(list(frozen))
    return paths


__all__ = ["IMPORTANT_ROAD_KEYS", "important_road_paths"]
