"""Pure encoded-polyline and route interpolation utilities."""

from __future__ import annotations

import math
from typing import TypeAlias

Coordinate: TypeAlias = tuple[float, float]


def encode_polyline(coords: list[Coordinate] | tuple[Coordinate, ...]) -> str:
    out: list[str] = []

    def encode_value(value: int) -> None:
        value = ~(value << 1) if value < 0 else value << 1
        while value >= 0x20:
            out.append(chr((0x20 | (value & 0x1F)) + 63))
            value >>= 5
        out.append(chr(value + 63))

    lat = lon = 0
    for latitude, longitude in coords:
        next_lat, next_lon = round(latitude * 1e5), round(longitude * 1e5)
        encode_value(next_lat - lat)
        encode_value(next_lon - lon)
        lat, lon = next_lat, next_lon
    return "".join(out)


def decode_polyline(points: str) -> list[Coordinate]:
    coords: list[Coordinate] = []
    lat = lon = index = 0
    while index < len(points):
        values: list[int] = []
        for _ in range(2):
            shift = result = 0
            while True:
                if index >= len(points):
                    return coords
                byte = ord(points[index]) - 63
                index += 1
                result |= (byte & 0x1F) << shift
                shift += 5
                if byte < 0x20:
                    break
            values.append(~(result >> 1) if result & 1 else result >> 1)
        lat += values[0]
        lon += values[1]
        coords.append((lat / 1e5, lon / 1e5))
    return coords


def polyline_length(coords: list[Coordinate] | tuple[Coordinate, ...]) -> float:
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(coords, coords[1:], strict=False)
    )


def point_and_tangent_at_fraction(
    coords: list[Coordinate] | tuple[Coordinate, ...], fraction: float
) -> tuple[Coordinate, float]:
    if len(coords) < 2:
        return coords[0], 0.0
    total = polyline_length(coords)
    target = min(max(fraction, 0.0), 1.0) * total
    travelled = 0.0
    for a, b in zip(coords, coords[1:], strict=False):
        segment = math.hypot(b[0] - a[0], b[1] - a[1])
        if segment and travelled + segment >= target:
            ratio = (target - travelled) / segment
            point = (a[0] + (b[0] - a[0]) * ratio, a[1] + (b[1] - a[1]) * ratio)
            return point, math.atan2(b[0] - a[0], b[1] - a[1])
        travelled += segment
    a, b = coords[-2], coords[-1]
    return b, math.atan2(b[0] - a[0], b[1] - a[1])


def polyline_slice(
    coords: list[Coordinate] | tuple[Coordinate, ...], f0: float, f1: float
) -> list[Coordinate]:
    if len(coords) < 2:
        return list(coords)
    f0, f1 = sorted((min(max(f0, 0.0), 1.0), min(max(f1, 0.0), 1.0)))
    total = polyline_length(coords)
    if total <= 0:
        return [coords[0], coords[-1]]
    start, _ = point_and_tangent_at_fraction(coords, f0)
    end, _ = point_and_tangent_at_fraction(coords, f1)
    out = [start]
    travelled = 0.0
    for index, (a, b) in enumerate(zip(coords, coords[1:], strict=False), start=1):
        travelled += math.hypot(b[0] - a[0], b[1] - a[1])
        if f0 < travelled / total < f1 and index < len(coords) - 1:
            out.append(b)
    out.append(end)
    return out
