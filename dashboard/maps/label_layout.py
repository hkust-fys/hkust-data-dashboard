"""Deterministic box placement, independent of map projection and drawing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple, Protocol

Rect = tuple[float, float, float, float]
LOCAL_SEARCH_RADIUS = 96.0
SEARCH_STEP = 6.0
_GLOBAL_ROAD_PENALTY = 48.0


class Occupancy(Protocol):
    def overlap(self, rect: Rect) -> tuple[int, int]: ...


class _Candidate(NamedTuple):
    rect: Rect
    displacement: float
    side: int


def rects_overlap(first: Rect, second: Rect, padding: float = 2) -> bool:
    return not (
        first[2] + padding <= second[0]
        or second[2] + padding <= first[0]
        or first[3] + padding <= second[1]
        or second[3] + padding <= first[1]
    )


def place_label_box(
    anchor: tuple[float, float],
    dimensions: tuple[float, float],
    size: tuple[int, int],
    occupied: Sequence[Rect],
    arrows: Sequence[Rect],
    *,
    scale: float = 1.0,
    traffic: Occupancy | None = None,
    important_roads: Occupancy | None = None,
) -> Rect:
    """Place a full-size singleton or stack beside its immutable arrow.

    Road-free placement is a constraint throughout the bounded local search.
    Only after exhausting that search may a local box cover a protected road.
    A canvas-wide search is reserved for label/arrow congestion, with bounded
    road penalties so a distant gap cannot drag the label across the map.
    """
    width, height = dimensions
    margin, gap, padding = 2 * scale, 10 * scale, 2 * scale
    step = SEARCH_STEP * scale

    def candidate(side: int, dx: float = 0, dy: float = 0) -> _Candidate:
        left = anchor[0] + (gap if side > 0 else -gap - width) + dx
        top = anchor[1] - height / 2 + dy
        return _Candidate(
            (left, top, left + width, top + height), math.hypot(dx, dy), side,
        )

    def arrow_safe(rect: Rect) -> bool:
        return not any(rects_overlap(rect, arrow, padding=0) for arrow in arrows)

    def collision_free(rect: Rect) -> bool:
        return not any(rects_overlap(rect, other, padding=padding) for other in occupied)

    def fits(rect: Rect) -> bool:
        return (
            margin <= rect[0] and margin <= rect[1]
            and rect[2] <= size[0] - margin and rect[3] <= size[1] - margin
            and arrow_safe(rect) and collision_free(rect)
        )

    def covers_road(rect: Rect) -> bool:
        return important_roads is not None and bool(important_roads.overlap(rect)[0])

    def score(item: _Candidate, *, local: bool = True) -> tuple:
        road_overlap = covers_road(item.rect)
        traffic_cost = 0.0
        if traffic is not None:
            overlap, area = traffic.overlap(item.rect)
            if overlap and area:
                traffic_cost = scale * (24 + 20 * overlap / area)
        cost = item.displacement + traffic_cost
        road_cost = _GLOBAL_ROAD_PENALTY * scale if road_overlap else 0.0
        return (
            (road_overlap, cost) if local else (cost + road_cost, road_cost)
        ) + (traffic_cost, item.side, item.rect[0], item.rect[1])

    # Keep ordinary layouts stable and cheap. A road-overlapping winner must
    # proceed to the same escape search as a label blocked by another label.
    row_step = height + 4 * scale
    preferred = []
    for dy in (0.0, -row_step, row_step, -2 * row_step, 2 * row_step):
        for side in (-1, 1):
            item = candidate(side, dy=dy)
            if fits(item.rect):
                preferred.append(item)
    best = min(preferred, key=score) if preferred else None
    if best is not None and not covers_road(best.rect):
        return best.rect

    # Search in two dimensions: loops and junctions need diagonal slots as
    # well as different rows. Both axes share a true Euclidean distance bound,
    # independent of the number of rows in a stacked label.
    local_candidates = list(preferred)
    steps = round(LOCAL_SEARCH_RADIUS / SEARCH_STEP)
    for iy in range(-steps, steps + 1):
        for ix in range(-steps, steps + 1):
            if ix * ix + iy * iy > steps * steps:
                continue
            for side in (-1, 1):
                item = candidate(side, ix * step, iy * step)
                if fits(item.rect):
                    local_candidates.append(item)
    if local_candidates:
        return min(local_candidates, key=score).rect

    # No collision-free local box exists. Search the canvas once, retaining
    # the best strict and relaxed choices without building large grid lists.
    best_grid = best_relaxed = None
    best_grid_score = best_relaxed_score = None
    grid_step = max(1, round(step))
    start = math.ceil(margin)
    for top in range(start, math.floor(size[1] - height - margin) + 1, grid_step):
        for left in range(start, math.floor(size[0] - width - margin) + 1, grid_step):
            rect = (float(left), float(top), left + width, top + height)
            if not arrow_safe(rect):
                continue
            item = _Candidate(
                rect,
                math.hypot(left + width / 2 - anchor[0], top + height / 2 - anchor[1]),
                0,
            )
            item_score = score(item, local=False)
            if best_relaxed_score is None or item_score < best_relaxed_score:
                best_relaxed, best_relaxed_score = rect, item_score
            if collision_free(rect) and (best_grid_score is None or item_score < best_grid_score):
                best_grid, best_grid_score = rect, item_score
    if best_grid is not None:
        return best_grid
    if best_relaxed is not None:
        return best_relaxed

    # Even the arrow-safe grid is full (or the label exceeds the viewport).
    # Preserve the full label and prefer a clamped side clear of the arrows.
    fallbacks = []
    for side in (-1, 1):
        item = candidate(side)
        left = min(max(margin, item.rect[0]), size[0] - width - margin)
        top = min(max(margin, item.rect[1]), size[1] - height - margin)
        rect = (left, top, left + width, top + height)
        if arrow_safe(rect):
            fallbacks.append(_Candidate(rect, 0, side))
    if fallbacks:
        return min(fallbacks, key=score).rect
    left = min(max(margin, anchor[0] + gap), size[0] - width - margin)
    top = min(max(margin, anchor[1] - height / 2), size[1] - height - margin)
    return left, top, left + width, top + height
