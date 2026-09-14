"""Road-loop regressions through the map renderer's label-layout boundary."""

import math

import pytest
from PIL import Image, ImageDraw

from dashboard.maps import renderer
from dashboard.models import Operator


@pytest.mark.parametrize("scale", [0.75, 0.9, 1.0])
@pytest.mark.parametrize("grouped", [False, True])
def test_labels_search_beside_loop_when_preferred_columns_cover_road(scale, grouped):
    class FixedWidthDraw:
        def textlength(self, _text, font=None):
            return 60 * scale

    size = (round(320 * scale), round(240 * scale))
    anchor = (160 * scale, 120 * scale)
    mask = Image.new("L", size, 0)
    # A hairpin crosses both preferred label columns. Other labels occupy
    # the clear space above/below it, but there is room beside the bend.
    ImageDraw.Draw(mask).ellipse(
        tuple(value * scale for value in (125, 80, 195, 160)),
        outline=255,
        width=round(14 * scale),
    )
    roads = renderer.TrafficOccupancy(mask)
    occupied = [
        tuple(value * scale for value in rect)
        for rect in ((0, 0, 320, 70), (0, 170, 320, 240))
    ]
    markers = [renderer.BusMarker(("91",), *anchor, Operator.KMB, 0)]
    if grouped:
        markers.append(renderer.BusMarker(("11",), *anchor, Operator.GMB, 0))
    metrics = renderer.RenderMetrics(scale)

    placements = renderer._layout_bus_labels(
        markers, FixedWidthDraw(), renderer._font(13), size, occupied,
        metrics=metrics, important_roads=roads,
    )

    assert len(placements) == 1
    placement = placements[0]
    assert placement.marker == anchor
    assert roads.overlap(placement.rect)[0] == 0
    assert all(not renderer._rects_overlap(placement.rect, rect) for rect in occupied)
    assert not renderer._rects_overlap(
        placement.rect, renderer._arrow_footprint(anchor, metrics), padding=0,
    )
    left, top, right, bottom = placement.rect
    assert right - left == pytest.approx(68 * scale)
    assert bottom - top == pytest.approx((36 if grouped else 18) * scale)
    assert 0 <= left < right <= size[0]
    assert 0 <= top < bottom <= size[1]
    displacement = min(
        math.hypot(left - anchor[0] - 10 * scale, (top + bottom) / 2 - anchor[1]),
        math.hypot(right - anchor[0] + 10 * scale, (top + bottom) / 2 - anchor[1]),
    )
    assert displacement <= 48 * scale
    assert placements == renderer._layout_bus_labels(
        reversed(markers), FixedWidthDraw(), renderer._font(13), size, occupied,
        metrics=metrics, important_roads=roads,
    )


@pytest.mark.parametrize("scale", [0.75, 1.0])
@pytest.mark.parametrize("grouped", [False, True])
def test_labels_escape_a_wide_protected_corridor(scale, grouped):
    class FixedWidthDraw:
        def textlength(self, _text, font=None):
            return 60 * scale

    size = (round(360 * scale), round(180 * scale))
    anchor = (180 * scale, 90 * scale)
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(
        tuple(value * scale for value in (90, 0, 270, 180)), fill=255,
    )
    roads = renderer.TrafficOccupancy(mask)
    markers = [renderer.BusMarker(("91",), *anchor, Operator.KMB, 0)]
    if grouped:
        markers.append(renderer.BusMarker(("11",), *anchor, Operator.GMB, 0))

    placement = renderer._layout_bus_labels(
        markers, FixedWidthDraw(), renderer._font(13), size,
        metrics=renderer.RenderMetrics(scale), important_roads=roads,
    )[0]

    assert placement.marker == anchor
    assert roads.overlap(placement.rect)[0] == 0
    # This clear slot needs more than 48 logical pixels of escape, but a
    # connector can still associate the label with its nearby road anchor.
    left, top, right, bottom = placement.rect
    assert (top + bottom) / 2 == pytest.approx(anchor[1])
    assert min(abs(right - anchor[0]), abs(left - anchor[0])) <= 106 * scale


@pytest.mark.parametrize("grouped", [False, True])
def test_unavoidable_protected_road_does_not_send_label_to_distant_clear_land(grouped):
    class FixedWidthDraw:
        def textlength(self, _text, font=None):
            return 60

    size = (640, 180)
    anchor = (100, 90)
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle((0, 0, 300, 180), fill=255)
    roads = renderer.TrafficOccupancy(mask)
    markers = [renderer.BusMarker(("91",), *anchor, Operator.KMB, 0)]
    if grouped:
        markers.append(renderer.BusMarker(("11",), *anchor, Operator.GMB, 0))

    placement = renderer._layout_bus_labels(
        markers, FixedWidthDraw(), renderer._font(13), size, important_roads=roads,
    )[0]

    assert placement.marker == anchor
    assert roads.overlap(placement.rect)[0] > 0
    left, top, right, bottom = placement.rect
    assert (top + bottom) / 2 == pytest.approx(anchor[1])
    assert min(abs(right - anchor[0]), abs(left - anchor[0])) == pytest.approx(10)
