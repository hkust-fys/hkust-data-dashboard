"""Concurrency and failure-isolation tests for the traffic-map pipeline."""

from __future__ import annotations

import asyncio

import pytest

from dashboard.providers.route_geometry import RouteGeometry, RouteLine, Stop


def _line() -> RouteLine:
    return RouteLine(
        "91",
        "KMB",
        "outbound",
        [Stop("a", "A", 22.33, 114.26), Stop("b", "B", 22.331, 114.261)],
    )


@pytest.mark.asyncio
async def test_capture_and_probe_run_at_the_same_time(monkeypatch):
    import dashboard.maps as maps

    capture_done = asyncio.Event()
    probe_started = asyncio.Event()
    release = asyncio.Event()

    async def capture(**_kwargs):
        await release.wait()
        capture_done.set()
        return b"google-base"

    async def geometry(*_args, **_kwargs):
        return RouteGeometry(routes=[_line()])

    async def probes(*_args, **_kwargs):
        probe_started.set()
        await release.wait()
        return []

    monkeypatch.setattr(maps, "capture_gmaps_base", capture)
    monkeypatch.setattr(maps, "fetch_route_geometry", geometry)
    monkeypatch.setattr(maps, "fetch_probe_etas", probes)
    monkeypatch.setattr(maps, "render_map", lambda *args, **kwargs: b"rendered")

    operation = asyncio.create_task(maps.fetch_traffic_map(object()))
    await asyncio.wait_for(probe_started.wait(), timeout=1)
    assert not capture_done.is_set()
    release.set()
    assert await operation == (b"rendered", [])


@pytest.mark.asyncio
async def test_cancellation_cleans_capture_and_probe_tasks(monkeypatch):
    import dashboard.maps as maps

    capture_cancelled = asyncio.Event()
    probe_cancelled = asyncio.Event()
    probe_started = asyncio.Event()

    async def capture(**_kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            capture_cancelled.set()
            raise

    async def geometry(*_args, **_kwargs):
        return RouteGeometry(routes=[_line()])

    async def probes(*_args, **_kwargs):
        probe_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            probe_cancelled.set()
            raise

    monkeypatch.setattr(maps, "capture_gmaps_base", capture)
    monkeypatch.setattr(maps, "fetch_route_geometry", geometry)
    monkeypatch.setattr(maps, "fetch_probe_etas", probes)

    operation = asyncio.create_task(maps.fetch_traffic_map(object()))
    await asyncio.wait_for(probe_started.wait(), timeout=1)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert capture_cancelled.is_set()
    assert probe_cancelled.is_set()


@pytest.mark.asyncio
async def test_geometry_failure_keeps_google_base_fallback(monkeypatch):
    import dashboard.maps as maps

    captured: dict[str, object] = {}

    async def capture(**_kwargs):
        return b"google-base"

    async def geometry(*_args, **_kwargs):
        raise RuntimeError("geometry unavailable")

    def render(*args, **_kwargs):
        captured["base"] = args[4]
        captured["stops"] = args[2]
        return b"rendered"

    monkeypatch.setattr(maps, "capture_gmaps_base", capture)
    monkeypatch.setattr(maps, "fetch_route_geometry", geometry)
    monkeypatch.setattr(maps, "render_map", render)

    assert await maps.fetch_traffic_map(object()) == (b"rendered", [])
    assert captured["base"] == b"google-base"
    assert len(captured["stops"]) == 2


@pytest.mark.asyncio
async def test_probe_failure_does_not_cancel_or_hide_google_base(monkeypatch):
    import dashboard.maps as maps

    async def capture(**_kwargs):
        return b"google-base"

    async def geometry(*_args, **_kwargs):
        return RouteGeometry(routes=[_line()])

    async def probes(*_args, **_kwargs):
        raise RuntimeError("probe unavailable")

    captured: dict[str, object] = {}

    def render(*args, **_kwargs):
        captured["base"] = args[4]
        return b"rendered"

    monkeypatch.setattr(maps, "capture_gmaps_base", capture)
    monkeypatch.setattr(maps, "fetch_route_geometry", geometry)
    monkeypatch.setattr(maps, "fetch_probe_etas", probes)
    monkeypatch.setattr(maps, "render_map", render)

    assert await maps.fetch_traffic_map(object()) == (b"rendered", [])
    assert captured["base"] == b"google-base"


@pytest.mark.asyncio
async def test_estimator_failure_does_not_cancel_or_hide_google_base(monkeypatch):
    import dashboard.maps as maps

    async def capture(**_kwargs):
        return b"google-base"

    async def geometry(*_args, **_kwargs):
        return RouteGeometry(routes=[_line()])

    async def probes(*_args, **_kwargs):
        return [object()]

    captured: dict[str, object] = {}

    def failed_estimate(*_args, **_kwargs):
        raise RuntimeError("invalid probe rows")

    def render(*args, **_kwargs):
        captured["base"] = args[4]
        captured["estimates"] = args[0]
        return b"rendered"

    monkeypatch.setattr(maps, "capture_gmaps_base", capture)
    monkeypatch.setattr(maps, "fetch_route_geometry", geometry)
    monkeypatch.setattr(maps, "fetch_probe_etas", probes)
    monkeypatch.setattr(maps, "estimate_bus_positions", failed_estimate)
    monkeypatch.setattr(maps, "render_map", render)

    assert await maps.fetch_traffic_map(object()) == (b"rendered", [])
    assert captured["base"] == b"google-base"
    assert captured["estimates"] == []
