import asyncio
from types import SimpleNamespace

import pytest

from dashboard import pipeline


@pytest.mark.asyncio
async def test_collect_all_publishes_settled_results_and_drains_on_cancel(monkeypatch):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def roads(*args, **kwargs):
        started.set()
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    async def quick(*args, **kwargs):
        return "ok"

    monkeypatch.setattr(pipeline.tracked_roads_provider, "fetch_tracked_roads", roads)
    monkeypatch.setattr(pipeline.transit, "fetch_transit_etas", quick)
    monkeypatch.setattr(pipeline.weather_provider, "fetch_weather_conditions", quick)
    monkeypatch.setattr(pipeline.traffic_provider, "fetch_traffic_data", quick)
    settings = SimpleNamespace(cache_dir=".cache")
    task = asyncio.create_task(pipeline.collect_all(object(), settings, include_traffic_map=False))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_shutdown_continues_after_component_failure_and_preserves_order(monkeypatch):
    calls = []

    def close(name, fail=False):
        async def operation():
            calls.append(name)
            if fail:
                raise RuntimeError(name)

        return operation

    monkeypatch.setattr(pipeline.maps, "shutdown_gmaps_browser", close("maps", True))
    monkeypatch.setattr(
        pipeline.route_geometry_provider, "shutdown_background_refreshes", close("geometry")
    )
    monkeypatch.setattr(
        pipeline.tracked_roads_provider, "shutdown_background_refreshes", close("roads")
    )
    monkeypatch.setattr(pipeline.transit, "shutdown_background_refreshes", close("transit"))
    await pipeline.shutdown_background_resources()
    assert calls == ["maps", "geometry", "roads", "transit"]
