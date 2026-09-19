import asyncio
from types import SimpleNamespace

import pytest

from dashboard import pipeline


def test_retained_rthk_snapshot_expires_at_presentation_without_mutating_td():
    from datetime import UTC, datetime, timedelta

    from dashboard.models import TrafficIncident

    published = datetime(2026, 9, 19, 8, tzinfo=UTC)
    rthk = TrafficIncident(
        "rthk", "RTHK update", "RTHK-only report", "", "", "", "ACTIVE",
        source="RTHK", announcement_time=published,
    )
    td = TrafficIncident(
        "td", "TD update", "TD active notice", "", "", "", "ACTIVE",
        source="TD", page_updated_at=published, related_reports=(rthk,),
    )
    results = {"traffic": ([], [td], [], published)}

    def text(payload):
        return "\n".join(embed.description or "" for embed in payload.embeds)

    recent = pipeline.to_payload(results, now=published + timedelta(hours=2))
    aged = pipeline.to_payload(results, now=published + timedelta(hours=4))
    extended = pipeline.to_payload(
        results, now=published + timedelta(hours=4), rthk_news_max_age_hours=6,
    )
    assert "RTHK-only report" in text(recent)
    assert "RTHK-only report" not in text(aged)
    assert "TD active notice" in text(aged)
    assert "RTHK-only report" in text(extended)
    assert td.related_reports == (rthk,)


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
