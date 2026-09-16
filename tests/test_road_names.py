import asyncio
import json
import time
from urllib.parse import parse_qs, urlsplit

import pytest

from dashboard.providers import road_names


def _feature(english: object, chinese: object) -> dict:
    return {
        "attributes": {
            "ENGLISHSTREETNAME": english,
            "CHINESESTREETNAME": chinese,
        }
    }


def _page(*features: dict, exceeded: bool = False) -> dict:
    return {"features": list(features), "exceededTransferLimit": exceeded}


class QueueClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, object, int]] = []

    async def fetch_json(self, url, headers=None, max_bytes=None):
        self.calls.append((url, headers, max_bytes))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


@pytest.fixture(autouse=True)
async def _reset_provider_state():
    await road_names.shutdown_background_refreshes()
    yield
    await road_names.shutdown_background_refreshes()


def test_normalize_name_handles_unicode_quotes_width_and_spacing():
    assert road_names.normalize_name("  HIRAM\u2019S\t Highway  ") == "hiram's highway"
    assert road_names.normalize_name("ＮＥＷ  Clear Water Bay Road") == (
        "new clear water bay road"
    )


@pytest.mark.asyncio
async def test_fetch_paginates_and_persists_complete_dictionary(tmp_path):
    client = QueueClient(
        [
            _page(
                _feature("Clear Water Bay Road", "清水灣道"),
                _feature("Hiram’s Highway", "西貢公路"),
                exceeded=True,
            ),
            _page(
                _feature("HIRAM'S HIGHWAY", "西貢公路"),
                _feature("New Clear Water Bay Road", "新清水灣道"),
            ),
        ]
    )

    result = await road_names.fetch_road_names(
        client, cache_dir=str(tmp_path), wait_for_refresh=True
    )

    assert result == {
        "clear water bay road": ("清水灣道",),
        "hiram's highway": ("西貢公路",),
        "new clear water bay road": ("新清水灣道",),
    }
    offsets = [parse_qs(urlsplit(call[0]).query)["resultOffset"] for call in client.calls]
    assert offsets == [["0"], ["2"]]
    assert all(call[1] is None for call in client.calls)
    assert all(call[2] == road_names.ROAD_NAMES_MAX_BYTES for call in client.calls)
    cached = json.loads(
        (tmp_path / "maps" / road_names.ROAD_NAMES_CACHE_NAME).read_text(encoding="utf-8")
    )
    assert cached["version"] == road_names.ROAD_NAMES_CACHE_VERSION
    assert cached["names"]["hiram's highway"] == ["西貢公路"]


@pytest.mark.asyncio
async def test_invalid_partial_refresh_keeps_last_good_cache(tmp_path):
    cache_dir = str(tmp_path)
    retained = road_names._CacheRecord(  # noqa: SLF001
        {"po lam road": ("寶琳路",)},
        time.time() - road_names.ROAD_NAMES_TTL_SECONDS - 1,
    )
    road_names._save_disk_cache(retained, cache_dir)  # noqa: SLF001
    road_names._startup_attempted.add(cache_dir)  # noqa: SLF001
    client = QueueClient(
        [
            _page(_feature("Clear Water Bay Road", "清水灣道"), exceeded=True),
            {"error": {"message": "upstream failure"}},
        ]
    )

    result = await road_names.fetch_road_names(
        client, cache_dir=cache_dir, wait_for_refresh=True
    )

    assert result == retained.names
    raw = json.loads(
        (tmp_path / "maps" / road_names.ROAD_NAMES_CACHE_NAME).read_text(encoding="utf-8")
    )
    assert raw["names"] == {"po lam road": ["寶琳路"]}
    assert cache_dir in road_names._refresh_retry_after  # noqa: SLF001


@pytest.mark.asyncio
async def test_cache_older_than_stale_limit_falls_back_on_network_failure(tmp_path):
    cache_dir = str(tmp_path)
    expired = road_names._CacheRecord(  # noqa: SLF001
        {"po lam road": ("寶琳路",)},
        time.time() - road_names.ROAD_NAMES_STALE_SECONDS - 1,
    )
    road_names._save_disk_cache(expired, cache_dir)  # noqa: SLF001
    client = QueueClient([RuntimeError("offline")])

    result = await road_names.fetch_road_names(
        client, cache_dir=cache_dir, wait_for_refresh=True
    )

    assert result == road_names.FALLBACK_ROAD_NAMES
    assert "po lam road" not in result


@pytest.mark.asyncio
async def test_startup_returns_saved_cache_and_starts_only_one_refresh(tmp_path):
    cache_dir = str(tmp_path)
    retained = road_names._CacheRecord(  # noqa: SLF001
        {"po lam road": ("寶琳路",)}, time.time()
    )
    road_names._save_disk_cache(retained, cache_dir)  # noqa: SLF001
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingClient:
        def __init__(self) -> None:
            self.calls = 0

        async def fetch_json(self, _url, headers=None, max_bytes=None):
            self.calls += 1
            started.set()
            await release.wait()
            return _page(_feature("Po Lam Road", "寶琳路"))

    client = BlockingClient()

    first = await road_names.fetch_road_names(client, cache_dir=cache_dir)
    await asyncio.wait_for(started.wait(), timeout=1)
    second = await road_names.fetch_road_names(client, cache_dir=cache_dir)

    assert first == retained.names
    assert second == retained.names
    assert client.calls == 1
    assert cache_dir in road_names._refresh_tasks  # noqa: SLF001

    release.set()
    await asyncio.wait_for(road_names._refresh_tasks[cache_dir], timeout=1)  # noqa: SLF001
    await asyncio.sleep(0)
    assert client.calls == 1
    assert await road_names.fetch_road_names(client, cache_dir=cache_dir) == retained.names
    assert client.calls == 1


@pytest.mark.asyncio
async def test_refreshes_again_after_24_hours_but_returns_cache_nonblocking(tmp_path):
    cache_dir = str(tmp_path)
    stale = road_names._CacheRecord(  # noqa: SLF001
        {"po lam road": ("寶琳路",)},
        time.time() - road_names.ROAD_NAMES_TTL_SECONDS - 1,
    )
    road_names._save_disk_cache(stale, cache_dir)  # noqa: SLF001
    road_names._startup_attempted.add(cache_dir)  # noqa: SLF001
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingClient:
        async def fetch_json(self, _url, headers=None, max_bytes=None):
            started.set()
            await release.wait()
            return _page(_feature("Clear Water Bay Road", "清水灣道"))

    result = await road_names.fetch_road_names(BlockingClient(), cache_dir=cache_dir)
    await asyncio.wait_for(started.wait(), timeout=1)

    assert result == stale.names
    assert cache_dir in road_names._refresh_tasks  # noqa: SLF001
    release.set()
    await asyncio.wait_for(road_names._refresh_tasks[cache_dir], timeout=1)  # noqa: SLF001


@pytest.mark.asyncio
async def test_shutdown_cancels_and_drains_background_refresh():
    cache_dir = "shutdown-road-names"
    task = asyncio.create_task(asyncio.sleep(60))
    road_names._refresh_tasks[cache_dir] = task  # noqa: SLF001
    road_names._startup_attempted.add(cache_dir)  # noqa: SLF001

    await road_names.shutdown_background_refreshes()

    assert task.cancelled()
    assert road_names._refresh_tasks == {}  # noqa: SLF001
    assert road_names._refresh_retry_after == {}  # noqa: SLF001
    assert road_names._startup_attempted == set()  # noqa: SLF001
