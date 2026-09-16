from urllib.parse import parse_qs, urlsplit

import pytest

from dashboard.http import FetchError
from dashboard.providers import official_roads


@pytest.mark.asyncio
async def test_official_geometry_paginates_and_preserves_bilingual_names(monkeypatch):
    monkeypatch.setattr(official_roads, "PAGE_SIZE", 1)
    calls = []

    class Client:
        async def fetch_json(self, url):
            params = parse_qs(urlsplit(url).query)
            calls.append(params)
            return {
                "features": [{
                    "attributes": {"ENGLISHSTREETNAME": "LUNG CHEUNG ROAD", "CHINESESTREETNAME": "龍翔道"},
                    "geometry": {"paths": [[[114.20, 22.33], [114.21, 22.33]]]},
                }],
                "exceededTransferLimit": len(calls) == 1,
            }

    ways = await official_roads.fetch_official_road_ways(Client(), [[(22.33, 114.2), (22.33, 114.21)]])
    assert [call["resultOffset"] for call in calls] == [["0"], ["1"]]
    assert ways[0]["points"] == [(22.33, 114.2), (22.33, 114.21)]
    assert ways[0]["name_zh"] == "龍翔道"
    assert calls[0]["outSR"] == ["4326"]


@pytest.mark.asyncio
async def test_official_geometry_rejects_partial_or_invalid_result(monkeypatch):
    monkeypatch.setattr(official_roads, "MAX_PAGES", 2)

    class Client:
        async def fetch_json(self, url):
            if "resultOffset=0" in url:
                return {"features": [{
                    "attributes": {"ENGLISHSTREETNAME": "ROAD"},
                    "geometry": {"paths": [[[114.2, 22.3], [114.21, 22.3]]]},
                }], "exceededTransferLimit": True}
            return {"error": {"message": "source unavailable"}}

    with pytest.raises(FetchError, match="Invalid official"):
        await official_roads.fetch_official_road_ways(Client(), [[(22.3, 114.2)]])


@pytest.mark.asyncio
async def test_official_geometry_enforces_pagination_limit(monkeypatch):
    monkeypatch.setattr(official_roads, "MAX_PAGES", 1)

    class Client:
        async def fetch_json(self, url):
            return {"features": [], "exceededTransferLimit": True}

    with pytest.raises(FetchError, match="pagination limit"):
        await official_roads.fetch_official_road_ways(Client(), [[(22.3, 114.2)]])
