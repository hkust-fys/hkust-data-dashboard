"""Transit provider tests: KMB/Citybus/GMB parsing, ordering, markers, and the
verified GMB directional-variant IDs."""

import asyncio
from datetime import UTC, timedelta
from types import SimpleNamespace

import pytest

from dashboard.models import EtaKind, Operator
from dashboard.providers import transit
from dashboard.providers.transit import (
    GMB_STOPS,
    KMB_STOPS,
    _fetch_citybus,
    _fetch_gmb,
    _fetch_kmb,
    group_etas,
)
from tests.fixtures import sample_data as s


@pytest.fixture(autouse=True)
def _reset_gmb_state(monkeypatch):
    """Keep shared gate/probe cooldown and caches isolated between tests."""
    monkeypatch.setattr(transit, "_gmb_cooldown_until", 0.0)
    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 20)
    transit._gmb_gate_cache._stored = None  # noqa: SLF001
    transit._gate_eta_cache.stored = None  # noqa: SLF001
    transit._probe_cache._store.clear()  # noqa: SLF001
    monkeypatch.setattr(transit, "_gate_network_refresh_at", None)
    monkeypatch.setattr(transit, "_probe_network_refresh_at", None)
    monkeypatch.setattr(transit, "_gate_refresh_task", None)
    monkeypatch.setattr(transit, "_probe_refresh_task", None)
    monkeypatch.setattr(transit, "_gate_refresh_waiters", 0)
    monkeypatch.setattr(transit, "_probe_refresh_waiters", 0)
    # Existing parser/rotation tests intentionally model successive refreshes;
    # cadence behavior is covered explicitly by the tests below.
    monkeypatch.setattr(transit, "TRANSIT_NETWORK_REFRESH_SECONDS", 0.0)


@pytest.mark.parametrize("operator", ["KMB", "CTB", "GMB"])
def test_probe_cache_ages_cached_countdowns_between_rotated_probes(operator):
    now = [100.0]
    cache = transit.ProbeEtaCache(ttl_seconds=900, clock=lambda: now[0])
    eta = transit.ProbeEta(operator, "11", "seq-1", "stop", 3, 4)
    cache.set("probe", [eta])

    now[0] += 65
    aged = cache.get("probe")[0]
    assert aged.minutes == pytest.approx(2.9166667)
    assert aged.cache_age_seconds == 65
    assert cache._store["probe"][1] == [eta]  # noqa: SLF001


def test_probe_cache_removes_departed_cached_rows():
    now = [100.0]
    cache = transit.ProbeEtaCache(ttl_seconds=900, clock=lambda: now[0])
    cache.set("probe", [transit.ProbeEta("GMB", "11", "seq-1", "stop", 3, 1)])

    now[0] += 120
    assert cache.get("probe") == []


def test_probe_cache_expires_only_after_multi_sweep_ceiling():
    """A long TTL keeps sweep rungs together but bounds outage staleness."""
    now = [100.0]
    cache = transit.ProbeEtaCache(ttl_seconds=420, clock=lambda: now[0])
    eta = transit.ProbeEta("KMB", "91M", "inbound", "stop", 3, 1000)
    cache.set("probe", [eta])

    now[0] += 419
    assert cache.get("probe")[0].minutes == pytest.approx(993.0166667)
    now[0] += 2
    assert cache.get("probe") is None
    assert "probe" not in cache._store  # noqa: SLF001


def test_gmb_probe_parser_uses_matching_stop_sequence_and_precise_timestamp():
    now = s.utc()
    probe = SimpleNamespace(
        operator="GMB", route="104", bound="seq-1", stop_id="gate",
        route_id=2007200, sequence=1, index=0,
    )
    raw = {
        "data": [
            {
                "enabled": True, "route_id": 2007200, "route_seq": 1,
                "stop_seq": 24,
                "eta": [{"diff": 1, "timestamp": (now + timedelta(minutes=1)).isoformat()}],
            },
            {
                "enabled": True, "route_id": 2007200, "route_seq": 1,
                "stop_seq": 1,
                "eta": [
                    {
                        "diff": 99,
                        "timestamp": (now + timedelta(seconds=90)).isoformat(),
                    },
                    {"diff": 4},
                ],
            },
        ]
    }
    rows = transit._parse_probe_etas(probe, raw, now)  # noqa: SLF001
    assert [row.minutes for row in rows] == pytest.approx([1.5, 4.0])
    assert {row.index for row in rows} == {0}


def test_gmb_repeated_physical_stop_occurrences_have_distinct_cache_keys():
    first = SimpleNamespace(
        operator="GMB", route="104", bound="seq-1", stop_id="gate", index=0,
    )
    returning = SimpleNamespace(
        operator="GMB", route="104", bound="seq-1", stop_id="gate", index=23,
    )
    assert transit._probe_cache_key(first) != transit._probe_cache_key(returning)  # noqa: SLF001
    assert transit._fetch_group_key(first) == transit._fetch_group_key(returning)  # noqa: SLF001


@pytest.mark.asyncio
async def test_gmb_probe_cap_rotates_across_all_groups(monkeypatch):
    probes = [
        SimpleNamespace(
            operator="GMB", route=f"R{index}", bound="seq-1", stop_id=f"stop-{index}",
            route_id=1, sequence=1, index=0,
        )
        for index in range(35)
    ]
    calls: list[str] = []

    async def fetch(_client, probe):
        calls.append(probe.stop_id)
        return {"data": [{"enabled": True, "route_id": 1, "route_seq": 1, "eta": [{"diff": 1}]}]}

    monkeypatch.setattr(transit, "_probe_cache", transit.ProbeEtaCache())
    monkeypatch.setattr(transit, "_gmb_cursor", 0)
    monkeypatch.setattr(transit, "_probe_cursor", 0)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    # Twenty GMB groups per cycle require three cycles to cover all 35.
    for _ in range(3):
        await transit.fetch_probe_etas(object(), probes)

    assert set(calls) == {f"stop-{index}" for index in range(35)}


@pytest.mark.asyncio
async def test_gmb_probe_default_batch_limit_is_twenty(monkeypatch):
    probes = [
        SimpleNamespace(
            operator="GMB", route=f"R{index}", bound="seq-1", stop_id=f"stop-{index}",
            route_id=1, sequence=1, index=0,
        )
        for index in range(25)
    ]
    calls: list[str] = []

    async def fetch(_client, probe):
        calls.append(probe.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    await transit.fetch_probe_etas(object(), probes)

    assert len(calls) == 20


@pytest.mark.asyncio
async def test_gmb_probe_rotation_continues_after_batch_reduction(monkeypatch):
    probes = [
        SimpleNamespace(
            operator="GMB", route=f"R{index}", bound="seq-1", stop_id=f"stop-{index}",
            route_id=1, sequence=1, index=0,
        )
        for index in range(40)
    ]
    calls: list[str] = []

    async def fetch(_client, probe):
        calls.append(probe.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "_probe_cache", transit.ProbeEtaCache())
    monkeypatch.setattr(transit, "_gmb_cursor", 0)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)

    await transit.fetch_probe_etas(object(), probes)
    first_batch = set(calls)
    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 17)
    await transit.fetch_probe_etas(object(), probes)
    second_batch = set(calls[20:])

    assert len(first_batch) == 20
    assert len(second_batch) == 17
    assert first_batch.isdisjoint(second_batch)


@pytest.mark.asyncio
async def test_gmb_probe_cap_respects_smaller_cycle_budget(monkeypatch):
    probes = [
        SimpleNamespace(
            operator="GMB", route=f"R{index}", bound="seq-1", stop_id=f"stop-{index}",
            route_id=1, sequence=1, index=0,
        )
        for index in range(20)
    ]
    calls: list[str] = []

    async def fetch(_client, probe):
        calls.append(probe.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "_probe_cache", transit.ProbeEtaCache())
    monkeypatch.setattr(transit, "_gmb_cursor", 0)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)

    await transit.fetch_probe_etas(object(), probes, max_per_cycle=5)

    assert len(calls) == 5


@pytest.mark.asyncio
async def test_gmb_probe_sweep_stops_after_first_403(monkeypatch):
    from dashboard.http import FetchError

    probes = [
        SimpleNamespace(
            operator="GMB", route=f"R{index}", bound="seq-1", stop_id=f"stop-{index}",
            route_id=1, sequence=1, index=0,
        )
        for index in range(5)
    ]
    calls: list[str] = []

    async def fetch(_client, probe):
        calls.append(probe.stop_id)
        raise FetchError("HTTP 403 for GMB", status_code=403)

    monkeypatch.setattr(transit, "_probe_cache", transit.ProbeEtaCache())
    monkeypatch.setattr(transit, "_gmb_cursor", 0)
    monkeypatch.setattr(transit, "_gmb_cooldown_until", 0)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)

    await transit.fetch_probe_etas(object(), probes)

    assert len(calls) == 1
    assert transit._gmb_cooldown_until > transit.time.monotonic()  # noqa: SLF001
    assert transit.GMB_GROUPS_PER_CYCLE == 17


def test_gmb_403_decrements_once_per_cooldown_episode(monkeypatch):
    transit._record_gmb_403()
    assert transit.GMB_GROUPS_PER_CYCLE == 17
    transit._record_gmb_403()
    assert transit.GMB_GROUPS_PER_CYCLE == 17


def test_gmb_403_reduces_after_each_cooldown_and_stops_at_floor(monkeypatch):
    expected = [17, 14, 11, 8, 5, 5]
    for limit in expected:
        transit._record_gmb_403()
        assert limit == transit.GMB_GROUPS_PER_CYCLE
        monkeypatch.setattr(transit, "_gmb_cooldown_until", 0.0)


class _StubClient:
    """Minimal aiohttp-free client exposing fetch_json."""

    def __init__(self, responses: dict[str, object]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    async def fetch_json(self, url: str, headers: dict[str, str] | None = None):
        self.calls.append(url)
        for key, value in self._responses.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise AssertionError(f"unexpected URL: {url}")

    def utcnow(self):
        from datetime import datetime

        return datetime.now(UTC)


@pytest.mark.asyncio
async def test_gate_refresh_interval_serves_cached_rows(monkeypatch):
    """A ten-second render must not trigger another operator sweep."""
    calls = 0
    row = s.eta_row("91", "Diamond Hill", "S", 5)

    async def fetch(_client, _now):
        nonlocal calls
        calls += 1
        return [row]

    monkeypatch.setattr(transit, "TRANSIT_NETWORK_REFRESH_SECONDS", 30.0)
    monkeypatch.setattr(transit, "_fetch_kmb", fetch)
    monkeypatch.setattr(transit, "_fetch_citybus", fetch)
    monkeypatch.setattr(transit, "_fetch_gmb", fetch)

    class Client:
        async def gather_any(self, coroutines):
            return await asyncio.gather(*coroutines)

    import asyncio

    first, _, _ = await transit.fetch_transit_etas(Client())
    second, _, _ = await transit.fetch_transit_etas(Client())
    assert calls == 3
    assert [r.minutes for r in first[0].rows] == [5, 5, 5]
    assert [r.minutes for r in second[0].rows] == [5, 5, 5]


@pytest.mark.asyncio
async def test_probe_refresh_interval_serves_aged_cache(monkeypatch):
    probe = SimpleNamespace(
        operator="GMB", route="11", bound="seq-1", stop_id="stop",
        route_id=1, sequence=1, index=0,
    )
    calls = 0

    async def fetch(_client, _probe):
        nonlocal calls
        calls += 1
        return {"data": [{"enabled": True, "route_id": 1, "route_seq": 1,
                           "eta": [{"diff": 4}]}]}

    monkeypatch.setattr(transit, "TRANSIT_NETWORK_REFRESH_SECONDS", 30.0)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    first = await transit.fetch_probe_etas(object(), [probe])
    second = await transit.fetch_probe_etas(object(), [probe])
    assert calls == 1
    assert first[0].minutes == pytest.approx(4)
    assert second[0].minutes <= first[0].minutes


@pytest.mark.asyncio
async def test_refresh_cadence_uses_deterministic_clock_and_ages_correctly(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(transit.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        transit, "_probe_cache", transit.ProbeEtaCache(clock=lambda: clock[0])
    )
    monkeypatch.setattr(transit, "TRANSIT_NETWORK_REFRESH_SECONDS", 30.0)
    gate_calls = 0
    row = s.eta_row("91", "Diamond Hill", "S", 5)

    async def gate_fetch(_client, _now):
        nonlocal gate_calls
        gate_calls += 1
        return [row]

    monkeypatch.setattr(transit, "_fetch_kmb", gate_fetch)
    monkeypatch.setattr(transit, "_fetch_citybus", gate_fetch)
    monkeypatch.setattr(transit, "_fetch_gmb", gate_fetch)

    class Client:
        async def gather_any(self, coroutines):
            import asyncio
            return await asyncio.gather(*coroutines)

    first, _, _ = await transit.fetch_transit_etas(Client())
    clock[0] += 10
    ten_seconds, _, _ = await transit.fetch_transit_etas(Client())
    assert gate_calls == 3
    assert ten_seconds[0].rows[0].minutes == first[0].rows[0].minutes
    clock[0] += 20
    resumed, _, _ = await transit.fetch_transit_etas(Client())
    assert gate_calls == 6
    assert resumed[0].rows[0].minutes == 5

    probe = SimpleNamespace(
        operator="GMB", route="11", bound="seq-1", stop_id="probe-stop",
        route_id=1, sequence=1, index=0,
    )
    probe_calls = 0

    async def probe_fetch(_client, _probe):
        nonlocal probe_calls
        probe_calls += 1
        return {"data": [{"enabled": True, "route_id": 1, "route_seq": 1,
                           "eta": [{"diff": 4}]}]}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", probe_fetch)
    first_probe = await transit.fetch_probe_etas(object(), [probe])
    clock[0] += 10
    aged_probe = await transit.fetch_probe_etas(object(), [probe])
    assert probe_calls == 1
    assert aged_probe[0].minutes == pytest.approx(first_probe[0].minutes - 1 / 6)
    clock[0] += 20
    await transit.fetch_probe_etas(object(), [probe])
    assert probe_calls == 2


@pytest.mark.asyncio
async def test_concurrent_probe_callers_receive_only_their_requested_markers(monkeypatch):
    probes = [
        SimpleNamespace(operator="GMB", route=route, bound="seq-1", stop_id=route,
                        route_id=1, sequence=1, index=0)
        for route in ("first", "second")
    ]
    started = asyncio.Event()
    release = asyncio.Event()

    async def fetch(_client, probe):
        started.set()
        await release.wait()
        return {"data": [{"enabled": True, "route_id": 1, "route_seq": 1,
                           "eta": [{"diff": 4}]}]}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    first_task = asyncio.create_task(transit.fetch_probe_etas(object(), probes[:1]))
    await started.wait()
    second_task = asyncio.create_task(transit.fetch_probe_etas(object(), probes[1:]))
    await asyncio.sleep(0)
    release.set()
    first, second = await asyncio.gather(first_task, second_task)
    assert {eta.route for eta in first} == {"first"}
    assert second == []


@pytest.mark.asyncio
async def test_canceling_one_probe_waiter_keeps_shared_refresh_alive(monkeypatch):
    probe = SimpleNamespace(operator="GMB", route="11", bound="seq-1", stop_id="stop",
                            route_id=1, sequence=1, index=0)
    started = asyncio.Event()
    release = asyncio.Event()

    async def fetch(_client, _probe):
        started.set()
        await release.wait()
        return {"data": [{"enabled": True, "route_id": 1, "route_seq": 1,
                           "eta": [{"diff": 4}]}]}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    first = asyncio.create_task(transit.fetch_probe_etas(object(), [probe]))
    await started.wait()
    second = asyncio.create_task(transit.fetch_probe_etas(object(), [probe]))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    release.set()
    result = await second
    assert result and result[0].route == "11"


@pytest.mark.asyncio
async def test_done_probe_task_inside_cadence_returns_cache(monkeypatch):
    monkeypatch.setattr(
        transit, "_probe_cache", transit.ProbeEtaCache(clock=lambda: 100.0)
    )
    probe = SimpleNamespace(operator="GMB", route="11", bound="seq-1", stop_id="stop",
                            route_id=1, sequence=1, index=0)
    cached = transit.ProbeEta("GMB", "11", "seq-1", "stop", 0, 3)
    transit._probe_cache.set(transit._probe_cache_key(probe), [cached])  # noqa: SLF001
    transit._probe_network_refresh_at = transit.time.monotonic()  # noqa: SLF001
    transit._probe_refresh_task = asyncio.create_task(asyncio.sleep(0))  # noqa: SLF001
    await asyncio.sleep(0)

    async def must_not_fetch(_client, _probe):
        raise AssertionError("refresh should be gated")

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", must_not_fetch)
    result = await transit.fetch_probe_etas(object(), [probe])
    assert result == [cached]


@pytest.mark.asyncio
async def test_kmb_filters_service_type_and_parses_kinds():
    # stub returns the same payload to every stop URL; the provider must filter
    # by route and service_type per stop.
    client = _StubClient({"/stop-eta/": s.kmb_json()})
    rows = await _fetch_kmb(client, s.utc())
    # 91 at two stops (S Diamond Hill, N Clear Water Bay): 2 entries each
    # 91M at two stops (S Diamond Hill, N Po Lam): 1 entry each
    assert len(rows) == 6
    assert all(r.operator == Operator.KMB for r in rows)
    assert {r.route for r in rows} == {"91", "91M"}
    assert {r.gate for r in rows} == {"S", "N"}
    kinds = {r.kind for r in rows}
    assert EtaKind.SCHEDULED in kinds
    assert EtaKind.MOVING_SLOWLY in kinds
    # service_type 2 (the short-run entry) never appears
    assert not any(r.minutes == 3 for r in rows)


@pytest.mark.asyncio
async def test_kmb_empty_list_is_safe():
    client = _StubClient({"/stop-eta/": s.kmb_json_empty()})
    rows = await _fetch_kmb(client, s.utc())
    assert rows == []


@pytest.mark.asyncio
async def test_kmb_deduplicates_shared_stop_endpoints():
    client = _StubClient({"/stop-eta/": s.kmb_json_empty()})
    await _fetch_kmb(client, s.utc())
    assert len(client.calls) == len({spec["stop"] for spec in KMB_STOPS})


@pytest.mark.asyncio
async def test_citybus_handles_empty_eta_and_kmb_cycle():
    client = _StubClient({"/eta/CTB/": s.citybus_json()})
    rows = await _fetch_citybus(client, s.utc())
    # only the entry with a real eta is kept
    assert len(rows) == 1
    assert rows[0].route == "792M"
    assert rows[0].gate == "N"


@pytest.mark.asyncio
async def test_gmb_uses_verified_directional_route_id():
    """South Gate stop 20013011 must use route 2004828 for 11B (not 2004827)."""
    route_tuples = GMB_STOPS[20013011]
    route_ids = [t[3] for t in route_tuples]
    assert 2004828 in route_ids
    assert 2004827 not in route_ids

    client = _StubClient({"/eta/stop/": s.gmb_json(20013011)})
    rows = await _fetch_gmb(client, s.utc())
    rows_11b = [r for r in rows if r.route == "11B"]
    assert rows_11b, "11B should be present"
    assert all(r.gate == "S" for r in rows_11b)
    assert rows_11b[0].kind == EtaKind.REALTIME
    assert rows_11b[1].kind == EtaKind.SCHEDULED


@pytest.mark.asyncio
async def test_gmb_delayed_remark():
    client = _StubClient({"/eta/stop/": s.gmb_json()})
    rows = await _fetch_gmb(client, s.utc())
    delayed = [r for r in rows if r.route == "11S"]
    assert delayed and delayed[0].kind == EtaKind.DELAYED


@pytest.mark.asyncio
async def test_one_failed_gmb_stop_does_not_hide_other_minibuses(caplog):
    client = _StubClient(
        {
            "/eta/stop/20013010": RuntimeError("one stop unavailable"),
            "/eta/stop/": s.gmb_json(),
        }
    )
    rows = await _fetch_gmb(client, s.utc())
    assert rows
    assert "20013010" in caplog.text


@pytest.mark.asyncio
async def test_gmb_gate_403_sets_shared_cooldown_and_serves_aged_cache():
    from dashboard.http import FetchError

    transit._gmb_gate_cache.set([s.eta_row("11", "Choi Hung", "S", 5, operator=Operator.GMB)])
    stamped, cached_rows = transit._gmb_gate_cache._stored  # noqa: SLF001
    transit._gmb_gate_cache._stored = (stamped - 61, cached_rows)  # noqa: SLF001

    class Client:
        def __init__(self):
            self.calls = 0

        async def fetch_json(self, _url):
            self.calls += 1
            raise FetchError("access denied", status_code=403)

    client = Client()
    await _fetch_gmb(client, s.utc())
    assert client.calls == 1
    assert transit._gmb_cooldown_until > transit.time.monotonic()  # noqa: SLF001

    rows = await _fetch_gmb(client, s.utc())
    assert client.calls == 1
    assert rows and rows[0].minutes == 4


@pytest.mark.asyncio
async def test_gmb_gate_cooldown_expiry_resumes_polling():
    transit._gmb_cooldown_until = transit.time.monotonic() + 60
    client = _StubClient({"/eta/stop/": s.gmb_json()})
    assert await _fetch_gmb(client, s.utc()) == []
    assert client.calls == []

    transit._gmb_cooldown_until = 0
    rows = await _fetch_gmb(client, s.utc())
    assert rows
    assert client.calls


def test_gmb_gate_cache_has_hard_ttl_for_live_and_unknown_etas():
    transit._gmb_gate_cache.set([
        s.eta_row("11", "Choi Hung", "S", 5, operator=Operator.GMB),
        s.eta_row("11B", "Choi Hung", "S", None, operator=Operator.GMB),
    ])
    stamped, rows = transit._gmb_gate_cache._stored  # noqa: SLF001
    transit._gmb_gate_cache._stored = (
        stamped - transit._gmb_gate_cache.TTL_SECONDS - 1,
        rows,
    )  # noqa: SLF001

    assert transit._gmb_gate_cache.get() == []
    assert transit._gmb_gate_cache._stored is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_concurrent_gate_and_probe_403_stop_followup_calls():
    import asyncio

    from dashboard.http import FetchError

    calls: list[str] = []
    first_started = asyncio.Event()
    release = asyncio.Event()

    class Client:
        async def fetch_json(self, url):
            calls.append(url)
            if len(calls) == 1:
                first_started.set()
                await release.wait()
            raise FetchError("access denied", status_code=403)

    probe = SimpleNamespace(
        operator="GMB", route="11", bound="seq-1", stop_id="20013010",
        route_id=2004791, sequence=1, index=0,
    )
    client = Client()
    gate_task = asyncio.create_task(_fetch_gmb(client, s.utc()))
    await first_started.wait()
    probe_task = asyncio.create_task(transit.fetch_probe_etas(client, [probe]))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(gate_task, probe_task)

    # One request may already be in flight on the other path; neither path
    # may start a second request after the shared cooldown is set.
    assert len(calls) <= 2
    assert transit.GMB_GROUPS_PER_CYCLE == 17


def test_gmb_config_has_both_gates_and_verified_stops():
    """The config covers the verified stop IDs and both 11B directional variants."""
    assert 20013010 in GMB_STOPS  # HKUST(S) 11
    assert 20012472 in GMB_STOPS  # HKUST(N) 11
    assert 20012474 in GMB_STOPS  # HKUST(N) 11M/11S/12
    assert 20015226 in GMB_STOPS  # HKUST(S) 104
    # South Gate 11B route 2004828 (the boarding direction); 2004827 is the
    # non-boarding variant at North Gate and is intentionally absent.
    south_11b = [t for t in GMB_STOPS[20013011] if t[0] == "11B"]
    assert south_11b and south_11b[0][3] == 2004828


def test_group_etas_stable_order_and_merging():
    rows = [
        s.eta_row("11B", "Choi Hung", "S", 5, EtaKind.SCHEDULED, Operator.GMB),
        s.eta_row("91", "Diamond Hill", "S", 2),
        s.eta_row("91", "Diamond Hill", "S", 20),
        s.eta_row("91M", "Po Lam", "N", 3),
        s.eta_row("792M", "Sai Kung", "N", 6, operator=Operator.CITYBUS),
        s.eta_row("291P", "Mong Kok", "S", 10),
        s.eta_row("12", "Po Lam", "N", 4, operator=Operator.GMB),
    ]
    groups = group_etas(rows)
    # North first, then South
    gates = [g.gate for g in groups]
    assert gates == ["N", "N", "N", "S", "S", "S"]
    # within North: buses before minibuses, numeric within operator
    assert [g.route for g in groups[:3]] == ["91M", "792M", "12"]
    # within South: KMB (91, 291P) before GMB (11B)
    assert [g.route for g in groups[3:]] == ["91", "291P", "11B"]
    # 91 rows merged into one group
    grp_91 = [g for g in groups if g.route == "91"][0]
    assert len(grp_91.rows) == 2


def test_group_etas_splits_gmb_circular_stops_by_stop_seq():
    """A circular GMB route (e.g. 104) appears at multiple stop_seq along its
    loop; the far-end stop's ETAs are the bus RETURNING to HKUST and must not
    be merged into the departure stop's group (which would make later ETAs
    drop below earlier ones)."""
    from dashboard.models import EtaRow, Operator

    # the departure stop at HKUST: 0, 21, 46
    departure = [
        EtaRow(
            route="104",
            destination="Kwun Tong",
            gate="S",
            operator=Operator.GMB,
            minutes=0,
            stop_seq=1,
        ),
        EtaRow(
            route="104",
            destination="Kwun Tong",
            gate="S",
            operator=Operator.GMB,
            minutes=21,
            stop_seq=1,
        ),
        EtaRow(
            route="104",
            destination="Kwun Tong",
            gate="S",
            operator=Operator.GMB,
            minutes=46,
            stop_seq=1,
        ),
    ]
    # the far-end stop (loop return): 30, 45 — must NOT appear
    loopback = [
        EtaRow(
            route="104",
            destination="Kwun Tong",
            gate="S",
            operator=Operator.GMB,
            minutes=30,
            stop_seq=24,
        ),
        EtaRow(
            route="104",
            destination="Kwun Tong",
            gate="S",
            operator=Operator.GMB,
            minutes=45,
            stop_seq=24,
        ),
    ]
    groups = group_etas(departure + loopback)
    g104 = [g for g in groups if g.route == "104"]
    # two distinct groups: departure stop and loopback stop
    assert len(g104) == 2
    dep = [g for g in g104 if g.stop_seq == 1][0]
    assert [r.minutes for r in dep.rows] == [0, 21, 46]  # strictly increasing
    ret = [g for g in g104 if g.stop_seq == 24][0]
    assert [r.minutes for r in ret.rows] == [30, 45]


def test_route_sort_key_numeric():
    from dashboard.providers.transit import _route_sort_key

    assert _route_sort_key("91") < _route_sort_key("291P")
    assert (
        _route_sort_key("11")
        < _route_sort_key("11B")
        < _route_sort_key("12")
        < _route_sort_key("104")
    )


def test_gmb_config_iteration_is_deterministic():
    # the config dict order is preserved (no sorting/reshuffle on load)
    keys = list(GMB_STOPS.keys())
    assert keys == [20013010, 20012472, 20013011, 20012474, 20015226]
