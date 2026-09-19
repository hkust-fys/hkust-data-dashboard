"""Transit provider tests: KMB/Citybus/GMB parsing, ordering, markers, and the
verified GMB directional-variant IDs."""

import asyncio
from dataclasses import fields
from datetime import UTC, timedelta
from types import SimpleNamespace

import pytest

from dashboard.http import RequestNotStarted
from dashboard.maps.positions import estimate_bus_positions
from dashboard.maps.tracker import MarkerTracker
from dashboard.models import EtaKind, Operator
from dashboard.providers import transit
from dashboard.providers.route_geometry import ProbeStop, RouteLine, Stop
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
    transit._probe_route_generations.clear()  # noqa: SLF001
    transit._probe_group_versions.clear()  # noqa: SLF001
    transit._probe_group_rows.clear()  # noqa: SLF001
    transit._probe_route_published_versions.clear()  # noqa: SLF001
    transit._probe_route_version_floors.clear()  # noqa: SLF001
    transit._probe_group_service_debt.clear()  # noqa: SLF001
    transit._probe_failed_groups.clear()  # noqa: SLF001
    monkeypatch.setattr(transit, "_probe_generation", 0)
    monkeypatch.setattr(transit, "_probe_attempt_generation", 0)
    monkeypatch.setattr(transit, "_probe_attempted_checkpoints", frozenset())
    monkeypatch.setattr(transit, "_probe_cold_cursor", 0)
    monkeypatch.setattr(transit, "_probe_priority_cursor", 0)
    transit._probe_priority_owed.clear()  # noqa: SLF001
    monkeypatch.setattr(transit, "_probe_background_cursor", 0)
    monkeypatch.setattr(transit, "_probe_background_gmb_cursor", 0)
    monkeypatch.setattr(transit, "_gate_network_refresh_at", None)
    monkeypatch.setattr(transit, "_probe_network_refresh_at", None)
    monkeypatch.setattr(transit, "_gate_refresh_task", None)
    monkeypatch.setattr(transit, "_probe_refresh_task", None)
    monkeypatch.setattr(transit, "_gate_refresh_waiters", 0)
    monkeypatch.setattr(transit, "_probe_refresh_waiters", 0)
    # Existing parser/rotation tests intentionally model successive refreshes;
    # cadence behavior is covered explicitly by the tests below.
    monkeypatch.setattr(transit, "TRANSIT_NETWORK_REFRESH_SECONDS", 0.0)


def _seed_warm_routes(probes):
    """Seed a genuinely published empty generation for quota-only tests."""
    grouped = {}
    for probe in probes:
        key = (probe.operator, probe.route, probe.bound)
        grouped.setdefault(key, []).append(probe)
    for route_key, route_probes in grouped.items():
        groups = {transit._fetch_group_key(probe) for probe in route_probes}  # noqa: SLF001
        versions = {group: 0 for group in groups}
        public = transit.ProbeRouteGeneration(
            route_key, (), 1, s.utc(),
            frozenset(probe.index for probe in route_probes),
        )
        transit._probe_route_generations[route_key] = transit._StoredProbeGeneration(  # noqa: SLF001
            public=public,
            topology_keys=frozenset(transit._probe_topology_key(probe) for probe in route_probes),  # noqa: SLF001
            group_keys=frozenset(groups),
            published_monotonic=0.0,
        )
        transit._probe_route_published_versions[route_key] = versions  # noqa: SLF001
        transit._probe_route_version_floors[route_key] = dict(versions)  # noqa: SLF001


@pytest.mark.parametrize("operator", ["KMB", "CTB", "GMB"])
def test_probe_cache_ages_cached_countdowns_between_rotated_probes(operator):
    now = [100.0]
    cache = transit.ProbeEtaCache(ttl_seconds=900, clock=lambda: now[0])
    eta = transit.ProbeEta(operator, "11", "seq-1", "stop", 3, 4)
    cache.set("probe", [eta])

    now[0] += 65
    aged = cache.get("probe")[0]
    assert aged.minutes == pytest.approx(4)
    assert aged.cache_age_seconds == 65
    assert cache._store["probe"][1] == [eta]  # noqa: SLF001


def test_probe_cache_removes_departed_cached_rows():
    now = [100.0]
    cache = transit.ProbeEtaCache(ttl_seconds=900, clock=lambda: now[0])
    cache.set("probe", [transit.ProbeEta("GMB", "11", "seq-1", "stop", 3, 1)])

    now[0] += 120
    assert cache.get("probe")[0].minutes == pytest.approx(1)


def test_probe_cache_expires_only_after_multi_sweep_ceiling():
    """A long TTL keeps sweep rungs together but bounds outage staleness."""
    now = [100.0]
    cache = transit.ProbeEtaCache(ttl_seconds=420, clock=lambda: now[0])
    eta = transit.ProbeEta("KMB", "91M", "inbound", "stop", 3, 1000)
    cache.set("probe", [eta])

    now[0] += 419
    assert cache.get("probe")[0].minutes == pytest.approx(1000)
    now[0] += 2
    assert cache.get("probe") is None
    assert "probe" not in cache._store  # noqa: SLF001


def test_probe_cache_uses_injected_wall_clock_for_absolute_age_and_expiry():
    mono = [0.0]
    wall = [s.utc()]
    cache = transit.ProbeEtaCache(clock=lambda: mono[0], wall_clock=lambda: wall[0])
    arrival = wall[0] + timedelta(seconds=90)
    eta = transit.ProbeEta("KMB", "X", "outbound", "stop", 1, 1.5, arrival_at=arrival)
    cache.set("probe", [eta])
    wall[0] += timedelta(seconds=30)
    assert cache.get("probe")[0].minutes == pytest.approx(1.5)
    wall[0] += timedelta(seconds=61)
    assert cache.get("probe")[0].minutes == pytest.approx(1.5)


def test_probe_cache_defensively_copies_source_rows():
    cache = transit.ProbeEtaCache()
    rows = [transit.ProbeEta("KMB", "X", "outbound", "stop", 1, 5)]
    cache.set("probe", rows)
    rows.clear()
    assert cache.get("probe")[0].minutes == 5


def test_probe_cache_retains_revision_for_successful_empty_response():
    cache = transit.ProbeEtaCache(clock=lambda: 100.0)
    cache.set("shared-stop", [], revision=41)
    assert cache.get("shared-stop") == []
    assert cache.revision("shared-stop") == 41


def test_probe_generation_public_shape_is_small_and_topology_is_canonical():
    assert {field.name for field in fields(transit.ProbeRouteGeneration)} == {
        "route_key", "rows", "generation", "collected_at",
        "observed_checkpoint_indices", "checkpoint_revisions",
    }
    first = SimpleNamespace(
        operator="GMB", route="11", bound="seq-1", stop_id="stop-a", index=2,
        route_id=7, sequence=3,
    )
    reordered = SimpleNamespace(
        operator="GMB", route="11", bound="seq-1", stop_id="stop-a", index=2,
        route_id=7, sequence=3,
    )
    changed = SimpleNamespace(
        operator="GMB", route="11", bound="seq-1", stop_id="stop-a", index=2,
        route_id=8, sequence=3,
    )
    assert transit._probe_topology_key(first) == transit._probe_topology_key(reordered)
    assert transit._probe_topology_key(first) != transit._probe_topology_key(changed)


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
    assert rows[1].arrival_at is None


def test_gmb_probe_parser_preserves_equal_consecutive_eta_entries():
    now = s.utc()
    arrival = now + timedelta(minutes=2)
    probe = SimpleNamespace(
        operator="GMB", route="11", bound="seq-1", stop_id="stop",
        route_id=2004791, sequence=1, index=8,
    )
    raw = {
        "data": [
            {
                "enabled": True,
                "route_id": 2004791,
                "route_seq": 1,
                "stop_seq": 9,
                "eta": [
                    {"eta_seq": 1, "diff": 2, "timestamp": arrival.isoformat()},
                    {"eta_seq": 2, "diff": 2, "timestamp": arrival.isoformat()},
                ],
            },
            {
                "enabled": True,
                "route_id": 2004791,
                "route_seq": 2,
                "stop_seq": 9,
                "eta": [
                    {"eta_seq": 1, "diff": 2, "timestamp": arrival.isoformat()},
                ],
            },
        ]
    }

    rows = transit._parse_probe_etas(probe, raw, now)  # noqa: SLF001

    assert len(rows) == 2
    assert [row.arrival_at for row in rows] == [arrival, arrival]
    assert {row.bound for row in rows} == {"seq-1"}


@pytest.mark.asyncio
async def test_probe_snapshot_preserves_source_minutes_and_ages_cache_metadata(monkeypatch):
    mono = [10.0]
    wall = [s.utc()]
    monkeypatch.setattr(transit.time, "monotonic", lambda: mono[0])
    monkeypatch.setattr(transit, "_probe_mono_clock", lambda: mono[0])
    monkeypatch.setattr(transit, "_probe_wall_clock", lambda: wall[0])
    monkeypatch.setattr(
        transit, "_probe_cache", transit.ProbeEtaCache(
            clock=lambda: mono[0]
        )
    )
    monkeypatch.setattr(transit, "TRANSIT_NETWORK_REFRESH_SECONDS", 1000.0)
    probe = SimpleNamespace(operator="GMB", route="R", bound="b", stop_id="s",
                            route_id=1, sequence=1, index=0)

    async def fetch(_client, _probe):
        return {"data": [{"enabled": True, "route_id": 1, "route_seq": 1,
                           "eta": [{"diff": 2}]}]}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    await transit.fetch_probe_snapshot(object(), [probe], max_per_cycle=1)
    mono[0] += 30
    wall[0] += timedelta(seconds=30)
    second = await transit.fetch_probe_snapshot(object(), [probe], max_per_cycle=1)
    assert second.rows[0].minutes == pytest.approx(2)
    mono[0] += 90
    wall[0] += timedelta(seconds=90)
    aged = await transit.fetch_probe_snapshot(object(), [probe], max_per_cycle=1)
    assert aged.rows[0].minutes == pytest.approx(2)
    assert aged.rows[0].cache_age_seconds == pytest.approx(120)


def test_probe_parser_preserves_absolute_eta_and_observation_times():
    now = s.utc()
    probe = SimpleNamespace(
        operator="KMB", route="91", bound="outbound", stop_id="stop", index=2,
    )
    arrival = now + timedelta(minutes=4)
    observed = now - timedelta(seconds=3)
    rows = transit._parse_probe_etas(
        probe,
        {"data": [{"route": "91", "dir": "O", "seq": 3,
                   "eta": arrival.isoformat(),
                   "data_timestamp": observed.isoformat()}]},
        now,
    )
    assert len(rows) == 1
    assert rows[0].arrival_at == arrival
    assert rows[0].observed_at == observed


def test_probe_parser_preserves_negative_eta_offset_while_countdown_stays_zero():
    now = s.utc()
    probe = SimpleNamespace(
        operator="KMB", route="91M", bound="outbound", stop_id="stop", index=7,
    )
    arrival = now - timedelta(seconds=36)
    rows = transit._parse_probe_etas(
        probe,
        {"data": [{"route": "91M", "dir": "O", "service_type": 1,
                   "seq": 8, "eta": arrival.isoformat()}]},
        now,
    )

    assert len(rows) == 1
    assert rows[0].minutes == 0
    assert rows[0].signed_minutes == pytest.approx(-0.6)


@pytest.mark.parametrize("offset", [-.25, .25, None])
def test_gmb_rounded_zero_preserves_precise_timestamp_or_rounding_uncertainty(offset):
    now = s.utc()
    probe = ProbeStop("GMB", "11", "seq-1", "stop", 1, 1, 3)
    eta = {"diff": 0}
    if offset is not None:
        eta["timestamp"] = (now + timedelta(minutes=offset)).isoformat()
    raw = {"data": [{"route_id": 1, "route_seq": 1, "stop_seq": 4, "eta": [eta]}]}
    parsed = transit._parse_probe_etas(probe, raw, now)[0]
    assert parsed.countdown_rounded is (offset is None)
    assert parsed.signed_minutes == offset
    assert parsed.minutes == (max(0, offset) if offset is not None else 0)


@pytest.mark.asyncio
async def test_terminus_response_drives_upstream_discovery_and_adjacent_http_queries(monkeypatch):
    probes = [ProbeStop("GMB", "11", "seq-1", str(index), 1, 1, index) for index in range(13)]
    now = s.utc()
    calls = []
    monkeypatch.setattr(transit, "_probe_wall_clock", lambda: now)
    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 8)

    async def fetch(_client, probe):
        calls.append(probe.index)
        minutes = (4, 8, 12) if probe.index == 12 else ((2, 7, 12) if probe.index == 6 else (1,))
        return {"data": [{"route_id": 1, "route_seq": 1, "stop_seq": probe.index + 1,
                          "eta": [{"timestamp": (now + timedelta(minutes=value)).isoformat()}
                                  for value in minutes]}]}

    planning_rows = []

    def local_queries(rows):
        planning_rows.append({row.index for row in rows})
        return {("GMB", "11", "seq-1"): ((9, 10),)} if rows else {}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    result = await transit.fetch_probe_snapshot(
        object(), probes, max_per_cycle=8, generation_probes=[probes[-1]],
        terminus_first=True, neighbour_pairs=local_queries,
    )
    assert calls[0] == 12
    assert 6 in calls and 0 in calls  # the third row hid the upstream population
    assert calls.index(10) == calls.index(9) + 1
    assert len(calls) == len(set(calls)) <= 8
    assert any({12, 6} <= indices for indices in planning_rows)
    assert {9, 10} <= {row.index for row in result.rows}


@pytest.mark.asyncio
async def test_zero_span_queries_are_not_split_to_fill_the_last_budget_slot(monkeypatch):
    probes = [ProbeStop("GMB", "11", "seq-1", str(index), 1, 1, index) for index in range(7)]
    calls = []

    async def fetch(_client, probe):
        calls.append(probe.index)
        return {"data": [{"route_id": 1, "route_seq": 1, "eta": [{"diff": 1}]}]}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    kwargs = dict(generation_probes=[probes[-1]], terminus_first=True,
                  neighbour_pairs={("GMB", "11", "seq-1"): ((2, 3, 4),)})
    await transit.fetch_probe_snapshot(object(), probes, max_per_cycle=2, **kwargs)
    assert calls == [6]
    calls.clear()
    await transit.fetch_probe_snapshot(object(), probes, max_per_cycle=4, **kwargs)
    assert calls == [6, 2, 3, 4]


@pytest.mark.asyncio
async def test_kmb_adaptive_termini_share_one_route_request(monkeypatch):
    probes = [ProbeStop("KMB", "91", bound, str(index), None, None, index)
              for bound in ("outbound", "inbound") for index in range(6)]
    calls = []

    async def fetch(_client, probe):
        calls.append(probe)
        return {"data": []}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    result = await transit.fetch_probe_snapshot(object(), probes, terminus_first=True)
    assert len(calls) == 1
    assert len(result.positioning_checkpoints) == 12


def test_adaptive_presentation_excludes_expired_interior_responses_without_aging_countdowns(monkeypatch):
    mono = [0.0]
    cache = transit.ProbeEtaCache(clock=lambda: mono[0])
    monkeypatch.setattr(transit, "_probe_cache", cache)
    probe = ProbeStop("GMB", "11", "seq-1", "stop", 1, 1, 3)
    row = transit.ProbeEta("GMB", "11", "seq-1", "stop", 3, 1, refresh_generation=1)
    key = transit._probe_cache_key(probe)
    cache.set(key, [row], revision=1)
    mono[0] = 20
    assert transit.read_probe_snapshot([probe], positioning_max_age_seconds=60).rows[0].minutes == 1
    mono[0] = 60
    assert transit.read_probe_snapshot([probe], positioning_max_age_seconds=60).rows == ()
    assert cache.get(key)[0].minutes == 1


@pytest.mark.asyncio
async def test_stale_empty_checkpoint_does_not_stop_upstream_discovery(monkeypatch):
    probes = [ProbeStop("GMB", "11", "seq-1", str(index), 1, 1, index) for index in range(7)]
    mono = [0.0]
    cache = transit.ProbeEtaCache(clock=lambda: mono[0])
    monkeypatch.setattr(transit, "_probe_cache", cache)
    cache.set(transit._probe_cache_key(probes[3]), [], revision=1)
    mono[0] = 61
    calls = []

    async def fetch(_client, probe):
        calls.append(probe.index)
        return {"data": [{"route_id": 1, "route_seq": 1,
                          "eta": [{"diff": value} for value in ((1, 2, 6) if probe.index == 6 else ())]}]}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    await transit.fetch_probe_snapshot(object(), probes, max_per_cycle=2,
                                       generation_probes=[probes[-1]], terminus_first=True)
    assert calls == [6, 3]


@pytest.mark.asyncio
async def test_gmb_cooldown_does_not_block_other_operators_local_sampling(monkeypatch):
    ctb = [ProbeStop("CTB", "X", "outbound", str(index), None, None, index) for index in range(5)]
    gmb = [ProbeStop("GMB", "11", "seq-1", "gmb", 1, 1, 0)]
    monkeypatch.setattr(transit, "_gmb_cooldown_until", float("inf"))
    calls = []

    async def fetch(_client, probe):
        calls.append((probe.operator, probe.index))
        return {"data": []}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    await transit.fetch_probe_snapshot(
        object(), [*ctb, *gmb], max_per_cycle=3, generation_probes=[ctb[-1], gmb[-1]],
        terminus_first=True, neighbour_pairs={("CTB", "X", "outbound"): ((1, 2),)},
    )
    assert calls == [("CTB", 4), ("CTB", 1), ("CTB", 2)]


def test_kmb_route_eta_parser_filters_by_one_based_sequence():
    now = s.utc()
    probe = SimpleNamespace(
        operator="KMB", route="91M", bound="outbound", stop_id="official-stop", index=2,
    )
    raw = {"data": [
        {"route": "91M", "dir": "O", "service_type": 1, "seq": 2,
         "eta": (now + timedelta(minutes=2)).isoformat()},
        {"route": "91M", "dir": "O", "service_type": 1, "seq": 3,
         "eta": (now + timedelta(minutes=3)).isoformat()},
    ]}
    rows = transit._parse_probe_etas(probe, raw, now)  # noqa: SLF001
    assert len(rows) == 1
    assert rows[0].index == 2
    assert rows[0].minutes == pytest.approx(3)


@pytest.mark.asyncio
async def test_priority_refresh_is_visible_before_next_complete_generation(monkeypatch):
    now = s.utc()
    route_key = ("CTB", "R", "outbound")
    probes = [
        SimpleNamespace(
            operator="CTB", route="R", bound="outbound", stop_id=f"s{index}",
            route_id=1, sequence=1, index=index,
        )
        for index in range(3)
    ]
    minutes = {"s0": 8, "s1": 10, "s2": 12}

    async def fetch(_client, probe):
        return {"data": [{
            "dir": "O",
            "eta": (now + timedelta(minutes=minutes[probe.stop_id])).isoformat(),
            "data_timestamp": now.isoformat(),
        }]}

    monkeypatch.setattr(transit, "_probe_wall_clock", lambda: now)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    first = await transit.fetch_probe_snapshot(object(), probes, max_per_cycle=3)
    assert len(first.complete_routes) == 1
    generation = first.complete_routes[0].generation

    minutes["s0"] = 3
    second = await transit.fetch_probe_snapshot(
        object(), probes, max_per_cycle=2, priorities={route_key: {0}}
    )
    assert second.complete_routes[0].generation == generation
    assert {row.index: row.minutes for row in second.positioning_rows}[0] == 3
    assert {row.index: row.minutes for row in second.complete_routes[0].rows}[0] == 8


@pytest.mark.asyncio
async def test_expired_generation_and_published_versions_are_evicted_together(monkeypatch):
    key = ("GMB", "R", "seq-1")
    public = transit.ProbeRouteGeneration(key, (), 1, s.utc())
    transit._probe_route_generations[key] = transit._StoredProbeGeneration(  # noqa: SLF001
        public=public, topology_keys=frozenset(), group_keys=frozenset({"GMB:stop"}),
        published_monotonic=10.0,
    )
    transit._probe_route_published_versions[key] = {"GMB:stop": 1}  # noqa: SLF001

    async def no_refresh(*args, **kwargs):
        return []

    monkeypatch.setattr(transit, "_probe_mono_clock", lambda: 1000.0)
    monkeypatch.setattr(transit, "fetch_probe_etas", no_refresh)
    await transit.fetch_probe_snapshot(object(), [], wait_for_refresh=False)
    assert key not in transit._probe_route_generations  # noqa: SLF001
    assert key not in transit._probe_route_published_versions  # noqa: SLF001


@pytest.mark.asyncio
async def test_ttl_expiry_requires_full_fresh_rebootstrap_before_republish(monkeypatch):
    mono = [100.0]
    wall = [s.utc()]
    monkeypatch.setattr(transit, "_probe_mono_clock", lambda: mono[0])
    monkeypatch.setattr(transit, "_probe_wall_clock", lambda: wall[0])
    monkeypatch.setattr(transit, "PROBE_GENERATION_TTL_SECONDS", 10.0)
    monkeypatch.setattr(
        transit, "_probe_cache", transit.ProbeEtaCache(
            ttl_seconds=10.0, clock=lambda: mono[0]
        )
    )
    probes = [SimpleNamespace(
        operator="GMB", route="TTL", bound="seq-1", stop_id=f"ttl-{i}",
        route_id=1, sequence=1, index=i,
    ) for i in range(4)]
    route_key = ("GMB", "TTL", "seq-1")
    calls: list[str] = []
    successful_fresh: set[str] = set()
    fail_anchor = [False]
    rebootstrap = [False]

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        if fail_anchor[0] and selected.stop_id == "ttl-0":
            raise RuntimeError("anchor unavailable")
        successful_fresh.add(selected.stop_id)
        if rebootstrap[0] and selected.stop_id != "ttl-0":
            return {"data": []}
        return {"data": [{"enabled": True, "route_id": 1, "route_seq": 1,
                          "stop_seq": selected.index + 1, "eta": [{"diff": 2}]}]}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    monkeypatch.setattr(transit, "_probe_background_cursor", 0)
    first = await transit.fetch_probe_snapshot(
        object(), probes, max_per_cycle=4, generation_probes=probes,
    )
    assert first.complete_routes and first.complete_routes[0].rows
    published = first.complete_routes[0].generation
    floor = dict(transit._probe_route_published_versions[route_key])  # noqa: SLF001

    calls.clear()
    await transit._refresh_probe_etas(object(), probes, max_per_cycle=1,
                                      generation_probes=probes)  # noqa: SLF001
    assert calls == ["ttl-0"]
    assert transit._probe_group_versions["GMB:ttl-0"] > floor["GMB:ttl-0"]  # noqa: SLF001
    calls.clear()
    mono[0] += 11
    wall[0] += timedelta(seconds=11)
    successful_fresh.clear()
    expired = await transit.fetch_probe_snapshot(
        object(), probes, max_per_cycle=1, generation_probes=probes,
    )
    assert expired.complete_routes == ()
    assert transit._probe_route_version_floors[route_key] == floor  # noqa: SLF001
    assert transit._probe_route_generations.get(route_key) is None  # noqa: SLF001

    fail_anchor[0] = True
    rebootstrap[0] = True
    fresh_calls: set[str] = set()
    for _ in range(6):
        mono[0] += 1
        wall[0] += timedelta(seconds=1)
        before = len(fresh_calls)
        partial = await transit.fetch_probe_snapshot(
            object(), probes, max_per_cycle=1, generation_probes=probes,
        )
        fresh_calls = set(successful_fresh)
        if len(fresh_calls) < 4:
            assert partial.complete_routes == ()
        if partial.complete_routes:
            break
        assert len(fresh_calls) >= before
    assert partial.complete_routes == ()
    assert {"ttl-1", "ttl-2", "ttl-3"} <= fresh_calls
    fail_anchor[0] = False
    rebuilt = partial
    for _ in range(4):
        mono[0] += 1
        wall[0] += timedelta(seconds=1)
        rebuilt = await transit.fetch_probe_snapshot(
            object(), probes, max_per_cycle=1, generation_probes=probes,
        )
        if rebuilt.complete_routes:
            break
    assert rebuilt.complete_routes
    assert len(fresh_calls | {"ttl-0"}) == 4
    assert rebuilt.complete_routes[0].generation > published
    assert rebuilt.complete_routes[0].observed_checkpoint_indices == {0, 1, 2, 3}


@pytest.mark.asyncio
async def test_mixed_active_resources_rotate_without_exceeding_total_cap(monkeypatch):
    def probe(operator, route, stop, index=0):
        return SimpleNamespace(operator=operator, route=route,
                               bound="seq-1" if operator == "GMB" else "outbound",
                               stop_id=stop, route_id=1, sequence=1, index=index)

    gmb = [probe("GMB", f"G{i}", f"fair-gmb-{i}") for i in range(8)]
    ctb = probe("CTB", "C", "fair-ctb")
    active = gmb + [ctb]
    priorities = {(item.operator, item.route, item.bound): {item.index} for item in active}
    _seed_warm_routes(active)
    calls: list[tuple[str, str]] = []

    async def fetch(_client, selected):
        calls.append((selected.operator, selected.stop_id))
        return {"data": []}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    monkeypatch.setattr(transit, "_probe_priority_cursor", 5)
    monkeypatch.setattr(transit, "_probe_background_cursor", 3)
    per_cycle = []
    for _ in range(3):
        calls.clear()
        await transit._refresh_probe_etas(object(), active, 4, priorities)  # noqa: SLF001
        unique = set(calls)
        assert len(calls) == len(unique) <= 4
        assert sum(operator == "GMB" for operator, _ in unique) <= 20
        per_cycle.append(unique)

    covered = per_cycle[0] | per_cycle[1] | per_cycle[2]
    assert {("GMB", item.stop_id) for item in gmb} <= covered
    assert ("CTB", ctb.stop_id) in covered


@pytest.mark.asyncio
async def test_supplemental_probe_churn_cannot_invalidate_sparse_baseline(monkeypatch):
    now = s.utc()
    route_key = ("CTB", "R", "outbound")

    def probe(index):
        return SimpleNamespace(
            operator="CTB", route="R", bound="outbound", stop_id=f"s{index}",
            route_id=1, sequence=1, index=index,
        )

    baseline = [probe(index) for index in (0, 3, 6, 9)]
    supplemental = probe(4)

    async def fetch(_client, selected):
        return {"data": [{
            "dir": "O",
            "eta": (now + timedelta(minutes=selected.index + 1)).isoformat(),
            "data_timestamp": now.isoformat(),
        }]}

    monkeypatch.setattr(transit, "_probe_wall_clock", lambda: now)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    first = await transit.fetch_probe_snapshot(
        object(), baseline, max_per_cycle=4, generation_probes=baseline
    )
    assert len(first.complete_routes) == 1
    generation = first.complete_routes[0].generation
    assert first.complete_routes[0].observed_checkpoint_indices == {0, 3, 6, 9}

    with_supplement = await transit.fetch_probe_snapshot(
        object(), baseline + [supplemental], max_per_cycle=1,
        priorities={route_key: {4}}, generation_probes=baseline,
    )
    assert with_supplement.complete_routes[0].generation == generation
    assert with_supplement.complete_routes[0].observed_checkpoint_indices == {0, 3, 6, 9}
    assert ("CTB", "R", "outbound", 4) in with_supplement.positioning_checkpoints

    without_supplement = await transit.fetch_probe_snapshot(
        object(), baseline, max_per_cycle=1, generation_probes=baseline
    )
    assert without_supplement.complete_routes[0].generation == generation


@pytest.mark.asyncio
async def test_realistic_cold_baseline_overflow_completes_within_two_cycles(monkeypatch):
    probes = [
        SimpleNamespace(
            operator="CTB", route="R", bound="outbound",
            stop_id=f"stop-{index:02d}", route_id=1, sequence=1, index=index,
        )
        for index in range(40)
    ]
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    await transit._refresh_probe_etas(  # noqa: SLF001
        object(), probes, 36, generation_probes=probes
    )
    assert len(calls) == 36
    await transit._refresh_probe_etas(  # noqa: SLF001
        object(), probes, 36, generation_probes=probes
    )

    assert set(calls) == {f"stop-{index:02d}" for index in range(40)}
    assert len(transit._probe_route_generations) == 1  # noqa: SLF001


@pytest.mark.asyncio
async def test_large_active_marker_probe_set_rotates_within_two_cycles(monkeypatch):
    route_key = ("CTB", "R", "outbound")

    def probe(index):
        return SimpleNamespace(
            operator="CTB", route="R", bound="outbound", stop_id=f"s{index:03d}",
            route_id=1, sequence=1, index=index,
        )

    baseline = [probe(index) for index in (0, 33, 66, 99)]
    supplemental = [probe(index) for index in range(100, 145)]
    calls: list[int] = []

    async def fetch(_client, selected):
        calls.append(selected.index)
        return {"data": []}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    await transit._refresh_probe_etas(  # noqa: SLF001
        object(), baseline, 36, generation_probes=baseline
    )
    calls.clear()
    priorities = {route_key: {probe.index for probe in supplemental}}
    active = baseline + supplemental
    await transit._refresh_probe_etas(  # noqa: SLF001
        object(), active, 36, priorities, generation_probes=baseline
    )
    assert len(calls) == 36
    await transit._refresh_probe_etas(  # noqa: SLF001
        object(), active, 36, priorities, generation_probes=baseline
    )

    assert {probe.index for probe in supplemental} <= set(calls)


@pytest.mark.asyncio
async def test_exact_cap_active_gmb_uses_full_allowance_before_background(monkeypatch):
    route_key = ("GMB", "R", "seq-1")

    def probe(index):
        return SimpleNamespace(
            operator="GMB", route="R", bound="seq-1", stop_id=f"s{index:02d}",
            route_id=1, sequence=1, index=index,
        )

    baseline = [probe(index) for index in range(20)]
    supplemental = [probe(index) for index in range(20, 40)]
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    await transit._refresh_probe_etas(  # noqa: SLF001
        object(), baseline, 36, generation_probes=baseline
    )
    assert len(transit._probe_route_generations) == 1  # noqa: SLF001
    calls.clear()

    priorities = {route_key: {probe.index for probe in supplemental}}
    active = baseline + supplemental
    per_cycle = []
    for _ in range(2):
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), active, 36, priorities, generation_probes=baseline
        )
        unique = set(calls)
        assert len(calls) == len(unique) <= 20
        per_cycle.append(unique)

    covered = per_cycle[0] | per_cycle[1]
    assert {probe.stop_id for probe in supplemental} <= covered
    assert covered & {probe.stop_id for probe in baseline}


@pytest.mark.asyncio
async def test_exact_cap_active_ctb_uses_full_total_before_background(monkeypatch):
    baseline = [
        SimpleNamespace(
            operator="CTB", route=f"B{index}", bound="outbound",
            stop_id=f"background-{index}", route_id=1, sequence=1, index=0,
        )
        for index in range(2)
    ]
    priority = [
        SimpleNamespace(
            operator="CTB", route=f"P{index}", bound="outbound",
            stop_id=f"priority-{index}", route_id=1, sequence=1, index=0,
        )
        for index in range(4)
    ]
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in priority
    }
    _seed_warm_routes(baseline)
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "_probe_priority_cursor", 2)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    active = baseline + priority
    per_cycle = []
    for _ in range(2):
        calls.clear()
        await transit._refresh_probe_etas(
            object(), active, 4, priorities, generation_probes=baseline
        )
        unique = set(calls)
        assert len(calls) == len(unique) <= 4
        per_cycle.append(unique)

    covered = per_cycle[0] | per_cycle[1]
    assert {item.stop_id for item in priority} <= covered
    assert covered & {item.stop_id for item in baseline}


@pytest.mark.asyncio
async def test_mixed_operator_priority_ring_resumes_at_capped_gmb_group(monkeypatch):
    def probe(operator, route, stop_id, index):
        return SimpleNamespace(
            operator=operator, route=route,
            bound="seq-1" if operator == "GMB" else "outbound",
            stop_id=stop_id, route_id=1, sequence=1, index=index,
        )

    baseline = [probe("GMB", "BASE", f"b{index}", index) for index in range(4)]
    priority = (
        [probe("CTB", "C", "ctb", 0)]
        + [probe("GMB", "P", f"p{index}", index) for index in range(4)]
        + [probe("KMB", "K", "kmb", 0)]
    )
    priorities = {
        ("CTB", "C", "outbound"): {0},
        ("GMB", "P", "seq-1"): {0, 1, 2, 3},
        ("KMB", "K", "outbound"): {0},
    }
    calls: list[tuple[str, str]] = []
    async def fetch(_client, selected):
        calls.append((selected.operator, selected.stop_id))
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 4)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    await transit._refresh_probe_etas(  # noqa: SLF001
        object(), baseline, 10, generation_probes=baseline
    )
    calls.clear()

    first_cycle = None
    for cycle in range(2):
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), baseline + priority, 10, priorities,
            generation_probes=baseline,
        )
        selected_priority = {
            stop for operator, stop in calls
            if operator == "GMB" and stop.startswith("p")
        }
        if cycle == 0:
            first_cycle = selected_priority
        else:
            assert "p3" in selected_priority
            assert first_cycle | selected_priority == {f"p{index}" for index in range(4)}


@pytest.mark.asyncio
async def test_priority_cursor_does_not_hide_now_fitting_gmb_ring(monkeypatch):
    probes = [
        SimpleNamespace(
            operator="GMB", route=f"R{index}", bound="seq-1",
            stop_id=f"priority-{index}", route_id=index, sequence=1, index=0,
        )
        for index in range(4)
    ]
    priorities = {
        (probe.operator, probe.route, probe.bound): {probe.index}
        for probe in probes
    }
    _seed_warm_routes(probes)
    calls: list[str] = []

    async def fetch(_client, probe):
        calls.append(probe.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 4)
    monkeypatch.setattr(transit, "_probe_priority_cursor", 2)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    await transit._refresh_probe_etas(object(), probes, 4, priorities)  # noqa: SLF001

    assert set(calls) == {probe.stop_id for probe in probes}


@pytest.mark.asyncio
async def test_sustained_gmb_priorities_reserve_rotating_background_capacity(monkeypatch):
    probes = [
        SimpleNamespace(
            operator="GMB", route=f"R{index}", bound="seq-1",
            stop_id=f"stop-{index}", route_id=index, sequence=1, index=0,
        )
        for index in range(8)
    ]
    priorities = {
        (probe.operator, probe.route, probe.bound): {probe.index}
        for probe in probes[:6]
    }
    calls: list[str] = []

    async def fetch(_client, probe):
        calls.append(probe.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 4)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    await transit._refresh_probe_etas(object(), probes, 4, priorities)  # noqa: SLF001
    await transit._refresh_probe_etas(object(), probes, 4, priorities)  # noqa: SLF001

    assert len(calls) == 8
    assert {"stop-6", "stop-7"} <= set(calls)


@pytest.mark.asyncio
async def test_steady_state_gmb_priorities_lead_before_background(monkeypatch):
    def probe(route, stop_id):
        return SimpleNamespace(
            operator="GMB", route=route, bound="seq-1", stop_id=stop_id,
            route_id=1, sequence=1, index=0,
        )

    baseline = [probe(f"B{index}", f"baseline-{index}") for index in range(20)]
    priority = [probe(f"P{index}", f"priority-{index}") for index in range(6)]
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in priority
    }
    _seed_warm_routes(baseline)
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 8)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    active = baseline + priority
    await transit._refresh_probe_etas(  # noqa: SLF001
        object(), active, 8, priorities, generation_probes=baseline
    )

    assert {item.stop_id for item in priority} <= set(calls)
    assert len({item.stop_id for item in baseline} & set(calls)) == 2


@pytest.mark.asyncio
async def test_live_gmb_priorities_bypass_unrelated_cold_backlog_within_two_cycles(
    monkeypatch,
):
    """A terminal singleton must refine while other routes are still cold."""
    def probe(route, stop_id, index=0):
        return SimpleNamespace(
            operator="GMB", route=route, bound="seq-1", stop_id=stop_id,
            route_id=1, sequence=1, index=index,
        )

    active_anchor = probe("11S", "active-anchor")
    active = [
        probe("11S", f"active-{index:02d}", index)
        for index in range(1, 15)
    ]
    cold = [
        probe(f"COLD-{index:02d}", f"cold-{index:02d}")
        for index in range(16)
    ]
    baseline = [active_anchor, *cold]
    _seed_warm_routes([active_anchor])
    priorities = {("GMB", "11S", "seq-1"): set(range(1, 15))}
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 8)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    per_cycle = []
    for _ in range(2):
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), [*baseline, *active], 8, priorities,
            generation_probes=baseline,
        )
        assert len(calls) == len(set(calls)) <= 8
        per_cycle.append(set(calls))

    covered = per_cycle[0] | per_cycle[1]
    assert {item.stop_id for item in active} <= covered
    assert {item.stop_id for item in cold} & covered


@pytest.mark.asyncio
async def test_single_gmb_slot_alternates_priority_and_cold_bootstrap(monkeypatch):
    def probe(route, stop_id, index=0):
        return SimpleNamespace(
            operator="GMB", route=route, bound="seq-1", stop_id=stop_id,
            route_id=1, sequence=1, index=index,
        )

    cold = probe("COLD", "cold")
    active = probe("11S", "active", 1)
    priorities = {("GMB", "11S", "seq-1"): {1}}
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 1)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    per_cycle = []
    for _ in range(2):
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), [cold, active], 1, priorities,
            generation_probes=[cold],
        )
        per_cycle.append(list(calls))

    assert per_cycle == [["active"], ["cold"]]
    assert ("GMB", "COLD", "seq-1") in transit._probe_route_generations  # noqa: SLF001


@pytest.mark.asyncio
async def test_pending_completions_cannot_hide_cold_gmb_bootstrap(monkeypatch):
    def probe(route, stop_id, index=0):
        return SimpleNamespace(
            operator="GMB", route=route, bound="seq-1", stop_id=stop_id,
            route_id=1, sequence=1, index=index,
        )

    first = [probe("A", f"a{index}", index) for index in range(2)]
    second = [probe("B", f"b{index}", index) for index in range(2)]
    cold = [probe("COLD", f"cold{index}", index) for index in range(4)]
    supplemental = [probe("11S", f"active{index}", index) for index in range(6)]
    fixed = [*first, *second]
    active = [first[0], second[0], *supplemental]
    priorities = {
        ("GMB", "A", "seq-1"): {0},
        ("GMB", "B", "seq-1"): {0},
        ("GMB", "11S", "seq-1"): set(range(6)),
    }
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 8)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    await transit._refresh_probe_etas(  # noqa: SLF001
        object(), fixed, 8, generation_probes=fixed,
    )
    await transit._refresh_probe_etas(  # noqa: SLF001
        object(), fixed, 1,
        {("GMB", "A", "seq-1"): {0}}, generation_probes=fixed,
    )

    per_cycle = []
    universe = [*fixed, *cold, *supplemental]
    baseline = [*fixed, *cold]
    for _ in range(4):
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), universe, 8, priorities, generation_probes=baseline,
        )
        assert len(calls) == len(set(calls)) <= 8
        per_cycle.append(set(calls))

    active_stops = {item.stop_id for item in active}
    assert all(active_stops <= (left | right)
               for left, right in zip(per_cycle, per_cycle[1:], strict=False))
    assert {item.stop_id for item in cold} <= set().union(*per_cycle)
    assert ("GMB", "COLD", "seq-1") in transit._probe_route_generations  # noqa: SLF001


@pytest.mark.asyncio
async def test_mixed_cold_resources_do_not_repeat_before_due_priorities(monkeypatch):
    """A changing cold resource slot cannot erase prior service obligations."""
    def probe(operator, route, stop_id):
        return SimpleNamespace(
            operator=operator,
            route=route,
            bound="seq-1" if operator == "GMB" else "outbound",
            stop_id=stop_id,
            route_id=1,
            sequence=1,
            index=0,
        )

    gmb = [probe("GMB", f"G{index}", f"gmb-{index}") for index in range(6)]
    ctb = [probe("CTB", f"C{index}", f"ctb-{index}") for index in range(4)]
    cold_ctb = probe("CTB", "COLD-X", "cold-ctb")
    cold_gmb = probe("GMB", "COLD-Y", "cold-gmb")
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in [*gmb, *ctb]
    }
    universe = [cold_ctb, cold_gmb, *gmb, *ctb]
    baseline = [cold_ctb, cold_gmb]
    transit._probe_group_service_debt.update({  # noqa: SLF001
        transit._fetch_group_key(item): 1  # noqa: SLF001
        for item in [*gmb, cold_ctb]
    })
    calls: list[tuple[str, str]] = []

    async def fetch(_client, selected):
        calls.append((selected.operator, selected.stop_id))
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 4)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    per_cycle = []
    for _ in range(2):
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), universe, 6, priorities, generation_probes=baseline,
        )
        assert len(calls) == len(set(calls)) <= 6
        assert sum(operator == "GMB" for operator, _ in calls) <= 4
        per_cycle.append(set(calls))

    covered = per_cycle[0] | per_cycle[1]
    assert {(item.operator, item.stop_id) for item in [*gmb, *ctb]} <= covered
    assert ("CTB", "COLD-X", "outbound") in transit._probe_route_generations  # noqa: SLF001
    assert ("GMB", "COLD-Y", "seq-1") in transit._probe_route_generations  # noqa: SLF001


@pytest.mark.asyncio
async def test_equal_debt_priorities_stay_balanced_as_cold_resource_changes(
    monkeypatch,
):
    def probe(operator, route, stop_id, index=0):
        return SimpleNamespace(
            operator=operator,
            route=route,
            bound="seq-1" if operator == "GMB" else "outbound",
            stop_id=stop_id,
            route_id=1,
            sequence=1,
            index=index,
        )

    gmb = [probe("GMB", f"G{index}", f"gmb-{index}") for index in range(2)]
    ctb = [probe("CTB", f"C{index}", f"ctb-{index}") for index in range(2)]
    cold_ctb = [
        probe("CTB", "COLD-X", f"cold-ctb-{index}", index)
        for index in range(2)
    ]
    cold_gmb = [
        probe("GMB", "COLD-Y", f"cold-gmb-{index}", index)
        for index in range(2)
    ]
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in [*gmb, *ctb]
    }
    universe = [*cold_ctb, *cold_gmb, *gmb, *ctb]
    baseline = [*cold_ctb, *cold_gmb]
    calls: list[tuple[str, str]] = []

    async def fetch(_client, selected):
        calls.append((selected.operator, selected.stop_id))
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 2)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    per_cycle = []
    for _ in range(4):
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), universe, 3, priorities, generation_probes=baseline,
        )
        assert len(calls) == len(set(calls)) <= 3
        assert sum(operator == "GMB" for operator, _ in calls) <= 2
        per_cycle.append(set(calls))

    active_stops = {(item.operator, item.stop_id) for item in [*gmb, *ctb]}
    assert all(
        active_stops <= (left | right)
        for left, right in zip(per_cycle, per_cycle[1:], strict=False)
    )
    assert ("CTB", "COLD-X", "outbound") in transit._probe_route_generations  # noqa: SLF001
    assert ("GMB", "COLD-Y", "seq-1") in transit._probe_route_generations  # noqa: SLF001


@pytest.mark.asyncio
async def test_half_split_tie_preserves_next_gmb_capacity(monkeypatch):
    def probe(operator, route, stop_id):
        return SimpleNamespace(
            operator=operator,
            route=route,
            bound="seq-1" if operator == "GMB" else "outbound",
            stop_id=stop_id,
            route_id=1,
            sequence=1,
            index=0,
        )

    gmb = [probe("GMB", f"G{index}", f"gmb-{index}") for index in range(3)]
    ctb = [probe("CTB", f"C{index}", f"ctb-{index}") for index in range(3)]
    cold_ctb = probe("CTB", "COLD-X", "cold-ctb")
    cold_gmb = probe("GMB", "COLD-Y", "cold-gmb")
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in [*gmb, *ctb]
    }
    baseline = [cold_ctb, cold_gmb]
    universe = [*baseline, *gmb, *ctb]
    transit._probe_group_service_debt.update({  # noqa: SLF001
        transit._fetch_group_key(item): 1  # noqa: SLF001
        for item in [*ctb, cold_ctb]
    })
    calls: list[tuple[str, str]] = []

    async def fetch(_client, selected):
        calls.append((selected.operator, selected.stop_id))
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 2)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    per_cycle = []
    for _ in range(2):
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), universe, 4, priorities, generation_probes=baseline,
        )
        assert len(calls) == len(set(calls)) <= 4
        assert sum(operator == "GMB" for operator, _ in calls) <= 2
        per_cycle.append(set(calls))

    active_stops = {(item.operator, item.stop_id) for item in [*gmb, *ctb]}
    assert active_stops <= (per_cycle[0] | per_cycle[1])


@pytest.mark.asyncio
async def test_deferred_cap_one_cold_gmb_cannot_reenter_background(monkeypatch):
    def probe(operator, route, stop_id, index=0):
        return SimpleNamespace(
            operator=operator,
            route=route,
            bound="seq-1" if operator == "GMB" else "outbound",
            stop_id=stop_id,
            route_id=1,
            sequence=1,
            index=index,
        )

    gmb = [probe("GMB", "G", "gmb")]
    ctb = [probe("CTB", f"C{index}", f"ctb-{index}") for index in range(3)]
    cold_ctb = [
        probe("CTB", "COLD-X", f"cold-ctb-{index}", index)
        for index in range(2)
    ]
    cold_gmb = [
        probe("GMB", "COLD-Y", f"cold-gmb-{index}", index)
        for index in range(2)
    ]
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in [*gmb, *ctb]
    }
    baseline = [*cold_ctb, *cold_gmb]
    universe = [*baseline, *gmb, *ctb]
    transit._probe_group_service_debt.update({  # noqa: SLF001
        transit._fetch_group_key(item): 1 for item in ctb  # noqa: SLF001
    })
    calls: list[tuple[str, str]] = []

    async def fetch(_client, selected):
        calls.append((selected.operator, selected.stop_id))
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 1)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    per_cycle = []
    for _ in range(6):
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), universe, 3, priorities, generation_probes=baseline,
        )
        assert len(calls) == len(set(calls)) <= 3
        assert sum(operator == "GMB" for operator, _ in calls) <= 1
        per_cycle.append(set(calls))

    active_stops = {(item.operator, item.stop_id) for item in [*gmb, *ctb]}
    assert all(
        active_stops <= (left | right)
        for left, right in zip(per_cycle, per_cycle[1:], strict=False)
    )
    assert ("CTB", "COLD-X", "outbound") in transit._probe_route_generations  # noqa: SLF001
    assert ("GMB", "COLD-Y", "seq-1") in transit._probe_route_generations  # noqa: SLF001


@pytest.mark.asyncio
async def test_cold_routes_fill_true_slack_and_publish_at_production_caps(monkeypatch):
    def probe(operator, route, stop_id, index=0):
        return SimpleNamespace(
            operator=operator,
            route=route,
            bound="seq-1" if operator == "GMB" else "outbound",
            stop_id=stop_id,
            route_id=1,
            sequence=1,
            index=index,
        )

    cold = [
        probe("GMB", f"COLD-{route}", f"cold-{anchor}-{route}", anchor)
        for anchor in range(4)
        for route in range(10)
    ]
    active = probe("CTB", "ACTIVE", "active")
    priorities = {("CTB", "ACTIVE", "outbound"): {0}}
    calls: list[tuple[str, str]] = []

    async def fetch(_client, selected):
        calls.append((selected.operator, selected.stop_id))
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 20)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    universe = [*cold, active]
    per_cycle = []
    for _ in range(2):
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), universe, 36, priorities, generation_probes=cold,
        )
        assert len(calls) == len(set(calls)) <= 36
        assert sum(operator == "GMB" for operator, _ in calls) <= 20
        assert ("CTB", "active") in calls
        per_cycle.append(set(calls))

    covered = per_cycle[0] | per_cycle[1]
    assert {(item.operator, item.stop_id) for item in cold} <= covered
    assert {
        ("GMB", f"COLD-{route}", "seq-1") for route in range(10)
    } <= set(transit._probe_route_generations)  # noqa: SLF001


@pytest.mark.asyncio
async def test_cold_route_anchors_complete_coherently_before_cache_expiry(monkeypatch):
    def probe(route, stop_id, index=0):
        return SimpleNamespace(
            operator="GMB",
            route=route,
            bound="seq-1",
            stop_id=stop_id,
            route_id=1,
            sequence=1,
                    index=index,
        )

    cold = [
        probe(f"COLD-{route}", f"cold-{anchor}-{route}", anchor)
        for anchor in range(4)
        for route in range(10)
    ]
    priority = [probe(f"P{index}", f"priority-{index}") for index in range(14)]
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in priority
    }
    clock = [0.0]
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 8)
    monkeypatch.setattr(transit._probe_cache, "_clock", lambda: clock[0])  # noqa: SLF001
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    per_cycle = []
    for _ in range(40):
        clock[0] += 30.0
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), [*cold, *priority], 36, priorities,
            generation_probes=cold,
        )
        assert len(calls) == len(set(calls)) <= 8
        per_cycle.append(set(calls))

    active_stops = {item.stop_id for item in priority}
    assert all(
        active_stops <= (left | right)
        for left, right in zip(per_cycle, per_cycle[1:], strict=False)
    )
    assert {item.stop_id for item in cold} <= set().union(*per_cycle)
    assert {
        ("GMB", f"COLD-{route}", "seq-1") for route in range(10)
    } <= set(transit._probe_route_generations)  # noqa: SLF001


@pytest.mark.asyncio
async def test_failed_partial_cold_route_yields_to_older_healthy_route(monkeypatch):
    def probe(route, stop_id, index=0):
        return SimpleNamespace(
            operator="GMB",
            route=route,
            bound="seq-1",
            stop_id=stop_id,
            route_id=1,
            sequence=1,
            index=index,
        )

    cold = [
        probe(route, f"{route.lower()}-{index}", index)
        for index in range(4)
        for route in ("A", "B")
    ]
    priority = [probe(f"P{index}", f"priority-{index}") for index in range(14)]
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in priority
    }
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        if selected.stop_id == "a-3":
            raise RuntimeError("anchor unavailable")
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 8)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    per_cycle = []
    for _ in range(12):
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), [*cold, *priority], 36, priorities,
            generation_probes=cold,
        )
        assert len(calls) == len(set(calls)) <= 8
        per_cycle.append(set(calls))

    active_stops = {item.stop_id for item in priority}
    assert all(
        active_stops <= (left | right)
        for left, right in zip(per_cycle, per_cycle[1:], strict=False)
    )
    assert ("GMB", "A", "seq-1") not in transit._probe_route_generations  # noqa: SLF001
    assert ("GMB", "B", "seq-1") in transit._probe_route_generations  # noqa: SLF001
    assert {f"b-{index}" for index in range(4)} <= set().union(*per_cycle)


@pytest.mark.asyncio
async def test_failed_cold_routes_cannot_starve_older_warm_refreshes(monkeypatch):
    def probe(route, stop_id):
        return SimpleNamespace(
            operator="GMB",
            route=route,
            bound="seq-1",
            stop_id=stop_id,
            route_id=1,
            sequence=1,
            index=0,
        )

    cold = [probe(f"COLD-{index}", f"cold-{index}") for index in range(2)]
    warm = [probe(f"WARM-{index}", f"warm-{index}") for index in range(4)]
    priority = [probe(f"P{index}", f"priority-{index}") for index in range(18)]
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in priority
    }
    _seed_warm_routes(warm)
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        if selected.stop_id.startswith("cold-"):
            raise RuntimeError("cold route unavailable")
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 20)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    per_cycle = []
    for _ in range(6):
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), [*cold, *warm, *priority], 36, priorities,
            generation_probes=[*cold, *warm],
        )
        assert len(calls) == len(set(calls)) <= 20
        per_cycle.append(set(calls))

    priority_stops = {item.stop_id for item in priority}
    assert all(priority_stops <= cycle for cycle in per_cycle)
    assert {item.stop_id for item in cold} <= set().union(*per_cycle)
    assert {item.stop_id for item in warm} <= set().union(*per_cycle)


@pytest.mark.asyncio
async def test_saturated_lifecycle_slot_rotates_from_failed_cold_to_warm(monkeypatch):
    def probe(route, stop_id, index=0):
        return SimpleNamespace(
            operator="GMB",
            route=route,
            bound="seq-1",
            stop_id=stop_id,
            route_id=1,
            sequence=1,
            index=index,
        )

    cold = probe("COLD", "cold")
    warm = [probe("WARM", f"warm-{index}", index) for index in range(4)]
    priority = [probe(f"P{index}", f"priority-{index}") for index in range(38)]
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in priority
    }
    _seed_warm_routes(warm)
    initial_generation = transit._probe_route_generations[  # noqa: SLF001
        ("GMB", "WARM", "seq-1")
    ].public.generation
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        if selected.stop_id == "cold":
            raise RuntimeError("cold route unavailable")
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 20)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    per_cycle = []
    for _ in range(10):
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), [cold, *warm, *priority], 36, priorities,
            generation_probes=[cold, *warm],
        )
        assert len(calls) == len(set(calls)) <= 20
        per_cycle.append(set(calls))

    priority_stops = {item.stop_id for item in priority}
    assert all(
        priority_stops <= (left | right)
        for left, right in zip(per_cycle, per_cycle[1:], strict=False)
    )
    assert "cold" in set().union(*per_cycle)
    assert {item.stop_id for item in warm} <= set().union(*per_cycle)
    assert transit._probe_route_generations[  # noqa: SLF001
        ("GMB", "WARM", "seq-1")
    ].public.generation > initial_generation


@pytest.mark.asyncio
async def test_oversized_priorities_rotate_while_cold_progresses(monkeypatch):
    def probe(operator, route, stop_id, index=0):
        return SimpleNamespace(
            operator=operator,
            route=route,
            bound="seq-1" if operator == "GMB" else "outbound",
            stop_id=stop_id,
            route_id=1,
            sequence=1,
            index=index,
        )

    gmb = [probe("GMB", f"G{index}", f"gmb-{index}") for index in range(39)]
    ctb = [probe("CTB", f"C{index}", f"ctb-{index}") for index in range(31)]
    cold_ctb = [
        probe("CTB", "COLD-X", f"cold-ctb-{index}", index)
        for index in range(2)
    ]
    cold_gmb = [probe("GMB", "COLD-Y", "cold-gmb")]
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in [*gmb, *ctb]
    }
    baseline = [*cold_ctb, *cold_gmb]
    universe = [*baseline, *gmb, *ctb]
    transit._probe_group_service_debt.update({  # noqa: SLF001
        transit._fetch_group_key(item): 1  # noqa: SLF001
        for item in [*ctb, *cold_ctb]
    })
    calls: list[tuple[str, str]] = []

    async def fetch(_client, selected):
        calls.append((selected.operator, selected.stop_id))
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 20)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    per_cycle = []
    for _ in range(4):
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), universe, 36, priorities, generation_probes=baseline,
        )
        assert len(calls) == len(set(calls)) <= 36
        assert sum(operator == "GMB" for operator, _ in calls) <= 20
        per_cycle.append(set(calls))

    active_stops = {(item.operator, item.stop_id) for item in [*gmb, *ctb]}
    assert all(active_stops <= set().union(*window)
               for window in zip(
                   per_cycle, per_cycle[1:], per_cycle[2:], strict=False,
               ))
    assert ("CTB", "COLD-X", "outbound") in transit._probe_route_generations  # noqa: SLF001
    assert ("GMB", "COLD-Y", "seq-1") in transit._probe_route_generations  # noqa: SLF001


@pytest.mark.asyncio
async def test_feasible_carry_preempts_warm_background_reservations(monkeypatch):
    def probe(route, stop_id):
        return SimpleNamespace(
            operator="GMB",
            route=route,
            bound="seq-1",
            stop_id=stop_id,
            route_id=1,
            sequence=1,
            index=0,
        )

    baseline = [probe(f"B{index}", f"background-{index}") for index in range(2)]
    priority = [probe(f"P{index}", f"priority-{index}") for index in range(5)]
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in priority
    }
    _seed_warm_routes(baseline)
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 3)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    active = [*baseline, *priority]
    per_cycle = []
    for _ in range(2):
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), active, 3, priorities, generation_probes=baseline,
        )
        assert len(calls) == len(set(calls)) <= 3
        per_cycle.append(set(calls))

    assert {item.stop_id for item in priority} <= (per_cycle[0] | per_cycle[1])
    assert {item.stop_id for item in baseline} & (per_cycle[0] | per_cycle[1])


@pytest.mark.asyncio
async def test_saturated_active_ring_cannot_starve_new_cold_route(monkeypatch):
    def probe(route, stop_id, index=0):
        return SimpleNamespace(
            operator="GMB",
            route=route,
            bound="seq-1",
            stop_id=stop_id,
            route_id=1,
            sequence=1,
            index=index,
        )

    priority = [probe(f"P{index}", f"priority-{index}") for index in range(40)]
    cold = [probe("COLD", f"cold-{index}", index) for index in range(4)]
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in priority
    }
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 20)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    per_cycle = []
    for _ in range(6):
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), [*cold, *priority], 36, priorities,
            generation_probes=cold,
        )
        assert len(calls) == len(set(calls)) <= 20
        per_cycle.append(set(calls))

    active_stops = {item.stop_id for item in priority}
    assert all(active_stops <= set().union(*window)
               for window in zip(
                   per_cycle, per_cycle[1:], per_cycle[2:], strict=False,
               ))
    assert {item.stop_id for item in cold} <= set().union(*per_cycle)
    assert ("GMB", "COLD", "seq-1") in transit._probe_route_generations  # noqa: SLF001


@pytest.mark.asyncio
async def test_twenty_three_gmb_priorities_clear_with_background_in_two_cycles(monkeypatch):
    def probe(route, stop_id, index=0):
        return SimpleNamespace(
            operator="GMB", route=route, bound="seq-1", stop_id=stop_id,
            route_id=1, sequence=1, index=index,
        )

    baseline = [probe(f"B{index}", f"baseline-{index}") for index in range(8)]
    priority = [probe(f"P{index}", f"priority-{index}") for index in range(23)]
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in priority
    }
    _seed_warm_routes(baseline)
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 15)
    monkeypatch.setattr(transit, "_probe_priority_cursor", 20)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    active = baseline + priority
    for _ in range(2):
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), active, 36, priorities, generation_probes=baseline
        )

    priority_calls = {item.stop_id for item in priority} & set(calls)
    assert priority_calls == {item.stop_id for item in priority}
    assert len([stop for stop in calls if stop.startswith("baseline-")]) > 0


@pytest.mark.asyncio
async def test_dynamic_priority_membership_services_persistent_group_within_two_cycles(
    monkeypatch,
):
    """Changing marker boundaries must not starve a continuously-needed group."""
    def probe(route, stop_id):
        return SimpleNamespace(
            operator="GMB", route=route, bound="seq-1", stop_id=stop_id,
            route_id=1, sequence=1, index=0,
        )

    baseline = [probe("BASE", "baseline")]
    persistent = probe("PERSISTENT", "persistent")
    _seed_warm_routes(baseline)
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    for frame in range(2):
        # Simulate a moving marker ring: each frame replaces the other
        # boundaries, while `persistent` remains required throughout.
        churn = [probe(f"CHURN-{frame}-{index}", f"churn-{frame}-{index}")
                 for index in range(3)]
        active = baseline + [persistent] + churn
        priorities = {
            (item.operator, item.route, item.bound): {item.index}
            for item in [persistent, *churn]
        }
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), active, max_per_cycle=1, priorities=priorities,
            generation_probes=baseline,
        )

    assert "persistent" in calls
    assert calls.index("persistent") < 2


@pytest.mark.asyncio
async def test_dynamic_priority_membership_does_not_starve_background_seed(
    monkeypatch,
):
    """Priority/background transitions must preserve due service for anchors."""
    def probe(stop_id, index):
        return SimpleNamespace(
            operator="GMB", route="BASE", bound="seq-1", stop_id=stop_id,
            route_id=1, sequence=1, index=index,
        )

    seed = probe("seed", 0)
    toggle_a = probe("toggle-a", 1)
    toggle_b = probe("toggle-b", 2)
    baseline = [seed, toggle_a, toggle_b]
    _seed_warm_routes(baseline)
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    monkeypatch.setattr(transit, "_probe_background_cursor", 1)
    for toggled in (toggle_a, toggle_b):
        priorities = {
            (toggled.operator, toggled.route, toggled.bound): {toggled.index}
        }
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), baseline, max_per_cycle=2, priorities=priorities,
            generation_probes=baseline,
        )

    assert "seed" in calls
    assert calls.index("seed") < 2


@pytest.mark.asyncio
async def test_gmb_background_reservation_yields_to_overdue_priority_groups(
    monkeypatch,
):
    def probe(route, stop_id):
        return SimpleNamespace(
            operator="GMB", route=route, bound="seq-1", stop_id=stop_id,
            route_id=1, sequence=1, index=0,
        )

    baseline = [probe("BASE", "base")]
    priority = [probe(f"P{index}", f"priority-{index}") for index in range(29)]
    active = baseline + priority
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in priority
    }
    _seed_warm_routes(baseline)
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 15)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    for _ in range(2):
        await transit._refresh_probe_etas(
            object(), active, max_per_cycle=36, priorities=priorities,
            generation_probes=baseline,
        )

    assert set(calls) == {"base", *(item.stop_id for item in priority)}
    assert len(calls) == len(set(calls)) == 30


@pytest.mark.asyncio
async def test_non_gmb_background_reservation_yields_to_overdue_priority_groups(
    monkeypatch,
):
    def probe(route, stop_id):
        return SimpleNamespace(
            operator="CTB", route=route, bound="outbound", stop_id=stop_id,
            route_id=1, sequence=1, index=0,
        )

    baseline = [probe("BASE", "base")]
    priority = [probe(f"P{index}", f"priority-{index}") for index in range(29)]
    active = baseline + priority
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in priority
    }
    _seed_warm_routes(baseline)
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    for _ in range(2):
        await transit._refresh_probe_etas(
            object(), active, max_per_cycle=15, priorities=priorities,
            generation_probes=baseline,
        )

    assert set(calls) == {"base", *(item.stop_id for item in priority)}
    assert len(calls) == len(set(calls)) == 30


@pytest.mark.asyncio
async def test_resource_quota_uses_next_group_debt_across_two_rings(monkeypatch):
    def probe(operator, route, stop_id):
        return SimpleNamespace(
            operator=operator, route=route,
            bound="seq-1" if operator == "GMB" else "outbound",
            stop_id=stop_id, route_id=1, sequence=1, index=0,
        )

    baseline = [probe("CTB", f"BASE-{i}", f"base-{i}") for i in range(2)]
    gmb = [probe("GMB", f"G{i}", f"gmb-{i}") for i in range(3)]
    ctb = [probe("CTB", f"C{i}", f"ctb-{i}") for i in range(3)]
    active = baseline + gmb + ctb
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in gmb + ctb
    }
    _seed_warm_routes(baseline)
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    for _ in range(2):
        await transit._refresh_probe_etas(
            object(), active, max_per_cycle=4, priorities=priorities,
            generation_probes=baseline,
        )

    assert set(calls) == {item.stop_id for item in active}
    assert len(calls) == len(set(calls)) == 8


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("gmb_total", "total_cap", "gmb_cap"), ((2, 1, 1), (4, 2, 2))
)
async def test_oversized_resources_still_rotate_each_resource(
    monkeypatch, gmb_total, total_cap, gmb_cap,
):
    def probe(operator, route, stop_id):
        return SimpleNamespace(
            operator=operator, route=route,
            bound="seq-1" if operator == "GMB" else "outbound",
            stop_id=stop_id, route_id=1, sequence=1, index=0,
        )

    gmb = [probe("GMB", f"G{i}", f"gmb-{i}") for i in range(gmb_total)]
    ctb = [probe("CTB", "C", "ctb")]
    active = gmb + ctb
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in active
    }
    _seed_warm_routes(active)
    calls: list[tuple[str, str]] = []
    all_calls: list[tuple[str, str]] = []

    async def fetch(_client, selected):
        calls.append((selected.operator, selected.stop_id))
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", gmb_cap)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    for _ in range(4):
        calls.clear()
        await transit._refresh_probe_etas(
            object(), active, max_per_cycle=total_cap, priorities=priorities,
            generation_probes=active,
        )
        assert len(calls) <= total_cap
        assert sum(operator == "GMB" for operator, _ in calls) <= gmb_cap
        all_calls.extend(calls)

    assert any(operator == "CTB" for operator, _ in all_calls)
    assert any(operator == "GMB" for operator, _ in all_calls)


@pytest.mark.asyncio
async def test_gmb_background_reservation_refills_remaining_total_slot(monkeypatch):
    def probe(operator, route, stop_id):
        return SimpleNamespace(
            operator=operator, route=route,
            bound="seq-1" if operator == "GMB" else "outbound",
            stop_id=stop_id, route_id=1, sequence=1, index=0,
        )

    background = probe("GMB", "BASE", "background")
    gmb = probe("GMB", "G", "gmb")
    ctb = [probe("CTB", f"C{i}", f"ctb-{i}") for i in range(2)]
    active = [background, gmb, *ctb]
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in [gmb, *ctb]
    }
    _seed_warm_routes([background])
    all_calls: list[tuple[str, str]] = []

    async def fetch(_client, selected):
        all_calls.append((selected.operator, selected.stop_id))
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 1)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    for _ in range(2):
        before = len(all_calls)
        await transit._refresh_probe_etas(
            object(), active, max_per_cycle=2, priorities=priorities,
            generation_probes=[background],
        )
        frame = all_calls[before:]
        assert len(frame) <= 2
        assert sum(operator == "GMB" for operator, _ in frame) <= 1

    assert {stop_id for _, stop_id in all_calls} >= {"gmb", "ctb-0", "ctb-1"}


@pytest.mark.asyncio
async def test_gmb_subcap_fills_spare_total_capacity_with_other_background(monkeypatch):
    def probe(operator, route, stop_id):
        return SimpleNamespace(
            operator=operator, route=route,
            bound="seq-1" if operator == "GMB" else "outbound",
            stop_id=stop_id, route_id=1, sequence=1, index=0,
        )

    baseline = (
        [probe("GMB", f"B{index}", f"gmb-background-{index}") for index in range(3)]
        + [probe("CTB", f"C{index}", f"ctb-background-{index}") for index in range(3)]
        + [probe("KMB", f"K{index}", f"kmb-background-{index}") for index in range(3)]
    )
    priority = [probe("GMB", f"P{index}", f"gmb-priority-{index}") for index in range(4)]
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in priority
    }
    _seed_warm_routes(baseline)
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 4)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    active = baseline + priority
    per_cycle = []
    for _ in range(2):
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), active, 8, priorities, generation_probes=baseline
        )
        unique = set(calls)
        assert len(calls) == len(unique) <= 8
        assert sum(item.startswith("gmb-") for item in unique) <= 4
        per_cycle.append(unique)

    covered = per_cycle[0] | per_cycle[1]
    assert {item.stop_id for item in priority} <= covered
    assert {"gmb-background-0", "ctb-background-0", "kmb-background-0"} <= covered


@pytest.mark.asyncio
async def test_pending_gmb_completion_respects_total_and_gmb_floors(monkeypatch):
    def probe(operator, route, index, stop):
        return SimpleNamespace(
            operator=operator, route=route,
            bound="seq-1" if operator == "GMB" else "outbound",
            stop_id=stop, route_id=1, sequence=1, index=index,
        )

    baseline = [probe("GMB", "BASE", i, f"base-{i}") for i in range(4)]
    active_gmb = [probe("GMB", f"GP{i}", 0, f"gmb-active-{i}") for i in range(10)]
    active_other = [probe("CTB", f"CP{i}", 0, f"ctb-active-{i}") for i in range(13)]
    active = baseline + active_gmb + active_other
    calls: list[tuple[str, str]] = []

    async def fetch(_client, selected):
        calls.append((selected.operator, selected.stop_id))
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 8)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    monkeypatch.setattr(transit, "_probe_priority_cursor", 3)
    await transit._refresh_probe_etas(object(), baseline, 4, generation_probes=baseline)  # noqa: SLF001
    route_key = ("GMB", "BASE", "seq-1")
    published = transit._probe_route_generations[route_key].public.generation  # noqa: SLF001
    await transit._refresh_probe_etas(object(), baseline, 1, generation_probes=baseline)  # noqa: SLF001

    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in active_gmb + active_other
    }
    per_cycle = []
    for _ in range(2):
        calls.clear()
        await transit._refresh_probe_etas(
            object(), active, 15, priorities, generation_probes=baseline,
        )  # noqa: SLF001
        unique = set(calls)
        assert len(unique) <= 15
        assert sum(operator == "GMB" for operator, _ in unique) <= 8
        per_cycle.append(unique)

    assert transit._probe_route_generations[route_key].public.generation > published  # noqa: SLF001
    covered = per_cycle[0] | per_cycle[1]
    assert {(item.operator, item.stop_id) for item in active_gmb + active_other} <= covered


@pytest.mark.asyncio
async def test_exact_gmb_subcap_still_seeds_gmb_background_and_rotates(monkeypatch):
    def probe(operator, route, stop):
        return SimpleNamespace(operator=operator, route=route,
                               bound="seq-1" if operator == "GMB" else "outbound",
                               stop_id=stop, route_id=1, sequence=1, index=0)

    baseline = [probe("GMB", f"B{i}", f"gmb-base-{i}") for i in range(4)]
    priority = [probe("GMB", f"P{i}", f"gmb-priority-{i}") for i in range(8)]
    background = [probe("CTB", f"C{i}", f"ctb-background-{i}") for i in range(4)]
    priorities = {(item.operator, item.route, item.bound): {0} for item in priority}
    calls: list[tuple[str, str]] = []

    async def fetch(_client, selected):
        calls.append((selected.operator, selected.stop_id))
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 8)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    active = baseline + priority + background
    per_cycle = []
    for _ in range(2):
        calls.clear()
        await transit._refresh_probe_etas(object(), active, 15, priorities)  # noqa: SLF001
        unique = set(calls)
        assert len(unique) <= 15
        assert sum(operator == "GMB" for operator, _ in unique) <= 8
        per_cycle.append(unique)

    covered = per_cycle[0] | per_cycle[1]
    assert {("GMB", item.stop_id) for item in baseline} & covered
    assert {("GMB", item.stop_id) for item in priority} <= covered


@pytest.mark.asyncio
async def test_tiny_total_capacity_does_not_reserve_equal_active_floor(monkeypatch):
    def probe(operator, route, index, stop):
        return SimpleNamespace(operator=operator, route=route,
                               bound="seq-1" if operator == "GMB" else "outbound",
                               stop_id=stop, route_id=1, sequence=1, index=index)

    baseline = [probe("GMB", "BASE", i, f"tiny-base-{i}") for i in range(2)]
    active = baseline + [probe("CTB", "ACTIVE", 0, "tiny-active")]
    route_key = ("GMB", "BASE", "seq-1")
    calls: list[tuple[str, str]] = []

    async def fetch(_client, selected):
        calls.append((selected.operator, selected.stop_id))
        return {"data": []}

    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 2)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    await transit._refresh_probe_etas(object(), baseline, 2, generation_probes=baseline)  # noqa: SLF001
    published = transit._probe_route_generations[route_key].public.generation  # noqa: SLF001
    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 1)
    await transit._refresh_probe_etas(object(), baseline, 1, generation_probes=baseline)  # noqa: SLF001
    calls.clear()
    await transit._refresh_probe_etas(
        object(), active, 1,
        {("CTB", "ACTIVE", "outbound"): {0}},
        generation_probes=baseline,
    )  # noqa: SLF001
    assert len(calls) == len(set(calls)) == 1
    assert transit._probe_route_generations[route_key].public.generation == published  # noqa: SLF001


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
    monkeypatch.setattr(transit, "_probe_background_cursor", 0)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    # The fair cold cursor covers the 35 groups in two capped cycles.
    for _ in range(2):
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
    monkeypatch.setattr(transit, "_gmb_cooldown_until", 0)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)

    await transit.fetch_probe_etas(object(), probes)

    assert len(calls) == 1
    assert transit._gmb_cooldown_until > transit.time.monotonic()  # noqa: SLF001
    assert transit.GMB_GROUPS_PER_CYCLE == 17
    first_group = transit._fetch_group_key(probes[0])  # noqa: SLF001
    later_group = transit._fetch_group_key(probes[1])  # noqa: SLF001
    assert transit._probe_group_service_debt[first_group] == 0  # noqa: SLF001
    assert transit._probe_group_service_debt[later_group] > 0  # noqa: SLF001

    # Once the cooldown clears, an unattempted group is due before the group
    # that actually received the 403; it must not be hidden by stale selection.
    monkeypatch.setattr(transit, "_gmb_cooldown_until", 0)
    monkeypatch.setattr(transit, "_probe_network_refresh_at", None)
    calls.clear()
    await transit.fetch_probe_etas(object(), probes, max_per_cycle=1)
    assert calls == ["stop-1"]


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
async def test_nonblocking_probe_refresh_publishes_for_a_later_map_frame(monkeypatch):
    probe = SimpleNamespace(
        operator="GMB", route="11", bound="seq-1", stop_id="stop",
        route_id=1, sequence=1, index=0,
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def fetch(_client, _probe):
        started.set()
        await release.wait()
        return {"data": [{
            "enabled": True, "route_id": 1, "route_seq": 1, "stop_seq": 1,
            "eta": [{"diff": 4}],
        }]}

    monkeypatch.setattr(transit, "TRANSIT_NETWORK_REFRESH_SECONDS", 30.0)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    immediate = await transit.fetch_probe_snapshot(
        object(), [probe], wait_for_refresh=False, generation_probes=[probe]
    )
    assert immediate.complete_routes == ()
    await asyncio.wait_for(started.wait(), timeout=1)
    refresh = transit._probe_refresh_task  # noqa: SLF001
    assert refresh is not None and not refresh.done()

    release.set()
    await refresh
    later = await transit.fetch_probe_snapshot(
        object(), [probe], wait_for_refresh=False, generation_probes=[probe]
    )
    assert len(later.complete_routes) == 1
    assert later.complete_routes[0].observed_checkpoint_indices == {0}


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ("failure", "cancellation"))
async def test_complete_kmb_generation_publishes_while_gmb_sweep_is_pending(
    monkeypatch, ending
):
    """A blocked later GMB request cannot hide an already complete KMB route."""
    def probe(operator, route, stop_id, index):
        return SimpleNamespace(
            operator=operator,
            route=route,
            bound="seq-1" if operator == "GMB" else "outbound",
            stop_id=stop_id,
            route_id=1,
            sequence=1,
            index=index,
        )

    kmb = probe("KMB", "91", "kmb", 0)
    gmb_first = probe("GMB", "11", "gmb-a", 0)
    gmb_blocked = probe("GMB", "11", "gmb-b", 1)
    probes = [kmb, gmb_first, gmb_blocked]
    blocked = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.Event()

    async def fetch(_client, selected):
        if selected.stop_id != "gmb-b":
            return {"data": []}  # successful empty responses are complete evidence
        blocked.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise RuntimeError("later GMB group failed")

    monkeypatch.setattr(transit, "TRANSIT_NETWORK_REFRESH_SECONDS", 30.0)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    initial = await transit.fetch_probe_snapshot(
        object(), probes, max_per_cycle=3, wait_for_refresh=False, generation_probes=probes,
    )
    assert initial.complete_routes == ()
    await asyncio.wait_for(blocked.wait(), timeout=1)
    refresh = transit._probe_refresh_task  # noqa: SLF001
    assert refresh is not None and not refresh.done()

    visible = await transit.fetch_probe_snapshot(
        object(), probes, max_per_cycle=3, wait_for_refresh=False, generation_probes=probes,
    )
    assert [route.route_key for route in visible.complete_routes] == [
        ("KMB", "91", "outbound"),
    ]
    assert visible.complete_routes[0].rows == ()
    assert ("GMB", "11", "seq-1") not in {
        route.route_key for route in visible.complete_routes
    }

    if ending == "failure":
        release.set()
        await refresh
    else:
        refresh.cancel()
        with pytest.raises(asyncio.CancelledError):
            await refresh
        assert cancelled.is_set()
    await asyncio.sleep(0)

    # Avoid launching a new sweep: this reads the retained publication after
    # the pending GMB work failed or was cancelled.
    monkeypatch.setattr(transit, "TRANSIT_NETWORK_REFRESH_SECONDS", 3_600.0)
    monkeypatch.setattr(transit, "_probe_network_refresh_at", transit.time.monotonic())
    retained = await transit.fetch_probe_snapshot(
        object(), probes, max_per_cycle=3, wait_for_refresh=False, generation_probes=probes,
    )
    assert [route.route_key for route in retained.complete_routes] == [
        ("KMB", "91", "outbound"),
    ]
    assert retained.complete_routes[0].generation == visible.complete_routes[0].generation


@pytest.mark.asyncio
async def test_shutdown_cancels_detached_probe_refresh(monkeypatch):
    probe = SimpleNamespace(
        operator="GMB", route="11", bound="seq-1", stop_id="stop",
        route_id=1, sequence=1, index=0,
    )
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def fetch(_client, _probe):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    await transit.fetch_probe_etas(object(), [probe], wait_for_refresh=False)
    await asyncio.wait_for(started.wait(), timeout=1)

    await transit.shutdown_background_refreshes()

    assert cancelled.is_set()
    assert transit._probe_refresh_task is None  # noqa: SLF001
    assert transit._probe_network_refresh_at is None  # noqa: SLF001


@pytest.mark.asyncio
async def test_probe_snapshot_publishes_whole_route_with_shared_generation(monkeypatch):
    mono = [100.0]
    wall = [s.utc()]
    monkeypatch.setattr(transit.time, "monotonic", lambda: mono[0])
    monkeypatch.setattr(transit, "_probe_mono_clock", lambda: mono[0])
    monkeypatch.setattr(transit, "_probe_wall_clock", lambda: wall[0])
    probes = [
        SimpleNamespace(
            operator="GMB", route="11", bound="seq-1", stop_id=stop,
            route_id=1, sequence=1, index=index,
        )
        for index, stop in enumerate(("first", "second"))
    ]

    async def fetch(_client, probe):
        return {
            "data": [{
                "enabled": True, "route_id": 1, "route_seq": 1,
                "stop_seq": probe.index + 1,
                "eta": [{"diff": 4}],
            }],
        }

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    snapshot = await transit.fetch_probe_snapshot(object(), probes, max_per_cycle=2)
    assert len(snapshot.routes) == 1
    route = snapshot.routes[0]
    assert route.generation > 0
    assert route.rows
    assert all(0 < row.refresh_generation < route.generation for row in route.rows)

    async def fail(_client, _probe):
        fail.calls += 1
        raise RuntimeError("temporary outage")
    fail.calls = 0

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fail)
    mono[0] += transit.TRANSIT_NETWORK_REFRESH_SECONDS + 1
    wall[0] += timedelta(seconds=1)
    retained = await transit.fetch_probe_snapshot(object(), probes, max_per_cycle=2)
    assert fail.calls == 2
    assert retained.routes[0].generation == route.generation
    assert [row.stop_id for row in retained.routes[0].rows] == [row.stop_id for row in route.rows]


@pytest.mark.asyncio
async def test_gmb_pending_route_completion_is_atomic_under_sustained_priorities(monkeypatch):
    """A moved anchor is visible for positioning, but cannot publish a split route."""
    now = s.utc()

    def probe(index, stop_id=None):
        return SimpleNamespace(
            operator="GMB", route="R", bound="seq-1",
            stop_id=stop_id or f"anchor-{index}", route_id=1, sequence=1,
            index=index,
        )

    baseline = [probe(index) for index in range(4)]
    supplemental = [probe(index, f"active-{index}") for index in range(4, 16)]
    active = baseline + supplemental
    route_key = ("GMB", "R", "seq-1")
    calls: list[str] = []
    partial_cycle = [False]

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        if partial_cycle[0]:
            partial_cycle[0] = False
            return {"data": [{"enabled": True, "route_id": 1, "route_seq": 1,
                              "stop_seq": selected.index + 1,
                              "eta": [{"diff": 0, "timestamp":
                                       (now - timedelta(minutes=2)).isoformat()}]}]}
        return {"data": []}

    monkeypatch.setattr(transit, "_probe_wall_clock", lambda: now)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 15)
    monkeypatch.setattr(transit, "_probe_background_cursor", 0)

    initial = await transit.fetch_probe_snapshot(
        object(), baseline, max_per_cycle=4, generation_probes=baseline,
    )
    assert initial.complete_routes and initial.complete_routes[0].rows == ()
    initial_revision_floors = dict(
        initial.complete_routes[0].checkpoint_revisions
    )
    assert set(initial_revision_floors) == set(range(4))
    assert all(revision > 0 for revision in initial_revision_floors.values())
    initial_generation = initial.complete_routes[0].generation

    calls.clear()
    partial_cycle[0] = True
    partial = await transit.fetch_probe_snapshot(
        object(), baseline, max_per_cycle=1, generation_probes=baseline,
    )
    advanced_stop = calls[0]
    assert any(row.stop_id == advanced_stop and row.minutes == 0 and row.signed_minutes < 0
               for row in partial.positioning_rows)
    assert partial.complete_routes[0].generation == initial_generation
    assert partial.complete_routes[0].rows == ()
    assert dict(partial.complete_routes[0].checkpoint_revisions) == (
        initial_revision_floors
    )

    calls.clear()
    priorities = {route_key: set(range(4, 16))}
    completed = await transit.fetch_probe_snapshot(
        object(), active, max_per_cycle=36, priorities=priorities,
        generation_probes=baseline,
    )
    assert completed.complete_routes[0].generation > initial_generation
    assert any(row.index == 0 and row.signed_minutes < 0
               for row in completed.complete_routes[0].rows)
    assert len(calls) == len(set(calls)) <= 15
    # The three groups left behind by the one-anchor background cycle all
    # arrive in one coherent completion sweep, despite P=16 priorities.
    assert ({f"anchor-{index}" for index in range(4)} - {advanced_stop}) <= set(calls)

    completion_calls = set(calls)
    calls.clear()
    await transit.fetch_probe_snapshot(
        object(), active, max_per_cycle=36, priorities=priorities,
        generation_probes=baseline,
    )
    assert len(calls) == len(set(calls)) <= 15
    covered = completion_calls | set(calls) | {advanced_stop}
    assert {item.stop_id for item in active} <= covered


@pytest.mark.asyncio
async def test_gmb_pending_routes_are_oldest_first_and_share_physical_fetch(monkeypatch):
    """Pending route completion is ordered, and a shared stop is fetched once."""
    def probe(route, index, stop):
        return SimpleNamespace(operator="GMB", route=route, bound="seq-1",
                               stop_id=stop, route_id=1, sequence=1, index=index)

    first = [probe("A", 0, "shared"), probe("A", 1, "a-1")]
    second = [probe("B", 0, "shared"), probe("B", 1, "b-1")]
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 3)
    both = first + second
    await transit._refresh_probe_etas(object(), both, 3, generation_probes=both)  # noqa: SLF001
    first_generation = transit._probe_route_generations[("GMB", "A", "seq-1")].public.generation  # noqa: SLF001
    second_generation = transit._probe_route_generations[("GMB", "B", "seq-1")].public.generation  # noqa: SLF001
    calls.clear()
    await transit._refresh_probe_etas(
        object(), both, 1, {("GMB", "A", "seq-1"): {0}},
        generation_probes=both,
    )  # noqa: SLF001
    assert calls == ["shared"]

    calls.clear()
    completed = await transit.fetch_probe_snapshot(
        object(), both, max_per_cycle=2,
        priorities={("GMB", "A", "seq-1"): {0}}, generation_probes=both,
    )
    assert calls.count("shared") == 1
    by_route = {route.route_key: route for route in completed.complete_routes}
    assert by_route[("GMB", "A", "seq-1")].generation > first_generation
    assert by_route[("GMB", "B", "seq-1")].generation == second_generation

    calls.clear()
    completed_again = await transit.fetch_probe_snapshot(
        object(), both, max_per_cycle=2,
        priorities={("GMB", "B", "seq-1"): {0}}, generation_probes=both,
    )
    assert calls.count("shared") == 1
    by_route = {route.route_key: route for route in completed_again.complete_routes}
    assert by_route[("GMB", "B", "seq-1")].generation > second_generation


@pytest.mark.asyncio
async def test_gmb_completion_yields_to_oversized_active_floor(monkeypatch):
    """A 3-group lifecycle completion cannot consume capacity owed to P=29."""
    def probe(index, stop_id):
        return SimpleNamespace(operator="GMB", route="R", bound="seq-1",
                               stop_id=stop_id, route_id=1, sequence=1, index=index)

    baseline = [probe(index, f"base-{index}") for index in range(4)]
    active = baseline + [probe(index, f"active-{index:02d}") for index in range(4, 29)]
    route_key = ("GMB", "R", "seq-1")
    calls: list[str] = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 15)
    monkeypatch.setattr(transit, "_probe_background_cursor", 0)
    monkeypatch.setattr(transit, "_probe_priority_cursor", 0)
    await transit._refresh_probe_etas(object(), baseline, 4, generation_probes=baseline)  # noqa: SLF001
    published = transit._probe_route_generations[route_key].public.generation  # noqa: SLF001
    await transit._refresh_probe_etas(object(), baseline, 1, generation_probes=baseline)  # noqa: SLF001

    calls.clear()
    priorities = {route_key: set(range(4, 29))}
    await transit._refresh_probe_etas(
        object(), active, 36, priorities, generation_probes=baseline,
    )  # noqa: SLF001
    assert transit._probe_route_generations[route_key].public.generation == published  # noqa: SLF001
    assert len(calls) == len(set(calls)) <= 15
    # One GMB slot is reserved for the baseline lifecycle work; completion
    # must not consume any further active-floor capacity.
    assert len(set(calls) & {f"active-{index:02d}" for index in range(4, 29)}) >= 14


@pytest.mark.asyncio
async def test_requested_lifecycle_census_leads_crowded_priority_ring(monkeypatch):
    """A fresh sparse census leads carry without breaking two-cycle service."""
    def probe(route, index, stop_id):
        return SimpleNamespace(
            operator="GMB", route=route, bound="seq-1", stop_id=stop_id,
            route_id=1, sequence=1, index=index,
        )

    baseline = [probe("R", index, f"base-{index}") for index in range(4)]
    competitors = [
        probe(f"C{index:02d}", 0, f"active-{index:02d}")
        for index in range(25)
    ]
    active = baseline + competitors
    route_key = ("GMB", "R", "seq-1")
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in competitors
    }
    priorities[route_key] = set(range(4))
    calls = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    _seed_warm_routes(baseline)
    monkeypatch.setattr(transit, "_probe_generation", 10)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 15)
    transit._probe_priority_owed.update(  # noqa: SLF001
        transit._fetch_group_key(item) for item in competitors  # noqa: SLF001
    )

    await transit._refresh_probe_etas(  # noqa: SLF001
        object(), active, 36, priorities,
        generation_probes=baseline,
        lifecycle_routes={route_key: 1},
    )

    assert transit._probe_route_generations[route_key].public.generation > 1  # noqa: SLF001
    assert {item.stop_id for item in baseline} <= set(calls)
    assert len(calls) == len(set(calls)) == 15
    assert len(set(calls) & {item.stop_id for item in competitors}) == 11

    first_cycle = set(calls)
    calls.clear()
    await transit._refresh_probe_etas(  # noqa: SLF001
        object(), active, 36, priorities,
        generation_probes=baseline,
        lifecycle_routes={route_key: 1},
    )
    assert len(calls) == len(set(calls)) == 15
    assert {item.stop_id for item in competitors} <= first_cycle | set(calls)


@pytest.mark.asyncio
async def test_successive_lifecycle_tokens_preserve_carried_service(monkeypatch):
    """Renewed censuses share a capped page with continuously owed probes."""
    def probe(route, index):
        return SimpleNamespace(
            operator="GMB", route=route, bound="seq-1",
            stop_id=f"{route}-{index}", route_id=index + 1,
            sequence=1, index=index,
        )

    first = [probe("A", index) for index in range(4)]
    second = [probe("B", index) for index in range(4)]
    competitors = [probe(f"C{index}", 0) for index in range(2)]
    probes = first + second + competitors
    route_keys = (("GMB", "A", "seq-1"), ("GMB", "B", "seq-1"))
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in probes
    }
    calls = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    _seed_warm_routes(first + second)
    monkeypatch.setattr(transit, "_probe_generation", 10)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 5)
    transit._probe_priority_owed.update(  # noqa: SLF001
        transit._fetch_group_key(item) for item in competitors  # noqa: SLF001
    )

    per_cycle = []
    for _ in range(6):
        tokens = {
            key: transit._probe_route_generations[key].public.generation  # noqa: SLF001
            for key in route_keys
        }
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), probes, 5, priorities,
            generation_probes=first + second,
            lifecycle_routes=tokens,
        )
        assert len(calls) == len(set(calls)) == 5
        per_cycle.append(set(calls))

    competitor_stops = {item.stop_id for item in competitors}
    assert all(
        competitor_stops <= left | right
        for left, right in zip(per_cycle, per_cycle[1:], strict=False)
    )
    assert all(
        transit._probe_route_generations[key].public.generation > 1  # noqa: SLF001
        for key in route_keys
    )


@pytest.mark.asyncio
async def test_fresh_gmb_census_uses_reserved_share_before_mixed_quota(monkeypatch):
    """The GMB quota scorer sees carry after a fresh census is reserved."""
    def probe(operator, route, index, stop_id):
        return SimpleNamespace(
            operator=operator, route=route, bound="seq-1", stop_id=stop_id,
            route_id=route, sequence=1, index=index,
        )

    baseline = [probe("GMB", "R", index, f"base-{index}") for index in range(4)]
    gmb_carry = [probe("GMB", "H", index, f"gmb-{index}") for index in range(4)]
    ctb = [probe("CTB", "X", index, f"ctb-{index}") for index in range(64)]
    probes = baseline + gmb_carry + ctb
    route_key = ("GMB", "R", "seq-1")
    priorities = {}
    for item in probes:
        priorities.setdefault(
            (item.operator, item.route, item.bound), set()
        ).add(item.index)
    calls = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    _seed_warm_routes(baseline)
    monkeypatch.setattr(transit, "_probe_generation", 10)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 8)
    initially_owed = [*gmb_carry, *ctb[:32]]
    transit._probe_priority_owed.update(  # noqa: SLF001
        transit._fetch_group_key(item) for item in initially_owed  # noqa: SLF001
    )

    per_cycle = []
    generations = []
    for _ in range(6):
        token = transit._probe_route_generations[  # noqa: SLF001
            route_key
        ].public.generation
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), probes, 36, priorities,
            generation_probes=baseline,
            lifecycle_routes={route_key: token},
        )
        assert len(calls) == len(set(calls)) == 36
        assert {item.stop_id for item in baseline} <= set(calls)
        per_cycle.append(set(calls))
        generations.append(
            transit._probe_route_generations[route_key].public.generation  # noqa: SLF001
        )

    assert generations == sorted(set(generations))
    carried_gmb_stops = {item.stop_id for item in gmb_carry}
    assert all(
        carried_gmb_stops <= left | right
        for left, right in zip(per_cycle, per_cycle[1:], strict=False)
    )


@pytest.mark.asyncio
async def test_fresh_other_census_cannot_occupy_entire_other_quota(monkeypatch):
    """A one-group KMB census cannot indefinitely displace owed Citybus."""
    def probe(operator, route, index, stop_id):
        return SimpleNamespace(
            operator=operator, route=route, bound="outbound", stop_id=stop_id,
            route_id=route, sequence=1, index=index,
        )

    baseline = [probe("KMB", "R", index, f"base-{index}") for index in range(4)]
    citybus = probe("CTB", "X", 0, "ctb-owed")
    gmb = [probe("GMB", "H", index, f"gmb-{index}") for index in range(8)]
    probes = baseline + [citybus, *gmb]
    route_key = ("KMB", "R", "outbound")
    priorities = {}
    for item in probes:
        priorities.setdefault(
            (item.operator, item.route, item.bound), set()
        ).add(item.index)
    calls = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    _seed_warm_routes(baseline)
    monkeypatch.setattr(transit, "_probe_generation", 10)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 5)
    transit._probe_priority_owed.update(  # noqa: SLF001
        transit._fetch_group_key(item) for item in [citybus, *gmb]  # noqa: SLF001
    )

    per_cycle = []
    for _ in range(6):
        token = transit._probe_route_generations[  # noqa: SLF001
            route_key
        ].public.generation
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), probes, 5, priorities,
            generation_probes=baseline,
            lifecycle_routes={route_key: token},
        )
        assert len(calls) == len(set(calls)) == 5
        assert calls[0] == "base-0"
        per_cycle.append(set(calls))

    # One recurring census leaves eight carry slots across two pages for nine
    # physical carry groups, so three pages is the tight feasible deadline.
    carried_stops = {citybus.stop_id, *(item.stop_id for item in gmb)}
    assert all(
        carried_stops <= first_page | second_page | third_page
        for first_page, second_page, third_page in zip(
            per_cycle, per_cycle[1:], per_cycle[2:], strict=False,
        )
    )


@pytest.mark.asyncio
async def test_two_slot_fresh_census_preserves_warm_background(monkeypatch):
    """A fresh request leaves both priority and background service room."""
    def probe(operator, route, index, stop_id):
        return SimpleNamespace(
            operator=operator, route=route, bound="outbound", stop_id=stop_id,
            route_id=route, sequence=1, index=index,
        )

    requested = [probe("KMB", "R", index, f"base-{index}") for index in range(4)]
    active = [probe("CTB", "H", 0, "active")]
    background = [
        probe("CTB", "W", index, f"warm-{index}") for index in range(4)
    ]
    probes = requested + active + background
    route_key = ("KMB", "R", "outbound")
    background_key = ("CTB", "W", "outbound")
    priorities = {
        route_key: set(range(4)),
        ("CTB", "H", "outbound"): {0},
    }
    calls = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    _seed_warm_routes(probes)
    initial_requested_generation = transit._probe_route_generations[  # noqa: SLF001
        route_key
    ].public.generation
    initial_background_generation = transit._probe_route_generations[  # noqa: SLF001
        background_key
    ].public.generation
    monkeypatch.setattr(transit, "_probe_generation", 10)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)

    per_cycle = []
    for _ in range(8):
        token = transit._probe_route_generations[  # noqa: SLF001
            route_key
        ].public.generation
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), probes, 2, priorities,
            generation_probes=probes,
            lifecycle_routes={route_key: token},
        )
        assert len(calls) == len(set(calls)) == 2
        per_cycle.append(set(calls))

    assert all(
        "active" in left | right
        for left, right in zip(per_cycle, per_cycle[1:], strict=False)
    )
    background_stops = {item.stop_id for item in background}
    assert all(background_stops & page for page in per_cycle)
    assert transit._probe_route_generations[  # noqa: SLF001
        route_key
    ].public.generation > initial_requested_generation
    assert transit._probe_route_generations[  # noqa: SLF001
        background_key
    ].public.generation > initial_background_generation


@pytest.mark.asyncio
async def test_successive_fresh_census_preserves_warm_gmb_background(monkeypatch):
    """Recurring lifecycle requests cannot disable warm-route rotation."""
    def probe(route, index):
        return SimpleNamespace(
            operator="GMB", route=route, bound="seq-1",
            stop_id=f"{route}-{index}", route_id=route,
            sequence=1, index=index,
        )

    requested = [probe("R", index) for index in range(4)]
    active = [probe("H", 0)]
    background = [probe("W", index) for index in range(4)]
    probes = requested + active + background
    route_key = ("GMB", "R", "seq-1")
    background_key = ("GMB", "W", "seq-1")
    priorities = {
        route_key: set(range(4)),
        ("GMB", "H", "seq-1"): {0},
    }
    calls = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    _seed_warm_routes(probes)
    initial_requested_generation = transit._probe_route_generations[  # noqa: SLF001
        route_key
    ].public.generation
    initial_background_generation = transit._probe_route_generations[  # noqa: SLF001
        background_key
    ].public.generation
    monkeypatch.setattr(transit, "_probe_generation", 10)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 5)
    transit._probe_priority_owed.update(  # noqa: SLF001
        transit._fetch_group_key(item) for item in active  # noqa: SLF001
    )

    per_cycle = []
    for _ in range(8):
        token = transit._probe_route_generations[  # noqa: SLF001
            route_key
        ].public.generation
        calls.clear()
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), probes, 5, priorities,
            generation_probes=probes,
            lifecycle_routes={route_key: token},
        )
        assert len(calls) == len(set(calls)) == 5
        per_cycle.append(set(calls))

    active_stops = {item.stop_id for item in active}
    assert all(
        active_stops <= left | right
        for left, right in zip(per_cycle, per_cycle[1:], strict=False)
    )
    background_stops = {item.stop_id for item in background}
    assert all(
        background_stops & (left | right)
        for left, right in zip(per_cycle, per_cycle[1:], strict=False)
    )
    assert transit._probe_route_generations[  # noqa: SLF001
        background_key
    ].public.generation > initial_background_generation
    assert transit._probe_route_generations[  # noqa: SLF001
        route_key
    ].public.generation > initial_requested_generation


@pytest.mark.asyncio
async def test_requested_lifecycle_censuses_are_oldest_first(monkeypatch):
    """When only one sparse route fits, its publication moves it behind peers."""
    def probe(route, index):
        return SimpleNamespace(
            operator="GMB", route=route, bound="seq-1",
            stop_id=f"{route}-{index}", route_id=index + 1,
            sequence=1, index=index,
        )

    first = [probe("A", index) for index in range(4)]
    second = [probe("B", index) for index in range(4)]
    probes = first + second
    first_key = ("GMB", "A", "seq-1")
    second_key = ("GMB", "B", "seq-1")
    calls = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    _seed_warm_routes(probes)
    monkeypatch.setattr(transit, "_probe_generation", 10)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 4)

    await transit._refresh_probe_etas(  # noqa: SLF001
        object(), probes, 4, generation_probes=probes,
        lifecycle_routes={first_key: 1, second_key: 1},
    )
    assert calls == [item.stop_id for item in first]
    first_generation = transit._probe_route_generations[first_key].public.generation  # noqa: SLF001
    assert transit._probe_route_generations[second_key].public.generation == 1  # noqa: SLF001

    calls.clear()
    await transit._refresh_probe_etas(  # noqa: SLF001
        object(), probes, 4, generation_probes=probes,
        lifecycle_routes={first_key: 1, second_key: 1},
    )
    assert calls == [item.stop_id for item in second]
    assert transit._probe_route_generations[first_key].public.generation == first_generation  # noqa: SLF001
    assert transit._probe_route_generations[second_key].public.generation > 1  # noqa: SLF001


@pytest.mark.asyncio
async def test_requested_lifecycle_403_yields_to_unattempted_priorities(monkeypatch):
    """A failed anchor retries behind groups skipped when its sweep stopped."""
    def probe(route, index, stop_id):
        return SimpleNamespace(
            operator="GMB", route=route, bound="seq-1", stop_id=stop_id,
            route_id=1, sequence=1, index=index,
        )

    baseline = [probe("R", index, f"base-{index}") for index in range(4)]
    competitors = [
        probe(f"C{index}", 0, f"active-{index}") for index in range(4)
    ]
    probes = baseline + competitors
    route_key = ("GMB", "R", "seq-1")
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in probes
    }
    calls = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        if selected.stop_id == "base-0":
            raise transit.FetchError("rate limited", status_code=403)
        return {"data": []}

    _seed_warm_routes(baseline)
    monkeypatch.setattr(transit, "_probe_generation", 10)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 4)

    per_cycle = []
    for _ in range(3):
        calls.clear()
        monkeypatch.setattr(transit, "_gmb_cooldown_until", 0)
        await transit._refresh_probe_etas(  # noqa: SLF001
            object(), probes, 4, priorities,
            generation_probes=baseline,
            lifecycle_routes={route_key: 1},
        )
        per_cycle.append(list(calls))

    assert per_cycle[0] == ["base-0"]
    assert per_cycle[1][:3] == ["base-1", "base-2", "base-3"]
    assert {item.stop_id for item in competitors} <= set().union(
        *(set(cycle) for cycle in per_cycle[1:])
    )


@pytest.mark.asyncio
async def test_requested_lifecycle_keeps_cold_route_reservation(monkeypatch):
    """A warm lifecycle request cannot consume the bounded cold-route slot."""
    def probe(route, index, stop_id):
        return SimpleNamespace(
            operator="GMB", route=route, bound="seq-1", stop_id=stop_id,
            route_id=1, sequence=1, index=index,
        )

    baseline = [probe("R", index, f"base-{index}") for index in range(4)]
    cold = [probe("COLD", index, f"cold-{index}") for index in range(4)]
    competitors = [
        probe(f"P{index:02d}", 0, f"active-{index:02d}")
        for index in range(30)
    ]
    probes = baseline + cold + competitors
    route_key = ("GMB", "R", "seq-1")
    priorities = {
        (item.operator, item.route, item.bound): {item.index}
        for item in baseline + competitors
    }
    calls = []

    async def fetch(_client, selected):
        calls.append(selected.stop_id)
        return {"data": []}

    _seed_warm_routes(baseline)
    monkeypatch.setattr(transit, "_probe_generation", 10)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", 20)

    await transit._refresh_probe_etas(  # noqa: SLF001
        object(), probes, 36, priorities,
        generation_probes=baseline + cold,
        lifecycle_routes={route_key: 1},
    )

    assert any(stop_id.startswith("cold-") for stop_id in calls)
    assert {item.stop_id for item in baseline} <= set(calls)
    assert len(calls) == len(set(calls)) == 20


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
    assert aged_probe[0].minutes == pytest.approx(first_probe[0].minutes)
    assert aged_probe[0].cache_age_seconds == pytest.approx(10)
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
    assert rows and rows[0].minutes == 5


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


@pytest.mark.asyncio
async def test_gmb_gate_request_not_started_serves_cache_without_cooldown():
    cached = s.eta_row("11", "Choi Hung", "S", 5, operator=Operator.GMB)
    transit._gmb_gate_cache.set([cached])

    class Client:
        calls = 0

        async def fetch_json(self, _url):
            self.calls += 1
            raise RequestNotStarted("admission closed")

    client = Client()
    rows = await _fetch_gmb(client, s.utc())

    assert client.calls == 1
    assert rows == [cached]
    assert transit._gmb_cooldown_until == 0  # noqa: SLF001


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


@pytest.mark.asyncio
async def test_probe_attempt_token_is_stable_for_cache_only_snapshot(monkeypatch):
    probe = SimpleNamespace(
        operator="GMB", route="11", bound="seq-1", stop_id="stop",
        route_id=1, sequence=1, index=0,
    )

    async def fetch(_client, _probe):
        return {"data": [{"enabled": True, "route_id": 1, "route_seq": 1,
                           "eta": [{"diff": 4}]}]}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    monkeypatch.setattr(transit, "TRANSIT_NETWORK_REFRESH_SECONDS", 900.0)
    first = await transit.fetch_probe_snapshot(object(), [probe])
    second = await transit.fetch_probe_snapshot(
        object(), [probe], wait_for_refresh=False,
    )
    assert first.probe_attempt_generation == 1
    assert second.probe_attempt_generation == first.probe_attempt_generation
    assert second.attempted_checkpoints == first.attempted_checkpoints


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_probe_attempt_token_counts_started_success_or_failure(monkeypatch, fails):
    probe = SimpleNamespace(
        operator="CTB", route="X", bound="outbound", stop_id="stop",
        route_id=1, sequence=1, index=2,
    )

    async def fetch(_client, _probe):
        if fails:
            raise RuntimeError("unavailable")
        return {"data": []}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    await transit._refresh_probe_etas(object(), [probe], generation_probes=[probe])  # noqa: SLF001
    assert transit._probe_attempt_generation == 1  # noqa: SLF001
    assert transit._probe_attempted_checkpoints == {
        ("CTB", "X", "outbound", 2),
    }  # noqa: SLF001


@pytest.mark.asyncio
async def test_malformed_probe_response_counts_as_attempt_without_partial_publish(
    monkeypatch,
):
    probes = [
        SimpleNamespace(
            operator="CTB", route="X", bound="outbound", stop_id=f"stop-{index}",
            route_id=1, sequence=1, index=index,
        )
        for index in range(2)
    ]
    calls = []

    async def fetch(_client, probe):
        calls.append(probe.index)
        return {"data": [None]} if probe.index == 0 else {"data": []}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    await transit._refresh_probe_etas(  # noqa: SLF001
        object(), probes, max_per_cycle=2, generation_probes=probes,
    )

    assert calls == [0, 1]
    assert transit._probe_attempt_generation == 1  # noqa: SLF001
    assert transit._probe_attempted_checkpoints == {  # noqa: SLF001
        ("CTB", "X", "outbound", 0),
        ("CTB", "X", "outbound", 1),
    }
    assert transit._probe_cache.get(  # noqa: SLF001
        transit._probe_cache_key(probes[0])  # noqa: SLF001
    ) is None
    assert transit._probe_cache.get(  # noqa: SLF001
        transit._probe_cache_key(probes[1])  # noqa: SLF001
    ) == []
    assert ("CTB", "X", "outbound") not in transit._probe_route_generations  # noqa: SLF001


@pytest.mark.asyncio
async def test_gmb_cooldown_skip_does_not_advance_probe_attempt_token(monkeypatch):
    probe = SimpleNamespace(
        operator="GMB", route="11S", bound="seq-1", stop_id="stop",
        route_id=1, sequence=1, index=0,
    )

    async def unexpected_fetch(_client, _probe):
        pytest.fail("a cooldown-skipped group must not start an HTTP request")

    monkeypatch.setattr(transit, "time", SimpleNamespace(monotonic=lambda: 10.0))
    monkeypatch.setattr(transit, "_gmb_cooldown_until", 20.0)
    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", unexpected_fetch)
    await transit._refresh_probe_etas(  # noqa: SLF001
        object(), [probe], max_per_cycle=1, generation_probes=[probe],
    )

    assert transit._probe_attempt_generation == 0  # noqa: SLF001
    assert transit._probe_attempted_checkpoints == frozenset()  # noqa: SLF001


@pytest.mark.asyncio
async def test_request_not_started_preserves_probe_service_accounting(monkeypatch):
    probe = SimpleNamespace(
        operator="GMB", route="11S", bound="seq-1", stop_id="stop",
        route_id=1, sequence=1, index=0,
    )
    route_key = (probe.operator, probe.route, probe.bound)
    group_key = transit._fetch_group_key(probe)  # noqa: SLF001
    _seed_warm_routes([probe])
    transit._probe_group_service_debt[group_key] = 7  # noqa: SLF001
    transit._probe_priority_owed.add(group_key)  # noqa: SLF001
    transit._probe_failed_groups.add(group_key)  # noqa: SLF001

    starts = []
    client = transit.HttpClient(
        SimpleNamespace(get=lambda *args, **kwargs: starts.append((args, kwargs)))
    )

    async def skip_before_start(_url):
        raise RequestNotStarted("admission closed")

    client._pace_origin = skip_before_start  # noqa: SLF001
    await transit._refresh_probe_etas(  # noqa: SLF001
        client, [probe], max_per_cycle=1,
        priorities={route_key: {probe.index}}, generation_probes=[probe],
    )

    assert starts == []
    assert transit._probe_attempt_generation == 0  # noqa: SLF001
    assert transit._probe_attempted_checkpoints == frozenset()  # noqa: SLF001
    assert transit._probe_group_service_debt[group_key] == 8  # noqa: SLF001
    assert group_key in transit._probe_priority_owed  # noqa: SLF001
    assert group_key in transit._probe_failed_groups  # noqa: SLF001
    assert transit._probe_route_generations[route_key].public.generation == 1  # noqa: SLF001


@pytest.mark.asyncio
async def test_retry_skip_retains_prior_started_failure_accounting(monkeypatch):
    probe = SimpleNamespace(
        operator="GMB", route="11S", bound="seq-1", stop_id="stop",
        route_id=1, sequence=1, index=0,
    )
    route_key = (probe.operator, probe.route, probe.bound)
    group_key = transit._fetch_group_key(probe)  # noqa: SLF001
    _seed_warm_routes([probe])
    transit._probe_group_service_debt[group_key] = 7  # noqa: SLF001
    transit._probe_priority_owed.add(group_key)  # noqa: SLF001
    starts = []

    class Response:
        status = 503
        headers = {"Content-Type": "application/json"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

    class Session:
        def get(self, *_args, **_kwargs):
            starts.append(transit.time.monotonic())
            return Response()

    client = transit.HttpClient(Session(), retry_attempts=2)
    pace_calls = 0

    async def block_retry(_url):
        nonlocal pace_calls
        pace_calls += 1
        if pace_calls == 2:
            transit._gmb_cooldown_until = transit.time.monotonic() + 60  # noqa: SLF001
            raise RequestNotStarted("retry admission closed")

    client._pace_origin = block_retry
    monkeypatch.setattr("dashboard.http.RETRY_BASE_DELAY", 0.0)

    tracker = MarkerTracker()
    remaining = {probe.index}
    tracker._priority_pending[route_key] = ((probe.index,), remaining)  # noqa: SLF001
    await transit._refresh_probe_etas(  # noqa: SLF001
        client, [probe], max_per_cycle=1,
        priorities={route_key: {probe.index}}, generation_probes=[probe],
    )
    tracker._ack_probe_attempts(SimpleNamespace(  # noqa: SLF001
        probe_attempt_generation=transit._probe_attempt_generation,  # noqa: SLF001
        attempted_checkpoints=transit._probe_attempted_checkpoints,  # noqa: SLF001
    ))

    assert len(starts) == 1
    assert transit._probe_attempt_generation == 1  # noqa: SLF001
    assert transit._probe_attempted_checkpoints == {  # noqa: SLF001
        (probe.operator, probe.route, probe.bound, probe.index),
    }
    assert transit._probe_group_service_debt[group_key] == 0  # noqa: SLF001
    assert group_key not in transit._probe_priority_owed  # noqa: SLF001
    assert group_key in transit._probe_failed_groups  # noqa: SLF001
    assert remaining == set()
    assert transit._probe_route_generations[route_key].public.generation == 1  # noqa: SLF001


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("total_cap", "gmb_cap", "competitor_count", "max_refinement_seconds"),
    [(3, 3, 0, 90), (8, 8, 11, 120), (36, 20, 35, 120)],
)
async def test_actual_provider_cadence_refines_coarse_marker_without_page_reset(
    monkeypatch,
    total_cap,
    gmb_cap,
    competitor_count,
    max_refinement_seconds,
):
    key = ("GMB", "11S", "seq-1")
    route_id = 2004826
    mono = [0.0]
    wall = [s.utc()]
    clock = SimpleNamespace(monotonic=lambda: mono[0])
    monkeypatch.setattr(transit, "time", clock)
    monkeypatch.setattr(transit, "_probe_mono_clock", lambda: mono[0])
    monkeypatch.setattr(transit, "_probe_wall_clock", lambda: wall[0])
    monkeypatch.setattr(
        transit,
        "_probe_cache",
        transit.ProbeEtaCache(clock=lambda: mono[0]),
    )
    monkeypatch.setattr(transit, "TRANSIT_NETWORK_REFRESH_SECONDS", 30.0)
    monkeypatch.setattr(transit, "GMB_GROUPS_PER_CYCLE", gmb_cap)

    stops = [
        Stop(f"target-{index}", f"Stop {index}", 22.33, 114.26 + index * 0.001)
        for index in range(29)
    ]
    route_line = RouteLine(
        "11S",
        "GMB",
        "seq-1",
        stops,
        [(stop.lat, stop.lon) for stop in stops],
        [float(index * 100) for index in range(len(stops))],
    )
    target_probes = [
        ProbeStop("GMB", "11S", "seq-1", stop.stop_id, route_id, 1, index)
        for index, stop in enumerate(stops)
    ]
    baseline = [target_probes[index] for index in (0, 12, 24, 28)]
    competitors = [
        ProbeStop(
            "GMB", f"COMP-{index}", "seq-1", f"competitor-{index}",
            3000000 + index, 1, 0,
        )
        for index in range(competitor_count)
    ]
    competitor_priorities = {
        (probe.operator, probe.route, probe.bound): {probe.index}
        for probe in competitors
    }
    calls_by_sweep = {}

    async def fetch(_client, probe):
        calls_by_sweep.setdefault(mono[0], []).append(probe)
        if probe.route != "11S":
            return {"data": []}
        eta = []
        if probe.index >= 8:
            arrival = wall[0] + timedelta(minutes=(probe.index - 7.5) * 2)
            eta = [{"eta_seq": 1, "diff": 1, "timestamp": arrival.isoformat()}]
        return {
            "generated_timestamp": wall[0].isoformat(),
            "data": [{
                "enabled": True,
                "route_id": route_id,
                "route_seq": 1,
                "stop_seq": probe.index + 1,
                "eta": eta,
            }],
        }

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    tracker = MarkerTracker()
    first_track_id = None
    first_marker_at = None
    target_attempts = set()
    marker = None

    for _tick in range(50):
        priorities = {
            route_key: set(indices)
            for route_key, indices in tracker.poll_priorities().items()
        }
        priorities.update(competitor_priorities)
        target_indices = {probe.index for probe in baseline}
        target_indices.update(priorities.get(key, ()))
        active = [target_probes[index] for index in sorted(target_indices)]
        active.extend(competitors)
        snapshot = await transit.fetch_probe_snapshot(
            object(),
            active,
            max_per_cycle=total_cap,
            priorities=priorities,
            wait_for_refresh=False,
            generation_probes=baseline,
        )
        observed = {}
        for operator, route, bound, index in snapshot.positioning_checkpoints:
            observed.setdefault((operator, route, bound), set()).add(index)
        candidates = estimate_bus_positions(
            snapshot.positioning_rows or (),
            [route_line],
            observed_checkpoint_indices=observed,
        )
        markers = await tracker.update(snapshot, candidates, [route_line])
        if markers:
            assert len(markers) == 1
            marker = markers[0]
            if first_track_id is None:
                first_track_id = marker.track_id
                first_marker_at = mono[0]
            assert marker.track_id == first_track_id
            if marker.bracket == (7.0, 8.0):
                break

        refresh = transit._probe_refresh_task  # noqa: SLF001
        if refresh is not None:
            await refresh
        for probe in calls_by_sweep.get(mono[0], ()):
            if probe.route == "11S":
                target_attempts.add(probe.index)
        mono[0] += 15.0
        wall[0] += timedelta(seconds=15)

    assert marker is not None
    assert marker.bracket == (7.0, 8.0)
    assert marker.position == pytest.approx(7.5)
    assert {7, 8} <= target_attempts
    assert first_marker_at is not None
    assert mono[0] - first_marker_at <= max_refinement_seconds
    for sweep in calls_by_sweep.values():
        assert len(sweep) <= total_cap
        assert sum(probe.operator == "GMB" for probe in sweep) <= gmb_cap


@pytest.mark.asyncio
async def test_probe_attempt_token_expands_shared_gmb_aliases(monkeypatch):
    probes = [
        SimpleNamespace(operator="GMB", route=route, bound="seq-1", stop_id="shared",
                        route_id=1, sequence=1, index=index)
        for route, index in (("A", 0), ("B", 3))
    ]

    async def fetch(_client, _probe):
        return {"data": []}

    monkeypatch.setattr(transit, "_fetch_raw_stop_eta", fetch)
    await transit._refresh_probe_etas(
        object(), probes[:1], generation_probes=probes,
    )  # noqa: SLF001
    assert transit._probe_attempted_checkpoints == {
        ("GMB", "A", "seq-1", 0), ("GMB", "B", "seq-1", 3),
    }  # noqa: SLF001
