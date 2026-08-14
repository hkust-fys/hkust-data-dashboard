"""Transit provider tests: KMB/Citybus/GMB parsing, ordering, markers, and the
verified GMB directional-variant IDs."""

from datetime import UTC

import pytest

from dashboard.models import EtaKind, Operator
from dashboard.providers.transit import (
    GMB_STOPS,
    KMB_STOPS,
    _fetch_citybus,
    _fetch_gmb,
    _fetch_kmb,
    group_etas,
)
from tests.fixtures import sample_data as s


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
