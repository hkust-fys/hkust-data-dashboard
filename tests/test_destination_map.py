"""Destination-map tests: bound-correct wording from gate ETA groups."""

from dashboard.maps import _destination_map
from dashboard.models import EtaKind, EtaRow, Operator, RouteEtaGroup
from dashboard.providers.route_geometry import RouteLine, Stop


def _line(operator: str, route: str, bound: str, destination: str, stops):
    return RouteLine(
        route, operator, bound,
        [Stop(sid, name, lat, lon) for sid, name, lat, lon in stops],
        destination=destination,
    )


def _group(route, dest, gate, operator=Operator.GMB):
    return RouteEtaGroup(
        route=route,
        destination=dest,
        gate=gate,
        operator=operator,
        rows=[EtaRow(route, dest, gate, operator, 5, EtaKind.REALTIME)],
    )


def test_gate_destination_lands_only_on_the_matching_direction():
    south = Stop("S-GATE", "H.K.U.S.T. (SOUTH)", 22.333360, 114.262881)
    north = Stop("N-GATE", "H.K.U.S.T. (NORTH)", 22.338678, 114.261946)
    lines = [
        # 11 seq-1 passes SOUTH gate heading to Choi Hung.
        _line("GMB", "11", "seq-1", "Choi Hung Station", [
            ("A", "Hang Hau Village", 22.312, 114.264),
            (south.stop_id, south.name, south.lat, south.lon),
            ("C", "Choi Hung", 22.334, 114.211),
        ]),
        # 11 seq-2 passes NORTH gate heading to Hang Hau.
        _line("GMB", "11", "seq-2", "Hang Hau Village", [
            ("C", "Choi Hung", 22.334, 114.211),
            (north.stop_id, north.name, north.lat, north.lon),
            ("A", "Hang Hau Village", 22.312, 114.264),
        ]),
    ]
    groups = [_group("11", "Choi Hung", "S"), _group("11", "Hang Hau", "N")]
    result = _destination_map(groups, lines)
    assert result[("GMB", "11", "seq-1")] == "Choi Hung"
    assert result[("GMB", "11", "seq-2")] == "Hang Hau"


def test_official_terminus_is_the_fallback_without_groups():
    line = _line("KMB", "91", "outbound", "DIAMOND HILL STATION", [
        ("A", "CWB Terminus", 22.287, 114.287),
        ("B", "Middle", 22.31, 114.27),
        ("C", "Diamond Hill", 22.34, 114.20),
    ])
    result = _destination_map([], [line])
    # Official termini pass through the shorthand normalizer.
    assert result[("KMB", "91", "outbound")] == "Diamond Hill"


def test_ambiguous_destination_does_not_overwrite_other_bound():
    south = Stop("S-GATE", "H.K.U.S.T. (SOUTH)", 22.333360, 114.262881)
    north = Stop("N-GATE", "H.K.U.S.T. (NORTH)", 22.338678, 114.261946)
    lines = [
        _line("CTB", "792M", "outbound", "Sai Kung", [
            ("X", "TKO Station", 22.307, 114.259),
            (south.stop_id, south.name, south.lat, south.lon),
            ("Y", "Sai Kung Pier", 22.288, 114.272),
        ]),
        _line("CTB", "792M", "inbound", "Tseung Kwan O", [
            ("Y", "Sai Kung Pier", 22.288, 114.272),
            (north.stop_id, north.name, north.lat, north.lon),
            ("X", "TKO Station", 22.307, 114.259),
        ]),
    ]
    # The South-gate group matches only the outbound line (which contains the
    # South stop); it must not relabel the inbound direction.
    groups = [_group("792M", "Sai Kung", "O", Operator.CITYBUS)]
    result = _destination_map(groups, lines)
    assert result[("CTB", "792M", "outbound")] == "Sai Kung"
