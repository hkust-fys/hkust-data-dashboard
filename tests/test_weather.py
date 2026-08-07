"""HKO weather provider tests: warning-code normalization, tolerance of empty
objects/null fields, and observation parsing."""

from dashboard.providers.weather import (
    KNOWN_WARNING_CODES,
    parse_observations,
    parse_warnings,
)
from tests.fixtures import sample_data as s


def test_parse_observations_sai_kung():
    snap = parse_observations(s.hko_rhrread())
    assert snap.temperature_c == 28.5
    assert snap.rainfall_mm == 0.0
    assert snap.humidity_pct == 71
    assert snap.source_time is not None


def test_parse_observations_tolerates_missing_sections():
    snap = parse_observations({})
    assert snap.temperature_c is None
    assert snap.rainfall_mm is None
    assert snap.humidity_pct is None


def test_parse_observations_ignores_wrong_station():
    raw = {
        "temperature": {"data": [{"place": "Wong Tai Sin", "value": 31.0}]},
        "rainfall": {"data": [{"place": "Wong Tai Sin", "max": 3.0}]},
        "humidity": {"data": [{"place": "Wong Tai Sin", "value": 60}]},
    }
    snap = parse_observations(raw)
    assert snap.temperature_c is None
    assert snap.rainfall_mm is None
    assert snap.humidity_pct is None


def test_parse_warnings_normalizes_codes_in_order():
    warnings = parse_warnings(s.hko_warnsum(("RAIN", "TC")), s.hko_warning_info())
    codes = [w.code for w in warnings]
    # known-code order: TC before RAIN regardless of input order
    assert codes == ["TC", "RAIN"]
    tc = warnings[0]
    assert tc.name == "Typhoon"
    assert tc.summary == "Tropical Cyclone Warning"
    assert tc.action == "Stay indoors"


def test_parse_warnings_empty_objects():
    assert parse_warnings({}) == []
    assert parse_warnings({"TC": {}}) == []
    assert parse_warnings(None) == []


def test_parse_warnings_unknown_codes_sorted_last():
    warnings = parse_warnings({"TC": {"code": "TC"}, "XYZ": {"code": "XYZ"}}, None)
    codes = [w.code for w in warnings]
    assert codes == ["TC", "XYZ"]
    assert KNOWN_WARNING_CODES[0] == "TC"


def test_parse_warnings_null_fields_tolerated():
    warnings = parse_warnings(
        {"HOT": {"code": "HOT"}},
        {"details": {"HOT": {"summary": None, "action": None}}},
    )
    assert warnings[0].summary == ""
    assert warnings[0].action == ""
