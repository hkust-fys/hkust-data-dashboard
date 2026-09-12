"""HKO weather provider tests: warning-code normalization, tolerance of empty
objects/null fields, and observation parsing."""

import io
from datetime import datetime

import pytest
from PIL import Image

from dashboard.models import WeatherWarning
from dashboard.providers.weather import (
    _fetch_warning_icons,
    _normalize_warning_icon,
    _warning_catalog_from_html,
    _warning_icon_cache,
    _warning_metadata_from_warntoday,
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
    warnings = parse_warnings(s.hko_warnsum(("WRAINA", "TC3")), s.hko_warning_info())
    codes = [w.code for w in warnings]
    # source order is retained within priority buckets: typhoon before rain.
    assert codes == ["TC3", "WRAINA"]
    tc = warnings[0]
    assert tc.name == "Strong Wind Signal No. 3"
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


def test_parse_warnings_null_fields_tolerated():
    warnings = parse_warnings(
        {"HOT": {"code": "HOT"}},
        {"details": {"HOT": {"summary": None, "action": None}}},
    )
    assert warnings[0].summary == ""
    assert warnings[0].action == ""


def test_pre_no8_statement_becomes_synthetic_warning():
    from dashboard.providers.weather import PRE_NO8_CODE

    info = {
        "details": {
            "TC": {
                "code": "TC",
                "statement": (
                    "The Hong Kong Observatory issues the Pre-No. 8 Special "
                    "Announcement at 6:15 p.m."
                ),
                "issueTime": "2026-08-21T18:15:00+08:00",
            }
        }
    }
    warnings = parse_warnings({"TC": {"code": "TC"}}, info)
    codes = [w.code for w in warnings]
    assert PRE_NO8_CODE in codes
    pre = next(w for w in warnings if w.code == PRE_NO8_CODE)
    assert pre.name == "Pre-No. 8 Special Announcement"
    assert pre.issued_at == datetime.fromisoformat("2026-08-21T18:15:00+08:00")


def test_no_pre_no8_without_the_statement():
    warnings = parse_warnings(
        {"TC": {"code": "TC"}},
        {"details": {"TC": {"code": "TC", "summary": "Gale Signal No. 8 NE in force"}}},
    )
    assert [w.code for w in warnings] == ["TC"]


def test_parse_warning_issued_time_does_not_use_later_update_time():
    warnings = parse_warnings(
        {
            "WHOT": {
                "code": "WHOT",
                "issueTime": "2026-08-05T06:45:00+08:00",
                "updateTime": "2026-08-13T06:45:00+08:00",
            }
        }
    )
    assert warnings[0].issued_at == datetime.fromisoformat("2026-08-05T06:45:00+08:00")


def test_live_wrain_uses_payload_code_and_official_metadata():
    catalog = _warning_catalog_from_html(
        '<img alt="AMBER RAINSTORM WARNING SIGNAL" src="/en/textonly/img/warn/images/rainamber.png">'
    )
    metadata = _warning_metadata_from_warntoday(s.hko_warntoday_wrain(), catalog)
    warnings = parse_warnings(
        s.hko_warnsum_wrain_live(), s.hko_warning_info_list(), warning_metadata=metadata
    )
    assert len(warnings) == 1
    assert warnings[0].code == "WRAINA"
    assert warnings[0].name == "Amber Rainstorm Warning Signal"
    assert warnings[0].icon_url == "https://www.hko.gov.hk/en/textonly/img/warn/images/rainamber.png"
    assert "WRAIN" not in warnings[0].name
    assert warnings[0].summary == ""


def test_live_strong_monsoon_code_resolves_unique_subtype_metadata_icon():
    catalog = _warning_catalog_from_html(
        '<img alt="STRONG MONSOON SIGNAL" src="../../../en/textonly/img/warn/images/sms.png">'
    )
    metadata = _warning_metadata_from_warntoday(s.hko_warntoday_monsoon(), catalog)
    warnings = parse_warnings(
        s.hko_warnsum_monsoon_live(), warning_metadata=metadata
    )
    assert warnings[0].code == "WMSGNL"
    assert warnings[0].icon_url == "https://www.hko.gov.hk/en/textonly/img/warn/images/sms.png"


def test_warning_metadata_prefix_fallback_rejects_ambiguous_variants():
    warnings = parse_warnings(
        {"WMSGNL": {"code": "WMSGNL"}},
        warning_metadata={
            "WMSGNL_MONSOON": ("Monsoon", "https://www.hko.gov.hk/images_e/msn.gif"),
            "WMSGNL_OTHER": ("Other", "https://www.hko.gov.hk/images_e/other.gif"),
        },
    )
    assert warnings[0].icon_url == ""


def test_warning_catalog_rejects_unsafe_and_ambiguous_images():
    catalog = _warning_catalog_from_html(
        '<img alt="safe" src="/en/textonly/img/warn/images/safe.png">'
        '<img alt="bad" src="https://evil.example/bad.png">'
        '<img alt="jpg" src="/en/textonly/img/warn/images/jpg.gif">'
        '<img alt="dup" src="/en/textonly/img/warn/images/a.png">'
        '<img alt="DUP" src="/en/textonly/img/warn/images/b.png">'
    )
    assert catalog["safe"] == "https://www.hko.gov.hk/en/textonly/img/warn/images/safe.png"
    assert "bad" not in catalog
    assert "jpg" not in catalog
    assert "dup" not in catalog


def test_warning_info_only_entry_does_not_become_active():
    warnings = parse_warnings(
        {"WRAIN": {"code": "WRAINA"}},
        {"details": [{"warningStatementCode": "WTS", "contents": ["Thunderstorm"]}]},
    )
    assert [warning.code for warning in warnings] == ["WRAINA"]


def test_warnsum_source_name_and_subtype_without_warntoday():
    warnings = parse_warnings(
        {"TC": {"code": "TC3", "type": "Strong Wind Signal No. 3", "name": "Tropical Cyclone Warning Signal"}},
        {"details": [{"warningStatementCode": "TC", "subtype": "TC3", "contents": ["Strong Wind Signal No. 3: take shelter"]}]},
    )
    assert warnings[0].name == "Strong Wind Signal No. 3"
    assert warnings[0].summary == "take shelter"


def test_warntoday_rejects_absolute_icon_url():
    metadata = _warning_metadata_from_warntoday({"WARNING_DATABASE": [{
        "WarningCode": "WTS", "WarningName": "Thunderstorm Warning",
        "Type": "", "Icon": "https://evil.example/icon.gif",
    }]})
    assert metadata["WTS"][1] == ""


def test_warntoday_rejects_protocol_relative_icon_url():
    metadata = _warning_metadata_from_warntoday({"WARNING_DATABASE": [{
        "WarningCode": "WTS", "WarningName": "Thunderstorm Warning",
        "Type": "", "Icon": "//evil.example/icon.gif",
    }]})
    assert metadata["WTS"][1] == ""


def test_warntoday_metadata_malformed_payload_fails_softly():
    assert _warning_metadata_from_warntoday(None) == {}  # type: ignore[arg-type]
    assert _warning_metadata_from_warntoday({"WARNING_DATABASE": {}}) == {}


@pytest.mark.asyncio
async def test_warning_icon_bytes_are_fetched_once_and_cached():
    class Client:
        calls = 0

        async def fetch_bytes(self, url, max_bytes):
            assert url == "https://www.hko.gov.hk/en/textonly/img/warn/images/rainamber.png"
            assert max_bytes == 256 * 1024
            self.calls += 1
            output = io.BytesIO()
            Image.new("RGB", (2, 2), "red").save(output, format="PNG")
            return output.getvalue()

    _warning_icon_cache.clear()
    client = Client()
    first = WeatherWarning(
        "WRAINA", "Amber Rainstorm Warning Signal",
        icon_url="https://www.hko.gov.hk/en/textonly/img/warn/images/rainamber.png",
    )
    second = WeatherWarning(
        "WRAINA", "Amber Rainstorm Warning Signal",
        icon_url="https://www.hko.gov.hk/en/textonly/img/warn/images/rainamber.png",
    )

    await _fetch_warning_icons(client, [first])
    await _fetch_warning_icons(client, [second])

    assert first.icon_data.startswith(b"\x89PNG")
    assert second.icon_data == first.icon_data
    assert client.calls == 1


def test_normalize_warning_png_writes_static_png():
    frames = [Image.new("RGBA", (8, 8), (255, 0, 0, 255))]
    output = io.BytesIO()
    frames[0].save(output, format="PNG")

    normalized = _normalize_warning_icon(output.getvalue())

    assert normalized is not None
    assert normalized.startswith(b"\x89PNG")
    with Image.open(io.BytesIO(normalized)) as image:
        assert image.format == "PNG"
        assert image.getbbox() is not None
        assert image.getpixel((4, 4))[:3] == (255, 0, 0)


@pytest.mark.asyncio
async def test_warning_gif_is_rejected_without_fetching():
    frames = [Image.new("RGBA", (8, 8), (255, 255, 255, 0)), Image.new("RGBA", (8, 8), (0, 0, 255, 255))]
    output = io.BytesIO()
    frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0, disposal=2)
    payload = output.getvalue()
    assert _normalize_warning_icon(payload) is None
    assert _normalize_warning_icon(b"not-a-png") is None

    class Client:
        calls = 0

        async def fetch_bytes(self, _url, max_bytes):
            assert max_bytes == 256 * 1024
            self.calls += 1
            return payload

    _warning_icon_cache.clear()
    client = Client()
    warning = WeatherWarning("WTS", "Thunderstorm Warning", icon_url="https://www.hko.gov.hk/images_e/ts.gif")
    await _fetch_warning_icons(client, [warning])
    assert warning.icon_data == b""
    assert client.calls == 0


@pytest.mark.asyncio
async def test_invalid_warning_icon_is_not_cached_and_retries():
    class Client:
        calls = 0

        async def fetch_bytes(self, _url, max_bytes):
            assert max_bytes == 256 * 1024
            self.calls += 1
            if self.calls == 1:
                return b"<html>temporary upstream error</html>"
            output = io.BytesIO()
            Image.new("RGB", (2, 2), "red").save(output, format="PNG")
            return output.getvalue()

    _warning_icon_cache.clear()
    client = Client()
    first = WeatherWarning(
        "WRAINA", "Amber Rainstorm Warning Signal",
        icon_url="https://www.hko.gov.hk/en/textonly/img/warn/images/rainamber.png",
    )
    second = WeatherWarning(
        "WRAINA", "Amber Rainstorm Warning Signal",
        icon_url="https://www.hko.gov.hk/en/textonly/img/warn/images/rainamber.png",
    )

    await _fetch_warning_icons(client, [first])
    await _fetch_warning_icons(client, [second])

    assert first.icon_data == b""
    assert second.icon_data.startswith(b"\x89PNG")
    assert client.calls == 2


@pytest.mark.asyncio
async def test_warning_icon_fetch_rejects_non_hko_but_accepts_minor_warning():
    class Client:
        calls = 0

        async def fetch_bytes(self, url, max_bytes):
            assert url == "https://www.hko.gov.hk/en/textonly/img/warn/images/ts.png"
            assert max_bytes == 256 * 1024
            self.calls += 1
            output = io.BytesIO()
            Image.new("RGB", (2, 2), "blue").save(output, format="PNG")
            return output.getvalue()

    warnings = [
        WeatherWarning("WRAINA", "Rain", icon_url="https://evil.example/rain.gif"),
        WeatherWarning(
            "WTS", "Thunderstorm", icon_url="https://www.hko.gov.hk/en/textonly/img/warn/images/ts.png"
        ),
    ]

    client = Client()
    await _fetch_warning_icons(client, warnings)

    assert warnings[0].icon_data == b""
    assert warnings[1].icon_data.startswith(b"\x89PNG")
    assert client.calls == 1
