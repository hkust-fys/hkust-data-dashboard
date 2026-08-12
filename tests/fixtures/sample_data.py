"""Shared test fixtures (sanitized; no keys, no real message content)."""

from datetime import UTC, datetime, timedelta

from dashboard.models import (
    EtaKind,
    EtaRow,
    ImageAsset,
    Operator,
    Roadwork,
    RouteEtaGroup,
    SpeedBand,
    TrafficCorridorStatus,
    TrafficIncident,
    TrafficObservation,
    WeatherSnapshot,
    WeatherWarning,
)


def utc(hours: int = 12, minutes: int = 0) -> datetime:
    """A deterministic "now" for fixtures: today at 12:00 UTC (or override).

    Callers that need a genuinely current timestamp for freshness checks should
    pass explicit hours/minutes or use datetime.now directly.
    """
    return datetime.now(UTC).replace(hour=hours, minute=minutes, second=0, microsecond=0)


def now_utc() -> datetime:
    """True current UTC time for freshness-sensitive fixtures."""
    return datetime.now(UTC)


# --------------------------------------------------------------------------
# Transit
# --------------------------------------------------------------------------

def kmb_json(stop: str = "B002CEF0DBC568F5") -> dict:
    now = utc()
    eta = (now + timedelta(minutes=4)).isoformat()
    # The fixture returns only route 91 data; the stub in tests routes the same
    # payload to every stop URL, so callers must filter by route.
    return {
        "type": "ETA",
        "version": "1.0",
        "generated_timestamp": now.isoformat(),
        "data": [
            {
                "co": "KMB", "route": "91", "dir": "I", "service_type": 1,
                "seq": 1, "dest_en": "DIAMOND HILL", "eta_seq": 1,
                "eta": eta, "rmk_en": "Scheduled Bus",
                "data_timestamp": now.isoformat(),
            },
            {
                "co": "KMB", "route": "91", "dir": "I", "service_type": 1,
                "seq": 2, "dest_en": "DIAMOND HILL", "eta_seq": 2,
                "eta": (now + timedelta(minutes=14)).isoformat(), "rmk_en": "",
                "data_timestamp": now.isoformat(),
            },
            {
                "co": "KMB", "route": "91", "dir": "I", "service_type": 2,
                "seq": 3, "dest_en": "SHORT RUN", "eta_seq": 1,
                "eta": (now + timedelta(minutes=3)).isoformat(), "rmk_en": "",
                "data_timestamp": now.isoformat(),
            },
            {
                "co": "KMB", "route": "91M", "dir": "I", "service_type": 1,
                "seq": 4, "dest_en": "PO LAM", "eta_seq": 1,
                "eta": (now + timedelta(minutes=9)).isoformat(), "rmk_en": "Moving slowly",
                "data_timestamp": now.isoformat(),
            },
        ],
    }


def kmb_json_empty() -> dict:
    return {"type": "ETA", "version": "1.0", "generated_timestamp": utc().isoformat(), "data": []}


def citybus_json() -> dict:
    now = utc()
    return {
        "type": "ETA", "version": "2.0", "generated_timestamp": now.isoformat(),
        "data": [
            {
                "co": "CTB", "route": "792M", "dir": "O", "seq": 1, "stop": "003130",
                "dest_en": "SAI KUNG", "eta_seq": 1,
                "eta": (now + timedelta(minutes=6)).isoformat(), "rmk_en": "",
                "data_timestamp": now.isoformat(),
            },
            {
                "co": "CTB", "route": "792M", "dir": "O", "seq": 2, "stop": "003130",
                "dest_en": "SAI KUNG", "eta_seq": 2,
                "eta": "", "rmk_en": "KMB Cycle",
                "data_timestamp": now.isoformat(),
            },
        ],
    }


def gmb_json(stop_id: int = 20013011) -> dict:
    now = utc()
    return {
        "type": "ETA-Stop",
        "version": "1.0",
        "generated_timestamp": now.isoformat(),
        "data": [
            {
                "route_id": 2004828, "route_seq": 1, "stop_seq": 1, "enabled": True,
                "eta": [
                    {
                        "eta_seq": 1, "diff": 2,
                        "timestamp": (now + timedelta(minutes=2)).isoformat(),
                        "remarks_en": "",
                    },
                    {
                        "eta_seq": 2, "diff": 15,
                        "timestamp": (now + timedelta(minutes=15)).isoformat(),
                        "remarks_en": "Scheduled",
                    },
                ],
            },
            {
                "route_id": 2004826, "route_seq": 1, "stop_seq": 8, "enabled": True,
                "eta": [
                    {
                        "eta_seq": 1, "diff": 30,
                        "timestamp": (now + timedelta(minutes=30)).isoformat(),
                        "remarks_en": "delayed",
                    }
                ],
            },
            {
                "route_id": 2004828, "route_seq": 9, "stop_seq": 9, "enabled": False,
                "eta": [
                    {
                        "eta_seq": 1, "diff": 1,
                        "timestamp": (now + timedelta(minutes=1)).isoformat(),
                        "remarks_en": "",
                    }
                ],
            },
        ],
    }


def eta_row(route: str = "91", dest: str = "Diamond Hill", gate: str = "S",
            minutes: int | None = 5, kind: EtaKind = EtaKind.REALTIME,
            operator: Operator = Operator.KMB) -> EtaRow:
    return EtaRow(
        route=route, destination=dest, gate=gate, operator=operator,
        minutes=minutes, kind=kind, source_time=utc(),
    )


def route_groups() -> list[RouteEtaGroup]:
    return [
        RouteEtaGroup(
            route="91", destination="Diamond Hill", gate="S", operator=Operator.KMB,
            rows=[eta_row("91", "Diamond Hill", "S", 2), eta_row("91", "Diamond Hill", "S", 20)],
        ),
        RouteEtaGroup(
            route="11B", destination="Choi Hung", gate="S", operator=Operator.GMB,
            rows=[eta_row("11B", "Choi Hung", "S", 5, EtaKind.SCHEDULED, Operator.GMB)],
        ),
    ]


# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------

def hko_rhrread() -> dict:
    return {
        "updateTime": utc().isoformat(),
        "temperature": {"data": [{"place": "Sai Kung", "value": 28.5}]},
        "rainfall": {"data": [{"place": "Sai Kung", "max": 0.0}]},
        "humidity": {"data": [{"place": "Sai Kung", "value": 71}]},
    }


def hko_warnsum(active: tuple[str, ...] = ("TC", "RAIN")) -> dict:
    result: dict = {"updateTime": utc().isoformat()}
    for code in active:
        result[code] = {"code": code, "issueDateTime": utc().isoformat()}
    return result


def hko_warning_info() -> dict:
    return {
        "details": {
            "TC": {"summary": "Tropical Cyclone Warning", "action": "Stay indoors"},
            "RAIN": {"summary": "Amber Rainstorm", "action": ""},
        }
    }


def weather_snapshot() -> WeatherSnapshot:
    return WeatherSnapshot(
        temperature_c=28.5, rainfall_mm=0.0, humidity_pct=71, source_time=utc()
    )


def weather_warnings() -> list[WeatherWarning]:
    return [
        WeatherWarning(code="TC", name="Typhoon", summary="Tropical Cyclone Warning"),
        WeatherWarning(code="RAIN", name="Rainstorm", summary="Amber Rainstorm"),
    ]


# --------------------------------------------------------------------------
# Traffic
# --------------------------------------------------------------------------

DETECTOR_CSV = (
    "AID_ID_Number,District,Road_EN,Road_TC,Road_SC,Easting,Northing,Latitude,Longitude,Direction,Rotation\n"
    "AID1001,Sai Kung,Clear Water Bay Road near Fei Ngo Shan Road - Eastbound,清水灣道,清水湾道,1,1,22.3370,114.2260,East,90\n"
    "AID1002,Wong Tai Sin,Lung Cheung Road near Diamond Hill MTR Station - Westbound,龍翔道,龙翔道,1,1,22.3420,114.2010,West,270\n"
    "AID1003,Sai Kung,Hiram's Highway near Pak Sha Wan - Northbound,西貢公路,西贡公路,1,1,22.3630,114.2710,North,0\n"
    "AID1004,Kowloon City,Nathan Road near Mody Road - Southbound,彌敦道,弥敦道,1,1,22.3100,114.1700,South,180\n"
)

DETECTOR_XML = """<?xml version="1.0" encoding="UTF-8"?>
<raw_speed_volume_list xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <date>2026-08-06</date>
  <periods>
    <period>
      <period_from>12:00:00</period_from>
      <period_to>12:00:30</period_to>
      <detectors>
        <detector>
          <detector_id>AID1001</detector_id>
          <direction>East</direction>
          <lanes>
            <lane><lane_id>Fast Lane</lane_id><speed>15</speed><occupancy>40</occupancy><volume>10</volume><s.d.>0</s.d.><valid>Y</valid></lane>
            <lane><lane_id>Slow Lane</lane_id><speed>18</speed><occupancy>45</occupancy><volume>8</volume><s.d.>0</s.d.><valid>Y</valid></lane>
          </lanes>
        </detector>
        <detector>
          <detector_id>AID1002</detector_id>
          <direction>West</direction>
          <lanes>
            <lane><lane_id>Fast Lane</lane_id><speed>35</speed><occupancy>20</occupancy><volume>5</volume><s.d.>0</s.d.><valid>Y</valid></lane>
          </lanes>
        </detector>
        <detector>
          <detector_id>AID1003</detector_id>
          <direction>North</direction>
          <lanes>
            <lane><lane_id>Fast Lane</lane_id><speed>55</speed><occupancy>10</occupancy><volume>3</volume><s.d.>0</s.d.><valid>Y</valid></lane>
          </lanes>
        </detector>
        <detector>
          <detector_id>AID1004</detector_id>
          <direction>South</direction>
          <lanes>
            <lane><lane_id>Fast Lane</lane_id><speed>60</speed><occupancy>5</occupancy><volume>1</volume><s.d.>0</s.d.><valid>Y</valid></lane>
          </lanes>
        </detector>
      </detectors>
    </period>
  </periods>
</raw_speed_volume_list>
"""

SPECIAL_NEWS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<trafficNews>
  <item>
    <identifier>TN-1</identifier>
    <title>Incident on Clear Water Bay Road</title>
    <description>Traffic accident near Fei Ngo Shan Road, slow.</description>
    <location>Clear Water Bay Road</location>
    <direction>towards Kowloon</direction>
    <status>Active</status>
    <start_time>{start}</start_time>
  </item>
  <item>
    <identifier>TN-2</identifier>
    <title>Roadworks on Lung Cheung Road</title>
    <description>Lane closure near Diamond Hill.</description>
    <location>Lung Cheung Road</location>
    <direction>inbound</direction>
    <status>Active</status>
    <start_time>{start}</start_time>
  </item>
  <item>
    <identifier>TN-3</identifier>
    <title>Incident on Nathan Road</title>
    <description>Unrelated to HKUST corridors.</description>
    <location>Nathan Road</location>
    <direction>inbound</direction>
    <status>Active</status>
    <start_time>{start}</start_time>
  </item>
</trafficNews>
""".format(start=utc().isoformat())

ROADWORKS_JSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "id": "RW-1",
                "description": "Roadworks on Hang Hau Road near Ying Yip Road",
                "road": "Hang Hau Road",
                "start_date": utc().isoformat(),
                "end_date": (utc() + timedelta(days=2)).isoformat(),
            },
            "geometry": {"type": "Point", "coordinates": [114.26, 22.32]},
        },
        {
            "type": "Feature",
            "properties": {
                "id": "RW-2",
                "description": "Roadworks on Tsim Sha Tsui",
                "road": "Salisbury Road",
            },
            "geometry": {"type": "Point", "coordinates": [114.17, 22.29]},
        },
    ],
}


def traffic_statuses() -> list[TrafficCorridorStatus]:
    return [
        TrafficCorridorStatus(
            name="Clear Water Bay Road",
            direction="",
            observations=[
                TrafficObservation(
                    corridor="Clear Water Bay Road",
                    direction="",
                    description="Clear Water Bay Road near Fei Ngo Shan Road",
                    latitude=22.337, longitude=114.226,
                    speed_kmh=15, volume=120, occupancy_pct=0.4,
                    capture_time=utc(), band=SpeedBand.RED,
                )
            ],
            capture_time=utc(),
        ),
        TrafficCorridorStatus(
            name="Lung Cheung Road",
            direction="",
            observations=[
                TrafficObservation(
                    corridor="Lung Cheung Road", direction="",
                    description="Lung Cheung Road near Diamond Hill",
                    latitude=22.342, longitude=114.201,
                    speed_kmh=35, volume=80, occupancy_pct=0.2,
                    capture_time=utc(), band=SpeedBand.AMBER,
                )
            ],
            capture_time=utc(),
        ),
    ]


def traffic_incidents() -> list[TrafficIncident]:
    return [
        TrafficIncident(
            identifier="TN-1", title="Incident on Clear Water Bay Road",
            description="Traffic accident near Fei Ngo Shan Road, slow.",
            road="Clear Water Bay Road", location="Clear Water Bay Road",
            direction="towards Kowloon", status="Active", start_time=utc(),
        )
    ]


def roadworks() -> list[Roadwork]:
    return [
        Roadwork(
            identifier="RW-1",
            description="Roadworks on Hang Hau Road near Ying Yip Road",
            road="Hang Hau Road", start_time=utc(),
            end_time=utc() + timedelta(days=2),
        )
    ]


def jpeg_bytes() -> bytes:
    return b"\xff\xd8\xff\xe0" + b"\x00" * 128


def bus_stop_assets() -> list[ImageAsset]:
    return [
        ImageAsset(
            filename="busstop-0.jpg", data=jpeg_bytes(), content_type="image/jpeg",
            label="North Gate bus stop", caption="live view", source_time=utc(),
        )
    ]
