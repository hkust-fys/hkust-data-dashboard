# HKUST Campus Data Dashboard

A Discord bot that edits one persistent dashboard message with public-transport
ETAs from the HKUST gates, HKO weather and warnings, traffic information, a
Google Maps traffic-layer base map, and live North/South Gate camera frames.

## What it shows

- KMB, Citybus, and green-minibus ETAs in a stable order. Scheduled or otherwise
  non-realtime estimates are labelled, and routes with no departures are hidden.
- A Google Maps traffic-layer screenshot with coarse bus estimates and official
  bus-stop markers offset beside their route direction. Bus markers are
  estimates, not vehicle GPS; the screenshot is the map base and is refreshed
  with each dashboard update.
- HKO observations and active warning signals, plus official TD incidents and
  roadworks relevant to the campus approaches.
- Fresh JPEG frames decoded from the official HKUST North and South Gate HLS
  streams. Camera failures do not stop the rest of the dashboard.
- A link to the official HKUST shuttle schedule. The bot does not scrape the
  timetable or call a private shuttle API.

Every embed carries the timestamp of the data it displays. Providers fail
independently, and bounded HTTP caches can serve a labelled stale response after
a transient fetch failure. Discord field, embed, attachment, and total-character
limits are enforced by the renderer.

## Data sources

The runtime uses official transit APIs from KMB, Citybus, and TD's GMB service;
HKO Open Data; TD detector data, traffic news, and roadworks; Google Maps
browser screenshots with the traffic layer; the official HKUST bus-stop live
view; and the
[official HKUST shuttle schedule](https://cso.ust.hk/tran/stud_sh_b) as a link.

HKUST's keyed bus-queue, people-count, and SSC indexes are not used because
their latest records were stale when rechecked in August 2026. RTHK traffic
pages and the shuttle timetable are not scraped.

## Install

Requires Python 3.11+.

```bash
git clone https://github.com/hkust-fys/hkust-data-dashboard.git
cd hkust-data-dashboard
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

`imageio-ffmpeg` supplies the ffmpeg executable used to decode HLS camera
segments. Startup performs an ffmpeg preflight; fix that dependency before
enabling camera frames if the check fails.

Copy `.env.example` to `.env` and set the required values. Never commit `.env`.

| Variable | Required | Purpose |
|---|---|---|
| `DISCORD_TOKEN` | production | Discord bot token |
| `ANNOUNCE_CHANNEL_ID` | production | Channel containing the dashboard |
| `DASHBOARD_MESSAGE_ID` | optional | Existing bot-authored dashboard message to reuse |
| `DEV_WEBHOOK` | development | One-shot preview webhook |
| `ALERT_ROLE_ID` | optional | Role used for configured critical alerts |
| `UPDATE_INTERVAL_SECONDS` | optional | Dashboard edit interval; default 30 seconds |
| `HTTP_TIMEOUT_SECONDS` | optional | Per-request timeout; default 10 seconds |
| `CACHE_DIR` | optional | Bounded cache directory; default `.cache` |
| `LOG_LEVEL` | optional | Standard Python log level; default `INFO` |

The bot needs `View Channel`, `Send Messages`, `Embed Links`, `Attach Files`,
and `Read Message History` in the target channel.

## Run

```bash
# Production: edit one persistent dashboard message
python bot.py

# Dry run: no Discord writes
python bot.py --dry-run --no-keys

# Development webhook: one-shot preview
python bot.py --dev-webhook
```

The Google Maps screenshot uses the configured 1920×1080 viewport and exact
traffic-layer base URL from `dashboard/maps/tiles.py`. Official TD route-stop
geometry is refreshed independently for marker placement. Other public sources
retain their own cadences, and their source timestamps—not the dashboard edit
time—are displayed.

## Test

Fixture tests require no network or credentials.

```bash
ruff check .
python -m compileall -q bot.py dashboard tests
python -m pytest -m "not live"
```

Opt-in smoke tests hit public endpoints:

```bash
python -m pytest -m live
```

The bot exposes a `hkust-dashboard` console command after installation. It also
supports one persistent message recovery: if the configured message is absent,
it finds its own bot-authored message or creates exactly one replacement.
