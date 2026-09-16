# HKUST Campus Data Dashboard

A Discord bot that edits one persistent dashboard message with public-transport
ETAs from the HKUST gates, HKO weather and warnings, traffic information, a
Google Maps traffic-layer base map, and live North/South Gate camera frames.

The runtime boundaries and provider dependency graph are described in
[docs/architecture.md](docs/architecture.md).

## What it shows

- KMB, Citybus, and green-minibus ETAs in a stable order. Scheduled or otherwise
  non-realtime estimates are labelled, and routes with no departures are hidden.
- A Google Maps traffic-layer screenshot with coarse bus estimates and official
  bus-stop markers offset beside their route direction. Bus markers are
  estimates, not vehicle GPS; the screenshot is the map base and uses the
  latest completed browser capture.
- HKO observations and active warning signals, TD special traffic notices and
  roadworks, and RTHK Chinese traffic reports mentioning roads the tracked buses use.
  Reports retain their source and available announcement time, including
  updates that an incident has cleared.
- Fresh JPEG frames decoded from the official HKUST North and South Gate HLS
  streams. Camera failures do not stop the rest of the dashboard.
- A link to the official HKUST shuttle schedule. The bot does not scrape the
  timetable or call a private shuttle API.

Source timestamps accompany the data they describe; traffic reports carry
individual timestamps instead of one combined source clock. Providers fail
independently, and bounded HTTP caches can serve a labelled stale response after
a transient fetch failure. Discord field, embed, attachment, and total-character
limits are enforced by the renderer.

## Data sources

The runtime uses official transit APIs from KMB, Citybus, and TD's GMB service;
HKO Open Data; TD detector data, traffic news, and roadworks; Google Maps
browser captures with the traffic layer; TD's full special-news page; RTHK's
current traffic-news page; the Lands Department's
[Road Centreline dataset](https://data.gov.hk/en-data/dataset/hk-landsd-openmap-road-centreline)
for bilingual road names and fallback road geometry; the official HKUST bus-stop
live view; and the
[official HKUST shuttle schedule](https://cso.ust.hk/tran/stud_sh_b) as a link.

HKUST's keyed bus-queue, people-count, and SSC indexes are not used because
their latest records were stale when rechecked in August 2026. The shuttle
timetable is not scraped.

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
| `ALERT_ROLE_ID` | optional | Role pinged for new congestion on Clear Water Bay Road / New Clear Water Bay Road |
| `UPDATE_INTERVAL_SECONDS` | optional | Dashboard edit interval; default/minimum 10 seconds |
| `HTTP_TIMEOUT_SECONDS` | optional | Per-request timeout; default 10 seconds |
| `CACHE_DIR` | optional | Bounded cache directory; default `.cache` |
| `LOG_LEVEL` | optional | Standard Python log level; default `INFO` |

The bot needs `View Channel`, `Send Messages`, `Embed Links`, `Attach Files`,
and `Read Message History` in the target channel.
Thread updates also need `Create Public Threads` and `Send Messages in Threads`.
For traffic pings, the configured role must be mentionable, or the bot needs
`Mention @everyone, @here, and All Roles` in that channel. Sends explicitly
allow only the configured role; weather and clearance messages allow no mentions.

## Run

```bash
# Production: edit one persistent dashboard message
python bot.py

# Dry run: no Discord writes
python bot.py --dry-run --no-keys

# Development webhook: one-shot preview
python bot.py --dev-webhook
```

The Google Maps screenshot uses a compact 960×540 viewport at zoom 14 and the
exact traffic-layer base URL from `dashboard/maps/tiles.py`; the traffic layer
is kept intact while dashboard markers and thin traffic-news rails are
added. The resulting WebP map attachment targets at most 100 KB for mobile
payloads. Official TD route-stop geometry is refreshed independently for marker
placement. Other public sources retain their own cadences, and their source
timestamps—not the dashboard edit time—are displayed.

The browser canvas is exported every presentation cycle (normally 10 seconds).
A replacement Google Maps page loads in the background before the active base
reaches one minute old. It replaces the active page only after its canvas has
finished loading and passed stability checks. Capture time and view-refresh
time are tracked separately; exporting unchanged pixels cannot renew the base's
age. Failures retry after 10 seconds. Any retained fallback is labelled, and a
base older than one minute is withheld.
TD and RTHK news pages are checked independently every 60 seconds. An empty
news result never means the roads are clear, and unavailable sources are
distinguished from a successfully checked page with no matching reports.
The incomplete special-traffic-news XML feed is excluded from runtime news
collection; TD detector measurements continue to use their separate XML feed.
English and Chinese names come from the official LandsD dataset catalogued on
data.gov.hk, served through CSDI. The dictionary loads from disk immediately,
refreshes in the background on startup and every 24 hours, and retains the last
good copy for at most 90 days. OSM bilingual tags supplement it. Road membership
is derived from official bus-route geometry, with LandsD centreline geometry
available when Overpass fails.

Weather warning changes and all relevant road reports go to the thread attached
to the dashboard message. An existing attached thread is reused; otherwise the
bot creates it from that message. RTHK matches show Chinese road names followed
by English names. Each TD report shows the page's update time, labelled as such;
RTHK reports show their individual publication times. TD's synchronized Chinese
page assists exact bilingual reconciliation. Unambiguous reports of the same
road, direction, landmark, cause and state share one entry with both source texts.
Conflicting or ambiguous reports remain separate, and a second source's
corroboration posts silently. The traffic links share one row; the map link is
labelled HKeMobility.

Notices on tracked roads remain visible even when no dashboard bus passes the
reported section. Named places are resolved with the official LandsD Location
Search API and compared with complete road centreline and bus-route geometry.
Only buses following that section and direction are listed. For notices without
a location, a bus covering at least half of the complete named road can be
listed as likely affected. An unresolved explicit landmark never falls back to
this road-wide estimate. Place/complete-road lookups have bounded seven-day
caches and a 30-minute failure retry interval.

Only the start of congestion on Clear Water Bay Road
or New Clear Water Bay Road can ping the traffic role; easing and clearance
updates never ping. Google traffic colours alone are not proof of a reportable
incident or the upstream data's age.

HKO warning icons are static official PNGs stitched into one PNG strip. Icon
bytes and the composed strip are cached; unchanged warnings reuse the uploaded
image while its Discord attachment URL remains valid. Changed warnings or an
expired attachment URL trigger a new upload.

## Test

The default test run is local and uses fixtures. Live endpoint checks, when
needed, should be run explicitly because they depend on upstream availability.

```bash
ruff check .
python -m compileall -q bot.py dashboard tests
python -m pytest
```

Use `.venv/Scripts/python.exe` on Windows. The suite does not require
production credentials; keep `.env` local.

The bot exposes a `hkust-dashboard` console command after installation. It also
supports one persistent message recovery: if the configured message is absent,
it finds its own bot-authored message or creates exactly one replacement.
