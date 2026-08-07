# HKUST Campus Data Dashboard

A Discord bot that keeps one concise, always-updated dashboard message in your
server: live bus/minibus ETAs at the HKUST gates, HKO weather and warning
signals, and Transport Department traffic status for the roads around campus —
including a generated traffic map, representative TD CCTV views, and the
official HKeMobility link.

## Features

- **One persistent message**: the bot edits a single dashboard message in place;
  embeds and attachments are replaced atomically. It never sends duplicate
  messages and never edits arbitrary user messages.
- **Transit ETAs**: KMB, Citybus, and green minibus departures at North/South
  Gate, with a stable route order, `*` = scheduled (not realtime), `!` = moving
  slowly, `‼` = delayed. Routes with no current departures are hidden.
- **Weather**: HKO observations for the Sai Kung station plus active warning
  signals (typhoon, rainstorm, thunderstorm, very hot/cold, monsoon, …).
- **Traffic**: TD detector speeds/volume/occupancy on HKUST-approach corridors
  (Clear Water Bay Road, Lung Cheung Road, Hiram's Highway, Hang Hau Road,
  Ying Yip Road, Tai Po Tsai Road, University Road, …), matching Special
  Traffic News notices, and planned roadworks.
  - Speed bands are dashboard heuristics: red < 20 km/h, amber 20–40 km/h,
    green > 40 km/h, gray = no fresh observation. Summaries say *observed
    slow/congested at monitored points* — they never claim full-road coverage.
  - A **traffic map** (1,024×600 PNG) overlays detector speeds on cached
    OpenStreetMap tiles, with `© OpenStreetMap contributors` attribution.
  - Two representative **TD CCTV point views** (Clear Water Bay Road and
    Lung Cheung Road) are attached and replaced on every edit so Discord does
    not cache stale camera frames. CCTV is a point view, not proof of corridor
    conditions.
  - The **HKeMobility** link provides the interactive territory-wide view.
- **Per-provider isolation**: one failing source never blocks the others; last
  good values are kept and marked stale rather than dropping the section.
- **Dry-run / dev-webhook modes** let you preview without touching Discord.

## Data sources

| Source | What it provides |
|---|---|
| [KMB ETA API](https://data.gov.hk/en-data/dataset/hk-td-tis_21-etakmb) | 91/91M/91P/291P ETAs |
| [Citybus ETA API](https://data.gov.hk/en-data/dataset/ctb-eta-transport-realtime-eta) | 792M ETAs |
| [GMB ETA API](https://data.gov.hk/en-data/dataset/hk-td-sm_7-real-time-arrival-data-of-gmb) | 11/11B/11M/11S/12/104 minibus ETAs (non-realtime, often inaccurate) |
| [HKO Open Data](https://data.weather.gov.hk/weatherAPI/opendata/) | rhrread observations, warnsum, warningInfo |
| [TD — Traffic Data of Strategic / Major Roads](https://data.gov.hk/en-data/dataset/hk-td-sm_4-traffic-data-strategic-major-roads) | detector metadata CSV + raw speed/volume/occupancy XML |
| [TD Special Traffic News](https://data.gov.hk/en-data/dataset/hk-td-tis_1-special-traffic-news) | active incident notices |
| [TD Roadworks](https://data.gov.hk/en-data/dataset/hk-td-tis_4-roadworks) | planned roadworks GeoJSON |
| [TD CCTV](https://data.gov.hk/en-data/dataset/hk-td-tis_12-cctv-snapshots) | two-minute camera snapshots (K627F, AID07117) |
| [OpenStreetMap tiles](https://www.openstreetmap.org/) | traffic-map background (attribution required) |
| [HKUST shuttle schedule](https://cso.ust.hk/tran/stud_sh_b) | official schedule link (not scraped) |
| [HKUST bus-stop live view](http://liveview.ust.hk/busstop/) | official camera page link |
| [HKeMobility](https://www.hkemobility.gov.hk/) | interactive territory-wide traffic map |

The HKUST Open Data keyed endpoints (bus-queue-data, people-count-pulse, SSC)
are **not used**: as of 2026-08 their indexes contain no data newer than
2023–2025, so any values would be misleading. If HKUST revives the datasets,
re-add them with freshness checks.

We do **not** scrape RTHK traffic pages (undocumented, restrictive reuse) or
reproduce Google Maps imagery.

## Installation

Requires **Python 3.11+**.

```bash
git clone https://github.com/hkust-fys/hkust-data-dashboard.git
cd hkust-data-dashboard
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Configuration

Create `.env` in the project root (see `.env.example` for a template) and fill
in:

| Variable | Required | Description |
|---|---|---|
| `DISCORD_TOKEN` | production | Bot token from the [Discord Developer Portal](https://discord.com/developers/applications) |
| `ANNOUNCE_CHANNEL_ID` | production | ID of the channel that hosts the dashboard |
| `DASHBOARD_MESSAGE_ID` | optional | Existing dashboard message to reuse; when unset the bot scans recent history for its marker and creates one if none exists |
| `DEV_WEBHOOK` | dev | Discord webhook URL for the one-shot dev mode |
| `ALERT_ROLE_ID` | optional | Role pinged in the status thread on critical alerts |
| `UPDATE_INTERVAL_SECONDS` | optional | Dashboard edit cadence (default 30, minimum 10) |
| `HTTP_TIMEOUT_SECONDS` | optional | Per-request timeout (default 10) |
| `LOG_LEVEL` | optional | `DEBUG`/`INFO`/`WARNING`/`ERROR` (default `INFO`) |
| `CACHE_DIR` | optional | Directory for the OSM tile cache (default `.cache`) |

> The token is private — never commit `.env`.

Discord permissions the bot needs in the target channel:
`View Channel`, `Send Messages`, `Embed Links`, `Attach Files`,
`Read Message History`.

## Running

**Production** (one persistent dashboard message):

```bash
python bot.py
```

On first start the bot creates its dashboard message and prints its ID; it then
edits that same message on every cycle. To pin a specific message later, set
`DASHBOARD_MESSAGE_ID` to that ID.

**Development webhook** (one-shot send to a debug webhook):

```bash
DEV_WEBHOOK=https://discord.com/api/webhooks/... python bot.py --dev-webhook
```

**Dry run** (no Discord at all; writes a text preview plus the map PNG under
`.private/`):

```bash
python bot.py --dry-run --no-keys
```

## Refresh cadence

| Source | Cadence |
|---|---|
| KMB/Citybus/GMB ETAs | ~30 s |
| TD detector observations | ~60 s |
| TD CCTV | ~120 s |
| HKO warning summary | ~60 s |
| HKO observations | ~10 min |
| TD Special Traffic News | ~5 min |
| TD roadworks | ~15 min |
| TD detector metadata | daily |
| OSM tile cache | 7 days (persistent, bounded) |

The dashboard shows each provider's own source timestamp; it never claims a
faster cadence than the source actually provides. No long-term raw API history
is stored — only the latest replaceable assets and the bounded tile cache.

## Testing

Fixture-only tests run with no network and no keys:

```bash
ruff check .
python -m compileall -q bot.py dashboard tests
python -m pytest -m "not live"
```

Opt-in live smoke tests (hit public HKO/TD/transit endpoints):

```bash
python -m pytest -m live
```

CI (`.github/workflows/tests.yml`) runs ruff, compileall, and fixture-only
pytest on Python 3.11/3.12.

## Troubleshooting

- **A section shows `Unavailable` or `Stale — last success …`**: that provider
  failed and the last good value is being kept. The bot keeps running and
  retries on the next cycle.
- **Traffic shows gray / no coverage**: no fresh TD detector observation for
  the matched corridors at that moment. This is expected — detectors only cover
  monitored points, not entire roads.
- **No traffic-map image**: OSM tiles could not be fetched; a neutral-background
  map is generated instead so the dashboard never fails.
- **Message deleted**: the bot recreates its dashboard message on the next
  cycle (one message only).
- **`HTTP 403` from data.gov.hk resources**: use a normal browser-like
  `User-Agent`; the dashboard already sends one.

## Deferred features

- HKUST Open Data gate queues, people counts, and SSC datasets (indexes stopped
  updating 2023–2025; re-add with freshness checks if revived)
- HKUST live sensor data (CO₂/temperature/humidity — not yet available)
- Restaurant schedules / menus
- Historical analytics and queue prediction
- Full shuttle schedule inside the main dashboard (official link instead)
