"""Live frame-by-frame source-to-marker timing diagnostic."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.config import Settings  # noqa: E402
from dashboard.http import HttpClient  # noqa: E402
from dashboard.maps import _authoritative_etas, _destination_map  # noqa: E402
from dashboard.maps.marker_audit import (  # noqa: E402
    _match,
    audit_marker_positions,
)
from dashboard.maps.positions import estimate_bus_positions  # noqa: E402
from dashboard.providers.route_geometry import (  # noqa: E402
    fetch_route_geometry,
    select_probe_stops,
)
from dashboard.providers.transit import (  # noqa: E402
    fetch_probe_etas,
    fetch_transit_etas,
)


def _issue_summary(frame: int, issue: dict) -> str:
    detail = issue.get("detail") or {}
    match = detail.get("match") or {}
    checkpoint = detail.get("checkpoint", detail.get("gate_index", "-"))
    reason = detail.get("reason", "one-to-one mismatch")
    return (
        f"ISSUE frame={frame} route={'/'.join(issue['key'])} "
        f"kind={issue['kind']} checkpoint={checkpoint} reason={reason} "
        f"unmatched_source={match.get('unmatched_source_values', [])} "
        f"source_tokens={match.get('unmatched_source_observations', [])} "
        f"unmatched_marker={match.get('unmatched_marker_values', [])} "
        f"marker_id={detail.get('marker_id', '-')} "
        f"position={detail.get('position', '-')} "
        f"marker_sources={detail.get('source_observations', [])} "
        f"gate_rows={detail.get('gate_rows', [])} "
        f"checkpoint_rows={detail.get('checkpoint_rows', [])} "
        f"route_markers={detail.get('route_markers', [])}"
    )


def _route_key(value: object) -> tuple[str, str, str]:
    operator = str(getattr(value, "operator", ""))
    if operator.startswith("Operator."):
        operator = {
            "Operator.KMB": "KMB",
            "Operator.CITYBUS": "CTB",
            "Operator.GMB": "GMB",
        }.get(operator, operator)
    return (
        operator,
        str(getattr(value, "route", "")),
        str(getattr(value, "bound", "") or ""),
    )


def _watched_state(
    key: tuple[str, str, str],
    probe_etas: list[object],
    authoritative: list[object],
    estimates: list[object],
) -> tuple[object, ...]:
    """Return a deterministic, compact source/ownership snapshot for one route."""
    sources = {"probe": probe_etas, "gate": authoritative}
    rows = tuple(
        (
            kind,
            index,
            int(row.index),
            round(float(row.minutes), 2),
            str(getattr(getattr(row, "kind", ""), "value", getattr(row, "kind", ""))),
        )
        for kind, source in sources.items()
        for index, row in enumerate(source)
        if getattr(row, "minutes", None) is not None and _route_key(row) == key
    )
    tracks = []
    for marker_id, marker in enumerate(estimates):
        if _route_key(marker) != key:
            continue
        owned = []
        for token in sorted(getattr(marker, "source_observations", ())):
            kind, index = token
            source = sources.get(kind, ())
            if not 0 <= index < len(source):
                owned.append((kind, index, "missing"))
                continue
            row = source[index]
            owned.append(
                (
                    kind,
                    index,
                    int(row.index),
                    round(float(row.minutes), 2),
                    round(int(row.index) - float(row.minutes) / 2.0, 3),
                )
            )
        tracks.append(
            (
                marker_id,
                round(float(marker.position), 3),
                bool(marker.unreliable),
                tuple(owned),
            )
        )
    return rows, tuple(tracks)


async def _run(
    frames: int,
    interval: float = 10.0,
    watch_routes: tuple[tuple[str, str, str], ...] = (),
) -> int:
    settings = Settings.from_env(require_keys=False)
    async with aiohttp.ClientSession() as session:
        client = HttpClient(session, timeout_seconds=settings.http_timeout_seconds)
        try:
            geometry = await fetch_route_geometry(
                client,
                cache_dir=settings.cache_dir,
            )
            lines = list(geometry.routes)
            probes = select_probe_stops(lines)
            failed_routes: set[str] = set()
            provider_failures: set[str] = set()
            conclusive_checks = 0
            total_checks = 0
            watched_previous: dict[tuple[str, str, str], tuple[object, ...]] = {}
            for frame_index in range(frames):
                groups, _latest, failed = await fetch_transit_etas(client)
                provider_failures.update(failed or [])
                probe_etas = await fetch_probe_etas(client, probes)
                destinations = _destination_map(groups, lines)
                authoritative = _authoritative_etas(groups, lines)
                estimates = estimate_bus_positions(
                    probe_etas,
                    lines,
                    destinations,
                    authoritative,
                )
                audit = audit_marker_positions(
                    probe_etas,
                    authoritative,
                    estimates,
                    lines,
                    frame_id=frame_index + 1,
                    seed=frame_index + 1,
                )
                total_checks += len(audit["checks"])
                inconclusive = sum(
                    bool(check.get("inconclusive")) for check in audit["checks"]
                )
                conclusive_checks += len(audit["checks"]) - inconclusive
                kinds = sorted({str(check.get("kind")) for check in audit["checks"]})
                print(
                    f"FRAME {frame_index + 1}/{frames} "
                    f"checks={len(audit['checks'])} "
                    f"kinds={','.join(kinds) or 'none'} "
                    f"inconclusive={inconclusive} markers={len(estimates)} "
                    f"observed_checkpoints={audit['stats'].get('observed_checkpoints', 0)} "
                    f"audited_checkpoints={audit['stats'].get('audited_checkpoints', 0)} "
                    f"uncovered_checkpoints={audit['stats'].get('uncovered_checkpoints', 0)} "
                    f"observed_rows={audit['stats'].get('observed_probe_rows', 0)} "
                    f"audited_rows={audit['stats'].get('audited_probe_rows', 0)} "
                    f"uncovered_rows={audit['stats'].get('uncovered_probe_rows', 0)} "
                    f"issues={len(audit['issues'])} "
                    f"status={'PASS' if audit['ok'] else 'FAIL'}"
                )
                for issue in audit["issues"]:
                    print(_issue_summary(frame_index + 1, issue))
                    failed_routes.add("/".join(issue["key"]))
                for key in watch_routes:
                    watched = _watched_state(
                        key, probe_etas, authoritative, estimates
                    )
                    if watched_previous.get(key) == watched:
                        continue
                    watched_previous[key] = watched
                    rows, tracks = watched
                    print(
                        f"WATCH frame={frame_index + 1} route={'/'.join(key)} "
                        f"rows={rows} tracks={tracks}"
                    )
                if frame_index + 1 < frames:
                    await asyncio.sleep(interval)
        except Exception as exc:  # noqa: BLE001
            print(f"STATUS INCONCLUSIVE reason={type(exc).__name__}")
            return 2

    if provider_failures or conclusive_checks == 0:
        reason = "upstream-failures" if provider_failures else "no-conclusive-checks"
        print(f"STATUS INCONCLUSIVE reason={reason}")
        return 2
    if failed_routes:
        print("STATUS FAIL routes=" + ",".join(sorted(failed_routes)))
        return 1
    print(f"STATUS PASS frames={frames} checks={total_checks}")
    return 0


def _self_test() -> int:
    matched = _match([20, 3], [3, 20], tolerance=0)
    assert matched["pair_indices"] == [(1, 0), (0, 1)]
    assert matched["cardinality"] == 2
    print("STATUS PASS self-test=identity-preserving-one-to-one")
    return 0


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", "--frames", dest="frames", type=int, default=4)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument(
        "--watch-route",
        action="append",
        default=[],
        metavar="OPERATOR/ROUTE/BOUND",
        help="print changed source rows and marker ownership for one route",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    if args.frames < 1:
        parser.error("--frames must be >= 1")
    if args.interval < 0:
        parser.error("--interval must be >= 0")
    watch_routes = []
    for value in args.watch_route:
        parts = tuple(part.strip() for part in value.split("/"))
        if len(parts) != 3 or not all(parts):
            parser.error("--watch-route must be OPERATOR/ROUTE/BOUND")
        watch_routes.append(parts)
    return asyncio.run(_run(args.frames, args.interval, tuple(watch_routes)))


if __name__ == "__main__":
    raise SystemExit(main())
