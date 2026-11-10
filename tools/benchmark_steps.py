"""Read-only, single-session timing of chain refreshes and replay quote windows."""
from datetime import date, datetime, timedelta
from pathlib import Path
from time import perf_counter
import argparse
import cProfile
import json
import sys
from dataclasses import replace
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qqq_sim.config import AppConfig, configure_environment, writable_path
from qqq_sim.data.asof import HistoricalData
from qqq_sim.data.catalog import file_hash
from qqq_sim.data.models import CatalogReport, Partition, PartitionKey
from qqq_sim.replay.calendar import local_datetime


def session_data(config, day):
    """Select one day's source files so a whole-archive scan isn't timed."""
    data = HistoricalData(config.underlying_root, config.options_root, cache_bytes=config.cache_bytes)
    partitions = []
    for kind in ("underlying", "greeks", "ohlcv", "oi_before", "trades"):
        folder = Path(config.underlying_root) if kind == "underlying" else Path(config.options_root) / kind
        paths = ([folder / f"QQQ_1m_{day}.parquet"] if kind == "underlying"
                 else sorted(folder.rglob(f"*{day:%Y%m%d}*.csv.gz")))
        for path in paths:
            expiry = None if kind == "underlying" else date.fromisoformat(path.name.split("exp=")[1][:10])
            stat = path.stat()
            partitions.append(Partition(PartitionKey(kind, day, expiry), str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    catalog = data.catalog
    catalog.report = CatalogReport(tuple(partitions), (), (config.underlying_root, config.options_root))
    catalog._index = {item.key: item for item in partitions}
    catalog._duplicate_paths = {key: tuple(item.path for item in partitions if item.key == key) for key in catalog._index}
    catalog._duplicate_paths = {key: paths for key, paths in catalog._duplicate_paths.items() if len(paths) > 1}
    catalog._duplicates = set(catalog._duplicate_paths)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--at", default="2023-06-05T10:01:00-04:00")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--replay", action="store_true", help="Also time persisted Step commands with the warmed session data")
    parser.add_argument("--output", default="runtime/results/step-performance.json")
    args = parser.parse_args()
    configure_environment()
    at = local_datetime(datetime.fromisoformat(args.at))
    config = AppConfig()
    data = session_data(config, at.date())
    before = {item.path: file_hash(item.path) for item in data.catalog.report.partitions}
    records = []
    profiler = cProfile.Profile() if args.profile else None
    for index in range(args.count):
        instant = at + timedelta(minutes=index)
        if profiler:
            profiler.enable()
        start = perf_counter()
        snapshot = data.snapshot(instant)
        snapshot_seconds = perf_counter() - start
        start = perf_counter()
        events = data.events_between(instant, instant + timedelta(minutes=1))
        event_seconds = perf_counter() - start
        if profiler:
            profiler.disable()
        record = {"at": instant.isoformat(), "snapshot_seconds": snapshot_seconds,
                  "event_window_seconds": event_seconds, "contracts": len(snapshot.quotes),
                  "events": len(events), "cache_bytes": data.cache.bytes,
                  "snapshot_id": snapshot.snapshot_id, "diagnostics": [item.code for item in snapshot.diagnostics]}
        records.append(record)
        print(json.dumps(record), flush=True)
    replay_records = []
    if args.replay:
        from qqq_sim.replay.controller import ReplayController
        config = replace(config, runtime_root="runtime/results/step-bench-" + uuid4().hex)
        # Reuse the selected real-session catalog. This excludes whole-archive
        # metadata enumeration from the Step measurement, as an open app does.
        with patch("qqq_sim.cli.make_data", return_value=data), patch.object(data.catalog, "scan", return_value=data.catalog.report):
            controller = ReplayController.create(config, at + timedelta(minutes=args.count - 1))
        try:
            for minutes in (1, 5):
                start = perf_counter()
                result = controller.advance(minutes=minutes)
                record = {"minutes": minutes, "seconds": perf_counter() - start,
                          "status": result.status, "decision_at": result.decision_at.isoformat(),
                          "reason": result.reason}
                replay_records.append(record)
                print(json.dumps({"replay": record}), flush=True)
        finally:
            controller.close()
    target = writable_path(args.output, create_parent=True)
    report = {"scope": "Single-session snapshots/event reads exclude startup, broker, persistence and UI; optional warm replay timings include broker/account work and persistence, excluding UI",
              "profiled": bool(profiler), "cache_budget_bytes": data.cache.max_bytes, "records": records,
              "replay_records": replay_records,
              "retained_cache_bytes": data.cache.bytes - data.cache._transient_bytes,
              "transient_cache_bytes": data.cache._transient_bytes,
              "source_hashes_unchanged": all(file_hash(path) == fingerprint for path, fingerprint in before.items())}
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if profiler:
        profiler.dump_stats(str(target.with_suffix(".prof")))


if __name__ == "__main__":
    main()
