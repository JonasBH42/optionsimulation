"""Isolated real-archive skip timings; source archives and user sessions are read only.

Examples:
  python tools/benchmark_fast_skips.py --minutes 5 --scenarios empty --cache warm
  python tools/benchmark_fast_skips.py --minutes 5 60 360 --next-session

Cold means the application's data cache is empty immediately before the command;
the operating system's file cache is not flushed. Warm means the destination
snapshot and replay indexes were loaded before timing. Startup, source
hashing, scenario preparation, and prewarming are recorded outside command time.
Ready means the normal newly opened session's caches and replay indexes are
retained, and its destination snapshot has never been requested.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_CEILING
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_steps import session_data
from qqq_sim.broker.models import OrderDraft, OrderLeg
from qqq_sim.config import AppConfig, configure_environment, writable_path
from qqq_sim.data.catalog import file_hash
from qqq_sim.data.models import CatalogReport, NormalizationMapping
from qqq_sim.replay.calendar import local_datetime
from qqq_sim.replay.controller import ReplayController
from qqq_sim.serialization import primitive
from qqq_sim.services.adapters import broker_contract


def selected_data(config, days, mapping):
    data = session_data(config, days[0])
    partitions = list(data.catalog.report.partitions)
    for day in days[1:]:
        partitions.extend(session_data(config, day).catalog.report.partitions)
    report = CatalogReport(tuple(partitions), (), (config.underlying_root, config.options_root))
    catalog = data.catalog
    catalog.report = report
    catalog._index = {item.key: item for item in partitions}
    paths = {key: tuple(item.path for item in partitions if item.key == key) for key in catalog._index}
    catalog._duplicate_paths = {key: value for key, value in paths.items() if len(value) > 1}
    catalog._duplicates = set(catalog._duplicate_paths)
    data.mappings = {"options": mapping}
    return data


def prepare_order(controller, scenario):
    if scenario == "empty":
        return
    candidates = [entry for entry in controller.snapshot.quotes
                  if entry.status in ("OK", "CURRENT") and entry.latest_event.valid
                  and entry.contract.expiration > controller.decision_at.date()
                  and entry.latest_event.ask is not None and Decimal(0) < entry.latest_event.ask <= Decimal(5)]
    if not candidates:
        raise RuntimeError("No fresh inexpensive non-expiring option is available for this benchmark scenario")
    # A one-contract long avoids naked-short assumptions and stock assignment.
    entry = min(candidates, key=lambda value: (value.latest_event.ask, value.contract))
    contract = broker_contract(entry.contract, controller.data.calendar)
    close = controller.data.calendar.session(contract.expiry).underlying_close
    debit = (entry.latest_event.ask * 100 + 100).quantize(Decimal(1), rounding=ROUND_CEILING)
    limit = -debit if scenario == "held" else Decimal(0)
    draft = OrderDraft("benchmark", (OrderLeg(contract, 1),), limit,
                       close - timedelta(minutes=1), close, tif="GTC")
    controller.submit(draft, controller.version, "benchmark-" + scenario)


def run_case(config, days, mapping, at, destination, scenario, cache_mode):
    prepared_at = perf_counter()
    data = selected_data(config, days, mapping)
    start = at - timedelta(minutes=1) if scenario == "held" else at
    with patch("qqq_sim.cli.make_data", return_value=data), patch.object(data.catalog, "scan", return_value=data.catalog.report):
        controller = ReplayController.create(config, start)
    try:
        prepare_order(controller, scenario)
        if scenario == "held":
            controller.advance(target=at)
            if not controller.broker.ledger.positions():
                raise RuntimeError("Held benchmark setup did not obtain an eligible later fill")
        setup_seconds = perf_counter() - prepared_at
        prewarm_seconds = 0.
        if cache_mode == "cold":
            data.cache.clear()
        elif cache_mode == "warm":
            warming = perf_counter()
            data.snapshot(destination)
            # Build bounded source indexes without materializing a full day's
            # normalized quote history or changing the account.
            data.prepare_replay_window(at, destination)
            prewarm_seconds = perf_counter() - warming
        counts = {"snapshots": 0, "event_windows": 0, "commits": 0}
        operation_seconds = {key: 0. for key in counts}
        snapshot, events, commit = data.snapshot, data.events_between, controller._commit
        def counted_snapshot(*args, **kwargs):
            counts["snapshots"] += 1
            started = perf_counter()
            try:
                return snapshot(*args, **kwargs)
            finally:
                operation_seconds["snapshots"] += perf_counter() - started
        def counted_events(*args, **kwargs):
            counts["event_windows"] += 1
            started = perf_counter()
            try:
                return events(*args, **kwargs)
            finally:
                operation_seconds["event_windows"] += perf_counter() - started
        def counted_commit(*args, **kwargs):
            counts["commits"] += 1
            started = perf_counter()
            try:
                return commit(*args, **kwargs)
            finally:
                operation_seconds["commits"] += perf_counter() - started
        with patch.object(data, "snapshot", counted_snapshot), patch.object(data, "events_between", counted_events), patch.object(controller, "_commit", counted_commit):
            started = perf_counter()
            result = controller.advance(target=destination, command_id="measured-skip")
            seconds = perf_counter() - started
        record = {"scenario": scenario, "cache": cache_mode, "start": at.isoformat(),
                  "requested": destination.isoformat(), "actual": result.decision_at.isoformat(),
                  "seconds": seconds, "setup_seconds": setup_seconds, "prewarm_seconds": prewarm_seconds,
                  "status": result.status, "reason": result.reason, "contracts": len(result.snapshot.quotes),
                  "snapshot_id": result.snapshot.snapshot_id, "data_cache_bytes": data.cache.bytes,
                  "cache_budget_bytes": data.cache.max_bytes, "operations": counts,
                  "operation_seconds": operation_seconds,
                  "positions": len(result.positions), "order_statuses": [order["status"] for order in result.orders],
                  "implementation_identity_hash": controller._implementation_identity["identity_hash"],
                  "session_path": str(controller.store.path)}
        saved = controller.store.restore(controller.branch_id)
        record["durable_state_matches"] = (saved["state"]["decision_at"] == result.decision_at.isoformat()
                                           and saved["state"]["broker"] == controller.broker.to_dict())
        return primitive(record)
    finally:
        controller.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--at", default="2023-06-05T10:00:00-04:00")
    parser.add_argument("--minutes", type=int, nargs="*", default=[5, 60, 360])
    parser.add_argument("--next-session", action="store_true")
    parser.add_argument("--scenarios", nargs="+", choices=("empty", "held", "active"), default=["empty", "held", "active"])
    parser.add_argument("--cache", nargs="+", choices=("cold", "ready", "warm"), default=["cold", "ready", "warm"])
    parser.add_argument("--output", default="runtime/results/fast-skip-performance.json")
    parser.add_argument("--background-workload", default=None, help="Describe concurrent work that may affect wall-clock timings")
    args = parser.parse_args()
    if any(minutes <= 0 for minutes in args.minutes):
        parser.error("--minutes must be positive")
    if not args.minutes and not args.next_session:
        parser.error("Select trading minutes or --next-session")
    configure_environment()
    root = writable_path("runtime/results/fast-skip-bench-" + uuid4().hex)
    root.mkdir(parents=True)
    at = local_datetime(datetime.fromisoformat(args.at))
    base_config = replace(AppConfig(), runtime_root=str(root))
    # Trading benchmark scenarios explicitly assume standard QQQ metadata. This
    # is local benchmark configuration, never a change to archive verification.
    mapping_record = {"metadata_verified": True, "confidence": "EXPLICIT_SYNTHETIC_SCENARIO"}
    mapping_path = root / "normalization.json"
    mapping_path.write_text(json.dumps(mapping_record), encoding="utf-8")
    config = replace(base_config, normalization_path=str(mapping_path))
    mapping = NormalizationMapping(**mapping_record)
    seed = selected_data(config, [at.date()], mapping)
    targets = [(str(minutes) + " trading minutes", seed.calendar.advance_trading_minutes(at, minutes))
               for minutes in args.minutes]
    if args.next_session:
        targets.append(("next session", seed.calendar.next_session(at.date()).exchange_open))
    days = []
    day, final_day = at.date(), max(target.date() for _, target in targets)
    while day <= final_day:
        if seed.calendar.session(day).status == "OPEN":
            days.append(day)
        day += timedelta(days=1)
    sources = selected_data(config, days, mapping).catalog.report.partitions
    hashes = {item.path: file_hash(item.path) for item in sources}
    records = []
    output = writable_path(args.output, create_parent=True)
    report = {"scope": "Isolated real-source replay commands including broker and persistence, excluding startup and UI",
              "cold_definition": "Application cache cleared; OS cache unchanged",
              "background_workload": args.background_workload,
              "scenario_assumptions": mapping_record, "at": at.isoformat(), "records": records,
              "benchmark_root": str(root), "source_hashes": hashes, "source_hashes_unchanged": None}
    for label, destination in targets:
        for scenario in args.scenarios:
            for cache_mode in args.cache:
                record = {"jump": label, **run_case(config, days, mapping, at, destination, scenario, cache_mode)}
                records.append(record)
                print(json.dumps(record), flush=True)
                output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["source_hashes_unchanged"] = all(file_hash(path) == expected for path, expected in hashes.items())
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["source_hashes_unchanged"] and all(record["status"] == "OK" and record["durable_state_matches"]
                                                        for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
