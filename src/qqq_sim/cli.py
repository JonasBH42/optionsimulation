"""GUI and explicit headless/offline workflows from one installed package."""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, date
import json
from pathlib import Path
import sys

from .config import AppConfig, configure_environment, writable_path
from .serialization import dumps, save_json


def instant(text):
    from .replay.calendar import local_datetime
    try:
        return local_datetime(datetime.fromisoformat(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parser():
    p = argparse.ArgumentParser(prog="python -m qqq_sim", description="Historical QQQ desktop simulator. No real orders. All outputs stay inside D:\\optionsimulation.")
    p.add_argument("--config", type=Path, help="Versioned JSON configuration; defaults to project archive roots and a $5,000 simulated account")
    p.add_argument("--demo", action="store_true", help="Explicitly select the synthetic fixture; never a historical-data fallback")
    p.add_argument("--at", type=instant, help="Initial NY decision instant; naive ISO times mean New York with DST validation")
    commands = p.add_subparsers(dest="command")
    audit = commands.add_parser("audit-data", help="Read-only archive/schema/coverage audit; does not build a model")
    audit.add_argument("--output", default="runtime/results/data-audit.json")
    audit.add_argument("--sample", type=int, default=3, help="Number of sample partition headers to inspect per dataset type")
    snap = commands.add_parser("snapshot", help="Export a causal completed-minute snapshot (raw event cutoff is exclusive)")
    snap.add_argument("--at", type=instant, required=True)
    snap.add_argument("--expiry", type=date.fromisoformat, action="append")
    snap.add_argument("--output", default="runtime/results/snapshot.json")
    replay = commands.add_parser("replay", help="Run persisted deterministic headless replay; every intervening minute/event is processed")
    replay.add_argument("--at", type=instant, required=True)
    replay.add_argument("--until", type=instant)
    replay.add_argument("--minutes", type=int, default=1)
    replay.add_argument("--script", type=Path, help="JSON commands: advance/submit/cancel/branch/export with explicit IDs")
    replay.add_argument("--session", help="Restore an existing local session ID")
    replay.add_argument("--output", default="runtime/results/replay.json")
    build = commands.add_parser("build-model", help="Explicit resumable offline build; never entered from normal analysis")
    build.add_argument("--build-config", type=Path, help="JSON build design, paths, budgets, seeds and numerical settings")
    build.add_argument("--output", help="Project-local pack directory; overrides build-config output_dir")
    build.add_argument("--fixture", action="store_true", help="Small nonproduction pack for plumbing validation; cannot pass production qualification")
    validation = commands.add_parser("validate-pack", help="Validate actual model pack bytes and report qualification gates")
    validation.add_argument("path", type=Path)
    validation.add_argument("--output", default="runtime/results/pack-validation.json")
    export = commands.add_parser("export", help="Export an existing frozen session; no new market calculations")
    export.add_argument("session", help="Session ID or project-local SQLite path")
    export.add_argument("--branch")
    export.add_argument("--output", default="runtime/results/session-export.json")
    saved = commands.add_parser("export-saved",help="Read-only forensic export of a saved session without loading current replay/account semantics")
    saved.add_argument("--session",required=True,help="Session ID or saved SQLite path")
    saved.add_argument("--branch",help="Saved branch ID; defaults to the stored active branch")
    saved.add_argument("--output",default="runtime/results/saved-session-export.json")
    oracle = commands.add_parser("reference", help="Explicit offline reference P50 calculation from a frozen JSON job")
    oracle.add_argument("job", type=Path)
    oracle.add_argument("--output", default="runtime/results/reference.json")
    oracle.add_argument("--checkpoint", help="Project-local resumable checkpoint; default is named by the immutable job hash")
    fast = commands.add_parser("fast-query", help="Evaluate a frozen job or prepared campaign request through the same spawned analysis worker as the desktop")
    fast.add_argument("request", type=Path)
    fast.add_argument("--pack", required=True, type=Path)
    fast.add_argument("--output", default="runtime/results/fast-query.json")
    commands.add_parser("p50-reference", aliases=["reference-wizard"], help="Three-stage terminal reference-input workflow with back navigation; use global --at for the historical snapshot")
    demo = commands.add_parser("create-fixture", help="Create labeled synthetic data for tests and demonstration")
    demo.add_argument("--output", default="runtime/fixtures/demo")
    commands.add_parser("doctor", help="Inspect interpreter, dependencies, paths and model availability")
    smoke=commands.add_parser("desktop-smoke",help="Test synthetic GUI open/close, pack loading, restore/export and rendering")
    smoke.add_argument("--output",default="runtime/results/desktop-smoke")
    smoke.add_argument("--pack",default="runtime/model_packs/validation_fixture",help="Existing tiny validation pack; no implicit build")
    return p


def make_data(config):
    from .data.asof import HistoricalData
    from .services.adapters import load_mapping
    from .services.calendars import load_calendar
    return HistoricalData(config.underlying_root, config.options_root, mappings={"options": load_mapping(config.normalization_path)},
                          calendar=load_calendar(config.calendar_path),cache_bytes=config.cache_bytes,
                          catalog_path=writable_path(Path(config.runtime_root)/"catalog/metadata.json"))


def demo_config(config, root="runtime/fixtures/demo"):
    from .data.fixtures import write_synthetic_dataset
    target = writable_path(root)
    existing = target / "SYNTHETIC_FIXTURE.json"
    roots = json.loads(existing.read_text(encoding="utf-8")) if existing.exists() else write_synthetic_dataset(target)
    if isinstance(roots, dict):
        underlying, options = roots["underlying_root"], roots["options_root"]
    else:
        underlying, options = roots
    mapping = save_json(target / "normalization.json", {"strike_divisor": "1", "iv_unit": "decimal", "metadata_verified": True,
                                                       "timestamp_precision": "subminute", "confidence": "EXPLICIT_SYNTHETIC_SCENARIO"})
    parameters = save_json(target / "parameters.json", {
        "rate_cc": 0.04, "dividend_yield_cc": 0.01, "sigma_path": 0.25,
        "known_at": "2026-11-09T09:30:00-05:00", "fixing_at": roots.get("expiry", "2026-11-13")+"T16:00:00-05:00",
        "source": "EXPLICIT_SYNTHETIC_SCENARIO"})
    return replace(config, underlying_root=str(underlying), options_root=str(options), normalization_path=str(mapping), parameters_path=str(parameters))


def error_record(exc):
    if hasattr(exc, "as_dict"):
        return exc.as_dict()
    if hasattr(exc, "to_dict"):
        return exc.to_dict()
    return {"status": "FAILED", "reason": getattr(exc, "code", type(exc).__name__), "message": str(exc)}


def main(argv=None):
    configure_environment()
    args = parser().parse_args(argv)
    try:
        config = AppConfig.load(args.config)
        if args.demo:
            config = demo_config(config)
        if args.command is None:
            from .app import launch
            return launch(config, args.at, synthetic=args.demo)
        if args.command == "doctor":
            from .services.startup import startup_report
            report=startup_report(config)
            print(dumps(report, pretty=True))
            return 0 if report["runtime_ready"] else 2
        if args.command == "desktop-smoke":
            from .ui.smoke import smoke
            result=smoke(args.output,args.pack)
            print(dumps(result,pretty=True))
            return 0 if result["status"]=="OK" else 2
        if args.command == "create-fixture":
            fixture = demo_config(config, args.output)
            path = fixture.save(writable_path(args.output) / "config.json")
            print(dumps({"status": "OK", "synthetic": True, "config": str(path)}))
            return 0
        if args.command == "audit-data":
            from .data.catalog import CatalogService
            catalog = CatalogService(config.underlying_root, config.options_root)
            report = catalog.scan()
            output = report.to_dict()
            output["source_access"] = "READ_ONLY"
            # Schema reads are bounded; full source hashes are bound lazily at actual use.
            samples, counts = [], {}
            for partition in report.partitions:
                kind = partition.key.data_type
                counts[kind] = counts.get(kind, 0)+1
                if sum(x["kind"] == kind for x in samples) >= max(0,args.sample):
                    continue
                try:
                    if partition.path.endswith(".parquet"):
                        import pyarrow.parquet as pq
                        meta = pq.ParquetFile(partition.path)
                        header, rows = str(meta.schema_arrow), meta.metadata.num_rows
                    else:
                        import gzip, csv
                        with gzip.open(partition.path, "rt", encoding="utf-8-sig", newline="") as f:
                            reader = csv.reader(f)
                            header, rows = next(reader), None
                    samples.append({"kind": kind, "path": partition.path, "schema": header, "rows": rows})
                except Exception as exc:
                    samples.append({"kind": kind, "path": partition.path, "error": error_record(exc)})
            output["samples"], output["partition_counts"] = samples, counts
            path = save_json(args.output, output)
            from collections import Counter
            print(dumps({"status": "OK" if report.ok else "PARTIAL", "partition_counts": counts,
                         "diagnostic_counts":dict(Counter(x.code for x in report.diagnostics)),
                         "diagnostic_sample":report.diagnostics[:10], "output": str(path)}, pretty=True))
            return 0 if report.ok else 2
        if args.command == "snapshot":
            data = make_data(config)
            snapshot = data.snapshot(args.at, expirations=args.expiry)
            path = save_json(args.output, snapshot)
            status="PARTIAL" if snapshot.spot is None or any(x.severity=="ERROR" for x in snapshot.diagnostics) else "OK"
            print(dumps({"status": status, "snapshot_id": snapshot.snapshot_id, "spot": snapshot.spot,
                         "contracts": len(snapshot.quotes), "diagnostics":snapshot.diagnostics,"output": str(path)}))
            return 0 if status=="OK" else 2
        if args.command == "replay":
            from .replay.controller import ReplayController
            controller = ReplayController.restore(config, args.session) if args.session else ReplayController.create(config, args.at)
            try:
                if args.script:
                    commands = json.loads(args.script.read_text(encoding="utf-8-sig"))
                    for command in commands:
                        controller.dispatch(command)
                else:
                    controller.advance(target=args.until, minutes=args.minutes, expected_version=controller.version)
                result = controller.export_record()
                path = save_json(args.output, result)
                print(dumps({"status": result["status"], "session_id": controller.session_id, "decision_at": controller.decision_at,
                             "account": result["account"], "output": str(path)}, pretty=True))
                return 0 if result["status"] == "OK" else 2
            finally:
                controller.close()
        if args.command == "build-model":
            from .risk.offline.build import build_model
            values = json.loads(args.build_config.read_text(encoding="utf-8-sig")) if args.build_config else {}
            values["output_dir"] = str(writable_path(args.output or values.get("output_dir","runtime/model_packs/qqq-production"), create_parent=True))
            if args.fixture:
                values["fixture"] = True
            result = build_model(values)
            print(dumps(result, pretty=True))
            return 0 if result.get("status") in ("OK", "QUALIFIED", "UNQUALIFIED", "COMPLETED", "VALIDATION_FIXTURE_BUILT", "RESEARCH_COMPLETE_UNQUALIFIED") else 2
        if args.command == "validate-pack":
            from .risk.offline.validate import validate_pack
            result = validate_pack(args.path)
            save_json(args.output, result)
            print(dumps(result, pretty=True))
            return 0 if result.get("status") in ("OK","QUALIFIED") else 2
        if args.command == "reference":
            from .risk.contracts import ReferenceJob
            from .risk.offline.hit_oracle import calculate_reference
            job=ReferenceJob.from_dict(json.loads(args.job.read_text(encoding="utf-8-sig")))
            checkpoint=writable_path(args.checkpoint or f"runtime/builds/reference/{job.hash}.json",create_parent=True)
            result = calculate_reference(job,checkpoint=checkpoint)
            save_json(args.output, result)
            print(dumps(result, pretty=True))
            return 0 if result.get("status") == "OK" else 2
        if args.command == "fast-query":
            from .risk.contracts import ReferenceJob, stable_hash
            import hashlib
            from .services.analysis import AnalysisCoordinator, AnalysisIdentity
            source = json.loads(args.request.read_text(encoding="utf-8-sig"))
            request = source if "job" in source else {"job": source}
            job = ReferenceJob.from_dict(request["job"])
            pack_path = args.pack.resolve()
            manifest = json.loads((pack_path / "manifest.json").read_text(encoding="utf-8-sig"))
            identity = AnalysisIdentity("HEADLESS_FROZEN_QUERY", "FROZEN", 0, job.hash,
                                        stable_hash(request), stable_hash(request.get("campaign")),
                                        stable_hash(request.get("proposal")), hashlib.sha256((pack_path / "manifest.json").read_bytes()).hexdigest(),
                                        stable_hash(request.get("limits")))
            coordinator = AnalysisCoordinator()
            try:
                returned_identity, result = coordinator.submit(identity, request["job"], str(pack_path),
                    request.get("campaign"), request.get("universe", ()), request.get("limits"), request.get("proposal")).result()
                path = save_json(args.output, {"identity": returned_identity, "frozen_request": request, "result": result})
                print(dumps({"status": result.get("status"), "reason": result.get("reason"),
                             "identity": returned_identity, "output": str(path)}, pretty=True))
                return 0 if result.get("status") == "OK" else 2
            finally:
                coordinator.close()
        if args.command in ("p50-reference", "reference-wizard"):
            from .services.reference_wizard import run_wizard
            return run_wizard(config, args.at)
        if args.command == "export-saved":
            from .services.saved_session_export import export_saved_session
            source=Path(args.session)
            if source.suffix!=".sqlite":source=Path(config.runtime_root)/"sessions"/args.session/"session.sqlite"
            path=export_saved_session(source,args.output,branch_id=args.branch)
            print(dumps({"status":"OK","output":str(path),"mode":"READ_ONLY_SAVED_STATE_EXPORT"}))
            return 0
        if args.command == "export":
            from .replay.controller import ReplayController
            controller = ReplayController.restore(config, args.session, args.branch)
            try:
                path = controller.export(args.output)
                print(dumps({"status": "OK", "output": str(path)}))
                return 0
            finally:
                controller.close()
    except KeyboardInterrupt:
        print(dumps({"status": "CANCELED", "reason": "USER_CANCELED"}))
        return 130
    except Exception as exc:
        print(dumps(error_record(exc), pretty=True), file=sys.stderr)
        return 2
    return 0
