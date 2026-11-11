"""Verify and assemble the scoped fast-skip standalone update.

The original distribution must already be backed up. Numerical qualification
reports are retained as explicitly historical evidence, not re-certified here.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import xml.etree.ElementTree as ET
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_source_manifest import current
from qqq_sim.broker.models import OrderDraft, OrderLeg
from qqq_sim.cli import demo_config, make_data
from qqq_sim.config import AppConfig, configure_environment, writable_path
from qqq_sim.data.fixtures import write_synthetic_dataset
from qqq_sim.replay.calendar import local_datetime
from qqq_sim.serialization import primitive
from qqq_sim.services.adapters import broker_contract


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = writable_path(path, create_parent=True)
    path.write_text(json.dumps(primitive(value), indent=2), encoding="utf-8")
    return path


def native_evidence(path, build_id, executable_sha256):
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("status") != "PASS" or record.get("build_id") != build_id
            or record.get("executable_sha256") != executable_sha256):
        raise RuntimeError("Native verification does not match this source and executable")
    return {"path": str(path), "sha256": sha(path), "result": record}


def performance_evidence():
    measured_path = ROOT / "runtime/results/fast-skip-benchmark-source-manifest.json"
    measured = json.loads(measured_path.read_text(encoding="utf-8"))
    build_id, sources, lock = current()
    changed = {key for key in set(sources) | set(measured["source_hashes"])
               if sources.get(key) != measured["source_hashes"].get(key)}
    compatibility_files = {"src/qqq_sim/services/_session_upgrade_policy.py", "src/qqq_sim/services/session_upgrade.py"}
    if changed - compatibility_files or lock != measured["lock_hash"]:
        raise AssertionError("Application behavior beyond the save compatibility policy changed after timing")
    paths = [ROOT / "runtime/results/fast-skip-performance-quiet.json",
             ROOT / "runtime/results/fast-skip-performance-trades-quiet.json"]
    return {"measured_build_id": measured["build_id"], "current_build_id": build_id,
            "changes_after_measurement": sorted(changed), "replay_data_broker_storage_sources_unchanged": True,
            "note": "Quiet timings were measured before the exact original-native save identity was added; the replay kernel is byte-identical.",
            "source_manifest": {"path": str(measured_path), "sha256": sha(measured_path)},
            "reports": [{"path": str(path), "sha256": sha(path)} for path in paths]}


def scenario_scripts(config, output):
    instant = local_datetime(datetime(2026, 11, 9, 10, 1))
    data = make_data(config)
    snapshot = data.snapshot(instant)
    keys = {(entry.contract.right, entry.contract.strike): broker_contract(entry.contract, data.calendar)
            for entry in snapshot.quotes}
    deadline = local_datetime(datetime(2026, 11, 13, 15, 30))
    opening = OrderDraft("release-check", (OrderLeg(keys["P", Decimal(600)], -1),
                                           OrderLeg(keys["P", Decimal(595)], 1)),
                         Decimal(50), deadline, deadline + timedelta(minutes=30))
    close = replace(opening, legs=tuple(OrderLeg(leg.contract, -leg.quantity) for leg in opening.legs),
                    limit_cash=Decimal(-500), intent="CLOSE")
    scripts = {}
    for runtime in ("source", "compiled"):
        script = [{"action": "submit", "command_id": "release-open", "draft": primitive(opening)},
                  {"action": "advance", "command_id": "release-held-skip", "minutes": 60},
                  {"action": "export", "output": str(output / f"{runtime}-held.json")},
                  {"action": "submit", "command_id": "release-close", "draft": primitive(close)},
                  {"action": "advance", "command_id": "release-close-skip", "minutes": 5}]
        scripts[runtime] = write_json(output / f"{runtime}-trade-script.json", script)
    scripts["working"] = write_json(output / "working-script.json", [
        {"action": "submit", "command_id": "release-working", "draft": primitive(replace(opening, limit_cash=Decimal(400), tif="GTC"))},
        {"action": "advance", "command_id": "release-working-skip", "minutes": 60}])
    return scripts


def original_native_cases(run, output, fixture_record_path, runtimes=("source", "compiled")):
    """Resume a real save made by the preserved original executable, untouched."""
    fixture = json.loads(fixture_record_path.read_text(encoding="utf-8"))
    original_path = writable_path(fixture["database"])
    original_hash = sha(original_path)
    if original_hash != fixture["database_sha256"] or sha(fixture["original_executable"]) != fixture["original_executable_sha256"]:
        raise AssertionError("Original native save or executable changed")
    original = json.loads(Path(fixture["export"]).read_text(encoding="utf-8"))
    case_root = output / ("original-native-cases-" + uuid4().hex)
    case_root.mkdir(parents=True)
    config = fixture["config"]
    reference_input = case_root / "old-native-reference.sqlite"
    shutil.copy2(original_path, reference_input)
    reference_after = case_root / "old-native-after-three-minutes.json"
    run("original_native_reference_resume", "original_native", ["--config", config, "replay",
        "--at", original["decision_at"], "--session", str(reference_input), "--minutes", "3",
        "--output", str(reference_after)])
    expected = json.loads(reference_after.read_text(encoding="utf-8"))
    pending_id = next(key for key, value in original["state"]["broker"]["orders"].items() if value["status"] == "WORKING")
    old_script = json.loads((Path(fixture["export"]).parent / "original-native-script.json").read_text(encoding="utf-8"))
    marketable_close = old_script[-1]["draft"]
    marketable_close["limit_cash"] = "-500"
    close_script = write_json(case_root / "close-after-upgrade.json", [
        {"action": "cancel", "order_id": pending_id, "command_id": "native-upgrade-cancel-pending"},
        {"action": "submit", "draft": marketable_close, "command_id": "native-upgrade-executable-close"},
        {"action": "advance", "minutes": 2, "command_id": "native-upgrade-close-fill"}])
    reference_closed = case_root / "old-native-closed.json"
    run("original_native_reference_close", "original_native", ["--config", config, "replay",
        "--at", expected["decision_at"], "--session", str(reference_input), "--script", str(close_script),
        "--output", str(reference_closed)])
    closed = json.loads(reference_closed.read_text(encoding="utf-8"))
    results = {}
    for runtime in runtimes:
        source = case_root / (runtime + "-legacy-input.sqlite")
        shutil.copy2(original_path, source)
        before_path = case_root / (runtime + "-before-resume.json")
        run(runtime + "_restores_original_native_save", runtime,
            ["--config", config, "export", str(source), "--output", str(before_path)])
        before = json.loads(before_path.read_text(encoding="utf-8"))
        if (before["decision_at"] != original["decision_at"] or before["account"] != original["account"]
                or before["state"]["broker"] != original["state"]["broker"]
                or before["implementation_identity"]["identity_hash"] == fixture["implementation_identity_hash"]):
            raise AssertionError(runtime + " changed the original native economic state while upgrading")
        after_path = case_root / (runtime + "-after-three-minutes.json")
        run(runtime + "_resumes_original_native_save", runtime, ["--config", config, "replay",
            "--at", original["decision_at"], "--session", str(source), "--minutes", "3", "--output", str(after_path)])
        after = json.loads(after_path.read_text(encoding="utf-8"))
        for field in ("decision_at", "account", "status"):
            if after[field] != expected[field]:
                raise AssertionError(runtime + " original-native continuation differs: " + field)
        if after["state"]["broker"] != expected["state"]["broker"]:
            raise AssertionError(runtime + " original-native broker continuation differs")
        closed_path = case_root / (runtime + "-closed.json")
        run(runtime + "_closes_original_native_trade", runtime, ["--config", config, "replay",
            "--at", after["decision_at"], "--session", str(source), "--script", str(close_script), "--output", str(closed_path)])
        actual_closed = json.loads(closed_path.read_text(encoding="utf-8"))
        if (actual_closed["account"] != closed["account"]
                or actual_closed["state"]["broker"] != closed["state"]["broker"]
                or actual_closed["decision_at"] != closed["decision_at"]):
            raise AssertionError(runtime + " original-native closing trade differs")
        if sha(source) != original_hash or sha(original_path) != original_hash:
            raise AssertionError("Original native saved input was modified")
        results[runtime] = {"status": "PASS", "input_database": str(source), "input_sha256_unchanged": True,
                            "economic_state_preserved": True, "resumed_like_original_native": True,
                            "pending_close_canceled_and_trade_closed_like_original_native": True,
                            "upgraded_export": str(before_path), "resumed_export": str(after_path),
                            "closed_export": str(closed_path)}
    return {"status": "PASS", "genuine_saved_identity": fixture["implementation_identity_hash"],
            "original_database": str(original_path), "original_database_sha256": original_hash,
            "original_database_unchanged": sha(original_path) == original_hash,
            "original_executable_sha256": fixture["original_executable_sha256"], "runtimes": results}


def verify(args):
    configure_environment()
    output = writable_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    binary = writable_path(args.executable)
    binary_hash = sha(binary)
    build_id = current()[0]
    fixture = output / "fixture"
    for day in (date(2026, 11, 9), date(2026, 11, 10)):
        write_synthetic_dataset(fixture, day=day)
    config = replace(demo_config(AppConfig(), fixture), runtime_root=str(output / "session_runtime"))
    config_path = config.save(output / "config.json")
    scripts = scenario_scripts(config, output)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    compiled_env = dict(env)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "QT_PLUGIN_PATH"):
        compiled_env.pop(key, None)
    compiled_env["PATH"] = r"C:\Windows\System32;C:\Windows"
    commands = []
    report = {"schema": "QQQ_FAST_SKIP_RELEASE_V1", "status": "FAIL", "build_id": build_id,
              "executable": str(binary), "executable_sha256": binary_hash, "commands": commands,
              "scope": "Replay optimization, simulation economic parity, saved-session interoperability and desktop smoke",
              "scientific_probability_qualification": "UNCHANGED; original reports are historical",
              "separate_clean_windows_account_tested": False}
    native_path = ROOT / "runtime/results/fast-skip-native-verification.json"
    old_fixture = json.loads(writable_path(args.original_native_fixture).read_text(encoding="utf-8"))
    prefixes = {"compiled": [str(binary)], "source": [str(ROOT / ".venv/Scripts/python.exe"), "-m", "qqq_sim"],
                "original_native": [old_fixture["original_executable"]]}

    def run(name, runtime, argv, extra_env=None):
        child_env = dict(env if runtime == "source" else compiled_env)
        if extra_env:
            child_env.update(extra_env)
        started = perf_counter()
        result = subprocess.run([*prefixes[runtime], *argv], cwd=ROOT, env=child_env, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=180)
        record = {"name": name, "runtime": runtime, "arguments": argv, "seconds": perf_counter() - started,
                  "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        commands.append(record)
        write_json(output / "result.json", report)
        if result.returncode:
            raise RuntimeError(name + " failed: " + result.stderr + result.stdout)
        return json.loads(result.stdout)

    def compare(name, source_path, compiled_path):
        source = json.loads(source_path.read_text(encoding="utf-8"))
        compiled = json.loads(compiled_path.read_text(encoding="utf-8"))
        for field in ("decision_at", "status", "account"):
            if source[field] != compiled[field]:
                raise AssertionError(name + " differs: " + field)
        if source["snapshot"]["snapshot_id"] != compiled["snapshot"]["snapshot_id"]:
            raise AssertionError(name + " snapshot differs")
        if source["state"]["broker"] != compiled["state"]["broker"]:
            raise AssertionError(name + " broker differs")
        return source, compiled

    try:
        report["performance_evidence"] = performance_evidence()
        doctor = run("compiled_identity", "compiled", ["doctor"])
        if doctor.get("build_id") != build_id:
            raise AssertionError("Compiled source identity is stale")
        for runtime in ("source", "compiled"):
            for label, start, tail in (
                ("360-minutes", "2026-11-09T09:30:00-05:00", ["--minutes", "360"]),
                ("next-session", "2026-11-09T10:00:00-05:00", ["--until", "2026-11-10T09:30:00-05:00"]),
                ("trade", "2026-11-09T10:01:00-05:00", ["--script", str(scripts[runtime])]),
                ("working", "2026-11-09T10:01:00-05:00", ["--script", str(scripts["working"])])):
                run(runtime + "_" + label, runtime, ["--config", str(config_path), "replay", "--at", start,
                    *tail, "--output", str(output / f"{runtime}-{label}.json")])
        pairs = {}
        for label in ("360-minutes", "next-session", "trade", "working", "held"):
            pairs[label] = compare(label, output / f"source-{label}.json", output / f"compiled-{label}.json")
        if not pairs["held"][0]["state"]["broker"]["ledger"]["lots"]:
            raise AssertionError("Held scenario has no option positions")
        if not all(order["status"] == "FILLED" for order in pairs["trade"][0]["state"]["broker"]["orders"].values()):
            raise AssertionError("Trade scenario did not fill its opening and closing orders")
        if not all(order["status"] == "WORKING" for order in pairs["working"][0]["state"]["broker"]["orders"].values()):
            raise AssertionError("Nonmarketable order should still be working")
        for source_runtime, destination_runtime in (("source", "compiled"), ("compiled", "source")):
            original = pairs["trade"][0 if source_runtime == "source" else 1]
            restored_path = output / (destination_runtime + "-restored-" + source_runtime + ".json")
            run(destination_runtime + "_restores_" + source_runtime, destination_runtime,
                ["--config", str(config_path), "export", original["session_id"], "--output", str(restored_path)])
            compare("restored " + source_runtime, output / f"{source_runtime}-trade.json", restored_path)
        report["original_native_upgrade_and_resume"] = original_native_cases(
            run, output, writable_path(args.original_native_fixture))
        screen_path = write_json(output / "screen.json", {"windowFrameMargins": True, "screens": [{
            "name": "FastSkipSmoke", "x": 0, "y": 0, "width": 2560, "height": 1440,
            "availableGeometry": {"x": 0, "y": 0, "width": 2560, "height": 1390},
            "logicalDpi": 96, "logicalBaseDpi": 96, "dpr": 1.0}]})
        gui = run("compiled_desktop", "compiled", ["desktop-smoke", "--pack", args.pack,
                  "--output", str(output / "desktop")],
                  {"QT_QPA_PLATFORM": "offscreen:configfile=" + screen_path.relative_to(ROOT).as_posix()})
        if gui.get("status") != "OK" or not gui.get("saved_and_restored"):
            raise AssertionError("Compiled desktop workflow failed")
        tests = writable_path(args.tests)
        suites = ET.parse(tests).findall(".//testsuite")
        test_counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
                       for key in ("tests", "failures", "errors", "skipped")}
        if not test_counts["tests"] or test_counts["failures"] or test_counts["errors"]:
            raise AssertionError("Passing source regression report required")
        report.update(status="PASS", economic_parity_cases=list(pairs),
                      source_compiled_restore="PASS_BOTH_DIRECTIONS", desktop_smoke="PASS_OFFSCREEN",
                      tests={**test_counts, "path": str(tests), "sha256": sha(tests)},
                      executable_unchanged_during_checks=sha(binary) == binary_hash,
                      sources_unchanged_during_checks=current()[0] == build_id)
        native = native_evidence(native_path, build_id, binary_hash)
        if native is not None:
            report["native_desktop_verification"] = native
        if not report["executable_unchanged_during_checks"] or not report["sources_unchanged_during_checks"]:
            raise AssertionError("Source or binary changed during verification")
    except Exception as error:
        report.update(status="FAIL", error=type(error).__name__, detail=str(error))
    write_json(output / "result.json", report)
    summary = {key: value for key, value in report.items() if key not in ("commands", "native_desktop_verification")}
    if report.get("native_desktop_verification"):
        summary["native_desktop"] = report["native_desktop_verification"]["result"]["status"]
    print(json.dumps(summary, indent=2))
    return report


def assemble(args, report):
    if report["status"] != "PASS":
        raise RuntimeError("Passing scoped verification required before assembly")
    backup_record = json.loads(writable_path(args.backup_manifest).read_text(encoding="utf-8"))
    backup = writable_path(backup_record["backup"])
    destination = writable_path(args.executable).parent
    if sha(backup / "QQQSimulator.exe") != backup_record["executable_sha256"]:
        raise RuntimeError("Original distribution backup integrity differs")
    for folder in ("licenses", "third_party_sources", "runtime", "docs"):
        shutil.copytree(backup / folder, destination / folder, dirs_exist_ok=True)
    shutil.copytree(backup / "evidence", destination / "evidence/historical-before-fast-skip", dirs_exist_ok=True)
    shutil.copy2(backup / "artifact-manifest.json", destination / "evidence/historical-before-fast-skip/original-artifact-manifest.json")
    for name in ("THIRD_PARTY_NOTICES.txt", "dependency-notices.json", "LICENSE_REVIEW.md", "verify-clean-account.ps1"):
        shutil.copy2(backup / name, destination / name)
    for name in ("README.md", "CHANGELOG.md", "requirements.lock", "config.example.json"):
        shutil.copy2(ROOT / name, destination / name)
    shutil.copytree(ROOT / "docs", destination / "docs", dirs_exist_ok=True)
    verification_path = destination / "evidence/fast-skip-verification.json"
    write_json(verification_path, report)
    shutil.copy2(writable_path(args.tests), destination / "evidence/fast-skip-tests.xml")
    shutil.copy2(ROOT / "runtime/results/build-source-manifest.json", destination / "evidence/build-source-manifest.json")
    for source, name in (
        ("runtime/results/fast-skip-windows-build.json", "fast-skip-windows-build.json"),
        ("runtime/results/fast-skip-windows-build.log", "fast-skip-windows-build.log"),
        ("runtime/results/fast-skip-test-evidence.json", "fast-skip-test-evidence.json"),
        ("runtime/results/fast-skip-benchmark-source-manifest.json", "fast-skip-benchmark-source-manifest.json"),
        ("runtime/results/fast-skip-performance-quiet.json", "fast-skip-performance-real.json"),
        ("runtime/results/fast-skip-performance-trades-quiet.json", "fast-skip-performance-trades-real.json"),
        ("runtime/results/fast-skip-native-verification.json", "fast-skip-native-verification.json"),
        ("runtime/results/fast-skip-native-desktop/result.json", "fast-skip-native-desktop.json")):
        path = ROOT / source
        if path.is_file():
            shutil.copy2(path, destination / "evidence" / name)
    old_native = report["original_native_upgrade_and_resume"]
    write_json(destination / "evidence/original-native-upgrade-and-resume.json", old_native)
    (destination / "PORTABLE.md").write_text(
        "# Portable application\n\nKeep this folder at `D:\\optionsimulation\\dist\\QQQSimulator.dist`. "
        "Run `QQQSimulator.exe`, or use `--demo` for synthetic observations.\n\n"
        "This update optimizes time skips. Its exact binary, source identity, economic comparisons, "
        "saved-session interoperability and desktop checks are recorded in "
        "`evidence/fast-skip-verification.json`.\n\n"
        "Reports under `evidence/historical-before-fast-skip` describe the previous release. "
        "They are historical numerical and release evidence, not certification of this updated binary. "
        "Probability qualification and separate clean Windows account testing remain incomplete. "
        "The supplied dependency notices, licenses and matching source archives are preserved.\n",
        encoding="utf-8")
    files = [{"path": path.relative_to(destination).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}
             for path in sorted(destination.rglob("*"))
             if path.is_file() and path.name not in ("artifact-manifest.json", "artifact-manifest.sha256")]
    manifest = {"schema": "QQQ_PORTABLE_ARTIFACT_V1", "build_id": report["build_id"],
                "executable_sha256": report["executable_sha256"], "scope": "FAST_SKIP_UPDATE",
                "verification": "evidence/fast-skip-verification.json", "files": files,
                "bytes": sum(record["bytes"] for record in files), "production_probability_qualified": False,
                "separate_clean_windows_account_tested": False}
    path = write_json(destination / "artifact-manifest.json", manifest)
    (destination / "artifact-manifest.sha256").write_text(sha(path) + "\n", encoding="ascii")
    print(json.dumps({"assembled": str(destination), "files": len(files), "bytes": manifest["bytes"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", default="dist/QQQSimulator.dist/QQQSimulator.exe")
    parser.add_argument("--output", default="runtime/results/fast-skip-release")
    parser.add_argument("--pack", default="runtime/model_packs/validation_fixture_v2")
    parser.add_argument("--tests", default="runtime/results/fast-skip-tests.xml")
    parser.add_argument("--backup-manifest", default="runtime/results/fast-skip-distribution-backup.json")
    parser.add_argument("--original-native-fixture", default="runtime/results/original-native-upgrade-fixture/fixture-record.json")
    assembly = parser.add_mutually_exclusive_group()
    assembly.add_argument("--assemble", action="store_true")
    assembly.add_argument("--assemble-only", action="store_true", help="Assemble from existing passing verification after native smoke finishes")
    args = parser.parse_args()
    if args.assemble_only:
        output = writable_path(args.output)
        report = json.loads((output / "result.json").read_text(encoding="utf-8"))
        if (report["status"] != "PASS" or current()[0] != report["build_id"]
                or sha(writable_path(args.executable)) != report["executable_sha256"]
                or sha(writable_path(args.tests)) != report["tests"]["sha256"]):
            raise RuntimeError("Passing verification no longer matches the source, executable, or tests")
        native = ROOT / "runtime/results/fast-skip-native-verification.json"
        proof = native_evidence(native, report["build_id"], report["executable_sha256"])
        if proof is None:
            raise RuntimeError("Passing native verification required before final assembly")
        report["native_desktop_verification"] = proof
        write_json(output / "result.json", report)
    else:
        report = verify(args)
    if (args.assemble or args.assemble_only) and report["status"] == "PASS":
        assemble(args, report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
