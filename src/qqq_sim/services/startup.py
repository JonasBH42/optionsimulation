"""Explicit local startup checks. No installation, model build or network access."""
from __future__ import annotations
from datetime import datetime, timedelta
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from zoneinfo import ZoneInfo

from qqq_sim.config import PROJECT_ROOT, writable_path

DEPENDENCIES = {
    "PySide6": ("PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"),
    "pyqtgraph": ("pyqtgraph",), "numpy": ("numpy",), "scipy": ("scipy.special",),
    "pyarrow": ("pyarrow.parquet",), "duckdb": ("duckdb",), "tzdata": ("tzdata",),
}
MINIMUM_RUNTIME_FREE_BYTES = 64 * 1024**2


def _probe_write(path):
    path=writable_path(path)
    path.mkdir(parents=True,exist_ok=True)
    probe=None
    try:
        with tempfile.NamedTemporaryFile(prefix=".startup-",suffix=".probe",dir=path,delete=False) as stream:
            probe=Path(stream.name)
            stream.write(b"QQQ runtime write probe\n")
            stream.flush();os.fsync(stream.fileno())
        if probe.read_bytes()!=b"QQQ runtime write probe\n":raise OSError("Runtime read-back mismatch")
    finally:
        if probe is not None:probe.unlink(missing_ok=True)
    return {"path":str(path),"write_read_fsync_delete":"PASS"}


def _probe_source(path):
    """Bounded read-only inspection; full partition validation remains lazy."""
    root=Path(path).resolve()
    if not root.is_dir():raise FileNotFoundError(str(root))
    pending=[(root,0)];visited=0
    while pending and visited<32:
        folder,depth=pending.pop();visited+=1
        with os.scandir(folder) as entries:
            for index,entry in enumerate(entries):
                if index>=8:break
                if entry.is_file():
                    with open(entry.path,"rb") as stream:stream.read(1)
                    return {"path":str(root),"sample_file":entry.path,"scope":"BOUNDED_ROOT_AND_ONE_FILE_READ"}
                if entry.is_dir() and depth<6:pending.append((Path(entry.path),depth+1))
    return {"path":str(root),"sample_file":None,"scope":"BOUNDED_DIRECTORY_READ_ONLY"}


def _pack_status(path):
    if not path:return {"status":"PACK_MISSING","available":False,"analytics_verified":False}
    folder=Path(path).resolve()
    result={"path":str(folder),"available":False,"analytics_verified":False}
    for name in ("manifest.json","validation.json"):
        file=folder/name
        if not file.is_file():return {**result,"status":"PACK_MISSING","missing_file":name}
        if file.stat().st_size>16*1024**2:return {**result,"status":"PACK_METADATA_TOO_LARGE"}
        raw=file.read_bytes()
        receipt=file.with_suffix(".sha256")
        if not receipt.is_file() or receipt.read_text(encoding="ascii").strip()!=hashlib.sha256(raw).hexdigest():
            return {**result,"status":"PACK_INTEGRITY_FAILURE","artifact":name}
        result[name.removesuffix(".json")]=json.loads(raw.decode("utf-8"))
    manifest,validation=result.pop("manifest"),result.pop("validation")
    if not isinstance(manifest,dict) or not isinstance(validation,dict) or manifest.get("schema_version")!="QQQ_P50_PACK_V1":
        return {**result,"status":"PACK_SCHEMA_UNSUPPORTED"}
    from qqq_sim.risk.contracts import stable_hash
    if validation.get("manifest_hash")!=stable_hash(manifest):return {**result,"status":"PACK_VALIDATION_MISMATCH"}
    capabilities=validation.get("capabilities",{})
    declared={name:bool(capabilities.get(name,{}).get("qualified",False)) for name in ("price","hold","managed")}
    # Availability is a bounded startup check. Never promote an unverified
    # artifact array/network from its metadata; normal loading verifies bytes.
    return {**result,"available":True,"pack_id":manifest.get("pack_id"),"declared_capabilities":declared,
            "status":"PACK_AVAILABLE_PENDING_FULL_VERIFICATION" if any(declared.values()) else "PACK_NOT_QUALIFIED",
            "integrity_scope":"MANIFEST_AND_VALIDATION_RECEIPTS_ONLY"}


def startup_report(config, *, minimum_free_bytes=MINIMUM_RUNTIME_FREE_BYTES):
    """Return actual local checks and only the setup actions that remain needed."""
    from qqq_sim._build_info import BUILD_ID,LOCK_HASH
    development=not (getattr(sys,"frozen",False) or "__compiled__" in globals())
    report={"schema_version":1,"python":sys.version,"python_executable":sys.executable,"venv":sys.prefix,
            "build_id":BUILD_ID,"build_dependency_lock_hash":LOCK_HASH,"platform":platform.platform(),
            "mode":"DEVELOPMENT" if development else "STANDALONE","dependencies":{},"checks":[],"issues":[],
            "source_roots":{},"model":config.pack_path or "PACK_MISSING","output_root":str(writable_path(config.runtime_root)),
            "real_orders_sent":False,"network_access_attempted":False}
    def check(name,operation,*,required=True,field=None,remedy=None):
        try:
            details=operation()
            report["checks"].append({"name":name,"status":"PASS","details":details})
            return details
        except Exception as exc:
            reason=getattr(exc,"code",type(exc).__name__)
            issue={"name":name,"status":"FAILED" if required else "UNAVAILABLE","reason":reason,
                   "message":str(exc),"required_for_runtime":required,"field":field,"remedy":remedy}
            report["checks"].append(issue);report["issues"].append(issue)
            return None
    def interpreter():
        if not ((3,12)<=sys.version_info[:2]<(3,13)):raise RuntimeError("PYTHON_3_12_REQUIRED")
        if development and Path(sys.prefix).resolve()!=PROJECT_ROOT/".venv":raise RuntimeError("PROJECT_VENV_REQUIRED")
        return {"version":platform.python_version(),"project_venv_active":not development or Path(sys.prefix).resolve()==PROJECT_ROOT/".venv"}
    check("interpreter",interpreter,remedy="Run .\\setup.ps1, then .\\launch.ps1 to use the project .venv.")
    for distribution,modules in DEPENDENCIES.items():
        def dependency(distribution=distribution,modules=modules):
            for module in modules:importlib.import_module(module)
            version=importlib.metadata.version(distribution)
            report["dependencies"][distribution]=version
            return {"version":version,"imported_modules":list(modules)}
        check("import:"+distribution,dependency,remedy="Run .\\setup.ps1 to restore the locked project environment." if development else "Restore the complete standalone distribution folder, including its native libraries.")
    def timezone():
        ny=ZoneInfo("America/New_York")
        if datetime(2026,1,15,tzinfo=ny).utcoffset()!=timedelta(hours=-5) or datetime(2026,7,15,tzinfo=ny).utcoffset()!=timedelta(hours=-4):
            raise RuntimeError("NEW_YORK_TIMEZONE_RULES_INVALID")
        return {"zone":"America/New_York","winter_offset_hours":-5,"summer_offset_hours":-4}
    check("timezone",timezone,remedy="Restore the locked tzdata dependency or the complete standalone distribution.")
    runtime=writable_path(config.runtime_root)
    for path in (runtime,*(runtime/name for name in ("catalog","sessions","results","model_packs","config")),PROJECT_ROOT/".tmp",PROJECT_ROOT/".cache"):
        check("runtime_write:"+str(path),lambda path=path:_probe_write(path),field="runtime_root",remedy="Choose a writable runtime_root inside D:\\optionsimulation or restore access to this directory.")
    def disk():
        usage=shutil.disk_usage(runtime)
        if usage.free<minimum_free_bytes:raise OSError(f"DISK_SPACE_LOW: {usage.free} free bytes; {minimum_free_bytes} required for ordinary runtime")
        return {"free_bytes":usage.free,"required_free_bytes":minimum_free_bytes,"scope":"ORDINARY_RUNTIME_ONLY_OFFLINE_BUILDS_HAVE_SEPARATE_BUDGETS"}
    check("disk_space",disk,field="runtime_root",remedy="Free space on the runtime drive; no files are removed automatically.")
    def schemas():
        config.validate()
        from qqq_sim.storage.events import SCHEMA_VERSION
        from qqq_sim.broker.fees import FeeProfile
        profile=FeeProfile()
        return {"config_schema":config.schema_version,"event_schema":SCHEMA_VERSION,"fee_profile":profile.id,"fee_profile_hash":profile.hash,
                "rule_mode":config.rule_mode,"rule_asof":config.rule_asof,"account_profile":config.account_profile}
    check("schema_and_rules",schemas,remedy="Use the supported versioned fields from config.example.json; preserve old sessions for explicit compatible restore.")
    if config.rule_mode=="strict_historical":
        report["issues"].append({"name":"historical_rules","status":"UNAVAILABLE","reason":"BROKER_RULE_HISTORY_MISSING","required_for_runtime":False,
            "field":"rule_mode","remedy":"Complete dated broker-rule inputs are required for strict historical trading; the included fixed-rule mode is an explicit counterfactual."})
    sources={}
    for field in ("underlying_root","options_root"):
        path=getattr(config,field)
        sources[field]=check("source:"+field,lambda path=path:_probe_source(path),required=False,field=field,
            remedy=f"Set {field} to a readable archive directory in your configuration, then launch with --config; or select Session > Open synthetic demonstration.")
        report["source_roots"][path]=sources[field] is not None
    calendar_ok=True
    if config.calendar_path:
        from qqq_sim.services.calendars import load_calendar
        calendar_ok=check("calendar_records",lambda:{"records":len(load_calendar(config.calendar_path).records),"declared_profile":config.calendar_profile},field="calendar_path",remedy="Correct the configured calendar JSON records and timezone offsets.") is not None
    pack=check("pack_metadata",lambda:_pack_status(config.pack_path),required=False,field="pack_path",remedy="Select an intact model-pack directory through Model > Load pack.")
    report["pack"]=pack or {"status":"PACK_UNREADABLE","available":False,"analytics_verified":False}
    if pack and pack["status"]!="PACK_AVAILABLE_PENDING_FULL_VERIFICATION":
        report["issues"].append({"name":"model_availability","status":"UNAVAILABLE","reason":pack["status"],"required_for_runtime":False,
            "field":"pack_path","remedy":"Browsing and simulation remain available. Select a qualified pack through Model > Load pack; Model > Offline build instructions describes the explicit build workflow."})
    report["runtime_ready"]=not any(item["required_for_runtime"] for item in report["issues"])
    report["can_open_session"]=report["runtime_ready"] and sources["underlying_root"] is not None and calendar_ok
    report["status"]="SETUP_REQUIRED" if not report["runtime_ready"] else "LIMITED" if report["issues"] else "READY"
    return report


def setup_guidance(report):
    lines=["Local startup checks: "+report["status"]]
    for issue in report["issues"]:
        field=(issue.get("field")+": ") if issue.get("field") else ""
        lines.extend(("",field+str(issue.get("reason",issue["name"]))))
        if issue.get("message"):lines.append(issue["message"])
        if issue.get("remedy"):lines.append(issue["remedy"])
    if report.get("pack",{}).get("status")=="PACK_AVAILABLE_PENDING_FULL_VERIFICATION":
        lines.extend(("","The configured pack is present. Normal model loading verifies all artifact bytes and capability gates before analysis."))
    return "\n".join(lines)
