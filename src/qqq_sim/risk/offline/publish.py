"""Validate saved model bytes in a fresh process before atomic publication."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
from ...config import writable_path
from ..contracts import RiskError,stable_hash
from ..pack_loader import file_hash
from .hit_oracle import atomic_json


def validate_and_publish(staging,destination,report_file):
    staging,destination,report_file=(writable_path(value) for value in (staging,destination,report_file))
    script="from qqq_sim.config import configure_environment; configure_environment(); from qqq_sim.risk.offline.validate import validate_pack; import sys; validate_pack(sys.argv[1],sys.argv[2])"
    frozen=bool(getattr(sys,"frozen",False))
    command=[sys.executable,"validate-pack",str(staging),"--output",str(report_file)] if frozen else [sys.executable,"-c",script,str(staging),str(report_file)]
    try:
        execution=subprocess.run(command,capture_output=True,text=True,timeout=120,check=False,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0) if os.name=="nt" else 0)
    except subprocess.TimeoutExpired as exc:
        raise RiskError("RESOURCE_LIMIT","Clean-process saved-byte validation exceeded its120-second cap") from exc
    if execution.returncode not in ((0,2) if frozen else (0,)) or not report_file.is_file():
        raise RiskError("PACK_CLEAN_PROCESS_VALIDATION_FAILED",returncode=execution.returncode,diagnostic=execution.stderr[-3000:])
    report=json.loads(report_file.read_text(encoding="utf-8"))
    if report.get("integrity")!="PASS":
        raise RiskError("PACK_VALIDATION_FAILED",report=report)
    manifest=json.loads((staging/"manifest.json").read_text(encoding="utf-8"))
    if report.get("manifest_hash")!=stable_hash(manifest):
        raise RiskError("PACK_CLEAN_PROCESS_MANIFEST_MISMATCH")
    validation=json.loads((staging/"validation.json").read_text(encoding="utf-8"))
    if validation.get("qualified") and not report.get("qualified"):
        validation["qualified"]=False
        validation["capabilities"]=report["capabilities"]
        validation["promotion_reason"]="SAVED_BYTES_DID_NOT_PASS_CLEAN_PROCESS_GATES"
        manifest["validation_status"]="UNQUALIFIED"
        manifest["pack_kind"]="PRODUCTION_BUILD_CHECKPOINT"
        atomic_json(staging/"manifest.json",manifest)
        (staging/"manifest.sha256").write_text(file_hash(staging/"manifest.json")+"\n",encoding="ascii")
        validation["manifest_hash"]=stable_hash(manifest)
        atomic_json(staging/"validation.json",validation)
        (staging/"validation.sha256").write_text(file_hash(staging/"validation.json")+"\n",encoding="ascii")
    if destination.exists():
        for file in staging.iterdir():
            existing=destination/file.name
            if not existing.is_file() or file_hash(existing)!=file_hash(file):
                raise RiskError("IMMUTABLE_PACK_COLLISION",artifact=file.name)
    else:
        destination.parent.mkdir(parents=True,exist_ok=True)
        os.replace(staging,destination)
    return {**report,"published_path":str(destination),"validated_in_clean_process":True}
