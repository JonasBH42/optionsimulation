"""Explicit resumable model build, safe artifacts, and nonproduction fixtures.

The research build records incomplete numerical stages instead of manufacturing
qualified probability weights. A fixture pack tests distribution and integrity;
it is never a fallback market model.
"""
from __future__ import annotations
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
from ...config import writable_path, configure_environment
from ..contracts import RiskError, canonical, stable_hash
from ..features import FEATURE_SCHEMA, TOKEN_FIELDS
from ..feature_layout import feature_contract
from ..pack_loader import file_hash
from .design import latin_hypercube, design_record, SEED_KEYS, reference_fixture_job
from .hit_oracle import atomic_json
from .validate import reference_fixtures, online_import_audit, validate_pack


def _write_array(folder, name, value):
    file = folder/(name+".npy")
    with file.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {"file": file.name, "bytes": file.stat().st_size, "sha256": file_hash(file)}


def build_model(config=None):
    configure_environment()
    config = dict(config or {})
    fixture = config.get("fixture", False) or config.get("profile") in ("fixture", "validation_fixture")
    research = bool(config.get("research", False))
    count = int(config.get("case_count", 64 if fixture else 32768))
    if count <= 0 or (not fixture and not research and count < 32768):
        raise RiskError("DESIGN_COVERAGE_INSUFFICIENT", "Production design starts at 32768 cases; use explicit fixture mode for smaller tests")
    model_config = {"fixture": fixture, "case_count": count, "seed": 20261109, "profile": "FROZEN_KNOWN_SURFACE_QQQ_V1", "feature_schema": FEATURE_SCHEMA, "dtype": "float64"}
    if not fixture:
        from .pipeline import run_pipeline
        return run_pipeline(config, model_config)
    build_id = stable_hash(model_config)[:20]
    destination = writable_path(config.get("output_dir", f"runtime/model_packs/{'fixture' if fixture else 'research'}_{build_id}"))
    work = writable_path(config.get("work_dir", f"runtime/builds/{build_id}"))
    work.mkdir(parents=True, exist_ok=True)
    checkpoint = work/"checkpoint.json"
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("config_hash") != stable_hash(model_config):
            raise RiskError("CHECKPOINT_MISMATCH")
        if destination.exists() and saved.get("pack_path") == str(destination):
            return {**saved, "validation": validate_pack(destination), "resumed": True}
    elif destination.exists():
        raise RiskError("OUTPUT_ALREADY_EXISTS", "Existing packs are immutable", path=str(destination))
    started = time.perf_counter()
    temporary = writable_path(str(destination)+".building")
    temporary.mkdir(parents=True, exist_ok=True)
    design = latin_hypercube(count)
    artifacts = {name: _write_array(temporary, "design_"+name, array) for name, array in design.items()}
    records = [design_record(i, design["coordinates"][i]) for i in range(count)]
    design_file = temporary/"design.jsonl"
    with design_file.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"))+"\n")
        stream.flush()
        os.fsync(stream.fileno())
    artifacts["design_records"] = {"file": design_file.name, "bytes": design_file.stat().st_size, "sha256": file_hash(design_file)}
    fixtures = reference_fixtures()
    atomic_json(temporary/"reference_fixtures.json", fixtures)
    artifacts["reference_fixtures"] = {"file": "reference_fixtures.json", "bytes": (temporary/"reference_fixtures.json").stat().st_size, "sha256": file_hash(temporary/"reference_fixtures.json")}
    # A concrete full reference input accompanies every pack, labeled synthetic.
    atomic_json(temporary/"example_reference_job.json", canonical(reference_fixture_job()))
    artifacts["example_reference_job"] = {"file": "example_reference_job.json", "bytes": (temporary/"example_reference_job.json").stat().st_size, "sha256": file_hash(temporary/"example_reference_job.json")}
    versions = {}
    for dependency in ("numpy", "scipy", "torch"):
        try:
            versions[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            versions[dependency] = None
    stages = {
        "B0": {"status": "COMPLETE", "description": "Frozen profiles, schema, seeds and versions"},
        "B1": {"status": "PARTIAL", "description": "Deterministic Latin-hypercube coordinates and family split persisted", "cases": count, "reason": "REFERENCE_QUOTED_UNIVERSE_AND_EXECUTED_CAMPAIGN_GENERATION_REQUIRED"},
        "B2": {"status": "INCOMPLETE", "reason": "ADAPTIVE_AMERICAN_PRICE_CELLS_REQUIRE_CONVERGED_LABELS"},
        "B3": {"status": "IMPLEMENTED_NOT_RELEASE_QUALIFIED", "reason": "INDEPENDENT_TRANSITION_GRID_REFINEMENT_REQUIRED"},
        "B4": {"status": "INCOMPLETE", "reason": "FULL_REFERENCE_LABELS_NOT_QUALIFIED"},
        "B5": {"status": "INCOMPLETE", "reason": "HOLD_TRAINING_REQUIRES_QUALIFIED_LABELS"},
        "B6": {"status": "IMPLEMENTED_NOT_RELEASE_QUALIFIED", "reason": "ACTION_GRAMMAR_COVERAGE_VALIDATION_REQUIRED"},
        "B7": {"status": "INCOMPLETE", "reason": "FITTED_POLICY_AND_INDEPENDENT_ROLLOUT_TRAINING_REQUIRED"},
        "B8": {"status": "FAILED_RELEASE_GATE", "reason": "STOCHASTIC_DOMAIN_UNQUALIFIED"},
        "B9": {"status": "SAFE_FIXTURE_ONLY" if fixture else "RESEARCH_ARTIFACT_ONLY", "reason": "NO_PRODUCTION_PROMOTION"},
    }
    manifest = {"schema_version": "QQQ_P50_PACK_V1", "pack_id": build_id, "pack_kind": "VALIDATION_FIXTURE" if fixture else "INCOMPLETE_RESEARCH_BUILD", "dtype": "float64", "model_config": model_config, "profiles": {"product": "STANDARD_QQQ", "reference": "P50_REFERENCE_V1", "managed": "FROZEN_KNOWN_SURFACE_QQQ_V1", "execution": "SIM_CONDITIONAL_RISK_CHECK_V1", "risk": "SIMULTANEOUS_DELIVERY_HEDGE_V1", "fee": "TT_COMPONENTS_20261028_SIM_V1"}, "monitoring_seconds": 15, "monitoring_status": "INTENDED_NOT_QUALIFIED", "feature_schema": FEATURE_SCHEMA, "position_token_order": TOKEN_FIELDS, "network_architecture": [256, 256, 128], "hold_head": [576, 3], "seeds": {"build": 20261109, "reference": 20261008, "semantic_keys": SEED_KEYS}, "artifacts": artifacts, "versions": versions, "hardware": {"machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version()}, "build_stages": stages, "coverage": {}, "excluded_regimes": ["ALL_STOCHASTIC_HOLD_AND_MANAGED_INFERENCE_PENDING_QUALIFICATION"], "domain_ceilings": {"legs": 8, "quantity": 20, "contracts": 512, "expiries": 8, "days": 30, "sessions": 32, "requested_times": 32, "candidates": 2000}, "validation_status": "UNQUALIFIED", "policy_id": None, "historical_market_observations_used": False}
    manifest['feature_contract']=feature_contract()
    atomic_json(temporary/"manifest.json", manifest)
    with (temporary/"manifest.sha256").open("w", encoding="ascii") as stream:
        stream.write(file_hash(temporary/"manifest.json")+"\n")
        stream.flush()
        os.fsync(stream.fileno())
    validation = {"manifest_hash": stable_hash(manifest), "qualified": False, "capabilities": {key: {"qualified": False, "independent_cases": 0, "reason": "FULL_CALIBRATION_NOT_COMPLETED"} for key in ("price", "hold", "managed")}, "reference_fixtures": fixtures, "online_import_boundary": online_import_audit()}
    atomic_json(temporary/"validation.json", validation)
    (temporary/"validation.sha256").write_text(file_hash(temporary/"validation.json")+"\n", encoding="ascii")
    actual_validation = validate_pack(temporary)
    if actual_validation.get("integrity") != "PASS":
        raise RiskError("PACK_VALIDATION_FAILED", report=actual_validation)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)
    report = {"status": "VALIDATION_FIXTURE_BUILT" if fixture else "INCOMPLETE", "production_qualified": False, "config_hash": stable_hash(model_config), "build_id": build_id, "pack_path": str(destination), "stages": stages, "elapsed_seconds": time.perf_counter()-started, "validation": validate_pack(destination)}
    atomic_json(checkpoint, report)
    atomic_json(work/"build_report.json", report)
    return report
