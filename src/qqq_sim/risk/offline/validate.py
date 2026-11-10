"""Validate the installed bytes and report measured numerical evidence."""
from __future__ import annotations
import ast
import dataclasses
import hashlib
import json
import math
import time
from decimal import Decimal
from pathlib import Path
import numpy as np
from ..contracts import RiskLeg, RiskError, stable_hash
from ..payoff import payoff_extrema, terminal_regions, piecewise_regions, region_membership
from ..pricing_fast import european_price, deterministic_american
from ..pack_loader import ModelPack
from .pricing_oracle import crr_price
from .hit_oracle import continuous_barrier_probability, wilson, atomic_json


def reference_fixtures(include_tree=True):
    measurements = []
    def measure(name, actual, expected, tolerance):
        error = abs(float(actual)-float(expected))
        measurements.append({"fixture": name, "actual": float(actual), "expected": float(expected), "absolute_error": error, "tolerance": tolerance, "passed": error <= tolerance})
    call = float(european_price("CALL", 100, 100, 1, .05, 0, .2))
    put = float(european_price("PUT", 100, 100, 1, .05, 0, .2))
    measure("R21_F05_CALL", call, 10.450583572186, 1e-11)
    measure("R21_F05_PUT", put, 5.573526022257, 1e-11)
    measure("R21_F05_PARITY", call-put, 100-100*math.exp(-.05), 1e-11)
    measure("R21_F05_FORWARD", european_price("CALL", 100*math.exp(.05), 100, 1, .05, 0, .2, "TERMINAL_FORWARD"), call, 1e-11)
    measure("R21_F08", deterministic_american("PUT", 80, 100, 1, -.05, .02), 26.711215773061994, 1e-11)
    measure("R21_F10", continuous_barrier_probability(math.log(1.1), .25, .2, 0), .3405384952145612, 1e-12)
    if include_tree:
        measure("R21_F07_CRR_2000", crr_price("PUT", 100, 100, 1, .05, 0, .2, 2000), 6.08998995255238, 1e-8)
        measure("R21_F07_CRR_4000", crr_price("PUT", 100, 100, 1, .05, 0, .2, 4000), 6.090180715022948, 1e-8)
    expiry = 1
    legs = (RiskLeg("SHORT767", "PUT", Decimal(767), -1, expiry, .2), RiskLeg("LONG765", "PUT", Decimal(765), 1, expiry, .2))
    payoff = payoff_extrema(legs, 80)
    measure("R21_F01_MAX", payoff["max_profit"], 80, 0)
    measure("R21_F01_LOSS", payoff["max_loss"], 120, 0)
    measure("R21_F01_TARGET_REGION", terminal_regions(legs, 80, 40)[0].lower, 766.6, 1e-10)
    point = piecewise_regions([1, 2, 3], [-1, 0, -1])
    measure("R21_F14_SINGLETON", float(region_membership(point, 2)), 1, 0)
    return {"status": "PASS" if all(x["passed"] for x in measurements) else "FAIL", "measurements": measurements, "scope": "ANALYTICAL_AND_FINITE_TREE_FIXTURES_ONLY", "full_reference_convergence_qualified": False}


ONLINE_AUDIT_FILES = ("hold.py", "policy.py", "envelope.py", "verify.py", "features.py", "pricing_fast.py", "pack_loader.py",
    "account_adapter.py", "account_features.py", "feature_layout.py", "universe.py", "universe_admission.py",
    "qualification.py", "contracts.py", "payoff.py", "greeks.py")


def online_import_audit():
    root = Path(__file__).parent.parent
    from ... import _build_info
    violations, audited_sources = [], []
    for name in ONLINE_AUDIT_FILES:
        key = "src/qqq_sim/risk/" + name
        path = root/name
        if path.is_file():
            source_bytes = path.read_bytes()
            origin = "FILESYSTEM"
        else:
            source = getattr(_build_info, "ONLINE_AUDIT_SOURCES", {}).get(key)
            expected = _build_info.SOURCE_HASHES.get(key)
            if not isinstance(source, str) or not isinstance(expected, str):
                raise RiskError("ONLINE_AUDIT_SOURCE_UNAVAILABLE", source=key)
            source_bytes = source.encode("utf-8")
            if hashlib.sha256(source_bytes).hexdigest() != expected:
                raise RiskError("ONLINE_AUDIT_SOURCE_INTEGRITY_FAILURE", source=key)
            origin = "VERIFIED_BUILD_SOURCE"
        digest = hashlib.sha256(source_bytes).hexdigest()
        audited_sources.append({"file": name, "sha256": digest, "origin": origin})
        tree = ast.parse(source_bytes.decode("utf-8-sig"), filename=key)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and ("offline" in node.module or node.module.startswith("torch")):
                violations.append({"file": name, "line": node.lineno, "module": node.module})
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "offline" in alias.name or alias.name.startswith("torch"):
                        violations.append({"file": name, "line": node.lineno, "module": alias.name})
    return {"passed": not violations, "violations": violations, "audited_sources": audited_sources}


def validate_pack(path, output_path=None):
    started = time.perf_counter()
    try:
        pack = ModelPack.load(path)
        fixture = reference_fixtures()
        imports = online_import_audit()
        # The loader verifies every declared file and saved validation basis.
        capabilities = pack.validation.get("capabilities", {})
        gates = {}
        for capability in ("price", "hold", "managed"):
            saved = capabilities.get(capability, {})
            count = int(saved.get("independent_cases", 0))
            qualified = bool(saved.get("qualified")) and count >= 200 and saved.get("pass_fraction", 0) >= .99
            if capability == "managed":
                qualified = qualified and bool(saved.get("independent_policy_rollouts"))
            reason = saved.get("reason") if not qualified else None
            if qualified:
                try:
                    pack.require_qualified(capability)
                except RiskError as exc:
                    qualified,reason=False,exc.code
            gates[capability] = {**saved, "qualified": qualified, "reason": reason}
        from ..hold import network
        loaded_byte_checks={}
        for capability,prefix,tolerance in (("hold","hold",.01),("managed","q_eval",.02)):
            key=capability+"_validation_features"
            required=(key,capability+"_validation_reference",capability+"_validation_uncertainty")
            if not all(name in pack.arrays for name in required):
                loaded_byte_checks[capability]={"passed":False,"reason":"VALIDATION_CASE_ARRAYS_MISSING"}
                gates[capability]["qualified"]=False
                continue
            values=pack.arrays[key]
            normalized=(values-pack.arrays[prefix+"_mean"])/pack.arrays[prefix+"_std"]
            output=network(pack.arrays,prefix,normalized)
            if capability=="hold":
                fractions=pack.arrays["hold_validation_bin_fractions"]
                hazards=np.logaddexp(0,output[:,:576])
                prediction=-np.expm1(-np.sum(hazards*fractions,axis=1))
                terminal=np.exp(-np.logaddexp(0,-output[:,-1]))
                prediction+=pack.arrays["hold_validation_terminal"]*(1-prediction)*terminal
            else:
                prediction=np.exp(-np.logaddexp(0,-output[:,0]))
            errors=np.abs(prediction-pack.arrays[capability+"_validation_reference"])+pack.arrays[capability+"_validation_uncertainty"]
            passed=len(errors)>=200 and float(np.mean(errors<=tolerance))>=.99
            loaded_byte_checks[capability]={"passed":passed,"independent_cases":len(errors),"pass_fraction":float(np.mean(errors<=tolerance)),"maximum_absolute_error_plus_reference_uncertainty":float(np.max(errors)),"tolerance":tolerance}
            gates[capability]["qualified"]=gates[capability]["qualified"] and passed
            if capability=="hold":
                from .train import within_bin_validation,HOLD_LABEL_SCHEMA,HOLD_LOSS_SCHEMA
                within_required=("hold_within_validation_labels","hold_within_validation_numerical_uncertainty","hold_within_validation_deterministic")
                if (not all(name in pack.arrays for name in within_required)
                        or pack.manifest.get("hold_label_schema")!=HOLD_LABEL_SCHEMA
                        or pack.manifest.get("hold_loss_schema")!=HOLD_LOSS_SCHEMA):
                    within={"qualified":False,"reason":"WITHIN_BIN_VALIDATION_ARTIFACTS_OR_SCHEMA_MISSING"}
                else:
                    within=within_bin_validation(output,pack.arrays[within_required[0]],pack.arrays[within_required[1]],deterministic=pack.arrays[within_required[2]])
                    within={key:value for key,value in within.items() if key not in ("case_errors","applicable_cases")}
                loaded_byte_checks[capability]["within_bin"]=within
                gates[capability]["qualified"]=gates[capability]["qualified"] and within["qualified"]
        qualified = all(x["qualified"] for x in gates.values()) and fixture["status"] == "PASS" and imports["passed"]
        result = {"status": "QUALIFIED" if qualified else "VALID_UNQUALIFIED_PACK", "qualified": qualified, "pack_id": pack.pack_id, "manifest_hash": pack.manifest_hash, "integrity": "PASS", "capabilities": gates,"loaded_array_prediction_checks":loaded_byte_checks, "reference_fixtures": fixture, "online_import_boundary": imports, "array_count": len(pack.arrays), "pack_kind": pack.manifest.get("pack_kind"), "exclusions": pack.manifest.get("excluded_regimes", [])}
    except (RiskError, OSError, ValueError) as exc:
        result = exc.as_dict() if isinstance(exc, RiskError) else {"status": "FAILED", "reason": "PACK_VALIDATION_ERROR", "message": str(exc)}
        result["qualified"] = False
    result["elapsed_seconds"] = time.perf_counter()-started
    if output_path:
        atomic_json(output_path, result)
    return result
