"""Immutable actual-initialization benchmarks and bounded fast inference."""
from __future__ import annotations
import copy
from dataclasses import asdict
import math
from pathlib import Path

from qqq_sim.broker.models import BrokerError, OrderDraft, OrderLeg, D, fingerprint
from qqq_sim.broker.service import Broker
from qqq_sim.serialization import primitive, digest
from qqq_sim.services.analysis import AnalysisIdentity


def freeze_initial_benchmark(controller, campaign_id, at, snapshot=None):
    """Called inside the fill transaction, before any later event is consumed."""
    c = controller
    campaign = c.broker.ledger.campaigns[campaign_id]
    if campaign.target is None or campaign.benchmark_id is not None:
        return
    basis = dict(campaign.initial_fill_basis or {})
    basis.update(rule_profile=c.config.rule_profile, operational_model=campaign.operational_model,
                 pack_hash=c._pack_hash, parameter_source_hash=c._lifecycle_registry.source_hash,
                 reference_input_hashes=c._reference_hashes)
    request = None
    reason = None
    try:
        snapshot = snapshot or c.data.snapshot(at, state_version=c.version+1, inclusive=True)
        held = c.broker.ledger.positions(campaign_id)
        held_keys = {contract.key for contract, quantity in held}
        from qqq_sim.services.adapters import broker_contract
        basis["entry_market_snapshot"] = primitive({"asof": snapshot.asof, "snapshot_id": snapshot.snapshot_id,
            "timing_profile": snapshot.timing_profile, "spot": snapshot.spot,
            "last_completed_candle": snapshot.candles[-1] if snapshot.candles else None,
            "quotes": [entry for entry in snapshot.quotes if broker_contract(entry.contract, c.data.calendar).key in held_keys]})
        if not c.config.pack_path:
            raise BrokerError("PACK_MISSING")
        from qqq_sim.services.analysis_inputs import prepare_analysis, cached_analysis_pack
        view = copy.copy(c)
        view.snapshot, view.decision_at = snapshot, at
        # Exact economic state is frozen, with prior analysis payloads removed
        # from account preview context to prevent recursively nested requests.
        state = c.broker.to_dict()
        for raw in state["ledger"]["campaigns"].values():
            raw["benchmark_inputs"] = None
            raw["benchmark_result"] = None
        view.broker = Broker.from_dict(state)
        pack = cached_analysis_pack(view)
        pack.require_qualified("hold")
        pack.require_qualified("managed")
        if not pack.manifest.get("policy_id"):
            raise BrokerError("FROZEN_POLICY_ID_MISSING")
        draft = OrderDraft(campaign_id, tuple(OrderLeg(contract, quantity) for contract, quantity in held), D(0),
                           campaign.deadline, campaign.hard_cutoff, intent="ADJUST",
                           operational_model=campaign.operational_model, standing_protection=campaign.standing_protection)
        prepared = prepare_analysis(view, draft, "INITIAL_FILL_BENCHMARK", current_position=True, register=False)
        if "job" not in prepared:
            raise BrokerError(prepared.get("reason", "INITIAL_BENCHMARK_INPUTS_UNAVAILABLE"))
        request = {key: primitive(prepared[key]) for key in ("job", "pack_path", "campaign", "universe", "limits")}
        request.update(pack_hash=c._pack_hash, policy_id=pack.manifest["policy_id"],
                       rule_hash=digest(c.config.rule_profile), initial_snapshot_hash=snapshot.snapshot_id)
        request["campaign"].update(initial_deadline_utc_us=request["job"]["deadline_utc_us"],
                                    active_deadline_utc_us=request["job"]["deadline_utc_us"])
    except Exception as exc:
        reason = getattr(exc, "code", "INITIAL_BENCHMARK_PREPARATION_FAILED")
        if reason == "DATA_REVISION_CHANGED":
            raise
        basis["benchmark_input_failure"] = {"reason": reason, "message": str(exc)}
    campaign.initial_fill_basis = primitive(basis)
    campaign.benchmark_id = fingerprint({"basis": basis, "request": request})
    campaign.benchmark_inputs = request
    campaign.benchmark_status = "PENDING" if request is not None else "UNKNOWN"
    campaign.benchmark_reason = "ASYNC_INITIAL_FILL_INFERENCE_PENDING" if request is not None else reason
    c.broker.ledger.append_event("benchmark-freeze:"+campaign.benchmark_id, "INITIAL_BENCHMARK_FROZEN", at,
        {"campaign_id": campaign_id, "benchmark_id": campaign.benchmark_id, "status": campaign.benchmark_status,
         "reason": campaign.benchmark_reason, "basis_hash": fingerprint(basis)})


def prepare_initial_benchmark(controller, campaign_id):
    campaign = controller.broker.ledger.campaigns[campaign_id]
    if campaign.benchmark_status != "PENDING" or campaign.benchmark_inputs is None:
        return {"campaign_id": campaign_id, "benchmark_id": campaign.benchmark_id,
                "status": campaign.benchmark_status, "reason": campaign.benchmark_reason}
    request = copy.deepcopy(campaign.benchmark_inputs)
    identity = AnalysisIdentity(controller.session_id, controller.branch_id, controller.version,
        request["initial_snapshot_hash"], campaign.benchmark_id, digest(request["campaign"]),
        digest(request["job"]), request["pack_hash"], request["rule_hash"])
    return {**request, "campaign_id": campaign_id, "benchmark_id": campaign.benchmark_id, "identity": identity}


def run_initial_benchmark(request):
    """Spawn entry point: score frozen policy, then evaluate that exact policy."""
    from qqq_sim.risk.contracts import ReferenceJob, RiskLeg, RiskError
    from qqq_sim.risk.hold import evaluate_hold
    from qqq_sim.risk.policy import select_policy_action, evaluate_policy_action
    result = {"campaign_id": request["campaign_id"], "benchmark_id": request["benchmark_id"],
              "pack_hash": request["pack_hash"], "policy_id": request["policy_id"],
              "status": "UNKNOWN", "p0_hold": None, "p0_policy": None, "p0_policy_range": None,
              "offline_calls": 0}
    try:
        from qqq_sim.services.analysis import _frozen_pack
        pack = _frozen_pack(request['pack_path'],request['pack_hash'])
        if pack.manifest.get("policy_id") != request["policy_id"]:
            raise RiskError("BENCHMARK_POLICY_ID_MISMATCH")
        job = ReferenceJob.from_dict(request["job"])
        universe = tuple(RiskLeg.from_dict(raw) for raw in request["universe"])
        hold = evaluate_hold(job, pack)
        result["hold"] = hold
        if hold.get("status") == "OK":
            result["p0_hold"] = hold.get("hold_p50")
        anchor=request['limits'].get('managed_anchor')
        if anchor is not None:
            result['managed_anchor']=anchor
            if anchor.get('status')!='OK':raise RiskError('MANAGED_ANCHOR_UNAVAILABLE')
        managed_raw=request['limits'].get('managed_job')
        managed_job=ReferenceJob.from_dict(managed_raw) if managed_raw else job
        selected = select_policy_action(managed_job, request["campaign"], universe, request["limits"], pack)
        policy = evaluate_policy_action(managed_job, request["campaign"], selected["action"], pack, universe)
        result.update(policy=policy, selection=selected)
        if policy.get("status") == "OK":
            result["p0_policy"] = policy.get("q_policy")
            result["p0_policy_range"] = policy.get("empirical_error_range")
        result["status"] = "KNOWN" if all(result[key] is not None for key in ("p0_hold", "p0_policy")) else "PARTIAL" if any(result[key] is not None for key in ("p0_hold", "p0_policy")) else "UNKNOWN"
        result["reason"] = None if result["status"] == "KNOWN" else policy.get("reason") or hold.get("reason") or "BENCHMARK_OUTSIDE_VALIDATED_DOMAIN"
    except Exception as exc:
        result["status"] = "PARTIAL" if result["p0_hold"] is not None else "UNKNOWN"
        result["reason"] = getattr(exc, "code", "INITIAL_BENCHMARK_INFERENCE_FAILED")
        result["message"] = str(exc)
    return result


def apply_benchmark_result(ledger, campaign_id, benchmark_id, result, at):
    campaign = ledger.campaigns[campaign_id]
    if campaign.benchmark_id != benchmark_id or result.get("benchmark_id") != benchmark_id or result.get("campaign_id") != campaign_id:
        raise BrokerError("INITIAL_BENCHMARK_IDENTITY_MISMATCH")
    if campaign.benchmark_result is not None:
        if campaign.benchmark_result == primitive(result):
            return False
        raise BrokerError("INITIAL_BENCHMARK_ALREADY_RECORDED")
    request = campaign.benchmark_inputs
    if campaign.benchmark_status != "PENDING" or request is None:
        raise BrokerError("INITIAL_BENCHMARK_NOT_PENDING")
    if result.get("pack_hash") != request["pack_hash"] or result.get("policy_id") != request["policy_id"]:
        raise BrokerError("INITIAL_BENCHMARK_MODEL_MISMATCH")
    for key in ("p0_hold", "p0_policy"):
        value = result.get(key)
        if value is not None and (isinstance(value, bool) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1):
            raise BrokerError("INVALID_BENCHMARK_PROBABILITY")
    campaign.p0_hold, campaign.p0_policy = result.get("p0_hold"), result.get("p0_policy")
    campaign.p0_policy_range = result.get("p0_policy_range")
    campaign.benchmark_policy_id = request["policy_id"]
    campaign.benchmark_deadline_utc_us = request["job"]["deadline_utc_us"]
    campaign.benchmark_hard_cutoff_utc_us = request["campaign"]["hard_cutoff_utc_us"]
    campaign.benchmark_status = "KNOWN" if campaign.p0_hold is not None and campaign.p0_policy is not None else "PARTIAL" if campaign.p0_hold is not None or campaign.p0_policy is not None else "UNKNOWN"
    campaign.benchmark_reason = result.get("reason")
    campaign.benchmark_result = primitive(result)
    ledger.append_event("benchmark-result:"+benchmark_id, "INITIAL_BENCHMARK_RECORDED", at,
        {"campaign_id": campaign_id, "benchmark_id": benchmark_id, "result": primitive(result)})
    return True
