"""Independent chronological managed-policy transitions and rollout accounting."""
from __future__ import annotations
import math
from dataclasses import dataclass, replace
from decimal import Decimal,ROUND_FLOOR,ROUND_CEILING
import numpy as np
from ..contracts import RiskLeg, RiskError, YEAR_US, canonical
from ..features import price_contract
from ..policy import action_fees
from ..verify import exact_poststate, risk_certificate


def fresh_account_limits(state,limits=None):
    """Each decision search receives a fresh bounded account-preview request."""
    return {key:value for key,value in (state.account_context or limits or {}).items() if key!="_account_preview_work_budget"}


def normal_quadrature(order=17):
    from scipy.linalg import eigh_tridiagonal
    eigenvalues, vectors = eigh_tridiagonal(np.zeros(order), np.sqrt(np.arange(1, order)/2))
    return np.sqrt(2)*eigenvalues, vectors[0]**2


def converged_expectation(function, tolerance=.0025):
    previous, passes, records = None, 0, []
    for n in (17, 33, 65, 129):
        nodes, weights = normal_quadrature(n)
        values = np.asarray([function(z) for z in nodes], dtype=np.float64)
        if np.any((values < 0) | (values > 1)):
            raise RiskError("INVALID_PROBABILITY_TARGET")
        value = float(np.dot(weights, values))
        error = abs(value-previous) if previous is not None else None
        records.append({"nodes": n, "value": value, "difference": error})
        passes = passes+1 if error is not None and error <= tolerance else 0
        if passes >= 2:
            return {"value": value, "status": "CONVERGED", "refinements": records}
        previous = value
    raise RiskError("CONTROL_QUADRATURE_UNRESOLVED", refinements=records)


@dataclass(frozen=True)
class ManagedState:
    job: object
    campaign: dict
    universe: tuple
    success: bool = False
    failed: bool = False
    events: tuple = ()
    account_context: dict | None = None

    def __post_init__(self):
        from ..account_features import bind_account_features
        object.__setattr__(self,'campaign',bind_account_features(self.campaign,self.account_context))


def transition(state, action, normal, pack, limits=None):
    job, campaign = state.job, dict(state.campaign)
    if state.success or state.failed:
        return state
    when = int(action["eligible_utc_us"])
    if when <= job.t0_utc_us:
        raise RiskError("ORDER_CAUSALITY_FAILURE")
    hard = campaign.get("hard_cutoff_utc_us")
    if hard is None:
        raise RiskError("NEEDS_CAMPAIGN_CUTOFF")
    campaign.setdefault("active_deadline_utc_us",job.deadline_utc_us)
    effective_deadline=min(hard,int(campaign["active_deadline_utc_us"]))
    stress_event=None
    if campaign.get("extra_latency_seconds"):
        delayed=when+int(campaign["extra_latency_seconds"])*1_000_000
        opportunities=[max(opening,delayed) for opening,close in job.eligible_intervals if max(opening,delayed)<close]
        if not opportunities:
            return ManagedState(job,campaign,state.universe,False,True,(*state.events,{"status":"LATENCY_NO_FUTURE_ELIGIBLE_EXECUTION"}),state.account_context)
        when=min(opportunities)
        action={**action,"eligible_utc_us":when}
        stress_event="EXTRA_EXECUTION_LATENCY"
    if campaign.get("forced_first_liquidation_nonfill") and action["family"]=="FULL_CLOSE" and not campaign.get("_first_liquidation_stress_used"):
        campaign["_first_liquidation_stress_used"]=True
        action={**action,"family":"HOLD","legs":[]}
        stress_event="FORCED_FIRST_LIQUIDATION_NONFILL"
    elif campaign.get("partial_ratio_fill_stress") and action["family"]!="HOLD" and action.get("legs"):
        ratio=math.gcd(*(abs(int(leg["quantity"])) for leg in action["legs"]))
        if ratio>1:
            action={**action,"legs":[{**leg,"quantity":int(leg["quantity"])//ratio} for leg in action["legs"]],"limit_cost_usd":str(Decimal(str(action["limit_cost_usd"]))/ratio)}
            stress_event="PARTIAL_ATOMIC_RATIO_FILL"
    if when>effective_deadline:
        return ManagedState(job,campaign,state.universe,False,True,(*state.events,{"status":"DEADLINE_UNRESOLVED_EXPOSURE","event_at_utc_us":effective_deadline,"open_contract_quantities":{x.contract_id:x.quantity for x in job.signed_legs}}),state.account_context)
    if when >= job.fixing_utc_us:
        raise RiskError("LIFECYCLE_MODEL_UNSUPPORTED", "Forecast cannot erase unresolved physical delivery")
    dt = (when-job.t0_utc_us)/YEAR_US
    spot = job.x0*math.exp((job.rate_cc-job.dividend_yield_cc-job.sigma_path**2/2)*dt+job.sigma_path*math.sqrt(dt)*float(normal))
    quotes = {}
    for leg in (*state.universe, *job.signed_legs):
        tau = (leg.fixing_utc_us-when)/YEAR_US
        if tau <= 0:
            continue
        value = float(price_contract(leg, spot, tau, job, pack))
        tick,half=Decimal(str(leg.tick)),Decimal(str(leg.half_spread))
        mark=Decimal(str(value))
        bid=(max(Decimal(0),mark-half)/tick).to_integral_value(rounding=ROUND_FLOOR)*tick
        ask=((mark+half)/tick).to_integral_value(rounding=ROUND_CEILING)*tick
        quotes[leg.contract_id] = replace(leg, bid=float(bid), ask=float(ask), known_at_utc_us=when)
    event = {"event_at_utc_us": when, "action_id": action["id"], "status": "HOLD"}
    if stress_event:
        event["conditional_stress"]=stress_event
    legs = tuple(replace(quotes[x.contract_id], quantity=x.quantity) for x in job.signed_legs)
    new_job = replace(job, t0_utc_us=when, deadline_utc_us=max(when, job.deadline_utc_us), x0=spot, signed_legs=legs, parameter_provenance=(),job_schema="MANAGED_STATE_V1",reporting_times_utc_us=tuple(t for t in job.reporting_times_utc_us if t>=when))
    if action["family"] != "HOLD":
        changes = []
        for raw in action["legs"]:
            requested = RiskLeg.from_dict(raw)
            if requested.contract_id not in quotes:
                raise RiskError("ACTION_CONTRACT_UNAVAILABLE")
            changes.append(replace(quotes[requested.contract_id], quantity=requested.quantity))
        cost = sum((x.quantity*x.multiplier*Decimal(str(x.ask if x.quantity > 0 else x.bid)) for x in changes), Decimal(0))
        fees = action_fees(new_job, changes)
        if cost > Decimal(action["limit_cost_usd"]):
            event["status"] = "NO_FILL_LIMIT"
        elif fees > Decimal(action.get("max_total_fees_usd", "50")):
            event["status"] = "NO_FILL_FEE_CAP"
        else:
            postlegs, updated = exact_poststate(legs, campaign, changes, cost, fees)
            certificate = risk_certificate(postlegs, updated, limits)
            if certificate["status"] != "VERIFIED_WITHIN_DECLARED_MODEL":
                event.update(status="NO_FILL_RISK", certificate=certificate)
            else:
                legs, campaign = postlegs, updated
                campaign["remaining_actions"] = campaign.get("remaining_actions", 12)-1
                if action["family"] == "ROLL_EXPIRATION":
                    campaign["remaining_rolls"] = campaign.get("remaining_rolls", 3)-1
                event.update(status="FILLED", premium_cost_usd=str(cost), fees_usd=str(fees))
    account_context=state.account_context
    if account_context is not None:
        from ..account_adapter import advance_isolated_account_context
        authoritative=advance_isolated_account_context(account_context,job,action,tuple(quotes.values()),when)
        account_context=authoritative["context"]
        legs=tuple(replace(quotes[key],quantity=quantity) for key,quantity in authoritative["positions"].items())
        if Decimal(authoritative["target_usd"])!=Decimal(str(state.campaign["target_usd"])):
            raise RiskError("ACCOUNT_TARGET_BASIS_MISMATCH")
        for key in ("net_cash_usd","gross_cash_usd","remaining_expense_budget_usd"):
            campaign[key]=authoritative[key]
        campaign["remaining_actions"]=state.campaign.get("remaining_actions",12)-int(authoritative["filled"])
        campaign["remaining_rolls"]=state.campaign.get("remaining_rolls",3)-int(authoritative["filled"] and action["family"]=="ROLL_EXPIRATION")
        event.update(status=authoritative["status"],premium_cost_usd=authoritative["premium_cost_usd"],fees_usd=authoritative["fees_usd"],broker_state_hash=authoritative["broker_state_hash"])
    flat = not legs
    campaign['adjustment_count']=state.campaign.get('adjustment_count',0)+int(event['status']=='FILLED')
    if flat and action.get('stop_after_fill') and event['status']=='FILLED':
        campaign['stopped_after_exit']=True
    success = flat and not campaign.get("unresolved_stock_obligation") and Decimal(str(campaign["net_cash_usd"])) >= Decimal(str(campaign["target_usd"])) and when <= effective_deadline
    failed = not success and (when >= effective_deadline or flat and action.get("stop_after_fill") and event["status"] == "FILLED")
    if legs:
        fixing=legs[0].fixing_utc_us
        new_job = replace(new_job, signed_legs=legs,fixing_utc_us=fixing,last_trading_utc_us=fixing if fixing!=job.fixing_utc_us else job.last_trading_utc_us,
                          deadline_utc_us=min(fixing,max(when,job.deadline_utc_us)),entry_cash_usd=Decimal(str(campaign.get("gross_cash_usd", campaign["net_cash_usd"]))))
    else:
        new_job=replace(new_job,signed_legs=(),entry_cash_usd=Decimal(str(campaign.get("gross_cash_usd",campaign["net_cash_usd"]))))
    campaign["open_contract_quantities"] = {x.contract_id: x.quantity for x in legs}
    campaign["economically_completed"] = flat
    return ManagedState(new_job, campaign, tuple(quotes.values()), success, failed, (*state.events, event),account_context)


def rollout(initial_state, first_action, policy_selector, pack, *, paths=20_000, seed=20261008, limits=None, max_steps=12_800, deadline=None,checkpoint=None,policy_identity=None):
    successes, failures, unsupported = 0, 0, 0
    stress_events={}
    from .hit_oracle import wilson,atomic_json
    from ..contracts import stable_hash
    from pathlib import Path
    import json
    completed=0
    identity=None
    if checkpoint is not None:
        from ...config import writable_path
        from ..pack_loader import file_hash
        from ..._build_info import SOURCE_HASHES
        if not policy_identity:raise RiskError("POLICY_CHECKPOINT_IDENTITY_REQUIRED")
        checkpoint=writable_path(checkpoint)
        module_file=Path(__file__)
        source_hash=file_hash(module_file) if module_file.is_file() else SOURCE_HASHES.get("src/qqq_sim/risk/offline/control_oracle.py")
        if not source_hash:raise RiskError("IMPLEMENTATION_IDENTITY_UNAVAILABLE")
        identity=stable_hash({"job":initial_state.job.hash,"campaign":initial_state.campaign,"universe":initial_state.universe,"account":fresh_account_limits(initial_state,limits),"action":first_action,"policy":policy_identity,"price_basis":getattr(pack,"basis_identity",getattr(pack,"manifest_hash",None)),"paths":paths,"seed":seed,"max_steps":max_steps,"transition_source":source_hash})
        if checkpoint.exists():
            saved=json.loads(checkpoint.read_text(encoding="utf-8"))
            digest=saved.pop("payload_sha256",None)
            if digest!=stable_hash(saved) or saved.get("identity")!=identity:
                raise RiskError("POLICY_CHECKPOINT_MISMATCH")
            completed=int(saved["completed_paths"])
            successes,failures,unsupported=(int(saved[key]) for key in ("successes","failures","unknown"))
            stress_events=saved["conditional_stress_events"]
            if successes+failures+unsupported!=completed or not 0<=completed<=paths:raise RiskError("POLICY_CHECKPOINT_MISMATCH")
    def save_progress():
        if checkpoint is None:return
        saved={"identity":identity,"completed_paths":completed,"successes":successes,"failures":failures,"unknown":unsupported,"conditional_stress_events":stress_events}
        atomic_json(checkpoint,{**saved,"payload_sha256":stable_hash(saved)})
    for path in range(completed,paths):
        import time
        if deadline is not None and time.monotonic() >= deadline:
            save_progress()
            raise RiskError("RESOURCE_LIMIT", "Policy rollout checkpoint required", completed_paths=path)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(40, path))))
        state, action = initial_state, first_action
        path_stress_events={}
        try:
            for step in range(max_steps):
                if deadline is not None and time.monotonic() >= deadline:
                    raise RiskError("RESOURCE_LIMIT", "Policy rollout checkpoint required", completed_paths=path)
                state = transition(state, action, rng.standard_normal(), pack, limits)
                if state.events and state.events[-1].get("conditional_stress"):
                    name=state.events[-1]["conditional_stress"]
                    path_stress_events[name]=path_stress_events.get(name,0)+1
                if state.success or state.failed:
                    break
                # Future random variates are not exposed to the selector.
                action = policy_selector(state)
            if state.success:
                successes += 1
            elif state.failed:
                failures += 1
            else:
                unsupported += 1
        except RiskError as exc:
            if exc.code == "RESOURCE_LIMIT":
                save_progress()
                raise
            if exc.code in ("NO_FUTURE_ELIGIBLE_EXECUTION", "NO_FEASIBLE_SUPPORTED_ACTION"):
                failures += 1
            else:
                unsupported += 1
        for name,count in path_stress_events.items():stress_events[name]=stress_events.get(name,0)+count
        completed=path+1
        if completed%128==0:save_progress()
    save_progress()
    return {"paths": paths, "successes": successes, "failures": failures, "unknown": unsupported, "probability_lower": successes/paths, "probability_upper": (successes+unsupported)/paths, "wilson_success": wilson(successes, paths), "status": "OK" if not unsupported else "COVERAGE_UNRESOLVED", "policy_event": "REALIZED_NET_TARGET","conditional_stress_events":stress_events}
