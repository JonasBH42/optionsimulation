"""Actual eligible-minute layers and an independent finite-horizon benchmark."""
from __future__ import annotations
import dataclasses
import math
import time
from collections import defaultdict
import numpy as np
from ..contracts import RiskError,stable_hash
from ..policy import generate_actions
from .control_oracle import ManagedState,transition,normal_quadrature,fresh_account_limits
from .hit_oracle import atomic_json


def decision_schedule(job,campaign):
    cutoff=min(campaign["hard_cutoff_utc_us"],campaign.get("active_deadline_utc_us",job.deadline_utc_us))
    times=[]
    for opening,close in job.eligible_intervals:
        start=max(opening,(job.t0_utc_us//60_000_000+1)*60_000_000)
        if start%60_000_000:
            start=((start+59_999_999)//60_000_000)*60_000_000
        end=min(close-1,cutoff)
        if start<=end:
            times.extend(range(start,end+1,60_000_000))
    return np.asarray(sorted(set(times)),dtype=np.int64)


def complete_minute_layers(cases,pack,*,deadline=None,checkpoint_dir=None):
    """Supply a conditional held state for every missing eligible-minute layer.

    Present market states are sampled by exact elapsed-time GBM, with HOLD over
    preceding opportunities. No hidden future normals enter a policy decision.
    Unsupported lifecycle states remain recorded exclusions rather than labels.
    """
    groups=defaultdict(list)
    counts=[]
    for case in cases:
        number=len(decision_schedule(case["job"],case["campaign"]))
        if number:
            groups[number].append(case)
            counts.append((number,case))
    if not counts:
        return [],{"status":"NO_DECISION_LAYERS","layers":[],"excluded":[]}
    maximum=max(number for number,_ in counts)
    sources={}
    for number,case in counts:
        split=case["record"]["split"]
        if split not in sources or number>sources[split][0]:
            sources[split]=(number,case)
    excluded=[]
    for split,(split_maximum,source) in sorted(sources.items()):
        schedule=decision_schedule(source["job"],source["campaign"])
        original=ManagedState(source["job"],source["campaign"],source["universe"],account_context=source.get("account_context"))
        for remaining in range(1,split_maximum+1):
            if any(case["record"]["split"]==split for case in groups[remaining]):
                continue
            if deadline is not None and time.monotonic()>=deadline:
                raise RiskError("RESOURCE_LIMIT","Conditional minute-layer generation checkpoint required",completed_layers=sorted(groups))
            at=int(schedule[split_maximum-remaining-1])
            if at>=source["job"].fixing_utc_us:
                excluded.append({"remaining":remaining,"split":split,"reason":"UNMODELED_EXPIRY_REQUIRES_ROLLED_OR_FLAT_STATE"})
                continue
            seed=np.random.SeedSequence(20261109,spawn_key=(1,17,remaining,int(source["record"]["index"])))
            random=np.random.Generator(np.random.PCG64(seed))
            action={"id":stable_hash([source["job"].hash,remaining,"HOLD_TO_CONDITIONAL_LAYER"]),"family":"HOLD","legs":[],"eligible_utc_us":at}
            try:
                advanced=transition(original,action,random.standard_normal(),pack,original.account_context)
                actual=len(decision_schedule(advanced.job,advanced.campaign))
                if advanced.success or advanced.failed or actual!=remaining:
                    raise RiskError("CONTROL_LAYER_STATE_MISMATCH",expected=remaining,actual=actual)
                record={**source["record"],"index":int(source["record"]["index"]),"conditional_layer":remaining,"split":split}
                derived={**source,"job":advanced.job,"campaign":advanced.campaign,"universe":advanced.universe,"record":record,"account_context":advanced.account_context,"layer_parent":source["job"].hash}
                groups[remaining].append(derived)
                if checkpoint_dir:
                    atomic_json(str(checkpoint_dir)+f"/layer_{remaining:05d}_{split}.json",derived)
            except RiskError as exc:
                if exc.code=="RESOURCE_LIMIT":
                    raise
                excluded.append({"remaining":remaining,"split":split,"reason":exc.code})
    ordered=[case for layer in sorted(groups) for case in groups[layer]]
    return ordered,{"status":"COMPLETE" if not excluded else "PARTIAL_LAYER_COVERAGE","layers":sorted(groups),"maximum_layer":maximum,"states":len(ordered),"excluded":excluded,"clock":"ELIGIBLE_60_SECOND_WITH_ACTUAL_CLOSED_GAP_DURATION"}


def finite_control_benchmark(initial,pack,*,limits=None,max_decisions=3,candidate_cap=4000,raw_cap=80000,node_budget=100000,deadline=None):
    """Independent exhaustive control on a small declared finite test subset.

    The horizon is never shortened to fit the budget. All failed quadrature or
    search budgets return an unavailable benchmark, not an optimistic optimum.
    """
    horizon=len(decision_schedule(initial.job,initial.campaign))
    if horizon>max_decisions:
        return {"status":"BENCHMARK_SUBSET_OUTSIDE_DOMAIN","decision_count":horizon,"supported_max_decisions":max_decisions}
    refinements=[]
    search_diagnostics={"maximum_candidates_returned":0,"maximum_raw_examined":0,"effective_candidate_cap":0,"effective_raw_cap":0}
    previous=None;passes=0
    for quadrature in (17,33,65,129):
        nodes,weights=normal_quadrature(quadrature)
        work=0
        memo={}
        def value(state):
            nonlocal work
            work+=1
            if work>node_budget or deadline is not None and time.monotonic()>=deadline:
                raise RiskError("RESOURCE_LIMIT","Independent control benchmark work cap",nodes=work)
            if state.success:return 1.
            if state.failed:return 0.
            if not len(decision_schedule(state.job,state.campaign)):return 0.
            key=stable_hash([state.job.hash,state.campaign])
            if key in memo:return memo[key]
            search=generate_actions(state.job,state.campaign,state.universe,fresh_account_limits(state,limits),max_candidates=candidate_cap,max_raw=raw_cap,offline_reference=True)
            search_diagnostics["maximum_candidates_returned"]=max(search_diagnostics["maximum_candidates_returned"],len(search["actions"]))
            search_diagnostics["maximum_raw_examined"]=max(search_diagnostics["maximum_raw_examined"],search.get("raw_attempts",0))
            search_diagnostics["effective_candidate_cap"]=search.get("candidate_cap",candidate_cap)
            search_diagnostics["effective_raw_cap"]=search.get("raw_cap",raw_cap)
            if search["truncated"]:
                raise RiskError("CONTROL_BENCHMARK_ACTION_TRUNCATED")
            values=[]
            for action in search["actions"]:
                estimate=math.fsum(float(w)*value(transition(state,action,float(z),pack,state.account_context or limits)) for z,w in zip(nodes,weights))
                values.append(estimate)
            result=max(values,default=0.)
            memo[key]=result
            return result
        try:
            current=value(initial)
        except RiskError as exc:
            return {"status":"BENCHMARK_UNRESOLVED","reason":exc.code,"details":exc.details,"refinements":refinements,"candidate_cap":candidate_cap,"raw_cap":raw_cap,"search_diagnostics":search_diagnostics}
        difference=None if previous is None else abs(current-previous)
        passes=passes+1 if difference is not None and difference<=.0025 else 0
        refinements.append({"quadrature_nodes":quadrature,"value":current,"difference":difference,"tree_nodes":work})
        if passes>=2:
            return {"status":"CONVERGED_FINITE_SUBSET_BENCHMARK","value":current,"refinements":refinements,"decision_count":horizon,"candidate_cap":candidate_cap,"raw_cap":raw_cap,"search_diagnostics":search_diagnostics,"global_optimality_claim":False}
        previous=current
    return {"status":"BENCHMARK_UNRESOLVED","reason":"CONTROL_QUADRATURE_UNRESOLVED","refinements":refinements}
