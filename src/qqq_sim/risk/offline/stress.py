"""Named conditional stress cases; no invented scenario-probability averaging."""
from __future__ import annotations
from dataclasses import replace
from ..contracts import RiskError,canonical
from ..verify import risk_certificate
from decimal import Decimal,ROUND_FLOOR,ROUND_CEILING
import time

REQUIRED_STRESSES=("PATH_IV_0.8","PATH_IV_1.2","LEG_IV_0.8","LEG_IV_1.2","PUT_SKEW_MINUS_0.05","PUT_SKEW_PLUS_0.05","DOUBLE_SPREAD","EXTRA_MINUTE_LATENCY","FAILED_FIRST_LIQUIDATION","SIMULTANEOUS_ASSIGNMENT","DELAYED_ASSIGNMENT_NOTICE","PARTIAL_RATIO_FILL")


def stress_cases(case):
    import math
    job,campaign=case["job"],case["campaign"]
    output=[]
    for name in REQUIRED_STRESSES:
        modified_job,modified_campaign=job,dict(campaign)
        try:
            if name.startswith("PATH_IV_"):
                modified_job=replace(job,sigma_path=job.sigma_path*float(name.rsplit("_",1)[1]))
            elif name.startswith("LEG_IV_"):
                factor=float(name.rsplit("_",1)[1])
                modified_job=replace(job,signed_legs=tuple(replace(x,sigma=x.sigma*factor,bid=None,ask=None) for x in job.signed_legs))
            elif name.startswith("PUT_SKEW"):
                sign=-1 if "MINUS" in name else 1
                modified_job=replace(job,signed_legs=tuple(replace(x,sigma=x.sigma+sign*.05*min(1,max(0,-math.log(float(x.strike)/job.x0)/.1)),bid=None,ask=None) if x.right=="PUT" else x for x in job.signed_legs))
            elif name=="DOUBLE_SPREAD":
                modified_job=replace(job,signed_legs=tuple(replace(x,half_spread=x.half_spread*2,bid=None,ask=None) for x in job.signed_legs))
            elif name=="EXTRA_MINUTE_LATENCY":
                modified_campaign["extra_latency_seconds"]=60
            elif name=="FAILED_FIRST_LIQUIDATION":
                modified_campaign["forced_first_liquidation_nonfill"]=True
            elif name=="DELAYED_ASSIGNMENT_NOTICE":
                modified_campaign["operational_model"]="DELAYED_NOTICE_STRESS"
            elif name=="PARTIAL_RATIO_FILL":
                modified_campaign["partial_ratio_fill_stress"]=True
            certificate=risk_certificate(modified_job.signed_legs,modified_campaign)
            output.append({"name":name,"job":modified_job,"campaign":modified_campaign,"certificate":certificate,"scenario_weights":None})
        except RiskError as exc:
            output.append({"name":name,"status":"STRESS_OUTSIDE_DOMAIN","reason":exc.code,"scenario_weights":None})
    return output


def stress_report(cases):
    reports=[]
    violations=0
    for case in cases:
        for scenario in stress_cases(case):
            entry={"case":case["job"].snapshot_id,"name":scenario["name"],"probability":None,"probability_reason":"STRESS_PROBABILITY_REQUIRES_SEPARATE_COVERAGE","scenario_weights":None}
            if "certificate" in scenario:
                entry["certificate"]=scenario["certificate"]
                # Delayed notice must explicitly withdraw the simultaneous model.
                if scenario["name"]=="DELAYED_ASSIGNMENT_NOTICE" and scenario["certificate"]["status"]=="VERIFIED_WITHIN_DECLARED_MODEL":
                    violations+=1
            else:
                entry.update(status=scenario["status"],reason=scenario["reason"])
            reports.append(entry)
    return {"required_scenarios":REQUIRED_STRESSES,"scenario_count":len(reports),"cases":reports,"accepted_known_constraint_violations":violations,"all_required_scenarios_exercised":bool(cases) and set(x["name"] for x in reports)==set(REQUIRED_STRESSES),"probability_stresses_qualified":False,"interpretation":"CONDITIONAL_SCENARIOS_NOT_PROBABILITY_WEIGHTED"}


def stress_policy_report(cases,selector,pack,*,paths=20000,limits=None,deadline=None,checkpoint_dir=None,policy_identity=None):
    """Independently execute each named conditional scenario, without mixing weights.

    Assignment notice/delivery needs a forecast lifecycle transition model; its
    absence remains an explicit unknown probability for that scenario. P50-M15
    permits risk results when stressed probability is outside modeled coverage.
    """
    from .control_oracle import ManagedState,rollout
    from ..features import price_contract
    from ..contracts import YEAR_US
    reports=[]
    for case in cases:
        for number,scenario in enumerate(stress_cases(case)):
            if deadline is not None and time.monotonic()>=deadline:
                raise RiskError("RESOURCE_LIMIT","Conditional stress rollout budget exhausted",completed_scenarios=len(reports))
            item={"case":case["job"].snapshot_id,"name":scenario["name"],"scenario_weights":None,"probability":None}
            if scenario["name"] in ("SIMULTANEOUS_ASSIGNMENT","DELAYED_ASSIGNMENT_NOTICE"):
                item.update(status="STRESS_LIFECYCLE_UNSUPPORTED",reason="PHYSICAL_DELIVERY_FORECAST_TRANSITION_UNAVAILABLE",certificate=scenario.get("certificate"))
                reports.append(item)
                continue
            if "job" not in scenario:
                reports.append({**item,"status":scenario["status"],"reason":scenario["reason"]})
                continue
            job,campaign=scenario["job"],scenario["campaign"]
            try:
                import math
                refreshed=[]
                all_legs={leg.contract_id:leg for leg in (*case["universe"],*job.signed_legs)}
                held={leg.contract_id:leg for leg in job.signed_legs}
                for leg in all_legs.values():
                    if leg.contract_id not in held:
                        if scenario["name"].startswith("LEG_IV_"):
                            leg=replace(leg,sigma=leg.sigma*float(scenario["name"].rsplit("_",1)[1]))
                        elif scenario["name"].startswith("PUT_SKEW") and leg.right=="PUT":
                            sign=-1 if "MINUS" in scenario["name"] else 1
                            leg=replace(leg,sigma=leg.sigma+sign*.05*min(1,max(0,-math.log(float(leg.strike)/job.x0)/.1)))
                        elif scenario["name"]=="DOUBLE_SPREAD":
                            leg=replace(leg,half_spread=2*leg.half_spread)
                    value=Decimal(str(float(price_contract(leg,job.x0,(leg.fixing_utc_us-job.t0_utc_us)/YEAR_US,job,pack))))
                    tick,half=Decimal(str(leg.tick)),Decimal(str(leg.half_spread))
                    bid=(max(Decimal(0),value-half)/tick).to_integral_value(rounding=ROUND_FLOOR)*tick
                    ask=((value+half)/tick).to_integral_value(rounding=ROUND_CEILING)*tick
                    refreshed.append(replace(leg,bid=float(bid),ask=float(ask),known_at_utc_us=job.t0_utc_us))
                mapping={leg.contract_id:leg for leg in refreshed}
                job=replace(job,signed_legs=tuple(replace(mapping[leg.contract_id],quantity=leg.quantity) for leg in job.signed_legs))
                # Existing original fills and target remain frozen. Changes in
                # current quotes are scenario observations, never rewritten fills.
                state=ManagedState(job,campaign,tuple(refreshed),account_context=case.get("account_context"))
                first=selector(state)
                from pathlib import Path
                from ..contracts import stable_hash
                checkpoint=None if checkpoint_dir is None else Path(checkpoint_dir)/(stable_hash([job.hash,scenario["name"]])+".json")
                result=rollout(state,first,selector,pack,paths=paths,seed=20261120+number,limits=limits,deadline=deadline,checkpoint=checkpoint,policy_identity=policy_identity)
                item.update(status=result["status"],probability=result["probability_lower"] if result["status"]=="OK" else None,rollout=result)
            except RiskError as exc:
                if exc.code=="RESOURCE_LIMIT":raise
                item.update(status="STRESS_COVERAGE_UNRESOLVED",reason=exc.code)
            reports.append(item)
    qualified=bool(reports) and paths>=20000 and all(x["status"]=="OK" for x in reports)
    classified=bool(reports) and set(item["name"] for item in reports)==set(REQUIRED_STRESSES) and all(item["status"] in ("OK","COVERAGE_UNRESOLVED","STRESS_OUTSIDE_DOMAIN","STRESS_COVERAGE_UNRESOLVED","STRESS_LIFECYCLE_UNSUPPORTED") for item in reports)
    return {"cases":reports,"paths_per_scenario":paths,"probability_stresses_qualified":qualified,"all_required_stress_outcomes_classified":classified,"independent_streams":True,"scenario_weights":None,"interpretation":"SEPARATE_CONDITIONAL_SCENARIOS_NOT_AVERAGED"}
