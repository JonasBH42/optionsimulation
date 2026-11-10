"""Empirical regime admission. Counts, errors and geometry are all mandatory."""
from __future__ import annotations
import math
import numpy as np
from .contracts import YEAR_US, RiskError


def regime_key(job, campaign=None,*,features=None):
    campaign=campaign or {}
    hours=(job.fixing_utc_us-job.t0_utc_us)/3_600_000_000
    time_bucket=next((label for ceiling,label in ((1,"lt1h"),(24,"1h_1d"),(96,"1_4d"),(240,"4_10d"),(721,"10_30d")) if hours<ceiling),"outside_time")
    sigma_bucket=next((label for ceiling,label in ((.15,".03_.15"),(.30,".15_.30"),(.60,".30_.60"),(2.00000001,".60_2")) if job.sigma_path<ceiling),"outside_sigma")
    legs=job.signed_legs
    family="vertical" if len(legs)==2 else "butterfly" if len(legs)==3 else "condor" if len(legs)==4 and len({x.right for x in legs})==2 else "asymmetric"
    if campaign.get("adjustment_count",0)>0:
        family="ongoing_"+family
    scale=max(float(job.target_usd),float(campaign.get("loss_cap_usd",1000)),1)
    liquidation=float(campaign.get("net_cash_usd",job.entry_cash_usd))
    quote_complete=all(x.bid is not None and x.ask is not None for x in legs)
    if quote_complete:
        liquidation+=sum(x.quantity*x.multiplier*(x.bid if x.quantity>0 else x.ask) for x in legs)
    near="near" if quote_complete and abs(liquidation-float(job.target_usd))<=.1*scale else "far"
    fraction=float(campaign.get("remaining_loss_budget_usd",1000))/float(campaign.get("loss_cap_usd",1000))
    budget="0_.1" if fraction<=.1 else ".1_.5" if fraction<=.5 else ".5_1"
    session="closed" if not any(a<=job.t0_utc_us<b for a,b in job.eligible_intervals) else "open"
    from .features import feature_missingness_key
    pattern='UNSPECIFIED' if features is None else feature_missingness_key(features)
    return ":".join((time_bucket,sigma_bucket,family,near,budget,session,'features='+pattern))


def conservative_case_error(case):
    """Auxiliary curves supply an already uncertainty-inclusive case error."""
    error=abs(float(case["prediction"])-float(case["reference"]))+float(case.get("reference_uncertainty",0))
    auxiliary=float(case.get("auxiliary_error",0))
    if not math.isfinite(error) or not math.isfinite(auxiliary) or auxiliary<0 or float(case.get("reference_uncertainty",0))<0:
        raise RiskError("QUALIFICATION_CASE_ERROR_INVALID")
    return max(error,auxiliary)


def qualify_regimes(training, training_keys, calibration, validation, *, tolerance, minimum=200):
    """Cases carry scaled encoding, prediction, label and reference uncertainty.

    No outcome-based merging: the initial adjacency rule is no merging. This
    deliberately excludes sparse regimes instead of hiding a failing subgroup.
    """
    training=np.asarray(training,dtype=np.float64)
    regimes={}
    allkeys=sorted(set(x["regime"] for x in calibration+validation))
    for key in allkeys:
        indices=[i for i,k in enumerate(training_keys) if k==key]
        cal=[x for x in calibration if x["regime"]==key]
        val=[x for x in validation if x["regime"]==key]
        reason=[]
        if len(indices)<8: reason.append("FEWER_THAN_EIGHT_TRAINING_NEIGHBORS")
        if len(cal)<minimum: reason.append("INSUFFICIENT_CALIBRATION_CASES")
        if len(val)<minimum: reason.append("INSUFFICIENT_INDEPENDENT_VALIDATION_CASES")
        residuals=np.array([conservative_case_error(x) for x in val])
        pass_fraction=float(np.mean(residuals<=tolerance)) if len(residuals) else 0.
        if pass_fraction<.99: reason.append("ACCURACY_TARGET_FAILED")
        if any(not x.get("reference_qualified",False) for x in cal+val): reason.append("UNQUALIFIED_REFERENCE_LABEL")
        distances=[]
        if len(indices)>=8:
            local=training[indices]
            for item in cal:
                difference=local-np.asarray(item["encoding"])
                squared=np.einsum("ij,ij->i",difference,difference)
                distances.append(math.sqrt(float(np.partition(squared,7)[7])/training.shape[1]))
        error=float(np.quantile(residuals,.99)) if len(residuals) else None
        maximum=float(np.max(residuals)) if len(residuals) else None
        regimes[key]={"qualified":not reason,"reasons":reason,"training_indices":indices,"calibration_count":len(cal),"independent_cases":len(val),"pass_fraction":pass_fraction,"p99_absolute_error":error,"maximum_absolute_error":maximum,"reference_uncertainty":0.,"eighth_neighbor_distance_99":float(np.quantile(distances,.99)) if distances else None,"tolerance":tolerance,"merging_rule":"NO_MERGING_V1","error_basis":"MAX_HEADLINE_AND_AUXILIARY_CASE_ERROR_INCLUDING_REFERENCE_UNCERTAINTY"}
    admitted={k:v for k,v in regimes.items() if v["qualified"]}
    return {"qualified":bool(admitted),"regimes":regimes,"admitted_regimes":sorted(admitted),"excluded_regimes":sorted(set(regimes)-set(admitted)),"independent_cases":sum(v["independent_cases"] for v in admitted.values()),"pass_fraction":min((v["pass_fraction"] for v in admitted.values()),default=0.),"calibration_count":sum(v["calibration_count"] for v in admitted.values()),"reason":None if admitted else "NO_REGIME_PASSED_ALL_COUNT_ACCURACY_AND_REFERENCE_GATES"}
