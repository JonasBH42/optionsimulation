"""Fast survival inference. This module never imports offline numerical code."""
from __future__ import annotations
import math
import time
import numpy as np
from .contracts import RiskError, YEAR_US, stable_hash
from .features import hold_features, exact_support, coverage_check, price_contract


def silu(x):
    return x*np.exp(-np.logaddexp(0, -x))


def network(arrays, prefix, inputs):
    x = np.asarray(inputs, dtype=np.float64)
    for index in range(4):
        wk, bk = f"{prefix}_w{index}", f"{prefix}_b{index}"
        if wk not in arrays or bk not in arrays:
            raise RiskError("PACK_NETWORK_MISSING")
        x = x@arrays[wk]+arrays[bk]
        if index < 3:
            x = silu(x)
    return x


def survival_bins(job, seconds):
    """At most 18 bins per session, exact requested observations included."""
    bins = []
    step = int(seconds*1_000_000)
    for start, close in job.eligible_intervals:
        lo, hi = max(start, job.t0_utc_us), min(close, job.last_trading_utc_us, job.fixing_utc_us)
        if lo >= hi:
            bins.extend([np.empty(0, dtype=np.int64) for _ in range(18)])
            continue
        observations = set(range(((lo+step-1)//step)*step, hi, step))
        if start >= job.t0_utc_us:
            observations.add(start)
        if lo <= job.t0_utc_us < hi:
            observations.add(job.t0_utc_us)
        observations.add(hi-1)
        observations.update(t for t in job.reports if lo <= t < hi)
        values = np.array(sorted(observations), dtype=np.int64)
        if len(values) == 1:
            bins.append(values)
            bins.extend([np.empty(0, dtype=np.int64) for _ in range(17)])
        else:
            bins.append(values[:1])
            bins.extend(np.array_split(values[1:-1], 16))
            bins.append(values[-1:])
    bins.extend([np.empty(0, dtype=np.int64) for _ in range(576-len(bins))])
    return bins


def survival_cdf(hazards, bins, instant):
    amount = math.fsum(float(h)*int(np.searchsorted(observations, instant, side="right"))/len(observations) for h, observations in zip(hazards, bins) if len(observations))
    return -math.expm1(-amount)


def evaluate_hold(job, pack=None):
    started = time.perf_counter()
    base = {"event": "UNCHANGED_THEORETICAL_FIRST_HIT", "target_usd": str(job.target_usd), "target_basis": job.target_basis, "job_hash": job.hash, "pack_id": getattr(pack, "pack_id", None), "hold_p50": None}
    try:
        if not job.signed_legs:
            raise RiskError("NO_OPEN_POSITION_HOLD_EVENT")
        if pack is None:
            raise RiskError("PACK_MISSING")
        pack.require_qualified("hold")
        exact_support(job)
        tau = (job.fixing_utc_us-job.t0_utc_us)/YEAR_US
        from .features import validated_leg_mark
        eps_port=min(.005,1e-4*max(1,abs(float(job.target_usd))))
        eps_leg=eps_port/max(1,sum(abs(x.quantity)*x.multiplier for x in job.signed_legs))
        residuals=[]
        for leg in job.signed_legs:
            if leg.bid is not None:
                residuals.append({"contract_id":leg.contract_id,**validated_leg_mark(leg,job.x0,tau,job.rate_cc,job.dividend_yield_cc,pack,job.state_kind,eps_leg)})
        base["per_leg_model_checks"]=residuals
        now = float(job.entry_cash_usd)+sum(x.quantity*x.multiplier*float(price_contract(x, job.x0, tau, job, pack)) for x in job.signed_legs)
        if any(a <= job.t0_utc_us < b for a, b in job.eligible_intervals) and job.t0_utc_us < job.last_trading_utc_us and now >= float(job.target_usd):
            base.update(status="OK", hold_p50=1., closing_only=1., terminal_inclusive_reference=1., empirical_error_range=[1., 1.], exact_reason="CURRENT_ELIGIBLE_MODEL_TARGET_ALREADY_MET")
            return base
        features = hold_features(job, pack)
        from .qualification import regime_key
        scaled, coverage = coverage_check(features, pack, regime=regime_key(job,features=features))
        output = network(pack.arrays, "hold", scaled)
        if output.shape != (579,):
            raise RiskError("PACK_HEAD_SCHEMA_MISMATCH")
        hazards = np.logaddexp(0, output[:576])
        companion = np.exp(-np.logaddexp(0, -output[576:]))
        bins = survival_bins(job, pack.manifest["monitoring_seconds"])
        close = survival_cdf(hazards, bins, job.fixing_utc_us)
        deadline = survival_cdf(hazards, bins, job.deadline_utc_us)
        union = close+(1-close)*float(companion[2])
        point = union if job.deadline_utc_us == job.fixing_utc_us else deadline
        e = coverage["p99_absolute_error"]+coverage.get("reference_uncertainty", 0)
        base.update(status="OK", hold_p50=point, closing_only=close, terminal_inclusive_reference=union, settlement_only=(1-close)*float(companion[2]), terminal_cent_pop=float(companion[0]), terminal_target_probability=float(companion[1]), deadline_endpoint=None, deadline_endpoint_reason="ENDPOINT_HEAD_NOT_QUALIFIED", empirical_error_range=[max(0, point-e), min(1, point+e)], uncertainty_interpretation="EMPIRICAL_ERROR_RANGE_NOT_MARKET_CONFIDENCE", curve=[{"deadline_utc_us": t, "closing_first_hit": survival_cdf(hazards, bins, t)} for t in job.reports], grid_hash=stable_hash([x.tolist() for x in bins]), coverage=coverage)
    except RiskError as exc:
        base.update(exc.as_dict())
    base["elapsed_seconds"] = time.perf_counter()-started
    base["offline_calls"] = 0
    return base
