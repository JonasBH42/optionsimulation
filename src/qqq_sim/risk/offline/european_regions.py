"""Adaptive European target topology with conservative derivative bounds."""
from __future__ import annotations
import math
import time
import numpy as np
from scipy.special import ndtr
from ..contracts import RiskError
from ..payoff import Interval,merge_intervals,piecewise_regions


def portfolio_lipschitz(job,lower,upper,tau):
    """Bound the derivative interval, including negative quantities/rates/carry.

    Every European option delta increases with its state variable. Summing its
    signed delta intervals tightens the specified global sum-of-absolute bound
    without assuming that different legs attain extrema at the same point.
    """
    lower,upper=np.asarray(lower),np.asarray(upper)
    dlow=np.zeros_like(lower);dhigh=np.zeros_like(upper)
    state_discount=math.exp(-(job.rate_cc if job.state_kind=="TERMINAL_FORWARD" else job.dividend_yield_cc)*tau)
    for leg in job.signed_legs:
        factor=leg.quantity*leg.multiplier
        eta=1 if leg.right=="CALL" else -1
        deltas=[]
        for spot in (lower,upper):
            forward=spot if job.state_kind=="TERMINAL_FORWARD" else spot*math.exp((job.rate_cc-job.dividend_yield_cc)*tau)
            if leg.sigma==0 or tau==0:
                delta=np.where(forward>float(leg.strike),1.,0.) if eta==1 else np.where(forward<float(leg.strike),-1.,0.)
            else:
                vol=leg.sigma*math.sqrt(tau)
                d1=np.log(forward/float(leg.strike))/vol+.5*vol
                delta=ndtr(d1) if eta==1 else -ndtr(-d1)
            deltas.append(factor*state_discount*delta)
        dlow+=np.minimum(*deltas)
        dhigh+=np.maximum(*deltas)
    global_bound=sum(abs(x.quantity)*x.multiplier for x in job.signed_legs)*math.exp(max(-(job.rate_cc if job.state_kind=="TERMINAL_FORWARD" else job.dividend_yield_cc),0)*tau)
    return np.minimum(global_bound,np.maximum(np.abs(dlow),np.abs(dhigh))+global_bound*1e-14)


def adaptive_regions(value,knots,lipschitz,*,error=0.,deadline=None,max_evaluations=2_000_000):
    """Return nominal/certain/possible unions; unresolved contacts remain possible.

    The callback is f(S)=portfolio mark minus the unchanged dollar target. A
    Lipschitz or roundoff budget never authorizes dropping an unresolved island.
    """
    points=np.asarray(sorted(set(map(float,knots))),dtype=np.float64)
    if len(points)<2 or np.any(np.diff(points)<=0):
        raise RiskError("INVALID_REGION_GRID")
    lower,upper=points[:-1],points[1:]
    certain=[];possible=[];nominal=[]
    evaluations=0;unresolved=0;depth=0
    while len(lower):
        if deadline is not None and time.monotonic()>deadline:
            raise RiskError("RESOURCE_LIMIT","European target subdivision work budget exceeded",evaluations=evaluations)
        evaluations+=len(lower)
        if evaluations>max_evaluations:
            raise RiskError("RESOURCE_LIMIT","European target topology remains unresolved at the allocation limit",evaluations=evaluations)
        middle=(lower+upper)/2
        f=np.asarray(value(middle),dtype=np.float64)
        bound=np.asarray(lipschitz(lower,upper) if callable(lipschitz) else lipschitz,dtype=np.float64)
        error_values=np.asarray(error(middle) if callable(error) else error,dtype=np.float64)
        if not np.all(np.isfinite(f)) or np.any(bound<0) or not np.all(np.isfinite(bound)):
            raise RiskError("MODEL_MARKS_INVALID")
        allowance=bound*(upper-lower)/2+error_values
        inside=f-allowance>0
        outside=f+allowance<0
        for a,b in zip(lower[inside],upper[inside]):
            interval=Interval(float(a),float(b))
            certain.append(interval);possible.append(interval);nominal.append(interval)
        unknown=~(inside|outside)
        tolerance=np.maximum(1e-8,1e-10*middle)
        small=unknown&((upper-lower<=tolerance)|(middle==lower)|(middle==upper))
        if np.any(small):
            a,m,b=lower[small],middle[small],upper[small]
            fa,fb=np.asarray(value(a)),np.asarray(value(b))
            evaluations+=2*len(a)
            unresolved+=len(a)
            for left,mid,right,vl,vm,vr in zip(a,m,b,fa,f[small],fb):
                possible.append(Interval(float(left),float(right)))
                nominal.extend(piecewise_regions([left,mid,right],[vl,vm,vr]))
        split=unknown&~small
        a,m,b=lower[split],middle[split],upper[split]
        lower=np.concatenate((a,m));upper=np.concatenate((m,b))
        # Merging periodically keeps long certified runs from growing in RAM.
        if depth%8==7:
            certain=list(merge_intervals(certain));possible=list(merge_intervals(possible));nominal=list(merge_intervals(nominal))
        depth+=1
    return {"regions":merge_intervals(nominal),"certain":merge_intervals(certain),"possible":merge_intervals(possible),
            "evaluations":evaluations,"unresolved_tolerance_intervals":unresolved,"subdivision_depth":depth,
            "method":"EUROPEAN_ANALYTIC_LIPSCHITZ_SUBDIVISION"}
