"""Portfolio Greek vectors from exact European formulas or admitted price cells."""
from __future__ import annotations
import math
import numpy as np
from scipy.special import ndtr
from .contracts import YEAR_US,RiskError


def portfolio_greeks(job,pack,legs=None):
    legs=job.signed_legs if legs is None else legs
    total=np.zeros(4)
    errors=np.zeros(4)
    for leg in legs:
        tau=(leg.fixing_utc_us-job.t0_utc_us)/YEAR_US
        sigma=leg.sigma;s=job.x0;k=float(leg.strike);r=job.rate_cc;q=job.dividend_yield_cc
        if tau<=0 or sigma==0:
            raise RiskError("GREEK_DISCONTINUITY",contract_id=leg.contract_id)
        if leg.exercise_style=="EUROPEAN" or leg.right=="CALL" and q==0 and r>=0:
            d1=(math.log(s/k)+(r-q+sigma*sigma/2)*tau)/(sigma*math.sqrt(tau));d2=d1-sigma*math.sqrt(tau)
            phi=math.exp(-d1*d1/2)/math.sqrt(2*math.pi);dq=math.exp(-q*tau);dr=math.exp(-r*tau)
            sign=1 if leg.right=="CALL" else -1
            delta=sign*dq*float(ndtr(sign*d1))
            gamma=dq*phi/(s*sigma*math.sqrt(tau))
            vega=.01*s*dq*phi*math.sqrt(tau)
            theta=(-s*dq*phi*sigma/(2*math.sqrt(tau))-sign*r*k*dr*float(ndtr(sign*d2))+sign*q*s*dq*float(ndtr(sign*d1)))/365
            value=np.array([delta,gamma,vega,theta]);error=np.zeros(4)
        else:
            if pack.price_basis is None:
                raise RiskError("PRICE_BASIS_MISSING")
            result=pack.price_basis.evaluate(leg.right,s,k,tau,r,q,sigma,greeks=True)
            if "greeks" not in result:
                raise RiskError(result.get("greeks_reason","GREEKS_UNVALIDATED"))
            value=np.array([result["greeks"][key] for key in ("delta","gamma","vega_per_1pct","theta_per_day")])
            error=np.asarray(result["greek_errors"])
        total+=leg.quantity*leg.multiplier*value
        errors+=abs(leg.quantity)*leg.multiplier*error
    if np.any(errors>.01):
        raise RiskError("PORTFOLIO_GREEK_ERROR_BUDGET_EXCEEDED",propagated_errors=errors.tolist())
    return {**dict(zip(("delta_shares","gamma_shares_per_dollar","vega_usd_per_1pct","theta_usd_per_calendar_day"),map(float,total))),"observed_error_envelope":errors.tolist(),"error_interpretation":"MEASURED_NUMERICAL_SENSITIVITY"}
