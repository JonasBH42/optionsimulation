"""Fixed float64 features; original exact state remains authoritative."""
from __future__ import annotations
import math
from dataclasses import replace
import numpy as np
from .contracts import YEAR_US, RiskError
from .payoff import payoff_array, payoff_extrema
from .pricing_fast import european_price, deterministic_american

from .feature_layout import (FEATURE_SCHEMA,TOKEN_FIELDS,HOLD_FEATURE_COUNT,MANAGED_STATE_COUNT,
    ACCOUNT_FIELDS,MISSING_SHOCK_SENTINEL,SHOCK_MASK_OFFSET,SHOCK_MASK_COUNT,ACCOUNT_MASK_OFFSET,CAMPAIGN_OFFSET)


def validated_leg_mark(leg,spot,tau,r,q,pack=None,state_kind="SPOT",eps_leg=1e-5):
    """Validate one leg's IV against its own quote; errors cannot cancel."""
    if tau<=0 or leg.sigma<=0:
        raise RiskError("LEG_IV_NOT_IDENTIFIABLE",contract_id=leg.contract_id)
    analytic=leg.exercise_style=="EUROPEAN" or leg.right=="CALL" and q==0 and r>=0
    if analytic:
        price=float(european_price(leg.right,spot,float(leg.strike),tau,r,q,leg.sigma,state_kind))
        d1=(math.log(spot/float(leg.strike))+(r-q+leg.sigma**2/2)*tau)/(leg.sigma*math.sqrt(tau))
        vega=.01*spot*math.exp(-q*tau)*math.exp(-d1*d1/2)/math.sqrt(2*math.pi)*math.sqrt(tau)
        result={"price":price,"error":0.,"vega_per_1pct":vega,"source":"EXACT_ANALYTICAL_MARK"}
    else:
        if pack is None or getattr(pack,"price_basis",None) is None:
            raise RiskError("PRICE_BASIS_MISSING",contract_id=leg.contract_id)
        if not hasattr(pack,"require_qualified"):
            raise RiskError("PACK_PRICE_VALIDATION_MISSING")
        pack.require_qualified("price")
        valued=pack.price_basis.evaluate(leg.right,spot,float(leg.strike),tau,r,q,leg.sigma,greeks=True)
        result={"price":valued["price"],"error":valued["error"],"vega_per_1pct":valued.get("greeks",{}).get("vega_per_1pct"),"source":"VALIDATED_AMERICAN_PRICE_BASIS"}
    if result["error"]>eps_leg:
        raise RiskError("PRICE_BASIS_ERROR_BUDGET_EXCEEDED",contract_id=leg.contract_id,error=result["error"],budget=eps_leg)
    if leg.bid is None or leg.ask is None:
        raise RiskError("MISSING_EXECUTABLE_QUOTE",contract_id=leg.contract_id)
    midpoint=(leg.bid+leg.ask)/2
    result["model_quote_residual"]=result["price"]-midpoint
    if abs(result["model_quote_residual"])>max((leg.ask-leg.bid)/2,eps_leg)+result["error"]:
        raise RiskError("MODEL_BASIS_TOO_LARGE",contract_id=leg.contract_id,model_price=result["price"],midpoint=midpoint)
    return result


def price_contract(leg, spot, tau, job, pack, sigma=None):
    sigma = leg.sigma if sigma is None else sigma
    if tau == 0 or sigma == 0:
        fn = deterministic_american if leg.exercise_style == "AMERICAN" else european_price
        return fn(leg.right, spot, float(leg.strike), tau, job.rate_cc, job.dividend_yield_cc, *(() if fn is deterministic_american else (sigma,)))
    if leg.exercise_style == "EUROPEAN" or leg.right == "CALL" and job.dividend_yield_cc == 0 and job.rate_cc >= 0:
        return european_price(leg.right, spot, float(leg.strike), tau, job.rate_cc, job.dividend_yield_cc, sigma, job.state_kind)
    if getattr(pack,'price_basis',None) is None:
        raise RiskError("PRICE_BASIS_MISSING")
    values = [pack.price_basis.evaluate(leg.right, float(s), float(leg.strike), tau, job.rate_cc, job.dividend_yield_cc, sigma)["price"] for s in np.atleast_1d(spot)]
    return np.asarray(values) if np.ndim(spot) else values[0]


def position_token(job,leg):
    tau=(leg.fixing_utc_us-job.t0_utc_us)/YEAR_US
    last=leg.last_trading_utc_us
    if last is None:
        if leg.fixing_utc_us!=job.fixing_utc_us:
            raise RiskError('LAST_TRADING_TIME_REQUIRED',contract_id=leg.contract_id)
        last=job.last_trading_utc_us
    return [1,math.log(float(leg.strike)/job.x0),tau,leg.quantity/20,1 if leg.right=='CALL' else -1,
            leg.sigma,leg.multiplier/100,leg.half_spread/job.x0,leg.tick/job.x0,
            float(leg.exercise_style=='AMERICAN'),(last-job.t0_utc_us)/YEAR_US,tau]


def _shock_values(leg,spots,tau,job,pack,factor):
    # Only unsupported shock coordinates become missing. Corrupt prices,
    # violated inequalities or stale packs remain hard failures.
    missing_codes={'PRICE_BASIS_OUTSIDE_DOMAIN','PRICE_BASIS_UNVALIDATED_CELL','PRICE_BASIS_MISSING'}
    try:
        values=price_contract(leg,spots,tau,job,pack,leg.sigma*factor)
        return np.asarray(values,dtype=np.float64),np.zeros(len(spots),dtype=bool)
    except RiskError as error:
        if error.code not in missing_codes:raise
    values=np.zeros(len(spots));missing=np.zeros(len(spots),dtype=bool)
    for index,spot in enumerate(spots):
        try:values[index]=price_contract(leg,spot,tau,job,pack,leg.sigma*factor)
        except RiskError as error:
            if error.code not in missing_codes:raise
            missing[index]=True
    return values,missing


def hold_features(job, pack, campaign=None):
    if len(job.signed_legs) > 8 or len(job.eligible_intervals) > 32 or len(job.reports) > 32:
        raise RiskError("OUTSIDE_MODEL_DOMAIN", "Feature token count exceeded")
    campaign = campaign or {}
    scale = max(float(job.target_usd), float(campaign.get("loss_cap_usd", 1000)), 1.)
    tau = (job.fixing_utc_us-job.t0_utc_us)/YEAR_US
    tokens = np.zeros((8, 12), dtype=np.float64)
    ordered = sorted(job.signed_legs, key=lambda x: (x.fixing_utc_us, x.right, x.strike, x.contract_id))
    for i, leg in enumerate(ordered):
        tokens[i] = position_token(job,leg)
    spots = job.x0*np.exp(np.linspace(-math.log(2), math.log(2), 65))
    marks = np.full(65, float(job.entry_cash_usd))
    liquid = np.full(65, float(campaign.get("net_cash_usd", job.entry_cash_usd)))
    low, high = marks.copy(), marks.copy()
    low_missing=np.zeros(65,dtype=bool);high_missing=low_missing.copy()
    close_quotes=[]
    for leg in job.signed_legs:
        leg_tau=(leg.fixing_utc_us-job.t0_utc_us)/YEAR_US
        value = price_contract(leg, spots, leg_tau, job, pack)
        marks += leg.quantity*leg.multiplier*value
        bid=np.floor(np.maximum(0,value-leg.half_spread)/leg.tick)*leg.tick
        ask=np.ceil((value+leg.half_spread)/leg.tick)*leg.tick
        side=bid if leg.quantity>0 else ask
        liquid += leg.quantity*leg.multiplier*side
        close_quotes.append((leg,bid,ask))
        shocked,missing=_shock_values(leg,spots,leg_tau,job,pack,.8)
        low+=leg.quantity*leg.multiplier*shocked;low_missing|=missing
        shocked,missing=_shock_values(leg,spots,leg_tau,job,pack,1.2)
        high+=leg.quantity*leg.multiplier*shocked;high_missing|=missing
    if close_quotes:
        from .policy import action_fees
        for index in range(65):
            changes=tuple(replace(leg,quantity=-leg.quantity,bid=float(bid[index]),ask=float(ask[index])) for leg,bid,ask in close_quotes)
            liquid[index]-=float(action_fees(job,changes))
    terminal = payoff_array(job.signed_legs, spots, job.entry_cash_usd)
    terminal_net = terminal-float(job.entry_cash_usd)+float(campaign.get("net_cash_usd", job.entry_cash_usd))
    shape = np.concatenate((marks, liquid, terminal, terminal_net, low, high))/scale
    shape[4*65:5*65][low_missing]=MISSING_SHOCK_SENTINEL
    shape[5*65:6*65][high_missing]=MISSING_SHOCK_SENTINEL
    shock_masks=np.concatenate((low_missing,high_missing)).astype(np.float64)
    kinks=np.zeros((8,4))
    for index,strike in enumerate(sorted({leg.strike for leg in ordered})):
        gross=float(payoff_array(job.signed_legs,float(strike),job.entry_cash_usd))
        net=gross-float(job.entry_cash_usd)+float(campaign.get('net_cash_usd',job.entry_cash_usd))
        kinks[index]=[1,math.log(float(strike)/job.x0),gross/scale,net/scale]
    sessions = np.zeros((32, 6))
    previous = job.t0_utc_us
    for i, (a, b) in enumerate(job.eligible_intervals):
        sessions[i] = [1, (a-job.t0_utc_us)/YEAR_US, (b-job.t0_utc_us)/YEAR_US, (a-previous)/YEAR_US, (b-a)/60_000_000, float(a <= job.fixing_utc_us <= b)]
        previous = b
    requested = np.zeros((32, 3))
    for i, when in enumerate(job.reports):
        requested[i] = [1, (when-job.t0_utc_us)/YEAR_US, float(when < job.last_trading_utc_us and any(a <= when < b for a, b in job.eligible_intervals))]
    extrema = payoff_extrema(job.signed_legs, job.entry_cash_usd)
    globals_ = np.array([job.x0/1000, job.rate_cc, job.dividend_yield_cc, job.sigma_path, tau, float(job.entry_cash_usd)/scale, float(job.target_usd)/scale, float(campaign.get("net_cash_usd", job.entry_cash_usd))/scale, float(campaign.get("remaining_expense_budget_usd", 100))/scale, *extrema["tail_slopes"], len(ordered)/8], dtype=np.float64)
    gaps=np.array([marks[32]-float(job.target_usd),liquid[32]-float(job.target_usd)])/scale
    result = np.concatenate((globals_, tokens.ravel(), shape, sessions.ravel(), requested.ravel(),shock_masks,kinks.ravel(),gaps))
    if result.shape!=(HOLD_FEATURE_COUNT,):raise RiskError('FEATURE_SCHEMA_MISMATCH')
    if not np.all(np.isfinite(result)):
        raise RiskError("INVALID_FEATURE_VALUE")
    return result


def managed_state_features(job,pack,campaign):
    state=hold_features(job,pack,campaign)
    record=campaign.get('account_features',{})
    values=record.get('values',{})
    scale=max(float(job.target_usd),float(campaign.get('loss_cap_usd',1000)),1.)
    account=np.zeros(len(ACCOUNT_FIELDS));known=np.zeros(len(ACCOUNT_FIELDS))
    for index,name in enumerate(ACCOUNT_FIELDS):
        if values.get(name) is not None:
            account[index]=float(values[name])/scale;known[index]=1.
    def clock(name,fallback):return (int(campaign.get(name,fallback))-job.t0_utc_us)/YEAR_US
    initial=campaign.get('initial_fill_utc_us')
    clocks=[clock('initial_deadline_utc_us',job.deadline_utc_us),clock('active_deadline_utc_us',job.deadline_utc_us),
            clock('hard_cutoff_utc_us',job.fixing_utc_us),0. if initial is None else (int(initial)-job.t0_utc_us)/YEAR_US,
            campaign.get('remaining_actions',12)/12,campaign.get('remaining_rolls',3)/3,campaign.get('adjustment_count',0)/12,
            float(initial is not None),float(bool(campaign.get('stopped') or campaign.get('stopped_after_exit'))),
            float(bool(campaign.get('economically_completed'))),float(bool(campaign.get('unresolved_stock_obligation'))),
            float(bool(campaign.get('protection_instruction_recorded')))]
    result=np.concatenate((state,account,known,clocks))
    if result.shape!=(MANAGED_STATE_COUNT,) or not np.all(np.isfinite(result)):raise RiskError('INVALID_FEATURE_VALUE')
    return result


def feature_missingness_key(features):
    from .contracts import stable_hash
    data=np.asarray(features)
    shocks=data[SHOCK_MASK_OFFSET:SHOCK_MASK_OFFSET+SHOCK_MASK_COUNT].astype(np.uint8)
    account=(data[ACCOUNT_MASK_OFFSET:CAMPAIGN_OFFSET].astype(np.uint8) if len(data)>=MANAGED_STATE_COUNT else None)
    if not shocks.any() and (account is None or np.all(account==1)):
        return 'COMPLETE'
    return stable_hash({'shocks':shocks.tolist(),'account_known':None if account is None else account.tolist()})[:24]


def exact_support(job):
    if len(job.signed_legs) > 8 or len(job.reports) > 32 or len(job.eligible_intervals) > 32:
        raise RiskError("OUTSIDE_MODEL_DOMAIN", "Count ceiling exceeded")
    if (job.fixing_utc_us-job.t0_utc_us)/YEAR_US > 30/365:
        raise RiskError("OUTSIDE_MODEL_DOMAIN", "Horizon exceeds 30 calendar days")
    if any(abs(x.quantity) > 20 or x.multiplier != 100 or not .5 <= float(x.strike)/job.x0 <= 1.5 for x in job.signed_legs):
        raise RiskError("OUTSIDE_MODEL_DOMAIN", "Position outside quantity/strike/multiplier support")
    if not -.02 <= job.rate_cc <= .20 or not 0 <= job.dividend_yield_cc <= .20 or not .03 <= job.sigma_path <= 2 or any(not .03 <= x.sigma <= 2 for x in job.signed_legs):
        raise RiskError("OUTSIDE_MODEL_DOMAIN", "Numerical input range not validated")


def coverage_check(features, pack, capability="hold", regime=None):
    prefix = capability+"_"
    required = (prefix+"mean", prefix+"std", prefix+"training_encodings")
    if not all(k in pack.arrays for k in required):
        raise RiskError("MODEL_COVERAGE_UNAVAILABLE")
    mean, std, training = (pack.arrays[k] for k in required)
    if features.shape != mean.shape or np.any(std < 1e-8):
        raise RiskError("FEATURE_SCHEMA_MISMATCH")
    scaled = (features-mean)/std
    coverage=pack.manifest.get("coverage", {}).get(capability, {})
    gate=coverage.get("regimes",{}).get(regime) if "regimes" in coverage else coverage
    if gate is None or not gate.get("qualified",True):
        raise RiskError("MODEL_COVERAGE_UNAVAILABLE",regime=regime)
    if "training_indices" in gate:
        training=training[np.asarray(gate["training_indices"],dtype=np.int64)]
    if len(training) < 8:
        raise RiskError("MODEL_COVERAGE_UNAVAILABLE")
    distances = []
    for start in range(0, len(training), 4096):
        distances.extend(np.sum((training[start:start+4096]-scaled)**2, axis=1).tolist())
    eighth = math.sqrt(float(np.partition(distances, 7)[7])/len(features))
    if eighth > gate.get("eighth_neighbor_distance_99", -1) or gate.get("calibration_count", 0) < 200:
        raise RiskError("MODEL_COVERAGE_UNAVAILABLE", eighth_neighbor_distance=eighth)
    return scaled, gate
