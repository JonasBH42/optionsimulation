"""Deterministic bounded admission of presently observed forecast contracts."""
from __future__ import annotations
from dataclasses import replace
import math

from .contracts import RiskError,RiskLeg,stable_hash,canonical


def admit_known_universe(job,observed,hard_cutoff_utc_us,*,max_contracts=512,max_expiries=8):
    """Keep exact holdings and bound optional observed actions with an audit.

    A held expiry may exceed the hard cutoff. Its true maturity and same-expiry
    management contracts remain available; an additional expiry may not extend
    beyond that cutoff. Current-expiry neighbors precede optional roll expiries.
    Within each expiry strikes are ranked by absolute log-moneyness, lower
    strike, right and exact identity. Omission never asserts infeasibility.
    """
    if not 1<=max_contracts<=512 or not 1<=max_expiries<=8:
        raise RiskError('INVALID_UNIVERSE_ADMISSION_BUDGET')
    held={leg.contract_id:leg for leg in job.signed_legs}
    held_expiries={leg.fixing_utc_us for leg in held.values()}
    if len(held)>max_contracts or len(held_expiries)>max_expiries:
        raise RiskError('UNSUPPORTED_PORTFOLIO_SIZE',held_contracts=len(held),held_expiries=len(held_expiries))
    by_id={};excluded=[]
    def reject(leg,reason):
        excluded.append({'contract_id':leg.contract_id,'fixing_utc_us':leg.fixing_utc_us,'reason':reason})
    for raw in observed:
        leg=raw if isinstance(raw,RiskLeg) else RiskLeg.from_dict(raw)
        old=by_id.get(leg.contract_id)
        if old is not None:
            if canonical(old)!=canonical(leg):
                raise RiskError('CONFLICTING_UNIVERSE_CONTRACT',contract_id=leg.contract_id)
            continue
        if leg.known_at_utc_us is None:
            reject(leg,'UNIVERSE_AVAILABILITY_UNKNOWN');continue
        if leg.known_at_utc_us>job.t0_utc_us:
            reject(leg,'FUTURE_UNIVERSE_RECORD');continue
        last_trade=leg.last_trading_utc_us
        if last_trade is None and leg.fixing_utc_us==job.fixing_utc_us:
            last_trade=job.last_trading_utc_us
            leg=replace(leg,last_trading_utc_us=last_trade)
        if last_trade is None:
            reject(leg,'UNIVERSE_CONTRACT_TIMES_MISSING');continue
        if leg.fixing_utc_us<=job.t0_utc_us or last_trade<=job.t0_utc_us:
            reject(leg,'UNIVERSE_EXPIRED_OR_STOPPED_CONTRACT');continue
        if leg.bid is None or leg.ask is None:
            reject(leg,'MISSING_EXECUTABLE_QUOTE');continue
        if leg.fixing_utc_us>hard_cutoff_utc_us and leg.fixing_utc_us not in held_expiries:
            reject(leg,'EXPIRY_EXTENDS_BEYOND_HARD_CUTOFF');continue
        by_id[leg.contract_id]=replace(leg,quantity=1)
    # Current contracts come from the already validated immutable HOLD job.
    # Never drop them merely because the optional quote list was filtered.
    for key,leg in held.items():
        other=by_id.get(key)
        if other is not None and (other.right,other.strike,other.fixing_utc_us,other.multiplier,other.exercise_style,other.deliverable_id)!=(leg.right,leg.strike,leg.fixing_utc_us,leg.multiplier,leg.exercise_style,leg.deliverable_id):
            raise RiskError('CONTRACT_IDENTITY_COLLISION',contract_id=key)
        if leg.known_at_utc_us is None or leg.known_at_utc_us>job.t0_utc_us or leg.bid is None or leg.ask is None:
            raise RiskError('CURRENT_UNIVERSE_CONTRACT_UNAVAILABLE',contract_id=key)
        by_id[key]=replace(leg,quantity=1,last_trading_utc_us=leg.last_trading_utc_us or job.last_trading_utc_us)
    extra_expiries=sorted({leg.fixing_utc_us for leg in by_id.values()}-held_expiries,
                          key=lambda when:(abs(when-hard_cutoff_utc_us),-when))
    selected_expiries=set(held_expiries)|set(extra_expiries[:max_expiries-len(held_expiries)])
    for key,leg in tuple(by_id.items()):
        if leg.fixing_utc_us not in selected_expiries:
            reject(leg,'DECLARED_EXPIRY_SEARCH_CAP');del by_id[key]
    expiry_order=sorted(held_expiries,key=lambda when:(abs(when-hard_cutoff_utc_us),-when))
    expiry_order.extend(when for when in extra_expiries if when in selected_expiries)
    rank={when:index for index,when in enumerate(expiry_order)}
    candidates=sorted((leg for key,leg in by_id.items() if key not in held),
        key=lambda leg:(rank[leg.fixing_utc_us],abs(math.log(float(leg.strike)/job.x0)),leg.strike,leg.right,leg.contract_id))
    admitted=[by_id[key] for key in sorted(held)]
    available=max_contracts-len(admitted)
    admitted.extend(candidates[:available])
    for leg in candidates[available:]:reject(leg,'DECLARED_CONTRACT_SEARCH_CAP')
    admitted=tuple(sorted(admitted,key=lambda leg:(leg.fixing_utc_us,leg.strike,leg.right,leg.contract_id)))
    audit={'rule':'M03_CURRENT_THEN_CURRENT_EXPIRY_NEIGHBORS_THEN_NEAREST_HARD_EXPIRIES_V1',
           'contract_cap':max_contracts,'expiry_cap':max_expiries,'held_contracts_retained':sorted(held),
           'considered_expiries_utc_us':sorted(selected_expiries),
           'admitted_expiries_utc_us':sorted({leg.fixing_utc_us for leg in admitted}),
           'admitted_contract_count':len(admitted),'excluded_contracts':sorted(excluded,key=lambda row:(row['contract_id'],row['reason'])),
           'interpretation':'DECLARED_BOUNDED_SEARCH_OMISSIONS_ARE_NOT_INFEASIBILITY_FINDINGS',
           'identity':stable_hash({'asof':job.t0_utc_us,'hard_cutoff':hard_cutoff_utc_us,'universe':admitted})}
    return admitted,audit
