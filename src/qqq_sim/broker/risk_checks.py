"""Conditional finite-budget operational certificates, never real-world guarantees."""
from dataclasses import dataclass
from .models import D


@dataclass(frozen=True)
class RiskCertificate:
    status: str
    model: str
    lower_bound: object = None
    loss_bound: object = None
    delivery_bound: object = None
    mandatory_delivery_fees: object = None
    optional_expense_reserve: object = None
    reason: str | None = None
    pairs: tuple = ()
    assumptions: tuple = ()


def certify_campaign(ledger,campaign_id,loss_cap=D(1000)):
    c=ledger.campaigns[campaign_id]; positions=ledger.positions(campaign_id)
    model=c.operational_model
    if ledger.shares(campaign_id):
        return RiskCertificate('UNKNOWN',model,reason='RESIDUAL_STOCK_EXPOSURE')
    if c.expense_remaining<0:
        return RiskCertificate('FAILED',model,reason='FUTURE_EXPENSE_BUDGET_EXCEEDED')
    shorts=[(contract,-q) for contract,q in positions if q<0]
    longs={contract.key:[contract,q] for contract,q in positions if q>0}
    if shorts and (model!='SIMULTANEOUS_DELIVERY_HEDGE_V1' or not c.standing_protection):
        return RiskCertificate('UNKNOWN',model,reason='UNVERIFIED_ASSIGNMENT_NOTICE_OR_PROTECTION')
    delivery=D(0); mandatory=D(0); pairs=[]
    for short,quantity in sorted(shorts,key=lambda x:x[0].key):
        candidates=sorted((entry for entry in longs.values() if entry[0].partition==short.partition and entry[0].right==short.right),
                          key=lambda entry:(abs(entry[0].strike-short.strike),entry[0].key))
        for item in candidates:
            long,available=item
            n=min(quantity,available)
            if not n: continue
            if short.multiplier!=100 or short.deliverable!='QQQ_STANDARD_100_SHARES':
                return RiskCertificate('UNKNOWN',model,reason='UNSUPPORTED_DELIVERABLE')
            eta=1 if short.right=='C' else -1
            delivery+=short.multiplier*n*max(D(0),eta*(long.strike-short.strike))
            mandatory+=D(10)*n
            pairs.append((short.key,long.key,n)); item[1]-=n; quantity-=n
            if not quantity: break
        if quantity: return RiskCertificate('UNKNOWN',model,reason='UNPROTECTED_SHORT_CONTRACT')
    # Unallocated longs can incur their own exercise group. They never supply
    # free protection or an unbudgeted exercise-fee assumption.
    mandatory+=D(5)*sum(item[1] for item in longs.values())
    if c.expense_remaining<mandatory:
        return RiskCertificate('FAILED',model,delivery_bound=delivery,mandatory_delivery_fees=mandatory,
                               reason='MANDATORY_DELIVERY_FEE_RESERVE_INSUFFICIENT',pairs=tuple(pairs))
    # Successful completed campaigns cannot consume a new action budget.
    future_budget=D(0) if c.target_realized else c.expense_remaining
    lower=c.net_cash-delivery-future_budget
    loss=max(D(0),-lower)
    return RiskCertificate('VERIFIED_WITHIN_DECLARED_MODEL' if loss<=loss_cap else 'FAILED',model,
                           lower,loss,delivery,mandatory,max(D(0),future_budget-mandatory),
                           None if loss<=loss_cap else 'CAMPAIGN_LOSS_CAP_EXCEEDED',tuple(pairs),
                           ('simultaneous_notice_and_delivery','standing_protective_exercise',
                            'atomic_netted_funding','finite_enforced_expense_budget') if shorts else ('long_only_nonnegative_terminal_value',))


def certify_session(ledger,loss_cap=D(1000)):
    certificates={key:certify_campaign(ledger,key,loss_cap) for key in sorted(ledger.campaigns)}
    unknown=[k for k,c in certificates.items() if c.lower_bound is None]
    if unknown: return {'status':'UNKNOWN','reason':'CAMPAIGN_CERTIFICATE_UNAVAILABLE','campaigns':certificates,'unknown':unknown}
    lower=sum((c.lower_bound for c in certificates.values()),D(0))
    # Unallocated income/expenses and external funding never replenish risk covertly.
    allocated=sum((c.net_cash for c in ledger.campaigns.values()),D(0))
    unallocated=ledger.economic_cash-D(ledger.initial_cash_cents)/100-ledger.external_contributions-allocated
    lower+=min(D(0),unallocated)
    loss=max(D(0),-lower)
    return {'status':'VERIFIED_WITHIN_DECLARED_MODEL' if loss<=loss_cap else 'FAILED',
            'reason':None if loss<=loss_cap else 'SESSION_AGGREGATE_LOSS_CAP_EXCEEDED',
            'lower_bound':lower,'loss_bound':loss,'campaigns':certificates}

