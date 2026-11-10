"""Read-only numeric account evidence for a frozen managed state."""
from decimal import Decimal
from .feature_layout import ACCOUNT_FIELDS


def account_risk_features(context,campaign_id=None):
    values={name:None for name in ACCOUNT_FIELDS}
    if not context or not context.get('broker_state'):
        return {'status':'UNAVAILABLE','reason':'ACCOUNT_CONTEXT_UNAVAILABLE','values':values}
    from qqq_sim.broker.service import Broker
    from qqq_sim.broker.risk_checks import certify_session
    try:
        broker=Broker.from_dict(context['broker_state'])
    except (KeyError,TypeError,ValueError) as error:
        from .contracts import RiskError
        raise RiskError('INVALID_ACCOUNT_CONTEXT',str(error)) from error
    summary=broker.account_summary()
    for name in ACCOUNT_FIELDS:
        if name in summary:values[name]=summary[name]
    values.update(unsettled_cash=broker.ledger.economic_cash-broker.ledger.settled_cash,
                  bp_cap=broker.bp_cap,loss_cap=broker.loss_cap,session_loss_cap=broker.session_loss_cap)
    certificate=certify_session(broker.ledger,broker.session_loss_cap)
    cid=campaign_id or context.get('campaign_id')
    current=certificate['campaigns'].get(cid)
    values['campaign_loss_bound']=None if current is None else current.loss_bound
    values['session_loss_bound']=certificate.get('loss_bound')
    other=[item.lower_bound for key,item in certificate['campaigns'].items() if key!=cid]
    values['other_campaign_lower_bound']=sum(other,Decimal(0)) if all(item is not None for item in other) else None
    available=context.get('available_cash_usd')
    values['campaign_available_cash']=Decimal(str(available)) if available is not None else summary['available_option_bp']
    for prefix in ('campaign','session'):
        loss=values[prefix+'_loss_bound']
        cap=broker.loss_cap if prefix=='campaign' else broker.session_loss_cap
        values[prefix+'_remaining_loss_budget']=None if loss is None else cap-loss
    return {'status':'KNOWN' if all(value is not None for value in values.values()) else 'PARTIAL',
            'reason':None if cid in broker.ledger.campaigns else 'DRAFT_POSTFILL_ACCOUNT_CONTEXT_UNAVAILABLE',
            'values':{name:None if value is None else str(value) for name,value in values.items()},
            'basis':'FROZEN_AUTHORITATIVE_ACCOUNT','campaign_id':cid}


def bind_account_features(campaign,context):
    """Refresh derived fields from this state, never a previous transition."""
    result=dict(campaign)
    record=account_risk_features(context,result.get('campaign_id'))
    result['account_features']=record
    values=record['values']
    if values.get('loss_cap') is not None:
        result['loss_cap_usd']=values['loss_cap']
    elif context and context.get('loss_cap_usd') is not None:
        result['loss_cap_usd']=context['loss_cap_usd']
    if values.get('campaign_remaining_loss_budget') is not None:
        result['remaining_loss_budget_usd']=values['campaign_remaining_loss_budget']
    elif 'remaining_loss_budget_usd' in result:
        result.pop('remaining_loss_budget_usd')
    return result
