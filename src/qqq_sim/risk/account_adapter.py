"""Bounded exact account preview for concrete fast-policy proposals."""
from datetime import datetime,timezone
from decimal import Decimal
from functools import reduce
from math import gcd
from qqq_sim.broker.models import BrokerError,OrderDraft,OrderLeg
from qqq_sim.broker.service import Broker
from qqq_sim.broker.margin import WorkBudget


def account_proposal_check(job,campaign,proposal,limits=None):
    limits=limits if limits is not None else {}
    if not limits.get('broker_state'):
        return {'status':'UNSUPPORTED','reason':'ACCOUNT_CONTEXT_UNAVAILABLE'}
    budget=limits.setdefault('_account_preview_work_budget',WorkBudget())
    if budget.request_nodes>=budget.request_limit:
        return {'status':'UNSUPPORTED','reason':'MARGIN_GROUPING_BUDGET_EXCEEDED','request_grouping_nodes':budget.request_nodes}
    try:
        broker=Broker.from_dict(limits['broker_state'])
        campaign_id=limits.get('campaign_id',campaign.get('campaign_id'))
        if campaign_id not in broker.ledger.campaigns:
            return {'status':'UNSUPPORTED','reason':'DRAFT_POSTFILL_ACCOUNT_CONTEXT_UNAVAILABLE'}
        mapping=limits.get('contract_id_mapping',{})
        canonical={contract.key:quantity for contract,quantity in broker.ledger.positions(campaign_id)}
        expected={mapping.get(leg.contract_id,leg.contract_id):leg.quantity for leg in job.signed_legs}
        if canonical!=expected: return {'status':'UNSUPPORTED','reason':'ACCOUNT_POSITION_CONTEXT_MISMATCH'}
        if proposal.get('family')=='HOLD' or not proposal.get('legs'):
            result=broker._check_poststate(broker.ledger,budget=budget)
            return {'status':'OK' if result['status']=='OK' else 'FAILED','reason':result.get('reason'),
                    'project_bp_used':str(result.get('project_bp_used')),'request_grouping_nodes':budget.request_nodes}
        contracts={contract.key:contract for contract,quantity in broker.ledger.positions()}
        contracts.update({key:quote.contract for key,quote in broker.quotes.items()})
        legs=[]
        for leg in proposal['legs']:
            raw=leg if isinstance(leg,dict) else {'contract_id':leg.contract_id,'quantity':leg.quantity}
            key=mapping.get(raw['contract_id'],raw['contract_id'])
            if key not in contracts: raise BrokerError('CONTRACT_NOT_YET_OBSERVED',contract=key)
            legs.append(OrderLeg(contracts[key],int(raw['quantity'])))
        units=reduce(gcd,(abs(leg.quantity) for leg in legs))
        limit=-Decimal(str(proposal.get('limit_cost_usd',proposal['premium_cost_usd'])))/units
        original=broker.ledger.campaigns[campaign_id]
        at=datetime.fromtimestamp(job.t0_utc_us/1e6,timezone.utc)
        draft=OrderDraft(campaign_id,tuple(legs),limit,original.deadline,original.hard_cutoff,
                         order_type=proposal.get('order_type','PROTECTED_LIMIT'),
                         tif=proposal.get('tif','DAY'),
                         max_total_fees=Decimal(str(proposal.get('max_total_fees_usd',50))),
                         intent='CLOSE' if proposal.get('family') in ('FULL_CLOSE','PARTIAL_CLOSE','EXIT_STOP') else 'ADJUST',
                         operational_model=original.operational_model,standing_protection=original.standing_protection)
        order=broker.submit(draft,at,broker.quotes,'policy-account:'+str(proposal.get('id','preview')),work_budget=budget)
        summary=broker.account_summary({})
        return {'status':'OK','reservation_usd':str(order.reservation),'project_bp_used':str(summary['project_bp_used']),
                'available_option_bp':str(summary['available_option_bp']),'request_grouping_nodes':budget.request_nodes,
                'conditional_fill_checks':'ALL_SUPPORTED_PARTIAL_RATIOS_PLUS_EXISTING_NO_FILL_STATE'}
    except BrokerError as error:
        return {'status':'UNSUPPORTED' if 'BUDGET' in error.code or 'UNSUPPORTED' in error.code else 'FAILED',
                'reason':error.code,'details':error.details,'request_grouping_nodes':budget.request_nodes}


def create_isolated_account_context(job,campaign,universe=()):
    """Materialize ONLY explicitly synthetic initial fills for offline designs.

    No market data are fetched or historical observations fabricated. The given
    job must declare synthetic provenance and executable bid/ask values.
    """
    from datetime import timedelta
    from decimal import ROUND_FLOOR
    from qqq_sim.broker.models import Contract,Quote
    from .contracts import RiskError
    if not str(job.snapshot_id).startswith('SYNTHETIC'):
        raise RiskError('SYNTHETIC_ACCOUNT_CONTEXT_REQUIRED')
    at=datetime.fromtimestamp(job.t0_utc_us/1e6,timezone.utc)
    fixing=datetime.fromtimestamp(job.fixing_utc_us/1e6,timezone.utc)
    deadline=datetime.fromtimestamp(job.deadline_utc_us/1e6,timezone.utc)
    cutoff=datetime.fromtimestamp(int(campaign['hard_cutoff_utc_us'])/1e6,timezone.utc)
    cid=campaign.get('campaign_id','SYNTHETIC_INITIAL')
    contracts={}; mapping={}; quotes={}
    for leg in tuple(job.signed_legs)+tuple(universe):
        if leg.bid is None or leg.ask is None: raise RiskError('SYNTHETIC_EXECUTABLE_QUOTE_REQUIRED')
        leg_fixing=datetime.fromtimestamp(leg.fixing_utc_us/1e6,timezone.utc)
        contract=Contract('QQQ',leg.right,leg.strike,leg_fixing.astimezone(__import__('zoneinfo').ZoneInfo('America/New_York')).date(),
                          multiplier=leg.multiplier,last_trade_at=leg_fixing,fixing_at=leg_fixing,
                          metadata_version='EXPLICIT_SYNTHETIC_STANDARD_QQQ_V1')
        contracts[leg.contract_id]=contract; mapping[leg.contract_id]=contract.key
        quotes[contract.key]=Quote(contract,Decimal(str(leg.bid)),Decimal(str(leg.ask)),at-timedelta(seconds=2),
                                  'SYNTHETIC_INITIAL_QUOTE:'+leg.contract_id)
    legs=tuple(OrderLeg(contracts[leg.contract_id],leg.quantity) for leg in job.signed_legs)
    units=reduce(gcd,(abs(leg.quantity) for leg in legs))
    premium=Decimal(str(campaign.get('gross_cash_usd',job.entry_cash_usd)))
    fees=premium-Decimal(str(campaign['net_cash_usd']))
    if fees<0: raise RiskError('SYNTHETIC_INITIAL_CASH_INCONSISTENT')
    limit=(premium/units).quantize(Decimal('1'),rounding=ROUND_FLOOR)
    broker=Broker()
    draft=OrderDraft(cid,legs,limit,deadline,cutoff,max_total_fees=max(Decimal('.01'),fees),
                     expense_budget=Decimal(str(campaign.get('remaining_expense_budget_usd',100))),
                     operational_model=campaign.get('operational_model','SIMULTANEOUS_DELIVERY_HEDGE_V1'),
                     standing_protection=campaign.get('protection_instruction_recorded',True))
    try:
        broker.submit(draft,at-timedelta(seconds=1),quotes,'SYNTHETIC_EXECUTED_INITIAL_ORDER')
        updated=[Quote(q.contract,q.bid,q.ask,at,'SYNTHETIC_EXECUTED_QUOTE:'+q.quote_id) for q in quotes.values()]
        broker.on_quotes(updated,at)
    except BrokerError as error:
        raise RiskError('SYNTHETIC_INITIAL_ACCOUNT_INFEASIBLE',reason=error.code) from error
    actual=broker.ledger.campaigns[cid]
    if actual.trade_cash!=premium or actual.net_cash!=Decimal(str(campaign['net_cash_usd'])) or actual.target!=job.target_usd:
        raise RiskError('SYNTHETIC_INITIAL_CASH_INCONSISTENT',expected_cash=str(premium),actual_cash=str(actual.trade_cash))
    return {'broker_state':broker.to_dict(),'campaign_id':cid,'contract_id_mapping':mapping,
            'available_cash_usd':str(broker.ledger.economic_cash),'other_campaign_net_usd':'0',
            'project_bp_cap_usd':'5000','loss_cap_usd':'1000','session_loss_cap_usd':'1000',
            'account_context_basis':'EXPLICIT_SYNTHETIC_EXECUTED_INITIAL_OPENING'}


def advance_isolated_account_context(context,old_job,action,quoted_legs,when_us):
    """Execute one declared synthetic next-opportunity IOC against modeled quotes.

    The exact Broker ledger is authoritative. Unfilled IOC orders are canceled
    after their one eligible opportunity; a failed exit cannot become flat.
    """
    from qqq_sim.broker.models import Quote
    from .contracts import RiskError
    if context.get('account_context_basis')!='EXPLICIT_SYNTHETIC_EXECUTED_INITIAL_OPENING':
        raise RiskError('SYNTHETIC_ACCOUNT_CONTEXT_REQUIRED')
    if when_us<=old_job.t0_utc_us: raise RiskError('ORDER_CAUSALITY_FAILURE')
    broker=Broker.from_dict(context['broker_state'])
    cid=context['campaign_id']; original=broker.ledger.campaigns[cid]
    before_cash=original.trade_cash; before_fees=original.expenses
    mapping=dict(context.get('contract_id_mapping',{}))
    contracts={key:quote.contract for key,quote in broker.quotes.items()}
    contracts.update({contract.key:contract for contract,q in broker.ledger.positions()})
    at=datetime.fromtimestamp(old_job.t0_utc_us/1e6,timezone.utc)
    when=datetime.fromtimestamp(when_us/1e6,timezone.utc)
    order=None
    if action.get('family')!='HOLD':
        legs=[]
        for raw in action['legs']:
            key=mapping.get(raw['contract_id'],raw['contract_id'])
            if key not in contracts: raise RiskError('ACTION_CONTRACT_UNAVAILABLE')
            legs.append(OrderLeg(contracts[key],int(raw['quantity'])))
        units=reduce(gcd,(abs(leg.quantity) for leg in legs))
        limit=-Decimal(str(action['limit_cost_usd']))/units
        try:
            draft=OrderDraft(cid,tuple(legs),limit,original.deadline,original.hard_cutoff,
                              order_type=action.get('order_type','PROTECTED_LIMIT'),tif=action.get('tif','DAY'),
                              max_total_fees=Decimal(str(action.get('max_total_fees_usd',50))),
                              intent='CLOSE' if action['family'] in ('FULL_CLOSE','PARTIAL_CLOSE','EXIT_STOP') else 'ADJUST',
                              operational_model=original.operational_model,standing_protection=original.standing_protection)
            order=broker.submit(draft,at,broker.quotes,'SYNTHETIC_POLICY:'+str(action['id'])+':'+str(old_job.t0_utc_us))
        except BrokerError as error:
            raise RiskError('ACCOUNT_TRANSITION_UNSUPPORTED',reason=error.code) from error
    quoted_legs=tuple(quoted_legs); quotes=[]
    for leg in quoted_legs:
        key=mapping.get(leg.contract_id,leg.contract_id)
        if key not in contracts: continue
        if leg.bid is None or leg.ask is None: continue
        tick=Decimal(str(getattr(leg,'tick',.01)))
        bid=Decimal(str(leg.bid)).quantize(tick)
        ask=Decimal(str(leg.ask)).quantize(tick)
        quotes.append(Quote(contracts[key],bid,ask,when,
                            'SYNTHETIC_FUTURE_QUOTE:'+str(when_us)+':'+leg.contract_id))
    # A delayed next opportunity can cross the DAY order's session boundary.
    # Preserve the declared TIF before considering that later quote batch.
    session_end=next((close for opening,close in old_job.eligible_intervals if close>old_job.t0_utc_us),None)
    if session_end is not None and when_us>=session_end:
        broker.expire_orders(when,session_close=datetime.fromtimestamp(session_end/1e6,timezone.utc))
    broker.on_quotes(quotes,when)
    if order and order.remaining_units:
        broker.cancel(order.order_id,when,command_id='SYNTHETIC_IOC_CANCEL:'+order.order_id)
    broker.settle_due(when)
    current=broker.ledger.campaigns[cid]
    updated={key:value for key,value in context.items() if not key.startswith('_')}
    updated['broker_state']=broker.to_dict()
    updated['available_cash_usd']=str(broker.ledger.economic_cash)
    reverse={value:key for key,value in mapping.items()}
    positions={reverse.get(contract.key,contract.key):q for contract,q in broker.ledger.positions(cid)}
    return {'context':updated,'filled':bool(order and order.status=='FILLED'),
            'status':'HOLD' if order is None else 'FILLED' if order.status=='FILLED' else 'NO_FILL',
            'order_status':order.status if order else None,
            'premium_cost_usd':str(-(current.trade_cash-before_cash)),
            'fees_usd':str(current.expenses-before_fees),'gross_cash_usd':str(current.trade_cash+current.income),
            'net_cash_usd':str(current.net_cash),'remaining_expense_budget_usd':str(current.expense_remaining),
            'positions':positions,'target_usd':str(current.target),'economically_completed':not positions and not broker.ledger.shares(cid),
            'order_observations':order.observations if order else [],'broker_state_hash':fingerprint_context(updated)}


def fingerprint_context(context):
    from qqq_sim.broker.models import fingerprint
    return fingerprint(context['broker_state'])
