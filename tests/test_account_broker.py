from datetime import date,datetime,timedelta
from decimal import Decimal as D
from zoneinfo import ZoneInfo
import copy
import pytest
from qqq_sim.broker.models import Contract,Quote,OrderLeg,OrderDraft,BrokerError,fingerprint
from qqq_sim.broker.fees import FeeEngine,FeeProfile,allocate_cents
from qqq_sim.broker.margin import MarginEngine,WorkBudget,payoff_extrema,naked_component
from qqq_sim.broker.service import Broker
from qqq_sim.broker.risk_checks import certify_campaign,certify_session
from qqq_sim.broker.settlement import stock_settlement_lag,settlement_date,daily_interest,daily_borrow

NY=ZoneInfo('America/New_York')
AT=datetime(2026,11,9,10,tzinfo=NY)
EXP=date(2026,11,13)


def contract(right='C',strike='105'):
    return Contract('QQQ',right,D(strike),EXP)


def spread_fixture(units=1,strict=True):
    short,long=contract(),contract(strike='110')
    quotes=[Quote(short,D('2'),D('2'),AT,'s0',units,units),Quote(long,D('1'),D('1'),AT,'l0',units,units)]
    draft=OrderDraft('campaign',(OrderLeg(short,-units),OrderLeg(long,units)),D(100),AT+timedelta(days=3),AT+timedelta(days=4))
    broker=Broker(strict=strict)
    return broker,short,long,quotes,draft


def open_spread(units=1):
    b,s,l,quotes,draft=spread_fixture(units)
    order=b.submit(draft,AT,quotes,'open')
    future=[Quote(q.contract,q.bid,q.ask,AT+timedelta(seconds=1),q.quote_id+'f',units,units) for q in quotes]
    fills=b.on_quotes(future,AT+timedelta(seconds=1))
    assert len(fills)==2
    return b,s,l,order


def test_exact_fee_fixtures_and_cumulative_caps():
    c=contract(); fee=FeeEngine()
    assert sum(fee.assess_option('buy',c,1,D(1),1,AT.date()).values())==D('1.12')
    assert sum(fee.assess_option('sell',c,-1,D(1),1,AT.date()).values())==D('1.13')
    amounts=[sum(fee.assess_option('sixteen',c,1,D(1),1,AT.date()).values()) for _ in range(16)]
    assert sum(amounts)==D('11.92')
    first=fee.assess_option('partial',c,6,D(1),6,AT.date())
    second=fee.assess_option('partial',c,6,D(1),6,AT.date()+timedelta(days=1))
    assert first['commission']==6 and second['commission']==4
    assert fee.assess_option('replace',c,6,D(1),6,AT.date())['commission']==6


def test_taf_waiver_and_proportional_cents():
    for dt,waived in [(date(2026,9,30),False),(date(2026,10,1),True),(date(2026,12,31),True),(date(2027,1,1),False)]:
        profile=FeeProfile(rule_asof=dt)
        assert (profile.taf_rate()==0)==waived
    allocations=allocate_cents(D('.02'),[('b',1),('a',1),('c',1)])
    assert allocations=={'a':D('.01'),'b':D('.01'),'c':D(0)}
    assert allocate_cents(D('-.02'),[('b',1),('a',1),('c',1)])=={k:-v for k,v in allocations.items()}


def test_credit_spread_economics_and_roundtrip_h0():
    b,s,l,order=open_spread()
    summary=b.account_summary({s.key:D(2),l.key:D(1)})
    assert summary['economic_cash']==D('5097.75')
    assert summary['collateral']==500
    assert summary['available_option_bp']==D('4597.75')
    assert summary['project_bp_used']==D('402.25')
    assert summary['equity']==D('4997.75')
    c=b.ledger.campaigns['campaign']; assert c.target==50 and c.original_max_profit==100
    close=OrderDraft('campaign',(OrderLeg(s,1),OrderLeg(l,-1)),D(-50),c.deadline,c.hard_cutoff,intent='CLOSE')
    b.submit(close,AT+timedelta(seconds=2),list(b.quotes.values()),'close')
    b.on_quotes([Quote(s,D(1),D(1),AT+timedelta(seconds=3),'sc'),Quote(l,D('.5'),D('.5'),AT+timedelta(seconds=3),'lc')],AT+timedelta(seconds=3))
    c=b.ledger.campaigns['campaign']
    assert c.target==50 and c.net_cash==D('47.50') and not c.target_realized
    assert c.realized_gross==50 and c.realized_fees==D('2.50')
    assert b.account_summary({})['economic_cash']==D('5047.50')
    assert b.account_summary({})['collateral']==0
    assert b.account_summary({})['project_bp_used']==0


def test_no_past_quote_fill_and_quote_update_all_legs():
    b,s,l,quotes,draft=spread_fixture()
    b.submit(draft,AT,quotes,'open')
    assert b.on_quotes(quotes,AT+timedelta(seconds=1))==[]
    assert b.on_quotes([Quote(s,D(2),D(2),AT+timedelta(seconds=1),'s1')],AT+timedelta(seconds=1))==[]
    assert len(b.on_quotes([Quote(l,D(1),D(1),AT+timedelta(seconds=2),'l1')],AT+timedelta(seconds=2)))==2


def test_unseen_same_timestamp_after_submission_is_eligible_but_seen_is_not():
    b,s,l,quotes,draft=spread_fixture()
    prior=[Quote(q.contract,q.bid,q.ask,AT-timedelta(seconds=1),q.quote_id+'prior') for q in quotes]
    order=b.submit(draft,AT,prior,'boundary')
    unseen=[Quote(q.contract,q.bid,q.ask,AT,q.quote_id+'atD') for q in quotes]
    assert len(b.on_quotes(unseen,AT))==2
    assert order.status=='FILLED'
    seen,_,_,quotes2,draft2=spread_fixture()
    seen.submit(draft2,AT,quotes2,'inclusive-seen')
    assert seen.on_quotes(quotes2,AT)==[]


def test_partial_fill_does_not_freeze_full_target_or_reuse_depth():
    b,s,l,quotes,draft=spread_fixture(2)
    order=b.submit(draft,AT,quotes,'open')
    future=[Quote(q.contract,q.bid,q.ask,AT+timedelta(seconds=1),q.quote_id+'f',1,1) for q in quotes]
    assert len(b.on_quotes(future,AT+timedelta(seconds=1)))==2
    assert order.status=='PARTIALLY_FILLED' and order.remaining_units==1
    assert b.ledger.campaigns['campaign'].target is None
    assert b.on_quotes(future,AT+timedelta(seconds=2))==[]
    second=[Quote(q.contract,q.bid,q.ask,AT+timedelta(seconds=3),q.quote_id+'2',1,1) for q in quotes]
    assert len(b.on_quotes(second,AT+timedelta(seconds=3)))==2
    assert b.ledger.campaigns['campaign'].target==100


def test_invalid_latest_quote_blocks_and_does_not_fallback():
    b,s,l,quotes,draft=spread_fixture(); b.submit(draft,AT,quotes,'open')
    assert b.on_quotes([Quote(s,D(3),D(2),AT+timedelta(seconds=1),'bad'),Quote(l,D(1),D(1),AT+timedelta(seconds=1),'good')],AT+timedelta(seconds=1))==[]
    assert not b.ledger.positions()


def test_strict_depth_missing_is_not_infinite():
    b,s,l,quotes,draft=spread_fixture()
    from dataclasses import replace
    order=b.submit(replace(draft,strict_depth=True),AT,quotes,'open')
    b.on_quotes([Quote(q.contract,q.bid,q.ask,AT+timedelta(seconds=1),q.quote_id+'f') for q in quotes],AT+timedelta(seconds=1))
    assert order.observations[-1]['reason']=='INSUFFICIENT_QUOTE_DEPTH'
    assert not b.ledger.positions()


def test_ordinary_market_is_not_silently_protected():
    b,s,l,quotes,draft=spread_fixture()
    from dataclasses import replace
    with pytest.raises(BrokerError,match='UNVERIFIED_EXECUTION_BOUND'): b.submit(replace(draft,order_type='MARKET'),AT,quotes)


def test_pending_order_reservation_and_duplicate_submission():
    b,s,l,quotes,draft=spread_fixture()
    order=b.submit(draft,AT,quotes,'same')
    assert order.reservation==450
    assert b.submit(draft,AT,quotes,'same') is order
    assert len(b.orders)==1 and not b.ledger.positions()
    assert b.account_summary({})['working_order_reserve']==450
    b.cancel(order.order_id,AT+timedelta(seconds=1)); assert b.account_summary({})['working_order_reserve']==0


def test_operational_certificate_is_distinct_from_terminal_width():
    b,s,l,o=open_spread(); c=b.ledger.campaigns['campaign']
    cert=certify_campaign(b.ledger,'campaign')
    assert cert.status=='VERIFIED_WITHIN_DECLARED_MODEL'
    assert cert.loss_bound==D('502.25') and cert.delivery_bound==500 and cert.mandatory_delivery_fees==10
    c.expense_remaining=D(10)
    assert certify_campaign(b.ledger,'campaign').loss_bound==D('412.25')
    c.operational_model='DELAYED_NOTICE'
    assert certify_campaign(b.ledger,'campaign').status=='UNKNOWN'


def test_unequal_condor_collateral_vs_delivery():
    positions=[(contract('P','90'),1),(contract('P','95'),-1),(contract('C','105'),-1),(contract('C','115'),1)]
    assessed=MarginEngine().evaluate(positions)
    assert assessed.collateral==1000 and assessed.components[0].name=='IRON_CONDOR'
    assert sum(abs(q) for comp in assessed.components for _,q in comp.legs)==4


def test_no_duplicate_hedge_and_bounded_grouping():
    s=contract(strike='100'); l=contract(strike='105')
    assert MarginEngine().evaluate([(s,-2),(l,1)]).status=='UNSUPPORTED'
    result=MarginEngine().evaluate([(s,-1),(l,1)],WorkBudget(state_limit=0))
    assert result.reason=='MARGIN_GROUPING_BUDGET_EXCEEDED'


def test_long_debit_cash_not_collateral_twice():
    b=Broker(); c=contract('P','100'); q=Quote(c,D(2),D(2),AT,'q0')
    draft=OrderDraft('long',(OrderLeg(c,1),),D(-200),AT+timedelta(days=3),AT+timedelta(days=4))
    b.submit(draft,AT,[q]); b.on_quotes([Quote(c,D(2),D(2),AT+timedelta(seconds=1),'q1')],AT+timedelta(seconds=1))
    s=b.account_summary({c.key:D(2)})
    assert s['economic_cash']==D('4798.88') and s['collateral']==0 and s['equity']==D('4998.88')
    assert s['project_bp_used']==D('201.12')


@pytest.mark.parametrize('right,position,shares,cash',[('C',1,100,-10000),('P',1,-100,10000),('C',-1,-100,10000),('P',-1,100,-10000)])
def test_four_physical_delivery_signs(right,position,shares,cash):
    from qqq_sim.accounting.ledger import Ledger,Campaign,Lot
    from qqq_sim.broker.lifecycle import LifecycleEngine
    ledger=Ledger(); c=contract(right,'100')
    ledger.campaigns['c']=Campaign('c',AT+timedelta(days=3),AT+timedelta(days=4),'EXPLICIT_ASSIGNMENT_EVENTS_V1',False)
    ledger.lots=[Lot('lot',c,'c',position,D(position*200))]
    before=ledger.economic_cash
    lifecycle=LifecycleEngine(ledger,FeeEngine())
    lifecycle.deliver('delivery','c',c,1,AT,AT.date()+timedelta(days=1))
    assert ledger.shares()==shares and ledger.economic_cash==before+cash-5 and not ledger.positions()
    assert ledger.stock_lots[0].basis==-D(cash)+D(position*200)
    assert not lifecycle.deliver('delivery','c',c,1,AT,AT.date()+timedelta(days=1))


def test_assignment_retains_hedge_and_no_intrinsic_cash():
    b,s,l,o=open_spread(); before=b.ledger.economic_cash
    b.lifecycle.deliver('assign','campaign',s,1,AT+timedelta(minutes=1),AT.date()+timedelta(days=1))
    assert b.ledger.shares()==-100 and b.ledger.positions()==[(l,1)]
    assert b.ledger.economic_cash==before+D(10500)-5
    assert certify_campaign(b.ledger,'campaign').reason=='RESIDUAL_STOCK_EXPOSURE'


def test_settlement_and_dividend_payment_preserve_equity():
    b,s,l,o=open_spread(); before=b.ledger.economic_cash
    b.settle_due(AT+timedelta(days=1)); assert b.ledger.economic_cash==before
    b.ledger.accrue_income('div','campaign',D(10),AT.date()+timedelta(days=2),AT)
    accrued=b.ledger.economic_cash; assert accrued==before+10
    b.settle_due(AT+timedelta(days=2)); assert b.ledger.economic_cash==accrued
    assert b.settle_due(AT+timedelta(days=2))==[]


def test_official_fixing_missing_is_typed_stop():
    b,s,l,o=open_spread()
    with pytest.raises(BrokerError,match='SETTLEMENT_REFERENCE_MISSING'):
        b.lifecycle.expiry(AT+timedelta(days=4),AT.date()+timedelta(days=5))


def test_snapshot_roundtrip_preserves_all_economics_and_fees():
    b,s,l,o=open_spread()
    before=fingerprint(b.to_dict()); restored=Broker.from_dict(b.to_dict())
    assert fingerprint(restored.to_dict())==before
    assert restored.account_summary({s.key:D(2),l.key:D(1)})==b.account_summary({s.key:D(2),l.key:D(1)})


def test_settlement_regime_and_financing():
    assert stock_settlement_lag(date(2017,9,4))==3
    assert stock_settlement_lag(date(2017,9,5))==2
    assert stock_settlement_lag(date(2024,5,28))==1
    assert settlement_date(date(2026,11,6),holidays={date(2026,11,9)})==date(2026,11,10)
    assert settlement_date(date(2026,11,10))==date(2026,11,12)  # Veterans Day banking holiday.
    assert daily_interest(-1000)==D(1000)*D('.11')/360
    with pytest.raises(BrokerError,match='BORROW_RATE_MISSING'): daily_borrow(-100,100,None)


def test_new_campaign_does_not_reset_session_loss_capacity():
    from qqq_sim.accounting.ledger import Campaign
    b,s,l,o=open_spread()
    old=Campaign('old',AT,AT,'LONG_ONLY',False,expense_remaining=D(0),status='FLAT',trade_cash=D(-600))
    b.ledger.campaigns['old']=old
    b.ledger.settled_cash_cents-=60000
    assert certify_session(b.ledger)['reason']=='SESSION_AGGREGATE_LOSS_CAP_EXCEEDED'


def test_payoff_infinity_and_penny_premium():
    c=contract()
    assert payoff_extrema([(c,1)]).maximum_unbounded
    assert payoff_extrema([(c,-1)]).minimum_unbounded
    q=Quote(c,D('4.01'),D('4.01'),AT,'tick')
    assert Broker._side(q,1)==D('4.01')


def test_paired_expiry_exercises_otm_protection_atomically():
    b,s,l,o=open_spread()
    at=AT.replace(day=13,hour=16)
    result=b.lifecycle.expiry(at,date(2026,11,16),official_fixing=D(107))
    assert result['fixing_basis']=='OFFICIAL_FIXING'
    assert not b.ledger.positions() and b.ledger.shares()==0
    c=b.ledger.campaigns['campaign']
    assert c.net_cash==D('-412.25')
    assert c.realized_gross==D(-400) and c.realized_fees==D('12.25')
    assert c.expense_remaining==90


def test_delivery_event_group_charges_once():
    fee=FeeEngine(); c=contract()
    assert fee.assess_delivery(c,AT.date(),'batch','EXERCISE')==5
    assert fee.assess_delivery(c,AT.date(),'batch','EXERCISE')==0
    assert fee.assess_delivery(c,AT.date(),'other','EXERCISE')==5


def test_historical_rules_require_complete_asof_record():
    from qqq_sim.broker.rule_registry import RuleRegistry,RuleRecord
    with pytest.raises(BrokerError,match='BROKER_RULE_HISTORY_MISSING'):
        RuleRegistry().resolve(AT.date(),AT,'historical_rules')
    partial=RuleRecord('partial',date(2020,1,1),date(2030,1,1),AT,{'clearing':'.1'},'historical_rules','fixture')
    with pytest.raises(BrokerError,match='BROKER_RULE_HISTORY_MISSING'):
        RuleRegistry([partial]).resolve(AT.date(),AT,'historical_rules')


def test_conditional_fee_cap_failure_cannot_truncate_fee_or_book_fill():
    from dataclasses import replace
    b,s,l,quotes,draft=spread_fixture()
    order=b.submit(replace(draft,max_total_fees=D(1)),AT,quotes)
    assert not b.on_quotes([Quote(q.contract,q.bid,q.ask,AT+timedelta(seconds=1),q.quote_id+'new') for q in quotes],AT+timedelta(seconds=1))
    assert order.observations[-1]['detail']=='ORDER_TOTAL_FEE_CAP_EXCEEDED'
    assert b.ledger.economic_cash==5000 and not b.ledger.positions()


def test_success_requires_actual_close_and_is_absorbing():
    b,s,l,o=open_spread(); c=b.ledger.campaigns['campaign']
    close=OrderDraft('campaign',(OrderLeg(s,1),OrderLeg(l,-1)),D(-40),c.deadline,c.hard_cutoff,intent='CLOSE')
    b.submit(close,AT+timedelta(seconds=2),list(b.quotes.values()),'close')
    b.on_quotes([Quote(s,D('.9'),D('.9'),AT+timedelta(seconds=3),'sclose'),Quote(l,D('.5'),D('.5'),AT+timedelta(seconds=3),'lclose')],AT+timedelta(seconds=3))
    c=b.ledger.campaigns['campaign']
    assert c.target_realized and c.net_cash==D('57.50') and c.target==50
    with pytest.raises(BrokerError,match='CAMPAIGN_ALREADY_SUCCEEDED'):
        b.submit(close,AT+timedelta(seconds=4),list(b.quotes.values()),'repeat')


def test_assignment_stock_reduction_uses_later_quote_and_preserves_fifo():
    b,s,l,o=open_spread()
    at=AT+timedelta(minutes=1)
    b.lifecycle.deliver('assign','campaign',s,1,at,AT.date()+timedelta(days=1))
    assert b.ledger.shares()==-100
    order=b.submit_stock_reduction('campaign',100,D(-107),at,'cover')
    assert b.on_stock_quote(106,107,at,'same-instant')==[]
    assert len(b.on_stock_quote(106,107,at+timedelta(seconds=1),'cover-quote'))==1
    assert b.ledger.shares()==0 and b.ledger.positions()==[(l,1)]
    assert order.status=='FILLED' and order.fees==D('.08')
    assert b.ledger.campaigns['campaign'].realized_gross==0
    assert Broker.from_dict(b.to_dict()).to_dict()==b.to_dict()


def test_multi_lot_partial_close_allocates_every_fee_cent():
    from qqq_sim.accounting.ledger import Ledger,Campaign,Lot
    from qqq_sim.broker.models import Fill
    ledger=Ledger(); option=contract()
    ledger.campaigns['c']=Campaign('c',AT+timedelta(days=3),AT+timedelta(days=4),'LONG_ONLY',False,
                                    status='ACTIVE',target=D(50),trade_cash=D(-400))
    ledger.lots=[Lot('lot'+str(i),option,'c',1,D(100)) for i in range(4)]
    fill=Fill('close3','order','c',option,-3,D(2),AT,'quote',D('.25'),{})
    ledger.apply_fill(fill,AT.date()+timedelta(days=1))
    assert ledger.campaigns['c'].realized_fees==D('.25')
    assert ledger.campaigns['c'].realized_gross==300
    assert len(ledger.lots)==1 and ledger.lots[0].basis==100


def test_settlement_event_order_identical_after_serialized_restart():
    b,s,l,o=open_spread()
    restored=Broker.from_dict(b.to_dict())
    at=AT+timedelta(days=2)
    b.settle_due(at); restored.settle_due(at)
    assert restored.to_dict()==b.to_dict()


def test_second_open_cannot_redefine_initial_partial_package():
    b,s,l,quotes,draft=spread_fixture(2)
    b.submit(draft,AT,quotes,'initial')
    b.on_quotes([Quote(q.contract,q.bid,q.ask,AT+timedelta(seconds=1),q.quote_id+'partial',1,1) for q in quotes],AT+timedelta(seconds=1))
    with pytest.raises(BrokerError,match='CAMPAIGN_INITIALIZATION_IN_PROGRESS'):
        b.submit(draft,AT+timedelta(seconds=2),b.quotes,'unrelated-initial-package')


def test_financing_subcent_cumulative_recognition_survives_restart():
    from qqq_sim.accounting.ledger import Campaign
    b=Broker(); b.ledger.settled_cash_cents=-1000
    b.ledger.campaigns['finance']=Campaign('finance',AT+timedelta(days=20),AT+timedelta(days=21),'EXPLICIT_FINANCING',False)
    due=date(2026,11,30)
    for i in range(2): b.financing.accrue(b.ledger,'finance',AT+timedelta(days=i),due)
    b=Broker.from_dict(b.to_dict())
    b.financing.accrue(b.ledger,'finance',AT+timedelta(days=2),due)
    # Three .003055... accruals recognize .01; daily isolated rounding loses it.
    assert b.ledger.campaigns['finance'].expenses==D('.01')
    before=b.to_dict()
    b.financing.accrue(b.ledger,'finance',AT+timedelta(days=2),due)
    assert b.to_dict()==before

