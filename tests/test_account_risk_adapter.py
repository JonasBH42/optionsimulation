from datetime import datetime,timedelta
from decimal import Decimal as D
from dataclasses import replace
from zoneinfo import ZoneInfo
from qqq_sim.risk.contracts import RiskLeg,ReferenceJob,utc_us
from qqq_sim.risk.account_adapter import create_isolated_account_context,account_proposal_check,advance_isolated_account_context
from qqq_sim.risk.policy import make_action,next_opportunity
from qqq_sim.broker.models import fingerprint
import pytest


def case():
    ny=ZoneInfo('America/New_York')
    start=datetime(2026,11,9,10,tzinfo=ny); fixing=datetime(2026,11,13,16,tzinfo=ny)
    intervals=tuple((utc_us(datetime(2026,11,9+i,9,30,tzinfo=ny)),utc_us(datetime(2026,11,9+i,16,tzinfo=ny))) for i in range(5))
    legs=(RiskLeg('SYNTHETIC:short','CALL',D(105),-1,utc_us(fixing),.25,bid=2.,ask=2.,known_at_utc_us=utc_us(start)),
          RiskLeg('SYNTHETIC:long','CALL',D(110),1,utc_us(fixing),.25,bid=1.,ask=1.,known_at_utc_us=utc_us(start)))
    job=ReferenceJob(utc_us(start),utc_us(fixing-timedelta(hours=1)),utc_us(fixing),utc_us(fixing),intervals,
                     100.,.04,.01,.25,legs,D(100),D(50),snapshot_id='SYNTHETIC_ACCOUNT_ADAPTER_TEST')
    campaign={'target_usd':'50','gross_cash_usd':'100','net_cash_usd':'97.75','remaining_expense_budget_usd':'100',
              'hard_cutoff_utc_us':utc_us(fixing),'operational_model':'SIMULTANEOUS_DELIVERY_HEDGE_V1',
              'protection_instruction_recorded':True}
    return job,campaign


def test_isolated_synthetic_account_is_real_ledger_and_preview_is_pure():
    job,campaign=case()
    limits=create_isolated_account_context(job,campaign,job.signed_legs)
    before=fingerprint(limits['broker_state'])
    proposal=make_action(job,'FULL_CLOSE',tuple(replace(leg,quantity=-leg.quantity) for leg in job.signed_legs),next_opportunity(job),campaign)
    check=account_proposal_check(job,campaign,proposal,limits)
    assert check['status']=='OK' and check['request_grouping_nodes']>0
    assert fingerprint(limits['broker_state'])==before


def test_missing_account_context_never_approves():
    job,campaign=case()
    assert account_proposal_check(job,campaign,{'family':'HOLD','legs':[]})['reason']=='ACCOUNT_CONTEXT_UNAVAILABLE'


def test_synthetic_next_opportunity_uses_actual_account_fees_and_no_fill():
    job,campaign=case(); context=create_isolated_account_context(job,campaign,job.signed_legs)
    when=next_opportunity(job)
    proposal=make_action(job,'FULL_CLOSE',tuple(replace(leg,quantity=-leg.quantity) for leg in job.signed_legs),when,campaign)
    # The stored close limit is100. Improved future sides close for50 plus.25.
    quoted=(replace(job.signed_legs[0],bid=1.,ask=1.,known_at_utc_us=when),
            replace(job.signed_legs[1],bid=.5,ask=.5,known_at_utc_us=when))
    result=advance_isolated_account_context(context,job,proposal,quoted,when)
    assert result['filled'] and result['positions']=={}
    assert D(result['premium_cost_usd'])==50 and D(result['fees_usd'])==D('.25')
    assert D(result['net_cash_usd'])==D('47.50') and D(result['target_usd'])==50
    worse=(replace(job.signed_legs[0],bid=3.,ask=3.,known_at_utc_us=when),quoted[1])
    failure=advance_isolated_account_context(context,job,proposal,worse,when)
    assert not failure['filled'] and failure['positions']=={leg.contract_id:leg.quantity for leg in job.signed_legs}
    assert D(failure['fees_usd'])==0 and D(failure['net_cash_usd'])==D('97.75')


def test_preview_and_transition_preserve_order_type_and_time_in_force(monkeypatch):
    from qqq_sim.broker.service import Broker
    job,campaign=case();context=create_isolated_account_context(job,campaign,job.signed_legs)
    # Conventional mode permits LIMIT; strict mode rejects it as unbounded.
    context['broker_state']['strict']=False
    when=next_opportunity(job)
    proposal=make_action(job,'FULL_CLOSE',tuple(replace(leg,quantity=-leg.quantity) for leg in job.signed_legs),when,campaign)
    proposal.update(order_type='LIMIT',tif='GTC')
    captured=[];original=Broker.submit
    def submit(self,draft,*args,**kwargs):
        captured.append((draft.order_type,draft.tif))
        return original(self,draft,*args,**kwargs)
    monkeypatch.setattr(Broker,'submit',submit)
    assert account_proposal_check(job,campaign,proposal,context)['status']=='OK'
    quoted=(replace(job.signed_legs[0],bid=1.,ask=1.,known_at_utc_us=when),
            replace(job.signed_legs[1],bid=.5,ask=.5,known_at_utc_us=when))
    assert advance_isolated_account_context(context,job,proposal,quoted,when)['filled']
    assert captured==[('LIMIT','GTC'),('LIMIT','GTC')]


@pytest.mark.parametrize('field,value,reason',[('order_type','UNKNOWN','UNSUPPORTED_ORDER_TYPE'),('tif','IOC','UNSUPPORTED_TIME_IN_FORCE')])
def test_unsupported_execution_fields_are_typed_refusals(field,value,reason):
    from qqq_sim.risk.contracts import RiskError
    job,campaign=case();context=create_isolated_account_context(job,campaign,job.signed_legs)
    when=next_opportunity(job)
    proposal=make_action(job,'FULL_CLOSE',tuple(replace(leg,quantity=-leg.quantity) for leg in job.signed_legs),when,campaign)
    proposal[field]=value
    assert account_proposal_check(job,campaign,proposal,context)['reason']==reason
    with pytest.raises(RiskError) as error:
        advance_isolated_account_context(context,job,proposal,job.signed_legs,when)
    assert error.value.code=='ACCOUNT_TRANSITION_UNSUPPORTED' and error.value.details['reason']==reason


def test_day_expiration_precedes_delayed_next_session_fill_while_gtc_survives():
    job,campaign=case();context=create_isolated_account_context(job,campaign,job.signed_legs)
    when=job.t0_utc_us+86400_000000
    proposal=make_action(job,'FULL_CLOSE',tuple(replace(leg,quantity=-leg.quantity) for leg in job.signed_legs),when,campaign)
    quoted=(replace(job.signed_legs[0],bid=1.,ask=1.,known_at_utc_us=when),
            replace(job.signed_legs[1],bid=.5,ask=.5,known_at_utc_us=when))
    day=advance_isolated_account_context(context,job,{**proposal,'tif':'DAY'},quoted,when)
    gtc=advance_isolated_account_context(context,job,{**proposal,'tif':'GTC'},quoted,when)
    assert not day['filled'] and day['order_status']=='EXPIRED'
    assert day['positions']=={leg.contract_id:leg.quantity for leg in job.signed_legs}
    assert gtc['filled'] and not gtc['positions']
