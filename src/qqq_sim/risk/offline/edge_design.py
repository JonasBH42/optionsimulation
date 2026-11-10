"""Mandatory untouched edge families with actual cashflows and checked topology.

These records are separate from the equally stratified base Latin hypercube.
Reserved expiries keep their actual contract universes out of all base anchor
windows; the complete groups remain validation-only in every adaptive cycle.
"""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime,timedelta
from decimal import Decimal
import math
import numpy as np

from ..contracts import NY,YEAR_US,RiskLeg,ReferenceJob,NumericalSettings,RiskError,utc_us,canonical
from ..payoff import payoff_extrema,terminal_regions
from ..policy import action_fees,make_action
from ..account_adapter import create_isolated_account_context,advance_isolated_account_context
from ...broker.service import Broker
from ...replay.calendar import CalendarService


FAMILIES=("NARROW","TANGENT","DISJOINT","NEAR_EXPIRY","LOW_BUDGET")
RESERVED_DATES=("2026-04-15","2026-05-13","2026-08-12","2026-09-16","2027-02-10")


def heldout_design_records(start_index,per_family=200):
    from .design import design_record,SEED
    if isinstance(start_index,bool) or int(start_index)!=start_index or start_index<0 or per_family<1:
        raise RiskError('INVALID_HELDOUT_DESIGN_COUNT')
    records=[]
    for family_number,family in enumerate(FAMILIES):
        for local_index in range(per_family):
            index=int(start_index)+len(records)
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(SEED,spawn_key=(5,family_number,local_index))))
            coordinates=rng.random(12)
            record=design_record(index,coordinates)
            record.update(heldout_edge_family=family,heldout_local_index=local_index,
                split='validation',shape_group='MANDATORY_HELDOUT:'+family,
                campaign_group='MANDATORY_HELDOUT:'+family,
                universe_group='RESERVED_EXPIRY:'+RESERVED_DATES[family_number],
                seed={'entropy':SEED,'spawn_key':[5,family_number,local_index]},
                anchor_date=RESERVED_DATES[family_number],spot=float(120+(37*local_index)%1331),
                sigma_anchor=.25+.001*(local_index%200),r=0.,q=0.,skew=0.,curvature=0.,term_slope=0.,
                quantity=2 if family in ('NARROW','TANGENT') else 1,expiry_count=1,
                half_spread=.01,desired_seconds=60 if family=='NEAR_EXPIRY' else 3600,
                training_eligible=False,calibration_eligible=False,augmentation_seed_eligible=False,
                provenance='MANDATORY_UNTOUCHED_SYNTHETIC_DESIGN',
                status='HELDOUT_COORDINATES_REQUIRE_REAL_LEDGER_AND_TOPOLOGY_CHECK')
            records.append(record)
    return records


def _cash(legs,opening=True):
    return -sum((leg.quantity*leg.multiplier*Decimal(str(leg.ask if leg.quantity>0 else leg.bid)) for leg in legs),Decimal(0))


def materialize_heldout_case(record,*,research=False,price_provider=None,numerical_settings=None,deadline=None):
    from .design import _check_deadline,_price,_quoted,_path_sigma,_surface_quality,_reports
    from ..pricing_fast import european_price
    from .design import SEED
    _check_deadline(deadline)
    family=record.get('heldout_edge_family')
    if family not in FAMILIES or research or record.get('split')!='validation' or record.get('training_eligible') is not False:
        raise RiskError('HELDOUT_DESIGN_PARTITION_VIOLATION')
    if record.get('calibration_eligible') is not False or record.get('augmentation_seed_eligible') is not False:
        raise RiskError('HELDOUT_DESIGN_PARTITION_VIOLATION')
    record=dict(record)
    index=int(record['index']); local_index=int(record['heldout_local_index'])
    family_number=FAMILIES.index(family)
    if record['anchor_date']!=RESERVED_DATES[family_number]:
        raise RiskError('HELDOUT_RESERVED_UNIVERSE_CHANGED')
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(SEED,spawn_key=(5,family_number,local_index))))
    record['generator_state_before']=canonical(rng.bit_generator.state)
    calendar=CalendarService()
    session=calendar.session(datetime.fromisoformat(record['anchor_date']).date())
    if session.status!='OPEN':
        raise RiskError('HELDOUT_CALENDAR_UNAVAILABLE')
    fixing=utc_us(session.underlying_close)
    seconds=(60,120,300)[local_index%3] if family=='NEAR_EXPIRY' else 3600
    opening=fixing-seconds*1_000_000
    center=float(record['spot']); sigma=float(record['sigma_anchor'])
    tau=seconds/(365*86400)
    sd=center*sigma*math.sqrt(tau)
    width=max(1,min(4,round(sd/2))) if family in ('NARROW','TANGENT','DISJOINT') else 1
    if family in ('NARROW','TANGENT','DISJOINT'):
        ratios=(1,-2,1) if family!='DISJOINT' else (-1,2,-1)
        strikes=(center-width,center,center+width)
    else:
        ratios=(1,-1);strikes=(center,center+width)
    quantity=2 if family in ('NARROW','TANGENT') else 1
    contracts={}
    # r=q=0 makes both American rights equal their exact European values.
    # This deliberately selected mathematical edge domain needs no approximate
    # trained basis to manufacture the accounting/topology fixture.
    provider=european_price
    record['pricing_reference_identity']='EXACT_AMERICAN_R0_Q0_EQUALS_EUROPEAN'
    for strike in range(int(center)-max(6,3*width),int(center)+max(6,3*width)+1):
        for right in ('CALL','PUT'):
            key=(right,float(strike))
            contracts[key]=RiskLeg(f'SYNTHETIC_HELDOUT:{record["anchor_date"]}:{right}:{strike}',right,Decimal(str(strike)),1,
                                   fixing,sigma,half_spread=.01,last_trading_utc_us=fixing)
    def quote_all(spot,when,templates=contracts):
        quoted={}
        for key,leg in templates.items():
            value=_price(leg.right,spot,float(leg.strike),(fixing-when)/YEAR_US,0.,0.,leg.sigma,provider,deadline)
            result=_quoted(leg,value,when)
            quoted[key]=result
        _surface_quality(quoted.values())
        return quoted
    def position(quotes,multiplier=1):
        return tuple(replace(quotes['CALL',strike],quantity=ratio*multiplier) for strike,ratio in zip(strikes,ratios))
    planned_close=None
    opening_spot=center
    attempts=[]
    if family in ('NARROW','TANGENT'):
        # Half of a two-unit opening is closed for the original per-unit debit.
        # The surviving one-unit maximum therefore equals originalH0 exactly;
        # an extra$1 close credit gives a2-cent-wide terminal success interval.
        gap=Decimal(1 if family=='NARROW' else 0)
        for distance in (2.,1.75,2.25,1.5,2.5):
            opening_spot=center-distance*sd
            quotes=quote_all(opening_spot,opening)
            unit=position(quotes)
            debit=-_cash(unit)
            desired=debit+gap
            frozen={key:replace(leg,half_spread=(leg.ask-leg.bid)/2) for key,leg in quotes.items()}
            when=opening+60_000_000
            def close_credit(spot):
                selected={key:leg for key,leg in frozen.items() if key[0]=='CALL' and key[1] in strikes}
                future=quote_all(spot,when,selected)
                return _cash(tuple(replace(leg,quantity=-leg.quantity) for leg in position(future)))
            left,right=opening_spot,center
            low,high=close_credit(left),close_credit(right)
            attempts.append({'opening_distance_sigma':distance,'opening_spot':opening_spot,
                             'unit_debit_usd':str(debit),'desired_close_credit_usd':str(desired),
                             'bracket_credits_usd':[str(low),str(high)]})
            if not low<=desired<=high or debit<=0 or debit>=width*100:
                continue
            for iteration in range(120):
                _check_deadline(deadline)
                middle=(left+right)/2
                actual=close_credit(middle)
                if actual==desired:
                    planned_close=(middle,when,desired,frozen)
                    break
                if actual<desired:left=middle
                else:right=middle
                if middle==left and right-left<1e-12:
                    break
            if planned_close:
                break
        if planned_close is None:
            raise RiskError('HELDOUT_EXACT_LEDGER_TOPOLOGY_UNRESOLVED',family=family,attempts=attempts)
        record['exact_close_search']=attempts
    else:
        quotes=quote_all(opening_spot,opening)
    legs=position(quotes,quantity)
    universe=tuple(replace(leg,half_spread=(leg.ask-leg.bid)/2) for leg in quotes.values())
    cash=_cash(legs)
    maximum=payoff_extrema(legs,cash)['max_profit']
    if not maximum.is_finite() or maximum<=0:
        raise RiskError('HELDOUT_NONPOSITIVE_OR_UNBOUNDED_ORIGINAL_TARGET')
    target=maximum/2
    intervals=((utc_us(session.exchange_open),fixing),)
    sigma_path,path_evidence=_path_sigma(universe,opening_spot,opening,fixing,0.,0.,provider,deadline)
    reports=_reports(opening,fixing,intervals,rng,False)
    job=ReferenceJob(opening,fixing-1,fixing,fixing,intervals,opening_spot,0.,0.,sigma_path,legs,cash,target,
        snapshot_id=f'SYNTHETIC_HELDOUT_{family}_{index}',calendar_profile_id='SYNTHETIC_QQQ_CONTRACTS_V1',
        numerical_settings=numerical_settings or NumericalSettings(),reporting_times_utc_us=reports,
        parameter_provenance=({'source':'MANDATORY_UNTOUCHED_SYNTHETIC_DESIGN','known_at_utc_us':opening,'path_iv_selection':path_evidence},))
    fees=action_fees(job,legs,current_legs=())
    campaign={'campaign_id':f'SYNTHETIC_HELDOUT_{family}_{index}','target_usd':str(target),
        'original_max_profit_usd':str(maximum),'gross_cash_usd':str(cash),'net_cash_usd':str(cash-fees),
        'remaining_expense_budget_usd':'10.01' if family=='LOW_BUDGET' else '100',
        'initial_deadline_utc_us':fixing-1,'active_deadline_utc_us':fixing-1,'hard_cutoff_utc_us':fixing,
        'remaining_actions':1 if family=='LOW_BUDGET' else 12,'remaining_rolls':0,
        'operational_model':'SIMULTANEOUS_DELIVERY_HEDGE_V1','protection_instruction_recorded':True,
        'initial_entry_events':[{'premium_cash_usd':str(cash),'fees_usd':str(fees),'event_at_utc_us':opening,
                                'provenance':'SYNTHETIC_EXECUTED_OPENING'}],
        'declared_adjustments':0,'executed_adjustments':0,'adjustment_events':[],'target_realized':False}
    context=create_isolated_account_context(job,campaign,universe)
    if planned_close:
        current_spot,when,desired,frozen=planned_close
        future=quote_all(current_spot,when,frozen)
        changes=tuple(replace(leg,quantity=-leg.quantity//2) for leg in legs)
        action=make_action(job,'PARTIAL_CLOSE',changes,when,campaign)
        outcome=advance_isolated_account_context(context,job,action,tuple(future.values()),when)
        if not outcome['filled'] or -Decimal(outcome['premium_cost_usd'])!=desired:
            raise RiskError('HELDOUT_EXACT_CLOSE_NOT_FILLED',outcome=outcome)
        context=outcome['context'];lookup={leg.contract_id:leg for leg in future.values()}
        legs=tuple(replace(lookup[key],quantity=n) for key,n in sorted(outcome['positions'].items()))
        for key in ('gross_cash_usd','net_cash_usd','remaining_expense_budget_usd'):
            campaign[key]=outcome[key]
        campaign.update(remaining_actions=11,declared_adjustments=1,executed_adjustments=1,
                        adjustment_events=[{'action':action,'status':'FILLED','premium_cost_usd':outcome['premium_cost_usd'],
                                            'fees_usd':outcome['fees_usd'],'event_at_utc_us':when,'broker_state_hash':outcome['broker_state_hash']}])
        dt=(when-opening)/YEAR_US
        normal=(math.log(current_spot/opening_spot)+.5*sigma_path*sigma_path*dt)/(sigma_path*math.sqrt(dt))
        campaign['adjustment_events'][0]['declared_gbm_normal']=normal
        universe=tuple(future.values())
        job=replace(job,t0_utc_us=when,x0=current_spot,signed_legs=legs,entry_cash_usd=Decimal(campaign['gross_cash_usd']),
                    reporting_times_utc_us=tuple(t for t in reports if t>=when),job_schema='MANAGED_STATE_V1')
    topology=terminal_regions(job.signed_legs,job.entry_cash_usd,job.target_usd)
    current_maximum=payoff_extrema(job.signed_legs,job.entry_cash_usd)['max_profit']
    audit={'intended_family':family,'terminal_intervals':canonical(topology),
           'original_target_usd':str(target),'current_maximum_gross_usd':str(current_maximum),
           'original_target_retained':job.target_usd==target,'validation_only':True}
    if family=='TANGENT':
        passed=current_maximum==target and len(topology)==1 and topology[0].lower==topology[0].upper
    elif family=='NARROW':
        passed=current_maximum-target==Decimal(1) and len(topology)==1 and 0<topology[0].upper-topology[0].lower<=.020000001
    elif family=='DISJOINT':
        passed=len(topology)>=2 and all(left.upper<right.lower for left,right in zip(topology,topology[1:]))
    elif family=='NEAR_EXPIRY':
        passed=0<job.fixing_utc_us-job.t0_utc_us<=300_000_000
        audit['remaining_seconds']=(job.fixing_utc_us-job.t0_utc_us)/1e6
    else:
        broker=Broker.from_dict(context['broker_state'])
        from ...broker.risk_checks import certify_campaign
        certificate=certify_campaign(broker.ledger,campaign['campaign_id'],broker.loss_cap)
        passed=campaign['remaining_actions']==1 and Decimal(campaign['remaining_expense_budget_usd'])==Decimal('10.01') and certificate.lower_bound is not None
        audit.update(remaining_actions=1,remaining_expense_budget_usd='10.01',operational_bound=canonical(certificate))
    audit['verified']=bool(passed)
    if not passed:
        raise RiskError('HELDOUT_EDGE_PROPERTY_NOT_ESTABLISHED',audit=audit)
    record.update(edge_validation=audit,path_iv_selection=path_evidence,actual_remaining_seconds=(fixing-job.t0_utc_us)/1e6,
                  generator_state_after=canonical(rng.bit_generator.state),status='MATERIALIZED_VERIFIED_HELDOUT_EDGE',
                  original_opening_spot=opening_spot,original_opening_utc_us=opening,quote_provenance='SYNTHETIC_DESIGN')
    return {'job':job,'campaign':campaign,'universe':universe,'record':record,'account_context':context,
            'scope':'MANDATORY_UNTOUCHED_EDGE_DESIGN','generator_limitations':[]}
