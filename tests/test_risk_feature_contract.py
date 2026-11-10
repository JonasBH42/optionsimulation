from copy import deepcopy
from dataclasses import replace
from datetime import datetime,timedelta
from decimal import Decimal as D
import math
from types import SimpleNamespace
import numpy as np
import pytest

from qqq_sim.risk.contracts import RiskError,RiskLeg,ReferenceJob,NY,YEAR_US,utc_us,canonical,stable_hash
from qqq_sim.risk.feature_layout import (FEATURE_SCHEMA,TOKEN_FIELDS,ACCOUNT_FIELDS,CAMPAIGN_FIELDS,ACTION_FIELDS,
    ORDER_TYPES,TIME_IN_FORCE,HOLD_FEATURE_COUNT,MANAGED_STATE_COUNT,MANAGED_FEATURE_COUNT,
    ACTION_SCALAR_OFFSET,ACTION_TOKEN_OFFSET,UNIVERSE_OFFSET,ACCOUNT_MASK_OFFSET,CAMPAIGN_OFFSET,
    KINK_OFFSET,SHOCK_MASK_OFFSET,SHOCK_MASK_COUNT,MISSING_SHOCK_SENTINEL,feature_contract,hold_masks,managed_masks)
from qqq_sim.risk.features import hold_features,managed_state_features,feature_missingness_key,coverage_check
from qqq_sim.risk.policy import action_features,action_fees
from qqq_sim.risk.account_features import account_risk_features
from qqq_sim.risk.account_adapter import create_isolated_account_context
from qqq_sim.risk.offline.design import reference_fixture_job
from qqq_sim.risk.offline.universe_train import initialize_universe_arrays
from qqq_sim.risk.pricing_fast import european_price
from qqq_sim.risk.qualification import regime_key


def context():
    job=reference_fixture_job()
    campaign={'net_cash_usd':'-202.25','target_usd':'150','loss_cap_usd':'1000','hard_cutoff_utc_us':job.fixing_utc_us,
        'account_features':{'values':{name:str(index+1) for index,name in enumerate(ACCOUNT_FIELDS)}}}
    pack=SimpleNamespace(arrays=initialize_universe_arrays(),price_basis=None)
    action={'family':'HOLD','premium_cost_usd':'0','fees_usd':'0','limit_cost_usd':'0','max_total_fees_usd':'50',
        'target_usd':'150','eligible_utc_us':job.t0_utc_us+60_000_000,'legs':[],
        'order_type':'PROTECTED_LIMIT','tif':'DAY','stop_after_fill':False}
    return job,campaign,pack,action


def test_contract_layout_and_unscaled_masks_match_actual_features():
    job,campaign,pack,action=context(); contract=feature_contract()
    hold=hold_features(job,pack,campaign);managed=managed_state_features(job,pack,campaign)
    action_values=action_features(job,campaign,action,pack)
    assert hold.shape==(HOLD_FEATURE_COUNT,) and managed.shape==(MANAGED_STATE_COUNT,)
    assert action_values.shape==(MANAGED_FEATURE_COUNT,)
    assert all(value.dtype==np.float64 and np.all(np.isfinite(value)) for value in (hold,managed,action_values))
    assert contract['schema']==FEATURE_SCHEMA
    assert (contract['hold_count'],contract['managed_state_count'],contract['managed_count'])==(len(hold),len(managed),len(action_values))
    assert np.array_equal(managed[:HOLD_FEATURE_COUNT],hold)
    assert np.array_equal(action_values[:MANAGED_STATE_COUNT],managed)
    assert all(0<=index<HOLD_FEATURE_COUNT for index in hold_masks())
    assert all(0<=index<MANAGED_FEATURE_COUNT for index in managed_masks())
    assert len(set(managed_masks()))==len(managed_masks())
    assert set(range(SHOCK_MASK_OFFSET,KINK_OFFSET)).issubset(hold_masks())
    assert set(range(ACCOUNT_MASK_OFFSET,CAMPAIGN_OFFSET)).issubset(managed_masks())
    assert ACTION_SCALAR_OFFSET+ACTION_FIELDS.index('stop_after_fill') in managed_masks()


@pytest.mark.parametrize('field,value,slot',[
    ('limit_cost_usd','17.50','limit_cost_div_A'),
    ('max_total_fees_usd','75.00','max_fees_div_A'),
    ('target_usd','200.00','original_target_div_A'),
    ('stop_after_fill',True,'stop_after_fill')])
def test_exact_action_constraints_change_only_their_declared_feature(field,value,slot):
    job,campaign,pack,action=context()
    before=action_features(job,campaign,action,pack)
    after=action_features(job,campaign,{**action,field:value},pack)
    index=ACTION_SCALAR_OFFSET+ACTION_FIELDS.index(slot)
    assert np.flatnonzero(before!=after).tolist()==[index]
    assert after[index]==pytest.approx(float(value) if field=='stop_after_fill' else float(value)/1000)
    assert stable_hash(action)==stable_hash(context()[-1])


@pytest.mark.parametrize('field,value,choices',[('order_type','LIMIT',ORDER_TYPES),('order_type','MARKET',ORDER_TYPES),('tif','GTC',TIME_IN_FORCE)])
def test_order_type_and_tif_distinctions_survive_encoding(field,value,choices):
    job,campaign,pack,action=context()
    before=action_features(job,campaign,action,pack)
    after=action_features(job,campaign,{**action,field:value},pack)
    start=ACTION_SCALAR_OFFSET+len(ACTION_FIELDS)+(len(ORDER_TYPES) if field=='tif' else 0)
    expected=np.zeros(len(choices));expected[choices.index(value)]=1
    assert np.array_equal(after[start:start+len(choices)],expected)
    changed=set(np.flatnonzero(before!=after))
    assert changed=={start+choices.index(action[field]),start+choices.index(value)}


@pytest.mark.parametrize('field,value',[('order_type','UNKNOWN'),('tif','IOC')])
def test_unsupported_execution_enum_does_not_alias_a_valid_action(field,value):
    job,campaign,pack,action=context()
    with pytest.raises(RiskError,match='EXECUTION_PROFILE_UNSUPPORTED'):
        action_features(job,campaign,{**action,field:value},pack)


def test_action_token_keeps_its_actual_fixing_and_stop_distinct_from_hold_expiry():
    job,campaign,pack,action=context()
    leg=replace(job.signed_legs[0],fixing_utc_us=job.fixing_utc_us+86400_000_000,
                last_trading_utc_us=job.fixing_utc_us+86340_000_000)
    values=action_features(job,campaign,{**action,'family':'HEDGE','legs':[canonical(leg)]},pack)
    token=values[ACTION_TOKEN_OFFSET:ACTION_TOKEN_OFFSET+len(TOKEN_FIELDS)]
    assert token[TOKEN_FIELDS.index('fixing_years')]==pytest.approx((leg.fixing_utc_us-job.t0_utc_us)/YEAR_US)
    assert token[TOKEN_FIELDS.index('last_trade_years')]==pytest.approx((leg.last_trading_utc_us-job.t0_utc_us)/YEAR_US)
    assert token[TOKEN_FIELDS.index('fixing_years')]!=token[TOKEN_FIELDS.index('last_trade_years')]
    assert not values[ACTION_TOKEN_OFFSET+len(TOKEN_FIELDS):UNIVERSE_OFFSET].any()


def account_case():
    start=datetime(2026,11,9,10,tzinfo=NY); fixing=datetime(2026,11,13,16,tzinfo=NY)
    intervals=tuple((utc_us(datetime(2026,11,9+i,9,30,tzinfo=NY)),utc_us(datetime(2026,11,9+i,16,tzinfo=NY))) for i in range(5))
    legs=(RiskLeg('SYNTHETIC:short','CALL',D(105),-1,utc_us(fixing),.25,bid=2.,ask=2.,known_at_utc_us=utc_us(start)),
          RiskLeg('SYNTHETIC:long','CALL',D(110),1,utc_us(fixing),.25,bid=1.,ask=1.,known_at_utc_us=utc_us(start)))
    job=ReferenceJob(utc_us(start),utc_us(fixing-timedelta(hours=1)),utc_us(fixing),utc_us(fixing),intervals,
        100.,.04,0.,.25,legs,D(100),D(50),snapshot_id='SYNTHETIC_FEATURE_ACCOUNT_TEST')
    campaign={'campaign_id':'features','target_usd':'50','gross_cash_usd':'100','net_cash_usd':'97.75',
        'remaining_expense_budget_usd':'100','hard_cutoff_utc_us':utc_us(fixing),
        'operational_model':'SIMULTANEOUS_DELIVERY_HEDGE_V1','protection_instruction_recorded':True}
    return job,campaign,create_isolated_account_context(job,campaign,legs)


def test_account_numeric_evidence_is_authoritative_read_only_and_changes_features():
    job,campaign,source=account_case(); before=stable_hash(source)
    evidence=account_risk_features(source)
    assert stable_hash(source)==before
    assert evidence['status']=='KNOWN'
    values=evidence['values']
    assert D(values['economic_cash'])==D('5097.75') and D(values['collateral'])==500
    assert D(values['available_option_bp'])==D('4597.75')
    assert D(values['equity'])==D('4997.75') and D(values['account_economic_pnl'])==D('-2.25')
    assert D(values['deficit'])==0
    pack=SimpleNamespace(price_basis=None)
    original=managed_state_features(job,pack,{**campaign,'account_features':evidence})
    changed=deepcopy(source);changed['broker_state']['bp_cap']='4000'
    other=account_risk_features(changed)
    after=managed_state_features(job,pack,{**campaign,'account_features':other})
    slot=HOLD_FEATURE_COUNT+ACCOUNT_FIELDS.index('bp_cap')
    assert original[slot]==5 and after[slot]==4
    assert np.flatnonzero(original!=after).tolist()==[slot]
    assert stable_hash(source)==before


def test_managed_transition_refreshes_account_features_and_preserves_prior_state():
    from qqq_sim.risk.offline.control_oracle import ManagedState,transition
    from qqq_sim.risk.policy import make_action,next_opportunity
    job,campaign,source=account_case()
    campaign['account_features']={'values':{'equity':'999999'}}
    campaign['remaining_loss_budget_usd']='999999'
    campaign_before=stable_hash(campaign); source_before=stable_hash(source)
    state=ManagedState(job,campaign,job.signed_legs,account_context=source)
    assert stable_hash(campaign)==campaign_before and stable_hash(source)==source_before
    assert state.campaign['account_features']['values']['equity']!='999999'
    assert state.campaign['remaining_loss_budget_usd']!='999999'
    state_before=stable_hash(state)
    close=make_action(job,'EXIT_STOP',tuple(replace(leg,quantity=-leg.quantity) for leg in job.signed_legs),next_opportunity(job),campaign)
    after=transition(state,close,0.,SimpleNamespace(price_basis=None))
    assert after.events[-1]['status']=='FILLED' and not after.job.signed_legs
    assert after.campaign['adjustment_count']==1 and after.campaign['stopped_after_exit']
    assert D(after.campaign['account_features']['values']['collateral'])==0
    assert D(state.campaign['account_features']['values']['collateral'])==500
    assert after.campaign['account_features']==account_risk_features(after.account_context,'features')
    assert after.campaign['remaining_loss_budget_usd']==after.campaign['account_features']['values']['campaign_remaining_loss_budget']
    assert after.campaign['remaining_loss_budget_usd']!=state.campaign['remaining_loss_budget_usd']
    assert stable_hash(state)==state_before and stable_hash(source)==source_before


def test_malformed_account_context_has_typed_failure():
    from qqq_sim.risk.offline.control_oracle import ManagedState
    job,campaign,pack,action=context()
    with pytest.raises(RiskError) as caught:
        ManagedState(job,campaign,(),account_context={'broker_state':{'corrupt':'missing authoritative fields'}})
    assert caught.value.code=='INVALID_ACCOUNT_CONTEXT'


@pytest.mark.parametrize('legacy',['schema','dimensions','missing_contract'])
def test_qualified_head_rejects_legacy_feature_contract_before_network_use(tmp_path,legacy):
    from qqq_sim.risk.pack_loader import ModelPack
    manifest={'pack_id':'CONTROLLED_SCHEMA_REJECTION_TEST','feature_schema':FEATURE_SCHEMA,'feature_contract':feature_contract(),
        'hold_label_schema':{'version':'QQQ_HOLD_LABEL_V2'},'hold_loss_schema':{'version':'QQQ_HOLD_LOSS_V2'}}
    if legacy=='schema':manifest['feature_schema']='EXACT_TOKENS_PAYOFF_SHAPES_V1'
    elif legacy=='dimensions':manifest['feature_contract']={**feature_contract(),'hold_count':786,'managed_count':1032}
    else:manifest.pop('feature_contract')
    # This controlled flag only reaches the schema guard; there are no trained
    # weights or claimed numerical accuracy in this test object.
    pack=ModelPack(tmp_path,manifest,{}, {'capabilities':{'hold':{'qualified':True,'within_bin_validation':{'qualified':True}}}})
    with pytest.raises(RiskError) as caught:pack.require_qualified('hold')
    assert caught.value.code==('FEATURE_SCHEMA_MISMATCH' if legacy=='schema' else 'FEATURE_CONTRACT_MISMATCH')


def test_unknown_account_value_has_a_distinct_mask_from_observed_zero():
    job,campaign,pack,action=context()
    known={name:'0' for name in ACCOUNT_FIELDS}
    missing={**known,'deficit':None}
    first=managed_state_features(job,pack,{**campaign,'account_features':{'values':known}})
    second=managed_state_features(job,pack,{**campaign,'account_features':{'values':missing}})
    offset=ACCOUNT_FIELDS.index('deficit')
    assert first[HOLD_FEATURE_COUNT+offset]==second[HOLD_FEATURE_COUNT+offset]==0
    assert first[ACCOUNT_MASK_OFFSET+offset]==1 and second[ACCOUNT_MASK_OFFSET+offset]==0
    assert feature_missingness_key(first)=='COMPLETE'
    assert feature_missingness_key(second)!='COMPLETE'
    unavailable=account_risk_features(None)
    assert unavailable['status']=='UNAVAILABLE' and all(x is None for x in unavailable['values'].values())


def test_kinks_and_target_gaps_preserve_gross_net_cash_and_executable_costs():
    job,campaign,pack,action=context();features=hold_features(job,pack,campaign)
    kinks=features[KINK_OFFSET:KINK_OFFSET+32].reshape(8,4)
    assert np.allclose(kinks[:2],[[1,0,-.2,-.20225],[1,math.log(1.05),.3,.29775]])
    assert not kinks[2:].any()
    tau=(job.fixing_utc_us-job.t0_utc_us)/YEAR_US
    gross=float(job.entry_cash_usd);net=float(campaign['net_cash_usd']);changes=[]
    for leg in job.signed_legs:
        fair=float(european_price(leg.right,job.x0,float(leg.strike),tau,job.rate_cc,job.dividend_yield_cc,leg.sigma))
        gross+=leg.quantity*leg.multiplier*fair
        bid=math.floor(max(0,fair-leg.half_spread)/leg.tick)*leg.tick
        ask=math.ceil((fair+leg.half_spread)/leg.tick)*leg.tick
        net+=leg.quantity*leg.multiplier*(bid if leg.quantity>0 else ask)
        changes.append(replace(leg,quantity=-leg.quantity,bid=bid,ask=ask))
    fee=float(action_fees(job,tuple(changes)));assert fee>0
    net-=fee
    assert features[-2:]==pytest.approx([(gross-150)/1000,(net-150)/1000])
    assert features[-1]<features[-2]


class ShockDomain:
    def __init__(self,error='PRICE_BASIS_UNVALIDATED_CELL',current_failure=False):
        self.error,self.current_failure=error,current_failure
    def evaluate(self,right,spot,strike,tau,r,q,sigma,**kwargs):
        if self.current_failure or sigma<.19 and spot>100 or sigma>.21 and spot<80:
            raise RiskError(self.error)
        return {'price':max(spot-strike,0)+sigma,'error':0.}


def shock_case(pack_basis):
    job,campaign,pack,action=context()
    job=replace(job,dividend_yield_cc=.01,signed_legs=tuple(replace(leg,exercise_style='AMERICAN') for leg in job.signed_legs))
    pack.price_basis=pack_basis
    return job,campaign,pack


def test_unavailable_shocks_are_masked_without_fabricating_zero_response():
    job,campaign,pack=shock_case(ShockDomain());before=job.hash
    features=hold_features(job,pack,campaign)
    spots=job.x0*np.exp(np.linspace(-math.log(2),math.log(2),65))
    expected=np.concatenate((spots>100,spots<80))
    assert np.array_equal(features[SHOCK_MASK_OFFSET:SHOCK_MASK_OFFSET+SHOCK_MASK_COUNT],expected)
    curve_offset=12+8*len(TOKEN_FIELDS)
    shocks=features[curve_offset+4*65:curve_offset+6*65]
    assert np.all(shocks[expected]==MISSING_SHOCK_SENTINEL)
    assert np.all(shocks[~expected]!=MISSING_SHOCK_SENTINEL) and np.all(np.isfinite(features))
    assert features[-2]!=MISSING_SHOCK_SENTINEL and job.hash==before
    assert feature_missingness_key(features)!='COMPLETE'
    altered=features.copy();altered[0]+=1
    assert feature_missingness_key(altered)==feature_missingness_key(features)


@pytest.mark.parametrize('reason',['PRICE_BASIS_CROSS_FACE_FAILURE','PACK_REVISION_CHANGED','PRICE_TREE_LOOKUP_LIMIT','MODEL_MARKS_INVALID'])
def test_corrupt_or_failed_shock_prices_remain_hard_failures(reason):
    job,campaign,pack=shock_case(ShockDomain(reason))
    with pytest.raises(RiskError) as caught:hold_features(job,pack,campaign)
    assert caught.value.code==reason


def test_missing_current_mark_is_not_downgraded_to_a_shock_mask():
    job,campaign,pack=shock_case(ShockDomain(current_failure=True))
    with pytest.raises(RiskError,match='PRICE_BASIS_UNVALIDATED_CELL'):
        hold_features(job,pack,campaign)


def test_missingness_requires_a_separately_admitted_coverage_regime():
    job,campaign,pack,action=context();complete=hold_features(job,pack,campaign)
    missing=complete.copy();missing[SHOCK_MASK_OFFSET]=1
    normal=regime_key(job,campaign,features=complete);unknown=regime_key(job,campaign,features=missing)
    assert normal!=unknown and normal.endswith('features=COMPLETE')
    gate={'qualified':True,'calibration_count':200,'eighth_neighbor_distance_99':1.}
    packed=SimpleNamespace(arrays={'hold_mean':np.zeros(HOLD_FEATURE_COUNT),'hold_std':np.ones(HOLD_FEATURE_COUNT),
        'hold_training_encodings':np.tile(complete,(8,1))},manifest={'coverage':{'hold':{'regimes':{normal:gate}}}})
    assert np.array_equal(coverage_check(complete,packed,regime=normal)[0],complete)
    with pytest.raises(RiskError,match='MODEL_COVERAGE_UNAVAILABLE'):
        coverage_check(missing,packed,regime=unknown)
