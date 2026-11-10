from dataclasses import replace
from datetime import datetime,timedelta,date
from decimal import Decimal as D
import json
from pathlib import Path
import pytest

from qqq_sim.risk.contracts import RiskLeg,ReferenceJob,RiskError,utc_us
from qqq_sim.risk.offline.design import reference_fixture_job
from qqq_sim.risk.universe_admission import admit_known_universe
from qqq_sim.risk.universe import universe_tokens


def quoted_job():
    job=reference_fixture_job(duration_seconds=86400)
    legs=tuple(replace(leg,bid=1.,ask=1.01,known_at_utc_us=job.t0_utc_us,last_trading_utc_us=job.last_trading_utc_us) for leg in job.signed_legs)
    return replace(job,signed_legs=legs)


def test_oversize_chain_retains_holdings_and_audits_bounded_expiry_contract_search():
    job=quoted_job();day=86400_000_000
    observed=list(job.signed_legs)
    for offset in range(11):
        for strike in range(50,151):
            for right in ('CALL','PUT'):
                observed.append(RiskLeg(f'{offset}:{right}:{strike}',right,D(strike),1,job.fixing_utc_us+offset*day,.2,
                    bid=1.,ask=1.01,known_at_utc_us=job.t0_utc_us,last_trading_utc_us=job.fixing_utc_us+offset*day))
    admitted,audit=admit_known_universe(job,observed,job.fixing_utc_us+12*day)
    assert len(admitted)==512
    assert len({leg.fixing_utc_us for leg in admitted})<=8
    assert {leg.contract_id for leg in job.signed_legs}<={leg.contract_id for leg in admitted}
    assert len(audit['considered_expiries_utc_us'])==8
    assert {'DECLARED_EXPIRY_SEARCH_CAP','DECLARED_CONTRACT_SEARCH_CAP'}<={row['reason'] for row in audit['excluded_contracts']}
    assert admit_known_universe(job,reversed(observed),job.fixing_utc_us+12*day)==(admitted,audit)
    assert universe_tokens(job,admitted).shape==(512,16)


def test_current_expiry_after_hard_cutoff_is_preserved_but_new_extension_is_excluded():
    job=quoted_job();hard=job.t0_utc_us+60_000_000
    future=replace(job.signed_legs[0],contract_id='new-expiry',fixing_utc_us=job.fixing_utc_us+86400_000_000,
                   last_trading_utc_us=job.fixing_utc_us+86400_000_000)
    unseen=replace(job.signed_legs[0],contract_id='future-quote',known_at_utc_us=job.t0_utc_us+1)
    admitted,audit=admit_known_universe(job,[future,unseen],hard)
    assert {leg.contract_id for leg in admitted}=={leg.contract_id for leg in job.signed_legs}
    assert all(leg.fixing_utc_us==job.fixing_utc_us for leg in admitted)
    assert {row['reason'] for row in audit['excluded_contracts']}=={'EXPIRY_EXTENDS_BEYOND_HARD_CUTOFF','FUTURE_UNIVERSE_RECORD'}


def test_current_holdings_cannot_be_silently_truncated():
    job=quoted_job()
    with pytest.raises(RiskError) as error:
        admit_known_universe(job,job.signed_legs,job.fixing_utc_us,max_contracts=1)
    assert error.value.code=='UNSUPPORTED_PORTFOLIO_SIZE'


@pytest.fixture
def controller(tmp_path):
    from qqq_sim.cli import demo_config
    from qqq_sim.config import AppConfig
    from qqq_sim.data.fixtures import write_synthetic_dataset
    from qqq_sim.replay.controller import ReplayController
    from qqq_sim.replay.calendar import local_datetime
    from qqq_sim.broker.models import OrderDraft,OrderLeg
    from qqq_sim.services.adapters import broker_contract
    root=tmp_path/'fixture';write_synthetic_dataset(root,minutes=45)
    config=replace(demo_config(AppConfig(),root),runtime_root=str(tmp_path/'runtime'))
    c=ReplayController.create(config,local_datetime(datetime(2026,11,9,10,1)))
    keys={(entry.contract.right,entry.contract.strike):broker_contract(entry.contract,c.data.calendar) for entry in c.snapshot.quotes}
    deadline=local_datetime(datetime(2026,11,13,15,30));hard=local_datetime(datetime(2026,11,30,16))
    draft=OrderDraft('campaign',(OrderLeg(keys['P',D(600)],-1),OrderLeg(keys['P',D(595)],1)),D(50),deadline,hard)
    c.submit(draft,c.version,'opening');c.advance(minutes=1)
    yield c
    c.close()


def holding_draft(c):
    from qqq_sim.broker.models import OrderDraft,OrderLeg
    campaign=c.broker.ledger.campaigns['campaign']
    return OrderDraft('campaign',tuple(OrderLeg(contract,quantity) for contract,quantity in c.broker.ledger.positions('campaign')),
                      D(0),campaign.deadline,campaign.hard_cutoff,intent='ADJUST')


def test_causal_multi_expiry_preparation_carries_real_stop_times(controller):
    from qqq_sim.services.analysis_inputs import prepare_analysis
    c=controller;expiry=date(2026,11,20)
    parameters=json.loads(Path(c.config.parameters_path).read_text(encoding='utf-8'))
    parameters['expirations']={expiry.isoformat():{'fixing_at':'2026-11-20T16:00:00-05:00','sigma_path':.7}}
    path=Path(c.config.runtime_root)/'future-expiry.json';path.write_text(json.dumps(parameters),encoding='utf-8')
    c.select_inputs(parameters_path=str(path))
    additions=[]
    for entry in c.snapshot.quotes:
        key=replace(entry.contract,expiration=expiry)
        quote=replace(entry.latest_event,contract=key,source_id=entry.latest_event.source_id+':later')
        additions.append(replace(entry,contract=key,quote=quote,latest_event=quote))
    hidden=additions[0]
    key=replace(hidden.contract,strike=D(999))
    quote=replace(hidden.latest_event,contract=key,available_at=c.decision_at+timedelta(seconds=1),source_id='UNSEEN_FUTURE')
    additions.append(replace(hidden,contract=key,quote=quote,latest_event=quote))
    c.snapshot=replace(c.snapshot,quotes=c.snapshot.quotes+tuple(additions))
    result=prepare_analysis(c,holding_draft(c),'multi-expiry',current_position=True)
    universe=tuple(RiskLeg.from_dict(raw) for raw in result['universe'])
    assert len({leg.fixing_utc_us for leg in universe})==2
    expected=utc_us(c.data.calendar.session(expiry).option_close)
    assert all(leg.last_trading_utc_us==expected for leg in universe if leg.fixing_utc_us==utc_us(datetime.fromisoformat('2026-11-20T16:00:00-05:00')))
    assert all(leg.strike!=D(999) for leg in universe)
    assert result['universe_exclusions']['INCOHERENT_SNAPSHOT']==1
    universe_tokens(ReferenceJob.from_dict(result['job']),universe)
    assert result['job']['sigma_path']==parameters['sigma_path']
    managed=result['limits']['managed_job']
    assert managed['sigma_path']==.7
    assert managed['fixing_utc_us']==result['job']['fixing_utc_us']
    assert ReferenceJob.from_dict(managed).signed_legs==ReferenceJob.from_dict(result['job']).signed_legs
    assert all(leg.half_spread==pytest.approx((leg.ask-leg.bid)/2) for leg in universe)
    assert all(leg.half_spread==pytest.approx((leg.ask-leg.bid)/2) for leg in ReferenceJob.from_dict(managed).signed_legs)
    assert managed['target_usd']==result['job']['target_usd']
    assert managed['entry_cash_usd']==result['job']['entry_cash_usd']
    assert result['managed_anchor']['anchor_expiry']=='2026-11-20'
    assert result['managed_anchor']['managed_job_hash']==ReferenceJob.from_dict(managed).hash


def test_oversize_live_preparation_admits512_instead_of_abstaining_whole_chain(controller):
    from qqq_sim.services.analysis_inputs import prepare_analysis
    c=controller
    existing={(entry.contract.right,entry.contract.strike) for entry in c.snapshot.quotes}
    template=c.snapshot.quotes[0];additions=[]
    for strike in range(450,750):
        for right in ('C','P'):
            if (right,D(strike)) in existing:continue
            key=replace(template.contract,right=right,strike=D(strike))
            quote=replace(template.latest_event,contract=key,source_id=f'OVERSIZE:{right}:{strike}')
            additions.append(replace(template,contract=key,quote=quote,latest_event=quote))
    c.snapshot=replace(c.snapshot,quotes=c.snapshot.quotes+tuple(additions))
    result=prepare_analysis(c,holding_draft(c),'oversize',current_position=True)
    assert len(result['universe'])==512
    assert result['universe_admission']['excluded_contracts']
    held={contract.key for contract,quantity in c.broker.ledger.positions('campaign')}
    assert held<={leg['contract_id'] for leg in result['universe']}


def test_fast_analysis_routes_hold_and_managed_laws_separately(monkeypatch):
    from types import SimpleNamespace
    from qqq_sim.services.analysis import AnalysisIdentity,run_fast_analysis
    from qqq_sim.risk.contracts import canonical
    from qqq_sim.services import analysis
    from qqq_sim.risk import hold,envelope
    original=quoted_job();managed=replace(original,sigma_path=.7)
    pack=SimpleNamespace(pack_id='test-pack')
    monkeypatch.setattr(analysis,'_frozen_pack',lambda *args:pack)
    seen=[]
    monkeypatch.setattr(hold,'evaluate_hold',lambda job,pack:{'status':'OK','hold_p50':job.sigma_path})
    def envelope_call(job,campaign,limits,universe,pack,*,hold_job):
        seen.append((hold_job.sigma_path,job.sigma_path))
        return {'status':'OK','hold':hold.evaluate_hold(hold_job,pack),'managed_probability':job.sigma_path}
    monkeypatch.setattr(envelope,'optimize_envelope',envelope_call)
    identity=AnalysisIdentity('s','b',1,'snapshot','selection','campaign','objective','pack','rules')
    campaign={'target_usd':str(original.target_usd)}
    limits={'managed_job':canonical(managed),'managed_anchor':{'status':'OK','anchor_expiry':'declared-later'}}
    _,result=run_fast_analysis(identity,canonical(original),'test-path',campaign,(),limits)
    assert seen==[(.2,.7)]
    assert result['hold']['hold_p50']==.2 and result['managed_probability']==.7
    limits['managed_job']=None;limits['managed_anchor']={'status':'UNSUPPORTED','reason':'MANAGED_ANCHOR_UNAVAILABLE'}
    _,unknown=run_fast_analysis(identity,canonical(original),'test-path',campaign,(),limits)
    assert unknown['hold']['hold_p50']==.2 and unknown['managed_probability'] is None
    assert unknown['reason']=='MANAGED_ANCHOR_UNAVAILABLE'
    assert seen==[(.2,.7)]


def test_initial_benchmark_uses_same_frozen_pack_loader_and_separate_laws(monkeypatch):
    from types import SimpleNamespace
    from qqq_sim.services import analysis
    from qqq_sim.services.benchmarks import run_initial_benchmark
    from qqq_sim.risk import hold,policy
    from qqq_sim.risk.contracts import canonical
    original=quoted_job();managed=replace(original,sigma_path=.7)
    calls=[]
    def load(path,expected):
        calls.append((path,expected));return SimpleNamespace(manifest={'policy_id':'frozen-policy'})
    monkeypatch.setattr(analysis,'_frozen_pack',load)
    monkeypatch.setattr(hold,'evaluate_hold',lambda job,pack:{'status':'OK','hold_p50':job.sigma_path})
    monkeypatch.setattr(policy,'select_policy_action',lambda job,*args:{'action':{'sigma':job.sigma_path}})
    monkeypatch.setattr(policy,'evaluate_policy_action',lambda job,campaign,action,*args:{'status':'OK','q_policy':job.sigma_path,'empirical_error_range':[.65,.75]})
    result=run_initial_benchmark({'campaign_id':'campaign','benchmark_id':'original','pack_path':'frozen-path','pack_hash':'frozen-hash',
        'policy_id':'frozen-policy','job':canonical(original),'universe':[],'campaign':{},
        'limits':{'managed_job':canonical(managed),'managed_anchor':{'status':'OK'}}})
    assert calls==[('frozen-path','frozen-hash')]
    assert result['p0_hold']==.2 and result['p0_policy']==.7 and result['status']=='KNOWN'
