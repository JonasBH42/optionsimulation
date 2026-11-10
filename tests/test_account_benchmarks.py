from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from qqq_sim.cli import demo_config
from qqq_sim.config import AppConfig
from qqq_sim.data.fixtures import write_synthetic_dataset
from qqq_sim.data.models import DataError
from qqq_sim.broker.models import BrokerError, OrderDraft, OrderLeg, fingerprint
from qqq_sim.replay.calendar import local_datetime
from qqq_sim.replay.controller import ReplayController
from qqq_sim.services.adapters import broker_contract


def start(tmp_path, monkeypatch, qualified=False):
    root=tmp_path/'fixture'
    write_synthetic_dataset(root,minutes=45)
    config=replace(demo_config(AppConfig(),root),runtime_root=str(tmp_path/'runtime'))
    if qualified:
        path=tmp_path/'qualified-test-pack'; path.mkdir()
        (path/'manifest.json').write_text('{"policy_id":"test-policy"}',encoding='utf-8')
        config=replace(config,pack_path=str(path))
        from qqq_sim.services import analysis_inputs
        # Qualification itself is tested by the model loader; this test drives
        # the asynchronous accounting protocol with a declared controlled pack.
        fake=SimpleNamespace(manifest={'policy_id':'test-policy'},require_qualified=lambda capability:None)
        monkeypatch.setattr(analysis_inputs,'cached_analysis_pack',lambda controller:fake)
    c=ReplayController.create(config,local_datetime(datetime(2026,11,9,10,1)))
    contracts={(entry.contract.right,entry.contract.strike):broker_contract(entry.contract,c.data.calendar) for entry in c.snapshot.quotes}
    deadline=local_datetime(datetime(2026,11,13,15,30))
    draft=OrderDraft('campaign',(OrderLeg(contracts['P',Decimal(600)],-1),OrderLeg(contracts['P',Decimal(595)],1)),Decimal(50),deadline,deadline+timedelta(minutes=30))
    c.submit(draft,c.version,'opening')
    c.advance(minutes=1,command_id='fill')
    return c


def response(request,hold=.42,policy=.38):
    return {'campaign_id':request['campaign_id'],'benchmark_id':request['benchmark_id'],
            'pack_hash':request['pack_hash'],'policy_id':request['policy_id'],'status':'KNOWN',
            'p0_hold':hold,'p0_policy':policy,'p0_policy_range':[.35,.41],'reason':None,'offline_calls':0}


def test_missing_pack_freezes_unknown_actual_fill_basis(tmp_path,monkeypatch):
    c=start(tmp_path,monkeypatch)
    try:
        campaign=c.broker.ledger.campaigns['campaign']
        assert campaign.benchmark_status=='UNKNOWN' and campaign.benchmark_reason=='PACK_MISSING'
        assert campaign.p0_hold is campaign.p0_policy is None
        assert campaign.initialized_at<c.decision_at
        assert campaign.initial_fill_basis['entry_market_snapshot']['asof']==campaign.initialized_at.isoformat()
        assert campaign.initial_fill_basis['entry_market_snapshot']['timing_profile']=='EVENT_TIME_INCLUSIVE_V1'
        assert campaign.initial_fill_basis['net_cash_usd']==str(campaign.net_cash)
        original=fingerprint(campaign.initial_fill_basis)
        c.advance(minutes=2)
        assert fingerprint(campaign.initial_fill_basis)==original
        assert not c.bundle().pending_benchmarks
    finally:c.close()


def test_pending_benchmark_uses_frozen_inputs_after_advance_and_is_idempotent(tmp_path,monkeypatch):
    c=start(tmp_path,monkeypatch,True)
    try:
        campaign=c.broker.ledger.campaigns['campaign']
        assert campaign.benchmark_status=='PENDING',campaign.benchmark_reason
        request=c.prepare_benchmark('campaign')
        initial_time=request['job']['t0_utc_us']
        assert request['job']['entry_cash_usd']==str(campaign.original_entry_cash)
        assert c.bundle().pending_benchmarks==({'campaign_id':'campaign','benchmark_id':request['benchmark_id']},)
        before=c.broker.ledger.economic_cash
        c.advance(minutes=2)
        assert c.prepare_benchmark('campaign')['job']['t0_utc_us']==initial_time
        result=response(request)
        c.record_benchmark('campaign',request['benchmark_id'],result,c.version,'benchmark-complete')
        assert campaign.p0_hold==.42 and campaign.p0_policy==.38
        assert campaign.benchmark_status=='KNOWN'
        assert c.broker.ledger.economic_cash==before
        version=c.version
        c.record_benchmark('campaign',request['benchmark_id'],result,0,'retry')
        assert c.version==version
        with pytest.raises(BrokerError,match='ALREADY_RECORDED'):
            c.record_benchmark('campaign',request['benchmark_id'],response(request,.5),c.version,'conflicting')
        expected=c.broker.to_dict(); session=c.session_id; config=c.config
        c.close(); c=None
        c=ReplayController.restore(config,session)
        assert c.broker.to_dict()==expected
    finally:
        if c:c.close()


def test_result_identity_and_save_failure_leave_pending_state(tmp_path,monkeypatch):
    c=start(tmp_path,monkeypatch,True)
    try:
        request=c.prepare_benchmark('campaign'); before=c.broker.to_dict(); version=c.version
        wrong=response(request); wrong['pack_hash']='changed'
        with pytest.raises(BrokerError,match='MODEL_MISMATCH'):
            c.record_benchmark('campaign',request['benchmark_id'],wrong,c.version,'wrong')
        assert c.broker.to_dict()==before
        monkeypatch.setattr(c.store,'transact',lambda *a,**k:(_ for _ in ()).throw(OSError('test full disk')))
        with pytest.raises(BrokerError) as error:
            c.record_benchmark('campaign',request['benchmark_id'],response(request),c.version,'failed')
        assert error.value.code=='SAVE_FAILED'
        assert c.broker.to_dict()==before and c.version==version
    finally:c.close()


def test_restore_rejects_new_partition_added_while_application_closed(tmp_path,monkeypatch):
    c=start(tmp_path,monkeypatch)
    session,config=c.session_id,c.config
    source=next(partition.path for partition in c.data.catalog.report.partitions if partition.key.data_type=='greeks')
    c.close()
    source=Path(source)
    target=source.with_name(source.name.replace('2026-11-13','2026-11-20'))
    target.write_bytes(source.read_bytes())
    with pytest.raises(DataError,match='membership changed'):
        ReplayController.restore(config,session)


def test_last_opportunity_requires_feasible_close():
    from qqq_sim.risk.offline.design import reference_fixture_job
    from qqq_sim.risk.policy import mandatory_last_opportunity
    from qqq_sim.risk.contracts import RiskError
    job=reference_fixture_job(duration_seconds=61)
    when=(job.t0_utc_us//60_000_000+1)*60_000_000
    hold={'family':'HOLD','eligible_utc_us':when,'premium_cost_usd':'0','legs':[],'id':'hold'}
    close={**hold,'family':'FULL_CLOSE','id':'close'}
    campaign={'hard_cutoff_utc_us':job.fixing_utc_us,'active_deadline_utc_us':job.deadline_utc_us}
    assert mandatory_last_opportunity(job,campaign,[hold,close]) is close
    with pytest.raises(RiskError,match='MANDATORY_LIQUIDATION_UNAVAILABLE'):
        mandatory_last_opportunity(job,campaign,[hold])
