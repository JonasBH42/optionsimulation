from dataclasses import replace
from datetime import datetime,timedelta
from decimal import Decimal as D
import json
from pathlib import Path
import pytest
from qqq_sim.config import AppConfig
from qqq_sim.cli import demo_config
from qqq_sim.data.fixtures import write_synthetic_dataset
from qqq_sim.replay.calendar import local_datetime
from qqq_sim.replay.controller import ReplayController
from qqq_sim.services.adapters import broker_contract
from qqq_sim.services.analysis_inputs import prepare_analysis
from qqq_sim.broker.models import OrderDraft,OrderLeg,BrokerError


@pytest.fixture
def controller(tmp_path):
    root=tmp_path/'fixture'; write_synthetic_dataset(root,minutes=45)
    config=replace(demo_config(AppConfig(),root),runtime_root=str(tmp_path/'runtime'))
    c=ReplayController.create(config,local_datetime(datetime(2026,11,9,10,1)))
    contracts={(entry.contract.right,entry.contract.strike):broker_contract(entry.contract,c.data.calendar) for entry in c.snapshot.quotes}
    deadline=local_datetime(datetime(2026,11,13,15,30))
    draft=OrderDraft('campaign',(OrderLeg(contracts['P',D(600)],-1),OrderLeg(contracts['P',D(595)],1)),D(50),deadline,deadline+timedelta(minutes=30))
    c.submit(draft,c.version,'open'); c.advance(minutes=1)
    yield c
    c.close()


def test_close_preview_keeps_current_hold_and_separates_changes(controller):
    c=controller; campaign=c.broker.ledger.campaigns['campaign']
    holdings=c.broker.ledger.positions('campaign')
    closing=OrderDraft('campaign',tuple(OrderLeg(contract,-q) for contract,q in holdings),D(-100),campaign.deadline,campaign.hard_cutoff,intent='CLOSE')
    result=prepare_analysis(c,closing,'selected-close')
    actual={leg['contract_id']:leg['quantity'] for leg in result['job']['signed_legs']}
    assert actual=={contract.key:q for contract,q in holdings}
    proposal={leg['contract_id']:leg['quantity'] for leg in result['proposal']['legs']}
    assert proposal=={contract.key:-q for contract,q in holdings}
    assert result['job']['entry_cash_usd']==str(campaign.trade_cash+campaign.income)
    assert result['job']['target_usd']==str(campaign.target)
    assert result['proposal']['exact_broker_preview']['status']=='OK'


def test_holding_analysis_has_no_duplicate_addition(controller):
    c=controller; campaign=c.broker.ledger.campaigns['campaign']
    holdings=c.broker.ledger.positions('campaign')
    draft=OrderDraft('campaign',tuple(OrderLeg(contract,q) for contract,q in holdings),D(50),campaign.deadline,campaign.hard_cutoff,intent='ADJUST')
    assert prepare_analysis(c,draft,'hold',current_position=True)['proposal'] is None
    assert prepare_analysis(c,draft,'additional-identical-spread')['proposal'] is not None
    campaign.target=None; campaign.status='INITIALIZING'
    with pytest.raises(BrokerError,match='CAMPAIGN_INITIALIZING'): prepare_analysis(c,draft,'hold')


def test_analytics_separates_option_trading_close_from_fixing_and_replay_coverage(controller):
    from qqq_sim.services.adapters import risk_job
    from qqq_sim.risk.contracts import utc_us,RiskError
    c=controller; campaign=c.broker.ledger.campaigns['campaign']
    held=tuple(OrderLeg(contract,q) for contract,q in c.broker.ledger.positions('campaign'))
    draft=OrderDraft('campaign',held,D(50),campaign.deadline,campaign.hard_cutoff,intent='ADJUST')
    parameters=json.loads(Path(c.config.parameters_path).read_text(encoding='utf-8'))
    context={'target_usd':str(campaign.target),'entry_cash_usd':str(campaign.trade_cash),'deadline':campaign.deadline}
    job=risk_job(c.snapshot,draft,parameters,c.data.calendar,context)
    actual=utc_us(held[0].contract.last_trade_at)
    assert job['last_trading_utc_us']==actual
    assert all(leg['last_trading_utc_us']==actual for leg in job['signed_legs'])
    assert job['last_trading_utc_us']>job['fixing_utc_us']
    assert job['eligible_intervals'][-1][1]==job['fixing_utc_us']
    parameters['last_trading_at']='2026-11-13T15:45:00-05:00'
    overridden=risk_job(c.snapshot,draft,parameters,c.data.calendar,context)
    assert overridden['last_trading_utc_us']==utc_us(parameters['last_trading_at'])
    parameters['last_trading_at']='2026-11-13T15:45:00'
    with pytest.raises(RiskError,match='INVALID_LAST_TRADING_TIMESTAMP'):
        risk_job(c.snapshot,draft,parameters,c.data.calendar,context)


def test_other_campaign_uses_bound_instead_of_cash(controller):
    from qqq_sim.accounting.ledger import Campaign
    c=controller; current=c.broker.ledger.campaigns['campaign']
    c.broker.ledger.campaigns['prior']=Campaign('prior',current.deadline,current.hard_cutoff,'LONG_ONLY',False,
                                               expense_remaining=D(0),status='FLAT',trade_cash=D(-600))
    c.broker.ledger.settled_cash_cents-=60000
    held=tuple(OrderLeg(contract,q) for contract,q in c.broker.ledger.positions('campaign'))
    draft=OrderDraft('campaign',held,D(50),current.deadline,current.hard_cutoff,intent='ADJUST')
    result=prepare_analysis(c,draft,'hold')
    assert result['limits']['other_campaign_net_usd']=='-600'
    assert result['limits']['other_campaign_bound_basis']=='CONSERVATIVE_FINAL_NET_LOWER_BOUNDS'


def test_branch_restores_parameter_selection_and_idempotency(controller):
    c=controller
    old_path=c.config.parameters_path
    old_hash=c._lifecycle_registry.source_hash
    checkpoint=c.store.checkpoint(c.branch_id)
    new_path=Path(c.config.runtime_root)/'new-parameters.json'
    values=json.loads(Path(old_path).read_text(encoding='utf-8')); values['sigma_path']=.3
    new_path.write_text(json.dumps(values),encoding='utf-8')
    c.select_inputs(parameters_path=str(new_path))
    c.advance(minutes=1,command_id='future-step')
    assert 'future-step' in c._advance_ids
    c.branch(checkpoint['sequence'])
    assert c.config.parameters_path==old_path and c._lifecycle_registry.source_hash==old_hash
    assert 'future-step' not in c._advance_ids


def test_normalization_change_cannot_reinterpret_live_session(controller):
    c=controller; path=Path(c.config.normalization_path)
    original=path.read_bytes()
    path.write_bytes(original+b' ')
    try:
        with pytest.raises(BrokerError,match='CALENDAR_OR_NORMALIZATION_REVISION_CHANGED'):
            c.advance(minutes=1)
    finally: path.write_bytes(original)


def test_bar_exclusive_boundary_quote_fills_after_command_at_same_wall_time(tmp_path):
    import csv,gzip
    root=tmp_path/'fixture'; write_synthetic_dataset(root,minutes=45)
    path=next((root/'raw_theta'/'greeks').glob('*.gz'))
    with gzip.open(path,'rt',encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream); fields=reader.fieldnames; rows=list(reader)
    for row in rows:
        if row['timestamp'].startswith('2026-11-09T10:01:30'):
            row['timestamp']='2026-11-09T10:01:00'
    with gzip.open(path,'wt',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    config=replace(demo_config(AppConfig(),root),runtime_root=str(tmp_path/'runtime'))
    at=local_datetime(datetime(2026,11,9,10,1)); c=ReplayController.create(config,at)
    try:
        contracts={(e.contract.right,e.contract.strike):broker_contract(e.contract,c.data.calendar) for e in c.snapshot.quotes}
        deadline=local_datetime(datetime(2026,11,13,15,30))
        draft=OrderDraft('boundary',(OrderLeg(contracts['P',D(600)],-1),OrderLeg(contracts['P',D(595)],1)),D(50),deadline,deadline+timedelta(minutes=30))
        c.submit(draft,c.version,'boundary-open'); c.advance(minutes=1)
        fills=[event for event in c.broker.ledger.events if event['event_type']=='FILL']
        assert len(fills)==2 and all(datetime.fromisoformat(event['at'])==at for event in fills)
    finally: c.close()


def test_save_failure_rolls_back_submission_and_advance_to_durable_prefix(controller,monkeypatch):
    c=controller; campaign=c.broker.ledger.campaigns['campaign']
    draft=OrderDraft('campaign',tuple(OrderLeg(contract,-q) for contract,q in c.broker.ledger.positions('campaign')),
                     D(-100),campaign.deadline,campaign.hard_cutoff,intent='CLOSE')
    before=c.broker.to_dict(); version=c.version; at=c.decision_at
    def fail(*args,**kwargs): raise OSError('injected disk-full write failure')
    monkeypatch.setattr(c.store,'transact',fail)
    with pytest.raises(BrokerError) as raised: c.submit(draft,c.version,'not-committed')
    assert raised.value.code=='SAVE_FAILED'
    assert c.broker.to_dict()==before and c.version==version
    result=c.advance(minutes=2,command_id='failed-advance')
    assert result.status=='PARTIAL' and result.reason['reason']=='SAVE_FAILED'
    assert c.decision_at==at and c.broker.to_dict()==before and c.version==version
    assert 'failed-advance' not in c._advance_ids
    assert c.store.restore(c.branch_id)['state']['broker']==before


def test_lost_commit_acknowledgment_recovers_committed_state_once(controller,monkeypatch):
    c=controller; transact=c.store.transact
    def lost(*args,**kwargs):
        transact(*args,**kwargs)
        raise OSError('injected acknowledgment loss after SQL COMMIT')
    monkeypatch.setattr(c.store,'transact',lost)
    before_version=c.version
    c.advance(minutes=1,command_id='committed-despite-lost-ack')
    assert c.status=='OK' and c.version==before_version+2
    expected=c.broker.to_dict()
    restored=ReplayController.restore(c.config,c.session_id)
    try:
        assert restored.broker.to_dict()==expected
        version=restored.version
        restored.advance(minutes=1,expected_version=0,command_id='committed-despite-lost-ack')
        assert restored.version==version and restored.broker.to_dict()==expected
    finally: restored.close()


def test_cancel_advance_preserves_committed_prefix_and_recovery_status(controller,monkeypatch):
    c=controller; start=c.decision_at; commit=c._commit; batches=[]
    monkeypatch.setattr('qqq_sim.replay.fast_forward.MAX_BATCH_SLOTS',2)
    monkeypatch.setattr('qqq_sim.replay.fast_forward.BATCH_SECONDS',float('inf'))
    def cancel_after_batch(command_id,payload):
        commit(command_id,payload)
        if command_id.startswith('batch:'):
            batches.append(command_id); c.cancel_requested.set()
    monkeypatch.setattr(c,'_commit',cancel_after_batch)
    result=c.advance(minutes=5,command_id='canceled-long-advance')
    assert result.status=='CANCELED' and len(batches)==1
    assert c.decision_at==start+timedelta(minutes=2)
    restored=ReplayController.restore(c.config,c.session_id)
    try:
        assert restored.status=='CANCELED' and restored.broker.to_dict()==c.broker.to_dict()
    finally: restored.close()


def test_export_write_failure_does_not_report_success(controller,monkeypatch):
    c=controller; before=c.broker.to_dict()
    def fail(*args,**kwargs): raise OSError('injected export write failure')
    monkeypatch.setattr('qqq_sim.replay.controller.save_json',fail)
    with pytest.raises(OSError,match='export write failure'):
        c.export(Path(c.config.runtime_root)/'failed-export.json')
    assert c.broker.to_dict()==before


def test_create_failure_closes_sqlite_handle(controller,monkeypatch):
    import sqlite3
    from qqq_sim.storage.events import EventStore
    seen=[]; original=EventStore.__init__
    def capture(self,*args,**kwargs): original(self,*args,**kwargs); seen.append(self)
    def fail(*args,**kwargs): raise BrokerError('INJECTED_BIND_FAILURE')
    monkeypatch.setattr(EventStore,'__init__',capture)
    monkeypatch.setattr(ReplayController,'_bind_day',fail)
    with pytest.raises(BrokerError,match='INJECTED_BIND_FAILURE'):
        ReplayController.create(controller.config,controller.decision_at)
    assert seen
    with pytest.raises(sqlite3.ProgrammingError,match='closed'):
        seen[-1].connection.execute('SELECT 1')


def test_worker_restore_updates_configuration_for_next_session(controller):
    from qqq_sim.ui.worker import SimulationWorker
    c=controller; worker=SimulationWorker(AppConfig()); failures=[]
    worker.failed.connect(lambda kind,error:failures.append((kind,error)))
    worker.execute({'kind':'restore','path':str(c.store.path),'id':'restore'})
    try:
        assert not failures and worker.config==c.config
        restored=worker.controller
        worker.execute({'kind':'open','at':c.decision_at,'id':'new-session'})
        assert not failures and worker.controller is not restored
        assert worker.controller.config.underlying_root==c.config.underlying_root
    finally:
        if worker.controller: worker.controller.close()


def test_worker_failed_session_switch_closes_new_handle_and_keeps_old(controller,monkeypatch):
    import sqlite3
    from qqq_sim.ui.worker import SimulationWorker
    from qqq_sim.storage.events import EventStore
    old=ReplayController.create(controller.config,controller.decision_at)
    worker=SimulationWorker(old.config); worker.controller=old
    failures=[]; worker.failed.connect(lambda kind,error:failures.append(error))
    seen=[]; original_init=EventStore.__init__; original_close=old.close
    def capture(self,*args,**kwargs): original_init(self,*args,**kwargs); seen.append(self)
    def fail_close(): raise OSError('injected previous-session checkpoint failure')
    monkeypatch.setattr(EventStore,'__init__',capture); monkeypatch.setattr(old,'close',fail_close)
    try:
        worker.execute({'kind':'open','at':old.decision_at,'id':'failed-switch'})
        assert failures and worker.controller is old
        assert old.store.connection.execute('SELECT 1').fetchone()[0]==1
        assert seen
        with pytest.raises(sqlite3.ProgrammingError,match='closed'): seen[-1].connection.execute('SELECT 1')
    finally: original_close()


def test_failed_parameter_selection_keeps_worker_and_session_configuration(controller):
    from qqq_sim.ui.worker import SimulationWorker
    c=controller; worker=SimulationWorker(c.config); worker.controller=c
    before=c.config; failures=[]; worker.failed.connect(lambda kind,error:failures.append(error))
    path=Path(c.config.runtime_root)/'invalid-lifecycle.json'
    path.write_text('{"lifecycle_events":[{"id":"missing-required-fields"}]}',encoding='utf-8')
    worker.execute({'kind':'load_parameters','path':str(path),'id':'invalid-input-selection'})
    assert failures and worker.config==before and c.config==before


def test_analysis_gate_requires_identity_and_coordinator_tracks_pending():
    from concurrent.futures import Future
    from qqq_sim.services.analysis import AnalysisCoordinator,AnalysisIdentity,ResultGate
    gate=ResultGate(); assert not gate.accepts(None)
    class FakePool:
        def submit(self,*args): return Future()
        def shutdown(self,**kwargs): self.closed=True
    coordinator=AnalysisCoordinator(); coordinator.pool=FakePool()
    identity=AnalysisIdentity('s','b',1,'snapshot','selection','campaign','objective','pack','rules')
    coordinator.submit(identity,{},None)
    assert coordinator.pending_identity==identity and coordinator.gate.accepts(identity)
    first=coordinator.pending
    assert coordinator.submit(identity,{},None) is first and not first.cancelled()
    coordinator.close()
    assert coordinator.pending_identity is None and not coordinator.gate.accepts(None)


def test_bundle_order_sequence_is_identical_after_restart(controller):
    c=controller; campaign=c.broker.ledger.campaigns['campaign']
    close=OrderDraft('campaign',tuple(OrderLeg(contract,-q) for contract,q in c.broker.ledger.positions('campaign')),
                     D(-100),campaign.deadline,campaign.hard_cutoff,intent='CLOSE')
    c.submit(close,c.version,'chronological-close'); c.advance(minutes=1)
    before=c.bundle()
    restored=ReplayController.restore(c.config,c.session_id)
    try:
        after=restored.bundle()
        assert before.orders==after.orders and before.campaigns==after.campaigns
        assert before.account==after.account and before.positions==after.positions
    finally: restored.close()


def test_roll_analysis_uses_explicit_fixing_for_each_change_leg(controller):
    from datetime import date
    from qqq_sim.risk.contracts import utc_us
    c=controller
    path=Path(c.config.runtime_root)/'expiry-inputs.json'
    params=json.loads(Path(c.config.parameters_path).read_text(encoding='utf-8'))
    params['expirations']={'2026-11-20':{'fixing_at':'2026-11-20T16:00:00-05:00'}}
    path.write_text(json.dumps(params),encoding='utf-8')
    c.select_inputs(parameters_path=str(path))
    additional=[]
    for entry in c.snapshot.quotes:
        contract=replace(entry.contract,expiration=date(2026,11,20))
        quote=replace(entry.latest_event,contract=contract,source_id=entry.latest_event.source_id+'next-expiry')
        additional.append(replace(entry,contract=contract,quote=quote,latest_event=quote))
    c.snapshot=replace(c.snapshot,quotes=c.snapshot.quotes+tuple(additional))
    campaign=c.broker.ledger.campaigns['campaign']
    changes=[OrderLeg(contract,-q) for contract,q in c.broker.ledger.positions('campaign')]
    for entry in additional:
        if entry.contract.right=='P' and entry.contract.strike in (D(595),D(600)):
            changes.append(OrderLeg(broker_contract(entry.contract,c.data.calendar),1 if entry.contract.strike==595 else -1))
    draft=OrderDraft('campaign',tuple(changes),D(-100),campaign.deadline,campaign.hard_cutoff,intent='ROLL')
    result=prepare_analysis(c,draft,'explicit-roll')
    assert len({x['fixing_utc_us'] for x in result['proposal']['legs']})==2
    assert len({x['fixing_utc_us'] for x in result['universe']})==1
    assert {row['reason'] for row in result['universe_admission']['excluded_contracts']}=={'EXPIRY_EXTENDS_BEYOND_HARD_CUTOFF'}
    assert result['proposal']['exact_broker_preview']['reason']=='ROLL_EXPIRY_EXCEEDS_HARD_CUTOFF'
    assert {x['fixing_utc_us'] for x in result['job']['signed_legs']}=={utc_us(local_datetime(datetime(2026,11,13,16)))}
    assert result['job']['target_usd']==str(campaign.target)


def test_flat_campaign_reentry_keeps_empty_hold_and_original_target(controller):
    c=controller;campaign=c.broker.ledger.campaigns['campaign']
    opening=tuple(OrderLeg(contract,q) for contract,q in c.broker.ledger.positions('campaign'))
    closing=OrderDraft('campaign',tuple(OrderLeg(x.contract,-x.quantity) for x in opening),D(-100),campaign.deadline,campaign.hard_cutoff,intent='CLOSE')
    c.submit(closing,c.version,'flatten');c.advance(minutes=1)
    assert not c.broker.ledger.positions('campaign')
    reentry=OrderDraft('campaign',opening,D(50),campaign.deadline,campaign.hard_cutoff,intent='ADJUST')
    result=prepare_analysis(c,reentry,'reentry')
    assert result['job']['signed_legs']==[]
    assert result['job']['job_schema']=='MANAGED_STATE_V1'
    assert result['campaign']['open_contract_quantities']=={}
    assert result['proposal'] is not None
    assert result['job']['target_usd']==str(campaign.target)

