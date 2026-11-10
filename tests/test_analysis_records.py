from dataclasses import replace
from datetime import datetime,timedelta
from decimal import Decimal as D
import json
import pytest

from qqq_sim.config import AppConfig
from qqq_sim.cli import demo_config
from qqq_sim.data.fixtures import write_synthetic_dataset
from qqq_sim.replay.calendar import local_datetime
from qqq_sim.replay.controller import ReplayController
from qqq_sim.services.adapters import broker_contract
from qqq_sim.services.analysis_inputs import prepare_analysis
from qqq_sim.services.analytic_records import register_analysis_request,record_archived_analysis
from qqq_sim.serialization import digest,primitive
from qqq_sim.broker.models import OrderDraft,OrderLeg,BrokerError


@pytest.fixture
def setup(tmp_path):
    root=tmp_path/'fixture'; write_synthetic_dataset(root,minutes=45)
    config=replace(demo_config(AppConfig(),root),runtime_root=str(tmp_path/'runtime'))
    controller=ReplayController.create(config,local_datetime(datetime(2026,11,9,10,1)))
    contracts={(e.contract.right,e.contract.strike):broker_contract(e.contract,controller.data.calendar) for e in controller.snapshot.quotes}
    deadline=local_datetime(datetime(2026,11,13,15,30))
    draft=OrderDraft('draft',(OrderLeg(contracts['P',D(600)],-1),OrderLeg(contracts['P',D(595)],1)),D(50),deadline,deadline+timedelta(minutes=30))
    yield controller,draft
    controller.close()


def test_freeze_does_not_change_economic_state_and_copies_inputs(setup):
    c,draft=setup; version=c.version; state=digest(c._state())
    prepared=prepare_analysis(c,draft,'selection')
    frozen=c.store.analysis_request(prepared['request_hash'])
    assert frozen['payload_hash']==digest(frozen['payload'])
    assert frozen['snapshot_payload_hash']==digest(primitive(c.snapshot))
    assert frozen['payload']['draft']['campaign_id']=='draft'
    prepared['job']['target_usd']='999999'
    assert c.store.analysis_request(prepared['request_hash'])['payload']['request']['job']['target_usd']!='999999'
    assert c.version==version and digest(c._state())==state
    assert c._analytic_record_cache.bytes<=c._analytic_record_cache.max_bytes


def test_result_idempotency_cache_and_conflicting_replacement(setup):
    c,draft=setup; prepared=prepare_analysis(c,draft,'selection'); version=c.version
    result={'status':'UNSUPPORTED','reason':'PACK_MISSING','hold_probability':None}
    first=c.record_analysis(prepared['identity'],result,prepared['request_hash'])
    duplicate=c.record_analysis(prepared['identity'],dict(result),prepared['request_hash'])
    assert first['record']==duplicate['record'] and first['eligible_for_current_display']
    assert first['automatic_action'] is False and c.version==version
    repeat=prepare_analysis(c,draft,'selection')
    assert repeat['cached_result']==result
    with pytest.raises(BrokerError,match='ANALYSIS_RESULT_CONFLICT'):
        c.record_analysis(prepared['identity'],{'status':'OK','hold_probability':.8})
    prepared['job']['target_usd']='999999'
    with pytest.raises(BrokerError,match='ANALYSIS_REQUEST_CONFLICT'):
        register_analysis_request(c,prepared,draft=draft)
    assert c.store.analysis_result(prepared['request_hash'])['payload']['result']==result


def test_stale_branch_results_export_original_inputs_without_economic_commit(setup):
    c,draft=setup; prepared=prepare_analysis(c,draft,'selection'); original_branch=c.branch_id
    c.advance(minutes=1); c.branch(reason='Inspect different future')
    version=c.version; state=digest(c._state())
    result={'status':'UNSUPPORTED','reason':'PACK_NOT_QUALIFIED','managed_probability':None}
    archived=c.record_analysis(prepared['identity'],result)
    assert archived['historical'] and not archived['eligible_for_current_display']
    record=c.export_record()['analyses'][0]
    assert record['branch_id']==original_branch and record['identity']['state_version']==prepared['identity'].state_version
    assert record['result_record']['payload']['result']==result
    assert c.version==version and digest(c._state())==state
    restored=ReplayController.restore(c.config,c.session_id)
    try: assert restored.export_record()['analyses']==c.export_record()['analyses']
    finally: restored.close()


def test_result_archives_original_session_after_session_switch(setup):
    from qqq_sim.storage.events import EventStore
    c,draft=setup; prepared=prepare_analysis(c,draft,'selection')
    second=ReplayController.create(c.config,c.decision_at)
    c.store.close()
    try:
        state=digest(second._state()); version=second.version
        result={'status':'CANCELED','reason':'SELECTION_CHANGED'}
        archived=second.record_analysis(prepared['identity'],result,prepared['request_hash'],prepared['archive_path'])
        assert archived['historical'] and not archived['eligible_for_current_display']
        assert second.export_record()['analyses']==[]
        with EventStore(prepared['archive_path']) as store:
            assert store.analysis_records(session_id=c.session_id)[0]['result_record']['payload']['result']==result
        assert second.version==version and digest(second._state())==state
    finally:
        second.close()
        c.store=EventStore(prepared['archive_path'])


def test_pending_and_null_failure_results_are_exported(setup):
    c,draft=setup; prepared=prepare_analysis(c,draft,'pending')
    assert c.export_record()['analyses'][0]['null_reason']=='ANALYSIS_PENDING'
    with pytest.raises(BrokerError,match='CURRENT_CAMPAIGN_REQUIRED') as caught:
        prepare_analysis(c,draft,'failed-input',current_position=True)
    metadata=caught.value.analysis_record
    assert metadata['already_archived'] and metadata['selection_hash']=='failed-input'
    assert metadata['state_version']==c.version and 'identity' not in metadata
    records=c.export_record()['analyses']
    failed=next(x for x in records if x['selection_hash']=='failed-input')
    assert failed['result_record']['payload']['result']['reason']=='CURRENT_CAMPAIGN_REQUIRED'
    assert failed['result_record']['payload']['result']['hold_probability'] is None


def test_original_archive_precedes_same_session_identity_in_restored_backup(setup,tmp_path):
    c,draft=setup
    backup=tmp_path/'older-session.sqlite'
    c.store.backup(backup)
    prepared=prepare_analysis(c,draft,'created-after-backup')
    restored=ReplayController.restore(c.config,str(backup))
    try:
        assert restored.session_id==c.session_id and restored.branch_id==c.branch_id
        assert restored.export_record()['analyses']==[]
        version=restored.version; state=digest(restored._state())
        result={'status':'UNSUPPORTED','reason':'PACK_NOT_QUALIFIED','managed_probability':None}
        recorded=restored.record_analysis(prepared['identity'],result,prepared['request_hash'],prepared['archive_path'])
        assert recorded['historical'] and not recorded['eligible_for_current_display']
        assert recorded['archive_path']==prepared['archive_path']
        assert c.export_record()['analyses'][0]['result_record']['payload']['result']==result
        assert restored.export_record()['analyses']==[]
        assert restored.version==version and digest(restored._state())==state
    finally: restored.close()


def test_worker_missing_pack_is_immediately_archived(setup):
    from qqq_sim.ui.worker import SimulationWorker
    c,draft=setup; worker=SimulationWorker(c.config); worker.controller=c
    outputs=[]; errors=[]
    worker.completed.connect(lambda kind,result:outputs.append((kind,result)))
    worker.failed.connect(lambda kind,result:errors.append((kind,result)))
    worker.execute({'kind':'analyze','draft':draft,'selection_hash':'missing-pack','version':c.version})
    assert not errors
    result=outputs[-1][1]
    assert result['reason']=='PACK_MISSING' and result['already_archived']
    assert 'identity' not in result and result['selection_hash']=='missing-pack'
    frozen=c.export_record()['analyses'][0]
    assert frozen['result_record']['payload']['result']['reason']=='PACK_MISSING'


@pytest.mark.parametrize('target',['request','result','result_identity','request_version'])
def test_analysis_hash_corruption_rejected(setup,target):
    c,draft=setup; prepared=prepare_analysis(c,draft,'selection')
    c.record_analysis(prepared['identity'],{'status':'OK','hold_probability':.3})
    if target=='request':
        c.store.connection.execute("UPDATE analysis_requests SET payload_json='{}'")
    elif target=='result':
        c.store.connection.execute("UPDATE analyses SET payload_json_or_path='{}'")
    elif target=='result_identity':
        payload=c.store.analysis_result(prepared['request_hash'])['payload']
        payload['identity']['branch_id']='wrong-branch'
        c.store.connection.execute('UPDATE analyses SET payload_json_or_path=?',(json.dumps(payload),))
    else:
        c.store.connection.execute('UPDATE analysis_requests SET state_version=1000')
    with pytest.raises(BrokerError,match='ANALYSIS_.*_HASH_MISMATCH'):
        c.export_record()
