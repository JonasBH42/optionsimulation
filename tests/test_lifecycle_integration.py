from dataclasses import replace
from datetime import date,datetime,timedelta
from decimal import Decimal as D
import json
import pytest
from qqq_sim.config import AppConfig
from qqq_sim.cli import demo_config
from qqq_sim.data.fixtures import write_synthetic_dataset
from qqq_sim.replay.calendar import local_datetime
from qqq_sim.replay.controller import ReplayController
from qqq_sim.services.adapters import broker_contract
from qqq_sim.broker.models import OrderDraft,OrderLeg,BrokerError


def fixture_config(tmp_path,day,events):
    root=tmp_path/'fixture'
    write_synthetic_dataset(root,day=day)
    config=demo_config(AppConfig(),root)
    path=root/'parameters.json'
    parameters=json.loads(path.read_text(encoding='utf-8'))
    parameters['lifecycle_events']=events
    path.write_text(json.dumps(parameters),encoding='utf-8')
    return replace(config,runtime_root=str(tmp_path/'runtime'))


def draft_for(c,deadline):
    keys={(entry.contract.right,entry.contract.strike):broker_contract(entry.contract,c.data.calendar) for entry in c.snapshot.quotes}
    return OrderDraft('campaign',(OrderLeg(keys['P',D(600)],-1),OrderLeg(keys['P',D(595)],1)),D(10),deadline,deadline+timedelta(minutes=1))


def test_official_fixing_consumed_before_paired_expiry(tmp_path):
    day=date(2026,11,13)
    events=[{'id':'close','kind':'OFFICIAL_FIXING','economic_at':'2026-11-13T16:00:00-05:00',
             'known_at':'2026-11-13T16:00:00-05:00','price':'597','expiry':day.isoformat()}]
    config=fixture_config(tmp_path,day,events)
    c=ReplayController.create(config,local_datetime(datetime(2026,11,13,15,57)))
    try:
        c.submit(draft_for(c,local_datetime(datetime(2026,11,13,15,59))),c.version,'entry')
        c.advance(minutes=1)
        assert c.broker.ledger.positions()
        c.advance(minutes=2)
        assert c.status=='OK' and not c.broker.ledger.positions() and c.broker.ledger.shares()==0
        types=[event['event_type'] for event in c.broker.ledger.events]
        assert types.index('LIFECYCLE_INPUT_CONSUMED')<types.index('ASSIGNMENT')
        assert c.broker.ledger.campaigns['campaign'].expenses>D(10)
    finally: c.close()


def test_missing_fixing_preserves_last_complete_minute(tmp_path):
    config=fixture_config(tmp_path,date(2026,11,13),[])
    c=ReplayController.create(config,local_datetime(datetime(2026,11,13,15,57)))
    try:
        c.submit(draft_for(c,local_datetime(datetime(2026,11,13,15,59))),c.version,'entry')
        c.advance(minutes=3)
        assert c.status=='PARTIAL' and c.reason['reason']=='SETTLEMENT_REFERENCE_MISSING'
        assert c.decision_at==local_datetime(datetime(2026,11,13,15,59))
        assert c.broker.ledger.positions()
    finally: c.close()


def test_late_notice_does_not_leak_and_revises_dividend(tmp_path):
    # No contract lookup is inferred from a future chain: an exact supplied
    # standard-product identity denotes an explicit assignment scenario.
    contract={'symbol':'QQQ','right':'P','strike':'600','expiry':'2026-11-13',
              'last_trade_at':'2026-11-13T16:15:00-05:00','metadata_version':'VERIFIED_STANDARD_QQQ_V1'}
    events=[{'id':'div','kind':'DIVIDEND','economic_at':'2026-11-09T10:03:30-05:00','known_at':'2026-11-09T09:30:00-05:00',
             'per_share':'.50','pay_date':'2026-11-10','campaign_id':'campaign'},
            {'id':'assignment','kind':'ASSIGNMENT','economic_at':'2026-11-09T10:03:00-05:00','known_at':'2026-11-09T10:04:30-05:00',
             'contract':contract,'quantity':1,'campaign_id':'campaign'}]
    config=fixture_config(tmp_path,date(2026,11,9),events)
    c=ReplayController.create(config,local_datetime(datetime(2026,11,9,10,1)))
    try:
        c.submit(draft_for(c,local_datetime(datetime(2026,11,13,15,59))),c.version,'entry')
        c.advance(minutes=3)
        assert c.decision_at.hour==10 and c.decision_at.minute==4
        assert c.broker.ledger.shares()==0 and c.broker.ledger.campaigns['campaign'].income==0
        c.advance(minutes=1)
        assert c.status=='OK' and c.broker.ledger.shares()==100
        assert c.broker.ledger.campaigns['campaign'].income==50
        assert c.broker.ledger.campaigns['campaign'].operational_model=='DELAYED_NOTICE_EXPLICIT_SCENARIO'
        assert len(c.broker.ledger.positions())==1
        state=c.broker.to_dict(); c._process_lifecycle_until(c.decision_at)
        assert c.broker.to_dict()==state
        assert all(p['campaign_id']=='campaign' for p in c.bundle().positions)
    finally: c.close()


def test_lifecycle_revision_change_is_rejected_on_restart(tmp_path):
    config=fixture_config(tmp_path,date(2026,11,9),[])
    c=ReplayController.create(config,local_datetime(datetime(2026,11,9,10,1)))
    session_id=c.session_id; c.close()
    with open(config.parameters_path,'a',encoding='utf-8') as handle: handle.write(' ')
    with pytest.raises(BrokerError,match='LIFECYCLE_SOURCE_REVISION_CHANGED'):
        ReplayController.restore(config,session_id)

