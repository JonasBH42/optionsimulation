import sqlite3
import pytest
from qqq_sim.storage import EventStore
from qqq_sim.broker.models import BrokerError


def test_commit_idempotency_restore_and_branch(tmp_path):
    with EventStore(tmp_path/'session.sqlite3') as store:
        ids=store.create_session({'initial_cash_cents':500000},initial_state={'cash':500000})
        branch=ids['branch_id']
        result=store.transact(branch,'open',0,{'action':'open'},
                              [{'event_id':'fill1','event_type':'FILL','payload':{'cash':100},'postings':[{'amount_cents':100,'bucket':'unsettled','category':'premium'}]}],
                              {'accepted':True},{'cash':500100})
        assert result['version']==1
        assert store.transact(branch,'open',0,{},[],{}, {'cash':0})==result
        assert store.restore(branch)['state']=={'cash':500100}
        with pytest.raises(BrokerError,match='STALE_STATE_VERSION'):
            store.transact(branch,'bad',0,{},[],{}, {})
        child=store.branch(branch,{},'test branch')
        store.transact(child['branch_id'],'close',1,{},[],{}, {'cash':500200})
        assert store.restore(branch)['state']=={'cash':500100}
        assert store.restore(child['branch_id'])['state']=={'cash':500200}
        export=store.export(branch,tmp_path/'export.json')
        assert export['restored']['state']=={'cash':500100}


def test_crash_after_commit_recovers_once(tmp_path):
    path=tmp_path/'crash.sqlite3'; store=EventStore(path)
    ids=store.create_session({}); branch=ids['branch_id']
    store.transact(branch,'fill',0,{},[{'event_id':'unique','payload':{'amount':5}}],{'ok':True},{'fills':1})
    store.connection.close()  # Simulated crash before publication/checkpoint.
    with EventStore(path) as recovered:
        assert recovered.restore(branch)['state']=={'fills':1}
        assert recovered.transact(branch,'fill',0,{},[],{}, {})['version']==1


def test_hash_tamper_detected_and_no_projection_trust(tmp_path):
    with EventStore(tmp_path/'tamper.sqlite3') as store:
        branch=store.create_session({})['branch_id']
        store.transact(branch,'one',0,{},[],{}, {'cash':1})
        store.connection.execute("UPDATE projections SET payload_json='{}'")
        assert store.restore(branch)['state']=={'cash':1}
        store.connection.execute("UPDATE events SET payload_json='{}'")
        with pytest.raises(BrokerError,match='EVENT_HASH_MISMATCH'): store.restore(branch)
        # Restore projection so orderly close can checkpoint it; corruption is never repaired silently.
        store.connection.execute("UPDATE projections SET payload_json='{\"version\":1,\"state\":{\"cash\":1}}'")


def test_write_boundary_enforced():
    with pytest.raises(BrokerError,match='WRITE_OUTSIDE_PROJECT_FORBIDDEN'):
        EventStore('C:/forbidden/qqq-test.sqlite3')


def test_unknown_schema_is_not_modified_before_refusal(tmp_path):
    path=tmp_path/'future-schema.sqlite3'
    db=sqlite3.connect(path); db.execute('PRAGMA user_version=99'); db.commit(); db.close()
    before=path.read_bytes()
    with pytest.raises(BrokerError,match='STORAGE_SCHEMA_UNSUPPORTED'): EventStore(path)
    assert path.read_bytes()==before
    assert not path.with_name(path.name+'-wal').exists()


def test_transaction_error_rolls_back_events_postings_commands_and_projection(tmp_path):
    with EventStore(tmp_path/'atomic.sqlite3') as store:
        branch=store.create_session({},initial_state={'cash':5000})['branch_id']
        def deny_projection(action,arg1,arg2,database,source):
            if action in (sqlite3.SQLITE_INSERT,sqlite3.SQLITE_UPDATE) and arg1=='projections': return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        store.connection.set_authorizer(deny_projection)
        with pytest.raises(sqlite3.DatabaseError):
            store.transact(branch,'not-committed',0,{},[{'event_id':'fill','payload':{'amount':100},
                'postings':[{'amount_cents':100,'bucket':'unsettled','category':'premium'}]}],{}, {'cash':5100})
        store.connection.set_authorizer(None)
        assert store.restore(branch)['state']=={'cash':5000}
        assert store.connection.execute('SELECT count(*) FROM events').fetchone()[0]==0
        assert store.connection.execute('SELECT count(*) FROM cash_postings').fetchone()[0]==0
        assert store.connection.execute('SELECT count(*) FROM commands').fetchone()[0]==0

