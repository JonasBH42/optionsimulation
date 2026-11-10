from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
import json
import pytest

from qqq_sim.config import AppConfig, writable_path, ConfigurationError
from qqq_sim.cli import demo_config
from qqq_sim.data.fixtures import write_synthetic_dataset
from qqq_sim.serialization import digest
from qqq_sim.replay.controller import ReplayController
from qqq_sim.replay.calendar import local_datetime
from qqq_sim.broker.models import OrderDraft, OrderLeg, BrokerError
from qqq_sim.services.adapters import broker_contract
from qqq_sim.services.analysis import AnalysisIdentity, ResultGate


@pytest.fixture
def config(tmp_path):
    root=tmp_path/"fixture"
    write_synthetic_dataset(root,minutes=45)
    result=demo_config(AppConfig(),root)
    return replace(result,runtime_root=str(tmp_path/"runtime"))


def spread(controller):
    contracts={(e.contract.right,e.contract.strike):broker_contract(e.contract,controller.data.calendar) for e in controller.snapshot.quotes}
    deadline=local_datetime(datetime(2026,11,13,15,30))
    return OrderDraft("campaign",(OrderLeg(contracts["P",Decimal(600)],-1),OrderLeg(contracts["P",Decimal(595)],1)),
                      Decimal(50),deadline,deadline+timedelta(minutes=30))


def test_replay_step_size_equivalence_and_restore(config):
    at=local_datetime(datetime(2026,11,9,10,1))
    first=ReplayController.create(config,at)
    second=ReplayController.create(config,at)
    try:
        for c in (first,second):
            c.submit(spread(c),c.version,"same-command")
            assert c.broker.ledger.positions()==[]
        first.advance(minutes=5)
        for _ in range(5):
            second.advance(minutes=1)
        assert first.status==second.status=="OK"
        assert first.broker.ledger.positions()
        assert first.broker.to_dict()==second.broker.to_dict()
        assert first.bundle().account==second.bundle().account
        expected=first.broker.to_dict()
        session=first.session_id
        first.close()
        first=None
        restored=ReplayController.restore(config,session)
        try:
            assert restored.broker.to_dict()==expected
            assert restored.decision_at==second.decision_at
            exported=restored.export(config.runtime_root+"/export.json")
            record=json.loads(exported.read_text(encoding="utf-8"))
            assert record["real_orders_sent"] is False
            assert record["state"]["broker"]==expected
        finally:
            restored.close()
    finally:
        if first:
            first.close()
        second.close()


def test_gap_commits_same_valid_prefix(config):
    # Rewrite only this test's synthetic fixture BEFORE either run binds it.
    root=writable_path(config.underlying_root).parent
    write_synthetic_dataset(root,minutes=45,missing_minutes=(33,))
    at=local_datetime(datetime(2026,11,9,10,1))
    a,b=(ReplayController.create(config,at) for _ in range(2))
    try:
        for c in (a,b):
            c.submit(spread(c),c.version,"identical-submit")
        a.advance(minutes=5)
        for _ in range(5):
            b.advance(minutes=1)
            if b.status!="OK": break
        assert a.status==b.status=="PARTIAL"
        assert a.decision_at==b.decision_at==local_datetime(datetime(2026,11,9,10,3))
        assert a.broker.to_dict()==b.broker.to_dict()
    finally:
        a.close(); b.close()


def test_preview_duplicate_submission_and_stale_version(config):
    c=ReplayController.create(config,local_datetime(datetime(2026,11,9,10,1)))
    try:
        draft=spread(c)
        before=digest(c.broker.to_dict())
        preview=c.submit(draft,c.version,"preview",preview=True)
        assert preview["status"]=="OK" and digest(c.broker.to_dict())==before
        prior_version=c.version
        c.submit(draft,c.version,"submit")
        version=c.version
        c.submit(draft,prior_version,"submit")
        assert c.version==version and len(c.broker.orders)==1
        with pytest.raises(BrokerError,match="STATE_VERSION_CONFLICT"):
            c.submit(draft,prior_version,"new-command")
    finally:
        c.close()


def test_branch_does_not_retain_future_fill(config):
    c=ReplayController.create(config,local_datetime(datetime(2026,11,9,10,1)))
    try:
        checkpoint=c.store.checkpoint(c.branch_id)
        c.submit(spread(c),c.version,"entry")
        c.advance(minutes=2)
        assert c.broker.ledger.positions()
        c.branch(checkpoint["sequence"])
        assert c.broker.ledger.positions()==[]
        assert c.broker.ledger.economic_cash==5000
        assert not c.broker.orders
    finally:
        c.close()


def test_output_boundary_and_stale_analysis():
    with pytest.raises(ConfigurationError):
        writable_path(r"D:\external-result.json")
    with pytest.raises(ConfigurationError):
        writable_path("../external-result.json")
    gate=ResultGate()
    identity=AnalysisIdentity("s","b",1,"snapshot","selection","ledger","objective","pack","rules")
    gate.request(identity)
    assert gate.accepts(identity)
    for field in identity.__dataclass_fields__:
        changed=replace(identity,**{field:2 if field=="state_version" else "different"})
        assert not gate.accepts(changed)
    gate.invalidate()
    assert not gate.accepts(identity)


def test_config_does_not_mislabel_rules_or_accept_binary_float_cash():
    with pytest.raises(ConfigurationError,match="UNSUPPORTED_RULE_PROFILE"):
        replace(AppConfig(),rule_asof="2012-01-01").validate()
    with pytest.raises(ConfigurationError,match="MONEY_REQUIRES_EXACT_DECIMAL_INPUT"):
        replace(AppConfig(),initial_cash=5000.675).validate()
    with pytest.raises(ConfigurationError,match="STRICT_FLAG_MUST_BE_BOOLEAN"):
        replace(AppConfig(),strict="false").validate()
