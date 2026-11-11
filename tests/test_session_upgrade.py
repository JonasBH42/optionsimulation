from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
import json
from pathlib import Path
import sqlite3

import pytest

from qqq_sim.broker.models import BrokerError, OrderDraft, OrderLeg
from qqq_sim.cli import demo_config
from qqq_sim.config import AppConfig
from qqq_sim.data.fixtures import write_synthetic_dataset
from qqq_sim.replay.calendar import local_datetime
from qqq_sim.replay.controller import ReplayController
from qqq_sim.serialization import digest
from qqq_sim.services import session_upgrade as upgrade
from qqq_sim.services.adapters import broker_contract
from qqq_sim.services._session_upgrade_policy import POLICY_ID, PREDECESSOR_IDENTITIES
from qqq_sim.services.runtime_identity import runtime_identity, verify_runtime_identity
from qqq_sim.storage.events import EventStore


@pytest.fixture(params=PREDECESSOR_IDENTITIES, ids=("original-source", "original-native"))
def legacy(tmp_path, request):
    root = tmp_path / "fixture"
    write_synthetic_dataset(root, minutes=40)
    config = replace(demo_config(AppConfig(), root), runtime_root=str(tmp_path / "runtime"))
    controller = ReplayController.create(config, local_datetime(datetime(2026, 11, 9, 10, 1)))
    state = controller._state()
    state["implementation_identity"] = deepcopy(request.param)
    controller.store.transact(controller.branch_id, "fixture-legacy", controller.version,
                              {"action": "FIXTURE_PREDECESSOR"}, [], {}, state)
    # Existing stored analyses are immutable historical records, even without a request.
    controller.store.record_analysis("old-analysis", controller.branch_id, controller.version,
                                     "selection", "original-pack", "UNAVAILABLE", {"reason": "ORIGINAL"})
    source = controller.store.path
    try:
        yield config, controller, state, source
    finally:
        controller.store.connection.close()


def _bytes(path):
    return {str(item): item.read_bytes() for item in (path, Path(str(path) + "-wal")) if item.exists()}


def _rows(db, table):
    return [tuple(row) for row in db.execute(f'SELECT * FROM "{table}" ORDER BY rowid')]


def test_predecessor_upgrade_copies_live_wal_preserves_original_history_and_economics(legacy):
    _, controller, old_state, source = legacy
    before = _bytes(source)
    original = {table: _rows(controller.store.connection, table) for table in
                ("events", "commands", "analyses", "cash_postings", "branches")}
    destination, branch_id = upgrade.prepare_session_restore(source)
    assert destination != source and destination.is_file()
    assert _bytes(source) == before
    with upgrade._verified_reader(destination) as (reader, session, recovered):
        assert recovered["branch_id"] == branch_id != controller.branch_id
        assert session["id"] == controller.session_id
        expected = deepcopy(old_state)
        expected["implementation_identity"] = runtime_identity()
        assert recovered["state"] == expected
        verify_runtime_identity(recovered["state"]["implementation_identity"])
        for table, rows in original.items():
            assert _rows(reader.connection, table)[:len(rows)] == rows
        row = reader.connection.execute("SELECT payload_json FROM events WHERE event_type='IMPLEMENTATION_UPGRADED'").fetchone()
        transition = json.loads(row[0])
        assert transition["policy_id"] == POLICY_ID
        assert transition["previous_identity"] == old_state["implementation_identity"]
        assert transition["economic_state_changed"] is False
        assert reader.restore(controller.branch_id)["state"] == old_state


def test_reopening_legacy_source_reuses_progressed_copy(legacy):
    _, _, _, source = legacy
    destination, branch_id = upgrade.prepare_session_restore(source)
    store = EventStore(destination)
    try:
        recovered = store.restore(branch_id)
        state = deepcopy(recovered["state"])
        state["decision_at"] = (datetime.fromisoformat(state["decision_at"]) + timedelta(minutes=1)).isoformat()
        store.transact(branch_id, "progress-after-upgrade", recovered["version"], {"action": "TEST_ADVANCE"}, [], {}, state)
    finally:
        store.connection.close()
    assert upgrade.prepare_session_restore(source) == (destination, branch_id)
    with upgrade._verified_reader(destination) as (_, _, recovered):
        assert recovered["state"] == state
    assert upgrade.prepare_session_restore(destination) == (destination, None)


@pytest.mark.parametrize("damage", ["missing", "invalid", "unknown"])
def test_upgrade_refuses_unrecognized_identity_without_writes(legacy, damage):
    _, controller, old_state, source = legacy
    state = deepcopy(old_state)
    if damage == "missing":
        state.pop("implementation_identity")
    else:
        saved = state["implementation_identity"]
        saved["content"]["source_digest"] = "f" * 64
        if damage == "unknown":
            saved["identity_hash"] = digest({"schema": saved["schema"], "content": saved["content"]})
    version = controller.store.restore(controller.branch_id)["version"]
    controller.store.transact(controller.branch_id, "damage", version, {}, [], {}, state)
    before = _bytes(source)
    with pytest.raises(BrokerError, match={"missing": "IDENTITY_MISSING", "invalid": "IDENTITY_INVALID",
                                          "unknown": "REVISION_CHANGED"}[damage]):
        upgrade.prepare_session_restore(source)
    assert _bytes(source) == before


def test_upgrade_requires_matching_current_runtime(legacy, monkeypatch):
    _, _, _, source = legacy
    different = runtime_identity()
    different["content"]["python_runtime"]["sqlite_library_version"] = "different"
    monkeypatch.setattr(upgrade, "runtime_identity", lambda: deepcopy(different))
    before = _bytes(source)
    with pytest.raises(BrokerError, match="REVISION_CHANGED"):
        upgrade.prepare_session_restore(source)
    assert _bytes(source) == before


def test_interrupted_publication_cleans_stage_and_keeps_source(legacy, monkeypatch):
    config, _, _, source = legacy
    before = _bytes(source)
    def fail(*args):
        raise OSError("Simulated publication failure")
    monkeypatch.setattr(upgrade.os, "link", fail)
    with pytest.raises(OSError, match="publication failure"):
        upgrade.prepare_session_restore(source)
    assert _bytes(source) == before
    assert not list((Path(config.runtime_root) / "session_upgrades").rglob("*.sqlite"))


def test_legacy_historical_branch_transition_is_atomic(legacy):
    _, _, _, source = legacy
    destination, _ = upgrade.prepare_session_restore(source)
    store = EventStore(destination)
    try:
        original_branch = store.connection.execute("SELECT id FROM branches WHERE parent_id IS NULL").fetchone()[0]
        before = _rows(store.connection, "branches")
        active = store.sessions()[0]["active_branch"]
        store.connection.execute("CREATE TEMP TRIGGER fail_transition BEFORE INSERT ON commands BEGIN SELECT RAISE(FAIL, 'test failure'); END")
        with pytest.raises(sqlite3.IntegrityError, match="test failure"):
            upgrade.upgrade_legacy_branch(store, original_branch)
        assert _rows(store.connection, "branches") == before
        assert store.sessions()[0]["active_branch"] == active
        store.connection.execute("DROP TRIGGER fail_transition")
        child = upgrade.upgrade_legacy_branch(store, original_branch)
        assert child["branch_id"] != active
        verify_runtime_identity(child["state"]["implementation_identity"])
        assert child["state"]["broker"] == store.restore(original_branch)["state"]["broker"]
    finally:
        store.connection.close()


def test_corrupt_reusable_copy_is_refused_and_never_overwritten(legacy):
    _, _, _, source = legacy
    destination, _ = upgrade.prepare_session_restore(source)
    connection = sqlite3.connect(destination)
    connection.execute("UPDATE events SET event_hash=? WHERE event_type='IMPLEMENTATION_UPGRADED'", ("0" * 64,))
    connection.commit()
    connection.close()
    before = destination.read_bytes()
    with pytest.raises(BrokerError, match="EVENT_HASH_MISMATCH"):
        upgrade.prepare_session_restore(source)
    assert destination.read_bytes() == before


def test_controller_restore_preserves_open_trade_order_and_legacy_branching(legacy):
    config, controller, predecessor_state, source = legacy
    # Populate economic state through ordinary commands before freezing an old build.
    current = controller._state()
    version = controller.store.restore(controller.branch_id)["version"]
    controller.store.transact(controller.branch_id, "fixture-current", version, {}, [], {}, current)
    controller.version = version + 1
    contracts = {(entry.contract.right, entry.contract.strike): broker_contract(entry.contract, controller.data.calendar)
                 for entry in controller.snapshot.quotes}
    deadline = local_datetime(datetime(2026, 11, 13, 15, 30))
    draft = OrderDraft("campaign", (OrderLeg(contracts["P", Decimal(600)], -1),
                                     OrderLeg(contracts["P", Decimal(595)], 1)),
                       Decimal(50), deadline, deadline + timedelta(minutes=30))
    controller.submit(draft, command_id="open-trade")
    controller.advance(minutes=1)
    assert controller.broker.ledger.positions()
    close = replace(draft, legs=tuple(OrderLeg(leg.contract, -leg.quantity) for leg in draft.legs),
                    limit_cash=Decimal(1000), intent="CLOSE")
    controller.submit(close, command_id="pending-close")
    old = controller._state()
    old["implementation_identity"] = deepcopy(predecessor_state["implementation_identity"])
    controller.store.transact(controller.branch_id, "fixture-final-predecessor", controller.version, {},
                              controller.broker.ledger.events, {}, old)
    legacy_sequence = controller.store.restore(controller.branch_id)["sequence"]
    before = _bytes(source)
    restored = ReplayController.restore(config, source)
    reference = None
    try:
        assert restored.store.path != source
        assert restored.broker.to_dict() == old["broker"]
        assert restored.decision_at.isoformat() == old["decision_at"]
        restored.branch(legacy_sequence)
        assert restored.broker.to_dict() == old["broker"]
        verify_runtime_identity(restored._state()["implementation_identity"])
        reference_path = restored.store.path.with_name("reference-step.sqlite")
        restored.store.backup(reference_path)
        reference = ReplayController.restore(config, reference_path)
        restored.advance(minutes=2)
        reference.advance(minutes=1)
        reference.advance(minutes=1)
        assert restored.status == reference.status == "OK"
        assert restored.decision_at == reference.decision_at
        assert restored.snapshot.snapshot_id == reference.snapshot.snapshot_id
        assert restored.broker.to_dict() == reference.broker.to_dict()
        assert restored.broker.ledger.positions()
        assert any(order.status == "WORKING" and order.draft.intent == "CLOSE"
                   for order in restored.broker.orders.values())
        assert _bytes(source) == before
    finally:
        if reference is not None:
            reference.close()
        restored.close()
