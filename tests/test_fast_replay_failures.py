"""Regression coverage for validated prefixes and matching failure snapshots."""
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal as D
import csv
import gzip
import json
from pathlib import Path

import pytest

from qqq_sim.cli import demo_config
from qqq_sim.broker.models import OrderDraft, OrderLeg
from qqq_sim.config import AppConfig
from qqq_sim.data.fixtures import write_synthetic_dataset
from qqq_sim.replay.calendar import local_datetime
from qqq_sim.replay.controller import ReplayController
from qqq_sim.services.adapters import broker_contract


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "fixture"
    write_synthetic_dataset(root, minutes=100)
    return replace(demo_config(AppConfig(), root), runtime_root=str(tmp_path / "runtime"))


def at(minute):
    return local_datetime(datetime(2026, 11, 9, 10, minute))


def spread(controller):
    contracts = {(entry.contract.right, entry.contract.strike): broker_contract(entry.contract, controller.data.calendar)
                 for entry in controller.snapshot.quotes}
    deadline = local_datetime(datetime(2026, 11, 13, 15, 30))
    return OrderDraft("campaign", (OrderLeg(contracts["P", D(600)], -1), OrderLeg(contracts["P", D(595)], 1)),
                      D(50), deadline, deadline + timedelta(minutes=30))


def set_events(config, events):
    path = Path(config.parameters_path)
    parameters = json.loads(path.read_text(encoding="utf-8"))
    parameters["lifecycle_events"] = events
    path.write_text(json.dumps(parameters), encoding="utf-8")


def step_to(controller, target):
    while controller.decision_at < target:
        following = controller.data.calendar.advance_trading_minutes(controller.decision_at, 1)
        controller.advance(target=min(target, following))
        if controller.status != "OK":
            break


def assert_equivalent(left, right):
    assert left.decision_at == right.decision_at
    assert left.status == right.status
    assert left.reason == right.reason
    assert left.snapshot.snapshot_id == right.snapshot.snapshot_id
    assert left.broker.to_dict() == right.broker.to_dict()
    assert left.bundle().account == right.bundle().account


def test_bad_quote_scale_keeps_same_complete_prefix_as_single_steps(config):
    # The ordinary reader detects inconsistent inferred units only once this
    # future timestamp arrives. Preparation must retain that failure barrier.
    mapping_path = Path(config.normalization_path)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping.pop("strike_divisor")
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    quotes_path = next((Path(config.options_root) / "greeks").glob("*.csv.gz"))
    with gzip.open(quotes_path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    bad = next(row for row in rows if row["timestamp"] == "2026-11-09T10:04:30" and row["strike"] == "600")
    bad["strike"] = "600000"
    with gzip.open(quotes_path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fast = ReplayController.create(config, at(1))
    reference = ReplayController.create(config, at(1))
    try:
        fast.advance(minutes=10)
        for _ in range(10):
            reference.advance(minutes=1)
            if reference.status != "OK":
                break
        assert reference.status == "PARTIAL"
        assert reference.reason["code"] == "INCONSISTENT_STRIKE_SCALE"
        assert fast.status == reference.status
        assert fast.reason == reference.reason
        assert fast.decision_at == reference.decision_at == at(4)
        assert fast.snapshot.snapshot_id == reference.snapshot.snapshot_id
        assert fast.broker.to_dict() == reference.broker.to_dict()
        assert fast.store.restore(fast.branch_id)["state"]["decision_at"] == at(4).isoformat()
    finally:
        fast.close()
        reference.close()


def test_second_batch_save_failure_returns_snapshot_for_durable_minute(config, monkeypatch):
    from qqq_sim.replay import fast_forward
    monkeypatch.setattr(fast_forward, "MAX_BATCH_SLOTS", 3)
    monkeypatch.setattr(fast_forward, "monotonic", lambda: 0.)
    controller = ReplayController.create(config, at(1))
    transaction = controller.store.transact
    attempts = []
    def transact(branch_id, command_id, expected_version, payload, events, result, state):
        if payload.get("action") == "ADVANCE_BATCH":
            attempts.append(payload["at"])
            if len(attempts) == 2:
                raise OSError("Second replay batch could not be saved")
        return transaction(branch_id, command_id, expected_version, payload, events, result, state)
    monkeypatch.setattr(controller.store, "transact", transact)
    try:
        bundle = controller.advance(minutes=10)
        assert attempts == [at(4).isoformat(), at(7).isoformat()]
        assert bundle.status == "PARTIAL" and bundle.reason["reason"] == "SAVE_FAILED"
        assert bundle.decision_at == at(4)
        assert bundle.snapshot.asof == bundle.decision_at
        expected = controller.data.snapshot(at(4), state_version=controller.version)
        assert bundle.snapshot.snapshot_id == expected.snapshot_id
        assert bundle.snapshot.state_version == bundle.version
        saved = controller.store.restore(controller.branch_id)
        assert saved["state"]["decision_at"] == bundle.snapshot.asof.isoformat()
        assert saved["state"]["broker"] == controller.broker.to_dict()
    finally:
        controller.close()


def test_data_blocked_order_is_retried_on_same_earliest_quote(config):
    fast = ReplayController.create(config, at(1))
    reference = ReplayController.create(config, at(1))
    try:
        for controller in (fast, reference):
            controller.submit(spread(controller), command_id="blocked-entry")
            next(iter(controller.broker.orders.values())).status = "DATA_BLOCKED"
        fast.advance(minutes=5)
        step_to(reference, at(6))
        assert_equivalent(fast, reference)
        order = next(iter(fast.broker.orders.values()))
        assert order.status == "FILLED"
        fills = [event for event in fast.broker.ledger.events if event["event_type"] == "FILL"]
        assert fills and all(datetime.fromisoformat(event["at"]) == at(1) + timedelta(seconds=30) for event in fills)
    finally:
        fast.close()
        reference.close()


def test_missing_overnight_financing_precedes_next_session_gap(config):
    root = Path(config.underlying_root).parent
    write_synthetic_dataset(root, day=date(2026, 11, 9))
    write_synthetic_dataset(root, day=date(2026, 11, 10), missing_minutes=(0,))
    set_events(config, [{"id": "assignment", "kind": "ASSIGNMENT", "campaign_id": "campaign",
                         "economic_at": "2026-11-09T15:58:00-05:00", "known_at": "2026-11-09T15:58:00-05:00",
                         "contract": {"symbol": "QQQ", "right": "P", "strike": "600", "expiry": "2026-11-13",
                                      "last_trade_at": "2026-11-13T16:15:00-05:00",
                                      "metadata_version": "VERIFIED_STANDARD_QQQ_V1"}, "quantity": 1}])
    start = local_datetime(datetime(2026, 11, 9, 15, 57))
    target = local_datetime(datetime(2026, 11, 10, 9, 31))
    fast = ReplayController.create(config, start)
    reference = ReplayController.create(config, start)
    try:
        for controller in (fast, reference):
            controller.submit(replace(spread(controller), limit_cash=D(10)), command_id="assigned-entry")
        fast.advance(target=target)
        step_to(reference, target)
        assert_equivalent(fast, reference)
        assert fast.status == "PARTIAL" and fast.reason["reason"] == "LIFECYCLE_REFERENCE_REQUIRED"
        assert fast.decision_at == local_datetime(datetime(2026, 11, 9, 16))
        assert fast.broker.ledger.shares("campaign") == 100
    finally:
        fast.close()
        reference.close()


def test_lifecycle_risk_review_generates_close_with_same_execution(config):
    set_events(config, [{"id": "bounded-review", "kind": "RISK_REVIEW", "campaign_id": "campaign",
                         "economic_at": "2026-11-09T10:04:00-05:00", "known_at": "2026-11-09T09:30:00-05:00",
                         "automatic": True, "minimum_package_cash": "-100", "max_total_fees": "50"}])
    fast = ReplayController.create(config, at(1))
    reference = ReplayController.create(config, at(1))
    try:
        for controller in (fast, reference):
            controller.submit(spread(controller), command_id="reviewed-entry")
            controller.advance(minutes=1)
            # A tighter explicit risk-desk limit makes the existing position
            # fail its delivery certificate while admitting a bounded close.
            controller.broker.loss_cap = D(200)
        fast.advance(target=at(7))
        step_to(reference, at(7))
        assert_equivalent(fast, reference)
        assert fast.status == "OK" and not fast.broker.ledger.positions()
        closing = [order for order in fast.broker.orders.values() if order.draft.intent == "CLOSE"]
        assert len(closing) == 1 and closing[0].status == "FILLED"
        assert closing[0].submitted_at == at(4)
        closing_fills = [event for event in fast.broker.ledger.events if event["event_type"] == "FILL"
                         and event["payload"]["order_id"] == closing[0].order_id]
        assert closing_fills and all(datetime.fromisoformat(event["at"]) == at(4) + timedelta(seconds=30)
                                     for event in closing_fills)
    finally:
        fast.close()
        reference.close()


def test_cancellation_after_first_durable_batch_keeps_no_future_economic_events(config, monkeypatch):
    from qqq_sim.replay import fast_forward
    monkeypatch.setattr(fast_forward, "MAX_BATCH_SLOTS", 3)
    monkeypatch.setattr(fast_forward, "monotonic", lambda: 0.)
    ticks = iter(range(100))
    monkeypatch.setattr("qqq_sim.replay.controller.monotonic", lambda: next(ticks) * .3)
    set_events(config, [{"id": "future-charge", "kind": "EXPENSE", "campaign_id": "campaign",
                         "economic_at": "2026-11-09T10:05:00-05:00", "known_at": "2026-11-09T10:05:00-05:00",
                         "amount": "13", "due_date": "2026-11-10"}])
    fast = ReplayController.create(config, at(1))
    reference = ReplayController.create(config, at(1))
    observed = []
    def stop_after_batch(record):
        saved = fast.store.restore(fast.branch_id)
        assert saved["state"]["decision_at"] == record["decision_at"].isoformat()
        observed.append(record)
        if record["completed_slots"] >= 3:
            fast.cancel_requested.set()
    try:
        for controller in (fast, reference):
            controller.submit(spread(controller), command_id="cancelable-entry")
        result = fast.advance(minutes=10, progress=stop_after_batch)
        step_to(reference, at(4))
        assert result.status == "CANCELED" and result.decision_at == at(4)
        assert result.snapshot.asof == at(4)
        assert fast.broker.to_dict() == reference.broker.to_dict()
        assert "lifecycle-input:future-charge" not in fast.broker.ledger.event_ids
        assert all(quote.event_at < at(4) for quote in fast.broker.quotes.values())
        assert observed[-1]["completed_slots"] == 3 and observed[-1]["status"] == "CANCELED"
    finally:
        fast.close()
        reference.close()


@pytest.mark.parametrize(("kind", "bad_field", "bad_value", "expected_minute"), (
    ("trades", "sequence", "unknown", 4),
    ("ohlcv", "close", "bad", 4),
    ("oi_before", "open_interest", "bad", 3),
))
def test_future_auxiliary_parse_error_keeps_same_durable_prefix(config, kind, bad_field, bad_value, expected_minute):
    path = next((Path(config.options_root) / kind).glob("*.csv.gz"))
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row["strike"] == "600" and row["right"] == "P"]
    if kind == "trades":
        bad = next(row for row in selected if row["timestamp"] == "2026-11-09T10:04:30")
    elif kind == "ohlcv":
        bad = next(row for row in selected if row["timestamp"] == "2026-11-09T10:04:00")
    else:
        bad = selected[0]
        bad["known_at"] = "2026-11-09T10:04:00"
    bad[bad_field] = bad_value
    names = list(dict.fromkeys(name for row in rows for name in row))
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    fast = ReplayController.create(config, at(1))
    reference = ReplayController.create(config, at(1))
    try:
        fast.advance(target=at(8))
        step_to(reference, at(8))
        assert reference.status == "PARTIAL"
        assert reference.decision_at == at(expected_minute)
        assert_equivalent(fast, reference)
        assert fast.snapshot.asof == fast.decision_at
        saved = fast.store.restore(fast.branch_id)
        assert saved["state"]["decision_at"] == at(expected_minute).isoformat()
        assert saved["state"]["broker"] == fast.broker.to_dict()
    finally:
        fast.close()
        reference.close()
