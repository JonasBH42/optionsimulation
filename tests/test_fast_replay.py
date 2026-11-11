"""Economic and recovery invariants for multi-minute replay commands."""
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal as D
from pathlib import Path
import csv
import gzip
import json

import pytest

from qqq_sim.broker.models import OrderDraft, OrderLeg
from qqq_sim.cli import demo_config
from qqq_sim.config import AppConfig
from qqq_sim.data.fixtures import write_synthetic_dataset
from qqq_sim.replay.calendar import local_datetime
from qqq_sim.replay.controller import ReplayController
from qqq_sim.services.adapters import broker_contract


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "fixture"
    for day in (date(2026, 11, 9), date(2026, 11, 10)):
        write_synthetic_dataset(root, day=day)
    return replace(demo_config(AppConfig(), root), runtime_root=str(tmp_path / "runtime"))


def at(hour=10, minute=1, day=9):
    return local_datetime(datetime(2026, 11, day, hour, minute))


def draft(controller, *, limit=D(50), tif="DAY", quantity=1):
    keys = {(entry.contract.right, entry.contract.strike): broker_contract(entry.contract, controller.data.calendar)
            for entry in controller.snapshot.quotes}
    return OrderDraft("campaign", (OrderLeg(keys["P", D(600)], -quantity),
                                   OrderLeg(keys["P", D(595)], quantity)), limit,
                      at(15, 30, 13), at(16, 0, 13), tif=tif)


def assert_same_market_and_account(left, right):
    assert left.decision_at == right.decision_at
    assert left.status == right.status
    assert left.reason == right.reason
    assert left.snapshot.snapshot_id == right.snapshot.snapshot_id
    assert left.snapshot.quotes == right.snapshot.quotes
    assert left.snapshot.candles == right.snapshot.candles
    assert left.snapshot.diagnostics == right.snapshot.diagnostics
    assert left.broker.to_dict() == right.broker.to_dict()
    assert left.bundle().account == right.bundle().account


def step_to(controller, target):
    while controller.decision_at < target:
        following = controller.data.calendar.advance_trading_minutes(controller.decision_at, 1)
        controller.advance(target=min(target, following))
        if controller.status != "OK":
            break


@pytest.mark.parametrize("minutes", (5, 60, 360))
def test_long_idle_skip_matches_single_minutes(config, minutes):
    left = ReplayController.create(config, at(9, 30))
    right = ReplayController.create(config, at(9, 30))
    try:
        target = left.data.calendar.advance_trading_minutes(left.decision_at, minutes)
        left.advance(target=target)
        step_to(right, target)
        assert_same_market_and_account(left, right)
    finally:
        left.close()
        right.close()


def test_idle_skip_builds_only_the_destination_full_chain(config, monkeypatch):
    controller = ReplayController.create(config, at())
    try:
        original = controller.data.snapshot
        loaded = []
        def snapshot(instant, **kwargs):
            loaded.append(instant)
            return original(instant, **kwargs)
        monkeypatch.setattr(controller.data, "snapshot", snapshot)
        destination = at(11, 1)
        result = controller.advance(target=destination)
        assert result.status == "OK" and result.decision_at == destination
        assert loaded == [destination]
    finally:
        controller.close()


@pytest.mark.parametrize("starting_state", ("working", "held", "canceled"))
def test_skip_keeps_earliest_fill_and_held_trade_history(config, starting_state):
    left = ReplayController.create(config, at())
    right = ReplayController.create(config, at())
    try:
        for controller in (left, right):
            controller.submit(draft(controller), controller.version, "identical-order")
            if starting_state == "held":
                controller.advance(minutes=1)
                assert controller.broker.ledger.positions()
            elif starting_state == "canceled":
                order_id = next(iter(controller.broker.orders))
                controller.cancel(order_id, controller.version, "identical-cancel")
        target = left.decision_at + timedelta(minutes=5)
        left.advance(target=target)
        step_to(right, target)
        assert_same_market_and_account(left, right)
        fills = [event for event in left.broker.ledger.events if event["event_type"] == "FILL"]
        if starting_state == "canceled":
            assert not fills
        else:
            assert fills
            assert all(datetime.fromisoformat(event["at"]) == at() + timedelta(seconds=30)
                       for event in fills)
    finally:
        left.close()
        right.close()


def test_partially_filled_order_consumes_each_quote_depth_once(config):
    config = replace(config, loss_cap="2000.00", session_loss_cap="2000.00")
    quote_path = next((Path(config.options_root) / "greeks").glob("*20261109*.csv.gz"))
    with gzip.open(quote_path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["bid_size"] = row["ask_size"] = "1"
    with gzip.open(quote_path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    left = ReplayController.create(config, at())
    right = ReplayController.create(config, at())
    try:
        for controller in (left, right):
            controller.submit(draft(controller, quantity=2), controller.version, "partial-entry")
            controller.advance(minutes=1)
            order = next(iter(controller.broker.orders.values()))
            assert order.status == "PARTIALLY_FILLED" and order.remaining_units == 1
        target = left.decision_at + timedelta(minutes=5)
        left.advance(target=target)
        step_to(right, target)
        assert_same_market_and_account(left, right)
        order = next(iter(left.broker.orders.values()))
        assert order.status == "FILLED" and order.filled_units == 2
        fill_times = {datetime.fromisoformat(event["at"]) for event in left.broker.ledger.events
                      if event["event_type"] == "FILL"}
        assert fill_times == {at() + timedelta(seconds=30), at() + timedelta(minutes=1, seconds=30)}
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("tif", ("DAY", "GTC"))
def test_next_session_retains_order_expiration_and_settlement(config, tif):
    left = ReplayController.create(config, at(15, 57))
    right = ReplayController.create(config, at(15, 57))
    try:
        for controller in (left, right):
            # An unattainable credit keeps this order working until its TIF acts.
            controller.submit(draft(controller, limit=D(400), tif=tif), controller.version, "working-close")
        target = at(9, 30, 10)
        left.advance(target=target)
        step_to(right, target)
        assert_same_market_and_account(left, right)
        assert next(iter(left.broker.orders.values())).status == ("EXPIRED" if tif == "DAY" else "WORKING")
    finally:
        left.close()
        right.close()


def test_skip_preserves_subminute_destination_barrier(config):
    left = ReplayController.create(config, at())
    right = ReplayController.create(config, at())
    try:
        target = at(10, 4) + timedelta(seconds=30)
        for controller in (left, right):
            controller.submit(draft(controller), controller.version, "same-boundary-order")
        left.advance(target=target)
        step_to(right, target)
        assert_same_market_and_account(left, right)
        assert all(entry.latest_event.available_at < target for entry in left.snapshot.quotes)
    finally:
        left.close()
        right.close()


def test_missing_minute_keeps_same_durable_prefix_and_idempotent_restore(config):
    root = Path(config.underlying_root).parent
    write_synthetic_dataset(root, minutes=390, missing_minutes=(34,))
    left = ReplayController.create(config, at())
    right = ReplayController.create(config, at())
    try:
        for controller in (left, right):
            controller.submit(draft(controller), controller.version, "before-gap")
        left.advance(minutes=10, command_id="partial-jump")
        step_to(right, at(10, 11))
        assert_same_market_and_account(left, right)
        assert left.status == "PARTIAL" and left.decision_at == at(10, 4)
        restored = ReplayController.restore(config, left.session_id)
        try:
            assert_same_market_and_account(left, restored)
            saved_version = restored.version
            restored.advance(minutes=10, expected_version=0, command_id="partial-jump")
            assert restored.version == saved_version
            assert_same_market_and_account(left, restored)
        finally:
            restored.close()
    finally:
        left.close()
        right.close()


def test_cancel_before_skip_does_not_advance_or_change_trade(config):
    controller = ReplayController.create(config, at())
    try:
        controller.submit(draft(controller), controller.version, "before-cancel")
        before = controller.broker.to_dict()
        def cancel(record):
            controller.cancel_requested.set()
        result = controller.advance(minutes=60, command_id="cancel-before-first-minute", progress=cancel)
        assert result.status == "CANCELED" and result.decision_at == at()
        assert controller.broker.to_dict() == before
        saved = controller.store.restore(controller.branch_id)
        assert saved["state"]["status"] == "CANCELED"
        assert saved["state"]["broker"] == json.loads(json.dumps(before))
    finally:
        controller.close()
