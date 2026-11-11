"""Opening jumps preserve source validation and resume ordinary execution."""
import csv
import gzip
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from qqq_sim.broker.models import OrderDraft, OrderLeg
from qqq_sim.cli import demo_config
from qqq_sim.config import AppConfig
from qqq_sim.data.asof import HistoricalData
from qqq_sim.data.fixtures import write_synthetic_dataset
from qqq_sim.replay.calendar import local_datetime
from qqq_sim.replay.controller import ReplayController
from qqq_sim.services.adapters import broker_contract


def instant(day, hour, minute):
    return local_datetime(datetime(2026, 11, day, hour, minute))


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "fixture"
    for day in (date(2026, 11, 9), date(2026, 11, 10)):
        write_synthetic_dataset(root, day=day)
    return replace(demo_config(AppConfig(), root), runtime_root=str(tmp_path / "runtime"))


def step_to(controller, target):
    while controller.decision_at < target:
        following = controller.data.calendar.advance_trading_minutes(controller.decision_at, 1)
        controller.advance(target=min(target, following))
        if controller.status != "OK":
            break


def same_state(left, right):
    assert (left.decision_at, left.status, left.reason) == (right.decision_at, right.status, right.reason)
    assert left.snapshot.snapshot_id == right.snapshot.snapshot_id
    assert left.broker.to_dict() == right.broker.to_dict()
    assert left.bundle().account == right.bundle().account


def trace_preparation(monkeypatch):
    original = HistoricalData.prepare_replay_window
    calls = []
    def prepare(data, start, end, **kwargs):
        calls.append((data, start, end))
        return original(data, start, end, **kwargs)
    monkeypatch.setattr(HistoricalData, "prepare_replay_window", prepare)
    return calls


def test_next_open_then_five_minutes_rebuilds_execution_window_and_fills_gtc(config, monkeypatch):
    prepared = trace_preparation(monkeypatch)
    start, opening = instant(9, 15, 59), instant(10, 9, 30)
    fast, reference = (ReplayController.create(config, start) for _ in range(2))
    snapshots = []
    original_snapshot = fast.data.snapshot
    def snapshot(at, **kwargs):
        snapshots.append(at)
        return original_snapshot(at, **kwargs)
    monkeypatch.setattr(fast.data, "snapshot", snapshot)
    try:
        for controller in (fast, reference):
            keys = {(entry.contract.right, entry.contract.strike): broker_contract(entry.contract, controller.data.calendar)
                    for entry in controller.snapshot.quotes}
            controller.submit(OrderDraft("overnight", (OrderLeg(keys["P", Decimal(600)], -1),
                                                        OrderLeg(keys["P", Decimal(595)], 1)), Decimal(50),
                                         instant(13, 15, 30), instant(13, 16, 0), tif="GTC"), command_id="gtc-entry")
        fast.advance(target=opening)
        step_to(reference, opening)
        same_state(fast, reference)
        assert next(iter(fast.broker.orders.values())).status == "WORKING"
        assert opening in snapshots
        assert all(start_at.date() == date(2026, 11, 9) for data, start_at, _ in prepared if data is fast.data)
        fast.advance(minutes=5)
        step_to(reference, opening + timedelta(minutes=5))
        same_state(fast, reference)
        assert any(start_at.date() == date(2026, 11, 10) for data, start_at, _ in prepared if data is fast.data)
        assert next(iter(fast.broker.orders.values())).status == "FILLED"
        fills = [event for event in fast.broker.ledger.events if event["event_type"] == "FILL"]
        assert fills and all(datetime.fromisoformat(event["at"]) == opening + timedelta(seconds=30) for event in fills)
    finally:
        fast.close()
        reference.close()


def test_next_open_missing_quote_schema_stops_at_same_prior_close(config, monkeypatch):
    path = next((Path(config.options_root) / "greeks").glob("*20261110*.csv.gz"))
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        names = [name for name in reader.fieldnames if name != "ask"]
        rows = [{name: value for name, value in row.items() if name != "ask"} for row in reader]
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    prepared = trace_preparation(monkeypatch)
    start, opening = instant(9, 15, 59), instant(10, 9, 30)
    fast, reference = (ReplayController.create(config, start) for _ in range(2))
    try:
        fast.advance(target=opening)
        step_to(reference, opening)
        same_state(fast, reference)
        assert fast.status == "PARTIAL" and fast.decision_at == instant(9, 16, 0)
        assert fast.reason["code"] == "REQUIRED_FIELD_MISSING"
        assert all(start_at.date() == date(2026, 11, 9) for data, start_at, _ in prepared if data is fast.data)
        saved = fast.store.restore(fast.branch_id)
        assert saved["state"]["decision_at"] == fast.decision_at.isoformat()
    finally:
        fast.close()
        reference.close()
