from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from qqq_sim.cli import demo_config
from qqq_sim.config import AppConfig
from qqq_sim.data.fixtures import write_synthetic_dataset
from qqq_sim.data.models import DataError
from qqq_sim.replay.calendar import local_datetime
from qqq_sim.replay.controller import ReplayController


@pytest.fixture
def controller(tmp_path):
    root=tmp_path/"fixture"
    write_synthetic_dataset(root,minutes=45)
    config=replace(demo_config(AppConfig(),root),runtime_root=str(tmp_path/"runtime"))
    result=ReplayController.create(config,local_datetime(datetime(2026,11,9,10,1)))
    try:
        yield result
    finally:
        result.close()


def test_progress_starts_before_loading_and_is_throttled(controller,monkeypatch):
    monkeypatch.setattr("qqq_sim.replay.controller.monotonic",lambda:1.)
    original=controller.data.snapshot
    loaded=[]
    def snapshot(*args,**kwargs):
        loaded.append(args[0])
        return original(*args,**kwargs)
    monkeypatch.setattr(controller.data,"snapshot",snapshot)
    records=[]
    def progress(record):
        if not records:
            assert not loaded
        records.append(record)
    start=controller.decision_at
    result=controller.advance(minutes=3,command_id="progress",progress=progress)
    assert len(records)==2
    assert records[0]["status"]=="RUNNING" and records[0]["decision_at"]==start
    assert records[0]["completed_slots"]==0
    assert records[-1]["status"]=="OK" and records[-1]["completed_slots"]==3
    assert records[-1]["decision_at"]==result.decision_at==start+timedelta(minutes=3)
    assert all(record["command_id"]=="progress" for record in records)


def test_partial_progress_only_reports_durable_prefix(controller,monkeypatch):
    ticks=iter(index*.3 for index in range(20))
    monkeypatch.setattr("qqq_sim.replay.controller.monotonic",lambda:next(ticks))
    original=controller.data.snapshot
    unavailable=controller.decision_at+timedelta(minutes=2)
    def snapshot(at,**kwargs):
        if at==unavailable:
            raise DataError("TEST_UNAVAILABLE", "Required observation missing")
        return original(at,**kwargs)
    monkeypatch.setattr(controller.data,"snapshot",snapshot)
    records=[]
    def progress(record):
        saved=controller.store.restore(controller.branch_id)
        assert saved["version"]==record["state_version"]
        assert saved["state"]["decision_at"]==record["decision_at"].isoformat()
        records.append(record)
    result=controller.advance(minutes=3,progress=progress)
    assert result.status=="PARTIAL"
    assert len(records)==3
    assert records[-1]["status"]=="PARTIAL"
    assert records[-1]["completed_slots"]==1
    assert all(record["decision_at"]<unavailable for record in records)


def test_progress_can_request_cancellation_before_first_slot(controller):
    start=controller.decision_at
    records=[]
    def progress(record):
        records.append(record)
        controller.cancel_requested.set()
    result=controller.advance(minutes=3,progress=progress)
    assert result.status=="CANCELED" and result.decision_at==start
    assert records[-1]["status"]=="CANCELED" and records[-1]["completed_slots"]==0


def test_failed_progress_observer_cannot_interrupt_replay(controller):
    start=controller.decision_at
    def progress(record):
        raise RuntimeError("Display disconnected")
    result=controller.advance(minutes=2,progress=progress)
    assert result.status=="OK" and result.decision_at==start+timedelta(minutes=2)
    saved=controller.store.restore(controller.branch_id)
    assert saved["state"]["decision_at"]==result.decision_at.isoformat()
