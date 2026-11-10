import csv
import gzip
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from qqq_sim.data.cache import ByteLRU
from qqq_sim.data.models import DataError, NormalizationMapping, Partition, PartitionKey
from qqq_sim.data.theta import ThetaReader, active_trades
from qqq_sim.data.trade_state import latest_trades
from qqq_sim.replay.calendar import CalendarService, NY


DAY=date(2026,11,9)
EXPIRY=date(2026,11,13)
OPEN=datetime(2026,11,9,9,30,tzinfo=NY)
MAPPING=NormalizationMapping(strike_divisor=Decimal(1),iv_unit="decimal",
                             metadata_verified=True,timestamp_precision="subminute")


def row(seconds,**values):
    return {"timestamp":(OPEN+timedelta(seconds=seconds)).isoformat(),"known_at":"",
            "expiration":EXPIRY.isoformat(),"strike":"600","right":"C",
            "price":"1","size":"10","sequence":"","condition":"REGULAR",
            "trade_id":f"T{seconds}","action":"TRADE","correction_of":"",**values}


def readers(tmp_path,rows):
    path=tmp_path/"trades.csv.gz"
    with gzip.open(path,"wt",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    stat=path.stat()
    partition=Partition(PartitionKey("trades",DAY,EXPIRY),str(path),stat.st_size,stat.st_mtime_ns)
    calendar=CalendarService()
    return ThetaReader(calendar,ByteLRU(1024*1024)),ThetaReader(calendar,ByteLRU(1024*1024)),partition


def selected_full(reader,partition,cutoff,spot,mapping,inclusive):
    observations=reader.trades(partition,cutoff,spot,mapping,inclusive=inclusive)
    groups={}
    for trade in active_trades(observations):
        if not trade.size:
            continue
        group=groups.get(trade.contract)
        if group is None or trade.event_at>group[0].event_at:
            groups[trade.contract]=[trade]
        elif trade.event_at==group[0].event_at:
            group.append(trade)
    selected={}
    for contract,group in groups.items():
        if all(trade.sequence is not None for trade in group):
            sequence=max(trade.sequence for trade in group)
            group=[trade for trade in group if trade.sequence==sequence]
        trade=max(group,key=lambda item:(item.known_at,item.source_id))
        if len({item.price for item in group})>1:
            trade=replace(trade,quality_flags=(*trade.quality_flags,"AMBIGUOUS_LAST_TRADE_ORDER"))
        selected[contract]=trade
    return selected


def assert_full_equivalent(reader,reference,partition,seconds,spot=600,mapping=MAPPING,inclusive=False):
    cutoff=OPEN+timedelta(seconds=seconds)
    try:
        expected=selected_full(reference,partition,cutoff,spot,mapping,inclusive)
    except DataError as expected_error:
        with pytest.raises(DataError) as actual_error:
            latest_trades(reader,partition,cutoff,spot,mapping,inclusive)
        assert actual_error.value.code==expected_error.code
        assert actual_error.value.message==expected_error.message
        return None
    actual=latest_trades(reader,partition,cutoff,spot,mapping,inclusive)
    assert actual==expected
    return actual


def test_warm_forward_repeat_rewind_and_eviction_match_full_ledger(tmp_path,monkeypatch):
    reader,reference,partition=readers(tmp_path,[row(15),row(75,price="2"),row(135,right="P")])
    calls=[]
    original=reader.trades
    def trades(*args,**kwargs):
        calls.append(kwargs.get("interval_start"))
        return original(*args,**kwargs)
    monkeypatch.setattr(reader,"trades",trades)
    assert_full_equivalent(reader,reference,partition,60)
    assert_full_equivalent(reader,reference,partition,120)
    assert calls==[None,OPEN+timedelta(seconds=60)]
    assert_full_equivalent(reader,reference,partition,120)
    assert len(calls)==2
    for seconds in (180,60,120,180):
        assert_full_equivalent(reader,reference,partition,seconds)
    reader.cache.put("memory-pressure",object(),reader.cache.max_bytes)
    assert_full_equivalent(reader,reference,partition,240)
    assert calls[-1] is None


@pytest.mark.parametrize("action,size",[("CANCEL","0"),("CORRECT","5"),("CORRECT","0")])
def test_correcting_latest_trade_recovers_the_preceding_observation(tmp_path,action,size):
    reader,reference,partition=readers(tmp_path,[row(15),row(30,price="2"),
        row(30,known_at=(OPEN+timedelta(seconds=90)).isoformat(),trade_id="revision",
            action=action,correction_of="T30",price="3",size=size)])
    assert_full_equivalent(reader,reference,partition,60)
    assert_full_equivalent(reader,reference,partition,90)
    assert_full_equivalent(reader,reference,partition,90,inclusive=True)
    assert_full_equivalent(reader,reference,partition,120)


def test_correction_of_older_trade_keeps_incremental_latest_group(tmp_path,monkeypatch):
    reader,reference,partition=readers(tmp_path,[row(15),row(30,price="2"),
        row(15,known_at=(OPEN+timedelta(seconds=90)).isoformat(),trade_id="cancel",
            action="CANCEL",correction_of="T15",size="0")])
    calls=[]
    original=reader.trades
    def trades(*args,**kwargs):
        calls.append(kwargs.get("interval_start"))
        return original(*args,**kwargs)
    monkeypatch.setattr(reader,"trades",trades)
    assert_full_equivalent(reader,reference,partition,60)
    selected=assert_full_equivalent(reader,reference,partition,120)
    assert next(iter(selected.values())).trade_id=="T30"
    assert calls==[None,OPEN+timedelta(seconds=60)]


def test_zero_size_remains_an_active_correction_target(tmp_path):
    reader,reference,partition=readers(tmp_path,[row(15),row(30,size="0"),
        row(45,known_at=(OPEN+timedelta(seconds=90)).isoformat(),trade_id="correct-zero",
            action="CORRECT",correction_of="T30",price="4",size="2")])
    selected=assert_full_equivalent(reader,reference,partition,60)
    assert next(iter(selected.values())).trade_id=="T15"
    selected=assert_full_equivalent(reader,reference,partition,120)
    assert next(iter(selected.values())).trade_id=="correct-zero"


def test_delayed_older_ordinary_trade_does_not_replace_latest(tmp_path):
    reader,reference,partition=readers(tmp_path,[row(30),
        row(15,known_at=(OPEN+timedelta(seconds=90)).isoformat(),price="2")])
    assert_full_equivalent(reader,reference,partition,60)
    selected=assert_full_equivalent(reader,reference,partition,120)
    assert next(iter(selected.values())).trade_id=="T30"


@pytest.mark.parametrize("sequences",[("",""),("1","2"),("2","1"),("1","1")])
def test_delayed_same_event_time_preserves_sequence_and_ambiguity(tmp_path,sequences):
    reader,reference,partition=readers(tmp_path,[row(30,sequence=sequences[0]),
        row(30,known_at=(OPEN+timedelta(seconds=90)).isoformat(),trade_id="second",
            sequence=sequences[1],price="2")])
    for seconds in (60,90,120,180):
        assert_full_equivalent(reader,reference,partition,seconds)


@pytest.mark.parametrize("change",[
    {"trade_id":"T15","price":"2"},
    {"action":"CANCEL","correction_of":"missing"},
    {"action":"CORRECT","correction_of":"missing"},
    {"price":"-1"},
    {"size":"1.5"},
])
def test_later_invalid_trade_or_ledger_operation_keeps_reader_error(tmp_path,change):
    reader,reference,partition=readers(tmp_path,[row(15),row(90,**change)])
    assert_full_equivalent(reader,reference,partition,60)
    assert assert_full_equivalent(reader,reference,partition,120) is None
    assert assert_full_equivalent(reader,reference,partition,180) is None


def test_duplicate_sources_and_reuse_of_canceled_trade_ids(tmp_path):
    reader,reference,partition=readers(tmp_path,[row(15),row(15),row(30),
        row(45,trade_id="cancel",action="CANCEL",correction_of="T15",size="0"),
        row(90,trade_id="T15",price="2"),row(90,trade_id="T15",price="2")])
    for seconds in (30,60,120,180):
        assert_full_equivalent(reader,reference,partition,seconds)


@pytest.mark.parametrize("precision",["subminute","minute"])
def test_inclusive_and_exclusive_transitions_preserve_correction_visibility(tmp_path,precision):
    reader,reference,partition=readers(tmp_path,[row(0),row(60),
        row(0,known_at=(OPEN+timedelta(seconds=120)).isoformat(),trade_id="cancel",
            action="CANCEL",correction_of="T0",size="0")])
    mapping=replace(MAPPING,timestamp_precision=precision)
    for seconds,inclusive in ((60,False),(60,True),(60,False),(120,False),
                              (120,True),(120,False),(180,False)):
        assert_full_equivalent(reader,reference,partition,seconds,mapping=mapping,inclusive=inclusive)


def test_new_phase30_record_cannot_use_a_later_sorted_phase20_target(tmp_path):
    reader,reference,partition=readers(tmp_path,[row(60),
        row(0,known_at=(OPEN+timedelta(seconds=120)).isoformat(),trade_id="correction",
            action="CORRECT",correction_of="T60",price="2")])
    mapping=replace(MAPPING,timestamp_precision="minute")
    assert_full_equivalent(reader,reference,partition,120,mapping=mapping)
    assert assert_full_equivalent(reader,reference,partition,121,mapping=mapping) is None


def test_unknown_conditions_and_conflicting_sequence_prices_keep_quality(tmp_path):
    reader,reference,partition=readers(tmp_path,[row(15,condition="UNKNOWN"),
        row(15,trade_id="second",condition="UNKNOWN",price="2")])
    selected=assert_full_equivalent(reader,reference,partition,60)
    flags=next(iter(selected.values())).quality_flags
    assert "UNKNOWN_TRADE_CONDITION" in flags and "AMBIGUOUS_LAST_TRADE_ORDER" in flags


def test_spot_inference_includes_canceled_contracts_and_allows_mixed_scales(tmp_path):
    reader,reference,partition=readers(tmp_path,[row(15,strike="600000"),
        row(30,strike="600"),row(45,trade_id="cancel",strike="600000",action="CANCEL",
                                  correction_of="T15",size="0")])
    mapping=replace(MAPPING,strike_divisor=None)
    for seconds,spot in ((20,600),(40,600),(60,600),(120,600000),(180,600),(240,None)):
        assert_full_equivalent(reader,reference,partition,seconds,spot=spot,mapping=mapping)


def test_canceled_strikes_still_constrain_future_inference_reuse(tmp_path):
    reader,reference,partition=readers(tmp_path,[row(15,strike="900000"),
        row(30,strike="600000"),row(45,trade_id="cancel",strike="600000",action="CANCEL",
                                  correction_of="T15",size="0")])
    mapping=replace(MAPPING,strike_divisor=None)
    assert_full_equivalent(reader,reference,partition,60,mapping=mapping)
    assert assert_full_equivalent(reader,reference,partition,120,spot=75,mapping=mapping) is None


def test_explicit_mapping_changes_and_streaming_cache_budget(tmp_path):
    reader,reference,partition=readers(tmp_path,[row(15,strike="600000"),row(90,strike="605000")])
    for seconds,spot,mapping in ((60,None,replace(MAPPING,strike_divisor=Decimal(1000))),
                                (120,None,replace(MAPPING,strike_divisor=Decimal(1000))),
                                (180,600000,MAPPING)):
        assert_full_equivalent(reader,reference,partition,seconds,spot=spot,mapping=mapping)
    reader.cache.clear()
    reader.cache.max_bytes=1
    assert_full_equivalent(reader,reference,partition,240)
    assert reader.cache.bytes<=reader.cache.max_bytes
