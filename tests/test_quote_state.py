import csv
import gzip
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from qqq_sim.data.cache import ByteLRU
from qqq_sim.data.models import DataError, NormalizationMapping, Partition, PartitionKey
from qqq_sim.data.quote_state import chain_quotes
from qqq_sim.data.theta import ThetaReader
from qqq_sim.replay.calendar import CalendarService, NY


DAY=date(2026,11,9)
EXPIRY=date(2026,11,13)
OPEN=datetime(2026,11,9,9,30,tzinfo=NY)
MAPPING=NormalizationMapping(strike_divisor=Decimal(1),iv_unit="decimal",
                             metadata_verified=True,timestamp_precision="subminute")


def row(seconds,**values):
    return {"timestamp":(OPEN+timedelta(seconds=seconds)).isoformat(),
            "known_at":"","expiration":EXPIRY.isoformat(),"strike":"600","right":"C",
            "bid":"1.00","ask":"1.10","sequence":"","implied_vol":"0.25",**values}


def readers(tmp_path,rows):
    path=tmp_path/"greeks.csv.gz"
    with gzip.open(path,"wt",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    stat=path.stat()
    partition=Partition(PartitionKey("greeks",DAY,EXPIRY),str(path),stat.st_size,stat.st_mtime_ns)
    calendar=CalendarService()
    return ThetaReader(calendar,ByteLRU(1024*1024)),ThetaReader(calendar,ByteLRU(1024*1024)),partition


def assert_full_equivalent(reader,reference,partition,seconds,spot=600,mapping=MAPPING,inclusive=False):
    cutoff=OPEN+timedelta(seconds=seconds)
    actual=chain_quotes(reader,partition,cutoff,spot,mapping,inclusive)
    quotes,diagnostics=reference.quotes(partition,cutoff,spot,mapping,inclusive)
    latest,valid={},{}
    for quote in quotes:
        latest[quote.contract]=quote
        if quote.valid:
            valid[quote.contract]=quote
    expected={contract:(quote,valid.get(contract)) for contract,quote in latest.items()}
    assert actual==(expected,diagnostics)
    return actual


def test_warm_forward_rewind_and_repeat_match_full_history(tmp_path,monkeypatch):
    reader,reference,partition=readers(tmp_path,[row(30),row(90,bid="2",ask="2.1"),
                                               row(150,strike="605"),row(210,right="P")])
    calls=[]
    original=reader.quotes
    def quotes(*args,**kwargs):
        calls.append(kwargs.get("interval_start"))
        return original(*args,**kwargs)
    monkeypatch.setattr(reader,"quotes",quotes)
    assert_full_equivalent(reader,reference,partition,60)
    assert_full_equivalent(reader,reference,partition,120)
    assert calls==[None,OPEN+timedelta(seconds=60)]
    assert_full_equivalent(reader,reference,partition,120)
    assert len(calls)==2
    for seconds in (240,120,180,60,240):
        assert_full_equivalent(reader,reference,partition,seconds)


def test_invalid_latest_retains_last_valid_then_accepts_new_valid(tmp_path):
    reader,reference,partition=readers(tmp_path,[row(30),row(90,bid="3",ask="1"),
                                               row(150,bid="2",ask="2.1")])
    assert_full_equivalent(reader,reference,partition,60)
    pairs,diagnostics=assert_full_equivalent(reader,reference,partition,120)
    latest,valid=next(iter(pairs.values()))
    assert not latest.valid and valid.valid and valid.event_at<latest.event_at
    assert not diagnostics
    assert_full_equivalent(reader,reference,partition,180)


def test_late_older_valid_observation_preserves_event_time_order(tmp_path):
    reader,reference,partition=readers(tmp_path,[row(30,bid="3",ask="1"),
        row(15,known_at=(OPEN+timedelta(seconds=90)).isoformat(),bid="2",ask="2.1")])
    pairs,_=assert_full_equivalent(reader,reference,partition,60)
    assert next(iter(pairs.values()))[1] is None
    pairs,_=assert_full_equivalent(reader,reference,partition,120)
    latest,valid=next(iter(pairs.values()))
    assert not latest.valid and valid.event_at==OPEN+timedelta(seconds=15)


@pytest.mark.parametrize("sequences",[("",""),("2","1"),("1","2"),("1","1")])
def test_late_same_timestamp_conflict_matches_complete_group(tmp_path,sequences):
    reader,reference,partition=readers(tmp_path,[row(30,sequence=sequences[0]),
        row(30,known_at=(OPEN+timedelta(seconds=90)).isoformat(),sequence=sequences[1],bid="2",ask="2.1")])
    assert_full_equivalent(reader,reference,partition,60)
    assert_full_equivalent(reader,reference,partition,120)
    assert_full_equivalent(reader,reference,partition,180)


def test_late_ambiguous_older_group_removes_previous_valid_mark(tmp_path):
    reader,reference,partition=readers(tmp_path,[row(15),row(30,bid="3",ask="1"),
        row(15,known_at=(OPEN+timedelta(seconds=90)).isoformat(),bid="2",ask="2.1")])
    pairs,_=assert_full_equivalent(reader,reference,partition,60)
    assert next(iter(pairs.values()))[1] is not None
    pairs,diagnostics=assert_full_equivalent(reader,reference,partition,120)
    assert next(iter(pairs.values()))[1] is None
    assert any(item.code=="AMBIGUOUS_EVENT_ORDER" for item in diagnostics)


@pytest.mark.parametrize("precision,rows",[
    ("subminute",[row(30),row(60,bid="2",ask="2.1"),row(90,bid="3",ask="3.1")]),
    ("minute",[row(0),row(60,bid="2",ask="2.1"),row(120,bid="3",ask="3.1")]),
])
def test_inclusive_exclusive_boundary_transitions(tmp_path,precision,rows):
    reader,reference,partition=readers(tmp_path,rows)
    mapping=replace(MAPPING,timestamp_precision=precision)
    for seconds,inclusive in ((60,False),(60,True),(60,False),(60,True),
                              (90,False),(120,True),(120,False),(180,False)):
        assert_full_equivalent(reader,reference,partition,seconds,mapping=mapping,inclusive=inclusive)


def test_spot_inference_revalidates_all_previously_observed_strikes(tmp_path):
    reader,reference,partition=readers(tmp_path,[row(30,strike="600000"),
                                               row(90,strike="610000")])
    mapping=replace(MAPPING,strike_divisor=None)
    for seconds,spot in ((60,600),(120,610),(180,610000),(240,None),(300,610)):
        assert_full_equivalent(reader,reference,partition,seconds,spot=spot,mapping=mapping)


def test_inferred_strike_range_keeps_old_contracts_without_new_quotes(tmp_path):
    reader,reference,partition=readers(tmp_path,[row(30,strike="600000"),
                                               row(30,strike="500000")])
    mapping=replace(MAPPING,strike_divisor=None)
    for seconds,spot in ((60,600),(120,5000),(180,5500),(240,600000)):
        assert_full_equivalent(reader,reference,partition,seconds,spot=spot,mapping=mapping)


def test_inconsistent_inferred_scale_across_incremental_windows_is_rejected(tmp_path):
    reader,reference,partition=readers(tmp_path,[row(30,strike="600"),row(90,strike="600000")])
    mapping=replace(MAPPING,strike_divisor=None)
    assert_full_equivalent(reader,reference,partition,60,mapping=mapping)
    for load in (lambda:chain_quotes(reader,partition,OPEN+timedelta(seconds=120),600,mapping),
                 lambda:reference.quotes(partition,OPEN+timedelta(seconds=120),600,mapping)):
        with pytest.raises(DataError,match="INCONSISTENT_STRIKE_SCALE"):
            load()


def test_mapping_change_and_explicit_strikes_do_not_reuse_old_identity(tmp_path):
    reader,reference,partition=readers(tmp_path,[row(30,strike="600000"),row(90,strike="605000")])
    for seconds,spot,mapping in ((60,600,replace(MAPPING,strike_divisor=Decimal(1000))),
                                (120,None,replace(MAPPING,strike_divisor=Decimal(1000))),
                                (180,600000,MAPPING),(240,600,MAPPING)):
        assert_full_equivalent(reader,reference,partition,seconds,spot=spot,mapping=mapping)


def test_old_diagnostics_are_retained_on_later_forward_reads(tmp_path):
    reader,reference,partition=readers(tmp_path,[row(30),row(90,expiration="2026-11-20"),row(150)])
    assert_full_equivalent(reader,reference,partition,60)
    for seconds in (120,180,240):
        _,diagnostics=assert_full_equivalent(reader,reference,partition,seconds)
        assert any(item.code=="EXPIRY_IDENTITY_MISMATCH" for item in diagnostics)


def test_cache_eviction_reconstructs_the_same_chain(tmp_path,monkeypatch):
    reader,reference,partition=readers(tmp_path,[row(30),row(90,bid="3",ask="1"),row(150)])
    assert_full_equivalent(reader,reference,partition,60)
    assert_full_equivalent(reader,reference,partition,120)
    reader.cache.put("memory-pressure",object(),reader.cache.max_bytes)
    calls=[]
    original=reader.quotes
    def quotes(*args,**kwargs):
        calls.append(kwargs.get("interval_start"))
        return original(*args,**kwargs)
    monkeypatch.setattr(reader,"quotes",quotes)
    assert_full_equivalent(reader,reference,partition,180)
    assert calls==[None]
