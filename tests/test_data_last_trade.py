from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
import csv
import gzip
from qqq_sim.data.asof import HistoricalData
from qqq_sim.data.fixtures import write_synthetic_dataset, synthetic_mappings
from qqq_sim.replay.calendar import NY


def dataset(tmp_path):
    manifest=write_synthetic_dataset(tmp_path/"data",minutes=4)
    return HistoricalData(manifest["underlying_root"],manifest["options_root"],synthetic_mappings())


def write_rows(data,kind,rows):
    path=next((data.catalog.options_root/kind).glob("*.csv.gz"))
    with gzip.open(path,"wt",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def trade(**changes):
    return {"symbol":"QQQ","expiration":"2026-11-13","strike":"600","right":"C","timestamp":"2026-11-09T09:30:20",
            "known_at":"2026-11-09T09:30:20","price":"1","size":"10","sequence":"0","condition":"REGULAR","trade_id":"T1","action":"TRADE","correction_of":"",**changes}


def selected(snapshot):
    return next(entry for entry in snapshot.quotes if entry.contract.strike==600 and entry.contract.right=="C")


def test_last_trade_corrections_are_prospective_and_exact_known_time_is_exclusive(tmp_path):
    data=dataset(tmp_path)
    write_rows(data,"trades",[trade(),trade(known_at="2026-11-09T09:32:00",price="2",sequence="1",trade_id="T2",action="CORRECT",correction_of="T1"),
                              trade(known_at="2026-11-09T09:33:00",price="0",size="0",sequence="2",trade_id="T3",action="CANCEL",correction_of="T2")])
    before=data.snapshot(datetime(2026,11,9,9,32,tzinfo=NY))
    after=data.snapshot(datetime(2026,11,9,9,32,1,tzinfo=NY))
    canceled=data.snapshot(datetime(2026,11,9,9,33,1,tzinfo=NY))
    assert selected(before).last_trade_price==1
    assert selected(after).last_trade_price==2 and selected(after).last_trade_at.minute==30
    assert selected(canceled).last_trade_price is None
    assert selected(before).last_trade_price==1  # Frozen snapshot survives later correction reads.


def test_minute_trade_is_known_at_completion_with_identical_cached_and_streamed_results(tmp_path):
    data=dataset(tmp_path)
    data.mappings["trades"]=replace(synthetic_mappings()["options"],timestamp_precision="minute")
    write_rows(data,"trades",[trade(timestamp="2026-11-09T09:30:00",known_at="2026-11-09T09:30:00")])
    assert selected(data.snapshot(datetime(2026,11,9,9,30,59,tzinfo=NY))).last_trade_price is None
    cached=data.snapshot(datetime(2026,11,9,9,31,tzinfo=NY))
    assert selected(cached).last_trade_price==1
    data.cache.clear();data.cache.max_bytes=1
    streamed=data.snapshot(datetime(2026,11,9,9,31,tzinfo=NY))
    assert streamed.snapshot_id==cached.snapshot_id


def test_equal_time_conflicting_last_trade_without_reliable_order_is_unavailable(tmp_path):
    data=dataset(tmp_path)
    write_rows(data,"trades",[trade(sequence=""),trade(sequence="",trade_id="T2",price="2")])
    entry=selected(data.snapshot(datetime(2026,11,9,9,31,tzinfo=NY)))
    assert entry.last_trade_price is None and "AMBIGUOUS_LAST_TRADE_ORDER" in entry.last_trade_quality
    write_rows(data,"trades",[trade(sequence="0"),trade(sequence="1",trade_id="T2",price="2")])
    entry=selected(data.snapshot(datetime(2026,11,9,9,31,tzinfo=NY)))
    assert entry.last_trade_price==2


def test_unknown_trade_condition_is_visible_with_quality_but_cannot_supply_execution_volume(tmp_path):
    from qqq_sim.data.theta import trade_derived_volume
    data=dataset(tmp_path)
    write_rows(data,"trades",[trade(condition="UNKNOWN_NUMERIC_123")])
    at=datetime(2026,11,9,9,31,tzinfo=NY)
    entry=selected(data.snapshot(at))
    assert entry.last_trade_price==1 and "UNKNOWN_TRADE_CONDITION" in entry.last_trade_quality
    observations=data.theta_reader.trades(data.catalog.get("trades",date(2026,11,9),date(2026,11,13)),at,600,data.mapping("trades"))
    assert not trade_derived_volume(observations)


def test_delayed_quote_publication_at_decision_is_phase30_not_already_observed(tmp_path):
    data=dataset(tmp_path)
    raw=trade(timestamp="2026-11-09T09:30:30",known_at="2026-11-09T09:31:00")
    row={key:raw[key] for key in ("symbol","expiration","strike","right","timestamp","known_at","sequence")}
    row.update(bid="1",ask="1.1",implied_vol=".25")
    write_rows(data,"greeks",[row])
    at=datetime(2026,11,9,9,31,tzinfo=NY)
    assert not data.snapshot(at).quotes
    assert data.snapshot(at,inclusive=True).quotes
    events=[event for event in data.events_between(at,at+timedelta(minutes=1)) if event.kind=="quote"]
    assert len(events)==1 and events[0].phase==30 and events[0].event_at==at
    data.mappings["greeks"]=replace(synthetic_mappings()["options"],timestamp_precision="minute")
    row.update(timestamp="2026-11-09T09:30:00",known_at="2026-11-09T09:32:00")
    write_rows(data,"greeks",[row])
    assert not data.snapshot(at+timedelta(minutes=1)).quotes
    delayed=[event for event in data.events_between(at+timedelta(minutes=1),at+timedelta(minutes=2)) if event.kind=="quote"]
    assert len(delayed)==1 and delayed[0].phase==30
