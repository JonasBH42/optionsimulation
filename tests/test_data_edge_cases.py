from datetime import date, datetime, timedelta
from decimal import Decimal
import csv
import gzip
from pathlib import Path
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from qqq_sim.data.asof import HistoricalData
from qqq_sim.data.fixtures import synthetic_mappings, write_synthetic_dataset
from qqq_sim.data.models import ContractKey, DataError, NormalizationMapping, TradeObservation
from qqq_sim.data.normalization import canonical_schema, row_timestamp
from qqq_sim.data.theta import trade_derived_volume
from qqq_sim.data.parameters import ParameterRegistry, ReferenceRecord
from qqq_sim.replay.calendar import NY


def make_data(tmp_path, minutes=4):
    manifest = write_synthetic_dataset(tmp_path / "data", minutes=minutes)
    return HistoricalData(manifest["underlying_root"], manifest["options_root"], synthetic_mappings())


def edit_gzip(path, change):
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        names, rows = reader.fieldnames, list(reader)
    change(rows)
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def test_zero_volume_invalid_geometry_and_duplicate_minute(tmp_path):
    data = make_data(tmp_path)
    path = next(data.catalog.underlying_root.glob("*.parquet"))
    rows = pq.read_table(path).to_pylist()
    rows[1]["volume"] = 0
    rows[2]["high"] = rows[2]["low"] - 1
    pq.write_table(pa.Table.from_pylist(rows), path)
    result = data.snapshot(datetime(2026, 11, 9, 9, 34, tzinfo=NY))
    assert len(result.candles) == 2
    assert {item.code for item in result.diagnostics} >= {"ZERO_VOLUME_EXCLUDED", "INVALID_CANDLE"}
    rows.append(dict(rows[0]))
    pq.write_table(pa.Table.from_pylist(rows), path)
    result = data.snapshot(datetime(2026, 11, 9, 9, 34, tzinfo=NY))
    assert "DUPLICATE_UNDERLYING_TIMESTAMP" in {item.code for item in result.diagnostics}


def test_boundary_is_not_391st_bar_and_end_labels_not_shifted_twice(tmp_path):
    data = make_data(tmp_path, minutes=390)
    path = next(data.catalog.underlying_root.glob("*.parquet"))
    rows = pq.read_table(path).to_pylist()
    extra = dict(rows[-1], regular_time=datetime(2026, 11, 9, 16))
    rows.append(extra)
    pq.write_table(pa.Table.from_pylist(rows), path)
    result = data.snapshot(datetime(2026, 11, 9, 16, tzinfo=NY))
    assert len(result.candles) == 390 and len(result.boundary_observations) == 1
    rows = [{**row, "regular_time": row["regular_time"] + timedelta(minutes=1)} for row in rows[:-1]]
    pq.write_table(pa.Table.from_pylist(rows), path)
    data.mappings["underlying"] = NormalizationMapping(bar_label_semantics="END")
    result = data.snapshot(datetime(2026, 11, 9, 9, 31, tzinfo=NY))
    assert len(result.candles) == 1 and result.candles[0].label.minute == 30


def test_mismatched_expiry_and_unknown_greek_do_not_invent_values(tmp_path):
    data = make_data(tmp_path)
    path = next((data.catalog.options_root / "greeks").glob("*.csv.gz"))
    def change(rows):
        rows[0]["expiration"] = "2026-12-18"
        rows[4]["delta"] = "malformed"
    edit_gzip(path, change)
    result = data.snapshot(datetime(2026, 11, 9, 9, 31, tzinfo=NY))
    assert len(result.quotes) == 9
    assert any(item.code == "EXPIRY_IDENTITY_MISMATCH" for item in result.diagnostics)
    assert any("VENDOR_GREEK_INVALID:delta" in entry.latest_event.quality_flags for entry in result.quotes)


def test_conflicting_same_time_without_sequence_cannot_fill(tmp_path):
    data = make_data(tmp_path)
    path = next((data.catalog.options_root / "greeks").glob("*.csv.gz"))
    def change(rows):
        duplicate = dict(rows[0], bid="3.00", ask="3.10", sequence="")
        rows[0]["sequence"] = ""
        rows.append(duplicate)
    edit_gzip(path, change)
    result = data.snapshot(datetime(2026, 11, 9, 9, 31, tzinfo=NY))
    assert any(entry.status == "INVALID" and not entry.latest_event.valid for entry in result.quotes)
    assert any(item.code == "AMBIGUOUS_EVENT_ORDER" for item in result.diagnostics)


def test_contradictory_redundant_timestamps_rejected():
    row = {"timestamp": "2026-11-09T09:30:00", "date": "20261109", "ms_of_day": "34201000"}
    with pytest.raises(DataError, match="CONTRADICTORY_TIMESTAMPS"):
        row_timestamp(row, canonical_schema(row), date(2026, 11, 9), NormalizationMapping())


def test_trade_correction_is_prospective_and_unknown_condition_excluded():
    contract = ContractKey(date(2026, 11, 13), Decimal(600), "C")
    t0 = datetime(2026, 11, 9, 10, tzinfo=NY)
    original = TradeObservation(contract, t0, t0, Decimal(2), 10, "t1", 1, "REGULAR", True, source_id="1")
    unknown = TradeObservation(contract, t0, t0, Decimal(2), 1000, "t2", 2, "UNKNOWN", False, source_id="2")
    correction = TradeObservation(contract, t0, t0 + timedelta(hours=1), Decimal(2), 4, "t3", 3, "REGULAR", True, "CORRECT", "t1", "3")
    assert trade_derived_volume([original, unknown])[contract] == 10
    assert trade_derived_volume([original, unknown, correction])[contract] == 4


def test_large_and_small_event_windows_identical(tmp_path):
    data = make_data(tmp_path, minutes=8)
    start = datetime(2026, 11, 9, 9, 31, tzinfo=NY)
    large = data.events_between(start, start + timedelta(minutes=5))
    small = tuple(event for i in range(5) for event in data.events_between(start + timedelta(minutes=i), start + timedelta(minutes=i + 1)))
    assert [event.key for event in large] == [event.key for event in small]


def test_early_close_fixture_contains_210_minutes(tmp_path):
    manifest = write_synthetic_dataset(tmp_path / "early", day=date(2026, 11, 27))
    data = HistoricalData(manifest["underlying_root"], manifest["options_root"], synthetic_mappings())
    result = data.snapshot(datetime(2026, 11, 27, 13, tzinfo=NY))
    assert len(result.candles) == 210
    assert result.candles[-1].label.strftime("%H:%M") == "12:59"
    assert result.candles[-1].available_at.strftime("%H:%M") == "13:00"


def test_required_expiry_missing_stops_but_sparse_quotes_do_not(tmp_path):
    data = make_data(tmp_path)
    start, end = datetime(2026, 11, 9, 9, 31, tzinfo=NY), datetime(2026, 11, 9, 9, 34, tzinfo=NY)
    key = ContractKey(date(2026, 11, 13), Decimal(600), "C")
    assert data.coverage_gap(start, end, [key]) is None
    unavailable = ContractKey(date(2026, 11, 20), Decimal(600), "C")
    assert data.coverage_gap(start, end, [unavailable.contract_id])[1] == "PARTITION_MISSING"
    path = next((data.catalog.options_root / "greeks").glob("*.csv.gz"))
    edit_gzip(path, lambda rows: rows.__setitem__(slice(None), [row for row in rows if "09:30" in row["timestamp"]]))
    assert data.coverage_gap(start, end, [key]) is None


def test_option_temporal_semantics_are_not_assumed(tmp_path):
    data = make_data(tmp_path)
    data.mappings["ohlcv"] = NormalizationMapping(strike_divisor=Decimal(1), bar_label_semantics="UNKNOWN")
    snapshot = data.snapshot(datetime(2026, 11, 9, 9, 31, tzinfo=NY))
    assert any(item.code == "TEMPORAL_SEMANTICS_UNVERIFIED" for item in snapshot.diagnostics)


def test_minute_quote_completion_precedes_barrier_without_reusing_seen_quote(tmp_path):
    data = make_data(tmp_path)
    path = next((data.catalog.options_root / "greeks").glob("*.csv.gz"))
    def correct_times(rows):
        for row in rows:
            stamp = datetime.fromisoformat(row["timestamp"])
            row["timestamp"] = stamp.replace(second=0).isoformat()
    edit_gzip(path, correct_times)
    data.mappings["options"] = NormalizationMapping(strike_divisor=Decimal(1), iv_unit="decimal", metadata_verified=True, timestamp_precision="minute")
    start = datetime(2026, 11, 9, 9, 31, tzinfo=NY)
    snapshot = data.snapshot(start)
    assert all(entry.latest_event.event_at.minute == 30 and entry.latest_event.available_at == start for entry in snapshot.quotes)
    events = [event for event in data.events_between(start, start + timedelta(minutes=1)) if event.kind == "quote"]
    assert len(events) == 10
    assert all(event.phase == 20 and event.event_at == start + timedelta(minutes=1) and event.payload.event_at.minute == 31 for event in events)
    assert not any(event.payload.source_id in {entry.latest_event.source_id for entry in snapshot.quotes} for event in events)


def test_reference_rate_units_require_explicit_compounding_and_scale():
    instant = datetime(2026, 11, 9, 10, tzinfo=NY)
    rate = ReferenceRecord("rate", "QQQ", 4, "annual_cc_percent", instant, instant, instant)
    carry = ReferenceRecord("continuous_carry", "QQQ", 1, "percent", instant, instant, instant)
    result = ParameterRegistry([rate, carry]).resolve("QQQ", instant)
    assert result["rate"] == 0.04 and result["carry"] is None
    assert "AMBIGUOUS_CARRY_UNITS" in result["reasons"]


def test_replay_default_marks_old_valid_quote_stale_without_hiding_it(tmp_path):
    data = make_data(tmp_path)
    path = next((data.catalog.options_root / "greeks").glob("*.csv.gz"))
    edit_gzip(path, lambda rows: rows.__setitem__(slice(None), [row for row in rows if "09:30" in row["timestamp"]]))
    snapshot = data.snapshot(datetime(2026, 11, 9, 9, 34, tzinfo=NY))
    assert snapshot.quotes and all(entry.status == "STALE" for entry in snapshot.quotes)
    assert all(entry.quote is not None and entry.age_seconds == 210 for entry in snapshot.quotes)
    unrestricted = data.snapshot(snapshot.asof, max_quote_age=None)
    assert all(entry.age_seconds == 210 for entry in unrestricted.quotes)
