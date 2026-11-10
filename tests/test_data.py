from __future__ import annotations
import csv
import gzip
from datetime import date, datetime, timedelta
from decimal import Decimal
import json
from pathlib import Path
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from qqq_sim.data.asof import HistoricalData
from qqq_sim.data.catalog import CatalogService, file_hash
from qqq_sim.data.fixtures import synthetic_mappings, write_synthetic_dataset
from qqq_sim.data.models import DataError, NormalizationMapping, stable_hash
from qqq_sim.data.normalization import canonical_schema, legacy_whole_session, parse_timestamp, resolve_iv, resolve_strike, row_timestamp
from qqq_sim.data.parameters import ParameterRegistry, ReferenceRecord
from qqq_sim.data.underlying import aggregate_candles, indicators
from qqq_sim.data.mirror import valid_mirror, write_mirror
from qqq_sim.replay.calendar import NY


@pytest.fixture
def data(tmp_path):
    manifest = write_synthetic_dataset(tmp_path / "data", minutes=10)
    return HistoricalData(manifest["underlying_root"], manifest["options_root"], synthetic_mappings())


def test_open_and_first_complete_minute(data):
    assert data.snapshot(datetime(2026, 11, 9, 9, 30, tzinfo=NY)).spot is None
    snapshot = data.snapshot(datetime(2026, 11, 9, 9, 31, tzinfo=NY))
    assert len(snapshot.candles) == 1 and snapshot.spot == 600.02
    assert len(snapshot.quotes) == 10
    assert all(entry.quote.event_at < snapshot.asof for entry in snapshot.quotes)
    assert all(entry.known_oi < 10000 for entry in snapshot.quotes)
    assert all(entry.volume == 10 for entry in snapshot.quotes)
    assert snapshot.spot != float(dict(snapshot.quotes[0].quote.raw_fields)["underlying_price"])


def test_partial_aggregation_and_indicators(data):
    snapshot = data.snapshot(datetime(2026, 11, 9, 9, 33, tzinfo=NY))
    candles = aggregate_candles(snapshot.candles, 5)
    assert len(candles) == 1 and candles[0].partial and candles[0].source_count == 3
    assert candles[0].volume == 3003
    assert indicators(snapshot.candles, 3)["sma"][:2] == (None, None)


def test_future_poisoning_keeps_snapshot_identical(data):
    cutoff = datetime(2026, 11, 9, 9, 33, tzinfo=NY)
    before = data.snapshot(cutoff).to_dict()
    source = next(data.catalog.underlying_root.glob("*.parquet"))
    rows = pq.read_table(source).to_pylist()
    for row in rows:
        if row["regular_time"] >= cutoff.replace(tzinfo=None):
            row.update(open=1e9, high=1e9, low=1e9, close=1e9, volume=1e9)
    pq.write_table(pa.Table.from_pylist(rows), source)
    quotes = next((data.catalog.options_root / "greeks").glob("*.csv.gz"))
    with gzip.open(quotes, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        names, values = reader.fieldnames, list(reader)
    for row in values:
        if parse_timestamp(row["timestamp"]) >= cutoff:
            row.update(bid="1000000000", ask="1000000001", implied_vol="4.0", strike="99999")
    with gzip.open(quotes, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(values)
    after = data.snapshot(cutoff).to_dict()
    assert before == after


def test_quote_boundary_exclusive_and_same_instant_next_advance(data):
    cutoff = datetime(2026, 11, 9, 9, 31, tzinfo=NY)
    path = next((data.catalog.options_root / "greeks").glob("*.csv.gz"))
    with gzip.open(path, "rt", newline="") as source:
        reader = csv.DictReader(source)
        names, rows = reader.fieldnames, list(reader)
    rows[0]["timestamp"] = "2026-11-09T09:31:00"
    with gzip.open(path, "wt", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    assert len(data.snapshot(cutoff).quotes) == 9
    assert len(data.snapshot(cutoff, inclusive=True).quotes) == 10
    events = data.events_between(cutoff, cutoff + timedelta(minutes=1))
    assert any(event.event_at == cutoff and event.kind == "quote" for event in events)
    assert all(event.event_at >= cutoff for event in events)


def test_invalid_latest_keeps_prior_only_as_stale_mark(data):
    path = next((data.catalog.options_root / "greeks").glob("*.csv.gz"))
    with gzip.open(path, "rt", newline="") as source:
        reader = csv.DictReader(source)
        names, rows = reader.fieldnames, list(reader)
    rows[1]["bid"], rows[1]["ask"] = "10", "1"
    with gzip.open(path, "wt", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    snapshot = data.snapshot(datetime(2026, 11, 9, 9, 32, tzinfo=NY))
    invalid = [entry for entry in snapshot.quotes if entry.status == "INVALID_LATEST"]
    assert len(invalid) == 1 and invalid[0].quote.valid and not invalid[0].latest_event.valid


def test_catalog_prefix_duplicates_and_bound_changes(data):
    report = data.catalog.scan()
    assert report.ok
    path = next((data.catalog.options_root / "greeks").glob("*.csv.gz"))
    duplicate = path.with_name("copy_" + path.name)
    duplicate.write_bytes(path.read_bytes())
    assert not data.catalog.scan().ok
    with pytest.raises(DataError, match="DUPLICATE_LOGICAL"):
        data.catalog.get("greeks", date(2026, 11, 9), date(2026, 11, 13))
    duplicate.unlink()
    data.catalog.scan()
    key = data.catalog.get("underlying", date(2026, 11, 9)).key
    data.catalog.bind_revision(keys=[key])
    source = Path(data.catalog.get("underlying", date(2026, 11, 9)).path)
    source.touch()
    with pytest.raises(DataError, match="DATA_REVISION_CHANGED"):
        data.catalog.verify_revision()


def test_missing_slot_not_substituted(tmp_path):
    manifest = write_synthetic_dataset(tmp_path / "gaps", minutes=8, missing_minutes=(2,))
    data = HistoricalData(manifest["underlying_root"], manifest["options_root"], synthetic_mappings())
    start, end = datetime(2026, 11, 9, 9, 31, tzinfo=NY), datetime(2026, 11, 9, 9, 36, tzinfo=NY)
    gap = data.coverage_gap(start, end)
    assert gap == (datetime(2026, 11, 9, 9, 32, tzinfo=NY), "MISSING_SCHEDULED_MINUTE")


def test_time_alias_units_and_legacy_selector():
    schema = canonical_schema(["date", "msofday", "strike_millis", "option_type", "iv", "bid_price", "ask_price"])
    assert schema["strike"] == "strike_millis"
    row = {"date": "20261109", "msofday": "34200000"}
    assert row_timestamp(row, schema, date(2026, 11, 9), NormalizationMapping()).hour == 9
    assert parse_timestamp("2026-11-09T15:00:00+00:00").hour == 10
    assert resolve_strike("600000", 600, NormalizationMapping())[0] == Decimal(600)
    with pytest.raises(DataError, match="AMBIGUOUS_STRIKE_SCALE"):
        resolve_strike("0.0001", 600, NormalizationMapping())
    assert resolve_iv("25", NormalizationMapping(iv_unit="percent"))[0] == 0.25
    assert resolve_iv("0.25", NormalizationMapping())[0] is None
    assert legacy_whole_session("00:00:00", "iv")


def test_no_future_reference_or_zero_rate_default():
    now = datetime(2026, 11, 9, 10, tzinfo=NY)
    record = ReferenceRecord("rate", "QQQ", 0.04, "annual_cc", now, now + timedelta(hours=1), now)
    registry = ParameterRegistry([record])
    assert registry.resolve("QQQ", now)["rate"] is None
    assert registry.resolve("QQQ", now)["carry"] is None
    assert registry.resolve("QQQ", now + timedelta(hours=2))["rate"] == 0.04


def test_mirror_atomic_identity_and_source_immutability(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("original", encoding="utf-8")
    before = file_hash(source)
    target = write_mirror(source, pa.table({"available_at": [1], "contract_key": ["x"], "session": ["2026-11-09"], "source_hash": [before]}), tmp_path / "mirror.parquet")
    assert valid_mirror(source, target) and file_hash(source) == before
    source.write_text("changed", encoding="utf-8")
    assert not valid_mirror(source, target)
