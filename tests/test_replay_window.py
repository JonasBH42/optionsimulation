import csv
import gzip
from dataclasses import replace
from datetime import date, datetime, timedelta

import pytest

from qqq_sim.data.asof import HistoricalData
from qqq_sim.data.fixtures import synthetic_mappings, write_synthetic_dataset
from qqq_sim.data.models import DataError
from qqq_sim.data.theta import IndexedPartition
from qqq_sim.replay.calendar import local_datetime
from qqq_sim.services.adapters import broker_contract


DAY = date(2026, 11, 9)
START = local_datetime(datetime(2026, 11, 9, 10, 1))


def make_data(tmp_path, *, precision="subminute", change_rows=None, inferred=False):
    roots = write_synthetic_dataset(tmp_path / "fixture", minutes=45)
    mapping = synthetic_mappings()["options"]
    if precision != "subminute" or change_rows is not None:
        from pathlib import Path
        path = next((Path(roots["options_root"]) / "greeks").glob("*.gz"))
        with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            names, rows = list(reader.fieldnames), list(reader)
        if precision == "minute":
            for row in rows:
                row["timestamp"] = datetime.fromisoformat(row["timestamp"]).replace(second=0).isoformat()
        if change_rows is not None:
            change_rows(rows)
            names = list(dict.fromkeys((*names, *(name for row in rows for name in row))))
        with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            writer.writerows(rows)
    mapping = replace(mapping, timestamp_precision=precision,
                      strike_divisor=None if inferred else mapping.strike_divisor)
    data = HistoricalData(roots["underlying_root"], roots["options_root"], mappings={"options": mapping})
    data.catalog.scan()
    return data


def terminal_reference(data, start, end):
    latest = {}
    while start < end:
        following = min(end, data.calendar.advance_trading_minutes(start, 1))
        for event in data.events_between(start, following):
            if event.kind == "quote":
                latest[event.payload.contract] = event
        start = following
    return tuple(sorted(latest.values(), key=lambda event: event.key))


@pytest.mark.parametrize("precision", ["subminute", "minute"])
def test_window_matches_causal_events_and_terminal_updates(tmp_path, precision):
    data = make_data(tmp_path, precision=precision)
    end = START + timedelta(minutes=8)
    window = data.prepare_replay_window(START, end)
    assert window.fast_quotes
    assert window.events_between(START, end) == data.events_between(START, end)
    assert window.terminal_quote_events(START, end) == terminal_reference(data, START, end)
    assert window.coverage_gap(START, end) is None


def test_terminal_updates_keep_invalid_latest_and_exclude_untouched(tmp_path):
    def change(rows):
        for row in rows:
            if row["strike"] == "590" and row["right"] == "C":
                if row["timestamp"] > "2026-11-09T10:02:30":
                    row["bid"], row["ask"] = "3", "1"
            if row["strike"] == "610" and row["right"] == "P":
                row["timestamp"] = "2026-11-09T09:30:30"
    data = make_data(tmp_path, change_rows=change)
    # Repeated pre-start observations conflict, so this intentionally uses the
    # fallback path while retaining the original invalid-event behavior.
    end = START + timedelta(minutes=5)
    window = data.prepare_replay_window(START, end)
    expected = terminal_reference(data, START, end)
    assert window.terminal_quote_events(START, end) == expected
    assert not any(event.payload.contract.strike == 610 and event.payload.contract.right == "P" for event in expected)
    assert not next(event.payload for event in expected if event.payload.contract.strike == 590 and event.payload.contract.right == "C").valid


def test_clean_terminal_updates_keep_invalid_latest(tmp_path):
    def change(rows):
        for row in rows:
            if row["strike"] == "590" and row["right"] == "C" and row["timestamp"] > "2026-11-09T10:02:30":
                row["bid"], row["ask"] = "3", "1"
    data = make_data(tmp_path, change_rows=change)
    end = START + timedelta(minutes=5)
    window = data.prepare_replay_window(START, end)
    assert window.fast_quotes
    assert window.terminal_quote_events(START, end) == terminal_reference(data, START, end)


@pytest.mark.parametrize("kind", ["delayed", "conflict", "inferred"])
def test_unusual_histories_use_same_terminal_arrival_behavior(tmp_path, kind):
    def change(rows):
        if kind == "delayed":
            rows[31]["known_at"] = "2026-11-09T10:04:15"
        elif kind == "conflict":
            duplicate = dict(rows[31], bid="3.01")
            rows.append(duplicate)
    data = make_data(tmp_path, change_rows=change, inferred=kind == "inferred")
    end = START + timedelta(minutes=6)
    window = data.prepare_replay_window(START, end)
    assert window.fast_quotes == (kind == "inferred")
    assert window.terminal_quote_events(START, end) == terminal_reference(data, START, end)


def test_clean_window_normalizes_only_selected_legs_or_terminal_rows(tmp_path, monkeypatch):
    data = make_data(tmp_path)
    end = START + timedelta(minutes=10)
    window = data.prepare_replay_window(START, end)
    full = data.events_between(START, end)
    contract = next(event.payload.contract for event in full if event.kind == "quote")
    actual_broker_contract = broker_contract(contract, data.calendar)
    original = data.theta_reader._contract
    normalized = []
    def count(*args, **kwargs):
        normalized.append(args[0]["timestamp"])
        return original(*args, **kwargs)
    monkeypatch.setattr(data.theta_reader, "_contract", count)
    selected = window.events_between(START, end, (actual_broker_contract,))
    assert selected == tuple(event for event in full if event.kind == "candle" or event.payload.contract == contract)
    assert len(normalized) == 10
    normalized.clear()
    terminal = window.terminal_quote_events(START, end, excluded_contracts=(actual_broker_contract,))
    assert len(terminal) == len(normalized) == 9
    assert all(event.payload.contract != contract for event in terminal)


def test_preparing_future_bad_rows_defers_original_failure(tmp_path):
    def change(rows):
        rows[40]["expiration"] = "2026-11-20"
    data = make_data(tmp_path, change_rows=change)
    window = data.prepare_replay_window(START, START + timedelta(minutes=12))
    assert window.fast_quotes
    assert window.events_between(START, START + timedelta(minutes=1)) == data.events_between(START, START + timedelta(minutes=1))


def test_preparation_honors_cancellation(tmp_path):
    data = make_data(tmp_path)
    with pytest.raises(InterruptedError):
        data.prepare_replay_window(START, START + timedelta(minutes=1), cancel=lambda: True)


def test_normalized_underlying_cache_reuses_candles_and_defers_future_errors(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    data = make_data(tmp_path)
    partition = data.catalog.get("underlying", DAY)
    rows = pq.read_table(partition.path).to_pylist()
    rows.append(dict(rows[40]))
    rows[39]["volume"] = 0
    pq.write_table(pa.Table.from_pylist(rows), partition.path)
    partition = data.catalog.get("underlying", DAY)
    first = data.underlying_reader.read(partition, START)
    later = data.underlying_reader.read(partition, START + timedelta(minutes=2))
    assert later[0][0] is first[0][0]
    assert not first[2]
    at_zero = data.underlying_reader.read(partition, START + timedelta(minutes=9))
    assert any(item.code == "ZERO_VOLUME_EXCLUDED" for item in at_zero[2])
    with pytest.raises(DataError, match="DUPLICATE_UNDERLYING_TIMESTAMP"):
        data.underlying_reader.read(partition, START + timedelta(minutes=10))
    assert data.underlying_reader.read(partition, START) == first


@pytest.mark.parametrize("problem", ["invalid_latest", "expiry", "size", "sequence", "timestamp", "inferred_first_valid"])
def test_indexed_destination_snapshot_preserves_full_history_identity(tmp_path, problem):
    def change(rows):
        if problem == "invalid_latest":
            rows[40]["bid"], rows[40]["ask"] = "9", "1"
        elif problem == "expiry":
            rows[40]["expiration"] = "2026-11-20"
        elif problem == "size":
            rows[40]["bid_size"] = "unavailable"
        elif problem == "sequence":
            rows[40]["sequence"] = "unknown"
        elif problem == "timestamp":
            rows[40]["timestamp"] = "bad-timestamp"
        elif problem == "inferred_first_valid":
            for row in rows[1:45]:
                row["bid"], row["ask"] = "9", "1"
    indexed = make_data(tmp_path, change_rows=change, inferred=problem == "inferred_first_valid")
    reference = HistoricalData(indexed.catalog.underlying_root, indexed.catalog.options_root,
                               mappings=indexed.mappings, calendar=indexed.calendar)
    reference.catalog.scan()
    end = START + timedelta(minutes=12)
    expected = reference.snapshot(end)
    assert indexed.prepare_replay_window(START, end).fast_quotes
    actual = indexed.snapshot(end)
    assert actual == expected
    assert actual.snapshot_id == expected.snapshot_id


def test_source_cache_owns_raw_temporal_and_replay_indexes_together(tmp_path):
    data = make_data(tmp_path)
    end = START + timedelta(minutes=8)
    data.snapshot(START)
    partition = data.catalog.get("greeks", DAY, date(2026, 11, 13))
    key = data.theta_reader._partition_cache_key(partition)
    original = data.cache.get(key)
    assert isinstance(original, IndexedPartition) and original.index is not None
    data.prepare_replay_window(START, end)
    prepared = data.cache.get(key)
    assert prepared.partition is original.partition and prepared.index is original.index
    assert prepared.replay_index.partition is original.partition
    data.snapshot(end)
    assert data.cache.get(key).replay_index is prepared.replay_index
    assert data.cache.bytes <= data.cache.max_bytes


def rewrite_partition(data, kind, change):
    partition = data.catalog.get(kind, DAY, date(2026, 11, 13))
    with gzip.open(partition.path, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    change(rows)
    names = list(dict.fromkeys(name for row in rows for name in row))
    with gzip.open(partition.path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("problem", ["numeric_condition", "same_time_conflict", "duplicate_id", "correction", "bad_sequence", "bad_size", "delayed"])
def test_prepared_trade_groups_match_reference_and_preserve_unsafe_flags(tmp_path, problem):
    data = make_data(tmp_path)
    def change(rows):
        if problem == "numeric_condition":
            for row in rows:
                row["condition"] = "18"
        elif problem == "same_time_conflict":
            rows.append(dict(rows[40], price="3.1415"))
        elif problem in ("duplicate_id", "correction"):
            for offset, row in enumerate(rows):
                row["trade_id"] = str(offset)
            if problem == "duplicate_id":
                rows[40]["trade_id"] = "39"
            else:
                rows[40].update(action="CANCEL", correction_of="39")
                for offset, row in enumerate(rows):
                    row.setdefault("action", "TRADE")
        elif problem == "bad_sequence":
            rows[40]["sequence"] = "unknown"
        elif problem == "bad_size":
            rows[40]["size"] = "unknown"
        else:
            rows[40]["known_at"] = "2026-11-09T10:12:15"
    rewrite_partition(data, "trades", change)
    reference = HistoricalData(data.catalog.underlying_root, data.catalog.options_root,
                               mappings=data.mappings, calendar=data.calendar)
    reference.catalog.scan()
    end = START + timedelta(minutes=12)
    window = data.prepare_replay_window(START, end)
    assert window.fast_quotes == (problem in ("numeric_condition", "same_time_conflict"))
    assert window.requires_snapshot_validation == (problem not in ("numeric_condition", "same_time_conflict"))
    assert data.snapshot(START) == reference.snapshot(START)
    if problem == "bad_sequence":
        with pytest.raises(ValueError):
            data.snapshot(end)
        with pytest.raises(ValueError):
            reference.snapshot(end)
    else:
        assert data.snapshot(end) == reference.snapshot(end)


@pytest.mark.parametrize("kind", ["ohlcv", "oi_before"])
def test_auxiliary_future_scalar_errors_require_snapshot_validation(tmp_path, kind):
    data = make_data(tmp_path)
    def change(rows):
        if kind == "ohlcv":
            rows[40]["close"] = "unknown"
        else:
            rows[0]["open_interest"] = "unknown"
            rows[0]["known_at"] = "2026-11-09T10:10:00"
    rewrite_partition(data, kind, change)
    window = data.prepare_replay_window(START, START + timedelta(minutes=12))
    assert window.fast_quotes and window.requires_snapshot_validation
    data.snapshot(START)
    with pytest.raises((ValueError, ArithmeticError)):
        data.snapshot(START + timedelta(minutes=12))
