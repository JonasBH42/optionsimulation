import csv
import gzip
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from qqq_sim.data.cache import ByteLRU
from qqq_sim.data.models import DataError, NormalizationMapping, Partition, PartitionKey
from qqq_sim.data.theta import IndexedPartition, ThetaReader
from qqq_sim.replay.calendar import CalendarService, NY


def stamp(text):
    return datetime.fromisoformat("2026-11-09T" + text).replace(tzinfo=NY)


def reader_for(tmp_path, observations, precision="subminute", budget=1024 * 1024):
    rows = [{"symbol": "QQQ", "expiration": "2026-11-13", "strike": "600", "right": "C",
             "timestamp": time, "bid": "2.00", "ask": "2.10", **fields}
            for time, fields in observations]
    names = tuple(dict.fromkeys(key for row in rows for key in row))
    path = tmp_path / "greeks_20261109_exp=2026-11-13.csv.gz"
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    stat = path.stat()
    partition = Partition(PartitionKey("greeks", date(2026, 11, 9), date(2026, 11, 13)),
                          str(path), stat.st_size, stat.st_mtime_ns)
    mapping = NormalizationMapping(strike_divisor=Decimal(1), timestamp_precision=precision)
    return ThetaReader(CalendarService(), ByteLRU(budget)), partition, mapping


def in_window(quote, start):
    completed_minute = (quote.timestamp_precision == "minute" and
                        quote.available_at == quote.event_at.replace(second=0, microsecond=0) + timedelta(minutes=1))
    return quote.available_at > start or quote.available_at == start and not completed_minute


@pytest.mark.parametrize("precision", ["subminute", "minute"])
@pytest.mark.parametrize("budget", [1024 * 1024, 1])
def test_quote_window_matches_full_history_with_delayed_knowledge_and_barriers(tmp_path, precision, budget):
    reader, partition, mapping = reader_for(tmp_path, [
        ("09:57:00", {}),
        ("09:58:00", {"known_at": "10:00:00", "sequence": "1"}),
        ("09:59:00", {}),
        ("09:59:30", {"known_at": "10:01:00", "sequence": "2"}),
        ("10:00:00", {}),
        ("10:01:00", {}),
    ], precision, budget)
    for start, end in ((stamp("10:00:00"), stamp("10:01:00")),
                       (stamp("10:01:00"), stamp("10:02:00"))):
        full, _ = reader.quotes(partition, end, 600, mapping)
        window, _ = reader.quotes(partition, end, 600, mapping, interval_start=start)
        assert window == tuple(quote for quote in full if in_window(quote, start))


@pytest.mark.parametrize("sequences", [("", ""), ("1", "1"), ("1", "2")])
def test_window_keeps_previously_known_same_time_quotes_for_conflict_resolution(tmp_path, sequences):
    reader, partition, mapping = reader_for(tmp_path, [
        ("09:59:00", {"known_at": "09:59:30", "bid": "2.00", "sequence": sequences[0]}),
        ("09:59:00", {"known_at": "10:00:30", "bid": "2.05", "sequence": sequences[1]}),
        # A not-yet-known conflict cannot contaminate either query.
        ("09:59:00", {"known_at": "10:02:00", "bid": "2.08", "sequence": ""}),
    ])
    start, end = stamp("10:00:00"), stamp("10:01:00")
    full, full_errors = reader.quotes(partition, end, 600, mapping)
    window, errors = reader.quotes(partition, end, 600, mapping, interval_start=start)
    assert window == tuple(quote for quote in full if in_window(quote, start))
    assert errors == full_errors
    if sequences != ("1", "2"):
        assert all(not quote.valid for quote in window)


def test_indexed_window_normalizes_only_arriving_timestamp_groups(tmp_path, monkeypatch):
    observations = [((stamp("10:00:00") + timedelta(minutes=minute, seconds=30)).isoformat(),
                     {"strike": str(strike), "sequence": str(minute)})
                    for minute in range(30) for strike in (590, 595, 600, 605, 610)]
    reader, partition, mapping = reader_for(tmp_path, observations)
    end, start = stamp("10:30:00"), stamp("10:29:00")
    full, _ = reader.quotes(partition, end, 600, mapping)
    normalized = []
    original = reader._contract

    def count(*args, **kwargs):
        normalized.append(args[0]["timestamp"])
        return original(*args, **kwargs)

    monkeypatch.setattr(reader, "_contract", count)
    window, _ = reader.quotes(partition, end, 600, mapping, interval_start=start)
    assert window == tuple(quote for quote in full if in_window(quote, start))
    assert len(normalized) == 5
    assert len(full) == 150


def test_expiry_mismatch_includes_source_row_for_volume_diagnostic(tmp_path):
    reader, partition, mapping = reader_for(tmp_path, [
        ("09:59:00", {"expiration": "2026-11-20"}),
    ])
    with pytest.raises(DataError) as error:
        reader.volumes(partition, stamp("10:01:00"), 600, mapping)
    assert error.value.code == "EXPIRY_IDENTITY_MISMATCH"
    assert error.value.details == {"path": partition.path, "row_number": 2}


def test_compact_raw_rows_preserve_exact_csv_strings_and_immutable_storage(tmp_path):
    reader, partition, _ = reader_for(tmp_path, [
        ("09:59:00", {"sequence": "000123", "vendor_note": 'comma,quote"\nNUL\x00\u05e9\u05dc\u05d5\u05dd'}),
        ("10:00:00", {"sequence": "001234", "vendor_note": ""}),
    ])
    with gzip.open(partition.path, "rt", newline="", encoding="utf-8") as stream:
        source_rows = list(csv.reader(stream))
    raw = reader.raw_partition(partition)
    names, rows = raw
    assert names == tuple(source_rows[0])
    assert list(rows) == [tuple(row) for row in source_rows[1:]]
    assert rows[-1] == tuple(source_rows[-1])
    assert rows[:1] == (tuple(source_rows[1]),)
    assert raw.rows.offsets.readonly and raw.row_numbers.readonly
    assert reader.cache.bytes == raw.size_bytes


def test_temporal_cache_owns_raw_storage_once_and_reuses_it_without_reopening(tmp_path, monkeypatch):
    reader, partition, mapping = reader_for(tmp_path, [
        ("09:59:00", {}), ("10:00:00", {}), ("10:01:00", {}),
    ])
    names, records = reader.visible_records(partition, stamp("10:01:00"), mapping)
    list(records)
    assert len(reader.cache._items) == 1
    cached, size = next(iter(reader.cache._items.values()))
    assert isinstance(cached, IndexedPartition)
    assert size == cached.partition.size_bytes + len(cached.index.records) * 320
    assert reader.raw_partition(partition) is cached.partition

    def no_open(*args, **kwargs):
        raise AssertionError("A warm time window must use the retained raw/index entry")

    monkeypatch.setattr("qqq_sim.data.theta.gzip.open", no_open)
    _, records = reader.visible_records(partition, stamp("10:02:00"), mapping, interval_start=stamp("10:01:00"))
    assert len(list(records)) == 1


def test_raw_partition_stops_at_budget_and_visible_reader_streams(tmp_path):
    reader, partition, mapping = reader_for(tmp_path, [
        ("09:59:00", {"vendor_note": "x" * 2000}),
    ], budget=1024)
    with pytest.raises(DataError, match="PARTITION_EXCEEDS_MEMORY_BUDGET"):
        reader.raw_partition(partition)
    quotes, _ = reader.quotes(partition, stamp("10:01:00"), 600, mapping)
    assert len(quotes) == 1
    assert reader.cache.bytes <= reader.cache.max_bytes
