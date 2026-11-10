import csv
import gzip
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from qqq_sim.data.cache import ByteLRU
from qqq_sim.data.models import NormalizationMapping, Partition, PartitionKey
from qqq_sim.data.theta import ThetaReader
from qqq_sim.replay.calendar import CalendarService, NY


def at(text):
    return datetime.fromisoformat("2026-11-09T" + text).replace(tzinfo=NY)


def make_trades(tmp_path, observations, precision="subminute", budget=1024 * 1024):
    rows = [{"symbol": "QQQ", "expiration": "2026-11-13", "strike": "600", "right": "C",
             "timestamp": stamp, "price": "2.00", "size": "10", "condition": "REGULAR", "action": "TRADE", **fields}
            for stamp, fields in observations]
    names = tuple(dict.fromkeys(key for row in rows for key in row))
    path = tmp_path / "trades_20261109_exp=2026-11-13.csv.gz"
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    stat = path.stat()
    partition = Partition(PartitionKey("trades", date(2026, 11, 9), date(2026, 11, 13)),
                          str(path), stat.st_size, stat.st_mtime_ns)
    mapping = NormalizationMapping(strike_divisor=Decimal(1), timestamp_precision=precision)
    return ThetaReader(CalendarService(), ByteLRU(budget)), partition, mapping


def arriving(trade, start, precision):
    phase20 = (precision == "minute" and
               trade.known_at == trade.event_at.replace(second=0, microsecond=0) + timedelta(minutes=1))
    return trade.known_at > start or trade.known_at == start and not phase20


@pytest.mark.parametrize("precision", ["subminute", "minute"])
@pytest.mark.parametrize("budget", [1024 * 1024, 1])
def test_trade_window_matches_causal_history_with_delayed_corrections_and_boundaries(tmp_path, precision, budget):
    reader, partition, mapping = make_trades(tmp_path, [
        ("09:57:00", {"trade_id": "original"}),
        ("09:57:00", {"trade_id": "correction", "action": "CORRECT", "correction_of": "original",
                      "known_at": "10:00:00", "size": "4"}),
        ("09:59:00", {"trade_id": "previous"}),
        ("09:58:00", {"trade_id": "cancel", "action": "CANCEL", "correction_of": "previous",
                      "known_at": "10:00:30"}),
        ("10:00:00", {"trade_id": "boundary", "condition": "UNKNOWN"}),
        ("09:58:30", {"trade_id": "late", "known_at": "10:01:00"}),
        ("10:01:00", {"trade_id": "next"}),
    ], precision, budget)
    for start, end in ((at("10:00:00"), at("10:01:00")), (at("10:01:00"), at("10:02:00"))):
        full = reader.trades(partition, end, 600, mapping)
        window = reader.trades(partition, end, 600, mapping, interval_start=start)
        assert window == tuple(trade for trade in full if arriving(trade, start, precision))


def test_trade_interval_normalizes_only_newly_known_rows(tmp_path, monkeypatch):
    reader, partition, mapping = make_trades(tmp_path, [
        ((at("10:00:00") + timedelta(minutes=minute, seconds=30)).isoformat(),
         {"trade_id": str(minute)}) for minute in range(30)
    ] + [("10:00:30", {"trade_id": "late-correction", "known_at": "10:29:30",
                        "action": "CORRECT", "correction_of": "0"})])
    end, start = at("10:30:00"), at("10:29:00")
    full = reader.trades(partition, end, 600, mapping)
    normalized = []
    original = reader._contract

    def count(*args, **kwargs):
        normalized.append(args[0]["trade_id"])
        return original(*args, **kwargs)

    monkeypatch.setattr(reader, "_contract", count)
    window = reader.trades(partition, end, 600, mapping, interval_start=start)
    assert window == tuple(trade for trade in full if arriving(trade, start, "subminute"))
    assert sorted(normalized) == ["29", "late-correction"]
    assert len(full) == 31
