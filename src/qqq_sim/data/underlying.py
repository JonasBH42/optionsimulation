"""Authoritative OHLCV reading, completed-bar availability, causal aggregation."""
from __future__ import annotations
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from pathlib import Path
from collections import defaultdict
from .models import Candle, DataError, Diagnostic, NormalizationMapping, Partition, stable_hash
from .normalization import parse_timestamp
from .cache import ByteLRU
from qqq_sim.replay.calendar import CalendarService

TIME_COLUMNS = ("regular_time", "datetime", "timestamp", "time", "date")
PRICE_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class UnderlyingIndex:
    candles: tuple
    boundaries: tuple
    diagnostics: tuple
    failures: tuple
    candle_times: tuple
    boundary_times: tuple

    def before(self, cutoff):
        # File order determines the first error, just as in the streaming reader.
        for available_at, error in self.failures:
            if available_at is None or available_at <= cutoff:
                if isinstance(error, DataError):
                    raise DataError(error.code, error.message, **error.details)
                raise type(error)(*error.args)
        return (self.candles[:bisect_right(self.candle_times, cutoff)],
                self.boundaries[:bisect_right(self.boundary_times, cutoff)],
                tuple(item for available_at, item in self.diagnostics if available_at <= cutoff))


class UnderlyingReader:
    def __init__(self, calendar: CalendarService, cache: ByteLRU | None = None):
        self.calendar, self.cache = calendar, cache or ByteLRU()

    def read(self, partition: Partition, cutoff: datetime, mapping: NormalizationMapping | None = None) -> tuple[tuple[Candle, ...], tuple[Candle, ...], tuple[Diagnostic, ...]]:
        import pyarrow.parquet as pq
        mapping = mapping or NormalizationMapping()
        if mapping.bar_label_semantics not in ("START", "END"):
            raise DataError("TEMPORAL_SEMANTICS_UNVERIFIED", mapping.bar_label_semantics)
        session = self.calendar.session(partition.key.session_date)
        normalized_key = (partition.path, partition.size, partition.mtime_ns,
                          "underlying-normalized-v1", stable_hash(mapping), session)
        normalized = self.cache.get(normalized_key)
        if normalized is not None:
            return normalized.before(cutoff)
        key = (partition.path, partition.size, partition.mtime_ns, "underlying-raw-v1", mapping.timestamp_column)
        raw = self.cache.get(key)
        if raw is None:
            schema = pq.read_schema(partition.path)
            timestamp_column = mapping.timestamp_column or next((name for name in TIME_COLUMNS if name in schema.names), None)
            missing = [name for name in PRICE_COLUMNS if name not in schema.names]
            if not timestamp_column or missing:
                raise DataError("REQUIRED_FIELD_MISSING", "Underlying timestamp/OHLCV schema invalid", missing=missing)
            table = pq.read_table(partition.path, columns=[timestamp_column, *PRICE_COLUMNS])
            raw = (timestamp_column, tuple(tuple(row[name] for name in (timestamp_column, *PRICE_COLUMNS)) for row in table.to_pylist()))
            self.cache.put(key, raw, max(table.nbytes * 4, len(raw[1]) * 256))
        timestamp_column, rows = raw
        candles, boundaries, diagnostics, failures, seen = [], [], [], [], set()
        for number, values in enumerate(rows, 1):
            try:
                stamp = parse_timestamp(values[0], partition.key.session_date)
            except (ValueError, TypeError) as error:
                failures.append((None, error))
                continue
            label = stamp - timedelta(minutes=1) if mapping.bar_label_semantics == "END" else stamp
            boundary = mapping.bar_label_semantics == "START" and stamp == session.underlying_close
            available_at = stamp if mapping.bar_label_semantics == "END" or boundary else stamp + timedelta(minutes=1)
            if stamp.date() != partition.key.session_date:
                failures.append((available_at, DataError("SESSION_IDENTITY_MISMATCH", f"Underlying row {number} differs from filename day")))
                continue
            if stamp.second != 0 or stamp.microsecond != 0:
                diagnostics.append((available_at, Diagnostic("INVALID_MINUTE_LABEL", "Underlying OHLCV label is not a minute boundary", partition.path, number)))
                continue
            if session.status != "OPEN":
                failures.append((available_at, DataError("CALENDAR_SOURCE_CONFLICT", "Observations exist on a modeled closed day")))
                continue
            if label < session.exchange_open:
                continue
            if label >= session.underlying_close and not boundary:
                if session.early_close and label.hour < 16:
                    failures.append((available_at, DataError("CALENDAR_SOURCE_CONFLICT", "RTH source extends after declared early close")))
                continue
            if stamp in seen:
                failures.append((available_at, DataError("DUPLICATE_UNDERLYING_TIMESTAMP", stamp.isoformat())))
                continue
            seen.add(stamp)
            try:
                opn, high, low, close, volume = map(float, values[1:])
            except (ValueError, TypeError):
                diagnostics.append((available_at, Diagnostic("INVALID_CANDLE", "Non-numeric OHLCV", partition.path, number)))
                continue
            if not all(math.isfinite(value) for value in (opn, high, low, close, volume)) or volume < 0 or high < low or high < max(opn, close) or low > min(opn, close):
                diagnostics.append((available_at, Diagnostic("INVALID_CANDLE", "Invalid finite/geometry/volume check", partition.path, number)))
                continue
            if volume == 0:
                diagnostics.append((available_at, Diagnostic("ZERO_VOLUME_EXCLUDED", "Canonical profile excludes zero-volume rows", partition.path, number, "INFO")))
                continue
            candle = Candle(label, available_at, opn, high, low, close, volume,
                            stable_hash((Path(partition.path).name, stamp, (opn, high, low, close, volume))), boundary=boundary)
            (boundaries if boundary else candles).append(candle)
        candles = tuple(sorted(candles, key=lambda row: row.label))
        boundaries = tuple(sorted(boundaries, key=lambda row: row.label))
        normalized = UnderlyingIndex(candles, boundaries, tuple(diagnostics), tuple(failures),
                                     tuple(row.available_at for row in candles), tuple(row.available_at for row in boundaries))
        self.cache.put(normalized_key, normalized, 1024 + len(rows) * 768)
        return normalized.before(cutoff)


def aggregate_candles(candles, minutes: int | str, calendar: CalendarService | None = None) -> tuple[Candle, ...]:
    """Input must already be revealed. Buckets anchor to each session's 09:30 open."""
    calendar = calendar or CalendarService()
    if minutes == "daily":
        minutes = 1440
    minutes = int(minutes)
    if minutes <= 0:
        raise ValueError("Aggregation minutes must be positive")
    groups = defaultdict(list)
    for candle in candles:
        if candle.boundary:
            continue
        session = calendar.session(candle.label.date())
        bucket = int((candle.label - session.exchange_open).total_seconds() // 60) // minutes
        groups[(candle.label.date(), bucket)].append(candle)
    result = []
    for (day, bucket), group in sorted(groups.items()):
        group.sort(key=lambda row: row.label)
        session = calendar.session(day)
        label = session.exchange_open + timedelta(minutes=bucket * minutes)
        end = min(label + timedelta(minutes=minutes), session.underlying_close)
        expected = int((end - label).total_seconds() // 60)
        result.append(Candle(label, group[-1].available_at, group[0].open, max(row.high for row in group), min(row.low for row in group), group[-1].close, sum(row.volume for row in group), stable_hash([row.source_id for row in group]), partial=len(group) < expected or group[-1].available_at < end, source_count=len(group)))
    return tuple(result)


def indicators(candles, period: int = 20) -> dict[str, tuple[float | None, ...]]:
    """Causal SMA/EMA/VWAP, computed only from the caller's revealed observations."""
    if period <= 0:
        raise ValueError("Period must be positive")
    closes, sma, ema, vwap = [], [], [], []
    cumulative_volume = cumulative_value = 0.0
    previous_day = None
    for row in candles:
        closes.append(row.close)
        sma.append(sum(closes[-period:]) / period if len(closes) >= period else None)
        ema.append(row.close if not ema else (2 / (period + 1)) * row.close + (1 - 2 / (period + 1)) * ema[-1])
        if row.label.date() != previous_day:
            cumulative_volume = cumulative_value = 0.0
        cumulative_volume += row.volume
        cumulative_value += ((row.high + row.low + row.close) / 3) * row.volume
        vwap.append(cumulative_value / cumulative_volume if cumulative_volume else None)
        previous_day = row.label.date()
    return {"sma": tuple(sma), "ema": tuple(ema), "vwap": tuple(vwap)}
