"""Streaming immutable Theta partitions with capability-specific validation."""
from __future__ import annotations
import csv
import gzip
import json
import math
from array import array
from collections import defaultdict
from dataclasses import dataclass
from bisect import bisect_left, bisect_right
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from .models import ContractKey, DataError, Diagnostic, NormalizationMapping, OptionQuote, Partition, TradeObservation, stable_hash
from .normalization import canonical_schema, decimal_value, field, normalize_right, parse_timestamp, resolve_iv, resolve_strike, row_timestamp, normalize_vendor_greeks
from .cache import ByteLRU
from qqq_sim.replay.calendar import CalendarService

GREEKS = ("delta", "theta", "vega", "rho", "epsilon", "lambda", "gamma", "vanna", "charm", "vomma", "veta", "vera", "speed", "zomma", "color", "ultima", "d1", "d2", "dual_delta", "dual_gamma", "iv_error")


def observation_availability(stamp, row, day, mapping):
    known = parse_timestamp(row["known_at"], day) if row.get("known_at") else stamp
    if known < stamp:
        raise DataError("INVALID_OBSERVATION_AVAILABILITY", "An observation cannot be known before its source event")
    if mapping.timestamp_precision == "minute":
        completion = stamp.replace(second=0, microsecond=0)+timedelta(minutes=1)
        known = max(known, completion)
        return known, 20 if known == completion else 30
    return known, 30


def visible_at(stamp, known, phase, cutoff, inclusive):
    return (stamp <= cutoff and known <= cutoff) if inclusive else stamp < cutoff and (known < cutoff or known == cutoff and phase < 25)


@dataclass(frozen=True)
class CompactRows:
    """Exact CSV field strings without a Python object for every stored field."""
    payload: bytes
    offsets: memoryview

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return tuple(json.loads(self.payload[self.offsets[index]:self.offsets[index+1]]))


@dataclass(frozen=True)
class RawPartition:
    names: tuple[str, ...]
    rows: CompactRows
    row_numbers: memoryview

    @property
    def size_bytes(self):
        return len(self.rows.payload) + self.rows.offsets.nbytes + self.row_numbers.nbytes + 1024 + sum(len(name) * 4 + 64 for name in self.names)

    def __iter__(self):
        yield self.names
        yield self.rows


@dataclass(frozen=True)
class PartitionRecords:
    partition: RawPartition

    def __len__(self):
        return len(self.partition.rows)

    def __getitem__(self, index):
        return self.partition.row_numbers[index], self.partition.rows[index]


@dataclass(frozen=True)
class TemporalIndex:
    """Independent source-time and availability indexes over immutable records."""
    names: tuple[str, ...]
    records: PartitionRecords
    raw_event_keys: tuple
    availability_keys: tuple
    unknown_time_records: tuple
    event_times: tuple

    def before(self, cutoff, inclusive=False):
        event_bound = bisect_right(self.raw_event_keys, (cutoff, float("inf"))) if inclusive else bisect_left(self.raw_event_keys, (cutoff, -1))
        available_bound = bisect_right(self.availability_keys, (cutoff, float("inf"), float("inf"))) if inclusive else bisect_left(self.availability_keys, (cutoff, 25, -1))
        available = {index for _, _, index in self.availability_keys[:available_bound]}
        indexes = [index for _, index in self.raw_event_keys[:event_bound] if index in available]
        return (self.records[index] for index in (*indexes, *self.unknown_time_records))

    def window(self, start, cutoff, inclusive=False):
        """Read arriving records and their same-source-time conflict groups.

        A previously known row can still invalidate an arriving quote with the
        same event timestamp. Expand those groups before resolving ambiguity;
        filtering raw rows by availability alone could create a false fill.
        """
        lower = bisect_left(self.availability_keys, (start, 25, -1))
        upper = (bisect_right(self.availability_keys, (cutoff, float("inf"), float("inf"))) if inclusive
                 else bisect_left(self.availability_keys, (cutoff, 25, -1)))
        stamps = {self.event_times[index] for _, _, index in self.availability_keys[lower:upper]}
        indexes = []
        for stamp in sorted(stamps):
            if stamp > cutoff or (stamp == cutoff and not inclusive):
                continue
            left = bisect_left(self.raw_event_keys, (stamp, -1))
            right = bisect_right(self.raw_event_keys, (stamp, float("inf")))
            indexes.extend(index for _, index in self.raw_event_keys[left:right])
        # The quote parser enforces the cutoff again, including availability of
        # any extra conflict-group rows. Malformed timestamps remain diagnosed.
        return (self.records[index] for index in (*indexes, *self.unknown_time_records))


@dataclass(frozen=True)
class IndexedPartition:
    """One cache entry owns both raw bytes and the time index referencing them."""
    partition: RawPartition
    mapping_identity: str
    index: TemporalIndex


class ThetaReader:
    def __init__(self, calendar: CalendarService, cache: ByteLRU | None = None):
        self.calendar, self.cache = calendar, cache or ByteLRU()

    def raw_partition(self, partition: Partition):
        key = self._partition_cache_key(partition)
        cached = self.cache.get(key)
        if cached is not None:
            return cached.partition if isinstance(cached, IndexedPartition) else cached
        try:
            with gzip.open(partition.path, "rt", encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                names = tuple(next(reader))
                payload = bytearray()
                offsets = array("Q", [0])
                row_numbers = array("Q")
                overhead = 1024 + sum(len(name) * 4 + 64 for name in names)
                for row in reader:
                    number = reader.line_num
                    if not row or all(not value.strip() for value in row):
                        continue
                    if len(row) != len(names):
                        raise DataError("MALFORMED_CSV_ROW", f"Expected {len(names)} fields, got {len(row)}", path=partition.path, row_number=number)
                    # JSON encodes only strings, retaining leading zeroes,
                    # embedded newlines, NULs and non-ASCII CSV values exactly.
                    payload.extend(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                    offsets.append(len(payload))
                    row_numbers.append(number)
                    estimated = len(payload) + len(offsets) * offsets.itemsize + len(row_numbers) * row_numbers.itemsize + overhead
                    if estimated > self.cache.max_bytes:
                        raise DataError("PARTITION_EXCEEDS_MEMORY_BUDGET", "Use bounded streaming access for this partition", path=partition.path)
                result = RawPartition(names, CompactRows(bytes(payload), memoryview(offsets.tobytes()).cast("Q")),
                                      memoryview(row_numbers.tobytes()).cast("Q"))
                self.cache.put(key, result, result.size_bytes)
                return result
        except (OSError, EOFError, StopIteration, csv.Error) as exc:
            raise DataError("INVALID_OR_TRUNCATED_PARTITION", str(exc), path=partition.path) from exc

    @staticmethod
    def _partition_cache_key(partition):
        return (partition.path, partition.size, partition.mtime_ns, "theta-compact-v1")

    def header(self, partition):
        try:
            with gzip.open(partition.path, "rt", encoding="utf-8-sig", newline="") as stream:
                return tuple(next(csv.reader(stream)))
        except (OSError, EOFError, StopIteration, csv.Error) as exc:
            raise DataError("INVALID_OR_TRUNCATED_PARTITION", str(exc), path=partition.path) from exc

    def iter_records(self, partition):
        """Small partitions use immutable LRU; large ones have bounded row streaming."""
        raw = None
        if partition.size * 8 <= self.cache.max_bytes:
            try:
                raw = self.raw_partition(partition)
            except DataError as exc:
                if exc.code != "PARTITION_EXCEEDS_MEMORY_BUDGET":
                    raise
        if raw is not None:
            return raw.names, zip(raw.row_numbers, raw.rows)
        return self._stream_records(partition)

    def _stream_records(self, partition):
        names = self.header(partition)
        def records():
            try:
                with gzip.open(partition.path, "rt", encoding="utf-8-sig", newline="") as stream:
                    reader = csv.reader(stream)
                    next(reader)
                    for row in reader:
                        if not row or all(not value.strip() for value in row):
                            continue
                        if len(row) != len(names):
                            raise DataError("MALFORMED_CSV_ROW", "Inconsistent field count", path=partition.path, row_number=reader.line_num)
                        yield reader.line_num, tuple(row)
            except (OSError, EOFError, csv.Error) as exc:
                raise DataError("INVALID_OR_TRUNCATED_PARTITION", str(exc), path=partition.path) from exc
        return names, records()

    def visible_records(self, partition, cutoff, mapping, inclusive=False, interval_start=None):
        """Binary-search private time indexes, with streaming for large partitions."""
        if partition.size * 8 > self.cache.max_bytes:
            return self.iter_records(partition)
        mapping_identity = stable_hash(mapping)
        key = self._partition_cache_key(partition)
        cached = self.cache.get(key)
        index = cached.index if isinstance(cached, IndexedPartition) and cached.mapping_identity == mapping_identity else None
        if index is None:
            try:
                raw_partition = self.raw_partition(partition)
            except DataError as exc:
                if exc.code != "PARTITION_EXCEEDS_MEMORY_BUDGET":
                    raise
                return self._stream_records(partition)
            names, records = raw_partition.names, PartitionRecords(raw_partition)
            if raw_partition.size_bytes + len(records) * 320 > self.cache.max_bytes:
                return names, iter(records)
            schema = canonical_schema(names)
            raw, available, unknown, event_times = [], [], [], []
            for offset, (_, values) in enumerate(records):
                row = dict(zip(names, values))
                stamp = None
                try:
                    stamp = row_timestamp(row, schema, partition.key.session_date, mapping)
                    known, phase = observation_availability(stamp,row,partition.key.session_date,mapping)
                    raw.append((stamp, offset))
                    available.append((known, phase, offset))
                except (ValueError, TypeError):
                    unknown.append(offset)
                event_times.append(stamp)
            index = TemporalIndex(names, records, tuple(sorted(raw)), tuple(sorted(available)), tuple(unknown), tuple(event_times))
            # The index retains the raw rows. Account that storage in this same
            # entry so eviction cannot leave an uncounted full-session copy.
            self.cache.put(key, IndexedPartition(raw_partition, mapping_identity, index), raw_partition.size_bytes + len(records) * 320)
        return index.names, (index.before(cutoff, inclusive) if interval_start is None
                             else index.window(interval_start, cutoff, inclusive))

    def _contract(self, row, schema, partition, spot, mapping, row_number=None):
        symbol = row.get("symbol", row.get("underlying_symbol", "QQQ")).strip().upper()
        root = row.get("root", "QQQ").strip().upper()
        if symbol != "QQQ" or root != "QQQ":
            raise DataError("CONTRACT_SYMBOL_MISMATCH", f"{symbol}/{root}")
        expiration = field(row, "expiration", schema)
        if expiration is not None:
            expiration_text = str(expiration)
            parsed_expiry = datetime.strptime(expiration_text, "%Y%m%d").date() if len(expiration_text) == 8 and expiration_text.isdigit() else date.fromisoformat(expiration_text[:10])
            if parsed_expiry != partition.key.expiration:
                raise DataError("EXPIRY_IDENTITY_MISMATCH", f"Row {parsed_expiry} differs from partition expiry",
                                path=partition.path, row_number=row_number)
        strike, divisor = resolve_strike(field(row, "strike", schema), spot, mapping, schema.get("strike", "strike"))
        return ContractKey(partition.key.expiration, strike, normalize_right(field(row, "right", schema)), metadata_verified=mapping.metadata_verified), divisor

    def quotes(self, partition: Partition, cutoff: datetime, spot: float | None, mapping: NormalizationMapping | None = None, inclusive: bool = False, spot_at=None, interval_start=None):
        """Read causal quotes, optionally restricted to one replay interval.

        Interval diagnostics concern the selected source-time groups. Full
        snapshots continue to validate the entire visible quote history.
        """
        mapping = mapping or NormalizationMapping()
        names, records = self.visible_records(partition, cutoff, mapping, inclusive, interval_start)
        schema = canonical_schema(names)
        missing = [key for key in ("strike", "right", "bid", "ask") if key not in schema]
        if missing:
            raise DataError("REQUIRED_FIELD_MISSING", "Theta quote schema invalid", missing=missing)
        session = self.calendar.session(partition.key.session_date)
        output, diagnostics, scales = [], [], set()
        mapping_identity = stable_hash(mapping)
        for number, values in records:
            row = dict(zip(names, values))
            try:
                stamp = row_timestamp(row, schema, partition.key.session_date, mapping)
                known_at, phase = observation_availability(stamp,row,partition.key.session_date,mapping)
                if not visible_at(stamp,known_at,phase,cutoff,inclusive):
                    continue
                if known_at < stamp:
                    raise DataError("INVALID_QUOTE_AVAILABILITY", "Quote cannot be known before its source event")
                if session.status != "OPEN":
                    raise DataError("CALENDAR_SOURCE_CONFLICT", "Visible quote exists on closed date")
                if not session.exchange_open <= stamp < session.underlying_close:
                    continue
                causal_spot = spot_at(stamp) if spot_at is not None else spot
                contract, divisor = self._contract(row, schema, partition, causal_spot, mapping, number)
                scales.add(divisor)
                quote_cache_key = ("canonical-quote-v1", partition.path, partition.size, partition.mtime_ns, number, mapping_identity, contract)
                cached_quote = self.cache.get(quote_cache_key)
                if cached_quote is not None:
                    output.append(cached_quote)
                    continue
                bid, ask = decimal_value(field(row, "bid", schema), nullable=True), decimal_value(field(row, "ask", schema), nullable=True)
                valid = bid is not None and ask is not None and bid >= 0 and ask > 0 and ask >= bid
                iv, iv_flags = resolve_iv(field(row, "implied_vol", schema), mapping)
                flags = list(iv_flags)
                if mapping.timestamp_precision == "minute" and (stamp.second or stamp.microsecond):
                    flags.append("TIMESTAMP_PRECISION_CONFLICT")
                    valid = False
                if not valid:
                    flags.append("INVALID_QUOTE")
                elif bid == 0:
                    flags.append("ZERO_BID")
                if not mapping.metadata_verified:
                    flags.append("CONTRACT_METADATA_UNVERIFIED")
                sequence = field(row, "sequence", schema)
                sequence = int(sequence) if sequence is not None else None
                greek_values = []
                for key in GREEKS:
                    if row.get(key) in (None, ""):
                        continue
                    try:
                        value = float(row[key])
                        if not math.isfinite(value):
                            raise ValueError()
                        greek_values.append((key, value))
                    except (ValueError, TypeError):
                        flags.append("VENDOR_GREEK_INVALID:" + key)
                greeks = tuple(greek_values)
                normalized_greeks, normalized_greek_units, greek_flags = normalize_vendor_greeks(greeks, mapping, contract.multiplier)
                flags.extend(greek_flags)
                raw = tuple(sorted(row.items()))
                source_id = stable_hash((Path(partition.path).name, raw))
                quote = OptionQuote(contract, stamp, known_at, bid, ask, iv, valid, source_id, sequence, mapping.timestamp_precision, tuple(flags), raw, greeks,
                                    int(row["bid_size"]) if row.get("bid_size") else None, int(row["ask_size"]) if row.get("ask_size") else None,
                                    number, partition.path, normalized_greeks, normalized_greek_units)
                output.append(quote)
                estimated = 1024 + sum(len(name) + len(value) + 128 for name, value in raw)
                self.cache.put(quote_cache_key, quote, estimated, transient=True)
            except (ValueError, TypeError) as exc:
                code = getattr(exc, "code", "INVALID_OPTION_ROW")
                diagnostics.append(Diagnostic(code, str(exc), partition.path, number))
        if len(scales) > 1:
            raise DataError("INCONSISTENT_STRIKE_SCALE", "Visible rows require conflicting unit mappings")
        grouped = defaultdict(list)
        for quote in output:
            grouped[(quote.contract, quote.event_at)].append(quote)
        resolved = []
        for key, group in grouped.items():
            unique = {item.source_id: item for item in group}
            values = list(unique.values())
            if len(values) > 1 and (any(item.sequence is None for item in values) or len({item.sequence for item in values}) != len(values)):
                diagnostics.append(Diagnostic("AMBIGUOUS_EVENT_ORDER", f"Conflicting same-time events for {key[0].contract_id}", partition.path))
                # No arbitrary winner can become an execution opportunity.
                from dataclasses import replace
                representative = min(values, key=lambda item: item.source_id)
                resolved.append(replace(representative, bid=None, ask=None, valid=False, iv=None, quality_flags=("AMBIGUOUS_EVENT_ORDER",)))
            else:
                resolved.extend(values)
        if interval_start is not None:
            resolved = [quote for quote in resolved if quote.available_at > interval_start or
                        quote.available_at == interval_start and not (
                            quote.timestamp_precision == "minute" and
                            quote.available_at == quote.event_at.replace(second=0, microsecond=0)+timedelta(minutes=1))]
        return tuple(sorted(resolved, key=lambda item: (item.event_at, item.sequence if item.sequence is not None else -1, item.source_id))), tuple(diagnostics)

    def daily_oi(self, partition: Partition, cutoff: datetime, spot, mapping: NormalizationMapping | None = None, post_session: bool = False):
        mapping = mapping or NormalizationMapping()
        names, records = self.iter_records(partition)
        schema = canonical_schema(names)
        if any(key not in schema for key in ("strike", "right", "open_interest")):
            raise DataError("REQUIRED_FIELD_MISSING", "OI needs strike, right and open_interest")
        session = self.calendar.session(partition.key.session_date)
        if partition.key.data_type == "oi" and not post_session:
            return {}
        result = {}
        for number, values in records:
            row = dict(zip(names, values))
            # Daily OI timestamp is a vendor reference, not publication evidence.
            known = parse_timestamp(row["known_at"], partition.key.session_date) if row.get("known_at") else session.exchange_open if partition.key.data_type == "oi_before" else None
            if known is None or known > cutoff:
                continue
            contract, _ = self._contract(row, schema, partition, spot, mapping, number)
            value = Decimal(str(field(row, "open_interest", schema)))
            if not value.is_finite() or value < 0 or value != value.to_integral_value():
                raise DataError("INVALID_OPEN_INTEREST", str(value))
            if contract in result and result[contract] != int(value):
                raise DataError("CONFLICTING_OI", contract.contract_id)
            result[contract] = int(value)
        return result

    def volumes(self, partition: Partition, cutoff: datetime, spot, mapping: NormalizationMapping | None = None):
        mapping = mapping or NormalizationMapping()
        if mapping.bar_label_semantics not in ("START", "END"):
            raise DataError("TEMPORAL_SEMANTICS_UNVERIFIED", mapping.bar_label_semantics)
        names, records = self.iter_records(partition)
        schema = canonical_schema(names)
        result = defaultdict(float)
        seen = set()
        session = self.calendar.session(partition.key.session_date)
        for number, values in records:
            row = dict(zip(names, values))
            stamp = row_timestamp(row, schema, partition.key.session_date, mapping)
            available = stamp if mapping.bar_label_semantics == "END" else stamp + timedelta(minutes=1)
            if available > cutoff:
                continue
            if stamp.second != 0 or stamp.microsecond != 0:
                raise DataError("INVALID_MINUTE_LABEL", "Option OHLCV label is not a minute boundary")
            label = stamp - timedelta(minutes=1) if mapping.bar_label_semantics == "END" else stamp
            if not session.exchange_open <= label < session.underlying_close:
                continue
            if row.get("known_at") and parse_timestamp(row["known_at"], partition.key.session_date) > cutoff:
                continue
            contract, _ = self._contract(row, schema, partition, spot, mapping, number)
            identity = (contract, stamp)
            if identity in seen:
                raise DataError("DUPLICATE_OPTION_BAR", str(identity))
            seen.add(identity)
            volume = float(row["volume"])
            prices = tuple(float(row[name]) for name in ("open", "high", "low", "close"))
            opn, high, low, close = prices
            if not all(math.isfinite(x) for x in (*prices, volume)) or volume < 0 or high < max(opn, close) or low > min(opn, close) or high < low:
                raise DataError("INVALID_OPTION_BAR", str(identity))
            result[contract] += volume
        return dict(result)

    def trades(self, partition: Partition, cutoff: datetime, spot, mapping: NormalizationMapping | None = None,
               allowed_conditions=frozenset({"REGULAR"}), inclusive=False, interval_start=None):
        """Retain unknown conditions diagnostically; corrections apply when known."""
        mapping = mapping or NormalizationMapping()
        names, records = self.visible_records(partition,cutoff,mapping,inclusive,interval_start)
        schema = canonical_schema(names)
        missing = [key for key in ("strike", "right", "last") if key not in schema]
        if missing or "size" not in names:
            raise DataError("REQUIRED_FIELD_MISSING", "Trade schema needs strike/right/price/size", missing=missing)
        session = self.calendar.session(partition.key.session_date)
        result = []
        mapping_identity=stable_hash(mapping)
        condition_identity=tuple(sorted(allowed_conditions))
        for number, values in records:
            row = dict(zip(names, values))
            stamp = row_timestamp(row, schema, partition.key.session_date, mapping)
            known,phase=observation_availability(stamp,row,partition.key.session_date,mapping)
            if not visible_at(stamp,known,phase,cutoff,inclusive) or not session.exchange_open <= stamp < session.underlying_close:
                continue
            if interval_start is not None and (known < interval_start or known == interval_start and phase < 25):
                continue
            contract, _ = self._contract(row, schema, partition, spot, mapping, number)
            cache_key=("canonical-trade-v2",partition.path,partition.size,partition.mtime_ns,number,mapping_identity,contract,condition_identity)
            cached=self.cache.get(cache_key)
            if cached is not None:
                result.append(cached)
                continue
            price = decimal_value(field(row, "last", schema))
            size = decimal_value(row["size"])
            if price < 0 or size < 0 or size != size.to_integral_value():
                raise DataError("INVALID_TRADE", "Price/size must be nonnegative; size integral")
            sequence = field(row, "sequence", schema)
            sequence = int(sequence) if sequence is not None else None
            condition = str(row.get("condition", row.get("conditions", "UNKNOWN")))
            action = row.get("action", "TRADE").strip().upper()
            if action not in ("TRADE", "CANCEL", "CORRECT"):
                raise DataError("UNKNOWN_TRADE_ACTION", action)
            source_id = stable_hash((Path(partition.path).name, row))
            trade_id = str(row.get("trade_id", source_id))
            correction_of = row.get("correction_of") or row.get("original_trade_id") or None
            if action in ("CANCEL", "CORRECT") and not correction_of:
                raise DataError("TRADE_CORRECTION_TARGET_MISSING", trade_id)
            flags=[] if condition in allowed_conditions else ["UNKNOWN_TRADE_CONDITION"]
            if mapping.timestamp_precision=="minute" and (stamp.second or stamp.microsecond):
                flags.append("TIMESTAMP_PRECISION_CONFLICT")
            eligible = not flags
            observation=TradeObservation(contract,stamp,known,price,int(size),trade_id,sequence,condition,eligible,action,correction_of,source_id,tuple(flags))
            result.append(observation)
            self.cache.put(cache_key,observation,768+sum(len(name)+len(value) for name,value in row.items()), transient=True)
        return tuple(sorted(result, key=lambda item: (item.known_at, item.event_at, item.sequence if item.sequence is not None else -1, item.source_id)))


def active_trades(observations):
    """Prospective as-of correction ledger; no double counting trade IDs."""
    active = {}
    seen = set()
    for item in observations:
        if item.source_id in seen:
            continue
        seen.add(item.source_id)
        if item.action in ("CANCEL", "CORRECT"):
            if item.correction_of not in active:
                raise DataError("TRADE_CORRECTION_TARGET_UNKNOWN", str(item.correction_of))
            active.pop(item.correction_of)
        if item.action != "CANCEL":
            if item.trade_id in active and active[item.trade_id] != item:
                raise DataError("CONFLICTING_TRADE_ID", item.trade_id)
            active[item.trade_id] = item
    return tuple(active.values())


def trade_derived_volume(observations):
    """Only recognized execution conditions contribute derived volume."""
    totals = defaultdict(int)
    for item in active_trades(observations):
        if item.execution_eligible:
            totals[item.contract] += item.size
    return dict(totals)
