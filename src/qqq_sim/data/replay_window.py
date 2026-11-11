"""Bounded replay indexes; unusual histories retain the causal reference reader."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from itertools import groupby

from .models import DataError, SourceEvent, stable_hash
from .normalization import canonical_schema, decimal_value, field, normalize_vendor_greeks, row_timestamp
from .theta import IndexedPartition, observation_availability
from qqq_sim.replay.calendar import local_datetime


@dataclass(frozen=True)
class QuoteReplayIndex:
    """Only offsets and contract/time metadata; no normalized quote history."""
    by_contract: dict
    size_bytes: int
    partition: object
    valid_by_contract: dict
    rejected_offsets: tuple
    unreplayable_offsets: frozenset
    spot_bounds: object


@dataclass(frozen=True)
class TradeReplayIndex:
    by_contract: dict
    partition: object
    spot_bounds: object
    size_bytes: int


def _contract_identity(contract):
    if isinstance(contract, str):
        if contract.startswith("LOCAL:"):
            parts = contract.split(":")
            return date.fromisoformat(parts[2]), Decimal(parts[4]), parts[3]
        parts = contract.split("|")
        if len(parts) >= 5:
            return date.fromisoformat(parts[2]), Decimal(parts[4]), parts[3]
        raise DataError("INVALID_CONTRACT_ID", contract)
    expiry = getattr(contract, "expiration", None) or getattr(contract, "expiry", None)
    if expiry is None:
        raise DataError("INVALID_CONTRACT_ID", str(contract))
    return expiry, Decimal(str(contract.strike)), contract.right


def _selectors(contracts):
    return None if contracts is None else {_contract_identity(contract) for contract in contracts}


def _check_cancel(cancel):
    if cancel is not None and cancel():
        raise InterruptedError("Replay canceled during data preparation")


def _index_key(reader, partition, mapping, spot_bounds=None):
    return ("replay-quote-offsets-v1", partition.path, partition.size, partition.mtime_ns,
            stable_hash(mapping), reader.calendar.session(partition.key.session_date),
            spot_bounds if mapping.strike_divisor is None else None)


def _prepare_index(reader, partition, mapping, cancel=None, spot_at=None, spot_bounds=None):
    key = _index_key(reader, partition, mapping, spot_bounds)
    source_key = reader._partition_cache_key(partition)
    retained = reader.cache.get(source_key)
    if isinstance(retained, IndexedPartition) and retained.replay_identity == key:
        return retained.replay_index
    if reader.cache.get(key) is False:
        return None
    names = reader.header(partition)
    schema = canonical_schema(names)
    explicit_units = mapping.strike_divisor is not None or schema.get("strike", "").lower() in ("strike_milli", "strike_millis")
    if (not explicit_units and spot_bounds is None
            or any(name not in schema for name in ("strike", "right", "bid", "ask"))):
        reader.cache.put(key, False, 64)
        return None
    try:
        raw = reader.raw_partition(partition)
        replay_size = len(raw.rows) * 176 + 2048
        temporal = retained.index if isinstance(retained, IndexedPartition) and retained.mapping_identity == stable_hash(mapping) else None
        estimated = raw.size_bytes + replay_size + (len(raw.rows) * 320 if temporal is not None else 0)
        if estimated > reader.cache.max_bytes:
            reader.cache.put(key, False, 64)
            return None
        normalize_vendor_greeks((), mapping, 100)
        session = reader.calendar.session(partition.key.session_date)
        by_contract, contracts = defaultdict(list), {}
        sequences, scales, valid_offsets, rejected, unreplayable = {}, set(), set(), [], set()
        for offset, values in enumerate(raw.rows):
            if offset % 256 == 0:
                _check_cancel(cancel)
            row = dict(zip(names, values))
            try:
                stamp = row_timestamp(row, schema, partition.key.session_date, mapping)
                known, phase = observation_availability(stamp, row, partition.key.session_date, mapping)
            except (ValueError, TypeError):
                # These rows are diagnostics, never economic quote events.
                rejected.append((None, offset))
                continue
            expected = stamp.replace(second=0, microsecond=0) + timedelta(minutes=1) if mapping.timestamp_precision == "minute" else stamp
            if known != expected or mapping.timestamp_precision == "minute" and (stamp.second or stamp.microsecond):
                reader.cache.put(key, False, 64)
                return None
            if not session.exchange_open <= stamp < session.underlying_close:
                continue
            identity = (row.get("symbol", row.get("underlying_symbol", "QQQ")), row.get("root", "QQQ"),
                        field(row, "expiration", schema), field(row, "strike", schema), field(row, "right", schema))
            if identity not in contracts:
                try:
                    contract, divisor = reader._contract(row, schema, partition, None if explicit_units else spot_bounds[0], mapping, raw.row_numbers[offset])
                    if not explicit_units:
                        other, other_divisor = reader._contract(row, schema, partition, spot_bounds[1], mapping, raw.row_numbers[offset])
                        if other != contract or other_divisor != divisor:
                            reader.cache.put(key, False, 64)
                            return None
                    contracts[identity] = (contract, divisor)
                except DataError as error:
                    if error.code in ("AMBIGUOUS_STRIKE_SCALE", "STRIKE_SCALE_REQUIRES_KNOWN_SPOT"):
                        reader.cache.put(key, False, 64)
                        return None
                    # Invalid contract identities are excluded by the ordinary
                    # row parser. Their diagnostics remain in final snapshots.
                    contracts[identity] = None
            normalized = contracts[identity]
            if normalized is None:
                rejected.append((stamp, offset))
                continue
            contract, divisor = normalized
            if not explicit_units and spot_at(stamp) is None:
                unreplayable.add(offset)
            scales.add(divisor)
            if len(scales) > 1:
                reader.cache.put(key, False, 64)
                return None
            try:
                sequence = field(row, "sequence", schema)
                sequence = int(sequence) if sequence is not None else None
                for name in ("bid_size", "ask_size"):
                    if row.get(name):
                        int(row[name])
            except (ValueError, TypeError):
                rejected.append((stamp, offset))
                continue
            bid = decimal_value(field(row, "bid", schema), nullable=True)
            ask = decimal_value(field(row, "ask", schema), nullable=True)
            if bid is not None and ask is not None and bid >= 0 and ask > 0 and ask >= bid:
                valid_offsets.add(offset)
            sequences[offset] = sequence
            by_contract[contract].append((known, phase, sequence or 0, offset))
        result = {}
        for contract, records in by_contract.items():
            records.sort()
            unique = []
            for _, grouped in groupby(records, key=lambda record: record[0]):
                group = list(grouped)
                seen = {}
                for record in group:
                    sequence = sequences[record[3]]
                    if sequence in seen:
                        if raw.rows[seen[sequence][3]] == raw.rows[record[3]]:
                            continue
                        reader.cache.put(key, False, 64)
                        return None
                    seen[sequence] = record
                if len(seen) > 1 and None in seen:
                    reader.cache.put(key, False, 64)
                    return None
                unique.extend(seen.values())
            result[contract] = tuple(sorted(unique))
        valid = {contract: tuple(record for record in records if record[3] in valid_offsets)
                 for contract, records in result.items()}
        rejected.sort(key=lambda item: (item[0] is None, item[0], item[1]))
        index = QuoteReplayIndex(result, replay_size, raw, valid, tuple(offset for _, offset in rejected),
                                 frozenset(unreplayable), spot_bounds if not explicit_units else None)
        reader.cache.put(source_key, IndexedPartition(raw, stable_hash(mapping), temporal, index, key, replay_size), estimated)
        return index
    except (DataError, ValueError, TypeError, OverflowError):
        # Preparation must not surface a future bad row before replay reaches
        # its original failure slot. The reference reader remains authoritative.
        reader.cache.put(key, False, 64)
        return None


def indexed_chain_quotes(reader, partition, cutoff, spot, mapping, inclusive=False):
    """Use a prepared clean index without expanding its full quote history.

    Rejected raw rows still run through the canonical parser in original
    temporal order, so diagnostic content and snapshot identities are unchanged.
    """
    cached = reader.cache.get(reader._partition_cache_key(partition))
    if not isinstance(cached, IndexedPartition) or cached.mapping_identity != stable_hash(mapping):
        return None
    index = cached.replay_index
    if not isinstance(index, QuoteReplayIndex):
        return None
    if index.spot_bounds is not None and (spot is None or not index.spot_bounds[0] <= spot <= index.spot_bounds[1]):
        return None
    bound = ((cutoff, float("inf"), float("inf"), float("inf")) if inclusive
             else (cutoff, 25, -float("inf"), -1))
    selected = set()
    for contract, records in index.by_contract.items():
        upper = bisect_right(records, bound) if inclusive else bisect_left(records, bound)
        if upper:
            selected.add(records[upper - 1][3])
        valid = index.valid_by_contract[contract]
        upper = bisect_right(valid, bound) if inclusive else bisect_left(valid, bound)
        if upper:
            selected.add(valid[upper - 1][3])
    quotes, diagnostics = reader.quotes(partition, cutoff, spot, mapping, inclusive,
                                       _row_offsets=(*index.rejected_offsets, *sorted(selected)),
                                       _raw_partition=index.partition)
    latest, valid = {}, {}
    for quote in quotes:
        latest[quote.contract] = quote
        if quote.valid:
            valid[quote.contract] = quote
    return {contract: (quote, valid.get(contract)) for contract, quote in latest.items()}, diagnostics


def _prepare_trade_index(reader, partition, mapping, spot_bounds, cancel=None):
    key = ("replay-trade-groups-v1", partition.path, partition.size, partition.mtime_ns,
           stable_hash(mapping), reader.calendar.session(partition.key.session_date), spot_bounds)
    source_key = reader._partition_cache_key(partition)
    retained = reader.cache.get(source_key)
    if isinstance(retained, IndexedPartition) and retained.trade_identity == key:
        return retained.trade_index
    if reader.cache.get(key) is False:
        return None
    try:
        raw = reader.raw_partition(partition)
        schema = canonical_schema(raw.names)
        if any(name not in schema for name in ("strike", "right", "last")) or "size" not in raw.names:
            reader.cache.put(key, False, 64)
            return None
        explicit = mapping.strike_divisor is not None or schema.get("strike", "").lower() in ("strike_milli", "strike_millis")
        if not explicit and spot_bounds is None:
            reader.cache.put(key, False, 64)
            return None
        size_bytes = len(raw.rows) * 176 + 2048
        if raw.size_bytes + size_bytes > reader.cache.max_bytes:
            reader.cache.put(key, False, 64)
            return None
        temporal = retained.index if isinstance(retained, IndexedPartition) and retained.mapping_identity == stable_hash(mapping) else None
        availability = None
        if temporal is not None:
            if temporal.unknown_time_records:
                reader.cache.put(key, False, 64)
                return None
            availability = [None] * len(raw.rows)
            for known, phase, offset in temporal.availability_keys:
                availability[offset] = known, phase
        session = reader.calendar.session(partition.key.session_date)
        contracts, by_contract, trade_ids = {}, defaultdict(list), set()
        for offset, values in enumerate(raw.rows):
            if offset % 256 == 0:
                _check_cancel(cancel)
            row = dict(zip(raw.names, values))
            if temporal is not None:
                stamp = temporal.event_times[offset]
                known, phase = availability[offset]
            else:
                stamp = row_timestamp(row, schema, partition.key.session_date, mapping)
                known, phase = observation_availability(stamp, row, partition.key.session_date, mapping)
            expected = stamp.replace(second=0, microsecond=0) + timedelta(minutes=1) if mapping.timestamp_precision == "minute" else stamp
            if known != expected:
                reader.cache.put(key, False, 64)
                return None
            if not session.exchange_open <= stamp < session.underlying_close:
                continue
            identity = (row.get("symbol", row.get("underlying_symbol", "QQQ")), row.get("root", "QQQ"),
                        field(row, "expiration", schema), field(row, "strike", schema), field(row, "right", schema))
            if identity not in contracts:
                contract, divisor = reader._contract(row, schema, partition, None if explicit else spot_bounds[0], mapping, raw.row_numbers[offset])
                if not explicit:
                    other, other_divisor = reader._contract(row, schema, partition, spot_bounds[1], mapping, raw.row_numbers[offset])
                    if other != contract or other_divisor != divisor:
                        reader.cache.put(key, False, 64)
                        return None
                contracts[identity] = contract
            contract = contracts[identity]
            price, quantity = decimal_value(field(row, "last", schema)), decimal_value(row["size"])
            if price < 0 or quantity < 0 or quantity != quantity.to_integral_value():
                reader.cache.put(key, False, 64)
                return None
            sequence = field(row, "sequence", schema)
            sequence = int(sequence) if sequence is not None else None
            if row.get("action", "TRADE").strip().upper() != "TRADE":
                reader.cache.put(key, False, 64)
                return None
            if "trade_id" in row:
                if row["trade_id"] in trade_ids:
                    reader.cache.put(key, False, 64)
                    return None
                trade_ids.add(row["trade_id"])
            if quantity:
                by_contract[contract].append((known, phase, stamp, sequence if sequence is not None else -1, offset))
        result = {contract: tuple(sorted(records)) for contract, records in by_contract.items()}
        index = TradeReplayIndex(result, raw, None if explicit else spot_bounds, size_bytes)
        # Clean terminal groups replace the larger general temporal index. The
        # shared entry continues to own its raw partition exactly once.
        reader.cache.put(source_key, IndexedPartition(raw, stable_hash(mapping), None, trade_index=index,
                                                      trade_identity=key, trade_size=size_bytes), raw.size_bytes + size_bytes)
        return index
    except (DataError, ValueError, TypeError, ArithmeticError, OverflowError):
        reader.cache.put(key, False, 64)
        return None


def indexed_trade_observations(reader, partition, cutoff, spot, mapping, inclusive=False):
    cached = reader.cache.get(reader._partition_cache_key(partition))
    if not isinstance(cached, IndexedPartition) or cached.mapping_identity != stable_hash(mapping):
        return None
    index = cached.trade_index
    if not isinstance(index, TradeReplayIndex):
        return None
    if index.spot_bounds is not None and (spot is None or not index.spot_bounds[0] <= spot <= index.spot_bounds[1]):
        return None
    bound = ((cutoff, float("inf")) if inclusive else (cutoff, 25))
    offsets = []
    for records in index.by_contract.values():
        upper = bisect_right(records, (*bound, cutoff, float("inf"), float("inf"))) if inclusive else bisect_left(records, (*bound, cutoff, -float("inf"), -1))
        if not upper:
            continue
        stamp = records[upper - 1][2]
        lower = upper - 1
        while lower and records[lower - 1][2] == stamp:
            lower -= 1
        offsets.extend(record[4] for record in records[lower:upper])
    return reader.trades(partition, cutoff, spot, mapping, inclusive=inclusive,
                         _row_offsets=offsets, _raw_partition=index.partition)


def _auxiliary_snapshot_safe(reader, partition, kind, cancel=None):
    key = ("snapshot-row-types-v1", partition.path, partition.size, partition.mtime_ns, kind)
    cached = reader.cache.get(key)
    if cached is not None:
        return cached
    try:
        names, records = reader.iter_records(partition)
        schema = canonical_schema(names)
        for offset, (_, values) in enumerate(records):
            if offset % 256 == 0:
                _check_cancel(cancel)
            row = dict(zip(names, values))
            if kind == "ohlcv":
                tuple(float(row[name]) for name in ("open", "high", "low", "close", "volume"))
            else:
                Decimal(str(field(row, "open_interest", schema)))
        safe = True
    except DataError:
        # The chain reader reports typed source problems as diagnostics.
        safe = True
    except (ValueError, TypeError, ArithmeticError, KeyError):
        safe = False
    reader.cache.put(key, safe, 64)
    return safe


class ReplayWindow:
    def __init__(self, data, start, end, cancel=None):
        self.data, self.start, self.end, self.cancel = data, local_datetime(start), local_datetime(end), cancel
        self.fast_quotes = True
        self.requires_snapshot_validation = False
        self._spot_contexts = {}
        day = self.start.date()
        while day <= self.end.date():
            _check_cancel(cancel)
            session = data.calendar.session(day)
            if session.status == "OPEN":
                spot_at, spot_bounds = self._spot_context(day)
                for expiry in data.catalog.expirations(day):
                    partition = data.catalog.get("greeks", day, expiry)
                    if _prepare_index(data.theta_reader, partition, data.mapping("greeks"), cancel, spot_at, spot_bounds) is None:
                        self.fast_quotes = False
                    try:
                        trade_partition = data.catalog.get("trades", day, expiry)
                        if _prepare_trade_index(data.theta_reader, trade_partition, data.mapping("trades"), spot_bounds, cancel) is None:
                            self.fast_quotes = False
                            self.requires_snapshot_validation = True
                    except DataError:
                        pass
                    for kind in ("oi_before", "ohlcv"):
                        try:
                            auxiliary = data.catalog.get(kind, day, expiry)
                            if not _auxiliary_snapshot_safe(data.theta_reader, auxiliary, kind, cancel):
                                self.requires_snapshot_validation = True
                        except DataError:
                            pass
            day += timedelta(days=1)

    def _spot_context(self, day):
        if day not in self._spot_contexts:
            data = self.data
            try:
                rows, _, _ = data.underlying_reader.read(data.catalog.get("underlying", day),
                                                        data.calendar.session(day).underlying_close, data.mapping("underlying"))
            except (DataError, ValueError, TypeError):
                rows = ()
            times = tuple(row.available_at for row in rows)
            def spot_at(stamp):
                offset = bisect_right(times, stamp) - 1
                return rows[offset].close if offset >= 0 else None
            bounds = (min(row.close for row in rows), max(row.close for row in rows)) if rows else None
            self._spot_contexts[day] = spot_at, bounds
        return self._spot_contexts[day]

    def coverage_gap(self, start, end, required_contracts=None):
        _check_cancel(self.cancel)
        return self.data.coverage_gap(start, end, required_contracts=required_contracts)

    def events_between(self, start, end, required_contracts=None):
        return self._events(start, end, required_contracts=required_contracts)

    def terminal_quote_events(self, start, end, excluded_contracts=()):
        excluded = _selectors(excluded_contracts)
        events = self._events(start, end, terminal_only=True, excluded=excluded)
        latest = {}
        for event in sorted(events, key=lambda item: item.key):
            if _contract_identity(event.payload.contract) not in excluded:
                latest[event.payload.contract] = event
        return tuple(sorted(latest.values(), key=lambda item: item.key))

    def _events(self, start, end, required_contracts=None, terminal_only=False, excluded=frozenset()):
        start, end = local_datetime(start), local_datetime(end)
        if end < start:
            raise DataError("REWIND_REQUIRES_BRANCH", "Event replay cannot move backwards")
        required = _selectors(required_contracts)
        data = self.data
        events, day = [], start.date()
        while day <= end.date():
            _check_cancel(self.cancel)
            session = data.calendar.session(day)
            if session.status != "OPEN":
                day += timedelta(days=1)
                continue
            cutoff = min(end, session.underlying_close)
            candles, _, _ = data.underlying_reader.read(data.catalog.get("underlying", day), cutoff, data.mapping("underlying"))
            if not terminal_only:
                events.extend(SourceEvent(candle.available_at, 20, candle.source_id, "candle", candle)
                              for candle in candles if start < candle.available_at <= end)
            spot = candles[-1].close if candles else None
            candle_times = tuple(candle.available_at for candle in candles)
            def causal_spot_at(stamp):
                offset = bisect_right(candle_times, stamp) - 1
                return candles[offset].close if offset >= 0 else None
            for expiry in data.catalog.expirations(day):
                _check_cancel(self.cancel)
                partition = data.catalog.get("greeks", day, expiry)
                mapping = data.mapping("greeks")
                indexed_spot_at, spot_bounds = self._spot_context(day)
                index = _prepare_index(data.theta_reader, partition, mapping, self.cancel, indexed_spot_at, spot_bounds)
                offsets = None
                if index is not None:
                    offsets = []
                    for contract, records in index.by_contract.items():
                        identity = _contract_identity(contract)
                        if required is not None and identity not in required or identity in excluded:
                            continue
                        lower = bisect_left(records, (start, 25, -float("inf"), -1))
                        upper = bisect_left(records, (end, 25, -float("inf"), -1))
                        chosen = records[max(lower, upper - 1):upper] if terminal_only else records[lower:upper]
                        offsets.extend(record[3] for record in chosen if record[3] not in index.unreplayable_offsets)
                    if not offsets:
                        continue
                if index is None and terminal_only:
                    # Only unusual partitions pay for reference minute windows;
                    # clean expiries retain their terminal-offset summaries.
                    quotes, current = [], start
                    while current < end:
                        _check_cancel(self.cancel)
                        following = min(end, data.calendar.advance_trading_minutes(current, 1))
                        slot_cutoff = min(following, session.underlying_close)
                        slot_spot = causal_spot_at(slot_cutoff)
                        batch, _ = data.theta_reader.quotes(partition, slot_cutoff, slot_spot, mapping,
                                                            spot_at=causal_spot_at, interval_start=current)
                        quotes.extend(batch)
                        current = following
                else:
                    quotes, _ = data.theta_reader.quotes(partition, cutoff, spot, mapping, spot_at=causal_spot_at,
                                                         interval_start=start, _row_offsets=offsets,
                                                         _raw_partition=index.partition if index is not None else None)
                for quote in quotes:
                    completed = quote.timestamp_precision == "minute" and quote.available_at == quote.event_at.replace(second=0, microsecond=0) + timedelta(minutes=1)
                    eligible = start < quote.available_at <= end if completed else start <= quote.available_at < end
                    identity = _contract_identity(quote.contract)
                    if eligible and (required is None or identity in required) and identity not in excluded:
                        events.append(SourceEvent(quote.available_at, 20 if completed else 30, quote.source_id,
                                                  "quote", quote, quote.sequence or 0))
            day += timedelta(days=1)
        return tuple(sorted(events, key=lambda item: item.key))
