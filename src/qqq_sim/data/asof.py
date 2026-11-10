"""Public causal data service. Private full-session caches never bypass cutoffs."""
from __future__ import annotations
from bisect import bisect_right
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from .cache import ByteLRU
from .catalog import CatalogService
from .models import ChainEntry, DataError, Diagnostic, MarketSnapshot, NormalizationMapping, SourceEvent, stable_hash
from .theta import ThetaReader
from .quote_state import chain_quotes
from .trade_state import latest_trades
from .underlying import UnderlyingReader
from qqq_sim.replay.calendar import CalendarService, local_datetime


class HistoricalData:
    def __init__(self, underlying_root: str | Path, options_root: str | Path,
                 mappings: dict[str, NormalizationMapping | dict] | None = None,
                 calendar: CalendarService | None = None, cache_bytes: int = 256 * 1024 * 1024, catalog_path=None):
        self.calendar = calendar or CalendarService()
        self.mappings = {key: value if isinstance(value, NormalizationMapping) else NormalizationMapping(**value) for key, value in (mappings or {}).items()}
        self.catalog = CatalogService(underlying_root, options_root, manifest_path=catalog_path, normalization_identity=stable_hash(self.mappings))
        self.cache = ByteLRU(cache_bytes)
        self.underlying_reader = UnderlyingReader(self.calendar, self.cache)
        self.theta_reader = ThetaReader(self.calendar, self.cache)

    def mapping(self, kind: str) -> NormalizationMapping:
        return self.mappings.get(kind, self.mappings.get("options", NormalizationMapping())) if kind != "underlying" else self.mappings.get(kind, NormalizationMapping())

    def underlying_asof(self, instant: datetime, visibility_profile=None):
        instant = local_datetime(instant)
        candles, _, _ = self.underlying_reader.read(self.catalog.get("underlying", instant.date()), instant, self.mapping("underlying"))
        return candles[-1] if candles else None

    def load_partition(self, partition_key, revision=None):
        item = self.catalog.get(partition_key.data_type, partition_key.session_date, partition_key.expiration)
        self.catalog.verify_revision()
        if partition_key.data_type == "underlying":
            end = self.calendar.session(partition_key.session_date).option_close
            return self.underlying_reader.read(item, end, self.mapping("underlying"))
        return self.theta_reader.raw_partition(item)

    def _chain(self, expiry: date, instant: datetime, spot: float | None, *, inclusive=False, max_quote_age=60.0, min_oi=None, max_relative_spread=None):
        item = self.catalog.get("greeks", instant.date(), expiry)
        by_contract, diagnostic_tuple = chain_quotes(self.theta_reader, item, instant, spot, self.mapping("greeks"), inclusive)
        diagnostics = list(diagnostic_tuple)
        oi, volumes = {}, {}
        for kind in ("oi_before", "ohlcv"):
            try:
                partition = self.catalog.get(kind, instant.date(), expiry)
                if kind == "oi_before":
                    oi = self.theta_reader.daily_oi(partition, instant, spot, self.mapping(kind))
                else:
                    volumes = self.theta_reader.volumes(partition, instant, spot, self.mapping(kind))
            except DataError as exc:
                diagnostics.append(Diagnostic(exc.code, f"{kind}: {exc.message}", exc.details.get("path"), exc.details.get("row_number"), severity="WARNING"))
        last_trades={}
        try:
            partition=self.catalog.get("trades",instant.date(),expiry)
            last_trades=latest_trades(self.theta_reader,partition,instant,spot,self.mapping("trades"),inclusive=inclusive)
        except DataError as exc:
            if exc.code not in ("PARTITION_MISSING","MISSING_EXPIRY","MISSING_SESSION"):
                diagnostics.append(Diagnostic(exc.code,f"trades: {exc.message}",exc.details.get("path"),exc.details.get("row_number"),severity="WARNING"))
        output = []
        for contract, (latest, valid) in sorted(by_contract.items()):
            age = (instant.astimezone(timezone.utc) - valid.event_at.astimezone(timezone.utc)).total_seconds() if valid else None
            status = "INVALID_LATEST" if not latest.valid else "CURRENT"
            if valid is None:
                status = "INVALID"
            elif latest.valid and max_quote_age is not None and age > max_quote_age:
                status = "STALE"
            if min_oi is not None and (contract not in oi or oi[contract] < min_oi):
                status = "OI_FILTER_REJECTED"
            if max_relative_spread is not None and (valid is None or valid.relative_spread is None or valid.relative_spread > max_relative_spread):
                status = "SPREAD_FILTER_REJECTED"
            trade=last_trades.get(contract)
            output.append(ChainEntry(contract, valid, latest, age, status, oi.get(contract), volumes.get(contract), "STORED_OPTION_OHLCV" if contract in volumes else None,
                                     last_trade_price=trade.price if trade and "AMBIGUOUS_LAST_TRADE_ORDER" not in trade.quality_flags else None,last_trade_at=trade.event_at if trade else None,
                                     last_trade_source=trade.source_id if trade else None,last_trade_quality=trade.quality_flags if trade else ()))
        return tuple(output), tuple(diagnostics)

    def chain_asof(self, expiry: date, instant: datetime, visibility_profile=None, **filters):
        instant = local_datetime(instant)
        underlying = self.underlying_asof(instant)
        return self._chain(expiry, instant, underlying.close if underlying else None, **filters)[0]

    def snapshot(self, decision_at: datetime, expirations=None, state_version: int = 0,
                 inclusive: bool = False, max_quote_age: float | None = 60.0,
                 min_oi: int | None = None, max_relative_spread: float | None = None) -> MarketSnapshot:
        decision_at = local_datetime(decision_at)
        self.catalog.verify_revision()
        diagnostics, candles, boundaries, entries = [], (), (), []
        session = self.calendar.session(decision_at.date())
        try:
            candles, boundaries, errors = self.underlying_reader.read(self.catalog.get("underlying", decision_at.date()), decision_at, self.mapping("underlying"))
            diagnostics.extend(errors)
        except DataError as exc:
            diagnostics.append(Diagnostic(exc.code, exc.message))
        spot = candles[-1].close if candles else None
        spot_age = (decision_at.astimezone(timezone.utc) - candles[-1].available_at.astimezone(timezone.utc)).total_seconds() if candles else None
        underlying_status = "CURRENT" if spot_age == 0 else "PRIOR_OBSERVATION" if spot_age is not None else "MISSING"
        if underlying_status == "PRIOR_OBSERVATION":
            diagnostics.append(Diagnostic("UNDERLYING_PRIOR_OBSERVATION", f"Latest completed authoritative candle is {spot_age:g} seconds old", severity="WARNING"))
        if spot is None:
            diagnostics.append(Diagnostic("UNDERLYING_NOT_AVAILABLE", "No completed authoritative candle at this decision instant", severity="WARNING"))
        chosen = self.catalog.expirations(decision_at.date()) if expirations is None else tuple(expirations)
        for expiry in chosen:
            if isinstance(expiry, str):
                expiry = date.fromisoformat(expiry)
            try:
                chain, errors = self._chain(expiry, decision_at, spot, inclusive=inclusive, max_quote_age=max_quote_age, min_oi=min_oi, max_relative_spread=max_relative_spread)
                entries.extend(chain)
                diagnostics.extend(errors)
            except DataError as exc:
                diagnostics.append(Diagnostic(exc.code, exc.message))
        entries = tuple(sorted(entries, key=lambda entry: entry.contract))
        # Causal row hashes are independent of unseen future bytes. Full archive
        # byte hashes live in the separately bound run revision, not its market state.
        identity = {"asof": decision_at, "candles": candles, "entries": entries, "exclusive": not inclusive,
                    "mappings": self.mappings, "calendar": session, "diagnostics": diagnostics}
        return MarketSnapshot(decision_at, spot, candles, entries, stable_hash(identity), state_version,
                              tuple(diagnostics), not inclusive, "EVENT_TIME_INCLUSIVE_V1" if inclusive else "QQQ_BAR_END_REPLAY_V1", session.scenario_flag, boundaries,
                              spot_age, candles[-1].source_id if candles else None, underlying_status)

    def events_between(self, start: datetime, end: datetime, required_contracts=None):
        """Event-granular quotes satisfy start<=available_at<end.

        Minute-only quotes and candles satisfy start<available_at<=end and
        complete at phase20, before the phase25 completed-bar snapshot.

        This consumes a boundary quote only on the advance following its phase25
        snapshot. All output is stable chronological (instant,phase,sequence,id).
        """
        start, end = local_datetime(start), local_datetime(end)
        if end < start:
            raise DataError("REWIND_REQUIRES_BRANCH", "Event replay cannot move backwards")
        required = set(required_contracts) if required_contracts is not None else None
        events, day = [], start.date()
        while day <= end.date():
            session = self.calendar.session(day)
            if session.status != "OPEN":
                day += timedelta(days=1)
                continue
            cutoff = min(end, session.underlying_close)
            partition = self.catalog.get("underlying", day)
            candles, _, _ = self.underlying_reader.read(partition, cutoff, self.mapping("underlying"))
            for candle in candles:
                if start < candle.available_at <= end:
                    events.append(SourceEvent(candle.available_at, 20, candle.source_id, "candle", candle))
            spot = candles[-1].close if candles else None
            available_keys = [candle.available_at for candle in candles]
            def causal_spot_at(stamp):
                offset = bisect_right(available_keys, stamp) - 1
                return candles[offset].close if offset >= 0 else None
            for expiry in self.catalog.expirations(day):
                quotes, _ = self.theta_reader.quotes(self.catalog.get("greeks", day, expiry), cutoff, spot, self.mapping("greeks"), spot_at=causal_spot_at, interval_start=start)
                for quote in quotes:
                    completed_minute = quote.timestamp_precision == "minute" and quote.available_at == quote.event_at.replace(second=0,microsecond=0)+timedelta(minutes=1)
                    eligible_time = start < quote.available_at <= end if completed_minute else start <= quote.available_at < end
                    if eligible_time and (required is None or quote.contract in required or quote.contract.contract_id in required):
                        events.append(SourceEvent(quote.available_at, 20 if completed_minute else 30, quote.source_id, "quote", quote, quote.sequence or 0))
            day += timedelta(days=1)
        return tuple(sorted(events, key=lambda event: event.key))

    def next_observed_bar(self, instant: datetime):
        instant = local_datetime(instant)
        day = instant.date()
        session = self.calendar.session(day)
        candles, _, _ = self.underlying_reader.read(self.catalog.get("underlying", day), session.underlying_close, self.mapping("underlying"))
        return next((candle for candle in candles if candle.available_at > instant), None)

    def coverage_gap(self, start: datetime, end: datetime, required_contracts=None):
        """First missing scheduled slot; never silently count observed rows instead."""
        by_day = {}
        required_expirations = set()
        for contract in required_contracts or ():
            if hasattr(contract, "expiration"):
                required_expirations.add(contract.expiration)
            elif hasattr(contract, "expiry"):
                required_expirations.add(contract.expiry)
            elif isinstance(contract, str) and contract.startswith("LOCAL:"):
                try:
                    required_expirations.add(date.fromisoformat(contract.split(":")[2]))
                except (IndexError, ValueError) as exc:
                    raise DataError("INVALID_CONTRACT_ID", contract) from exc
            else:
                raise DataError("INVALID_CONTRACT_ID", "Use ContractKey, contract with expiry, or LOCAL identity")
        for label in self.calendar.scheduled_between(start, end):
            if label.date() not in by_day:
                try:
                    rows, _, _ = self.underlying_reader.read(self.catalog.get("underlying", label.date()), end, self.mapping("underlying"))
                    by_day[label.date()] = {candle.label for candle in rows}
                    for expiration in required_expirations:
                        if expiration >= label.date():
                            partition = self.catalog.get("greeks", label.date(), expiration)
                            names = self.theta_reader.header(partition)
                            from .normalization import canonical_schema
                            schema = canonical_schema(names)
                            if any(field not in schema for field in ("strike", "right", "bid", "ask")):
                                return label, "REQUIRED_QUOTE_SCHEMA_MISSING"
                except DataError as exc:
                    return label, exc.code
            if label not in by_day[label.date()]:
                return label, "MISSING_SCHEDULED_MINUTE"
        return None
