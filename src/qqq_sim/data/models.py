"""Immutable observation contracts; no price or unit is synthesized here."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


def json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (Decimal, Path)):
        return str(value)
    return value


def stable_hash(value: Any) -> str:
    return sha256(json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class DataError(ValueError):
    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.details = code, message, details

    def to_dict(self) -> dict[str, Any]:
        return {"status": "FAILED", "code": self.code, "message": self.message, "details": json_value(self.details)}


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    path: str | None = None
    row_number: int | None = None
    severity: str = "ERROR"


@dataclass(frozen=True, order=True)
class PartitionKey:
    data_type: str
    session_date: date
    expiration: date | None = None


@dataclass(frozen=True)
class Partition:
    key: PartitionKey
    path: str
    size: int
    mtime_ns: int
    sha256: str | None = None
    schema: tuple[str, ...] = ()
    row_count: int | None = None
    schema_fingerprint: str | None = None
    normalization_version: str | None = None
    contract_count: int | None = None
    earliest_raw_at: datetime | None = None
    latest_raw_at: datetime | None = None
    quality_status: str = "UNSCANNED"
    mirror_identity: str | None = None
    audited_at: datetime | None = None


@dataclass(frozen=True)
class CatalogReport:
    partitions: tuple[Partition, ...]
    diagnostics: tuple[Diagnostic, ...]
    roots: tuple[str, ...]
    schema_version: int = 1
    source_identity: str | None = None

    @property
    def ok(self) -> bool:
        return not any(item.severity == "ERROR" for item in self.diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return json_value(self)


@dataclass(frozen=True, order=True)
class ContractKey:
    expiration: date
    strike: Decimal
    right: str
    underlying: str = "QQQ"
    root: str = "QQQ"
    multiplier: int = 100
    deliverable_id: str = "STANDARD_100_SHARES"
    exercise_style: str = "AMERICAN"
    settlement: str = "PHYSICAL"
    currency: str = "USD"
    metadata_verified: bool = False

    def __post_init__(self) -> None:
        if self.right not in ("C", "P") or self.strike <= 0 or self.multiplier <= 0:
            raise DataError("INVALID_CONTRACT", "Positive strike/multiplier and C/P right required")

    @property
    def contract_id(self) -> str:
        return f"LOCAL:{self.root}:{self.expiration.isoformat()}:{self.right}:{self.strike.normalize()}:{self.multiplier}:{self.deliverable_id}"


@dataclass(frozen=True)
class Candle:
    label: datetime
    available_at: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source_id: str
    partial: bool = False
    source_count: int = 1
    boundary: bool = False


@dataclass(frozen=True)
class OptionQuote:
    contract: ContractKey
    event_at: datetime
    available_at: datetime
    bid: Decimal | None
    ask: Decimal | None
    iv: float | None
    valid: bool
    source_id: str
    sequence: int | None = None
    timestamp_precision: str = "subminute"
    quality_flags: tuple[str, ...] = ()
    raw_fields: tuple[tuple[str, str], ...] = ()
    vendor_greeks: tuple[tuple[str, float], ...] = ()
    bid_size: int | None = None
    ask_size: int | None = None
    source_row_number: int | None = None
    source_path: str | None = None
    normalized_greeks: tuple[tuple[str, float], ...] = ()
    normalized_greek_units: tuple[tuple[str, str], ...] = ()

    @property
    def mid(self) -> Decimal | None:
        return (self.bid + self.ask) / 2 if self.valid and self.bid is not None and self.ask is not None else None

    @property
    def relative_spread(self) -> float | None:
        mid = self.mid
        return float((self.ask - self.bid) / mid) if mid and mid > 0 else None


@dataclass(frozen=True)
class ChainEntry:
    contract: ContractKey
    quote: OptionQuote | None
    latest_event: OptionQuote
    age_seconds: float | None
    status: str
    known_oi: int | None = None
    volume: float | None = None
    volume_source: str | None = None
    universe_basis: str = "OBSERVATION_LIMITED"
    last_trade_price: Decimal | None = None
    last_trade_at: datetime | None = None
    last_trade_source: str | None = None
    last_trade_quality: tuple[str,...] = ()


@dataclass(frozen=True)
class MarketSnapshot:
    asof: datetime
    spot: float | None
    candles: tuple[Candle, ...]
    quotes: tuple[ChainEntry, ...]
    snapshot_id: str
    state_version: int = 0
    diagnostics: tuple[Diagnostic, ...] = ()
    exclusive_event_cutoff: bool = True
    timing_profile: str = "QQQ_BAR_END_REPLAY_V1"
    calendar_scenario: bool = True
    boundary_observations: tuple[Candle, ...] = ()
    spot_age_seconds: float | None = None
    spot_source: str | None = None
    underlying_status: str = "MISSING"

    @property
    def quotes_by_contract(self) -> dict[str, ChainEntry]:
        return {entry.contract.contract_id: entry for entry in self.quotes}

    def to_dict(self) -> dict[str, Any]:
        return json_value(self)


@dataclass(frozen=True)
class SourceEvent:
    event_at: datetime
    phase: int
    stable_id: str
    kind: str
    payload: Any
    sequence: int = 0

    @property
    def key(self) -> tuple[datetime, int, int, str]:
        return self.event_at, self.phase, self.sequence, self.stable_id


@dataclass(frozen=True)
class TradeObservation:
    contract: ContractKey
    event_at: datetime
    known_at: datetime
    price: Decimal
    size: int
    trade_id: str
    sequence: int | None
    condition: str
    execution_eligible: bool
    action: str = "TRADE"
    correction_of: str | None = None
    source_id: str = ""
    quality_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class NormalizationMapping:
    timestamp_column: str | None = None
    source_timezone: str = "America/New_York"
    bar_label_semantics: str = "START"
    strike_divisor: Decimal | None = None
    iv_unit: str | None = None
    greek_units: tuple[tuple[str, str], ...] = ()
    metadata_verified: bool = False
    mapping_version: str = "QQQ_NORMALIZATION_V1"
    timestamp_precision: str = "minute"
    confidence: str = "EXPLICIT_CONTRACT"
    rate: float | None = None
    carry: float | None = None
