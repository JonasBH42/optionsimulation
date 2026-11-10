"""Exact local simulation contracts. No broker transport is implemented."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from hashlib import sha256
import json
from math import gcd
from functools import reduce
from typing import Any

D = Decimal
CENT = D('0.01')
PROFILE_ID = 'TT_COMPONENTS_20261028_SIM_V1'


class BrokerError(ValueError):
    def __init__(self, code: str, message: str = '', **details):
        self.code, self.details = code, details
        super().__init__(message or code)

    def as_dict(self):
        return {'status': 'UNSUPPORTED', 'reason': self.code, 'message': str(self), 'details': self.details}


def decimal(value) -> Decimal:
    result = value if isinstance(value, Decimal) else D(str(value))
    if not result.is_finite():
        raise BrokerError('NONFINITE_MONEY')
    return result


def cents(value) -> int:
    return int((decimal(value) * 100).quantize(D(1), rounding=ROUND_HALF_UP))


def money(value) -> Decimal:
    return D(cents(value)) / 100


def encode(value: Any):
    if is_dataclass(value):
        return encode(asdict(value))
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, set):
        return sorted(encode(v) for v in value)
    return value


def canonical(value) -> str:
    return json.dumps(encode(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value) -> str:
    return sha256(canonical(value).encode('utf-8')).hexdigest()


def require_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise BrokerError('NAIVE_TIMESTAMP')


@dataclass(frozen=True)
class Contract:
    symbol: str
    right: str
    strike: Decimal
    expiry: date
    multiplier: int = 100
    deliverable: str = 'QQQ_STANDARD_100_SHARES'
    style: str = 'AMERICAN'
    currency: str = 'USD'
    root: str = 'QQQ'
    last_trade_at: datetime | None = None
    fixing_at: datetime | None = None
    metadata_version: str = 'SCENARIO_STANDARD_QQQ_V1'

    def __post_init__(self):
        object.__setattr__(self, 'strike', decimal(self.strike))
        object.__setattr__(self, 'right', self.right.upper()[0])
        if self.right not in ('C', 'P') or self.strike < 0 or self.multiplier <= 0:
            raise BrokerError('INVALID_CONTRACT')
        if isinstance(self.expiry, str):
            object.__setattr__(self, 'expiry', date.fromisoformat(self.expiry))

    @property
    def key(self) -> str:
        return '|'.join([self.symbol, self.root, self.expiry.isoformat(), self.right,
                         format(self.strike.normalize(), 'f'), str(self.multiplier),
                         self.deliverable, self.style, self.currency,
                         self.fixing_at.isoformat() if self.fixing_at else ''])

    @property
    def partition(self):
        return (self.symbol, self.expiry, self.multiplier, self.deliverable, self.currency,
                self.fixing_at)

    def intrinsic(self, spot) -> Decimal:
        spot = decimal(spot)
        return max(D(0), spot-self.strike if self.right == 'C' else self.strike-spot)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key in ('last_trade_at', 'fixing_at'):
            if data.get(key): data[key] = datetime.fromisoformat(data[key])
        return cls(**data)


@dataclass(frozen=True)
class Quote:
    contract: Contract
    bid: Decimal | None
    ask: Decimal | None
    event_at: datetime
    quote_id: str
    bid_size: int | None = None
    ask_size: int | None = None
    precision: str = 'event'
    source_sequence: int = 0

    def __post_init__(self):
        require_aware(self.event_at)
        for key in ('bid', 'ask'):
            if getattr(self, key) is not None: object.__setattr__(self, key, decimal(getattr(self, key)))

    @property
    def valid(self):
        return (self.bid is not None and self.ask is not None and self.bid >= 0 and
                self.ask > 0 and self.bid <= self.ask)

    @property
    def midpoint(self):
        return (self.bid+self.ask)/2 if self.valid else None


@dataclass(frozen=True)
class OrderLeg:
    contract: Contract
    quantity: int

    def __post_init__(self):
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) or not self.quantity:
            raise BrokerError('NONINTEGER_OR_ZERO_QUANTITY')


def normalize_legs(legs) -> tuple[OrderLeg, ...]:
    contracts, quantities = {}, {}
    for leg in legs:
        contracts[leg.contract.key] = leg.contract
        quantities[leg.contract.key] = quantities.get(leg.contract.key, 0) + leg.quantity
    result = tuple(OrderLeg(contracts[k], q) for k, q in sorted(quantities.items()) if q)
    if not result: raise BrokerError('EMPTY_ORDER')
    return result


@dataclass(frozen=True)
class OrderDraft:
    campaign_id: str
    legs: tuple[OrderLeg, ...]
    limit_cash: Decimal
    deadline: datetime
    hard_cutoff: datetime
    order_type: str = 'PROTECTED_LIMIT'
    max_total_fees: Decimal = D('50')
    tif: str = 'DAY'
    intent: str = 'OPEN'
    target_fraction: Decimal = D('.50')
    expense_budget: Decimal = D('100')
    operational_model: str = 'SIMULTANEOUS_DELIVERY_HEDGE_V1'
    standing_protection: bool = True
    strict_depth: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'legs', normalize_legs(self.legs))
        for key in ('limit_cash', 'max_total_fees', 'target_fraction', 'expense_budget'):
            object.__setattr__(self, key, decimal(getattr(self, key)))
        require_aware(self.deadline); require_aware(self.hard_cutoff)
        if self.deadline > self.hard_cutoff: raise BrokerError('DEADLINE_EXCEEDS_HARD_CUTOFF')
        if self.order_type not in ('MARKET', 'LIMIT', 'PROTECTED_LIMIT'):
            raise BrokerError('UNSUPPORTED_ORDER_TYPE')
        if self.tif not in ('DAY', 'GTC'): raise BrokerError('UNSUPPORTED_TIME_IN_FORCE')
        if self.max_total_fees < 0 or self.expense_budget < 0:
            raise BrokerError('NEGATIVE_FEE_BUDGET')
        # Limits are dollars for ONE normalized ratio unit, .01 premium points = $1.
        multipliers = {leg.contract.multiplier for leg in self.legs}
        if multipliers != {100}: raise BrokerError('UNSUPPORTED_PACKAGE_TICK')
        if self.limit_cash % D('1') != 0: raise BrokerError('INVALID_PACKAGE_TICK')

    @property
    def units(self):
        return reduce(gcd, (abs(x.quantity) for x in self.legs))

    @property
    def ratio(self):
        return tuple(OrderLeg(x.contract, x.quantity//self.units) for x in self.legs)


@dataclass
class Order:
    order_id: str
    draft: OrderDraft
    submitted_at: datetime
    submitted_sequence: int
    submission_quote_minutes: dict[str, str]
    remaining_units: int
    submission_quote_ids: dict[str,str] = field(default_factory=dict)
    status: str = 'WORKING'
    filled_units: int = 0
    fees: Decimal = D(0)
    reservation: Decimal = D(0)
    last_signature: str | None = None
    last_attempt_at: datetime | None = None
    observations: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class Fill:
    fill_id: str
    order_id: str
    campaign_id: str
    contract: Contract
    quantity: int
    price: Decimal
    event_at: datetime
    quote_id: str
    fees: Decimal
    fee_components: dict
    assumption: str = 'SIMULATED_FROM_QUOTES'


@dataclass
class StockReductionOrder:
    order_id: str
    campaign_id: str
    quantity: int
    remaining: int
    minimum_cash_per_share: Decimal
    submitted_at: datetime
    max_total_fees: Decimal = D(50)
    fees: Decimal = D(0)
    status: str = 'WORKING'
    last_quote_id: str | None = None
    submitted_sequence: int = 0

