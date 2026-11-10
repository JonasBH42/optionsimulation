"""Immutable, independently serializable P50_REFERENCE_V1 input contracts."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

YEAR_US = 31_536_000_000_000
NY = ZoneInfo("America/New_York")


class RiskError(ValueError):
    def __init__(self, code: str, message: str = "", **details: Any):
        self.code, self.details = code, details
        super().__init__(message or code)

    def as_dict(self):
        return {"status": "UNSUPPORTED", "reason": self.code, "message": str(self), "details": self.details}


def finite(value: Any, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise RiskError("INVALID_CONTRACT_OR_UNITS", f"{name} must be a finite number")
    try:
        x = float(value)
    except (ValueError, TypeError) as exc:
        raise RiskError("INVALID_CONTRACT_OR_UNITS", f"Invalid {name}") from exc
    if not math.isfinite(x):
        raise RiskError("INVALID_CONTRACT_OR_UNITS", f"Nonfinite {name}")
    return x


def utc_us(value: int | str | datetime) -> int:
    if isinstance(value, bool):
        raise RiskError("INVALID_TIMESTAMP")
    if isinstance(value, int):
        return value
    dt = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(dt, datetime) or dt.tzinfo is None:
        raise RiskError("INVALID_TIMESTAMP", "An explicit timezone is required")
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = dt.astimezone(timezone.utc) - epoch
    return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds


def ny_time(text: str) -> int:
    naive = datetime.fromisoformat(text)
    if naive.tzinfo is not None:
        return utc_us(naive)
    candidates = set()
    for fold in (0, 1):
        aware = naive.replace(tzinfo=NY, fold=fold)
        back = aware.astimezone(timezone.utc).astimezone(NY).replace(tzinfo=None)
        if back == naive:
            candidates.add(utc_us(aware))
    if len(candidates) != 1:
        raise RiskError("AMBIGUOUS_OR_NONEXISTENT_LOCAL_TIME", text)
    return candidates.pop()


def canonical(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return canonical(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return "+infinity" if value > 0 else "-infinity" if value < 0 else None
    if hasattr(value, "item"):
        return canonical(value.item())
    return value


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RiskLeg:
    contract_id: str
    right: str
    strike: Decimal
    quantity: int
    fixing_utc_us: int
    sigma: float
    multiplier: int = 100
    exercise_style: str = "AMERICAN"
    deliverable_id: str = "100_QQQ_SHARES"
    bid: float | None = None
    ask: float | None = None
    known_at_utc_us: int | None = None
    half_spread: float = 0.01
    tick: float = 0.01
    last_trading_utc_us: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "right", {"C": "CALL", "P": "PUT"}.get(self.right.upper(), self.right.upper()))
        object.__setattr__(self, "strike", Decimal(str(self.strike)))
        object.__setattr__(self, "fixing_utc_us", utc_us(self.fixing_utc_us))
        if self.last_trading_utc_us is not None:
            object.__setattr__(self,"last_trading_utc_us",utc_us(self.last_trading_utc_us))
        if self.right not in ("CALL", "PUT") or self.exercise_style not in ("AMERICAN", "EUROPEAN"):
            raise RiskError("INVALID_CONTRACT_OR_UNITS", "Invalid option right/style")
        for name in ("quantity", "multiplier"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int) or (name == "multiplier" and val <= 0):
                raise RiskError("INVALID_CONTRACT_OR_UNITS", f"{name} must be an integer")
        if self.quantity == 0 or finite(self.strike, "strike") <= 0 or not self.contract_id:
            raise RiskError("INVALID_CONTRACT_OR_UNITS")
        if not 0 <= finite(self.sigma, "sigma") <= 8:
            raise RiskError("LEG_IV_OUTSIDE_MODEL_DOMAIN")
        if finite(self.tick, "tick") <= 0 or finite(self.half_spread, "half_spread") < 0:
            raise RiskError("INVALID_CONTRACT_OR_UNITS")
        if (self.bid is None) != (self.ask is None):
            raise RiskError("INCOHERENT_SNAPSHOT", "Bid/ask must be supplied together")
        if self.bid is not None and not (0 <= finite(self.bid, "bid") <= finite(self.ask, "ask") and self.ask > 0):
            raise RiskError("INCOHERENT_SNAPSHOT", "Crossed or invalid quote")

    @classmethod
    def from_dict(cls, raw: dict):
        raw = dict(raw)
        if "fixing" in raw:
            raw["fixing_utc_us"] = utc_us(raw.pop("fixing"))
        return cls(**raw)


@dataclass(frozen=True)
class NumericalSettings:
    production_paths: int = 200_000
    pilot_paths: int = 20_000
    region_paths: int = 20_000
    shard_size: int = 4096
    seed: int = 20261008
    monitoring_seconds: tuple[float, ...] = (60, 30, 15, 7.5, 3.75, 1.875)
    space_nodes: int = 1601
    max_space_nodes: int = 12801
    max_refinements: int = 4
    max_grid_points: int = 2_000_000
    max_work_seconds: float = 600.0
    memory_budget_mb: int = 256
    workers: int = 1

    def __post_init__(self):
        for k in ("production_paths", "pilot_paths", "region_paths", "shard_size", "space_nodes", "max_space_nodes", "max_grid_points", "workers"):
            if isinstance(getattr(self, k), bool) or not isinstance(getattr(self, k), int) or getattr(self, k) <= 0:
                raise RiskError("INVALID_NUMERICAL_SETTINGS", k)
        if not self.monitoring_seconds or any(x <= 0 for x in self.monitoring_seconds):
            raise RiskError("INVALID_NUMERICAL_SETTINGS")


@dataclass(frozen=True)
class ReferenceJob:
    t0_utc_us: int
    deadline_utc_us: int
    fixing_utc_us: int
    last_trading_utc_us: int
    eligible_intervals: tuple[tuple[int, int], ...]
    x0: float
    rate_cc: float
    dividend_yield_cc: float
    sigma_path: float
    signed_legs: tuple[RiskLeg, ...]
    entry_cash_usd: Decimal
    target_usd: Decimal
    snapshot_id: str = "EXPLICIT_SCENARIO"
    snapshot_hash: str = ""
    calculation_mode: str = "EXPLICIT_REFERENCE"
    calendar_profile_id: str = "EXPLICIT_INTERVALS"
    state_kind: str = "SPOT"
    target_basis: str = "GROSS_ORIGINAL_TARGET"
    optional_targets_usd: tuple[Decimal, ...] = ()
    reporting_times_utc_us: tuple[int, ...] = ()
    parameter_provenance: tuple[dict, ...] = ()
    numerical_settings: NumericalSettings = field(default_factory=NumericalSettings)
    model_profile: str = "FROZEN_IV_CONTINUOUS_Q_V1"
    job_schema: str = "P50_REFERENCE_V1"

    def __post_init__(self):
        for key in ("t0_utc_us", "deadline_utc_us", "fixing_utc_us", "last_trading_utc_us"):
            object.__setattr__(self, key, utc_us(getattr(self, key)))
        object.__setattr__(self, "entry_cash_usd", Decimal(str(self.entry_cash_usd)))
        object.__setattr__(self, "target_usd", Decimal(str(self.target_usd)))
        object.__setattr__(self, "signed_legs", tuple(self.signed_legs))
        object.__setattr__(self, "eligible_intervals", tuple((utc_us(a), utc_us(b)) for a, b in self.eligible_intervals))
        object.__setattr__(self, "reporting_times_utc_us", tuple(sorted(set(utc_us(t) for t in self.reporting_times_utc_us))))
        object.__setattr__(self, "optional_targets_usd", tuple(Decimal(str(h)) for h in self.optional_targets_usd))
        for key in ("x0", "rate_cc", "dividend_yield_cc", "sigma_path", "entry_cash_usd", "target_usd"):
            finite(getattr(self, key), key)
        if self.x0 <= 0 or not 0 <= self.sigma_path <= 8 or self.state_kind not in ("SPOT", "TERMINAL_FORWARD"):
            raise RiskError("INVALID_CONTRACT_OR_UNITS")
        if not self.t0_utc_us <= self.deadline_utc_us <= self.fixing_utc_us or self.fixing_utc_us <= self.t0_utc_us:
            raise RiskError("INVALID_DEADLINE")
        if any(t < self.t0_utc_us or t > self.fixing_utc_us for t in self.reporting_times_utc_us):
            raise RiskError("INVALID_DEADLINE", "Reporting times must lie between t0 and actual fixing")
        if not self.signed_legs and self.job_schema != "MANAGED_STATE_V1":
            raise RiskError("INVALID_CONTRACT_OR_UNITS", "At least one option leg is required")
        if len({x.contract_id for x in self.signed_legs}) != len(self.signed_legs):
            raise RiskError("INVALID_CONTRACT_OR_UNITS", "Net identical contracts before pricing")
        if any(x.fixing_utc_us != self.fixing_utc_us for x in self.signed_legs):
            raise RiskError("INCOMPATIBLE_FIXING")
        if len({(x.deliverable_id, x.multiplier) for x in self.signed_legs}) > 1:
            raise RiskError("INVALID_CONTRACT_OR_UNITS", "Mixed deliverables")
        if self.state_kind == "TERMINAL_FORWARD" and any(x.exercise_style != "EUROPEAN" for x in self.signed_legs):
            raise RiskError("INVALID_CONTRACT_OR_UNITS", "Forward coordinates require European legs")
        end = None
        for a, b in self.eligible_intervals:
            if a >= b or (end is not None and a < end):
                raise RiskError("INVALID_CALENDAR_INTERVALS")
            end = b
        for record in self.parameter_provenance:
            if "known_at_utc_us" in record and utc_us(record["known_at_utc_us"]) > self.t0_utc_us:
                raise RiskError("FUTURE_PARAMETER", fields=record.get("name"))
        if any(leg.known_at_utc_us is not None and utc_us(leg.known_at_utc_us) > self.t0_utc_us for leg in self.signed_legs):
            raise RiskError("INCOHERENT_SNAPSHOT", "A future quote cannot enter the job")

    @property
    def hash(self):
        return stable_hash(self)

    @property
    def computation_hash(self):
        data=canonical(self)
        for key in ("max_work_seconds","memory_budget_mb","workers","max_grid_points"):
            data["numerical_settings"].pop(key,None)
        return stable_hash(data)

    @property
    def targets(self):
        return tuple(sorted(set((self.target_usd, *self.optional_targets_usd))))

    @property
    def reports(self):
        return tuple(sorted(set((self.deadline_utc_us, self.fixing_utc_us, *self.reporting_times_utc_us))))

    @classmethod
    def from_dict(cls, raw: dict):
        raw = dict(raw)
        raw["signed_legs"] = tuple(x if isinstance(x, RiskLeg) else RiskLeg.from_dict(x) for x in raw["signed_legs"])
        raw["numerical_settings"] = NumericalSettings(**raw.get("numerical_settings", {}))
        return cls(**raw)


normalize_job = ReferenceJob.from_dict
