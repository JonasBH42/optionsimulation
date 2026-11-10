"""As-of reference records: missing rates/carry remain missing, never zero."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date
from typing import Any, Iterable
import math
from .models import DataError, stable_hash
from .normalization import parse_timestamp


@dataclass(frozen=True)
class ReferenceRecord:
    kind: str
    symbol: str
    value: Any
    unit: str
    observed_at: datetime
    known_at: datetime
    effective_from: datetime
    effective_to: datetime | None = None
    contract_id_or_null: str | None = None
    expiry_or_tenor: str | None = None
    basis: str = "HISTORICAL_RECORD"
    scenario_flag: bool = False
    identity_hash: str = ""

    def __post_init__(self):
        if not self.identity_hash:
            object.__setattr__(self, "identity_hash", stable_hash({key: getattr(self, key) for key in self.__dataclass_fields__ if key != "identity_hash"}))
        if self.known_at < self.observed_at:
            raise DataError("INVALID_REFERENCE_TIMING", "known_at precedes observed_at")


class ParameterRegistry:
    def __init__(self, records: Iterable[ReferenceRecord] = ()):
        self.records = tuple(records)

    def get(self, kind: str, symbol: str, instant: datetime, contract_id: str | None = None, tenor: str | None = None) -> ReferenceRecord | None:
        eligible = [record for record in self.records if record.kind == kind and record.symbol == symbol
                    and record.contract_id_or_null == contract_id and record.expiry_or_tenor == tenor
                    and record.known_at <= instant and record.effective_from <= instant
                    and (record.effective_to is None or instant < record.effective_to)]
        if not eligible:
            return None
        latest = max(record.known_at for record in eligible)
        choices = [record for record in eligible if record.known_at == latest]
        if len({stable_hash((record.value, record.unit, record.basis)) for record in choices}) > 1:
            raise DataError("CONFLICTING_REFERENCE_RECORDS", kind)
        return sorted(choices, key=lambda record: record.identity_hash)[0]

    def resolve(self, symbol: str, instant: datetime, horizon=None, profile=None) -> dict:
        rates, carry = self.get("rate", symbol, instant), self.get("continuous_carry", symbol, instant)
        reasons = []
        resolved = []
        for name, record in (("RATE", rates), ("CARRY", carry)):
            if record is None:
                reasons.append(name + "_MISSING")
                resolved.append(None)
                continue
            units = {"annual_cc": 1, "annual_cc_decimal": 1, "annual_continuously_compounded_decimal": 1,
                     "annual_cc_percent": 100, "annual_continuously_compounded_percent": 100}
            divisor = units.get(record.unit)
            if divisor is None:
                reasons.append("AMBIGUOUS_" + name + "_UNITS")
                resolved.append(None)
                continue
            try:
                value = float(record.value) / divisor
                if isinstance(record.value, bool) or not math.isfinite(value):
                    raise ValueError()
            except (ValueError, TypeError):
                reasons.append("INVALID_" + name)
                resolved.append(None)
                continue
            resolved.append(value)
        return {"status": "UNSUPPORTED" if reasons else "OK", "rate": resolved[0],
                "carry": resolved[1], "records": tuple(value for value in (rates, carry) if value),
                "reasons": tuple(reasons), "scenario_flag": any(record.scenario_flag for record in (rates, carry) if record),
                "normalized_unit": "annual_continuously_compounded_decimal"}

    @classmethod
    def from_json(cls, path):
        import json
        records = []
        for item in json.loads(open(path, encoding="utf-8").read()):
            for key in ("observed_at", "known_at", "effective_from", "effective_to"):
                if item.get(key):
                    item[key] = parse_timestamp(item[key])
            records.append(ReferenceRecord(**item))
        return cls(records)
