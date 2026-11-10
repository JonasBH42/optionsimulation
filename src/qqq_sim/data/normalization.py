"""Conservative schema aliases, units, and New York timestamp interpretation."""
from __future__ import annotations
from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
import math
from typing import Any, Callable
from functools import lru_cache
from .models import DataError, NormalizationMapping
from qqq_sim.replay.calendar import NY, local_datetime

ALIASES = {
    "timestamp": ("timestamp", "datetime", "regular_time", "time", "quote_time", "date_time"),
    "ms_of_day": ("ms_of_day", "msofday", "milliseconds_of_day"),
    "strike": ("strike", "strike_price", "strikeprice", "strike_milli", "strike_millis"),
    "right": ("right", "option_type", "type", "put_call", "cp"),
    "bid": ("bid", "bid_price"), "ask": ("ask", "ask_price"),
    "last": ("last", "last_price", "price"),
    "implied_vol": ("implied_vol", "implied_volatility", "impliedvolatility", "iv"),
    "underlying_price": ("underlying_price", "underlying_px", "underlying"),
    "open_interest": ("open_interest", "openinterest", "oi"),
    "expiration": ("expiration", "expiry", "expiration_date"),
    "sequence": ("sequence", "seq", "source_sequence"),
}


def canonical_schema(names) -> dict[str, str]:
    lower = {str(name).strip().lower(): name for name in names}
    return {key: next(lower[alias] for alias in aliases if alias in lower) for key, aliases in ALIASES.items() if any(alias in lower for alias in aliases)}


def field(row: dict, name: str, schema: dict[str, str], default=None):
    value = row.get(schema.get(name, name), default)
    return default if value is None or value == "" else value


@lru_cache(maxsize=16384)
def parse_timestamp(value: Any, day: date | None = None) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return local_datetime(value)
    if isinstance(value, date):
        return datetime.combine(value, time(), NY)
    string = str(value).strip()
    if not string:
        raise DataError("INVALID_TIMESTAMP", "Empty timestamp")
    try:
        if "T" not in string and " " not in string and ":" in string:
            if day is None:
                raise ValueError("Time-only value requires session date")
            value = datetime.combine(day, time.fromisoformat(string))
        else:
            value = datetime.fromisoformat(string.replace("Z", "+00:00"))
        return local_datetime(value)
    except ValueError as exc:
        if getattr(exc, "code", None):
            raise
        raise DataError("INVALID_TIMESTAMP", string) from exc


def row_timestamp(row: dict, schema: dict, day: date, mapping: NormalizationMapping, *, daily: bool = False) -> datetime:
    chosen = mapping.timestamp_column or schema.get("timestamp")
    first = parse_timestamp(row[chosen], day) if chosen and row.get(chosen) not in (None, "") else None
    ms_value = field(row, "ms_of_day", schema)
    second = None
    if ms_value is not None:
        ms = Decimal(str(ms_value))
        if ms != ms.to_integral_value() or not 0 <= ms < 86_400_000:
            raise DataError("INVALID_MS_OF_DAY", str(ms_value))
        date_value = row.get("date", day.isoformat())
        date_string = str(date_value)
        parsed_day = datetime.strptime(date_string, "%Y%m%d").date() if len(date_string) == 8 and date_string.isdigit() else date.fromisoformat(date_string[:10])
        second = datetime.combine(parsed_day, time(), NY) + timedelta(milliseconds=int(ms))
    if first is not None and second is not None and first != second:
        raise DataError("CONTRADICTORY_TIMESTAMPS", "timestamp and date/ms_of_day disagree")
    result = first or second
    if result is None and daily:
        result = datetime.combine(day, time(9, 30), NY)
    if result is None:
        raise DataError("TIMESTAMP_MISSING", "No recognized timestamp or date/ms_of_day")
    if result.date() != day:
        raise DataError("SESSION_IDENTITY_MISMATCH", f"Row {result.date()} differs from partition {day}")
    return result


def decimal_value(value, *, nullable: bool = False) -> Decimal | None:
    if nullable and value in (None, ""):
        return None
    try:
        parsed = Decimal(str(value))
        if not parsed.is_finite():
            raise InvalidOperation()
        return parsed
    except (InvalidOperation, TypeError, ValueError) as exc:
        if nullable:
            return None
        raise DataError("NONFINITE_VALUE", str(value)) from exc


def normalize_right(value: str) -> str:
    values = {"C": "C", "CALL": "C", "P": "P", "PUT": "P"}
    result = values.get(str(value).strip().upper())
    if result is None:
        raise DataError("INVALID_OPTION_RIGHT", str(value))
    return result


def resolve_strike(raw, spot: float | None, mapping: NormalizationMapping, header: str = "strike") -> tuple[Decimal, Decimal]:
    value = decimal_value(raw)
    if value <= 0:
        raise DataError("INVALID_STRIKE", str(raw))
    if mapping.strike_divisor is not None:
        divisor = Decimal(mapping.strike_divisor)
        if divisor <= 0:
            raise DataError("INVALID_STRIKE_SCALE", str(divisor))
        return value / divisor, divisor
    # Header meaning is explicit schema evidence, unlike a plausible numeric range.
    if header.lower() in ("strike_milli", "strike_millis"):
        return value / Decimal(1000), Decimal(1000)
    if spot is None or spot <= 0:
        raise DataError("STRIKE_SCALE_REQUIRES_KNOWN_SPOT", "Supply causal authoritative spot or explicit scale mapping")
    candidates = [divisor for divisor in (Decimal(1), Decimal(1000)) if Decimal(str(spot)) / 10 <= value / divisor <= Decimal(str(spot)) * 10]
    if len(candidates) != 1:
        raise DataError("AMBIGUOUS_STRIKE_SCALE", "No unique scale agrees with known spot", raw=str(raw), candidates=[str(x) for x in candidates])
    return value / candidates[0], candidates[0]


def resolve_iv(raw, mapping: NormalizationMapping, acceptance: Callable[[float], bool] | None = None) -> tuple[float | None, tuple[str, ...]]:
    if raw in (None, ""):
        return None, ("IV_MISSING",)
    parsed = decimal_value(raw, nullable=True)
    if parsed is None or parsed < 0:
        return None, ("IV_INVALID",)
    if mapping.iv_unit in ("decimal", "percent"):
        value = float(parsed / (100 if mapping.iv_unit == "percent" else 1))
        if not 0 <= value <= 4:
            return None, ("IV_OUTSIDE_NUMERIC_DOMAIN",)
        return value, ()
    # Magnitude by itself is not proof. A unique repricing-consistent candidate is.
    if acceptance is not None:
        candidates = [x for x in (float(parsed), float(parsed / 100)) if 0 <= x <= 4 and acceptance(x)]
        if len(candidates) == 1:
            return candidates[0], ()
    return None, ("AMBIGUOUS_IV_UNITS",)


def legacy_whole_session(selector: str | time | None, kind: str) -> bool:
    return kind in ("greeks", "iv", "ohlcv") and str(selector) == "00:00:00"


def normalize_vendor_greeks(values, mapping, multiplier):
    """Convert only explicitly documented units; preserve raw vendor values."""
    supplied = {}
    for name, unit in mapping.greek_units:
        if name in supplied and supplied[name] != unit:
            raise DataError("CONFLICTING_GREEK_UNITS", name)
        supplied[name] = unit
    converted, units, flags = [], [], []
    for name, value in values:
        declared = supplied.get(name)
        divisor = 1.
        if declared is not None and declared.endswith("_per_contract"):
            divisor = float(multiplier)
            declared = declared.removesuffix("_per_contract")
        table = {}
        if name in ("delta", "dual_delta"):
            table = {"decimal": (1., "per_unit"), "per_unit": (1., "per_unit"), "percent": (.01, "per_unit")}
        elif name in ("gamma", "dual_gamma"):
            table = {"per_dollar": (1., "per_dollar")}
        elif name == "theta":
            table = {"per_day": (1., "per_day"), "per_year_act365": (1/365, "per_day")}
        elif name in ("vega", "rho", "epsilon"):
            table = {"per_1pct": (1., "per_1pct"), "per_unit": (.01, "per_1pct")}
        selected = table.get(declared)
        if selected is None:
            flags.append("VENDOR_GREEK_UNITS_UNVERIFIED:"+name)
            continue
        scale, canonical_unit = selected
        converted.append((name, value*scale/divisor))
        units.append((name, canonical_unit))
    return tuple(converted), tuple(units), tuple(flags)
