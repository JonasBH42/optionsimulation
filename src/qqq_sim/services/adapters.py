"""Explicit conversions between immutable market and accounting contracts."""
from __future__ import annotations
from dataclasses import fields,replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path

from qqq_sim.broker.models import Contract, Quote, BrokerError
from qqq_sim.data.models import NormalizationMapping


def load_mapping(path=None):
    if path is None:
        return NormalizationMapping()
    values = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    unknown = set(values) - {f.name for f in fields(NormalizationMapping)}
    if unknown:
        raise ValueError(f"UNKNOWN_MAPPING_FIELDS: {sorted(unknown)}")
    if values.get("strike_divisor") is not None:
        values["strike_divisor"] = Decimal(str(values["strike_divisor"]))
    if "greek_units" in values:
        values["greek_units"] = tuple(tuple(v) for v in values["greek_units"])
    return NormalizationMapping(**values)


def broker_contract(key, calendar):
    session = calendar.session(key.expiration)
    return Contract(symbol=key.underlying, right=key.right, strike=key.strike,
                    expiry=key.expiration, multiplier=key.multiplier,
                    deliverable="QQQ_STANDARD_100_SHARES" if key.deliverable_id == "STANDARD_100_SHARES" else key.deliverable_id,
                    style=key.exercise_style, currency=key.currency, root=key.root,
                    last_trade_at=session.option_close, fixing_at=session.fixing_at,
                    metadata_version="VERIFIED_STANDARD_QQQ_V1" if key.metadata_verified else "UNVERIFIED_ARCHIVE_METADATA")


def broker_quote(quote, calendar):
    return Quote(broker_contract(quote.contract, calendar), quote.bid, quote.ask,
                 quote.event_at, quote.source_id, quote.bid_size, quote.ask_size,
                 "minute" if quote.timestamp_precision == "minute" else "event", quote.sequence or 0)


def snapshot_quotes(snapshot, calendar):
    # An invalid latest update blocks that contract. Do not trade a previous good quote.
    return {broker_contract(entry.contract, calendar).key: broker_quote(entry.latest_event, calendar)
            for entry in snapshot.quotes}


def require_trade_metadata(draft, snapshot, calendar):
    known = {broker_contract(entry.contract, calendar).key: entry for entry in snapshot.quotes}
    for leg in draft.legs:
        entry = known.get(leg.contract.key)
        if entry is None:
            raise BrokerError("CONTRACT_NOT_YET_OBSERVED", leg.contract.key)
        if not entry.contract.metadata_verified:
            raise BrokerError("CONTRACT_METADATA_UNVERIFIED", "Supply verified standard-contract metadata or an explicit synthetic scenario mapping")
        if entry.status not in ("OK", "CURRENT") or not entry.latest_event.valid:
            raise BrokerError("INSUFFICIENT_QUOTE_QUALITY", entry.status)


def probability_snapshot_quality(snapshot, selected_entries, settings=None):
    """Section15.2 historical profile, distinct from60s execution eligibility."""
    from qqq_sim.risk.contracts import RiskError
    from qqq_sim.data.normalization import parse_timestamp
    settings = settings or {}
    quote_limit = float(settings.get("prior_quote_max_age_seconds", 300))
    iv_limit = float(settings.get("prior_iv_max_age_seconds", 300))
    dispersion_limit = float(settings.get("prior_quote_dispersion_seconds", 60))
    if any(not __import__("math").isfinite(value) or value < 0 for value in (quote_limit, iv_limit, dispersion_limit)):
        raise RiskError("INVALID_INPUT_QUALITY_PROFILE")
    instant = snapshot.asof.astimezone(timezone.utc)
    expected_candle = snapshot.asof.replace(second=0, microsecond=0)
    current = snapshot.candles[-1] if snapshot.candles else None
    if current is None or current.available_at != expected_candle or snapshot.spot is None:
        raise RiskError("STALE_AUTHORITATIVE_SPOT", "A current completed QQQ candle is required; prior substitution is disabled")
    earliest, latest, quote_ages, iv_ages = [], [], [], []
    for entry in selected_entries:
        quote = entry.latest_event
        if not quote.valid or quote.available_at > snapshot.asof:
            raise RiskError("INCOHERENT_SNAPSHOT")
        source = dict(quote.raw_fields)
        side_stamps = [quote.event_at]
        for field in ("bid_timestamp", "ask_timestamp", "bid_time", "ask_time"):
            if source.get(field):
                side_stamps.append(parse_timestamp(source[field], snapshot.asof.date()))
        iv_stamp = quote.event_at
        for field in ("iv_timestamp", "iv_time", "implied_vol_timestamp"):
            if source.get(field):
                iv_stamp = parse_timestamp(source[field], snapshot.asof.date())
                break
        if "USER_OVERRIDE_IV" in quote.quality_flags:
            iv_stamp = snapshot.asof
        if any(stamp > snapshot.asof for stamp in (*side_stamps, iv_stamp)):
            raise RiskError("INCOHERENT_SNAPSHOT", "A quote or IV timestamp is later than t0")
        lower = min(stamp.astimezone(timezone.utc) for stamp in side_stamps)
        upper = max(stamp.astimezone(timezone.utc) for stamp in side_stamps)
        if quote.timestamp_precision == "minute":
            upper += timedelta(seconds=60)
        age = (instant - lower).total_seconds()
        iv_age = (instant - iv_stamp.astimezone(timezone.utc)).total_seconds()
        if age > quote_limit:
            raise RiskError("STALE_PROBABILITY_QUOTE", age_seconds=age, limit_seconds=quote_limit)
        if iv_age > iv_limit:
            raise RiskError("STALE_PROBABILITY_IV", age_seconds=iv_age, limit_seconds=iv_limit)
        earliest.append(lower); latest.append(upper); quote_ages.append(age); iv_ages.append(iv_age)
    dispersion = (max(latest) - min(earliest)).total_seconds() if earliest else 0.
    if dispersion > dispersion_limit:
        raise RiskError("PROBABILITY_QUOTE_DISPERSION", seconds=dispersion, limit_seconds=dispersion_limit)
    return {"profile": "REPLAY_QQQ_MINUTE_V1", "quote_max_age_seconds": quote_limit,
            "iv_max_age_seconds": iv_limit, "dispersion_limit_seconds": dispersion_limit,
            "quote_ages_seconds": quote_ages, "iv_ages_seconds": iv_ages, "dispersion_seconds": dispersion,
            "spot_age_seconds": (instant-current.available_at.astimezone(timezone.utc)).total_seconds()}


def parameters_for_expiry(parameters,expiry):
    """Select an explicitly dated fixing record; never infer it from a candle."""
    from qqq_sim.risk.contracts import RiskError
    if not parameters:
        raise RiskError("NEEDS_MODEL_INPUT")
    result=dict(parameters)
    overrides=result.pop("expirations",{})
    if not isinstance(overrides,dict):
        raise RiskError("INVALID_EXPIRATION_PARAMETERS")
    if expiry.isoformat() in overrides:
        record=overrides[expiry.isoformat()]
        if not isinstance(record,dict) or "fixing_at" not in record:
            raise RiskError("EXPIRATION_FIXING_REQUIRED",expiry=expiry.isoformat())
        result.update(record)
    fixing=result.get("fixing_at")
    if fixing is None:
        raise RiskError("EXPIRATION_FIXING_REQUIRED",expiry=expiry.isoformat())
    stamp=datetime.fromisoformat(fixing)
    from qqq_sim.replay.calendar import NY
    if stamp.tzinfo is None or stamp.astimezone(NY).date()!=expiry:
        raise RiskError("INCOMPATIBLE_FIXING",expiry=expiry.isoformat())
    return result


def risk_change_legs(snapshot,draft,parameters,calendar,campaign,pack=None):
    """A roll's change vector may span expiries; the held HOLD job may not."""
    from qqq_sim.risk.contracts import RiskLeg
    groups={}
    for leg in draft.legs:
        groups.setdefault(leg.contract.expiry,[]).append(leg)
    changes=[]
    for expiry,legs in sorted(groups.items()):
        scoped=parameters_for_expiry(parameters,expiry)
        job=risk_job(snapshot,replace(draft,legs=tuple(legs)),scoped,calendar,campaign,pack=pack)
        changes.extend(RiskLeg.from_dict(x) for x in job["signed_legs"])
    by_key={broker_contract(entry.contract,calendar).key:entry for entry in snapshot.quotes}
    probability_snapshot_quality(snapshot,[by_key[x.contract.key] for x in draft.legs])
    return tuple(changes)


def last_trading_for_expiry(parameters,expiry,calendar,contract=None):
    """Keep the contract's trading cutoff distinct from its economic fixing."""
    from qqq_sim.risk.contracts import RiskError
    from qqq_sim.replay.calendar import NY
    explicit=parameters.get("last_trading_at",parameters.get("last_trade_at"))
    value=(datetime.fromisoformat(explicit) if explicit is not None else
           contract.last_trade_at if contract is not None and contract.last_trade_at is not None else
           calendar.session(expiry).option_close)
    if value.tzinfo is None or value.astimezone(NY).date()!=expiry:
        raise RiskError("INVALID_LAST_TRADING_TIMESTAMP",expiry=expiry.isoformat())
    return value


def risk_job(snapshot, draft, parameters, calendar, campaign=None, current_position=None, quality_settings=None, pack=None):
    """Construct risk input only from explicit, known-as-of parameter records."""
    from qqq_sim.risk.contracts import RiskError, utc_us
    if not parameters:
        raise RiskError("NEEDS_MODEL_INPUT", "Load dated rate, carry, path-IV and fixing inputs")
    exact_legs=tuple(current_position if current_position is not None else draft.legs)
    expiries={x.contract.expiry for x in exact_legs}
    if len(expiries)>1:
        raise RiskError("INCOMPATIBLE_FIXING","HOLD requires one current-position expiry; use the separate roll change vector")
    if expiries:
        parameters=parameters_for_expiry(parameters,next(iter(expiries)))
    required = {"rate_cc", "dividend_yield_cc", "known_at", "fixing_at"}
    if required - set(parameters):
        raise RiskError("NEEDS_MODEL_INPUT", "Missing parameter fields", fields=sorted(required-set(parameters)))
    known_at = datetime.fromisoformat(parameters["known_at"])
    if known_at.tzinfo is None or known_at > snapshot.asof:
        raise RiskError("FUTURE_PARAMETER")
    fixing = datetime.fromisoformat(parameters["fixing_at"])
    if fixing.tzinfo is None:
        raise RiskError("INVALID_TIMESTAMP")
    if snapshot.spot is None:
        raise RiskError("MISSING_AUTHORITATIVE_SPOT")
    by_key = {broker_contract(entry.contract, calendar).key: entry for entry in snapshot.quotes}
    legs = []
    cash = Decimal(0)
    selected = [by_key.get(leg.contract.key) for leg in exact_legs]
    if any(entry is None for entry in selected):
        raise RiskError("INCOHERENT_SNAPSHOT")
    quality = probability_snapshot_quality(snapshot, selected, quality_settings)
    for leg in exact_legs:
        entry = by_key.get(leg.contract.key)
        if entry is None or not entry.latest_event.valid:
            raise RiskError("INCOHERENT_SNAPSHOT")
        quote = entry.latest_event
        if quote.iv is None:
            raise RiskError("LEG_IV_NOT_IDENTIFIABLE")
        if leg.contract.expiry != fixing.astimezone(snapshot.asof.tzinfo).date():
            raise RiskError("INCOMPATIBLE_FIXING")
        legs.append({"contract_id": leg.contract.key, "right": leg.contract.right,
                     "strike": str(leg.contract.strike), "quantity": leg.quantity,
                     "multiplier": leg.contract.multiplier, "fixing_utc_us": utc_us(fixing),
                     "last_trading_utc_us":utc_us(last_trading_for_expiry(parameters,leg.contract.expiry,calendar,leg.contract)),
                     "sigma": quote.iv, "bid": float(quote.bid), "ask": float(quote.ask),
                     "half_spread":float((quote.ask-quote.bid)/2),
                     "known_at_utc_us": utc_us(quote.available_at), "exercise_style": leg.contract.style})
        cash -= leg.quantity * leg.contract.multiplier * (quote.ask if leg.quantity > 0 else quote.bid)
    from qqq_sim.risk.contracts import RiskLeg
    from qqq_sim.risk.payoff import payoff_extrema
    # Exact original target is kept when an open campaign is selected.
    if campaign:
        target = campaign["target_usd"]
        cash = Decimal(campaign["entry_cash_usd"])
    else:
        extrema = payoff_extrema(tuple(RiskLeg.from_dict(x) for x in legs), cash)
        maximum = extrema["max_profit"]
        if not maximum.is_finite():
            raise RiskError("UNBOUNDED_MAXIMUM_PROFIT")
        target = maximum * draft.target_fraction
    path_iv_record = None
    sigma_path = parameters.get("sigma_path")
    if sigma_path is None:
        from .path_iv import local_atm_path_iv
        weight = sum(abs(leg.quantity)*leg.contract.multiplier for leg in exact_legs)
        eps_leg = min(.005, 1e-4*max(1., abs(float(target))))/max(weight, 1)
        sigma_path, path_iv_record = local_atm_path_iv(snapshot, fixing.astimezone(snapshot.asof.tzinfo).date(), parameters, pack, eps_leg)
    intervals = []
    day = snapshot.asof.date()
    from datetime import timedelta
    while day <= fixing.astimezone(snapshot.asof.tzinfo).date():
        session = calendar.session(day)
        if session.status == "OPEN":
            intervals.append((utc_us(session.exchange_open), utc_us(session.underlying_close)))
        day += timedelta(days=1)
    deadline=campaign.get('deadline',draft.deadline) if campaign else draft.deadline
    last_trading=min((leg['last_trading_utc_us'] for leg in legs),default=
                     utc_us(last_trading_for_expiry(parameters,fixing.astimezone(snapshot.asof.tzinfo).date(),calendar)))
    return {"t0_utc_us": utc_us(snapshot.asof), "deadline_utc_us": utc_us(deadline),
            "fixing_utc_us": utc_us(fixing), "last_trading_utc_us": last_trading,
            "eligible_intervals": intervals, "x0": snapshot.spot,
            "rate_cc": parameters["rate_cc"], "dividend_yield_cc": parameters["dividend_yield_cc"],
            "sigma_path": sigma_path, "signed_legs": legs,
            "entry_cash_usd": str(cash), "target_usd": str(target),
            "snapshot_id": snapshot.snapshot_id, "snapshot_hash": snapshot.snapshot_id,
            "parameter_provenance": [{"known_at_utc_us": utc_us(known_at), "source": parameters.get("source", "EXPLICIT_SCENARIO")}, {"input_quality": quality}, *([path_iv_record] if path_iv_record else [])],
            "calendar_profile_id": "RTH_ONLY_SCENARIO",
            "job_schema":"P50_REFERENCE_V1" if legs else "MANAGED_STATE_V1"}
