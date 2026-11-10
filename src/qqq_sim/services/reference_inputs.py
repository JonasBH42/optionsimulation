"""Pure parsing and exact scoped inputs for the three-stage reference wizard."""
from __future__ import annotations
from dataclasses import replace
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
import json
import numbers


def finite_decimal(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite numeric value")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite numeric value") from exc
    if not result.is_finite():
        raise ValueError(f"{label} must be a finite numeric value")
    return result


def normalize_extra_target_pcts(value):
    values = value if isinstance(value, (tuple, list)) else (value,)
    output = {Decimal("0.50")}
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (numbers.Real, Decimal)):
            raise ValueError("EXTRA_TARGET_PCTS accepts numeric percentages, not strings or booleans")
        try:
            number = Decimal(str(item))
        except InvalidOperation as exc:
            raise ValueError("Invalid EXTRA_TARGET_PCTS numeric value") from exc
        if not number.is_finite() or not 0 < number <= 100:
            raise ValueError("EXTRA_TARGET_PCTS values must be finite and in (0,100]")
        output.add(number / 100)
    return tuple(sorted(output))


def normalize_extra_day_offsets(value):
    values = value if isinstance(value, (tuple, list)) else (value,)
    if any(isinstance(item, bool) or not isinstance(item, numbers.Integral) or item <= 0 for item in values):
        raise ValueError("EXTRA_DAY_OFFSETS accepts positive integer calendar-day offsets")
    return tuple(sorted(set(map(int, values))))


def position_identifier(symbol, expiry, root, legs):
    amounts = {}
    for right, strike, quantity in legs:
        key = right.upper(), Decimal(str(strike))
        amounts[key] = amounts.get(key, 0) + quantity
    text = ";".join(f"{right}:{format(strike.normalize(), 'f')}:{quantity:+d}" for (right, strike), quantity in sorted(amounts.items()) if quantity)
    return f"{symbol.upper()}|{expiry.isoformat()}|{root}|{text}"


def resolve_trade_override(identifier, midpoint_cash, overrides=None):
    overrides = overrides or {}
    item = overrides.get(identifier)
    if item is None:
        return {"entry_cash_dollars": Decimal(midpoint_cash), "entry_context": "hypothetical", "round_trip_cost_dollars": None,
                "matched": False, "unmatched_override_count": len(overrides)}
    if not isinstance(item, dict) or set(item) - {"entry_cash_dollars", "entry_context", "round_trip_cost_dollars"}:
        raise ValueError("TRADE_OVERRIDES entry has unsupported fields")
    context = item.get("entry_context", "hypothetical")
    if context not in ("hypothetical", "existing") or context == "existing" and "entry_cash_dollars" not in item:
        raise ValueError("TRADE_OVERRIDES existing context requires original signed entry_cash_dollars")
    cash = finite_decimal(item.get("entry_cash_dollars", midpoint_cash), "entry_cash_dollars")
    cost = finite_decimal(item["round_trip_cost_dollars"], "round_trip_cost_dollars") if "round_trip_cost_dollars" in item else None
    if not cash.is_finite() or cost is not None and (not cost.is_finite() or cost < 0):
        raise ValueError("TRADE_OVERRIDES cash must be finite; fixed cost must be nonnegative")
    return {"entry_cash_dollars": cash, "entry_context": context, "round_trip_cost_dollars": cost,
            "matched": True, "unmatched_override_count": len(overrides) - 1}


def resolve_extra_dates(asof, fixing, offsets):
    from qqq_sim.replay.calendar import add_calendar_days
    included, outside = [], []
    for offset in offsets:
        value = add_calendar_days(asof, offset)
        (included if value <= fixing else outside).append({"offset_days": offset, "at": value.isoformat()})
    return tuple(included), tuple(outside)


def resolve_market_overrides(symbol, expiry, asof, base, overrides):
    from qqq_sim.replay.calendar import local_datetime
    result, applied, changed, ivs = dict(base or {}), [], {}, {}
    if not overrides:
        return result, (), ivs
    entries = overrides if isinstance(overrides, (list, tuple)) else (overrides,) if isinstance(overrides, dict) and "symbol" in overrides else tuple(overrides.values()) if isinstance(overrides, dict) else ()
    if not entries:
        raise ValueError("MARKET_INPUT_OVERRIDES requires dated symbol-scoped records")
    aliases = {"r": "rate_cc", "q": "dividend_yield_cc", "path_iv": "sigma_path",
               "rate_cc": "rate_cc", "dividend_yield_cc": "dividend_yield_cc", "sigma_path": "sigma_path"}
    for item in entries:
        if not isinstance(item, dict) or not {"symbol", "effective_from", "effective_to", "provenance"}.issubset(item):
            raise ValueError("MARKET_INPUT_OVERRIDES requires symbol, effective_from/to and provenance")
        if str(item["symbol"]).upper() != symbol or item.get("expiration") not in (None, expiry.isoformat()):
            continue
        start, end = (local_datetime(datetime.fromisoformat(item[key])) for key in ("effective_from", "effective_to"))
        if not start <= asof < end:
            continue
        known = local_datetime(datetime.fromisoformat(item.get("known_at", item["effective_from"])))
        if known > asof:
            raise ValueError("FUTURE_PARAMETER: MARKET_INPUT_OVERRIDES known_at exceeds snapshot")
        for supplied, normalized in aliases.items():
            if supplied in item:
                value = finite_decimal(item[supplied], "Market override annual decimal")
                if normalized in changed and changed[normalized] != float(value):
                    raise ValueError("CONFLICTING_MARKET_OVERRIDES")
                changed[normalized] = float(value)
        for contract, raw in item.get("contract_ivs", {}).items():
            value = finite_decimal(raw, "Contract IV override")
            if value < 0 or contract in ivs and ivs[contract] != float(value):
                raise ValueError("INVALID_OR_CONFLICTING_CONTRACT_IV_OVERRIDE")
            ivs[contract] = float(value)
        if "fixing_at" in item:
            if "fixing_at" in changed and changed["fixing_at"] != item["fixing_at"]:
                raise ValueError("CONFLICTING_FIXING_OVERRIDES")
            changed["fixing_at"] = item["fixing_at"]
        old_known = local_datetime(datetime.fromisoformat(result["known_at"])) if result.get("known_at") else known
        result["known_at"] = max(old_known, known).isoformat()
        applied.append({"label": "USER_OVERRIDE", "provenance": item["provenance"], "effective_from": start.isoformat(), "effective_to": end.isoformat(), "fields": sorted(key for key in aliases if key in item)})
    result.update(changed)
    if applied:
        result["source"] = "USER_OVERRIDE: " + "; ".join(str(item["provenance"]) for item in applied)
    return result, tuple(applied), ivs


def build_reference_request(config, at, symbol, expiry, legs, deadline, *, trade_overrides=None, market_overrides=None,
                            product_overrides=None, extra_target_pcts=(), extra_day_offsets=(), engine_defaults=None, data=None):
    from qqq_sim.cli import make_data
    from qqq_sim.broker.models import OrderDraft, OrderLeg
    from qqq_sim.services.adapters import broker_contract, risk_job
    from qqq_sim.risk.contracts import ReferenceJob, NumericalSettings, utc_us, stable_hash
    from qqq_sim.risk.payoff import payoff_extrema
    from qqq_sim.replay.calendar import local_datetime
    if symbol != "QQQ":
        raise ValueError("PRODUCT_ADAPTER_UNSUPPORTED: local historical provider supports QQQ")
    if at is None:
        raise ValueError("NEEDS_MODEL_INPUT: supply --at for the historical snapshot")
    at = local_datetime(datetime.fromisoformat(at) if isinstance(at, str) else at)
    data = make_data(config) if data is None else data
    snapshot = data.snapshot(at, expirations=[expiry])
    parameters = json.loads(Path(config.parameters_path).read_text(encoding="utf-8-sig")) if config.parameters_path else {}
    parameters, applied, ivs = resolve_market_overrides(symbol, expiry, snapshot.asof, parameters, market_overrides)
    if ivs:
        entries = []
        for entry in snapshot.quotes:
            if entry.contract.contract_id in ivs:
                value = ivs[entry.contract.contract_id]
                latest = replace(entry.latest_event, iv=value, quality_flags=(*entry.latest_event.quality_flags, "USER_OVERRIDE_IV"))
                mark = replace(entry.quote, iv=value, quality_flags=(*entry.quote.quality_flags, "USER_OVERRIDE_IV")) if entry.quote else None
                entry = replace(entry, latest_event=latest, quote=mark)
            entries.append(entry)
        snapshot = replace(snapshot, quotes=tuple(entries), snapshot_id=stable_hash({"source_snapshot": snapshot.snapshot_id, "contract_iv_overrides": ivs}))
    product = (product_overrides or {}).get(symbol, {})
    if not isinstance(product, dict) or set(product) - {"root", "eligible_intervals"}:
        raise ValueError("PRODUCT_OVERRIDES supports exact root and explicit eligible_intervals only")
    root = product.get("root", "QQQ")
    contracts = {(entry.contract.right, entry.contract.strike): entry.contract for entry in snapshot.quotes if entry.contract.root == root}
    identifier = position_identifier(symbol, expiry, root, legs)
    missing = [(right, str(strike)) for right, strike, _ in legs if (right, strike) not in contracts]
    if not legs or missing:
        raise ValueError(f"CONTRACT_NOT_OBSERVED: {missing}; position_id={identifier}")
    order_legs = tuple(OrderLeg(broker_contract(contracts[right, strike], data.calendar), quantity) for right, strike, quantity in legs)
    draft = OrderDraft("reference", order_legs, Decimal(0), deadline, deadline)
    pack = None
    if parameters.get("sigma_path") is None and config.pack_path:
        from qqq_sim.risk.pack_loader import ModelPack
        pack = ModelPack.load(config.pack_path)
    model_input = risk_job(snapshot, draft, parameters, data.calendar, quality_settings=engine_defaults, pack=pack)
    if "eligible_intervals" in product:
        intervals = []
        for pair in product["eligible_intervals"]:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("PRODUCT_OVERRIDES eligible_intervals requires opening/closing timestamp pairs")
            opening, closing = (local_datetime(datetime.fromisoformat(value)) for value in pair)
            if opening.date() != closing.date():
                raise ValueError("PRODUCT_OVERRIDES sessions must remain inside one supported QQQ session")
            session = data.calendar.session(opening.date())
            if session.status != "OPEN" or not session.exchange_open <= opening < closing <= session.underlying_close:
                raise ValueError("PRODUCT_OVERRIDES session exceeds supported underlying/option intersection")
            intervals.append((utc_us(opening), utc_us(closing)))
        if not intervals:
            raise ValueError("PRODUCT_OVERRIDES requires at least one eligible interval")
        model_input["eligible_intervals"] = sorted(intervals)
        model_input["calendar_profile_id"] = "USER_OVERRIDE_QQQ_SESSION_SUBSET"
    quote_index = {entry.contract.contract_id: entry.latest_event for entry in snapshot.quotes}
    midpoint_cash = -sum((quantity * contracts[right, strike].multiplier * quote_index[contracts[right, strike].contract_id].mid for right, strike, quantity in legs), Decimal(0))
    override = resolve_trade_override(identifier, midpoint_cash, trade_overrides)
    cash = override["entry_cash_dollars"]
    job = ReferenceJob.from_dict({**model_input, "entry_cash_usd": str(cash)})
    extrema = payoff_extrema(job.signed_legs, cash)
    maximum = extrema["max_profit"]
    if not maximum.is_finite() or maximum <= 0:
        raise ValueError(f"UNDEFINED_MAX_PROFIT_TARGET: position_id={identifier}")
    defaults = engine_defaults or {}
    if defaults.get("default_defined_risk_only", True) and not extrema["max_loss"].is_finite():
        raise ValueError(f"UNBOUNDED_LOSS: default new-trade gate; position_id={identifier}")
    fractions = normalize_extra_target_pcts(extra_target_pcts)
    targets = {maximum * fraction for fraction in fractions}
    cost, net_spec = override["round_trip_cost_dollars"], None
    if cost is not None:
        net_maximum = maximum - cost
        if net_maximum <= 0:
            raise ValueError("NET_MAX_PROFIT_NONPOSITIVE: fixed cost exhausts maximum profit")
        gross_equivalent = cost + net_maximum / 2
        targets.add(gross_equivalent)
        net_spec = {"fixed_round_trip_cost_dollars": str(cost), "net_max_profit_dollars": str(net_maximum),
                    "net_target_dollars": str(net_maximum / 2), "gross_equivalent_threshold": str(gross_equivalent),
                    "terminal_cost_status": "TERMINAL_COST_CONVENTION_NOT_SPECIFIED"}
    fixing = local_datetime(datetime.fromisoformat(parameters["fixing_at"]))
    included, outside = resolve_extra_dates(snapshot.asof, fixing, normalize_extra_day_offsets(extra_day_offsets))
    settings = NumericalSettings(production_paths=defaults.get("production_paths", 200000), pilot_paths=defaults.get("pilot_paths", 20000),
                                 shard_size=defaults.get("batch_size", 4096), seed=defaults.get("random_seed", 20261008), memory_budget_mb=defaults.get("working_memory_budget_mib", 256))
    job = replace(job, target_usd=maximum / 2, optional_targets_usd=tuple(sorted(targets - {maximum / 2})),
                  reporting_times_utc_us=tuple(utc_us(item["at"]) for item in included), numerical_settings=settings)
    metadata = {"position_id": identifier, "entry_context": override["entry_context"], "entry_cash_dollars": str(cash),
                "entry_basis": "EXACT_POSITION_OVERRIDE" if override["matched"] else "HYPOTHETICAL_NET_MIDPOINT",
                "unmatched_trade_overrides": override["unmatched_override_count"], "original_max_profit_dollars": str(maximum),
                "original_target_dollars": str(maximum / 2), "prior_target_hit_status": "UNKNOWN" if override["entry_context"] == "existing" else "NOT_APPLICABLE_HYPOTHETICAL",
                "extra_target_fractions": [str(value) for value in fractions], "extra_dates": included, "outside_horizon_dates": outside,
                "market_overrides": applied, "product_overrides": product, "net_fixed_cost_companion": net_spec, "snapshot_id": snapshot.snapshot_id}
    return job, metadata


def attach_reference_companions(job, result, metadata):
    """Keep gross headline untouched; fixed closing fees define a separate event."""
    output = {**result, "controller": metadata}
    cost = metadata.get("net_fixed_cost_companion")
    if cost is None:
        return output
    companion = {**cost, "status": result.get("status"), "deadline_closing_probability": None,
                 "through_last_trading_closing_probability": None, "terminal_inclusive_probability": None,
                 "event": "FIXED_COST_NET_TARGET_ON_ELIGIBLE_CLOSING_OBSERVATION"}
    paths = result.get("production_paths", 0)
    if result.get("status") in ("OK", "PRELIMINARY") and paths > 0:
        target = Decimal(cost["gross_equivalent_threshold"])
        index = job.targets.index(target)
        histograms = result.get("first_hit_histograms", ())
        if index < len(histograms):
            histogram = histograms[index]
            deadline_count = sum(count for instant, count in histogram.items() if int(instant) <= job.deadline_utc_us)
            companion["deadline_closing_probability"] = deadline_count / paths
            companion["through_last_trading_closing_probability"] = sum(histogram.values()) / paths
            companion["reporting_dates"] = [{"at_utc_us": instant, "closing_probability": sum(count for observed, count in histogram.items() if int(observed) <= instant) / paths} for instant in job.reports]
    output["net_fixed_cost_companion"] = companion
    return output
