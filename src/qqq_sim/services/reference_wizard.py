"""The specified three-stage CLI, separate from ordinary fast desktop queries."""
from __future__ import annotations
from collections import Counter
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
from .reference_inputs import normalize_extra_target_pcts, normalize_extra_day_offsets, position_identifier, resolve_trade_override, resolve_extra_dates, build_reference_request, attach_reference_companions

DEFAULT_SYMBOL = ""
TRADE_OVERRIDES = {}
MARKET_INPUT_OVERRIDES = {}
PRODUCT_OVERRIDES = {}
EXTRA_TARGET_PCTS = ()
EXTRA_DAY_OFFSETS = ()
ENGINE_DEFAULTS = {
    "profile": "TT_ALIGNED_APPROVED_FALLBACK_V1", "target_fraction": 0.50,
    "primary_profit_basis": "gross", "premium_interest_accrual": False,
    "initial_monitor_step_seconds": 60, "minimum_pilot_step_seconds": 1.875,
    "production_paths": 200000, "batch_size": 4096, "random_seed": 20261008,
    "pilot_paths": 20000, "grid_tolerance_probability": 0.0025,
    "monitoring": "VERIFIED_ELIGIBLE_OPTION_SESSIONS", "terminal_settlement_reference": True,
    "default_dividend_profile": "continuous_yield", "production_bridge_correction": False,
    "default_defined_risk_only": True, "snapshot_collection_seconds": 15,
    "live_spot_max_age_seconds": 2, "live_option_max_age_seconds": 10,
    "live_quote_dispersion_seconds": 5, "live_iv_max_age_seconds": 60,
    "prior_quote_max_age_seconds": 300, "prior_quote_dispersion_seconds": 60,
    "prior_iv_max_age_seconds": 300, "basis_tolerance_fraction": 0.10,
    "cache_session_retention": 2, "working_memory_budget_mib": 256,
}


def parse_leg(text):
    pieces = text.lower().split()
    if len(pieces) != 3 or pieces[0] not in ("p", "c") or pieces[2] not in ("s", "l"):
        raise ValueError("Use p/c strike s/l, for example: p 500 s")
    try:
        strike = Decimal(pieces[1])
    except InvalidOperation as exc:
        raise ValueError("Strike must be a positive Decimal") from exc
    if not strike.is_finite() or strike <= 0:
        raise ValueError("Strike must be a positive Decimal")
    return pieces[0].upper(), strike, -1 if pieces[2] == "s" else 1


def net_legs(raw):
    amounts = Counter()
    for right, strike, quantity in raw:
        amounts[right, strike] += quantity
    return tuple((right, strike, q) for (right, strike), q in sorted(amounts.items()) if q)


def run_wizard(config, at, input_fn=input, output_fn=print):
    from qqq_sim.cli import error_record
    from qqq_sim.replay.calendar import local_datetime
    from qqq_sim.risk.offline.hit_oracle import calculate_reference
    from qqq_sim.serialization import dumps
    from qqq_sim.config import writable_path
    interactive_symbol = not bool(DEFAULT_SYMBOL.strip())
    symbol = DEFAULT_SYMBOL.strip().upper()
    expiry, raw, stage = None, [], 0 if interactive_symbol else 1
    output_fn("Historical reference — explicit offline calculation; three trade-entry stages.")
    if symbol:
        output_fn(f"Symbol: {symbol}")
    while True:
        try:
            prompt = {0: "Symbol: ", 1: "Stage 1 - Expiration [YYYY-MM-DD]: ",
                      2: "Stage 2 - Enter p/c strike s/l; done, undo, e, q: ",
                      3: "Stage 3 - Deadline, New York [YYYY-MM-DD HH:MM]: "}[stage]
            value = input_fn(prompt).strip()
            if value.lower() in ("q", "quit"):
                return 0
            if value.lower() == "e":
                stage = max(0 if interactive_symbol else 1, stage-1)
                continue
            if stage == 0:
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9.\-]{0,15}", value):
                    raise ValueError("Enter a valid symbol; QQQ is the historical provider's supported product")
                if symbol != value.upper():
                    expiry, raw = None, []
                symbol, stage = value.upper(), 1
                output_fn(f"Symbol: {symbol}")
            elif stage == 1:
                changed = date.fromisoformat(value)
                if changed != expiry:
                    raw = []
                expiry, stage = changed, 2
            elif stage == 2:
                if value.lower() == "undo":
                    if raw:
                        raw.pop()
                elif value.lower() == "done":
                    if not net_legs(raw):
                        raise ValueError("The net position is empty")
                    stage = 3
                else:
                    raw.append(parse_leg(value))
            else:
                deadline = local_datetime(datetime.fromisoformat(value))
                root = PRODUCT_OVERRIDES.get(symbol, {}).get("root", symbol)
                output_fn(f"position_id={position_identifier(symbol, expiry, root, net_legs(raw))}")
                job, metadata = build_reference_request(config, at, symbol, expiry, net_legs(raw), deadline,
                    trade_overrides=TRADE_OVERRIDES, market_overrides=MARKET_INPUT_OVERRIDES,
                    product_overrides=PRODUCT_OVERRIDES, extra_target_pcts=EXTRA_TARGET_PCTS,
                    extra_day_offsets=EXTRA_DAY_OFFSETS, engine_defaults=ENGINE_DEFAULTS)
                if metadata["unmatched_trade_overrides"]:
                    output_fn(f"Ignored {metadata['unmatched_trade_overrides']} unmatched TRADE_OVERRIDES entries; exact position ID required.")
                checkpoint = writable_path(Path(config.runtime_root) / "builds" / "reference_wizard" / (job.hash + ".json"))
                try:
                    output_fn(dumps(attach_reference_companions(job, calculate_reference(job, checkpoint=checkpoint), metadata), pretty=True))
                except KeyboardInterrupt:
                    output_fn("Calculation canceled; completed reference checkpoints are retained.")
                    stage = 2
                    continue
                stage, raw = 2, []
        except (EOFError, KeyboardInterrupt):
            return 0
        except Exception as exc:
            output_fn(dumps(error_record(exc), pretty=True))
