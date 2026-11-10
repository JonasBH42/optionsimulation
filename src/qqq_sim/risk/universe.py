"""Fast fixed-schema M09 future-contract encoder; no offline imports."""
from __future__ import annotations
import math
import numpy as np
from .contracts import RiskError, YEAR_US, canonical, stable_hash

UNIVERSE_TOKEN_FIELDS = ("mask", "log_moneyness", "remaining_years", "right_sign", "frozen_iv", "multiplier_div100",
                         "half_spread_div_spot", "tick_div_spot", "american", "last_trade_years", "fixing_years",
                         "bid_div_spot", "ask_div_spot", "available_now", "remaining_decisions_div12800", "expiry_ordinal_div8")
UNIVERSE_FEATURE_SCHEMA = "M09_UNIVERSE_16_64_64_POOL154_V1"
UNIVERSE_FEATURE_SIZE = 154


def _last_trade(job, leg):
    explicit = getattr(leg, "last_trading_utc_us", None)
    if explicit is not None:
        return int(explicit)
    if leg.fixing_utc_us == job.fixing_utc_us:
        return job.last_trading_utc_us
    raise RiskError("UNIVERSE_CONTRACT_TIMES_MISSING", contract_id=leg.contract_id)


def _remaining_decisions(job, last_trade):
    total = 0
    for opening, closing in job.eligible_intervals:
        first = max(opening, (job.t0_utc_us // 60_000_000 + 1) * 60_000_000)
        stop = min(closing, last_trade)
        if first < stop:
            total += (stop - 1 - first) // 60_000_000 + 1
    if total > 12800:
        raise RiskError("UNIVERSE_OUTSIDE_MODEL_DOMAIN", "More than12800 remaining decision opportunities")
    return total


def universe_tokens(job, universe):
    universe = tuple(universe)
    if len(universe) > 512 or len({leg.contract_id for leg in universe}) != len(universe):
        raise RiskError("UNIVERSE_OUTSIDE_MODEL_DOMAIN", "Contract count or duplicate identity")
    expiries = sorted({leg.fixing_utc_us for leg in universe})
    if len(expiries) > 8:
        raise RiskError("UNIVERSE_OUTSIDE_MODEL_DOMAIN", "More than8 expiries")
    output = np.zeros((512, 16), dtype=np.float64)
    ordered = sorted(universe, key=lambda leg: (leg.fixing_utc_us, leg.right, leg.strike, leg.contract_id))
    for row, leg in enumerate(ordered):
        if leg.known_at_utc_us is None:
            raise RiskError("UNIVERSE_AVAILABILITY_UNKNOWN", contract_id=leg.contract_id)
        if leg.known_at_utc_us > job.t0_utc_us:
            raise RiskError("FUTURE_UNIVERSE_RECORD", contract_id=leg.contract_id)
        remaining = (leg.fixing_utc_us - job.t0_utc_us) / YEAR_US
        if remaining <= 0:
            raise RiskError("UNIVERSE_EXPIRED_CONTRACT", contract_id=leg.contract_id)
        last_trade = _last_trade(job, leg)
        quote_known = leg.bid is not None and leg.ask is not None
        available = quote_known and last_trade > job.t0_utc_us and any(start <= job.t0_utc_us < end for start, end in job.eligible_intervals)
        output[row] = [1, math.log(float(leg.strike) / job.x0), remaining, 1 if leg.right == "CALL" else -1,
                       leg.sigma, leg.multiplier / 100, leg.half_spread / job.x0, leg.tick / job.x0,
                       float(leg.exercise_style == "AMERICAN"), (last_trade - job.t0_utc_us) / YEAR_US, remaining,
                       float(leg.bid) / job.x0 if quote_known else 0, float(leg.ask) / job.x0 if quote_known else 0,
                       float(available), _remaining_decisions(job, min(last_trade, leg.fixing_utc_us)) / 12800,
                       (expiries.index(leg.fixing_utc_us) + 1) / 8]
    if not np.all(np.isfinite(output)):
        raise RiskError("INVALID_UNIVERSE_FEATURE")
    return output


def universe_identity(job, universe):
    return stable_hash({"asof": job.t0_utc_us, "universe": canonical(sorted(universe, key=lambda leg: leg.contract_id)), "schema": UNIVERSE_FEATURE_SCHEMA})


def _silu(value):
    return value * np.exp(-np.logaddexp(0, -value))


def universe_features(job, universe, pack):
    tokens = universe_tokens(job, universe)
    count = int(tokens[:, 0].sum())
    if count == 0:
        return np.zeros(UNIVERSE_FEATURE_SIZE, dtype=np.float64)
    arrays = pack.arrays
    expected = {"universe_w0": (16, 64), "universe_b0": (64,), "universe_w1": (64, 64), "universe_b1": (64,)}
    for name, shape in expected.items():
        if name not in arrays or arrays[name].shape != shape or arrays[name].dtype != np.float64 or not np.all(np.isfinite(arrays[name])):
            raise RiskError("UNIVERSE_ENCODER_MISSING_OR_INVALID", field=name)
    active = tokens[:count]
    hidden = _silu(active @ arrays["universe_w0"] + arrays["universe_b0"])
    hidden = _silu(hidden @ arrays["universe_w1"] + arrays["universe_b1"])
    expiry_summary = np.zeros((8, 3), dtype=np.float64)
    for ordinal in range(1, 9):
        subset = active[np.rint(active[:, 15] * 8).astype(int) == ordinal]
        if len(subset):
            expiry_summary[ordinal - 1] = [1, subset[0, 2], len(subset) / 512]
    result = np.concatenate((hidden.sum(axis=0) / math.sqrt(count), hidden.max(axis=0),
                             [count / 512, np.count_nonzero(expiry_summary[:, 0]) / 8], expiry_summary.ravel()))
    if not np.all(np.isfinite(result)):
        raise RiskError("INVALID_UNIVERSE_FEATURE")
    return result


def universe_feature_mask_indices(offset=0):
    return tuple(offset + 130 + 3 * index for index in range(8))


def hold_flag_indices():
    """All current schema flags and masks bypass standardization."""
    from .feature_layout import hold_masks
    return hold_masks()
