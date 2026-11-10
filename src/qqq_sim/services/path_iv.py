"""Causal bounded ATM estimator using an already qualified fast pricing basis."""
from __future__ import annotations
from datetime import datetime
from decimal import Decimal
import math
import statistics
from qqq_sim.risk.contracts import RiskError, RiskLeg, YEAR_US, utc_us


def local_atm_path_iv(snapshot, expiry, parameters, pack, eps_leg):
    from .adapters import probability_snapshot_quality
    if pack is None or not pack.validation.get("capabilities", {}).get("price", {}).get("qualified", False):
        raise RiskError("MISSING_PATH_IV", "Local ATM fallback requires a qualified packed pricing capability")
    from qqq_sim.risk.features import validated_leg_mark
    fixing = datetime.fromisoformat(parameters["fixing_at"])
    tau = (utc_us(fixing)-utc_us(snapshot.asof))/YEAR_US
    r, q = float(parameters["rate_cc"]), float(parameters["dividend_yield_cc"])
    if tau <= 0 or snapshot.spot is None:
        raise RiskError("MISSING_PATH_IV")
    forward = snapshot.spot * math.exp((r-q)*tau)
    near = sorted((entry for entry in snapshot.quotes if entry.contract.expiration == expiry
                   and abs(math.log(float(entry.contract.strike)/forward)) <= .05),
                  key=lambda entry: (abs(math.log(float(entry.contract.strike)/forward)), entry.contract.strike, entry.contract.right, entry.contract.contract_id))
    valid, rejected, selected = [], [], []
    for entry in near:
        quote = entry.latest_event
        try:
            probability_snapshot_quality(snapshot, [entry])
            if quote.iv is None or quote.iv <= 0 or quote.mid is None:
                raise RiskError("LEG_IV_NOT_IDENTIFIABLE")
            key = entry.contract
            leg = RiskLeg(contract_id=key.contract_id, right="CALL" if key.right == "C" else "PUT", strike=key.strike,
                          quantity=1, multiplier=key.multiplier, fixing_utc_us=utc_us(fixing), sigma=quote.iv,
                          bid=float(quote.bid), ask=float(quote.ask), known_at_utc_us=utc_us(quote.available_at), exercise_style=key.exercise_style)
            mark = validated_leg_mark(leg, snapshot.spot, tau, r, q, pack)
            error, price = float(mark["error"]), float(mark["price"])
            half_spread = float((quote.ask-quote.bid)/2)
            residual = abs(price-float(quote.mid))
            if error > eps_leg or residual+error > max(half_spread, eps_leg):
                raise RiskError("VENDOR_IV_REPRICING_FAILED")
            if mark.get("vega_per_1pct") is None:
                raise RiskError("LEG_IV_NOT_IDENTIFIABLE", "Packed vega is unavailable at this cell boundary")
            vega = float(mark["vega_per_1pct"])*100
            if not math.isfinite(vega) or vega <= 0:
                raise RiskError("LEG_IV_NOT_IDENTIFIABLE", "A validated positive vega is required")
            width = max(half_spread, leg.tick/2, eps_leg)
            valid.append({"contract_id": key.contract_id, "strike": str(key.strike), "right": key.right, "sigma": quote.iv,
                          "vega_per_unit": vega, "half_spread": half_spread, "raw_weight": (vega/width)**2,
                          "quote_id": quote.source_id, "known_at": quote.available_at.isoformat(), "quote": [str(quote.bid), str(quote.ask)],
                          "model_price": price, "price_error": error, "model_residual": residual,
                          "iv_basis": "SCOPED_OVERRIDE" if "USER_OVERRIDE_IV" in quote.quality_flags else "REPRICED_NORMALIZED_VENDOR_IV",
                          "inverse_iv_uncertainty": None, "inverse_iv_status": "NOT_INVERTED_ACCEPTED_VENDOR_OR_OVERRIDE"})
            selected.append(entry)
            if len(valid) == 10:
                break
        except RiskError as exc:
            rejected.append({"contract_id": entry.contract.contract_id, "reason": exc.code})
    if len(valid) < 4 or len({item["strike"] for item in valid}) < 2 or {item["right"] for item in valid} != {"C", "P"}:
        raise RiskError("MISSING_PATH_IV", "ATM fallback requires four valid contracts, two strikes and both rights", rejected=rejected, valid_count=len(valid))
    quality = probability_snapshot_quality(snapshot, selected)
    cap = 25*statistics.median(item["raw_weight"] for item in valid)
    total = sum(min(item["raw_weight"], cap) for item in valid)
    for item in valid:
        item["weight"] = min(item["raw_weight"], cap)/total
    ordered = sorted(valid, key=lambda item: (item["sigma"], item["contract_id"]))
    def quantile(fraction):
        cumulative = 0.
        for item in ordered:
            cumulative += item["weight"]
            if cumulative >= fraction:
                return item["sigma"]
        return ordered[-1]["sigma"]
    median, span = quantile(.5), quantile(.9)-quantile(.1)
    if span > max(.03, .30*median):
        raise RiskError("MISSING_PATH_IV", "ATM IV dispersion exceeds supported profile", weighted_span=span, weighted_median=median, contributors=valid)
    return median, {"source": "CAUSAL_VALIDATED_LOCAL_ATM", "pack_id": pack.pack_id, "forward_proxy": forward,
                    "weighted_median": median, "weighted_90_10_span": span, "contributors": valid,
                    "rejected": rejected, "input_quality": quality}
