"""Exact structural constraints; modeled probability never overrides a failed cap."""
from __future__ import annotations
from dataclasses import replace
from decimal import Decimal
from .contracts import RiskError, canonical, stable_hash
from .payoff import payoff_extrema


def risk_certificate(legs, campaign, limits=None):
    limits = limits or {}
    cash = Decimal(str(campaign.get("net_cash_usd", campaign.get("entry_cash_usd", 0))))
    expense = Decimal(str(campaign.get("remaining_expense_budget_usd", 100)))
    other = Decimal(str(campaign.get("other_contingent_liability_usd", 0)))
    loss_cap = Decimal(str(limits.get("loss_cap_usd", 1000)))
    available = Decimal(str(limits.get("available_cash_usd", 5000)))
    session_cap = Decimal(str(limits.get("session_loss_cap_usd", 1000)))
    session_realized = Decimal(str(limits.get("other_campaign_net_usd", 0)))
    profile = campaign.get("operational_model", "SIMULTANEOUS_DELIVERY_HEDGE_V1")
    result = {"profile": profile, "status": "UNKNOWN", "assumptions": ["SIMULTANEOUS_PROTECTIVE_DELIVERY", "NO_NOTICE_LAG", "STANDING_PROTECTION_INSTRUCTION", "NO_INTRABATCH_FULL_STRIKE_FINANCING"], "target_usd": campaign.get("target_usd"), "pairs": []}
    if profile != "SIMULTANEOUS_DELIVERY_HEDGE_V1" or campaign.get("unresolved_stock_obligation") or campaign.get("protection_instruction_recorded", True) is not True:
        return {**result, "reason": "UNSUPPORTED_OPERATIONAL_ASSUMPTIONS"}
    if expense < 0 or other < 0 or any(abs(x.quantity) > 20 or x.multiplier != 100 or x.deliverable_id != "100_QQQ_SHARES" for x in legs) or len(legs) > 8:
        return {**result, "reason": "UNSUPPORTED_POSITION_OR_RESERVE"}
    if len({x.fixing_utc_us for x in legs}) > 1:
        return {**result, "reason": "MIXED_EXPIRY_UNSUPPORTED"}
    longs = []
    shorts = []
    for leg in legs:
        (longs if leg.quantity > 0 else shorts).extend([leg]*abs(leg.quantity))
    delivery = Decimal(0)
    for short in sorted(shorts, key=lambda x: (x.fixing_utc_us, x.right, x.strike, x.contract_id)):
        possible = [(i, long) for i, long in enumerate(longs) if (long.fixing_utc_us, long.right, long.deliverable_id, long.multiplier) == (short.fixing_utc_us, short.right, short.deliverable_id, short.multiplier)]
        if not possible:
            return {**result, "reason": "UNMATCHED_SHORT_DELIVERY"}
        index, long = min(possible, key=lambda pair: (abs(pair[1].strike-short.strike), pair[1].strike, pair[1].contract_id))
        longs.pop(index)
        eta = 1 if short.right == "CALL" else -1
        amount = short.multiplier*max(eta*(long.strike-short.strike), Decimal(0))
        delivery += amount
        result["pairs"].append({"short": short.contract_id, "long": long.contract_id, "adverse_delivery_cash_usd": str(amount)})
    unpaired_exercise = campaign.get("unpaired_long_exercise_policy", "RESERVE_EXERCISE")
    mandatory = Decimal(10*len(shorts)+(0 if unpaired_exercise == "DECLARED_DNE_PERMITTED" else 5*len(longs)))
    terminal = payoff_extrema(legs, cash)
    operational_loss = max(Decimal(0), -cash+delivery+expense+other)
    bound = max(operational_loss, terminal["max_loss"]+other)
    session_bound = max(Decimal(0), bound-session_realized)
    result.update(delivery_cash_bound_usd=str(delivery), mandatory_delivery_fees_usd=str(mandatory), remaining_expense_budget_usd=str(expense), optional_expense_reserve_usd=str(expense-mandatory), terminal_loss_bound_usd=str(terminal["max_loss"]), operational_loss_bound_usd=str(bound), session_loss_bound_usd=str(session_bound), net_cash_usd=str(cash), pairing_rule="P50_M06_DETERMINISTIC_NEAREST_STRIKE", pairing_is_global_optimum=False, unpaired_long_exercise_policy=unpaired_exercise)
    failures = []
    if expense < mandatory:
        failures.append("MANDATORY_EXPENSE_RESERVE_INSUFFICIENT")
    if bound > loss_cap:
        failures.append("CAMPAIGN_LOSS_CAP_EXCEEDED")
    if session_bound > session_cap:
        failures.append("SESSION_LOSS_CAP_EXCEEDED")
    if delivery+mandatory+other > available:
        failures.append("DELIVERY_FUNDING_INSUFFICIENT")
    result.update(status="FAILED" if failures else "VERIFIED_WITHIN_DECLARED_MODEL", reasons=failures)
    return result


def exact_poststate(legs, campaign, changes, premium_cost_usd, fees_usd):
    quantities = {x.contract_id: x for x in legs}
    for change in changes:
        old = quantities.get(change.contract_id)
        if old is not None and (old.right, old.strike, old.fixing_utc_us, old.multiplier, old.deliverable_id) != (change.right, change.strike, change.fixing_utc_us, change.multiplier, change.deliverable_id):
            raise RiskError("CONTRACT_IDENTITY_COLLISION")
        quantity = change.quantity+(old.quantity if old else 0)
        if quantity:
            quantities[change.contract_id] = replace(change, quantity=quantity)
        else:
            quantities.pop(change.contract_id, None)
    fees = Decimal(str(fees_usd))
    if fees < 0:
        raise RiskError("INVALID_FEE")
    new = dict(campaign)
    new["net_cash_usd"] = str(Decimal(str(campaign.get("net_cash_usd", 0)))-Decimal(str(premium_cost_usd))-fees)
    new["gross_cash_usd"] = str(Decimal(str(campaign.get("gross_cash_usd", campaign.get("net_cash_usd", 0))))-Decimal(str(premium_cost_usd)))
    new["remaining_expense_budget_usd"] = str(Decimal(str(campaign.get("remaining_expense_budget_usd", 100)))-fees)
    # The original target and original benchmarks are deliberately copied unchanged.
    return tuple(sorted(quantities.values(), key=lambda x: (x.fixing_utc_us, x.right, x.strike, x.contract_id))), new


def verify_adjustment(job, campaign, proposal, limits=None, pack=None):
    from .contracts import RiskLeg
    changes = tuple(x if isinstance(x, RiskLeg) else RiskLeg.from_dict(x) for x in proposal.get("legs", ()))
    if "premium_cost_usd" not in proposal or "fees_usd" not in proposal:
        return {"status": "UNSUPPORTED", "reason": "MISSING_EXACT_EXECUTION_OR_FEES", "probability": None}
    fees = Decimal(str(proposal["fees_usd"]))
    if fees > Decimal(str(proposal.get("max_total_fees_usd", 50))):
        return {"status": "REJECTED", "reason": "ORDER_FEE_AUTHORIZATION_EXCEEDED", "probability": None}
    if proposal.get("eligible_utc_us", job.t0_utc_us+1) <= job.t0_utc_us:
        return {"status": "REJECTED", "reason": "ORDER_CAUSALITY_FAILURE", "probability": None}
    legs, updated = exact_poststate(job.signed_legs, campaign, changes, proposal["premium_cost_usd"], fees)
    certificate = risk_certificate(legs, updated, limits)
    current_certificate = risk_certificate(job.signed_legs, campaign, limits)
    result = {"status": "OK" if certificate["status"] == "VERIFIED_WITHIN_DECLARED_MODEL" and current_certificate["status"] == "VERIFIED_WITHIN_DECLARED_MODEL" else "REJECTED", "poststate": canonical(legs), "campaign": canonical(updated), "certificate": certificate, "no_fill_certificate": current_certificate, "probability": None, "probability_reason": "MANAGED_MODEL_COVERAGE_UNAVAILABLE", "proposal_hash": stable_hash(proposal)}
    from .account_adapter import account_proposal_check
    result["account_check"]=account_proposal_check(job,campaign,proposal,limits or {})
    if result["account_check"]["status"]!="OK":
        result["status"]="UNSUPPORTED" if result["account_check"]["status"]=="UNSUPPORTED" else "REJECTED"
        result["reason"]=result["account_check"].get("reason","ACCOUNT_CHECK_FAILED")
    if pack is not None and result["status"] == "OK":
        from .policy import evaluate_policy_action
        result["policy"] = evaluate_policy_action(job, campaign, proposal, pack)
    return result
