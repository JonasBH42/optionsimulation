"""Reachable explicit candidate points, with honest bounded-search comparisons."""
from __future__ import annotations
from .contracts import RiskError
from .policy import generate_actions, evaluate_policy_actions
from .hold import evaluate_hold
from .verify import verify_adjustment


def form_envelope(candidates, benchmark=None, benchmark_range=None, truncated=False):
    covered = [x for x in candidates if x.get("q_policy") is not None and x.get("feasible", True)]
    if not covered:
        return {"status": "UNSUPPORTED", "reason": "NO_FEASIBLE_SUPPORTED_ACTION", "v_search": None, "points": [], "truncated": truncated}
    best = max(x["q_policy"] for x in covered)
    required = best if benchmark is None else min(benchmark, best)
    error = max(x.get("error", 0) for x in covered)
    points = []
    for item in covered:
        lower, upper = item.get("empirical_error_range", [item["q_policy"], item["q_policy"]])
        grade = "ROBUST_WITHIN_EMPIRICAL_RANGE" if lower >= required-error else "OUTSIDE" if upper < required-error else "AMBIGUOUS"
        if item["q_policy"] >= required-error:
            points.append({**item, "grade": grade})
    preservation = "UNKNOWN_OR_INCOMPARABLE_INITIAL_BENCHMARK"
    if benchmark_range is not None:
        if any(x.get("empirical_error_range", [x["q_policy"]]*2)[0] >= benchmark_range[1] for x in covered):
            preservation = "YES"
        elif all(x.get("empirical_error_range", [x["q_policy"]]*2)[1] < benchmark_range[0] for x in covered):
            preservation = "NO_WITHIN_SEARCH"
        else:
            preservation = "UNCERTAIN"
    return {"status": "OK", "v_search": best, "required_probability": required, "benchmark_shortfall": None if benchmark is None else max(0, benchmark-best), "benchmark_preservation": preservation, "points": points, "truncated": truncated, "interpretation": "BEST_SEARCHED_FIRST_ACTION_FOLLOWED_BY_FROZEN_POLICY", "interpolated_portfolios_authorized": False}


def optimize_envelope(job, campaign, account_limits=None, candidate_universe=(), pack=None,*,hold_job=None):
    result = {"hold": evaluate_hold(hold_job or job, pack), "managed_probability": None, "pack_id": getattr(pack, "pack_id", None), "target_usd": str(job.target_usd)}
    try:
        if pack is None:
            raise RiskError("PACK_MISSING")
        pack.require_qualified("managed")
        search = generate_actions(job, campaign, candidate_universe, account_limits)
        values = []
        evaluations=evaluate_policy_actions(job,campaign,search["actions"],pack,candidate_universe)
        for action,evaluation in zip(search["actions"],evaluations):
            try:
                from .contracts import RiskLeg,canonical
                from .verify import exact_poststate
                from .payoff import payoff_extrema
                from .greeks import portfolio_greeks
                changes=tuple(RiskLeg.from_dict(x) for x in action["legs"])
                postlegs,postcampaign=exact_poststate(job.signed_legs,campaign,changes,action["premium_cost_usd"],action["fees_usd"])
                greek_vector=portfolio_greeks(job,pack,postlegs)
                values.append({**evaluation,"greeks":greek_vector,"payoff":canonical(payoff_extrema(postlegs,postcampaign["net_cash_usd"])), "construction": action, "feasible": True})
            except RiskError as exc:
                values.append({**exc.as_dict(),"q_policy":None,"construction":action,"feasible":True,"greek_profile_unavailable":True})
        comparable = (campaign.get("benchmark_policy_id") == getattr(pack, "manifest", {}).get("policy_id")
                      and campaign.get("benchmark_deadline_utc_us") == campaign.get("active_deadline_utc_us",job.deadline_utc_us)
                      and campaign.get("benchmark_hard_cutoff_utc_us") == campaign.get("hard_cutoff_utc_us"))
        envelope = form_envelope(values, campaign.get("p0_policy") if comparable else None, campaign.get("p0_policy_range") if comparable else None, search["truncated"])
        result.update(envelope=envelope, search={k: v for k, v in search.items() if k != "actions"}, status=envelope["status"], managed_probability=envelope["v_search"], evaluated_candidates=values)
    except RiskError as exc:
        result.update(exc.as_dict())
    return result

__all__ = ["optimize_envelope", "verify_adjustment", "form_envelope"]
