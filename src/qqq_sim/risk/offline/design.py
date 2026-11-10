"""Persisted, semantic-seed Latin hypercube counterfactual design records."""
from __future__ import annotations
import math
import time
import statistics
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
import numpy as np
from ..contracts import RiskLeg, ReferenceJob, NumericalSettings, RiskError, NY, utc_us, canonical, stable_hash, YEAR_US

SEED = 20261109
SEED_KEYS = {"DESIGN": 1, "TRAIN_INIT": 2, "TRAIN_SHUFFLE": 3, "CALIBRATION": 4, "VALIDATION": 5, "POLICY_ROLLOUT": 6}
ANCHORS = ("2026-01-05", "2026-03-06", "2026-06-18", "2026-10-30", "2026-11-25")
EXPIRY_OFFSETS = (0, 1, 2, 4, 7, 14, 21, 30)
WIDTHS = (1, 2, 5, 10, 20)
OFFSETS = (-10, -5, -2, -1, 0, 1, 2, 5, 10)


def latin_hypercube(count=32768, dimensions=12, cycle=0):
    coordinates = np.empty((count, dimensions), dtype=np.float64)
    permutations = np.empty((count, dimensions), dtype=np.int64)
    jitters = np.empty_like(coordinates)
    for dimension in range(dimensions):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(SEED, spawn_key=(1, cycle, dimension))))
        permutation = rng.permutation(count)
        jitter = rng.random(count)
        coordinates[:, dimension] = (permutation+jitter)/count
        permutations[:, dimension], jitters[:, dimension] = permutation, jitter
    return {"coordinates": coordinates, "permutations": permutations, "jitters": jitters}


def design_record(index, coordinate, cycle=0):
    u = coordinate
    spot = math.exp(math.log(100)+u[0]*math.log(15))
    seconds = math.exp(math.log(60)+u[1]*math.log(30*86400/60))
    sigma = math.exp(math.log(.03)+u[2]*math.log(2/.03))
    family = ("VERTICAL", "CONDOR_BUTTERFLY", "ASYMMETRIC", "ONGOING_CAMPAIGN")[index % 4]
    shape_group = f"{family}:{index//4%128}"
    group_hash = int(stable_hash(shape_group)[:8], 16)%100
    split = "train" if group_hash < 70 else "calibration" if group_hash < 85 else "validation"
    return {"index": index, "cycle": cycle, "coordinates": coordinate.tolist(), "seed": {"entropy": SEED, "spawn_key": [1, cycle, index]}, "spot": spot, "desired_seconds": seconds, "sigma_anchor": sigma, "r": -.02+.22*u[3], "q": .2*u[4], "skew": -.5+u[5], "curvature": u[6], "term_slope": -.5+u[7], "cash_gap": -1+2*u[8], "quantity": (1, 2, 5, 10, 20)[index%5], "expiry_count": (1, 2, 4, 8)[index%4], "anchor_date": ANCHORS[index%5], "family": family, "shape_group": shape_group, "split": split, "provenance": "SYNTHETIC_DESIGN", "half_spread": math.ceil((.005+.195*u[9])*100)/100, "status": "DESIGN_COORDINATES_REQUIRING_VALIDATED_QUOTES_AND_CAMPAIGN"}


def reference_fixture_job(style="EUROPEAN", sigma_path=.2, duration_seconds=300):
    start = datetime(2026, 11, 9, 10, 0, tzinfo=NY)
    end = start+timedelta(seconds=duration_seconds)
    t0, expiry = utc_us(start), utc_us(end)
    legs = (RiskLeg("SYNTHETIC_C100", "CALL", Decimal(100), 1, expiry, .2, exercise_style=style), RiskLeg("SYNTHETIC_C105", "CALL", Decimal(105), -1, expiry, .2, exercise_style=style))
    return ReferenceJob(t0, expiry, expiry, expiry, ((t0, expiry),), 100., .05, 0., sigma_path, legs, Decimal(-200), Decimal(150), snapshot_id="SYNTHETIC_REFERENCE_FIXTURE", calendar_profile_id="SYNTHETIC_QQQ_CONTRACTS_V1")


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise RiskError("RESOURCE_LIMIT", "Synthetic generation checkpoint required")


def _price(right, spot, strike, tau, r, q, sigma, provider, deadline):
    """One bounded reference call. A supplied basis is qualified by its caller."""
    _check_deadline(deadline)
    from ..pricing_fast import european_price
    from .pricing_oracle import american_price
    if provider is not None:
        value = float(provider(right, spot, strike, tau, r, q, sigma))
    elif right == "CALL" and q == 0 and r >= 0:
        value = float(european_price(right, spot, strike, tau, r, q, sigma))
    else:
        seconds = 30 if deadline is None else min(30, max(.001, deadline-time.monotonic()))
        value = american_price(right, spot, strike, tau, r, q, sigma, eps=1e-7, max_seconds=seconds).price
    intrinsic = max(0, (spot-strike) if right == "CALL" else (strike-spot))
    upper = spot if right == "CALL" else strike*math.exp(max(0, -r*tau))
    if not math.isfinite(value) or value < intrinsic-1e-6 or value > upper+1e-6:
        raise RiskError("DESIGN_PRICE_ARBITRAGE", right=right, strike=strike, price=value)
    return value


def _quoted(leg, value, when):
    tick, half = Decimal(str(leg.tick)), Decimal(str(leg.half_spread))
    value = Decimal(str(value))
    bid = (max(Decimal(0), value-half)/tick).to_integral_value(rounding=ROUND_FLOOR)*tick
    ask = ((value+half)/tick).to_integral_value(rounding=ROUND_CEILING)*tick
    return replace(leg, bid=float(bid), ask=float(ask), known_at_utc_us=when)


def _surface_quality(universe):
    """Reject an executable vertical or adjacent butterfly arbitrage."""
    groups = {}
    for leg in universe:
        groups.setdefault((leg.fixing_utc_us, leg.right), []).append(leg)
    for (_, right), group in groups.items():
        group.sort(key=lambda leg: leg.strike)
        for left, right_leg in zip(group, group[1:]):
            if right == "CALL" and right_leg.bid > left.ask+1e-8 or right == "PUT" and left.bid > right_leg.ask+1e-8:
                raise RiskError("DESIGN_QUOTE_ARBITRAGE", contracts=[left.contract_id, right_leg.contract_id])
        for left, middle, right_leg in zip(group, group[1:], group[2:]):
            weight = float((right_leg.strike-middle.strike)/(right_leg.strike-left.strike))
            if middle.bid > weight*left.ask+(1-weight)*right_leg.ask+1e-8:
                raise RiskError("DESIGN_QUOTE_ARBITRAGE", contracts=[left.contract_id, middle.contract_id, right_leg.contract_id])


def _path_sigma(universe, spot, t0, hard, r, q, provider, deadline, *, research=False):
    """P50-R12 weighted median from known quotes, never the hidden generator IV."""
    anchor = min({leg.fixing_utc_us for leg in universe}, key=lambda t: (abs(t-hard), -t))
    tau = (anchor-t0)/YEAR_US
    forward = spot*math.exp((r-q)*tau)
    near = sorted((leg for leg in universe if leg.fixing_utc_us == anchor and abs(math.log(float(leg.strike)/forward)) <= .05),
                  key=lambda leg: (abs(math.log(float(leg.strike)/forward)), leg.strike, leg.right, leg.contract_id))
    valid, rejected = [], []
    for leg in near:
        bump = max(1e-5, 1e-3*max(leg.sigma, .01))
        low = max(0., leg.sigma-bump)
        upper = _price(leg.right, spot, float(leg.strike), tau, r, q, leg.sigma+bump, provider, deadline)
        lower = _price(leg.right, spot, float(leg.strike), tau, r, q, low, provider, deadline)
        vega = (upper-lower)/(leg.sigma+bump-low)
        if not math.isfinite(vega) or vega <= 0:
            rejected.append({"contract_id": leg.contract_id, "reason": "LEG_IV_NOT_IDENTIFIABLE"})
            continue
        half = max((leg.ask-leg.bid)/2, leg.tick/2, 1e-7)
        valid.append({"contract_id": leg.contract_id, "strike": str(leg.strike), "right": leg.right,
                      "sigma": leg.sigma, "vega_per_unit": vega, "half_spread": half,
                      "raw_weight": (vega/half)**2, "iv_basis": "SYNTHETIC_REFERENCE_REPRICED_IV"})
        if len(valid) == 10:
            break
    # The diagnostic call-only domain is explicitly excluded from production.
    if len(valid) < (2 if research else 4) or len({v["strike"] for v in valid}) < 2 or not research and {v["right"] for v in valid} != {"CALL", "PUT"}:
        raise RiskError("MANAGED_ANCHOR_UNAVAILABLE", valid_count=len(valid), rejected=rejected)
    cap = 25*statistics.median(v["raw_weight"] for v in valid)
    total = sum(min(v["raw_weight"], cap) for v in valid)
    for item in valid:
        item["weight"] = min(item["raw_weight"], cap)/total
    ordered = sorted(valid, key=lambda v: (v["sigma"], v["contract_id"]))
    def quantile(fraction):
        cumulative = 0.
        for item in ordered:
            cumulative += item["weight"]
            if cumulative >= fraction:
                return item["sigma"]
        return ordered[-1]["sigma"]
    sigma = quantile(.5)
    spread = quantile(.9)-quantile(.1)
    if spread > max(.03, .30*sigma):
        raise RiskError("MANAGED_ANCHOR_UNAVAILABLE", reason="ATM_IV_DISPERSION", weighted_span=spread)
    return sigma, {"source": "SYNTHETIC_VALIDATED_LOCAL_ATM", "anchor_fixing_utc_us": anchor,
                   "hard_cutoff_mismatch_us": hard-anchor, "weighted_median": sigma,
                   "weighted_90_10_span": spread, "contributors": valid, "rejected": rejected}


def _reports(t0, fixing, intervals, rng, closed):
    def eligible_nearest(target):
        choices = [min(max(target, max(t0, a)), min(fixing-1, b-1)) for a, b in intervals if max(t0, a) < min(fixing, b)]
        if not choices:
            raise RiskError("DESIGN_NO_ELIGIBLE_OBSERVATION")
        return min(choices, key=lambda t: (abs(t-target), t))
    result = {fixing, fixing-1}
    result.update(eligible_nearest(t0+int((fixing-t0)*fraction)) for fraction in (.25, .5, .75))
    # Seeded real subminute precision; round neither to a minute nor a date.
    result.add(eligible_nearest(t0+int((fixing-t0)*float(rng.uniform(.1, .9)))))
    if closed:
        result.add(t0)
    return tuple(sorted(result))


def _template(record, spot, research):
    index, family = record["index"], record["family"]
    center = round(spot)
    width = WIDTHS[index % len(WIDTHS)]
    middle = center+OFFSETS[(index//4) % len(OFFSETS)]
    if index//36 % 4 == 3 and not research:
        middle = max(1, round(spot*(.75 if index%2 else 1.25)))
    if research:
        middle = math.ceil(spot)
    orientation = 1 if research or (index//4) % 2 == 0 else -1
    right = "CALL" if research or (index//8) % 2 == 0 else "PUT"
    if family in ("VERTICAL", "ONGOING_CAMPAIGN"):
        structure = [(right, middle, orientation), (right, middle+width, -orientation)]
        name = "VERTICAL"
    elif family == "CONDOR_BUTTERFLY" and (research or (index//4) % 3 != 2):
        right_width = width if (index//4) % 3 == 0 else WIDTHS[(index+1) % 5]
        # Unequal wings require matching wing slopes to keep both tails flat.
        divisor = math.gcd(width, right_width)
        left_n, right_n = right_width//divisor, width//divisor
        structure = [(right, middle-width, orientation*left_n), (right, middle, -orientation*(left_n+right_n)), (right, middle+right_width, orientation*right_n)]
        name = "EQUAL_BUTTERFLY" if width == right_width else "UNEQUAL_BUTTERFLY"
    elif family == "CONDOR_BUTTERFLY":
        other = WIDTHS[(index+1) % 5]
        structure = [("PUT", middle-width-1, 1), ("PUT", middle-1, -1), ("CALL", middle+1, -1), ("CALL", middle+other+1, 1)]
        name = "IRON_CONDOR"
    else:
        first, second = ((1, 1), (1, 2), (2, 1))[(index//4) % 3]
        structure = [(right, middle-2*width, first), (right, middle-width, -first), (right, middle, second), (right, middle+width, -second)]
        name = "ASYMMETRIC_"+str(first)+":"+str(second)
    quantity = 1 if research else int(record["quantity"])
    structure = [(right, strike, n*quantity) for right, strike, n in structure]
    if any(strike <= 0 or abs(n) > 20 for right, strike, n in structure):
        raise RiskError("DESIGN_TEMPLATE_OUTSIDE_DOMAIN", template=name)
    return structure, name


def _execute_history(job, campaign, universe, context, rng, count, provider, deadline, record):
    from ..policy import generate_actions, next_opportunity
    from ..account_adapter import advance_isolated_account_context
    events = []
    for step in range(count):
        _check_deadline(deadline)
        if campaign.get("target_realized") or campaign.get("stopped_after_exit"):
            break
        eligibility = next_opportunity(job)
        if eligibility >= min(job.fixing_utc_us, int(campaign["active_deadline_utc_us"])):
            raise RiskError("DESIGN_HISTORY_HORIZON_EXHAUSTED", completed_adjustments=len(events))
        # The same supported grammar and exact account checks used by live fast
        # proposals; only the declared exploratory selection policy differs.
        search = generate_actions(job, campaign, universe, dict(context), max_candidates=64, max_raw=2000)
        preferred = ("FULL_CLOSE", "REPOSITION", "PARTIAL_CLOSE", "HEDGE", "ROLL_EXPIRATION") if step == 0 else ("REPOSITION", "HEDGE", "ROLL_EXPIRATION", "PARTIAL_CLOSE", "FULL_CLOSE")
        if record.get("trajectory_policy") == "REPOSITION_FIRST":
            preferred = ("REPOSITION", "PARTIAL_CLOSE", "FULL_CLOSE", "HEDGE", "ROLL_EXPIRATION")
        if record.get("trajectory_policy") == "EXIT_STOP":
            preferred = ("EXIT_STOP", *preferred)
        candidates = [a for a in search["actions"] if a["family"] != "HOLD" and (a["family"] != "EXIT_STOP" or record.get("trajectory_policy") == "EXIT_STOP")]
        if not candidates:
            raise RiskError("DESIGN_NO_FEASIBLE_ADJUSTMENT", rejected=search.get("rejected"), completed_adjustments=len(events))
        action = min(candidates, key=lambda a: (preferred.index(a["family"]) if a["family"] in preferred else 99, a["id"]))
        normal = float(rng.standard_normal())
        dt = (eligibility-job.t0_utc_us)/YEAR_US
        spot = job.x0*math.exp((job.rate_cc-job.dividend_yield_cc-job.sigma_path**2/2)*dt+job.sigma_path*math.sqrt(dt)*normal)
        quotes = []
        for leg in universe:
            if leg.fixing_utc_us <= eligibility:
                continue
            value = _price(leg.right, spot, float(leg.strike), (leg.fixing_utc_us-eligibility)/YEAR_US,
                           job.rate_cc, job.dividend_yield_cc, leg.sigma, provider, deadline)
            quotes.append(_quoted(leg, value, eligibility))
        _surface_quality(quotes)
        outcome = advance_isolated_account_context(context, job, action, quotes, eligibility)
        context = outcome["context"]
        lookup = {leg.contract_id: leg for leg in quotes}
        legs = tuple(replace(lookup[key], quantity=quantity) for key, quantity in sorted(outcome["positions"].items()))
        campaign = {**campaign, **{key: outcome[key] for key in ("gross_cash_usd", "net_cash_usd", "remaining_expense_budget_usd", "economically_completed")}}
        campaign["remaining_actions"] -= int(outcome["filled"])
        campaign["remaining_rolls"] -= int(outcome["filled"] and action["family"] == "ROLL_EXPIRATION")
        campaign["target_realized"] = not legs and Decimal(campaign["net_cash_usd"]) >= job.target_usd
        campaign["stopped_after_exit"] = bool(not legs and outcome["filled"] and action.get("stop_after_fill"))
        fixing = legs[0].fixing_utc_us if legs else job.fixing_utc_us
        job = replace(job, t0_utc_us=eligibility, x0=spot, signed_legs=legs, fixing_utc_us=fixing,
                      last_trading_utc_us=fixing, deadline_utc_us=min(int(campaign["active_deadline_utc_us"]), fixing),
                      entry_cash_usd=Decimal(campaign["gross_cash_usd"]), job_schema="MANAGED_STATE_V1",
                      reporting_times_utc_us=tuple(t for t in job.reporting_times_utc_us if eligibility <= t <= fixing))
        universe = tuple(quotes)
        events.append({"step": step, "normal": normal, "observation_utc_us": eligibility,
                       "spot": spot, "action": action, "status": outcome["status"],
                       "premium_cost_usd": outcome["premium_cost_usd"], "fees_usd": outcome["fees_usd"],
                       "gross_cash_usd": outcome["gross_cash_usd"], "net_cash_usd": outcome["net_cash_usd"],
                       "broker_state_hash": outcome["broker_state_hash"], "order_observations": outcome["order_observations"],
                       "generator_rejections": search.get("rejected", {})})
    campaign["adjustment_events"] = events
    campaign["declared_adjustments"] = count
    campaign["executed_adjustments"] = sum(event["status"] == "FILLED" for event in events)
    campaign["state_class"] = "REALIZED_TARGET" if campaign.get("target_realized") else "FAILED_STOP" if campaign.get("stopped_after_exit") else "FLAT_BELOW_TARGET_REENTRY" if not job.signed_legs else "OPEN_CAMPAIGN"
    return job, campaign, universe, context


def _open_account(job, campaign, universe, rng, calendar, provider, deadline, record):
    """An off-session observation must inherit an earlier eligible real fill."""
    from ..account_adapter import create_isolated_account_context, advance_isolated_account_context
    from ..policy import action_fees, make_action
    from ..payoff import payoff_extrema
    if any(a <= job.t0_utc_us < b for a, b in job.eligible_intervals):
        return job, campaign, create_isolated_account_context(job, campaign, universe)
    observed = datetime.fromtimestamp(job.t0_utc_us/1e6, NY)
    day = observed.date()-timedelta(days=1)
    previous = calendar.session(day)
    while previous.status != "OPEN":
        day -= timedelta(days=1)
        previous = calendar.session(day)
    opening = utc_us(previous.underlying_close-timedelta(minutes=2))
    dt = (job.t0_utc_us-opening)/YEAR_US
    normal = float(rng.standard_normal())
    factor = math.exp((job.rate_cc-job.dividend_yield_cc-job.sigma_path**2/2)*dt+job.sigma_path*math.sqrt(dt)*normal)
    opening_spot = job.x0/factor
    original_quotes = tuple(_quoted(leg, _price(leg.right, opening_spot, float(leg.strike), (leg.fixing_utc_us-opening)/YEAR_US,
        job.rate_cc, job.dividend_yield_cc, leg.sigma, provider, deadline), opening) for leg in universe)
    _surface_quality(original_quotes)
    lookup = {leg.contract_id: leg for leg in original_quotes}
    legs = tuple(replace(lookup[leg.contract_id], quantity=leg.quantity) for leg in job.signed_legs)
    cash = -sum((leg.quantity*leg.multiplier*Decimal(str(leg.ask if leg.quantity > 0 else leg.bid)) for leg in legs), Decimal(0))
    maximum = payoff_extrema(legs, cash)["max_profit"]
    if not maximum.is_finite() or maximum <= 0:
        raise RiskError("DESIGN_NONPOSITIVE_OR_UNBOUNDED_TARGET")
    target = maximum/2
    intervals = ((utc_us(previous.exchange_open), utc_us(previous.underlying_close)), *job.eligible_intervals)
    original = replace(job, t0_utc_us=opening, x0=opening_spot, signed_legs=legs, entry_cash_usd=cash, target_usd=target,
                       eligible_intervals=intervals, parameter_provenance=())
    fees = action_fees(original, legs, current_legs=())
    campaign = {**campaign, "target_usd": str(target), "original_max_profit_usd": str(maximum),
                "gross_cash_usd": str(cash), "net_cash_usd": str(cash-fees),
                "initial_entry_events": [{"premium_cash_usd": str(cash), "fees_usd": str(fees),
                                           "event_at_utc_us": opening, "provenance": "SYNTHETIC_EXECUTED_OPENING"}]}
    context = create_isolated_account_context(original, campaign, original_quotes)
    # No trading decision or quote execution is invented during this interval.
    hold = make_action(original, "HOLD", (), job.t0_utc_us, campaign)
    result = advance_isolated_account_context(context, original, hold, universe, job.t0_utc_us)
    record["closed_period_history"] = {"opening_utc_us": opening, "opening_spot": opening_spot,
        "observation_utc_us": job.t0_utc_us, "observed_spot": job.x0, "normal": normal,
        "construction": "SYNTHETIC_PRIOR_SPOT_BACKSOLVED_FOR_SAMPLED_CURRENT_SPOT", "execution_during_closed_period": False}
    return replace(job, entry_cash_usd=cash, target_usd=target), campaign, result["context"]


def materialize_case(record, *, research=False, price_provider=None, numerical_settings=None,deadline=None):
    """Turn persisted coordinates into actual quoted positions and original cash.

    Research mode is a named call-only, q=0, short-horizon diagnostic domain.
    Its labels can exercise every pipeline stage but never qualify the broad
    32,768-state production domain by silently substituting easy examples.
    """
    if record.get('heldout_edge_family'):
        from .edge_design import materialize_heldout_case
        return materialize_heldout_case(record,research=research,price_provider=price_provider,
                                      numerical_settings=numerical_settings,deadline=deadline)
    from ..payoff import payoff_extrema
    from ..policy import action_fees
    from ...replay.calendar import CalendarService
    _check_deadline(deadline)
    record = dict(record)
    record.setdefault("trajectory_policy", ("CLOSE_REENTER", "REPOSITION_FIRST", "EXIT_STOP", "CLOSE_REENTER")[(int(record["index"])//16) % 4])
    calendar = CalendarService()
    index, u = record["index"], record["coordinates"]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(SEED, spawn_key=(1, int(record.get("cycle", 0)), int(index)))))
    record["generator_state_before"] = canonical(rng.bit_generator.state)
    anchor = datetime.fromisoformat(record["anchor_date"]).replace(hour=10, tzinfo=NY)
    session = calendar.session(anchor.date())
    if session.status != "OPEN":
        session = calendar.next_session(anchor.date())
    # The final continuous coordinate spans eligible intraday observations,
    # including near-close/high-gamma cases; dates alone must not collapse all
    # production examples to the old fixed10:00 six-hour minimum lifetime.
    if not research:
        eligible_seconds = (session.underlying_close-session.exchange_open).total_seconds()
        anchor = session.exchange_open+timedelta(seconds=float(u[11])*(eligible_seconds-60))
    if index%10 == 0 and not research:
        anchor = anchor.replace(hour=8, minute=0, second=0, microsecond=0)
    desired = anchor+timedelta(seconds=record["desired_seconds"])
    if research:
        # True actual fixing remains explicit; diagnostics use synthetic expiries.
        anchor = session.underlying_close-timedelta(minutes=5+index%5)
        desired = session.underlying_close-timedelta(seconds=1)
    fixing_day = desired.date()
    expiry_session = calendar.session(fixing_day)
    while expiry_session.status != "OPEN" or expiry_session.underlying_close <= desired:
        expiry_session = calendar.next_session(expiry_session.local_date)
    expiry = expiry_session.underlying_close
    if (expiry-anchor).total_seconds() > 30*86400:
        raise RiskError("DESIGN_HORIZON_EXCEEDED")
    t0, fixing = utc_us(anchor), utc_us(expiry)
    hard = t0+30*86400*1_000_000 if not research else fixing-1
    intervals = tuple((utc_us(s.exchange_open), utc_us(s.underlying_close)) for s in calendar.intervals("QQQ", anchor.date(), datetime.fromtimestamp(hard/1e6, NY).date()) if s.status == "OPEN")
    spot = record["spot"]
    sigma = record["sigma_anchor"]
    r, q = record["r"], 0. if research else record["q"]
    center = round(spot)
    family = record["family"]
    structure, template = _template(record, spot, research)
    record["template"] = template
    tau_anchor = (fixing-t0)/YEAR_US
    half = record["half_spread"] if not research else .01
    by_identity = {}
    candidates = {fixing}
    for offset in EXPIRY_OFFSETS:
        candidate_session = calendar.session(anchor.date()+timedelta(days=offset))
        if candidate_session.status != "OPEN":
            candidate_session = calendar.next_session(candidate_session.local_date)
        value = utc_us(candidate_session.underlying_close)
        if t0 < value <= hard:
            candidates.add(value)
    requested_count = 1 if research else int(record["expiry_count"])
    expiries = [fixing]+[t for t in sorted(candidates-{fixing}, key=lambda t: (abs(t-hard), -t))][:requested_count-1]
    required = {(fixing, right, strike) for right, strike, _ in structure}
    strikes = {center+offset for offset in range(-32, 33) if center+offset > 0}
    strikes.update(max(1, round(spot*fraction)) for fraction in (.5, .75, 1.25, 1.5))
    # Fractional strikes require an explicit synthetic product adapter field.
    for increment in record.get("minimum_strike_increments", ()):
        value = Decimal(str(increment))
        if value <= 0:
            raise RiskError("INVALID_SYNTHETIC_STRIKE_INCREMENT")
        strikes.update(float(Decimal(center)+value*n) for n in range(-4, 5) if Decimal(center)+value*n > 0)
    ladder = {(when, right, strike) for when in expiries for strike in strikes for right in ("PUT", "CALL")}
    if research:
        ladder = {(fixing, "CALL", center+i) for i in range(-4, 5)}
    expiry_rank = {when: rank for rank, when in enumerate(sorted(expiries, key=lambda t: (abs(t-hard), -t)))}
    ranked = sorted(ladder-required, key=lambda item: (abs(math.log(item[2]/spot)), item[2], expiry_rank[item[0]], item[1]))
    identities = sorted(required)+ranked[:512-len(required)]
    excluded = ranked[512-len(required):]
    record["universe_admission"] = {"requested_expiries": requested_count, "candidate_fixings_utc_us": sorted(candidates),
        "selected_fixings_utc_us": sorted(expiries), "contract_cap": 512,
        "rule": "CURRENT_CONTRACTS_THEN_MONEYNESS_STRIKE_EXPIRY_DISTANCE_RIGHT",
        "excluded_contracts": [{"fixing_utc_us": t, "right": right, "strike": str(strike)} for t, right, strike in excluded],
        "excluded_expiries_utc_us": sorted(candidates-set(expiries))}
    for when, right, strike in identities:
        _check_deadline(deadline)
        k = math.log(strike/spot)
        tau = (when-t0)/YEAR_US
        vol = sigma if research else sigma+record["skew"]*k+record["curvature"]*k*k+record["term_slope"]*(math.sqrt(tau)-math.sqrt(tau_anchor))
        if not .03 <= vol <= 2:
            raise RiskError("DESIGN_SURFACE_OUTSIDE_DOMAIN", fixing_utc_us=when, strike=strike, sigma=vol)
        value = _price(right, spot, strike, tau, r, q, vol, price_provider, deadline)
        identifier = f"SYNTHETIC:{when}:{right}:{Decimal(str(strike)).normalize()}"
        leg = RiskLeg(identifier, right, Decimal(str(strike)), 1, when, vol, half_spread=half, last_trading_utc_us=when)
        leg = _quoted(leg, value, t0)
        by_identity[when, right, strike] = replace(leg, half_spread=(leg.ask-leg.bid)/2)
    universe = tuple(by_identity.values())
    _surface_quality(universe)
    path_sigma, path_evidence = _path_sigma(universe, spot, t0, hard, r, q, price_provider, deadline, research=research)
    record["path_iv_selection"] = path_evidence
    record["actual_remaining_seconds"] = (fixing-t0)/1e6
    legs = tuple(replace(by_identity[fixing, right, strike], quantity=n) for right, strike, n in structure)
    cash = -sum((x.quantity*x.multiplier*Decimal(str(x.ask if x.quantity > 0 else x.bid)) for x in legs), Decimal(0))
    extrema = payoff_extrema(legs, cash)
    if not extrema["max_profit"].is_finite() or extrema["max_profit"] <= 0 or not extrema["max_loss"].is_finite():
        raise RiskError("DESIGN_NONPOSITIVE_OR_UNBOUNDED_TARGET")
    target = extrema["max_profit"]/2
    reports = _reports(t0, fixing, intervals, rng, index%10 == 0 and not research)
    settings = numerical_settings or NumericalSettings()
    job = ReferenceJob(t0, fixing-1, fixing, fixing, intervals, spot, r, q, path_sigma, legs, cash, target, snapshot_id=f"SYNTHETIC_DESIGN_{index}", calendar_profile_id="SYNTHETIC_QQQ_CONTRACTS_V1", reporting_times_utc_us=reports, numerical_settings=settings,
                       parameter_provenance=({"source": "SYNTHETIC_DESIGN", "known_at_utc_us": t0, "path_iv_selection": path_evidence},))
    fees = action_fees(job, legs, current_legs=())
    campaign = {"campaign_id": "SYNTHETIC_DESIGN_"+str(record.get("cycle", 0))+"_"+str(index), "target_usd": str(target), "original_max_profit_usd": str(extrema["max_profit"]), "gross_cash_usd": str(cash), "net_cash_usd": str(cash-fees), "remaining_expense_budget_usd": "100", "hard_cutoff_utc_us": hard,
                "initial_deadline_utc_us": fixing-1, "active_deadline_utc_us": fixing-1, "remaining_actions": 12, "remaining_rolls": 3, "operational_model": "SIMULTANEOUS_DELIVERY_HEDGE_V1", "protection_instruction_recorded": True,
                "initial_entry_events": [{"premium_cash_usd": str(cash), "fees_usd": str(fees), "provenance": "SYNTHETIC_EXECUTED_OPENING"}], "target_realized": False,
                "requested_deadlines_utc_us": sorted(set((*reports, hard))), "declared_adjustments": 0, "executed_adjustments": 0, "adjustment_events": [], "state_class": "OPEN_CAMPAIGN"}
    context = None
    if not research:
        job, campaign, context = _open_account(job, campaign, universe, rng, calendar, price_provider, deadline, record)
        original_hash = stable_hash(context["broker_state"])
        campaign["initial_broker_state_hash"] = original_hash
        if family == "ONGOING_CAMPAIGN":
            count = int(record.get("adjustment_count", (index//4) % 4))
            if count not in range(4):
                raise RiskError("DESIGN_ADJUSTMENT_COUNT_OUTSIDE_DOMAIN")
            job, campaign, universe, context = _execute_history(job, campaign, universe, context, rng, count, price_provider, deadline, record)
            if campaign["adjustment_events"]:
                # A new forecast at the generated present re-normalizes its
                # observable ATM state. The earlier trajectory's law is frozen
                # and is separately retained, so no old draw is reinterpreted.
                record["history_path_iv_selection"] = path_evidence
                current_sigma, current_evidence = _path_sigma(universe, job.x0, job.t0_utc_us, hard,
                    r, q, price_provider, deadline)
                record["path_iv_selection"] = current_evidence
                job = replace(job, sigma_path=current_sigma, parameter_provenance=({"source": "SYNTHETIC_DESIGN",
                    "known_at_utc_us": job.t0_utc_us, "path_iv_selection": current_evidence},))
    else:
        campaign["initial_entry_events"][0]["provenance"] = "SYNTHETIC_MODELED_OPENING_ACCOUNT_UNQUALIFIED"
    record["generator_state_after"] = canonical(rng.bit_generator.state)
    record["status"] = "MATERIALIZED_SYNTHETIC_CASE"
    record["quote_provenance"] = "SYNTHETIC_DESIGN"
    return {"job": job, "campaign": campaign, "universe": universe, "record": record, "account_context": context,
            "scope": "RESEARCH_Q0_AMERICAN_CALLS" if research else "FULL_DESIGN", "generator_limitations": ["RESEARCH_DOMAIN_EXCLUDES_PRODUCTION_ACCOUNT_TRAJECTORIES"] if research else []}
