"""Bounded deterministic actions and separate evaluated-policy probabilities."""
from __future__ import annotations
import math
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from functools import reduce
import numpy as np
from .contracts import RiskError, RiskLeg, canonical, stable_hash, NY, YEAR_US
from .features import managed_state_features,position_token,coverage_check
from .feature_layout import (ACTION_FAMILIES,ACTION_FIELDS,ORDER_TYPES,TIME_IN_FORCE,MANAGED_FEATURE_COUNT,MANAGED_STATE_COUNT)
from .hold import network
from .verify import exact_poststate, risk_certificate

ACTION_ORDER = ACTION_FAMILIES


def next_opportunity(job, after=None):
    now = job.t0_utc_us if after is None else after
    for start, end in job.eligible_intervals:
        candidate = max(start, (now//60_000_000+1)*60_000_000)
        if now < candidate < min(end, job.last_trading_utc_us):
            return candidate
    raise RiskError("NO_FUTURE_ELIGIBLE_EXECUTION")


def action_fees(job, changes, current_legs=None, order_id="FORECAST_ORDER"):
    from ..broker.fees import FeeEngine
    from ..broker.models import Contract
    engine = FeeEngine()
    positions = {x.contract_id: x.quantity for x in (current_legs or job.signed_legs)}
    trade_date = datetime.fromtimestamp(job.t0_utc_us/1e6, timezone.utc).astimezone(NY).date()
    total = Decimal(0)
    for leg in changes:
        old = positions.get(leg.contract_id, 0)
        closing = min(abs(old), abs(leg.quantity)) if old*leg.quantity < 0 else 0
        opening = abs(leg.quantity)-closing
        if leg.bid is None or leg.ask is None:
            raise RiskError("MISSING_EXECUTABLE_QUOTE", contract_id=leg.contract_id)
        price = Decimal(str(leg.ask if leg.quantity > 0 else leg.bid))
        expiry = datetime.fromtimestamp(leg.fixing_utc_us/1e6, timezone.utc).astimezone(NY).date()
        contract = Contract(symbol="QQQ", right=leg.right[0], strike=leg.strike, expiry=expiry, multiplier=leg.multiplier)
        components = engine.assess_option(order_id, contract, leg.quantity, price, opening, trade_date)
        total += sum(components.values(), Decimal(0))
    return total


def make_action(job, family, changes, eligibility, campaign):
    changes = tuple(changes)
    cost = Decimal(0)
    for leg in changes:
        side = leg.ask if leg.quantity > 0 else leg.bid
        if side is None:
            raise RiskError("MISSING_EXECUTABLE_QUOTE", contract_id=leg.contract_id)
        cost += leg.quantity*leg.multiplier*Decimal(str(side))
    fees = action_fees(job, changes) if changes else Decimal(0)
    action = {"family": family, "legs": canonical(changes), "premium_cost_usd": str(cost), "limit_cost_usd": str(cost), "fees_usd": str(fees), "max_total_fees_usd": "50.00", "eligible_utc_us": eligibility, "stop_after_fill": family == "EXIT_STOP", "execution_profile": "SIM_CONDITIONAL_RISK_CHECK_V1", "target_usd": str(campaign.get("target_usd", job.target_usd))}
    action["id"] = stable_hash(action)
    return action


def generate_actions(job, campaign, universe=(), limits=None, max_candidates=2000, max_raw=40_000,*,offline_reference=False):
    """Stream the M07 grammar into bounded, individually checked family pools."""
    limits = limits or {}
    universe = tuple(universe)
    if len(universe) > 512 or len({x.fixing_utc_us for x in universe}) > 8:
        raise RiskError("UNIVERSE_OUTSIDE_MODEL_DOMAIN")
    if len({x.contract_id for x in universe}) != len(universe):
        raise RiskError("UNIVERSE_OUTSIDE_MODEL_DOMAIN", "Duplicate contract identity")
    if "hard_cutoff_utc_us" not in campaign:
        raise RiskError("NEEDS_CAMPAIGN_CUTOFF")
    ceiling_candidates,ceiling_raw=(4000,80000) if offline_reference else (2000,40000)
    max_candidates, max_raw = min(ceiling_candidates, int(max_candidates)), min(ceiling_raw, int(max_raw))
    if max_candidates < 1 or max_raw < 1:
        raise RiskError("INVALID_CANDIDATE_BUDGET")
    eligibility = next_opportunity(job)
    hard = campaign["hard_cutoff_utc_us"]
    if eligibility > min(hard, campaign.get("active_deadline_utc_us", job.deadline_utc_us)):
        raise RiskError("NO_FUTURE_ELIGIBLE_EXECUTION")
    actions, ids, attempted, rejected = [], set(), 0, {}
    counters = {name: 0 for name in ACTION_ORDER}
    truncations, duplicate_count = {}, 0
    close = tuple(replace(leg, quantity=-leg.quantity) for leg in job.signed_legs)
    can_act = campaign.get("remaining_actions", 12) > 0
    source_state_hash = stable_hash({"job": job.hash, "campaign": campaign})

    def reject(reason):
        rejected[reason] = rejected.get(reason, 0) + 1

    def net_transactions(legs):
        # Match the broker's atomic package netting, including shared butterfly
        # strikes and unchanged contracts in a complete reposition.
        by_id = {}
        for leg in legs:
            old = by_id.get(leg.contract_id)
            if old is not None and (old.right, old.strike, old.fixing_utc_us, old.multiplier, old.deliverable_id) != (leg.right, leg.strike, leg.fixing_utc_us, leg.multiplier, leg.deliverable_id):
                raise RiskError("CONTRACT_IDENTITY_COLLISION")
            by_id[leg.contract_id] = replace(leg, quantity=leg.quantity+(old.quantity if old else 0)) if old is None or leg.quantity+old.quantity else None
            if by_id[leg.contract_id] is None:
                del by_id[leg.contract_id]
        return tuple(by_id.values())

    def offer(family, raw_legs):
        nonlocal attempted, duplicate_count
        attempted += 1
        try:
            legs = net_transactions(raw_legs)
            if family != "HOLD" and not legs:
                reject("EMPTY_NET_TRANSACTION")
                return
            action = make_action(job, family, legs, eligibility, campaign)
            if any(leg.fixing_utc_us <= eligibility for leg in legs):
                raise RiskError("EXPIRY_BEFORE_ELIGIBILITY")
            if Decimal(action["fees_usd"]) > Decimal(action["max_total_fees_usd"]):
                raise RiskError("ORDER_FEE_AUTHORIZATION_EXCEEDED")
            postlegs, updated = exact_poststate(job.signed_legs, campaign, legs, action["premium_cost_usd"], action["fees_usd"])
            if family != "HOLD":
                certificate = risk_certificate(postlegs, updated, limits)
                if certificate["status"] != "VERIFIED_WITHIN_DECLARED_MODEL":
                    raise RiskError("RISK_CERTIFICATE_FAILED", certificate=certificate)
                action["certificate"] = certificate
            action["resulting_inventory"] = canonical(postlegs)
            action["resulting_cash_usd"] = updated["net_cash_usd"]
            action["resulting_gross_cash_usd"] = updated["gross_cash_usd"]
            action["resulting_expense_budget_usd"] = updated["remaining_expense_budget_usd"]
            action["resulting_action_count_remaining"] = campaign.get("remaining_actions", 12)-int(family != "HOLD")
            action["resulting_roll_count_remaining"] = campaign.get("remaining_rolls", 3)-int(family == "ROLL_EXPIRATION")
            action["reentry_permitted"] = family != "EXIT_STOP"
            action["source_state_hash"] = source_state_hash
            action["id"] = stable_hash({key: value for key, value in action.items() if key != "id"})
            if action["id"] in ids:
                duplicate_count += 1
                return
            if not limits.get("explicit_unqualified_research_account", False):
                from .account_adapter import account_proposal_check
                check = account_proposal_check(job, campaign, action, limits)
                if check["status"] != "OK":
                    raise RiskError(check.get("reason", "ACCOUNT_CHECK_FAILED"), account_check=check)
                action["account_check"] = check
            else:
                action["account_check"] = {"status": "RESEARCH_ONLY", "reason": "FULL_ACCOUNT_STATE_NOT_QUALIFIED"}
            actions.append(action)
            ids.add(action["id"])
            counters[family] += 1
        except RiskError as exc:
            reject(exc.code)

    def compulsory():
        yield "HOLD", ()
        if not can_act:
            return
        if close:
            yield "FULL_CLOSE", close
            yield "EXIT_STOP", close
        divisor = reduce(math.gcd, (abs(leg.quantity) for leg in job.signed_legs), 0)
        for quantity in range(1, divisor):
            yield "PARTIAL_CLOSE", tuple(replace(leg, quantity=-(leg.quantity//divisor)*quantity) for leg in job.signed_legs)
        # Pair identities come from the same delivery certificate used by the
        # final risk check. Reducing one pair can differ from a whole-position
        # gcd reduction even when the gcd is one.
        current = {leg.contract_id: leg for leg in job.signed_legs}
        certificate = risk_certificate(job.signed_legs, campaign, limits)
        seen = set()
        for pair in certificate.get("pairs", ()):
            key = pair["short"], pair["long"]
            if key in seen:
                continue
            seen.add(key)
            amounts = {key[0]: 1, key[1]: -1}
            yield "PARTIAL_CLOSE", tuple(replace(leg, quantity=amounts[leg.contract_id]) for leg in job.signed_legs if leg.contract_id in amounts)

    admitted = []
    for leg in universe:
        if leg.known_at_utc_us is None:
            reject("UNIVERSE_AVAILABILITY_UNKNOWN")
        elif leg.known_at_utc_us > job.t0_utc_us:
            reject("FUTURE_UNIVERSE_RECORD")
        elif leg.bid is None or leg.ask is None or leg.fixing_utc_us <= eligibility:
            reject("UNIVERSE_CONTRACT_UNAVAILABLE")
        else:
            admitted.append(leg)
    expiries = sorted({leg.fixing_utc_us for leg in admitted}, key=lambda when: (abs(when-job.fixing_utc_us), when))
    components, local_contracts = {}, {}
    for expiry in expiries:
        pool, local = [], []
        capped = False
        for right in ("PUT", "CALL"):
            contracts = sorted((leg for leg in admitted if leg.fixing_utc_us == expiry and leg.right == right), key=lambda leg: (leg.strike, leg.contract_id))
            if not contracts:
                continue
            anchors = [leg.strike for leg in job.signed_legs if leg.fixing_utc_us == expiry and leg.right == right]
            anchors.append(min(contracts, key=lambda leg: (abs(float(leg.strike)-job.x0), leg.strike, leg.contract_id)).strike)
            indices = set()
            for anchor in anchors:
                at = min(range(len(contracts)), key=lambda index: (abs(contracts[index].strike-anchor), contracts[index].strike))
                indices.update(range(max(0, at-8), min(len(contracts), at+9)))
            neighborhood = [contracts[index] for index in sorted(indices)]
            local.extend(neighborhood)
            for separation in range(1, len(neighborhood)):
                for lower in range(len(neighborhood)-separation):
                    for orientation in (-1, 1):
                        if len(pool) == 1024:
                            capped = True
                            break
                        pool.append((replace(neighborhood[lower], quantity=orientation), replace(neighborhood[lower+separation], quantity=-orientation)))
                    if capped:
                        break
                if capped:
                    break
            if capped:
                break
        if capped:
            truncations["COMPONENT_POOL_BUDGET_REACHED"] = truncations.get("COMPONENT_POOL_BUDGET_REACHED", 0)+1
        components[expiry], local_contracts[expiry] = tuple(pool), tuple(local)

    def diagonal_pairs(left, right, same=False):
        for diagonal in range(len(left)+len(right)-1):
            for i in range(max(0, diagonal-len(right)+1), min(len(left), diagonal+1)):
                j = diagonal-i
                if not same or i < j:
                    yield left[i], right[j]

    def bounded_combinations(expiry):
        pool = components[expiry]
        puts = tuple(pair for pair in pool if pair[0].right == "PUT")[:64]
        calls = tuple(pair for pair in pool if pair[0].right == "CALL")[:64]
        for put, call in diagonal_pairs(puts, calls):
            if max(leg.strike for leg in put) < min(leg.strike for leg in call):
                yield put+call, None
            else:
                yield (), "FOUR_LEG_COMPONENT_OVERLAP"
        for group in (puts, calls):
            for first, second in diagonal_pairs(group, group, same=True):
                combined = net_transactions(first+second)
                ordered = sorted(combined, key=lambda leg: leg.strike)
                if len(ordered) == 3 and [leg.quantity for leg in ordered] in ([1, -2, 1], [-1, 2, -1]):
                    yield tuple(ordered), None
                else:
                    yield (), "NOT_A_SUPPORTED_BUTTERFLY_COMBINATION"

    def family_stream(family):
        if not can_act:
            return
        for expiry in expiries:
            if family == "ROLL_EXPIRATION":
                if not job.fixing_utc_us < expiry <= hard or campaign.get("remaining_rolls", 3) <= 0:
                    continue
            elif expiry != job.fixing_utc_us:
                continue
            prefix = () if family == "HEDGE" else close
            for pair in components[expiry]:
                for quantity in range(1, 21):
                    yield family, prefix+tuple(replace(leg, quantity=leg.quantity*quantity) for leg in pair)
            if family == "HEDGE":
                for leg in local_contracts[expiry]:
                    for quantity in range(1, 21):
                        yield family, (replace(leg, quantity=quantity),)
            else:
                for structure, reason in bounded_combinations(expiry):
                    if reason:
                        yield family, (), reason
                        continue
                    for quantity in range(1, 21):
                        yield family, prefix+tuple(replace(leg, quantity=leg.quantity*quantity) for leg in structure)

    names = ("COMPULSORY", "HEDGE", "REPOSITION", "ROLL_EXPIRATION")
    streams = {"COMPULSORY": iter(compulsory()), **{name: iter(family_stream(name)) for name in names[1:]}}
    exhausted, accepted = set(), dict.fromkeys(names, 0)
    quotas = {"COMPULSORY": 32, "HEDGE": 384, "REPOSITION": 768, "ROLL_EXPIRATION": 768}

    def consume(name, target):
        nonlocal attempted
        while name not in exhausted and accepted[name] < target and attempted < max_raw and len(actions) < max_candidates:
            try:
                family, legs, *reason = next(streams[name])
            except StopIteration:
                exhausted.add(name)
                break
            before = len(actions)
            if reason:
                attempted += 1
                reject(reason[0])
            else:
                offer(family, legs)
            accepted[name] += len(actions)-before

    for name in names:
        consume(name, quotas[name])
    # Resume suspended iterators, so unused roll/reposition allocations can
    # increase hedges without rebuilding or rescanning the previous candidates.
    for name in names:
        consume(name, accepted[name]+max_candidates-len(actions))
    if attempted >= max_raw and len(exhausted) != len(names):
        truncations["RAW_CANDIDATE_BUDGET_REACHED"] = 1
    if len(actions) >= max_candidates and len(exhausted) != len(names):
        truncations["ACCEPTED_CANDIDATE_BUDGET_REACHED"] = 1

    def retained_order(action):
        legs = tuple(RiskLeg.from_dict(item) for item in action["legs"])
        distance = sum(min((abs(leg.strike-held.strike) for held in job.signed_legs if held.right == leg.right), default=abs(leg.strike-Decimal(str(job.x0)))) for leg in legs)
        adverse = Decimal(action.get("certificate", {}).get("operational_loss_bound_usd", "0"))
        details = tuple((leg.fixing_utc_us, 0 if leg.right == "PUT" else 1, leg.strike, leg.quantity, leg.contract_id) for leg in legs)
        return (ACTION_ORDER.index(action["family"]), Decimal(action["premium_cost_usd"])+Decimal(action["fees_usd"]), adverse, distance, details, action["id"])

    actions.sort(key=retained_order)
    return {"actions": actions, "family_counts": counters, "raw_attempts": attempted, "rejections": rejected,
            "duplicate_count": duplicate_count, "truncations": truncations, "truncated": bool(truncations),
            "family_allocations": accepted, "candidate_cap": max_candidates, "raw_cap": max_raw,"budget_profile":"OFFLINE_CONTROL_REFERENCE_2X_V1" if offline_reference else "ONLINE_BOUNDED_P50_M07",
            "search_label": "BOUNDED_SEARCH_NOT_GLOBAL_OPTIMUM"}


def action_features(job, campaign, action, pack, universe=(),*,state_features=None,universe_encoding=None):
    from .universe import universe_features
    state = managed_state_features(job,pack,campaign) if state_features is None else state_features
    if np.shape(state)!=(MANAGED_STATE_COUNT,):raise RiskError('FEATURE_SCHEMA_MISMATCH')
    family = np.zeros(len(ACTION_ORDER))
    family[ACTION_ORDER.index(action["family"])] = 1
    scale = max(float(job.target_usd),float(campaign.get('loss_cap_usd',1000)),1.)
    numbers = np.array([float(action['premium_cost_usd'])/scale,float(action['fees_usd'])/scale,
        (action['eligible_utc_us']-job.t0_utc_us)/YEAR_US,
        float(action.get('limit_cost_usd',action['premium_cost_usd']))/scale,
        float(action.get('max_total_fees_usd',50))/scale,
        float(action.get('target_usd',campaign.get('target_usd',job.target_usd)))/scale,
        float(bool(action.get('stop_after_fill',False)))])
    order_type=action.get('order_type','PROTECTED_LIMIT');tif=action.get('tif','DAY')
    if order_type not in ORDER_TYPES or tif not in TIME_IN_FORCE:raise RiskError('EXECUTION_PROFILE_UNSUPPORTED')
    execution=np.array([float(order_type==value) for value in ORDER_TYPES]+[float(tif==value) for value in TIME_IN_FORCE])
    tokens = np.zeros((16, 12))
    if len(action["legs"]) > 16:
        raise RiskError("ACTION_OUTSIDE_MODEL_DOMAIN")
    for i, raw in enumerate(action["legs"]):
        leg = RiskLeg.from_dict(raw)
        tokens[i] = position_token(job,leg)
    opportunities=universe_features(job,universe,pack) if universe_encoding is None else universe_encoding
    result=np.concatenate((state,family,numbers,execution,tokens.ravel(),opportunities))
    if result.shape!=(MANAGED_FEATURE_COUNT,) or not np.all(np.isfinite(result)):raise RiskError('INVALID_FEATURE_VALUE')
    return result


def evaluate_policy_action(job, campaign, action, pack, universe=()):
    result = {"q_policy": None, "event": "REALIZED_NET_TARGET_AFTER_FIRST_ACTION_THEN_FROZEN_POLICY", "policy_id": getattr(pack, "manifest", {}).get("policy_id"), "action_id": action.get("id"), "hold_p50_is_different_event": True}
    try:
        if campaign.get("unresolved_stock_obligation"):
            raise RiskError("LIFECYCLE_MODEL_UNSUPPORTED")
        if campaign.get("economically_completed") and not job.signed_legs:
            if Decimal(str(campaign["net_cash_usd"])) >= Decimal(str(campaign["target_usd"])):
                return {**result,"status":"OK","q_policy":1.,"exact":True}
            if campaign.get("stopped") or job.t0_utc_us>=campaign.get("active_deadline_utc_us",campaign.get("hard_cutoff_utc_us",job.deadline_utc_us)):
                return {**result,"status":"OK","q_policy":0.,"exact":True}
        if pack is None:
            raise RiskError("PACK_MISSING")
        pack.require_qualified("managed")
        features = action_features(job, campaign, action, pack,universe)
        from .qualification import regime_key
        scaled, coverage = coverage_check(features, pack, "managed",regime_key(job,campaign,features=features))
        raw = float(network(pack.arrays, "q_eval", scaled)[0])
        p = math.exp(-float(np.logaddexp(0, -raw)))
        error = coverage["p99_absolute_error"]+coverage.get("reference_uncertainty", 0)
        result.update(status="OK", q_policy=p, empirical_error_range=[max(0, p-error), min(1, p+error)], error=error, policy_id=pack.manifest["policy_id"])
    except RiskError as exc:
        result.update(exc.as_dict())
    return result


def evaluate_policy_actions(job,campaign,actions,pack,universe=()):
    """Reuse current bases and run numeric heads in bounded 256-action batches."""
    from .universe import universe_features
    if not actions:
        return []
    if campaign.get("economically_completed") and not job.signed_legs:
        return [evaluate_policy_action(job,campaign,action,pack,universe) for action in actions]
    outcomes=[{"q_policy":None,"event":"REALIZED_NET_TARGET_AFTER_FIRST_ACTION_THEN_FROZEN_POLICY","policy_id":getattr(pack,"manifest",{}).get("policy_id"),"action_id":action.get("id"),"hold_p50_is_different_event":True} for action in actions]
    try:
        if campaign.get("unresolved_stock_obligation"):
            raise RiskError("LIFECYCLE_MODEL_UNSUPPORTED")
        if pack is None:
            raise RiskError("PACK_MISSING")
        pack.require_qualified("managed")
        state=managed_state_features(job,pack,campaign)
        opportunities=universe_features(job,universe,pack)
        from .qualification import regime_key
        regime=regime_key(job,campaign,features=state)
        for start in range(0,len(actions),256):
            batch,indices,errors=[],[],[]
            for index in range(start,min(start+256,len(actions))):
                try:
                    features=action_features(job,campaign,actions[index],pack,universe,state_features=state,universe_encoding=opportunities)
                    scaled,coverage=coverage_check(features,pack,"managed",regime)
                    batch.append(scaled);indices.append(index)
                    errors.append(coverage["p99_absolute_error"]+coverage.get("reference_uncertainty",0))
                except RiskError as exc:
                    outcomes[index].update(exc.as_dict())
            if batch:
                probabilities=np.exp(-np.logaddexp(0,-network(pack.arrays,"q_eval",np.asarray(batch,dtype=np.float64))[:,0]))
                for index,p,error in zip(indices,probabilities,errors):
                    outcomes[index].update(status="OK",q_policy=float(p),empirical_error_range=[max(0.,float(p)-error),min(1.,float(p)+error)],error=error,policy_id=pack.manifest["policy_id"])
    except RiskError as exc:
        for outcome in outcomes:
            outcome.update(exc.as_dict())
    return outcomes


def mandatory_last_opportunity(job, campaign, actions):
    """Choose feasible liquidation when the next event is the final decision."""
    actions = tuple(actions)
    if not actions:
        raise RiskError("NO_FEASIBLE_SUPPORTED_ACTION")
    effective = min(int(campaign.get("active_deadline_utc_us", job.deadline_utc_us)), int(campaign["hard_cutoff_utc_us"]))
    when = min(int(action["eligible_utc_us"]) for action in actions)
    try:
        later = next_opportunity(job, after=when)
    except RiskError as exc:
        if exc.code != "NO_FUTURE_ELIGIBLE_EXECUTION":
            raise
        later = effective+1
    if later <= effective and campaign.get("remaining_actions",12)>1:
        return None
    family = "FULL_CLOSE" if job.signed_legs else "HOLD"
    choices = [action for action in actions if action["family"] == family]
    if not choices:
        raise RiskError("MANDATORY_LIQUIDATION_UNAVAILABLE")
    return min(choices, key=lambda action: (Decimal(str(action.get("certificate", {}).get("operational_loss_bound_usd",0))),
               Decimal(action["premium_cost_usd"]), sum(abs(int(leg["quantity"])) for leg in action["legs"]), action["id"]))


def select_policy_action(job, campaign, universe, limits, pack, *, error=0., max_candidates=2000):
    """The frozen Q_score policy, with its own encoder and declared tie rule.

    Scores choose actions; only the separately fitted Q_eval is a probability.
    This fast selector has no dependency on offline integration or training.
    """
    from types import SimpleNamespace
    if pack is None:
        raise RiskError("PACK_MISSING")
    pack.require_qualified("managed")
    names = ("universe_w0", "universe_b0", "universe_w1", "universe_b1")
    required = ("q_score_mean", "q_score_std", *("q_score_"+name for name in names))
    if any(key not in pack.arrays for key in required):
        raise RiskError("FROZEN_POLICY_ARRAYS_MISSING")
    arrays = dict(pack.arrays)
    arrays.update({name: pack.arrays["q_score_"+name] for name in names})
    score_pack = SimpleNamespace(arrays=arrays, price_basis=pack.price_basis)
    search = generate_actions(job, campaign, universe, limits, max_candidates=max_candidates)
    mandatory = mandatory_last_opportunity(job,campaign,search['actions'])
    if mandatory is not None:
        return {'action':mandatory,'selection_score':None,'policy_id':pack.manifest.get('policy_id'),
                'selector':'MANDATORY_LAST_OPPORTUNITY_LIQUIDATION','tie_error':error,
                'search':{key:value for key,value in search.items() if key!='actions'}}
    from .universe import universe_features
    state_features=managed_state_features(job,score_pack,campaign)
    universe_encoding=universe_features(job,universe,score_pack)
    choices = []
    for action in search["actions"]:
        raw = action_features(job, campaign, action, score_pack, universe,state_features=state_features,universe_encoding=universe_encoding)
        scaled = (raw-pack.arrays["q_score_mean"])/pack.arrays["q_score_std"]
        if not np.all(np.isfinite(scaled)):
            raise RiskError("INVALID_FROZEN_POLICY_FEATURE")
        score = float(np.exp(-np.logaddexp(0, -network(pack.arrays, "q_score", scaled)[0])))
        risk = Decimal(str(action.get("certificate", {}).get("operational_loss_bound_usd", 0)))
        size = sum(abs(int(leg["quantity"])) for leg in action["legs"])
        choices.append((score, risk, Decimal(action["premium_cost_usd"]), size, action["id"], action))
    if not choices:
        raise RiskError("NO_FEASIBLE_SUPPORTED_ACTION")
    best = max(choice[0] for choice in choices)
    selected = min((choice for choice in choices if best-choice[0] <= error), key=lambda choice: choice[1:5])
    return {"action": selected[-1], "selection_score": selected[0], "policy_id": pack.manifest.get("policy_id"),
            "selector": "LOWER_RISK_COST_SIZE_ID_V1", "tie_error": error,
            "search": {key: value for key, value in search.items() if key != "actions"}}


def bellman_value(weights, values, unknown_mass=0.):
    weights, values = np.asarray(weights), np.asarray(values)
    if np.any(weights < 0) or np.any((values < 0) | (values > 1)) or abs(float(np.sum(weights))+unknown_mass-1) > 1e-12:
        raise RiskError("INVALID_TRANSITION_MASS")
    lower = float(np.dot(weights, values))
    return {"lower": lower, "upper": lower+unknown_mass}
