from dataclasses import replace
from decimal import Decimal
import pytest
from qqq_sim.risk.offline.design import reference_fixture_job
from qqq_sim.risk.policy import generate_actions


def setup(strikes=(100, 105), condor=False):
    job = reference_fixture_job(duration_seconds=3600)
    template = job.signed_legs[0]
    def contract(right, strike, quantity=1):
        return replace(template, contract_id=f"SYNTHETIC_{right}_{strike}", right=right, strike=Decimal(strike), quantity=quantity,
                       bid=2., ask=2.05, known_at_utc_us=job.t0_utc_us)
    if condor:
        held = (contract("PUT", 95), contract("PUT", 100, -1), contract("CALL", 105, -1), contract("CALL", 110))
        universe = held
    else:
        held = (contract("CALL", 100), contract("CALL", 105, -1))
        universe = tuple(contract("CALL", strike) for strike in strikes)
    job = replace(job, signed_legs=held)
    campaign = {"net_cash_usd": "0", "gross_cash_usd": "0", "target_usd": "150", "remaining_expense_budget_usd": "1000",
                "hard_cutoff_utc_us": job.fixing_utc_us, "remaining_actions": 12, "remaining_rolls": 3}
    limits = {"loss_cap_usd": "1000000000", "session_loss_cap_usd": "1000000000", "available_cash_usd": "1000000000",
              "explicit_unqualified_research_account": True}
    return job, campaign, universe, limits


def test_compulsory_pair_reduction_exists_when_whole_position_gcd_is_one():
    job, campaign, universe, limits = setup(condor=True)
    result = generate_actions(job, campaign, universe, limits)
    reductions = [action for action in result["actions"] if action["family"] == "PARTIAL_CLOSE"]
    assert len(reductions) >= 2
    assert any(len(action["legs"]) == 2 and len(action["resulting_inventory"]) == 2 for action in reductions)
    assert {action["family"] for action in result["actions"]} >= {"HOLD", "FULL_CLOSE", "EXIT_STOP"}


def test_single_long_protective_hedges_retain_current_position():
    job, campaign, universe, limits = setup()
    result = generate_actions(job, campaign, universe, limits)
    singles = [action for action in result["actions"] if action["family"] == "HEDGE" and len(action["legs"]) == 1 and action["legs"][0]["quantity"] > 0]
    assert singles
    for action in singles:
        assert action["resulting_action_count_remaining"] == 11
        assert action["target_usd"] == campaign["target_usd"]


def test_same_right_butterflies_use_diagonal_components_and_net_shared_strike():
    job, campaign, universe, limits = setup((100, 105, 110))
    result = generate_actions(job, campaign, universe, limits)
    butterflies = []
    for action in result["actions"]:
        inventory = sorted(action["resulting_inventory"], key=lambda leg: Decimal(leg["strike"]))
        if action["family"] == "REPOSITION" and len(inventory) == 3 and [leg["quantity"] for leg in inventory] == [1, -2, 1]:
            butterflies.append(action)
    assert butterflies
    assert all(len({leg["contract_id"] for leg in action["legs"]}) == len(action["legs"]) for action in butterflies)
    assert all(action["certificate"]["status"] == "VERIFIED_WITHIN_DECLARED_MODEL" for action in butterflies)


def test_unused_roll_quota_is_reused_and_every_retained_identity_is_deterministic():
    job, campaign, universe, limits = setup(tuple(range(90, 116)))
    first = generate_actions(job, campaign, universe, limits)
    second = generate_actions(job, campaign, universe[::-1], limits)
    assert len(first["actions"]) == 2000
    assert first["family_counts"]["ROLL_EXPIRATION"] == 0
    assert first["family_counts"]["HEDGE"] > 384
    assert first["raw_attempts"] <= 40_000 and first["truncated"]
    assert [action["id"] for action in first["actions"]] == [action["id"] for action in second["actions"]]
    assert len({action["id"] for action in first["actions"]}) == len(first["actions"])


def test_offline_control_reference_expands_actual_budget_without_changing_online_cap():
    job,campaign,universe,limits=setup(tuple(range(90,116)))
    normal=generate_actions(job,campaign,universe,limits,max_candidates=4000,max_raw=80000)
    reference=generate_actions(job,campaign,universe,limits,max_candidates=4000,max_raw=80000,offline_reference=True)
    assert normal["candidate_cap"]==2000 and normal["raw_cap"]==40000
    assert reference["candidate_cap"]==4000 and reference["raw_cap"]==80000
    assert len(reference["actions"])>len(normal["actions"])
    assert reference["raw_attempts"]>normal["raw_attempts"]


def test_raw_budget_counts_rejected_actions_and_reports_reason():
    job, campaign, universe, limits = setup((100, 105, 110))
    result = generate_actions(job, campaign, universe, {**limits, "loss_cap_usd": "1"}, max_raw=7)
    assert result["raw_attempts"] == 7
    assert result["truncations"]["RAW_CANDIDATE_BUDGET_REACHED"] == 1
    assert result["rejections"]["RISK_CERTIFICATE_FAILED"] > 0


def test_action_identity_includes_resulting_cash_and_reentry_rule():
    job, campaign, universe, limits = setup()
    first = generate_actions(job, campaign, universe, limits)
    changed = generate_actions(job, {**campaign, "net_cash_usd": "1"}, universe, limits)
    one = next(action for action in first["actions"] if action["family"] == "HOLD")
    two = next(action for action in changed["actions"] if action["family"] == "HOLD")
    assert one["id"] != two["id"]
    stopped = next(action for action in first["actions"] if action["family"] == "EXIT_STOP")
    closed = next(action for action in first["actions"] if action["family"] == "FULL_CLOSE")
    assert stopped["id"] != closed["id"] and not stopped["reentry_permitted"] and closed["reentry_permitted"]
