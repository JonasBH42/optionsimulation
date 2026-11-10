"""Exact synthetic state generation; no archive or future observed rows."""
from dataclasses import replace
from decimal import Decimal
import math
import time

import numpy as np
import pytest

from qqq_sim.broker.service import Broker
from qqq_sim.risk.contracts import RiskError, YEAR_US, canonical, stable_hash
from qqq_sim.risk.offline.design import design_record, materialize_case, _path_sigma
from qqq_sim.risk.pricing_fast import european_price


def record(index=7, **updates):
    value = design_record(index, np.full(12, .5))
    value.update(spot=100., sigma_anchor=.3, skew=0., curvature=0., term_slope=.2,
                 r=0., q=0., quantity=1, half_spread=.01, desired_seconds=3*86400,
                 anchor_date="2026-01-05", expiry_count=8, adjustment_count=0)
    value.update(updates)
    return value


def materialize(value):
    # With r=q=0, the European formula is also the exact American price for
    # both rights. This isolates generation/accounting from PDE convergence.
    return materialize_case(value, price_provider=european_price)


def test_full_term_surface_admits_eight_expiries_and_records_every_exclusion():
    value = record()
    case = materialize(value)
    job, universe = case["job"], case["universe"]
    assert len(universe) == 512
    assert len({leg.fixing_utc_us for leg in universe}) == 8
    assert case["record"]["universe_admission"]["excluded_contracts"]
    tau_anchor = (job.fixing_utc_us-job.t0_utc_us)/YEAR_US
    for leg in universe:
        k = math.log(float(leg.strike)/value["spot"])
        tau = (leg.fixing_utc_us-job.t0_utc_us)/YEAR_US
        expected = value["sigma_anchor"]+value["skew"]*k+value["curvature"]*k*k+value["term_slope"]*(math.sqrt(tau)-math.sqrt(tau_anchor))
        assert leg.sigma == pytest.approx(expected)
        assert leg.last_trading_utc_us == leg.fixing_utc_us
        assert leg.known_at_utc_us == job.t0_utc_us
    assert job.sigma_path == case["record"]["path_iv_selection"]["weighted_median"]
    assert job.sigma_path != value["sigma_anchor"]
    assert {row["right"] for row in case["record"]["path_iv_selection"]["contributors"]} == {"CALL", "PUT"}
    broker = Broker.from_dict(case["account_context"]["broker_state"])
    campaign = broker.ledger.campaigns[case["campaign"]["campaign_id"]]
    assert campaign.target == job.target_usd
    assert campaign.trade_cash == job.entry_cash_usd
    assert campaign.net_cash == Decimal(case["campaign"]["net_cash_usd"])


def test_declared_adjustments_execute_actual_fills_and_nonfills_keep_original_target():
    value = record(adjustment_count=3)
    case = materialize(value)
    campaign = case["campaign"]
    assert campaign["declared_adjustments"] == 3
    assert len(campaign["adjustment_events"]) == 3
    assert {event["status"] for event in campaign["adjustment_events"]} == {"FILLED", "NO_FILL"}
    assert campaign["executed_adjustments"] == sum(event["status"] == "FILLED" for event in campaign["adjustment_events"])
    for event in campaign["adjustment_events"]:
        if event["status"] == "NO_FILL":
            assert Decimal(event["fees_usd"]) == Decimal(event["premium_cost_usd"]) == 0
    broker = Broker.from_dict(case["account_context"]["broker_state"])
    ledger_campaign = broker.ledger.campaigns[campaign["campaign_id"]]
    assert ledger_campaign.target == Decimal(campaign["original_max_profit_usd"])/2 == case["job"].target_usd
    assert ledger_campaign.net_cash == Decimal(campaign["net_cash_usd"])
    mapping = case["account_context"]["contract_id_mapping"]
    assert {contract.key: qty for contract, qty in broker.ledger.positions(campaign["campaign_id"])} == {
        mapping[leg.contract_id]: leg.quantity for leg in case["job"].signed_legs}
    assert len([order for order in broker.orders.values() if order.status == "FILLED"]) == 1+campaign["executed_adjustments"]
    assert stable_hash(canonical(case)) == stable_hash(canonical(materialize(value)))


def test_closed_observation_inherits_previous_eligible_opening():
    case = materialize(record(20, expiry_count=2, term_slope=0.))
    audit = case["record"]["closed_period_history"]
    assert audit["opening_utc_us"] < case["job"].t0_utc_us
    assert not any(a <= case["job"].t0_utc_us < b for a, b in case["job"].eligible_intervals)
    broker = Broker.from_dict(case["account_context"]["broker_state"])
    fills = [event for event in broker.ledger.events if event["event_type"] == "FILL"]
    assert fills
    assert all(event["at"] < "2026-01-05" for event in fills)
    assert audit["execution_during_closed_period"] is False
    assert case["job"].t0_utc_us in case["job"].reporting_times_utc_us


def test_surface_outside_support_rejected_without_clipping():
    with pytest.raises(RiskError) as error:
        materialize(record(sigma_anchor=1.99, term_slope=.5))
    assert error.value.code == "DESIGN_SURFACE_OUTSIDE_DOMAIN"


def test_generation_budget_precedes_price_work():
    called = []
    with pytest.raises(RiskError) as error:
        materialize_case(record(), price_provider=lambda *args: called.append(args), deadline=time.monotonic()-1)
    assert error.value.code == "RESOURCE_LIMIT"
    assert not called


def test_path_sigma_cannot_use_unobserved_generator_parameter():
    case = materialize(record())
    job = case["job"]
    sigma, audit = _path_sigma(case["universe"], job.x0, job.t0_utc_us,
                              case["campaign"]["hard_cutoff_utc_us"], 0., 0., european_price, None)
    assert sigma == job.sigma_path
    assert audit == case["record"]["path_iv_selection"]
