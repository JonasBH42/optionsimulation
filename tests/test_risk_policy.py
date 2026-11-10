from dataclasses import replace
from decimal import Decimal
import math
import numpy as np
import pytest
from qqq_sim.risk.contracts import RiskLeg, RiskError, canonical
from qqq_sim.risk.verify import risk_certificate, exact_poststate, verify_adjustment
from qqq_sim.risk.policy import bellman_value
from qqq_sim.risk.envelope import form_envelope, optimize_envelope
from qqq_sim.risk.hold import survival_bins, survival_cdf, evaluate_hold
from qqq_sim.risk.offline.design import reference_fixture_job
from qqq_sim.risk.offline.control_oracle import normal_quadrature, converged_expectation


def protected_spread():
    return (RiskLeg("SP100","PUT",Decimal(100),-1,1,.2),RiskLeg("LP95","PUT",Decimal(95),1,1,.2))


def test_operational_bound_distinct_terminal_bound():
    legs=protected_spread()
    out=risk_certificate(legs,{"net_cash_usd":"97.75","target_usd":"50"})
    assert out["status"]=="VERIFIED_WITHIN_DECLARED_MODEL"
    assert Decimal(out["operational_loss_bound_usd"])==Decimal("502.25")
    assert Decimal(out["terminal_loss_bound_usd"])==Decimal("402.25")
    out=risk_certificate(legs,{"net_cash_usd":"97.75","remaining_expense_budget_usd":"10"})
    assert Decimal(out["operational_loss_bound_usd"])==Decimal("412.25")


def test_condor_both_assignment_wings_reserved():
    legs=(*protected_spread(),RiskLeg("SC105","CALL",Decimal(105),-1,1,.2),RiskLeg("LC110","CALL",Decimal(110),1,1,.2))
    out=risk_certificate(legs,{"net_cash_usd":"200"})
    assert Decimal(out["delivery_cash_bound_usd"])==1000
    assert Decimal(out["terminal_loss_bound_usd"])==300


def test_prior_loss_cannot_reset_with_new_campaign():
    out=risk_certificate(protected_spread(),{"net_cash_usd":"-100","remaining_expense_budget_usd":"450"})
    assert out["status"]=="FAILED" and "CAMPAIGN_LOSS_CAP_EXCEEDED" in out["reasons"]
    out=risk_certificate(protected_spread(),{"net_cash_usd":"100"},{"other_campaign_net_usd":"-800"})
    assert "SESSION_LOSS_CAP_EXCEEDED" in out["reasons"]


def test_delayed_notice_and_unmatched_short_unknown():
    assert risk_certificate(protected_spread(),{"net_cash_usd":100,"operational_model":"DELAYED_NOTICE"})["status"]=="UNKNOWN"
    assert risk_certificate(protected_spread()[:1],{"net_cash_usd":100})["reason"]=="UNMATCHED_SHORT_DELIVERY"


def test_original_target_and_realized_cash_preserved():
    campaign={"net_cash_usd":"170","gross_cash_usd":"200","target_usd":"100","remaining_expense_budget_usd":"100","p0_policy":.6}
    changes=tuple(replace(x,quantity=-x.quantity) for x in protected_spread())
    legs,out=exact_poststate(protected_spread(),campaign,changes,20,2)
    assert not legs and out["target_usd"]=="100" and out["p0_policy"]==.6
    assert Decimal(out["net_cash_usd"])==148 and Decimal(out["remaining_expense_budget_usd"])==98


def test_policy_probability_arithmetic_and_unknown_mass():
    assert bellman_value([.25,.75],[1,0])=={"lower":.25,"upper":.25}
    assert bellman_value([.997],[0],.003)=={"lower":0,"upper":.003}
    with pytest.raises(RiskError): bellman_value([.997],[1],0)


def test_envelope_honest_shortfall_and_feasible_filter():
    candidates=[{"q_policy":.25,"empirical_error_range":[.24,.26],"error":.01},{"q_policy":.4,"empirical_error_range":[.39,.41],"error":.01,"feasible":False}]
    out=form_envelope(candidates,.7,[.69,.71],True)
    assert out["v_search"]==.25 and out["benchmark_preservation"]=="NO_WITHIN_SEARCH"
    assert out["truncated"] and not out["interpolated_portfolios_authorized"]


def test_closed_period_cdf_constant_and_monotonic():
    job=reference_fixture_job()
    bins=survival_bins(job,15)
    hazard=np.full(576,.02)
    values=[survival_cdf(hazard,bins,job.t0_utc_us+i*10_000_000) for i in range(30)]
    assert values==sorted(values)
    close=survival_cdf(hazard,bins,job.fixing_utc_us)
    assert survival_cdf(hazard,bins,job.fixing_utc_us+1_000_000)==close
    assert .6+(1-.6)*.25==pytest.approx(.7)


def test_missing_pack_abstains_without_candidate_search():
    job=reference_fixture_job()
    assert evaluate_hold(job)["reason"]=="PACK_MISSING"
    assert optimize_envelope(job,{})["reason"]=="PACK_MISSING"


def test_gauss_hermite_exact_low_moments():
    z,w=normal_quadrature(17)
    assert w.sum()==pytest.approx(1,abs=1e-14)
    assert np.dot(w,z)==pytest.approx(0,abs=1e-14)
    assert np.dot(w,z*z)==pytest.approx(1,abs=1e-14)
    out=converged_expectation(lambda z:.25)
    assert out["value"]==pytest.approx(.25) and out["status"]=="CONVERGED"
