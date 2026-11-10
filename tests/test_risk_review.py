"""Independent review regressions for shard safety and numerical semantics."""
from dataclasses import replace
import math
import numpy as np
import pytest
from types import SimpleNamespace
from decimal import Decimal
from qqq_sim.risk.contracts import RiskError, YEAR_US
from qqq_sim.risk.contracts import canonical
from qqq_sim.risk.offline.design import reference_fixture_job
from qqq_sim.risk.offline.hit_oracle import _simulate, observation_grid
from qqq_sim.risk.offline.control_oracle import ManagedState, transition
from qqq_sim.risk.payoff import Interval


def test_checkpoint_rejects_changed_region_artifacts(tmp_path):
    job = reference_fixture_job(sigma_path=0, duration_seconds=3)
    times = np.array([job.t0_utc_us, job.t0_utc_us + 1_000_000, job.fixing_utc_us])
    target = float(job.target_usd)
    first = {(int(t), target): (Interval(0, math.inf),) if t == times[1] else () for t in times}
    changed = {(int(t), target): () for t in times}
    checkpoint = tmp_path / "reference_checkpoint.json"
    original = _simulate(job, times, [first], 64, 10, checkpoint=checkpoint)
    assert original["counts"][0, 0, -1] == 1
    with pytest.raises(RiskError, match="CHECKPOINT_MISMATCH"):
        _simulate(job, times, [changed], 64, 10, checkpoint=checkpoint)


def test_checkpoint_rejects_changed_monitoring_schedule(tmp_path):
    job = reference_fixture_job(sigma_path=0, duration_seconds=3)
    times = np.array([job.t0_utc_us, job.t0_utc_us + 1_000_000, job.fixing_utc_us])
    regions = {(int(t), float(job.target_usd)): (Interval(0, math.inf),) for t in times}
    checkpoint = tmp_path / "monitor_checkpoint.json"
    _simulate(job, times, [regions], 64, 10, monitoring_sets=[{int(times[1])}], checkpoint=checkpoint)
    with pytest.raises(RiskError, match="CHECKPOINT_MISMATCH"):
        _simulate(job, times, [regions], 64, 10, monitoring_sets=[set()], checkpoint=checkpoint)


def test_equal_aggregate_probabilities_do_not_hide_pathwise_disagreement():
    job = reference_fixture_job(sigma_path=0.2, duration_seconds=3)
    count = 128
    times = np.array([job.t0_utc_us, job.t0_utc_us + 1_000_000, job.fixing_utc_us])
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(job.numerical_settings.seed, spawn_key=(10, 0))))
    dt = 1_000_000 / YEAR_US
    drift = job.rate_cc - job.dividend_yield_cc - job.sigma_path ** 2 / 2
    spots = job.x0 * np.exp(drift * dt + job.sigma_path * math.sqrt(dt) * generator.standard_normal(count))
    median = float(np.median(spots))
    target = float(job.target_usd)
    left = {(int(t), target): (Interval(0, median),) if t == times[1] else () for t in times}
    right = {(int(t), target): (Interval(median, math.inf),) if t == times[1] else () for t in times}
    result = _simulate(job, times, [left, right], count, 10)
    assert result["counts"][0, 0, -1] == result["counts"][1, 0, -1] == count // 2
    assert result["disagreements"][0, 0, -1] == count


def test_out_of_domain_path_is_unresolved_not_a_missed_hit():
    job = reference_fixture_job(sigma_path=0, duration_seconds=2)
    times, _ = observation_grid(job, 1)
    regions = {(int(t), float(job.target_usd)): () for t in times}
    with pytest.raises(RiskError, match="exited the validated domain"):
        _simulate(job, times, [regions], 64, 10, domain=(job.x0 + 1, job.x0 + 2))


def test_production_resume_preserves_counts_and_first_hit_histograms(tmp_path):
    job = reference_fixture_job(sigma_path=0.2, duration_seconds=3)
    times, _ = observation_grid(job, 1)
    regions = {(int(t), float(job.target_usd)): (Interval(job.x0, math.inf),) for t in times}
    checkpoint = tmp_path / "resume_checkpoint.json"
    first = _simulate(job, times, [regions], 5000, 10, checkpoint=checkpoint)
    resumed = _simulate(job, times, [regions], 5000, 10, checkpoint=checkpoint)
    assert np.array_equal(first["counts"], resumed["counts"])
    assert np.array_equal(first["terminal"], resumed["terminal"])
    assert first["histograms"] == resumed["histograms"]
    assert first["paths"] == resumed["paths"] == 5000


def test_reporting_timestamps_cannot_disable_later_shard_time_budget(monkeypatch):
    import qqq_sim.risk.offline.hit_oracle as oracle
    job = reference_fixture_job(sigma_path=.2, duration_seconds=3)
    times = np.array([job.t0_utc_us, job.fixing_utc_us])
    regions = {(int(t), float(job.target_usd)): () for t in times}
    ticks = iter([0., 0., 0., 0., 20., 20., 20.])
    monkeypatch.setattr(oracle.time, "monotonic", lambda: next(ticks, 20.))
    with pytest.raises(RiskError, match="Path work budget exceeded"):
        _simulate(job, times, [regions], 5000, 10, deadline=10.)


def test_managed_full_close_then_hold_cannot_resurrect_inventory():
    job = reference_fixture_job(sigma_path=0, duration_seconds=300)
    campaign = {"net_cash_usd": "-200", "gross_cash_usd": "-200", "target_usd": "150",
                "remaining_expense_budget_usd": "100", "hard_cutoff_utc_us": job.fixing_utc_us,
                "remaining_actions": 12, "remaining_rolls": 3}
    close = {"id": "close", "family": "FULL_CLOSE", "legs": canonical(tuple(replace(leg, quantity=-leg.quantity) for leg in job.signed_legs)),
             "eligible_utc_us": job.t0_utc_us + 60_000_000, "limit_cost_usd": "1000", "max_total_fees_usd": "50", "stop_after_fill": False}
    pack = SimpleNamespace(price_basis=None)
    closed = transition(ManagedState(job, campaign, job.signed_legs), close, 0, pack)
    assert closed.events[-1]["status"] == "FILLED"
    assert closed.campaign["open_contract_quantities"] == {}
    assert not closed.success and not closed.failed
    held = transition(closed, {"id": "hold", "family": "HOLD", "legs": [], "eligible_utc_us": job.t0_utc_us + 120_000_000}, 0, pack)
    assert held.campaign["open_contract_quantities"] == {}
    assert held.campaign["net_cash_usd"] == closed.campaign["net_cash_usd"]
    assert held.campaign["target_usd"] == "150"


def test_managed_roll_updates_option_fixing_without_resetting_campaign_target():
    job = reference_fixture_job(sigma_path=0, duration_seconds=300)
    new_fixing = job.fixing_utc_us + 300_000_000
    rolled_legs = tuple(replace(leg, contract_id=leg.contract_id + "_NEXT", fixing_utc_us=new_fixing) for leg in job.signed_legs)
    campaign = {"net_cash_usd": "-200", "gross_cash_usd": "-200", "target_usd": "150", "original_target_usd": "150",
                "remaining_expense_budget_usd": "100", "hard_cutoff_utc_us": new_fixing,
                "remaining_actions": 12, "remaining_rolls": 3, "p0_policy": 0.6}
    changes = tuple(replace(leg, quantity=-leg.quantity) for leg in job.signed_legs) + rolled_legs
    action = {"id": "roll", "family": "ROLL_EXPIRATION", "legs": canonical(changes),
              "eligible_utc_us": job.t0_utc_us + 60_000_000, "limit_cost_usd": "1000", "max_total_fees_usd": "50", "stop_after_fill": False}
    state = transition(ManagedState(job, campaign, (*job.signed_legs, *rolled_legs)), action, 0, SimpleNamespace(price_basis=None))
    assert state.events[-1]["status"] == "FILLED"
    assert state.job.fixing_utc_us == new_fixing
    assert all(leg.fixing_utc_us == new_fixing for leg in state.job.signed_legs)
    assert state.campaign["target_usd"] == state.campaign["original_target_usd"] == "150"
    assert state.campaign["hard_cutoff_utc_us"] == new_fixing
    assert state.campaign["p0_policy"] == 0.6
