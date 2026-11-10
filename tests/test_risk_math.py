from dataclasses import replace
from decimal import Decimal
import math
import numpy as np
import pytest

from qqq_sim.risk.contracts import RiskLeg, RiskError, ny_time, canonical, ReferenceJob
from qqq_sim.risk.payoff import payoff_extrema, original_target, terminal_regions, piecewise_regions, region_membership
from qqq_sim.risk.pricing_fast import european_price, deterministic_american, tensor_chebyshev
from qqq_sim.risk.offline.pricing_oracle import crr_price, generator_coefficients, solve_lcp, american_price, invert_iv
from qqq_sim.risk.offline.hit_oracle import observation_grid, endpoint_probability, wilson, continuous_barrier_probability, transition_matrix, _simulate
from qqq_sim.risk.offline.design import reference_fixture_job, latin_hypercube, design_record


@pytest.mark.parametrize('seconds',[1,15,61])
def test_short_american_time_grids_actually_refine(seconds):
    from qqq_sim.risk.offline.pricing_oracle import AmericanGrid
    year=365*86400
    counts=[];grids=[]
    for scale in (1,2,4):
        grid=AmericanGrid('PUT',100,.3,.02,.01,(50,200),nodes=81,time_scale=scale)
        recorded=[];step=grid._step
        def observed(dt):
            step(dt);recorded.append(grid.tau)
        grid._step=observed
        grid.advance(seconds/year)
        counts.append(len(recorded));grids.append(recorded)
        assert grid.tau==seconds/year
    assert counts==[math.ceil(seconds/60)*scale for scale in (1,2,4)]
    np.testing.assert_allclose(grids[0],grids[1][1::2],rtol=0,atol=1e-18)
    np.testing.assert_allclose(grids[1],grids[2][1::2],rtol=0,atol=1e-18)


def leg(right, strike, qty):
    return RiskLeg(f"{right}{strike}:{qty}", right, Decimal(str(strike)), qty, 1, .2)


@pytest.mark.parametrize("legs,cash,maximum,loss,target,region", [
    ((leg("PUT",767,-1),leg("PUT",765,1)),80,80,120,40,766.6),
    ((leg("CALL",100,1),leg("CALL",105,-1)),-200,300,200,150,103.5),
])
def test_exact_verticals(legs,cash,maximum,loss,target,region):
    out=payoff_extrema(legs,cash)
    assert out["max_profit"]==maximum and out["max_loss"]==loss
    assert original_target(legs,cash)==target
    assert terminal_regions(legs,cash,target)[0].lower==pytest.approx(region)


def test_condor_complete_topology_and_cent_pop():
    legs=(leg("PUT",90,1),leg("PUT",95,-1),leg("CALL",105,-1),leg("CALL",110,1))
    out=payoff_extrema(legs,200)
    assert out["max_profit"]==200 and out["max_loss"]==300
    region=terminal_regions(legs,200,100)
    assert len(region)==1 and region[0].lower==94 and region[0].upper==106
    put=(leg("PUT",767,-1),leg("PUT",765,1))
    assert terminal_regions(put,80,.01)[0].lower==pytest.approx(766.2001)
    assert not terminal_regions(put,80,0,strict=True)[0].lower_closed


def test_unbounded_extrema_never_grid_approximations():
    assert payoff_extrema((leg("CALL",100,1),),-500)["max_profit"].is_infinite()
    assert payoff_extrema((leg("CALL",100,-1),),500)["max_loss"].is_infinite()
    with pytest.raises(RiskError,match="UNDEFINED_MAX_PROFIT_TARGET"):
        original_target((leg("CALL",100,1),),-500)
    assert canonical(float("inf"))=="+infinity"


def test_singletons_and_flat_regions_retained():
    regions=piecewise_regions([1,2,3],[-1,0,-1])
    assert len(regions)==1 and regions[0].lower==regions[0].upper==2
    assert region_membership(regions,[1.999,2,2.001]).tolist()==[False,True,False]
    assert not piecewise_regions([1,2,3],[-1,0,-1],strict=True)
    assert len(piecewise_regions([1,2,3],[0,0,0]))==1


def test_bsm_parity_forward_and_limits():
    call=european_price("CALL",100,100,1,.05,0,.2)
    put=european_price("PUT",100,100,1,.05,0,.2)
    assert call==pytest.approx(10.450583572186,abs=1e-11)
    assert put==pytest.approx(5.573526022257,abs=1e-11)
    assert call-put==pytest.approx(100-100*math.exp(-.05),abs=1e-11)
    assert european_price("CALL",100*math.exp(.05),100,1,.05,0,.2,"TERMINAL_FORWARD")==pytest.approx(call,abs=1e-11)
    assert european_price("PUT",80,100,0,.05,0,.2)==20
    assert european_price("CALL",0,100,1,.05,0,.2)==0


def test_american_deterministic_negative_rate():
    assert deterministic_american("PUT",80,100,1,-.05,.02)==pytest.approx(26.711215773061994,abs=1e-11)
    assert american_price("CALL",100,100,1,.05,0,.2).price==pytest.approx(10.450583572186,abs=1e-11)


def test_independent_crr_reference():
    assert crr_price("PUT",100,100,1,.05,0,.2,2000)==pytest.approx(6.08998995255238,abs=1e-8)
    assert crr_price("PUT",100,100,1,.05,0,.2,4000)==pytest.approx(6.090180715022948,abs=1e-8)


def test_monotone_generator_and_complementarity():
    lower,upper=generator_coefficients(.001,-.1,.2,.1)
    assert lower>=0 and upper>=0
    later=np.array([1.,.75,.5,.25,0.])
    obstacle=np.array([1.,.7,.3,0.,0.])
    values,residual,iterations=solve_lcp(later,obstacle,.01,2,3,.02,(1.,0.))
    assert np.all(values>=obstacle-1e-9)
    assert residual<1e-8 and iterations>0


def test_iv_inversion_and_intrinsic_identifiability():
    price=float(european_price("PUT",100,100,.2,.04,.01,.27))
    result=invert_iv("PUT",100,100,.2,.04,.01,price)
    assert result["sigma"]==pytest.approx(.27,abs=1e-8)
    with pytest.raises(RiskError,match="plateau"):
        invert_iv("PUT",100,100,.2,0,0,0)


def test_ny_dst_rejects_ambiguous_and_nonexistent():
    with pytest.raises(RiskError): ny_time("2026-03-08 02:30")
    with pytest.raises(RiskError): ny_time("2026-11-01 01:30")
    assert ny_time("2026-03-08 10:00")-ny_time("2026-03-07 10:00")==23*3600*1_000_000


def test_observation_exclusive_close_and_true_fixing():
    job=reference_fixture_job(duration_seconds=300)
    times,eligible=observation_grid(job,60)
    assert job.fixing_utc_us in times
    assert not eligible[-1]
    assert job.fixing_utc_us-1 in times
    assert job.t0_utc_us in times
    changed=replace(job,deadline_utc_us=job.t0_utc_us+60_000_000)
    assert changed.fixing_utc_us==job.fixing_utc_us


def test_first_hit_remembers_hit_and_separates_terminal():
    from qqq_sim.risk.payoff import Interval
    job=reference_fixture_job(sigma_path=0,duration_seconds=3)
    times=np.array([job.t0_utc_us,job.t0_utc_us+1_000_000,job.t0_utc_us+2_000_000,job.fixing_utc_us])
    target=float(job.target_usd)
    regions={(int(t),target):() for t in times}
    regions[(int(times[1]),target)]=(Interval(0,math.inf),)
    out=_simulate(job,times,[regions],200000,10)
    assert out["paths"]==1 and out["counts"][0,0,-1]==1
    assert list(out["histograms"][0])==[int(times[1])]
    assert out["terminal"][0,0]==0


def test_closed_gap_not_hit():
    from qqq_sim.risk.payoff import Interval
    job=reference_fixture_job(sigma_path=0,duration_seconds=3)
    job=replace(job,eligible_intervals=((job.t0_utc_us,job.t0_utc_us+500000),))
    times=np.array([job.t0_utc_us,job.t0_utc_us+1_000_000,job.fixing_utc_us])
    regions={(int(t),float(job.target_usd)):(Interval(0,math.inf),) if t==times[1] else () for t in times}
    out=_simulate(job,times,[regions],200000,10)
    assert out["counts"].sum()==0


def test_endpoint_vs_constant_barrier():
    assert continuous_barrier_probability(math.log(1.1),.25,.2,0)==pytest.approx(.3405384952145612,abs=1e-12)
    job=reference_fixture_job(duration_seconds=int(.25*365*86400))
    job=replace(job,rate_cc=.02,dividend_yield_cc=0)
    from qqq_sim.risk.payoff import Interval
    assert endpoint_probability(job,job.fixing_utc_us,(Interval(110,math.inf),))==pytest.approx(.1702692476072806,abs=1e-12)


def test_wilson_not_zero_width_at_extremes():
    assert wilson(0,200000)[1]>0 and wilson(200000,200000)[0]<1
    interval=wilson(100000,200000)
    assert (interval[1]-interval[0])/2==pytest.approx(.0021913,abs=1e-7)


def test_truncated_normal_kernel_preserves_mass():
    matrix,tail=transition_matrix(np.linspace(-2,2,81),.1,.2,.03)
    assert np.min(matrix)>=-1e-10
    assert np.max(np.abs(matrix.sum(axis=1)+tail-1))<1e-10
    assert tail[0]>.4


def test_latin_hypercube_stratification_and_determinism():
    one=latin_hypercube(64)
    two=latin_hypercube(64)
    assert np.array_equal(one["coordinates"],two["coordinates"])
    for column in one["coordinates"].T:
        assert sorted(np.floor(column*64).astype(int))==list(range(64))
    assert design_record(0,one["coordinates"][0])["provenance"]=="SYNTHETIC_DESIGN"


def test_invalid_raw_inputs_never_filled_with_defaults():
    with pytest.raises(RiskError): RiskLeg("x","CALL",Decimal(100),True,1,.2)
    with pytest.raises(RiskError): RiskLeg("x","CALL",Decimal(100),1,1,float("nan"))
    job=reference_fixture_job()
    with pytest.raises(RiskError): replace(job,deadline_utc_us=job.fixing_utc_us+1)
    with pytest.raises(RiskError): replace(job,parameter_provenance=({"name":"rate","known_at_utc_us":job.t0_utc_us+1},))
