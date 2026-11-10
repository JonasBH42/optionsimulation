from dataclasses import replace
import pickle
import numpy as np
import pytest
from qqq_sim.risk.contracts import RiskError
from qqq_sim.risk.payoff import Interval
from qqq_sim.risk.offline.region_store import RegionStore,region_digest
from qqq_sim.risk.offline.hit_oracle import _simulate,observation_grid
from qqq_sim.risk.offline.design import reference_fixture_job


def test_independent_dense_allocation_respects_budget_before_kernel(monkeypatch):
    import qqq_sim.risk.offline.hit_oracle as module
    job=reference_fixture_job()
    job=replace(job,numerical_settings=replace(job.numerical_settings,memory_budget_mb=1))
    monkeypatch.setattr(module,"transition_matrix",lambda *args:pytest.fail("Dense kernel must not allocate"))
    with pytest.raises(RiskError) as error:
        module.independent_first_hit(job,np.array([job.t0_utc_us,job.fixing_utc_us]),{},job.target_usd,nodes=2049)
    assert error.value.code=="RESOURCE_LIMIT"
    assert error.value.details["estimated_additional_peak_bytes"]>400_000_000


def test_stochastic_singleton_closing_region_is_not_a_qualified_zero():
    from decimal import Decimal
    from qqq_sim.risk.offline.hit_oracle import calculate_reference
    job=reference_fixture_job(sigma_path=1.,duration_seconds=60)
    template=job.signed_legs[0]
    legs=tuple(replace(template,contract_id=str(strike),strike=Decimal(strike),quantity=quantity,sigma=0) for strike,quantity in ((100,1),(105,-2),(110,1)))
    job=replace(job,x0=104.9,signed_legs=legs,rate_cc=0.,entry_cash_usd=Decimal(0),target_usd=Decimal(500),numerical_settings=replace(job.numerical_settings,monitoring_seconds=(60,30,15),space_nodes=101,production_paths=128,pilot_paths=128,region_paths=128))
    result=calculate_reference(job)
    assert result["reason"]=="SINGLETON_CLOSING_EVENT_UNRESOLVED"
    assert result["hold_p50"] is None and not result.get("release_qualified",False)


def test_narrow_stochastic_island_cannot_pass_a_zero_endpoint_hit_plateau():
    from decimal import Decimal
    from qqq_sim.risk.offline.hit_oracle import calculate_reference
    job=reference_fixture_job(sigma_path=1.,duration_seconds=60)
    template=job.signed_legs[0]
    legs=tuple(replace(template,contract_id=str(strike),strike=Decimal(strike),quantity=quantity,sigma=0) for strike,quantity in ((100,1),(105,-2),(110,1)))
    job=replace(job,x0=104.9,signed_legs=legs,rate_cc=0.,entry_cash_usd=Decimal(0),target_usd=Decimal("499.99"),numerical_settings=replace(job.numerical_settings,monitoring_seconds=(60,30,15),space_nodes=101,production_paths=128,pilot_paths=128,region_paths=128))
    result=calculate_reference(job)
    assert result["reason"]=="NARROW_CLOSING_ISLAND_UNRESOLVED"
    assert result["hold_p50"] is None
    assert result["details"]["log_width"]>0


@pytest.mark.parametrize("target,reason",[("500","SINGLETON_CLOSING_EVENT_UNRESOLVED"),("499.99999","DETERMINISTIC_CLOSING_TOPOLOGY_UNRESOLVED")])
def test_deterministic_between_observation_butterfly_crossing_never_qualifies_zero(target,reason):
    from decimal import Decimal
    import math
    from qqq_sim.risk.offline.hit_oracle import calculate_reference, _portfolio
    job=reference_fixture_job(style="AMERICAN",sigma_path=0.,duration_seconds=60)
    template=job.signed_legs[0]
    legs=tuple(replace(template,contract_id=str(strike),strike=Decimal(strike),quantity=quantity,sigma=0,bid=None,ask=None) for strike,quantity in ((100,1),(105,-2),(110,1)))
    # With r=0,q>0 every deterministic American call exercises immediately.
    # The exact spot crosses the butterfly peak at0.5s, outside every monitor.
    spot=105*math.exp(.2*.5/31536000)
    job=replace(job,x0=spot,signed_legs=legs,rate_cc=0.,dividend_yield_cc=.2,entry_cash_usd=Decimal(0),target_usd=Decimal(target),numerical_settings=replace(job.numerical_settings,space_nodes=101,production_paths=128,pilot_paths=128,region_paths=128))
    assert _portfolio(job,np.array([105.]),59.5/31536000)[0][0]>=float(job.target_usd)
    for seconds in (0.,1.875,3.75,7.5,15.,30.,60.):
        state=spot*math.exp(-.2*seconds/31536000)
        assert _portfolio(job,np.array([state]),(60-seconds)/31536000)[0][0]<float(job.target_usd)
    result=calculate_reference(job)
    assert result["reason"]==reason
    assert result["hold_p50"] is None and not result.get("release_qualified",False)


def test_proven_static_deterministic_butterfly_remains_available():
    from decimal import Decimal
    from qqq_sim.risk.offline.hit_oracle import calculate_reference
    job=reference_fixture_job(sigma_path=0.,duration_seconds=60)
    template=job.signed_legs[0]
    legs=tuple(replace(template,contract_id=str(strike),strike=Decimal(strike),quantity=quantity,sigma=0,bid=None,ask=None) for strike,quantity in ((100,1),(105,-2),(110,1)))
    job=replace(job,x0=104.9,signed_legs=legs,rate_cc=0.,dividend_yield_cc=0.,entry_cash_usd=Decimal(0),target_usd=Decimal(500),numerical_settings=replace(job.numerical_settings,space_nodes=101,production_paths=128,pilot_paths=128,region_paths=128))
    result=calculate_reference(job)
    assert result["hold_p50"]==0.
    assert result["release_qualified"]


def test_independent_domain_uses_path_law_not_strikes_or_leg_volatility():
    import math
    from decimal import Decimal
    from qqq_sim.risk.offline.hit_oracle import independent_path_domain,initial_domain
    job=reference_fixture_job(sigma_path=.2,duration_seconds=300)
    domain,bound=independent_path_domain(job)
    assert bound<=1e-8
    assert math.log(domain[1]/domain[0])<.01
    assert math.log(initial_domain(job)[1]/initial_domain(job)[0])>1.
    changed=replace(job,signed_legs=tuple(replace(leg,strike=Decimal(10000+index),sigma=5.) for index,leg in enumerate(job.signed_legs)))
    assert independent_path_domain(changed)==(domain,bound)


def test_independent_refinement_uses_declared_spatial_cap():
    from qqq_sim.risk.offline.hit_oracle import independent_node_schedule
    assert list(independent_node_schedule(12801))==[129,257,513,1025,2049,4097,8193,12801]
    assert list(independent_node_schedule(513))==[129,257,513]
    assert list(independent_node_schedule(41))==[41]


def test_known_single_observation_barrier_converges_with_explicit_tail_bounds():
    from qqq_sim.risk.offline.hit_oracle import independent_first_hit
    job=reference_fixture_job(sigma_path=.2,duration_seconds=86400)
    job=replace(job,rate_cc=.5*.2**2,dividend_yield_cc=0.)
    observation=job.fixing_utc_us-1
    times=np.array([job.t0_utc_us,observation,job.fixing_utc_us],dtype=np.int64)
    regions={(int(instant),float(job.target_usd)):(Interval(job.x0,float("inf")),) if instant==observation else () for instant in times}
    errors=[]
    for nodes in (129,257,513):
        result=independent_first_hit(job,times,regions,job.target_usd,nodes=nodes,terminal_inclusive=False)
        assert 0<=result["lower"]<=result["upper"]<=1
        assert result["upper"]-result["lower"]<1e-6
        assert result["analytic_path_exit_bound"]<=1e-8
        assert result["probability_preservation"]["checked_every_transition"]
        assert not result["probability_preservation"]["clipping_applied"]
        errors.append(abs((result["lower"]+result["upper"])/2-.5))
    assert 0<errors[2]<errors[1]<errors[0]
    assert .4<errors[2]/errors[1]<.6
    assert errors[2]<.005


@pytest.mark.parametrize("defect",["nonfinite","outside_bounds","reversed_interval"])
def test_independent_recursion_refuses_probability_corruption(monkeypatch,defect):
    import qqq_sim.risk.offline.hit_oracle as module
    job=reference_fixture_job(duration_seconds=1)
    times=np.array([job.t0_utc_us,job.fixing_utc_us],dtype=np.int64)
    regions={(int(t),float(job.target_usd)):() for t in times}
    def bad_matrix(grid,*args):
        matrix=np.eye(len(grid))
        tails=np.full(len(grid),float("nan") if defect=="nonfinite" else 2. if defect=="outside_bounds" else -.1)
        return matrix,tails
    monkeypatch.setattr(module,"transition_matrix",bad_matrix)
    with pytest.raises(RiskError) as caught:
        module.independent_first_hit(job,times,regions,job.target_usd,nodes=41,terminal_inclusive=False)
    assert caught.value.code==("TRANSITION_PROBABILITY_NONFINITE" if defect=="nonfinite" else "TRANSITION_PROBABILITY_BOUNDS_FAILURE")


def test_reference_resource_exit_retains_completed_independent_refinements(monkeypatch):
    import qqq_sim.risk.offline.hit_oracle as module
    job=reference_fixture_job(duration_seconds=1)
    job=replace(job,x0=50.,numerical_settings=replace(job.numerical_settings,production_paths=4096))
    calls=[]
    def interrupted(*args,**kwargs):
        calls.append(kwargs["nodes"])
        if len(calls)>1:
            raise RiskError("RESOURCE_LIMIT","Explicit test allocation limit",resource="memory")
        return {"lower":0.,"upper":0.,"nodes":kwargs["nodes"]}
    monkeypatch.setattr(module,"independent_first_hit",interrupted)
    result=module.calculate_reference(job)
    assert result["reason"]=="RESOURCE_LIMIT"
    assert result["hold_p50"] is None
    assert result["independent_probability_validation"][0]["refinements"][0]["nodes"]==129


@pytest.mark.parametrize('offsets,expected_builds',[(list(range(11)),1),([0,1,2,8,9,10],3)])
def test_independent_transition_reuse_matches_uncached_recurrence(monkeypatch,offsets,expected_builds):
    import math
    from qqq_sim.risk.contracts import YEAR_US
    from qqq_sim.risk.payoff import region_membership
    import qqq_sim.risk.offline.hit_oracle as module
    job=reference_fixture_job(sigma_path=.2,duration_seconds=10)
    times=np.array([job.t0_utc_us+x*1_000_000 for x in offsets],dtype=np.int64)
    regions={(int(t),float(job.target_usd)):(Interval(job.x0*(1.0001+(i%2)*.00001),float('inf')),) for i,t in enumerate(times)}
    original=module.transition_matrix;calls=[]
    def measured(*args):
        calls.append(args[1]);return original(*args)
    monkeypatch.setattr(module,'transition_matrix',measured)
    actual=module.independent_first_hit(job,times,regions,job.target_usd,nodes=41,terminal_inclusive=False)
    domain,_=module.independent_path_domain(job);grid=np.linspace(math.log(domain[0]),math.log(domain[1]),41);spots=np.exp(grid)
    lo=np.zeros(41)
    if module.eligible(job,int(times[-1])):
        lo[region_membership(regions[(int(times[-1]),float(job.target_usd))],spots)]=1.
    hi=lo.copy();mu=job.rate_cc-job.dividend_yield_cc-job.sigma_path**2/2
    for before,after in zip(times[-2::-1],times[:0:-1]):
        matrix,tails=original(grid,(int(after)-int(before))/YEAR_US,job.sigma_path,mu)
        lo,hi=matrix@lo,matrix@hi+tails
        if module.eligible(job,int(before)):
            hit=region_membership(regions[(int(before),float(job.target_usd))],spots)
            lo[hit],hi[hit]=1.,1.
    assert actual['lower']==float(np.interp(math.log(job.x0),grid,lo))
    assert actual['upper']==float(np.interp(math.log(job.x0),grid,hi))
    assert len(calls)==actual['transition_matrix_builds']==expected_builds
    assert actual['transition_matrix_reuses']==len(times)-1-expected_builds
    assert actual['maximum_cached_transition_bytes']==41*42*8


def test_region_blocks_preserve_all_intervals_and_detect_corruption(tmp_path):
    expected={}
    store=RegionStore({"job":"test","domain":[1,200]},block_bytes=1024,root=tmp_path)
    for t in reversed(range(80)):
        expected[(t,50.)]=tuple(Interval(float(i),float(i)+.5,i%2==0,True) for i in range(t%13))
        store[(t,50.)]=expected[(t,50.)]
    store.freeze()
    assert store.statistics()["disk_backed"]
    assert dict(store.ordered_items())==expected
    assert region_digest(store)==region_digest(expected)
    clone=pickle.loads(pickle.dumps(store))
    assert dict(clone.ordered_items())==expected
    assert len(clone._cache)<=1
    with pytest.raises(RiskError,match="IMMUTABLE_REGION_STORE"):
        store[(999,50.)]=()
    store.close();clone.close()
    from pathlib import Path
    first=next(Path(store.folder).glob("intervals-*.npy"))
    with first.open("r+b") as stream:
        stream.seek(-1,2);stream.write(b"z")
    with pytest.raises(RiskError,match="REGION_STORE_INTEGRITY_FAILURE"):
        dict(clone.ordered_items())


def test_spawn_shards_match_serial_and_resume_across_worker_counts(tmp_path):
    job=reference_fixture_job(sigma_path=.2,duration_seconds=3)
    job=replace(job,numerical_settings=replace(job.numerical_settings,workers=1,memory_budget_mb=1024))
    times,_=observation_grid(job,1)
    regions=RegionStore({"job":job.computation_hash},block_bytes=1024,root=tmp_path)
    for t in times:
        regions[(int(t),float(job.target_usd))]=(Interval(job.x0,float("inf")),)
    regions.freeze()
    serial=_simulate(job,times,[regions],9000,10)
    parallel_job=replace(job,numerical_settings=replace(job.numerical_settings,workers=2))
    checkpoint=tmp_path/"shards.json"
    parallel=_simulate(parallel_job,times,[regions],9000,10,checkpoint=checkpoint)
    assert parallel["workers_used"]==2
    for key in ("counts","terminal","disagreements"):
        np.testing.assert_array_equal(serial[key],parallel[key])
    assert serial["identity"]==parallel["identity"]
    assert serial["histograms"]==parallel["histograms"]
    resumed=_simulate(job,times,[regions],9000,10,checkpoint=checkpoint)
    assert resumed["histograms"]==parallel["histograms"]
    assert resumed["paths"]==9000


def test_adaptive_topology_keeps_narrow_island_and_tangent_contact():
    from qqq_sim.risk.offline.european_regions import adaptive_regions
    from qqq_sim.risk.payoff import region_membership
    # Both initial endpoints are negative: a sign-change-only scan misses this.
    f=lambda x:1e-6-(x-.3712345)**2
    result=adaptive_regions(f,[0.,1.],2.,max_evaluations=1_000_000)
    assert region_membership(result["certain"],.3712345)
    assert not region_membership(result["possible"],.5)
    tangent=adaptive_regions(lambda x:-(x-.5)**2,[0.,.5,1.],2.,max_evaluations=1_000_000)
    assert region_membership(tangent["possible"],.5)
    assert not region_membership(tangent["certain"],.5)
    assert tangent["unresolved_tolerance_intervals"]>0


def test_european_derivative_bound_handles_carry_and_short_legs():
    from qqq_sim.risk.offline.european_regions import portfolio_lipschitz
    from qqq_sim.risk.offline.hit_oracle import _portfolio
    job=reference_fixture_job(sigma_path=.2,duration_seconds=3600)
    job=replace(job,dividend_yield_cc=-.02,signed_legs=tuple(replace(x,exercise_style="EUROPEAN",sigma=.3) for x in job.signed_legs))
    lower=np.array([400.,450.,500.,550.]);upper=lower+20
    bound=portfolio_lipschitz(job,lower,upper,.1)
    for fraction in np.linspace(0,1,11):
        point=lower+fraction*(upper-lower)
        derivative=(_portfolio(job,point+1e-4,.1)[0]-_portfolio(job,point-1e-4,.1)[0])/2e-4
        assert np.all(np.abs(derivative)<=bound+1e-5)


def test_reference_source_identity_works_without_packaged_py_files(monkeypatch,tmp_path):
    from qqq_sim.risk.offline import source_identity
    from qqq_sim import _build_info
    identity=source_identity.implementation_identity()
    monkeypatch.setattr(_build_info,'SOURCE_HASHES',identity['files'])
    monkeypatch.setattr(source_identity,'__file__',str(tmp_path/'qqq_sim/risk/offline/source_identity.py'))
    assert source_identity.implementation_identity()==identity
