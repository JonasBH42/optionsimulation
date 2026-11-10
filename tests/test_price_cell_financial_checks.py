import itertools
import math

import numpy as np
import pytest

from qqq_sim.risk.offline import price_basis as builder
from qqq_sim.risk.pricing_fast import european_price


@pytest.mark.parametrize("right,spot,value,greeks,expected", [
    ("CALL",2.,.9,[1.,0.,0.,0.],"INTRINSIC_LOWER_BOUND"),
    ("PUT",.5,.4,[-1.,0.,0.,0.],"INTRINSIC_LOWER_BOUND"),
    ("CALL",1.,0.,[.5,.1,.01,0.],"EUROPEAN_LOWER_BOUND"),
    ("PUT",1.,0.,[-.5,.1,.01,0.],"EUROPEAN_LOWER_BOUND"),
    ("CALL",1.,1.01,[1.,0.,0.,0.],"FINANCIAL_UPPER_BOUND"),
    ("PUT",1.,1.01,[-1.,0.,0.,0.],"FINANCIAL_UPPER_BOUND"),
    ("CALL",1.,.2,[-.1,.1,.01,0.],"SPOT_MONOTONICITY"),
    ("PUT",1.,.2,[.1,.1,.01,0.],"SPOT_MONOTONICITY"),
    ("CALL",1.,.2,[.1,.1,.01,0.],"STRIKE_MONOTONICITY"),
    ("CALL",1.,.2,[.5,-.1,.01,0.],"SPOT_CONVEXITY"),
    ("PUT",1.,.2,[-.5,.1,-.01,0.],"NONNEGATIVE_VEGA"),
])
def test_each_financial_inequality_detects_a_bad_polynomial(right,spot,value,greeks,expected):
    result=builder.financial_inequality_checks(right,[math.log(spot),.2,0.,0.],value,greeks)
    assert not result[expected]["passed"]
    assert result[expected]["violation_beyond_uncertainty"]>0


def test_negative_rate_put_upper_bound_and_independent_uncertainty():
    point=[math.log(.001),.01,-.002,0.]
    value=math.exp(.002)-.001
    checks=builder.financial_inequality_checks("PUT",point,value,[-1.,0.,0.,0.])
    assert value>1. and all(check["passed"] for check in checks.values())
    uncertain=builder.financial_inequality_checks("PUT",point,value,[-1.,-2e-9,-3e-9,0.],
        greek_uncertainty=np.array([0.,2.1e-9,3.1e-9,0.]))
    assert all(check["passed"] for check in uncertain.values())
    # Measured derivative disagreement is not reused as a tolerance that could
    # make an arbitrarily bad polynomial certify its own financial shape.
    certain=builder.financial_inequality_checks("PUT",point,value,[-1.,-2e-9,-3e-9,0.])
    assert not certain["SPOT_CONVEXITY"]["passed"]
    assert not certain["NONNEGATIVE_VEGA"]["passed"]


@pytest.mark.parametrize("right",["CALL","PUT"])
def test_analytical_zero_carry_european_cell_passes_all_american_inequalities(right):
    # At r=q=0 there is no early-exercise advantage for either right.
    from scipy.special import ndtr
    for spot in (.5,1.,2.):
        sigma=.3
        d1=math.log(spot)/sigma+sigma/2
        density=math.exp(-d1*d1/2)/math.sqrt(2*math.pi)
        value=float(european_price(right,spot,1.,1.,0.,0.,sigma))
        delta=float(ndtr(d1))-(right=="PUT")
        greeks=[delta,density/(spot*sigma),.01*spot*density,0.]
        checks=builder.financial_inequality_checks(right,[math.log(spot),sigma,0.,0.],value,greeks,price_uncertainty=1e-15)
        assert all(check["passed"] for check in checks.values())


def test_midplane_uses_signed_variation_and_physical_width_not_residual_magnitude():
    coordinates=np.asarray(list(itertools.product((-1.,-.5,0.,.5,1.),repeat=4)))
    # A large constant offset carries no refinement information. Off-midplane
    # interactions must not override the specified central-axis rule.
    residual=100+coordinates[:,0]*.1+coordinates[:,1]*2+1e4*coordinates[:,2]*coordinates[:,3]
    bounds=np.array([[0.,10.],[0.,1.],[0.,.01],[0.,.01]])
    axis,variation=builder.midplane_residual_axis(bounds,coordinates,residual)
    assert axis==1
    np.testing.assert_allclose(variation,[.02,4.,0.,0.],atol=1e-12)
    # Signed residual changes from -1 to +1 have variation 2, not zero.
    axis,variation=builder.midplane_residual_axis(np.tile([0.,2.],(4,1)),coordinates,coordinates[:,0])
    assert axis==0 and variation[0]==1.


def test_midplane_ties_are_x_then_w_then_a_then_b_and_both_rights_count():
    coordinates=np.asarray(list(itertools.product((-1.,-.5,0.,.5,1.),repeat=4)))
    bounds=np.tile([0.,2.],(4,1))
    axis,_=builder.midplane_residual_axis(bounds,coordinates,np.column_stack((coordinates.sum(axis=1),-coordinates.sum(axis=1))))
    assert axis==0
    axis,variation=builder.midplane_residual_axis(bounds,coordinates,np.column_stack((coordinates[:,0],3*coordinates[:,3])))
    assert axis==3 and variation[3]==3.


def test_correct_exercise_cell_passes_prices_without_clipping(monkeypatch):
    # Deep exercised calls, negligible puts, negative carry: controlled exact
    # reference data exercises all 881 validation points and all seven gates.
    class ExerciseOracle:
        max_solve_seconds=120
        def value(self,right,point):
            return (math.exp(point[0])-1 if right=="CALL" else 0.),0.
    def derivatives(oracle,right,point,deadline=None):
        return np.array([1.,0.,0.,0.]) if right=="CALL" else np.zeros(4),np.zeros(4)
    monkeypatch.setattr(builder,"independent_normalized_greeks",derivatives)
    bounds=np.array([[math.log(2),math.log(2)+1e-4],[.001,.002],[-.002,-.001],[.01,.02]])
    result=builder.build_cell(bounds,ExerciseOracle())
    assert result["accepted"] and result["financial_validation"]["passed"]
    assert result["errors"].max()<=1e-10
    assert all(item["checks"]==2*881 for item in result["financial_validation"]["checks"].values())
    assert len(result["midplane_residual_variation"])==4


def test_violated_inequality_refines_instead_of_admitting_reference_matching_cell(monkeypatch,tmp_path):
    bounds=np.array([[0.,.01],[.1,.2],[0.,.01],[0.,.01]])
    monkeypatch.setattr(builder,"initial_cells",lambda:[bounds])
    class Oracle:
        calls=0
        def __init__(self,*args,**kwargs): pass
    monkeypatch.setattr(builder,"OraclePriceCache",Oracle)
    observed=[]
    def cell(b,oracle,**kwargs):
        observed.append(b.copy())
        if len(observed)>1:
            from qqq_sim.risk.contracts import RiskError
            raise RiskError("RESOURCE_LIMIT")
        return {"accepted":False,"split_axis":2,"errors":np.zeros(2),"financial_validation":{"passed":False}}
    monkeypatch.setattr(builder,"build_cell",cell)
    result=builder.build_price_basis(tmp_path/"basis",budget_seconds=3)
    import json
    saved=json.loads((tmp_path/"basis/basis_checkpoint.json").read_text())
    assert not result["accepted"] and len(saved["pending"])==2
    assert saved["pending"][0]["refinement_reason"]=="CELL_FINANCIAL_INEQUALITY_FAILED"
    assert saved["pending"][0]["bounds"][2]==[0.,.005]
