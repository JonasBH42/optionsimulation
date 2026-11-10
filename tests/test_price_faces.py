import json
import math
import numpy as np
import pytest
from qqq_sim.risk.contracts import RiskError
from qqq_sim.risk.pricing_fast import PriceBasis
from qqq_sim.risk.offline import price_basis as builder


def cell(xlow, xhigh, value=.5, slope=0., wlow=.1, whigh=.2):
    bounds=np.array([[xlow,xhigh],[wlow,whigh],[0.,.01],[0.,.01]],dtype=np.float64)
    coefficients=np.zeros((2,4,4,4,4),dtype=np.float64)
    coefficients[:,0,0,0,0]=value+slope*(xlow+xhigh)/2
    coefficients[:,1,0,0,0]=slope*(xhigh-xlow)/2
    return {"bounds":bounds,"coefficients":coefficients,"errors":np.zeros(2),"greek_errors":np.zeros((2,4))}


class ConstantOracle:
    def __init__(self,*args,**kwargs):
        self.calls=0
        self.max_solve_seconds=120
    def value(self,right,point):
        self.calls+=1
        return .5,0.


def test_all_shared_face_grid_points_and_independent_derivatives_are_checked():
    oracle=ConstantOracle()
    gate=builder.validate_neighbor_faces(cell(-.2,0),cell(0,.2),oracle=oracle)
    assert gate["price_ok"] and gate["greeks_qualified"]
    assert gate["faces"][0]["validation_points_per_right"]==125
    assert gate["independent_reference_checked"] and oracle.calls>125*2*3


def test_equal_face_prices_do_not_certify_neighboring_derivatives():
    gate=builder.validate_neighbor_faces(cell(-.2,0,slope=.1),cell(0,.2,slope=-.1))
    assert gate["price_ok"] and not gate["greeks_qualified"]
    assert gate["maximum_greek_disagreement"][0,0]==pytest.approx(.2)
    assert gate["maximum_greek_disagreement"][0,1]==pytest.approx(.2)
    assert gate["maximum_price_disagreement"].max()<1e-14


def test_coarse_fine_faces_use_exact_overlap_and_ignore_corner_only_contacts():
    coarse=cell(-.2,0)
    fine1,fine2=cell(0,.2,wlow=.1,whigh=.15),cell(0,.2,wlow=.15,whigh=.2)
    one=builder.validate_neighbor_faces(coarse,fine1)
    two=builder.validate_neighbor_faces(coarse,fine2)
    assert one["faces"][0]["bounds"][1]==[.1,.15]
    assert two["faces"][0]["bounds"][1]==[.15,.2]
    corner=cell(0,.2,wlow=.2,whigh=.3)
    assert not builder.validate_neighbor_faces(coarse,corner)["adjacent"]


def test_online_face_lookup_rejects_price_jump_and_reports_one_sided_greeks():
    left,right=cell(-.2,0),cell(0,.2,value=.51)
    basis=PriceBasis(np.array([left["bounds"],right["bounds"]]),np.array([left["coefficients"],right["coefficients"]]),np.zeros((2,2)),np.zeros((2,2,4)))
    with pytest.raises(RiskError,match="PRICE_BASIS_CROSS_FACE_FAILURE"):
        basis.evaluate("CALL",1.,1.,1.,.005,.005,.15)
    left,right=cell(-.2,0,slope=.1),cell(0,.2,slope=-.1)
    basis=PriceBasis(np.array([left["bounds"],right["bounds"]]),np.array([left["coefficients"],right["coefficients"]]),np.zeros((2,2)),np.zeros((2,2,4)))
    result=basis.evaluate("CALL",1.,1.,1.,.005,.005,.15,greeks=True)
    assert result["cell"]==0 and result["price"]==.5
    assert result["greeks_reason"]=="GREEK_DISCONTINUITY"
    assert "delta" not in result["greeks"] and result["greek_intervals"]["delta"][0]<0<result["greek_intervals"]["delta"][1]


def fake_build(bounds,oracle,**kwargs):
    result=cell(*bounds[0],value=.6 if bounds[0,0]>=0 and np.diff(bounds[0])[0]>.15 else .5)
    result["bounds"]=bounds
    return {**result,"accepted":True,"validation_count":881,"split_axis":0,"greeks_status":"VALIDATED","greeks_qualified":True}


def prepare_builder(monkeypatch):
    monkeypatch.setattr(builder,"initial_cells",lambda:[cell(-.2,0)["bounds"],cell(0,.2)["bounds"]])
    monkeypatch.setattr(builder,"OraclePriceCache",ConstantOracle)
    monkeypatch.setattr(builder,"build_cell",fake_build)


def test_failed_cross_face_cell_is_refined_before_admission(tmp_path,monkeypatch):
    prepare_builder(monkeypatch)
    result=builder.build_price_basis(tmp_path/"basis",budget_seconds=30)
    assert result["status"]=="COMPLETE" and len(result["accepted"])==3
    assert {item["id"] for item in result["accepted"]}=={"0","1/0","1/1"}
    assert all(item["greeks_qualified"] for item in result["accepted"])
    assert any(item.get("refinement_reason")=="CROSS_FACE_PRICE_REFERENCE_FAILED" for item in result["accepted"])
    assert all(item["cross_face_status"]=="VALIDATED" for item in result["accepted"])
    resumed=builder.build_price_basis(tmp_path/"basis",budget_seconds=30)
    assert resumed["accepted"]==result["accepted"]


def test_interrupted_face_check_cannot_publish_unchecked_cell(tmp_path,monkeypatch):
    prepare_builder(monkeypatch)
    actual=builder.validate_neighbor_faces
    def interrupted(*args,**kwargs):
        raise RiskError("RESOURCE_LIMIT","Stop during face validation")
    monkeypatch.setattr(builder,"validate_neighbor_faces",interrupted)
    result=builder.build_price_basis(tmp_path/"basis",budget_seconds=30)
    assert result["status"]=="CHECKPOINTED" and len(result["accepted"])==1 and result["pending_cells"]==1
    saved=json.loads((tmp_path/"basis"/"basis_checkpoint.json").read_text())
    assert saved["accepted"][0]["id"]=="0"
    monkeypatch.setattr(builder,"validate_neighbor_faces",actual)
    assert builder.build_price_basis(tmp_path/"basis",budget_seconds=30)["status"]=="COMPLETE"


def test_finite_difference_uncertainty_is_amplified_for_gamma():
    class Uncertain(ConstantOracle):
        def value(self,right,point):
            return .5,1e-8
    values,uncertainty=builder.independent_normalized_greeks(Uncertain(),"CALL",np.array([0.,.15,.005,.005]))
    assert not values.any()
    assert uncertainty[1]>.5  # Four price-error terms divided by the final h_S^2.
    assert uncertainty[0]>0 and uncertainty[2]>0 and uncertainty[3]>0


def test_greek_only_face_failure_keeps_prices_and_marks_derivatives_unavailable(tmp_path,monkeypatch):
    prepare_builder(monkeypatch)
    def derivative_jump(bounds,oracle,**kwargs):
        result=cell(*bounds[0],slope=.1 if bounds[0,0]<0 else -.1)
        return {**result,"accepted":True,"validation_count":881,"split_axis":0,"greeks_status":"VALIDATED","greeks_qualified":True}
    monkeypatch.setattr(builder,"build_cell",derivative_jump)
    result=builder.build_price_basis(tmp_path/"basis",budget_seconds=30)
    assert result["status"]=="COMPLETE" and len(result["accepted"])==2
    assert all(not item["greeks_qualified"] for item in result["accepted"])
    for item in result["accepted"]:
        with np.load(tmp_path/"basis"/item["file"],allow_pickle=False) as arrays:
            assert arrays["errors"].max()<1e-10
            assert arrays["greek_errors"][:,0].min()>=.2


def test_each_prescribed_price_validation_point_also_receives_derivative_check(monkeypatch):
    calls=[]
    def derivative(oracle,right,point,deadline=None):
        calls.append((right,tuple(point)))
        return np.zeros(4),np.zeros(4)
    monkeypatch.setattr(builder,"independent_normalized_greeks",derivative)
    result=builder.build_cell(cell(-.02,0)["bounds"],ConstantOracle())
    # A positive constant call price agrees with this fake oracle but violates
    # strike monotonicity; price agreement cannot bypass the financial checks.
    assert not result["accepted"] and result["validation_count"]==625+256
    assert not result["financial_validation"]["checks"]["STRIKE_MONOTONICITY"]["passed"]
    assert len(calls)==2*(625+256)
