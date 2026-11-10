from dataclasses import replace
import copy
import math

import numpy as np
import pytest
import torch

from qqq_sim.risk.contracts import RiskError,stable_hash
from qqq_sim.risk.hold import survival_bins
from qqq_sim.risk.offline.design import reference_fixture_job
from qqq_sim.risk.offline.hit_oracle import observation_grid
from qqq_sim.risk.offline.pipeline import _hold_label
from qqq_sim.risk.offline.train import (HOLD_LABEL_SIZE,HOLD_LABEL_BASE_SIZE,HOLD_WITHIN_BIN_SLOTS,
    hold_loss_components,hold_training_loss,train_network,within_bin_validation)


def label(paths, hits):
    result=np.zeros(HOLD_LABEL_SIZE)
    result[0],result[576]=hits,paths-hits
    result[577:580]=.5
    result[580]=1
    return result


def test_multinomial_loss_weights_independent_paths_not_cases():
    labels=torch.tensor(np.array([label(1,1),label(9,9)]),dtype=torch.float64)
    output=torch.zeros((2,579),dtype=torch.float64)
    output[:,0]=torch.tensor([math.log(.9/.1),math.log(.1/.9)])
    actual=hold_training_loss(output,labels)
    expected=(-math.log(.9)-9*math.log(.1))/10
    assert actual.item()==pytest.approx(expected,abs=1e-7)
    assert abs(actual.item()-(-math.log(.9)-math.log(.1))/2)>.5
    replicated=torch.tensor(np.array([label(1,1)]*10),dtype=torch.float64)
    repeated_output=torch.cat((output[:1],output[1:].repeat(9,1)))
    assert hold_training_loss(repeated_output,replicated).item()==pytest.approx(actual.item())


def test_within_bin_early_and_late_hits_produce_distinct_loss_without_changing_bin_counts():
    early,late=label(100,100),label(100,100)
    early[HOLD_LABEL_BASE_SIZE:HOLD_LABEL_BASE_SIZE+5]=[1,0,.5,100,1000000]
    late[HOLD_LABEL_BASE_SIZE:HOLD_LABEL_BASE_SIZE+5]=[1,0,.5,0,1000000]
    output=torch.zeros((2,579),dtype=torch.float64)
    output[:,0]=math.log(math.expm1(1))
    terms=hold_loss_components(output,torch.tensor(np.array([early,late])))
    np.testing.assert_allclose(terms["multinomial_nll"],[terms["multinomial_nll"][0]]*2)
    predicted=-math.expm1(-.5)
    np.testing.assert_allclose(terms["within_bin_mse"],[(predicted-1)**2,predicted**2])


def test_empty_survivor_set_does_not_fabricate_conditional_terminal_observation():
    expected=torch.tensor(np.array([label(100,100)]))
    output=torch.zeros((1,579),dtype=torch.float64,requires_grad=True)
    loss=hold_training_loss(output,expected)
    loss.backward()
    assert output.grad[0,-1]==0


@pytest.mark.parametrize("hits",[0,100])
def test_extreme_hazards_and_zero_counts_have_finite_exact_loss_and_gradient(hits):
    output=torch.zeros((1,579),dtype=torch.float64,requires_grad=True)
    with torch.no_grad(): output[0,0]=-1000
    expected=torch.tensor(np.array([label(100,hits)]))
    loss=hold_training_loss(output,expected)
    loss.backward()
    assert torch.isfinite(loss) and torch.all(torch.isfinite(output.grad))
    assert loss.item()==pytest.approx(1000. if hits else 0.)


def test_legacy_schema_requires_explicit_unqualified_compatibility():
    output=torch.zeros((1,579),dtype=torch.float64)
    expected=torch.tensor(np.array([label(100,50)[:1156]]))
    with pytest.raises(RiskError,match="HOLD_LABEL_SCHEMA_MISMATCH"):
        hold_training_loss(output,expected)
    assert torch.isfinite(hold_training_loss(output,expected,allow_legacy=True))
    with pytest.raises(RiskError,match="HOLD_LABEL_SCHEMA_MISMATCH"):
        train_network(np.zeros((2,3)),np.array([label(100,50)[:1156]]*2),[0],[1],epochs=1)


def reference_result(job, histogram):
    times,_=observation_grid(job,15)
    return {"production_paths":100,"grid_seconds":15,"grid_hash":stable_hash(times.tolist()),
            "first_hit_histograms":[histogram],"settlement_only":0.,"terminal_cent_pop":.5}


def test_sampled_deadlines_are_existing_frozen_observations_and_retain_early_late_hits():
    job=reference_fixture_job(duration_seconds=3600)
    bins=survival_bins(job,15)
    chosen=next(b for b,times in enumerate(bins) if len(times)>1)
    first,last=map(int,(bins[chosen][0],bins[chosen][-1]))
    early_result=reference_result(job,{str(first):100})
    late_result=reference_result(job,{str(last):100})
    frozen=copy.deepcopy(early_result)
    early,late=_hold_label(job,early_result),_hold_label(job,late_result)
    np.testing.assert_array_equal(early[:1156],late[:1156])
    samples=early[HOLD_LABEL_BASE_SIZE:].reshape(HOLD_WITHIN_BIN_SLOTS,5)
    later=late[HOLD_LABEL_BASE_SIZE:].reshape(HOLD_WITHIN_BIN_SLOTS,5)
    valid=samples[:,0]==1
    selected=np.flatnonzero(valid&(samples[:,1]==chosen))[0]
    assert samples[selected,3]==100 and later[selected,3]==0
    times,_=observation_grid(job,15)
    assert set(samples[valid,4].astype(np.int64))<=set(times)
    assert all(0<f<1 for f in samples[valid,2])
    np.testing.assert_array_equal(_hold_label(job,early_result),early)
    assert early_result==frozen


def test_changed_exact_reporting_time_cannot_reuse_old_histogram():
    job=reference_fixture_job(duration_seconds=3600)
    result=reference_result(job,{str(job.t0_utc_us):100})
    changed=replace(job,reporting_times_utc_us=(job.t0_utc_us+3_333_333,))
    with pytest.raises(RiskError) as error:
        _hold_label(changed,result)
    assert error.value.code=="HOLD_LABEL_GRID_MISMATCH"
    invalid=reference_result(job,{str(job.t0_utc_us+3_333_333):100})
    with pytest.raises(RiskError,match="HOLD_LABEL_GRID_MISMATCH"):
        _hold_label(job,invalid)


def test_versioned_labels_fit_and_report_schema():
    features=np.random.default_rng(24).normal(size=(4,3))
    labels=np.array([label(100,20),label(200,150),label(100,30),label(300,240)])
    result=train_network(features,labels,[0,1],[2,3],epochs=1)
    assert result["hold_label_schema"]["version"]=="QQQ_HOLD_LABEL_V2"
    assert result["hold_loss_schema"]["version"]=="QQQ_HOLD_LOSS_V2"
    assert not result["legacy_hold_labels_unqualified"]


def test_heldout_within_bin_gate_counts_cases_once_and_detects_wrong_shape():
    one=label(200000,100000)
    one[HOLD_LABEL_BASE_SIZE:HOLD_LABEL_BASE_SIZE+5]=[1,0,.5,50000,1000000]
    labels=np.repeat(one[None,:],200,axis=0)
    output=np.zeros((200,579))
    # Exact sampled CDF .25 at half the bin's observations.
    hazard=-2*math.log(.75)
    output[:,0]=math.log(math.expm1(hazard))
    gate=within_bin_validation(output,labels)
    assert gate["qualified"] and gate["independent_cases"]==200 and gate["sampled_deadlines"]==200
    few=within_bin_validation(output[:1],labels[:1])
    assert not few["qualified"] and few["independent_cases"]==1
    output[:,0]=math.log(math.expm1(2.))
    failed=within_bin_validation(output,labels)
    assert not failed["qualified"] and failed["reason"]=="WITHIN_BIN_CDF_ERROR_GATE_FAILED"
    assert failed["maximum_absolute_error_plus_reference_uncertainty"]>.3


def test_heldout_sampling_uncertainty_and_singletons_are_not_false_evidence():
    one=label(10,5)
    one[HOLD_LABEL_BASE_SIZE:HOLD_LABEL_BASE_SIZE+5]=[1,0,.5,2,1000000]
    labels=np.repeat(one[None,:],200,axis=0)
    output=np.zeros((200,579));output[:,0]=math.log(math.expm1(-2*math.log(.8)))
    assert not within_bin_validation(output,labels)["qualified"]
    assert within_bin_validation(output,labels,deterministic=np.ones(200,dtype=bool))["qualified"]
    labels[:,HOLD_LABEL_BASE_SIZE:]=0
    singleton=within_bin_validation(output,labels)
    assert not singleton["qualified"] and singleton["singleton_only_cases"]==200
    assert singleton["independent_cases"]==0


def test_pack_validation_rechecks_within_bin_predictions_from_loaded_weights(monkeypatch):
    from types import SimpleNamespace
    from qqq_sim.risk.offline import validate
    from qqq_sim.risk.offline.train import HOLD_LABEL_SCHEMA,HOLD_LOSS_SCHEMA
    import qqq_sim.risk.hold as hold
    one=label(200000,100000)
    one[HOLD_LABEL_BASE_SIZE:HOLD_LABEL_BASE_SIZE+5]=[1,0,.5,50000,1000000]
    labels=np.repeat(one[None,:],200,axis=0)
    outputs=np.zeros((200,579));outputs[:,0]=math.log(math.expm1(-2*math.log(.75)))
    fractions=np.zeros((200,576));fractions[:,0]=1.
    arrays={"hold_validation_features":np.zeros((200,3)),"hold_mean":np.zeros(3),"hold_std":np.ones(3),
        "hold_validation_bin_fractions":fractions,"hold_validation_terminal":np.zeros(200),
        "hold_validation_reference":np.full(200,1-.75**2),"hold_validation_uncertainty":np.zeros(200),
        "hold_within_validation_labels":labels,"hold_within_validation_numerical_uncertainty":np.zeros(200),
        "hold_within_validation_deterministic":np.zeros(200),
        "managed_validation_features":np.zeros((200,3)),"q_eval_mean":np.zeros(3),"q_eval_std":np.ones(3),
        "managed_validation_reference":np.full(200,.5),"managed_validation_uncertainty":np.zeros(200)}
    gates={key:{"qualified":True,"independent_cases":200,"pass_fraction":1.,"independent_policy_rollouts":True} for key in ("price","hold","managed")}
    pack=SimpleNamespace(arrays=arrays,manifest={"hold_label_schema":HOLD_LABEL_SCHEMA,"hold_loss_schema":HOLD_LOSS_SCHEMA},
        validation={"capabilities":gates},pack_id="CONTROLLED_TEST",manifest_hash="CONTROLLED_TEST",require_qualified=lambda capability:None)
    monkeypatch.setattr(validate.ModelPack,"load",lambda path:pack)
    monkeypatch.setattr(validate,"reference_fixtures",lambda:{"status":"PASS"})
    monkeypatch.setattr(validate,"online_import_audit",lambda:{"passed":True})
    monkeypatch.setattr(hold,"network",lambda arrays,prefix,features:outputs if prefix=="hold" else np.zeros((len(features),1)))
    assert validate.validate_pack("unused")["qualified"]
    arrays["hold_within_validation_labels"][:,HOLD_LABEL_BASE_SIZE+3]=40000
    result=validate.validate_pack("unused")
    assert not result["qualified"]
    assert result["loaded_array_prediction_checks"]["hold"]["passed"]  # Same correct headline.
    assert not result["loaded_array_prediction_checks"]["hold"]["within_bin"]["qualified"]
    arrays["hold_within_validation_labels"][:,HOLD_LABEL_BASE_SIZE+3]=50000
    def unsupported_contract(capability):
        raise RiskError("PACK_FEATURE_SCHEMA_MISMATCH")
    pack.require_qualified=unsupported_contract
    unsupported=validate.validate_pack("unused")
    assert not unsupported["qualified"]
    assert unsupported["capabilities"]["hold"]["reason"]=="PACK_FEATURE_SCHEMA_MISMATCH"


def test_coverage_error_range_includes_within_bin_error_and_gate():
    from qqq_sim.risk.qualification import qualify_regimes
    row={"regime":"test","encoding":[0.,0.],"prediction":.5,"reference":.5,"reference_uncertainty":.001,
         "auxiliary_error":.009,"reference_qualified":True}
    calibration=[dict(row) for _ in range(200)]
    validation=[dict(row) for _ in range(200)]
    report=qualify_regimes(np.zeros((8,2)),["test"]*8,calibration,validation,tolerance=.01)
    assert report["qualified"]
    assert report["regimes"]["test"]["p99_absolute_error"]==pytest.approx(.009)
    for item in validation[:3]:item["auxiliary_error"]=.011
    refused=qualify_regimes(np.zeros((8,2)),["test"]*8,calibration,validation,tolerance=.01)
    assert not refused["qualified"] and refused["regimes"]["test"]["pass_fraction"]==.985
