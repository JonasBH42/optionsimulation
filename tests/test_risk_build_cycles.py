from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest
from qqq_sim.risk.contracts import RiskError,canonical
from qqq_sim.risk.offline.design import design_record,latin_hypercube,reference_fixture_job
from qqq_sim.risk.offline.adaptive import augmented_records,prepare_next_cycle
from qqq_sim.risk.offline.control_layers import decision_schedule,complete_minute_layers,finite_control_benchmark
from qqq_sim.risk.offline.control_oracle import ManagedState,transition


def _campaign(job):
    return {"net_cash_usd":"-200","gross_cash_usd":"-200","target_usd":"150","remaining_expense_budget_usd":"100","hard_cutoff_utc_us":job.fixing_utc_us,"active_deadline_utc_us":job.fixing_utc_us-1,"remaining_actions":12,"remaining_rolls":3}


def test_adaptive_adds_only_new_training_and_repeats_deterministically(tmp_path):
    lhs=latin_hypercube(4)
    records=[design_record(i,lhs["coordinates"][i]) for i in range(4)]
    original=json.dumps(records,sort_keys=True)
    failures=[{"regime":"bad","p99_error":.1,"case_indices":[1]}]
    added=augmented_records(records,failures,1)
    assert len(added)==8192 and added[0]["index"]==4 and added[-1]["index"]==8195
    assert all(row["split"]=="train" and row["adaptive_parent_index"]==1 for row in added)
    assert added==augmented_records(records,failures,1)
    assert json.dumps(records,sort_keys=True)==original
    assert any(row["family"]!=records[1]["family"] for row in added)
    capped=prepare_next_cycle(tmp_path,records,[{"case_index":1,"regime":"bad","prediction":.8,"reference":.1}],6)
    assert capped["status"]=="ADAPTIVE_BUDGET_EXHAUSTED" and capped["next_cycle"] is None


def test_minute_layers_cover_each_original_split_and_preserve_target(tmp_path):
    base=reference_fixture_job(duration_seconds=300)
    cases=[]
    for i,split in enumerate(("train","calibration","validation")):
        job=replace(base,snapshot_id=f"SYNTHETIC:{split}")
        cases.append({"job":job,"campaign":_campaign(job),"universe":job.signed_legs,"record":{"index":i,"split":split}})
    layers,report=complete_minute_layers(cases,SimpleNamespace(price_basis=None),checkpoint_dir=tmp_path)
    assert report["status"]=="COMPLETE" and report["layers"]==[1,2,3,4]
    assert len(layers)==12
    for count in range(1,5):
        subset=[case for case in layers if len(decision_schedule(case["job"],case["campaign"]))==count]
        assert {case["record"]["split"] for case in subset}=={"train","calibration","validation"}
        assert all(case["campaign"]["target_usd"]=="150" for case in subset)


def test_control_benchmark_never_shortens_horizon_and_converges_exact_terminal(monkeypatch):
    import qqq_sim.risk.offline.control_layers as module
    job=reference_fixture_job(duration_seconds=300)
    state=ManagedState(job,_campaign(job),job.signed_legs)
    outside=finite_control_benchmark(state,SimpleNamespace(price_basis=None),max_decisions=2)
    assert outside["status"]=="BENCHMARK_SUBSET_OUTSIDE_DOMAIN"
    short=replace(job,eligible_intervals=((job.t0_utc_us,job.t0_utc_us+61_000_000),))
    state=ManagedState(short,_campaign(short),short.signed_legs)
    monkeypatch.setattr(module,"generate_actions",lambda *a,**k:{"truncated":False,"actions":[{"id":"exact"}]})
    monkeypatch.setattr(module,"transition",lambda current,*a,**k:replace(current,success=True))
    result=finite_control_benchmark(state,SimpleNamespace(price_basis=None))
    assert result["status"]=="CONVERGED_FINITE_SUBSET_BENCHMARK"
    assert result["value"]==pytest.approx(1) and len(result["refinements"])==3


def test_transition_executes_latency_and_one_failed_close_as_separate_stresses():
    job=reference_fixture_job(sigma_path=0)
    action={"id":"close","family":"FULL_CLOSE","legs":canonical(tuple(replace(leg,quantity=-leg.quantity) for leg in job.signed_legs)),"eligible_utc_us":job.t0_utc_us+60_000_000,"limit_cost_usd":"1000","max_total_fees_usd":"50","stop_after_fill":False}
    campaign={**_campaign(job),"forced_first_liquidation_nonfill":True}
    state=transition(ManagedState(job,campaign,job.signed_legs),action,0,SimpleNamespace(price_basis=None))
    assert state.job.signed_legs and state.events[-1]["conditional_stress"]=="FORCED_FIRST_LIQUIDATION_NONFILL"
    assert state.campaign["net_cash_usd"]=="-200"
    state=transition(state,{**action,"eligible_utc_us":job.t0_utc_us+120_000_000},0,SimpleNamespace(price_basis=None))
    assert not state.job.signed_legs
    delayed=transition(ManagedState(job,{**_campaign(job),"extra_latency_seconds":60},job.signed_legs),{**action,"family":"HOLD","legs":[]},0,SimpleNamespace(price_basis=None))
    assert delayed.job.t0_utc_us==job.t0_utc_us+120_000_000


def test_pipeline_semantic_change_rejects_checkpoint_and_resources_retry(tmp_path,monkeypatch):
    import qqq_sim.risk.offline.pipeline as module
    from qqq_sim.risk.offline.build import build_model
    config={"research":True,"case_count":3,"budget_seconds":10,"epochs":1,"work_dir":str(tmp_path/"work"),"output_dir":str(tmp_path/"pack")}
    calls=[]
    def factory(record,**kwargs):
        job=reference_fixture_job()
        return {"job":job,"campaign":_campaign(job),"universe":job.signed_legs,"record":record,"scope":"TEST","generator_limitations":[]}
    def limited(*args,**kwargs):
        calls.append(1)
        return {"status":"UNAVAILABLE","reason":"RESOURCE_LIMIT","hold_p50":None}
    monkeypatch.setattr(module,"materialize_case",factory)
    monkeypatch.setattr(module,"calculate_reference",limited)
    assert build_model(config)["status"]=="CHECKPOINTED"
    assert build_model({**config,"budget_seconds":20})["status"]=="CHECKPOINTED"
    assert len(calls)==2
    assert list((tmp_path/"work"/"cases"/"attempts").glob("*.json"))
    with pytest.raises(RiskError,match="CHECKPOINT_MISMATCH"):
        build_model({**config,"epochs":2})


def test_policy_rollout_completed_paths_resume_exactly(tmp_path,monkeypatch):
    import qqq_sim.risk.offline.control_oracle as module
    job=reference_fixture_job()
    initial=ManagedState(job,_campaign(job),job.signed_legs)
    action={"id":"first"}
    pack=SimpleNamespace(price_basis=None,basis_identity="basis-1")
    def terminal(state,action,normal,*args):return replace(state,success=normal>0,failed=normal<=0)
    monkeypatch.setattr(module,"transition",terminal)
    expected=module.rollout(initial,action,lambda state:action,pack,paths=500)
    count=0
    def interrupted(*args):
        nonlocal count
        count+=1
        if count==130:raise RiskError("RESOURCE_LIMIT")
        return terminal(*args)
    monkeypatch.setattr(module,"transition",interrupted)
    with pytest.raises(RiskError):module.rollout(initial,action,lambda state:action,pack,paths=500,checkpoint=tmp_path/"paths.json",policy_identity="frozen")
    saved=json.loads((tmp_path/"paths.json").read_text())
    assert saved["completed_paths"]==129
    monkeypatch.setattr(module,"transition",terminal)
    actual=module.rollout(initial,action,lambda state:action,pack,paths=500,checkpoint=tmp_path/"paths.json",policy_identity="frozen")
    assert actual==expected
    with pytest.raises(RiskError) as error:module.rollout(initial,action,lambda state:action,SimpleNamespace(price_basis=None,basis_identity="different"),paths=500,checkpoint=tmp_path/"paths.json",policy_identity="frozen")
    assert error.value.code=="POLICY_CHECKPOINT_MISMATCH"


def test_offline_preview_budget_never_mutates_persisted_account_context():
    from qqq_sim.risk.offline.control_oracle import fresh_account_limits
    from qqq_sim.broker.service import Broker
    context={"broker_state":Broker().to_dict(),"campaign_id":"original","_account_preview_work_budget":"spent transient"}
    job=reference_fixture_job()
    state=ManagedState(job,_campaign(job),job.signed_legs,account_context=context)
    fresh=fresh_account_limits(state)
    assert "_account_preview_work_budget" not in fresh
    fresh["_account_preview_work_budget"]="new"
    assert context["_account_preview_work_budget"]=="spent transient"


def test_price_oracle_default_uses_full_remaining_stage_not_hidden_120s_cap(tmp_path,monkeypatch):
    import qqq_sim.risk.offline.price_basis as module
    budgets=[]
    def oracle(directory,epsilon,max_solve_seconds):
        budgets.append(max_solve_seconds)
        return SimpleNamespace(calls=0,max_solve_seconds=max_solve_seconds)
    def incomplete(*args,**kwargs):raise RiskError("RESOURCE_LIMIT","unfinished PDE",steps=10)
    monkeypatch.setattr(module,"OraclePriceCache",oracle)
    monkeypatch.setattr(module,"build_cell",incomplete)
    first=module.build_price_basis(tmp_path/"shared",budget_seconds=500,max_initial_cells=1)
    second=module.build_price_basis(tmp_path/"explicit",budget_seconds=500,max_initial_cells=1,max_solve_seconds=17)
    assert budgets==[500,17]
    assert first["last_stop"]["unfinished_single_pde_solve_restarts"] is False
    assert first["last_stop"]["details"]["steps"]==10
    assert second["per_oracle_budget_seconds"]==17
