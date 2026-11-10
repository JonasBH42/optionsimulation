from types import SimpleNamespace
import numpy as np
import pytest
from qqq_sim.risk.contracts import RiskError
from qqq_sim.risk.offline.training_store import NumericRows,UniverseRows,ControlCases,ControlLabelCache
from qqq_sim.risk.offline.train import train_network
from qqq_sim.risk.offline.design import reference_fixture_job
from qqq_sim.risk.offline.control_oracle import ManagedState


def test_streamed_rows_and_shared_token_pool(tmp_path):
    rows=NumericRows(tmp_path/"features.bin",3)
    for number in range(300):rows.append([number,number+1,number+2])
    assert isinstance(rows.array(),np.memmap)
    np.testing.assert_array_equal(np.asarray(rows)[[0,299]],[[0,1,2],[299,300,301]])
    tokens=UniverseRows(tmp_path/"tokens")
    first=np.zeros((512,16));second=first.copy();second[0,0]=1
    for _ in range(100):tokens.append(first)
    tokens.append(second)
    assert len(tokens.pool)==2 and tokens.shape==(101,512,16)
    np.testing.assert_array_equal(tokens[[0,100]][:,0,0],[0,1])
    rows.close();tokens.close()


def test_disk_batches_fit_identical_numeric_head(tmp_path):
    features=np.random.default_rng(42).normal(size=(12,5))
    labels=np.linspace(0,1,12)[:,None]
    disk=NumericRows(tmp_path/"features.bin",5)
    target=NumericRows(tmp_path/"target.bin",1)
    for x,y in zip(features,labels):disk.append(x);target.append(y)
    expected=train_network(features,labels,list(range(8)),list(range(8,12)),kind="q_eval",output_size=1,epochs=2)
    actual=train_network(disk,target,list(range(8)),list(range(8,12)),kind="q_eval",output_size=1,epochs=2,training_directory=tmp_path/"training")
    for name,array in expected["arrays"].items():np.testing.assert_array_equal(array,actual["arrays"][name])
    assert isinstance(actual["arrays"]["q_eval_training_encodings"],np.memmap)
    disk.close();target.close()


def test_case_and_label_cache_roundtrip_and_hash_rejection(tmp_path):
    job=reference_fixture_job()
    state=ManagedState(job,{"target_usd":"150"},job.signed_legs)
    cases=ControlCases(tmp_path/"cases")
    cases.append((state,{"id":"first"}));cases.append((state,{"id":"second"}))
    restored,action=cases[1]
    assert restored.job.hash==job.hash and action["id"]=="second"
    assert len(list((tmp_path/"cases").glob("*.json")))==1
    cache=ControlLabelCache(tmp_path/"labels.sqlite3")
    cache.put("label",{"value":.7})
    assert cache.get("label")=={"value":.7}
    cache.connection.execute("UPDATE labels SET body=?",('{"value":0.2}',));cache.connection.commit()
    with pytest.raises(RiskError,match="CHECKPOINT_MISMATCH"):cache.get("label")
    cases.close();cache.close()


def test_disk_reserve_guard_precedes_allocation(tmp_path,monkeypatch):
    import qqq_sim.risk.offline.training_store as module
    monkeypatch.setattr(module.shutil,"disk_usage",lambda path:SimpleNamespace(total=100<<30,free=1<<30))
    with pytest.raises(RiskError) as error:NumericRows(tmp_path/"large.bin",1032)
    assert error.value.code=="RESOURCE_LIMIT"
    assert not (tmp_path/"large.bin").exists()


def test_epoch_resume_restores_adam_best_weights_and_shuffle_exactly(tmp_path,monkeypatch):
    from qqq_sim.risk.offline.train_checkpoint import TrainingCheckpoint
    features=np.random.default_rng(7).normal(size=(32,5))
    labels=np.linspace(0,1,32)[:,None]
    options={"kind":"q_eval","output_size":1,"epochs":4}
    expected=train_network(features,labels,list(range(24)),list(range(24,32)),**options)
    original=TrainingCheckpoint.save
    def interrupt(checkpoint,metadata,arrays):
        original(checkpoint,metadata,arrays)
        if metadata.get("next_epoch")==1:raise RiskError("RESOURCE_LIMIT")
    monkeypatch.setattr(TrainingCheckpoint,"save",interrupt)
    with pytest.raises(RiskError):train_network(features,labels,list(range(24)),list(range(24,32)),training_directory=tmp_path,**options)
    monkeypatch.setattr(TrainingCheckpoint,"save",original)
    actual=train_network(features,labels,list(range(24)),list(range(24,32)),training_directory=tmp_path,**options)
    assert actual["resumed_completed_epochs"]==1 and actual["history"]==expected["history"]
    for name,value in expected["arrays"].items():np.testing.assert_array_equal(value,actual["arrays"][name])
    with pytest.raises(RiskError) as error:
        train_network(features+1,labels,list(range(24)),list(range(24,32)),training_directory=tmp_path,**options)
    assert error.value.code=="TRAINING_CHECKPOINT_MISMATCH"


def test_scaler_progress_resumes_without_restarting_completed_rows(tmp_path,monkeypatch):
    import qqq_sim.risk.offline.train as module
    import json
    features=np.random.default_rng(11).normal(size=(2050,3));labels=np.full((2050,1),.4)
    options={"kind":"q_eval","output_size":1,"epochs":1}
    expected=train_network(features,labels,list(range(2048)),[2048,2049],**options)
    ticks=iter((0.,2.))
    monkeypatch.setattr(module,"time",SimpleNamespace(monotonic=lambda:next(ticks)))
    with pytest.raises(RiskError):train_network(features,labels,list(range(2048)),[2048,2049],deadline=1.,training_directory=tmp_path,**options)
    metadata=json.loads((tmp_path/"training_checkpoint.json").read_text())
    assert metadata["phase"]=="SCALER" and metadata["seen"]==1024
    actual=train_network(features,labels,list(range(2048)),[2048,2049],training_directory=tmp_path,**options)
    for name,value in expected["arrays"].items():np.testing.assert_array_equal(value,actual["arrays"][name])
