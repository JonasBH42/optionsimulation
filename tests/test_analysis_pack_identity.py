import json
from dataclasses import replace
import pytest

from qqq_sim.services import analysis
from qqq_sim.services.analysis import AnalysisIdentity,run_fast_analysis
from qqq_sim.risk.contracts import RiskError,canonical,stable_hash
from qqq_sim.risk.pack_loader import ModelPack,file_hash
from qqq_sim.risk.offline.design import reference_fixture_job


def write_pack(path,name='original'):
    path.mkdir(exist_ok=True)
    manifest={'schema_version':'QQQ_P50_PACK_V1','pack_id':name,'artifacts':{}}
    validation={'manifest_hash':stable_hash(manifest),'capabilities':{'hold':{'qualified':False,'reason':'TEST_UNQUALIFIED'}}}
    for stem,value in (('manifest',manifest),('validation',validation)):
        target=path/(stem+'.json')
        target.write_text(json.dumps(value),encoding='utf-8')
        (path/(stem+'.sha256')).write_text(file_hash(target),encoding='ascii')
    return file_hash(path/'manifest.json')


@pytest.fixture
def prepared(tmp_path):
    analysis._packs.clear()
    path=tmp_path/'pack'; expected=write_pack(path)
    identity=AnalysisIdentity('session','branch',1,'snapshot','selection','campaign','objective',expected,'rules')
    yield path,identity,canonical(reference_fixture_job())
    analysis._packs.clear()


def test_fast_analysis_reuses_only_identical_frozen_pack(prepared,monkeypatch):
    path,identity,job=prepared; calls=[]; load=ModelPack.load
    def observed(target):
        calls.append(str(target)); return load(target)
    monkeypatch.setattr(ModelPack,'load',observed)
    first=run_fast_analysis(identity,job,str(path))
    second=run_fast_analysis(identity,job,str(path))
    assert first[0]==second[0]==identity
    assert first[1]['reason']==second[1]['reason']=='PACK_NOT_QUALIFIED'
    assert len(calls)==1


def test_different_valid_pack_cannot_use_original_request_identity(prepared):
    path,identity,job=prepared
    write_pack(path,'replacement')
    assert ModelPack.load(path).pack_id=='replacement'
    with pytest.raises(RiskError,match='PACK_REVISION_CHANGED'):
        run_fast_analysis(identity,job,str(path))


def test_replacement_during_load_rejected_before_inference(prepared,monkeypatch):
    path,identity,job=prepared; load=ModelPack.load
    def replaced(target):
        result=load(target)
        write_pack(path,'changed-during-load')
        return result
    monkeypatch.setattr(ModelPack,'load',replaced)
    with pytest.raises(RiskError,match='PACK_REVISION_CHANGED'):
        run_fast_analysis(identity,job,str(path))
    assert not analysis._packs


def test_cached_pack_rejects_changed_validation_receipt(prepared):
    path,identity,job=prepared
    run_fast_analysis(identity,job,str(path))
    validation=json.loads((path/'validation.json').read_text())
    validation['new_evidence']='A new report is a different cached pack state.'
    (path/'validation.json').write_text(json.dumps(validation),encoding='utf-8')
    (path/'validation.sha256').write_text(file_hash(path/'validation.json'),encoding='ascii')
    assert ModelPack.load(path).pack_id=='original'
    with pytest.raises(RiskError,match='PACK_REVISION_CHANGED'):
        run_fast_analysis(identity,job,str(path))
