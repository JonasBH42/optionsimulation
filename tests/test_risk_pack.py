import json
import shutil
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from qqq_sim.risk.contracts import RiskError,stable_hash
from qqq_sim.risk.pack_loader import ModelPack
from qqq_sim.risk.offline.build import build_model
from qqq_sim.risk.offline.validate import validate_pack,online_import_audit
from qqq_sim.risk.offline.design import reference_fixture_job
from qqq_sim.risk.hold import evaluate_hold
from qqq_sim.risk.qualification import qualify_regimes
from qqq_sim.risk.offline.price_basis import transform_lobatto
from qqq_sim.risk.pricing_fast import tensor_chebyshev


def test_fixture_pack_actual_bytes_and_abstention(tmp_path):
    path=tmp_path/'pack'
    report=build_model({'fixture':True,'case_count':8,'output_dir':str(path),'work_dir':str(tmp_path/'work')})
    assert not report['production_qualified']
    pack=ModelPack.load(path)
    for i in range(100):
        result=evaluate_hold(replace(reference_fixture_job(),x0=95+i/10),pack)
        assert result['reason']=='PACK_NOT_QUALIFIED' and result['hold_p50'] is None
        assert result['pack_id']==pack.pack_id and result['offline_calls']==0
    assert validate_pack(path)['integrity']=='PASS'
    assert online_import_audit()['passed']


def test_validation_receipt_detects_qualification_edit(tmp_path):
    path=tmp_path/'pack'
    build_model({'fixture':True,'case_count':8,'output_dir':str(path),'work_dir':str(tmp_path/'work')})
    validation=json.loads((path/'validation.json').read_text())
    validation['qualified']=True
    (path/'validation.json').write_text(json.dumps(validation),encoding='utf-8')
    with pytest.raises(RiskError,match='PACK_INTEGRITY_FAILURE'):
        ModelPack.load(path)


def test_array_integrity_detects_single_byte_change(tmp_path):
    path=tmp_path/'pack'
    build_model({'fixture':True,'case_count':8,'output_dir':str(path),'work_dir':str(tmp_path/'work')})
    file=path/'design_coordinates.npy'
    data=bytearray(file.read_bytes());data[-1]^=1;file.write_bytes(data)
    with pytest.raises(RiskError,match='PACK_INTEGRITY_FAILURE'):
        ModelPack.load(path)


def test_regime_promotion_is_evidence_conditional_not_hardcoded_false():
    training=np.zeros((8,3))
    row={'regime':'supported','encoding':[0,0,0],'prediction':.5,'reference':.5,'reference_uncertainty':.001,'reference_qualified':True}
    qualified=qualify_regimes(training,['supported']*8,[dict(row) for _ in range(200)],[dict(row) for _ in range(200)],tolerance=.01)
    assert qualified['qualified'] and qualified['regimes']['supported']['eighth_neighbor_distance_99']==0
    insufficient=qualify_regimes(training,['supported']*8,[dict(row) for _ in range(199)],[dict(row) for _ in range(200)],tolerance=.01)
    assert not insufficient['qualified']
    bad={**row,'reference_qualified':False}
    assert not qualify_regimes(training,['supported']*8,[dict(bad) for _ in range(200)],[dict(row) for _ in range(200)],tolerance=.01)['qualified']


def test_tensor_chebyshev_transform_reproduces_polynomial():
    z=np.cos(np.pi*np.arange(4)/3)
    grids=np.meshgrid(z,z,z,z,indexing='ij')
    values=1+2*grids[0]-3*grids[1]**2+.5*grids[2]*grids[3]
    coefficients=transform_lobatto(values)
    point=np.array([.13,-.27,.45,.63])
    expected=1+2*point[0]-3*point[1]**2+.5*point[2]*point[3]
    assert tensor_chebyshev(coefficients,point)==pytest.approx(expected,abs=1e-13)


def test_reference_checkpoint_identity_ignores_only_runtime_budgets():
    job=reference_fixture_job()
    other=replace(job,numerical_settings=replace(job.numerical_settings,max_work_seconds=1,workers=4,memory_budget_mb=512))
    assert job.hash!=other.hash and job.computation_hash==other.computation_hash
    changed=replace(job,numerical_settings=replace(job.numerical_settings,production_paths=1000))
    assert changed.computation_hash!=job.computation_hash
