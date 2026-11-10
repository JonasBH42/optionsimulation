"""Adversarial review of shard receipts and bounded region buffers."""
from dataclasses import replace
import copy
import json
import numpy as np
import pytest

from qqq_sim.risk.contracts import RiskError, stable_hash
from qqq_sim.risk.offline.design import reference_fixture_job
from qqq_sim.risk.offline.hit_oracle import _simulate, observation_grid
from qqq_sim.risk.offline.region_store import RegionStore
from qqq_sim.risk.payoff import Interval


@pytest.fixture
def checkpoint_case(tmp_path):
    job=reference_fixture_job(sigma_path=.2,duration_seconds=3)
    job=replace(job,numerical_settings=replace(job.numerical_settings,shard_size=64),
                optional_targets_usd=(100,),reporting_times_utc_us=(job.t0_utc_us+1_000_000,))
    times,_=observation_grid(job,1)
    variants=[]
    for offset in (0.,.001):
        variants.append({(int(t),float(target)):(Interval(100.+offset+(float(target)-100)/10000,float('inf')),)
                         for t in times for target in job.targets})
    file=tmp_path/'shards.json'
    original=_simulate(job,times,variants,100,10,checkpoint=file)
    return job,times,variants,file,original


def rewrite(file,change,rehash=True):
    saved=json.loads(file.read_text(encoding='utf-8'))
    change(saved)
    if rehash:
        saved['aggregate_sha256']=stable_hash({key:value for key,value in saved.items() if key!='aggregate_sha256'})
    file.write_text(json.dumps(saved),encoding='utf-8')


@pytest.mark.parametrize('change,validation',[
    (lambda s:s['counts'][0][0].__setitem__(0,101),'COUNTS_SHAPE_TYPE_OR_RANGE'),
    (lambda s:s['counts'][0][0].__setitem__(0,1.0),'COUNTS_SHAPE_TYPE_OR_RANGE'),
    (lambda s:s['counts'][0][0].__setitem__(0,-1),'COUNTS_SHAPE_TYPE_OR_RANGE'),
    (lambda s:s.__setitem__('counts',[[[0]]]),'COUNTS_SHAPE_TYPE_OR_RANGE'),
    (lambda s:s['completed'].append(0),'DUPLICATE_OR_UNORDERED_COMPLETED_SHARD'),
    (lambda s:s.__setitem__('completed',[-1]),'COMPLETED_SHARD_RANGE'),
    (lambda s:s.__setitem__('completed',[2]),'COMPLETED_SHARD_RANGE'),
    (lambda s:s['histograms'].pop(),'HISTOGRAM_COUNT'),
    (lambda s:s['histograms'][0].__setitem__('1',1),'HISTOGRAM_TIME_NOT_MONITORED'),
    (lambda s:s['terminal'][0].__setitem__(0,100),'HIT_AND_TERMINAL_CONSERVATION'),
])
def test_rehashed_malformed_aggregate_is_rejected(checkpoint_case,change,validation):
    job,times,variants,file,_=checkpoint_case
    rewrite(file,change)
    with pytest.raises(RiskError) as error:
        _simulate(job,times,variants,100,10,checkpoint=file)
    assert error.value.code=='CHECKPOINT_MISMATCH'
    assert error.value.details['validation']==validation


def test_aggregate_corruption_without_checksum_update_is_rejected(checkpoint_case):
    job,times,variants,file,_=checkpoint_case
    rewrite(file,lambda s:s['histograms'][0].clear(),rehash=False)
    with pytest.raises(RiskError) as error:
        _simulate(job,times,variants,100,10,checkpoint=file)
    assert error.value.details['validation']=='AGGREGATE_CHECKSUM'


def test_untouched_checkpoint_restores_exact_statistics(checkpoint_case):
    job,times,variants,file,original=checkpoint_case
    resumed=_simulate(job,times,variants,100,10,checkpoint=file)
    for field in ('counts','terminal','disagreements'):
        np.testing.assert_array_equal(resumed[field],original[field])
    assert resumed['histograms']==original['histograms']


def test_region_write_buffer_is_fixed_numpy_storage_and_record_limit_is_explicit(tmp_path):
    store=RegionStore({'job':'bounded-review'},block_bytes=4096,max_record_bytes=2048,root=tmp_path)
    for instant in range(100):
        store[(instant,1.)]=(Interval(1.,2.),)
    assert store.folder is not None
    assert isinstance(store._buffer,np.ndarray)
    assert store._buffer.nbytes<=4096
    before=len(store)
    with pytest.raises(RiskError) as error:
        store[(101,1.)]=tuple(Interval(i,i+.5) for i in range(9))
    assert error.value.code=='RESOURCE_LIMIT'
    assert len(store)==before
    store.freeze()
    assert store._buffer.nbytes==0
    assert all(value==(Interval(1.,2.),) for key,value in store.ordered_items())
    assert all(block['bytes']<=4096 for block in store._blocks)
    store.close()
