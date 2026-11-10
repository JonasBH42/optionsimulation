from concurrent.futures.process import BrokenProcessPool
import time
import pytest
from qqq_sim.services.analysis import AnalysisCoordinator,AnalysisIdentity


def _crash_isolated_worker(*args):
    import os
    os._exit(27)


def test_spawned_inference_worker_crash_recovers_for_next_query(monkeypatch):
    import qqq_sim.services.analysis as module
    original=module.run_fast_analysis
    coordinator=AnalysisCoordinator()
    identity=AnalysisIdentity('session','branch',1,'snapshot','selection','campaign','objective','pack','rules')
    try:
        monkeypatch.setattr(module,'run_fast_analysis',_crash_isolated_worker)
        failed=coordinator.submit(identity,{},None)
        with pytest.raises(BrokenProcessPool): failed.result(timeout=20)
        deadline=time.monotonic()+5
        while coordinator.pool is not None and time.monotonic()<deadline:
            time.sleep(.01)
        assert coordinator.pool is None
        monkeypatch.setattr(module,'run_fast_analysis',original)
        recovered=coordinator.submit(identity,{},None)
        result_identity,result=recovered.result(timeout=20)
        assert result_identity==identity and result['reason']=='PACK_MISSING'
        assert coordinator.gate.accepts(result_identity)
    finally:
        coordinator.close()
