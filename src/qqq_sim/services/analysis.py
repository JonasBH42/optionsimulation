from __future__ import annotations
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from threading import RLock,Thread


@dataclass(frozen=True)
class AnalysisIdentity:
    session_id: str
    branch_id: str
    state_version: int
    snapshot_hash: str
    selection_hash: str
    campaign_hash: str
    objective_hash: str
    pack_id: str
    rule_hash: str


class ResultGate:
    def __init__(self):
        self.current = None

    def request(self, identity):
        self.current = identity

    def accepts(self, identity):
        return identity is not None and self.current is not None and identity == self.current

    def invalidate(self):
        self.current = None


_packs = {}


def _frozen_pack(pack_path,expected_manifest_hash):
    """Load only the requested byte revision and guard cached mapped files."""
    from pathlib import Path
    from qqq_sim.risk.pack_loader import ModelPack,file_hash
    from qqq_sim.risk.contracts import RiskError
    root=Path(pack_path).resolve()
    manifest=root/'manifest.json'
    if not manifest.is_file():
        raise RiskError('PACK_MISSING',path=str(root))
    if file_hash(manifest)!=expected_manifest_hash:
        raise RiskError('PACK_REVISION_CHANGED',expected_manifest_hash=expected_manifest_hash)
    cache_key=(str(root),expected_manifest_hash)
    cached=_packs.get(cache_key)
    if cached is None:
        pack=ModelPack.load(root)
        if file_hash(manifest)!=expected_manifest_hash:
            raise RiskError('PACK_REVISION_CHANGED',expected_manifest_hash=expected_manifest_hash)
        paths={manifest,root/'manifest.sha256',root/'validation.json',root/'validation.sha256'}
        paths.update(root/artifact['file'] for artifact in pack.manifest.get('artifacts',{}).values())
        states=[]
        for path in sorted(paths):
            stat=path.stat()
            states.append((path,stat.st_size,stat.st_mtime_ns,stat.st_ino))
        _packs.clear()
        _packs[cache_key]=(pack,tuple(states))
    else:
        pack,states=cached
        for path,size,mtime,inode in states:
            if not path.is_file():
                raise RiskError('PACK_REVISION_CHANGED',artifact=str(path))
            stat=path.stat()
            if (stat.st_size,stat.st_mtime_ns,stat.st_ino)!=(size,mtime,inode):
                raise RiskError('PACK_REVISION_CHANGED',artifact=str(path))
    return pack


def run_fast_analysis(identity, job_data, pack_path, campaign=None, universe=(), limits=None, proposal=None):
    """Spawn entry point. This module must never import an offline engine."""
    if not pack_path:
        return identity, {"status": "UNSUPPORTED", "reason": "PACK_MISSING", "hold_probability": None, "managed_probability": None}
    from qqq_sim.risk.contracts import ReferenceJob, RiskLeg
    from qqq_sim.risk.hold import evaluate_hold
    pack = _frozen_pack(pack_path,identity.pack_id)
    job=ReferenceJob.from_dict(job_data)
    if campaign is None:
        return identity, evaluate_hold(job, pack)
    from qqq_sim.risk.account_features import bind_account_features
    campaign=bind_account_features(campaign,limits)
    from qqq_sim.risk.envelope import optimize_envelope, verify_adjustment
    anchor=(limits or {}).get('managed_anchor')
    managed_raw=(limits or {}).get('managed_job')
    managed_job=ReferenceJob.from_dict(managed_raw) if managed_raw else job
    managed_available=anchor is None or anchor.get('status')=='OK'
    if managed_available:
        result=optimize_envelope(managed_job,campaign,limits,tuple(RiskLeg.from_dict(x) for x in universe),pack,hold_job=job)
    else:
        result={'status':'UNSUPPORTED','reason':'MANAGED_ANCHOR_UNAVAILABLE','hold':evaluate_hold(job,pack),'managed_probability':None}
    if anchor is not None:result['managed_anchor']=anchor
    result["original_benchmarks"]={"hold":campaign.get("p0_hold"),"policy":campaign.get("p0_policy"),"status":campaign.get("benchmark_status","UNKNOWN")}
    result["original_target_usd"]=campaign["target_usd"]
    if proposal is not None:
        broker_preview=proposal.get('exact_broker_preview')
        if broker_preview and broker_preview.get('status')!='OK':
            result['proposal_check']={'status':'REJECTED','reason':'EXACT_BROKER_PREVIEW_FAILED',
                                      'broker_preview':broker_preview,'probability':None}
        else:
            result["proposal_check"]=verify_adjustment(managed_job,campaign,proposal,limits,pack if managed_available else None)
    return identity, result


class AnalysisCoordinator:
    def __init__(self):
        self.gate = ResultGate()
        self.pool = None
        self.pending = None
        self.pending_identity = None
        self.benchmark_futures = set()
        self._lock=RLock()
        self._closed=False

    @staticmethod
    def _new_pool():
        return ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))

    def _discard_pool(self,pool):
        if self.pool is pool:
            self.pool=None
            pool.shutdown(wait=False,cancel_futures=True)

    def _completed(self,future,pool):
        if future.cancelled(): return
        try: error=future.exception()
        except Exception: return
        if isinstance(error,BrokenProcessPool):
            # CPython invokes broken-pool callbacks while holding the executor's
            # shutdown lock. Shutting down synchronously here deadlocks that
            # management thread. Cleanup belongs to a separate short-lived
            # thread, after the callback releases the executor lock.
            Thread(target=self._recover_broken_pool,args=(pool,),daemon=True).start()

    def _recover_broken_pool(self,pool):
        with self._lock:
            self._discard_pool(pool)

    def submit(self, identity, job_data, pack_path, campaign=None, universe=(), limits=None, proposal=None):
        with self._lock:
            if self._closed: raise RuntimeError('ANALYSIS_COORDINATOR_CLOSED')
            self.gate.request(identity)
            if (self.pending is not None and self.pending_identity==identity and not self.pending.cancelled()
                and (not self.pending.done() or self.pending.exception() is None)):
                return self.pending
            self.pending_identity=identity
            if self.pending is not None: self.pending.cancel()
            if self.pool is None: self.pool=self._new_pool()
            pool=self.pool
            try:
                future=pool.submit(run_fast_analysis,identity,job_data,pack_path,campaign,universe,limits,proposal)
            except BrokenProcessPool:
                self._discard_pool(pool)
                self.pool=self._new_pool(); pool=self.pool
                future=pool.submit(run_fast_analysis,identity,job_data,pack_path,campaign,universe,limits,proposal)
            self.pending=future
            future.add_done_callback(lambda result,owner=pool:self._completed(result,owner))
            return future

    def submit_benchmark(self,request):
        """Independent frozen work; changing the selected draft cannot cancel it."""
        from qqq_sim.services.benchmarks import run_initial_benchmark
        with self._lock:
            if self._closed: raise RuntimeError('ANALYSIS_COORDINATOR_CLOSED')
            if self.pool is None: self.pool=self._new_pool()
            pool=self.pool
            try:
                future=pool.submit(run_initial_benchmark,request)
            except BrokenProcessPool:
                self._discard_pool(pool)
                self.pool=self._new_pool(); pool=self.pool
                future=pool.submit(run_initial_benchmark,request)
            self.benchmark_futures.add(future)
            future.add_done_callback(lambda result,owner=pool:self._completed(result,owner))
            future.add_done_callback(self.benchmark_futures.discard)
            return future

    def close(self):
        with self._lock:
            self._closed=True
            self.gate.invalidate()
            self.pending_identity=None
            if self.pending is not None: self.pending.cancel()
            for future in tuple(self.benchmark_futures): future.cancel()
            if self.pool: self._discard_pool(self.pool)
