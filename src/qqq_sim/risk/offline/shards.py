"""Pure spawned shard workers; no prices, training, broker I/O or GUI state."""
from __future__ import annotations
from collections import Counter
import math
import time
import numpy as np
from ..contracts import RiskError,YEAR_US
from ..payoff import payoff_array,region_membership

_context=None


def initialize(context):
    global _context
    _context=context


def run_initialized(shard):
    try:
        return run_shard(_context,shard)
    except RiskError as exc:
        return {"shard":shard,"error":exc.as_dict()}


def run_shard(context,shard,clock=time.monotonic):
    job,times,region_variants,paths,phase,monitors,domain,deadline=context
    settings=job.numerical_settings
    targets=tuple(map(float,job.targets))
    reports=job.reports
    variant_count=len(region_variants)*len(monitors)
    n=min(settings.shard_size,paths-shard*settings.shard_size)
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(settings.seed,spawn_key=(phase,shard))))
    spot=np.full(n,job.x0,dtype=np.float64)
    sentinel=np.iinfo(np.int64).max
    hits=np.full((variant_count,len(targets),n),sentinel,dtype=np.int64)
    previous=job.t0_utc_us
    deterministic=job.sigma_path==0
    for instant in times:
        if clock()>deadline:
            raise RiskError("RESOURCE_LIMIT","Path shard interrupted before commit")
        instant=int(instant)
        dt=(instant-previous)/YEAR_US
        if dt>0:
            drift=(job.rate_cc-job.dividend_yield_cc if job.state_kind=="SPOT" else 0)-job.sigma_path**2/2
            normal=0. if deterministic else rng.standard_normal(n)
            spot*=np.exp(drift*dt+job.sigma_path*math.sqrt(dt)*normal)
        if not np.all(np.isfinite(spot)):
            raise RiskError("NUMERICAL_DOMAIN_UNRESOLVED")
        if domain is not None and (np.any(spot<domain[0]) or np.any(spot>domain[1])):
            raise RiskError("NUMERICAL_DOMAIN_UNRESOLVED","A simulated state exited the validated domain; no clipping was applied")
        for ri,regions in enumerate(region_variants):
            for mi,monitor in enumerate(monitors):
                vi=ri*len(monitors)+mi
                if instant in monitor:
                    for ti,target in enumerate(targets):
                        found=(hits[vi,ti]==sentinel)&region_membership(regions[(instant,target)],spot)
                        hits[vi,ti,found]=instant
        previous=instant
    counts=np.zeros((variant_count,len(targets),len(reports)),dtype=np.int64)
    terminal=np.zeros((variant_count,len(targets)),dtype=np.int64)
    histograms=[{} for _ in range(variant_count*len(targets))]
    disagreements=np.zeros((max(0,variant_count-1),len(targets),len(reports)),dtype=np.int64)
    for vi in range(variant_count):
        for ti,target in enumerate(targets):
            for di,report_at in enumerate(reports):
                counts[vi,ti,di]=np.count_nonzero(hits[vi,ti]<=report_at)
            terminal[vi,ti]=np.count_nonzero((hits[vi,ti]==sentinel)&(payoff_array(job.signed_legs,spot,job.entry_cash_usd)>=target))
            unique,nums=np.unique(hits[vi,ti][hits[vi,ti]!=sentinel],return_counts=True)
            histograms[vi*len(targets)+ti]=dict(zip(map(int,unique),map(int,nums)))
            if vi>0:
                for di,report_at in enumerate(reports):
                    disagreements[vi-1,ti,di]=np.count_nonzero((hits[vi,ti]<=report_at)!=(hits[vi-1,ti]<=report_at))
    from .resource_metrics import process_memory
    return {"shard":shard,"paths":n,"counts":counts,"terminal":terminal,"histograms":histograms,"disagreements":disagreements,"process_memory":process_memory()}
