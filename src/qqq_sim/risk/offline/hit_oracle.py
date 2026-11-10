"""Offline shared-path first-hit calculation with session and terminal separation."""
from __future__ import annotations
import dataclasses
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri, log_ndtr
from ..contracts import ReferenceJob, RiskError, YEAR_US, canonical, stable_hash
from ..payoff import piecewise_regions, region_membership, terminal_regions, payoff_array
from ..pricing_fast import european_price, deterministic_american
from .pricing_oracle import AmericanGrid
from .region_store import RegionStore,region_digest
from .source_identity import implementation_identity


def wilson(k, n, z=1.959963984540054):
    if n <= 0 or not 0 <= k <= n:
        raise RiskError("INVALID_SAMPLE_COUNT")
    p = k/n
    d = 1+z*z/n
    center = (p+z*z/(2*n))/d
    half = z/d*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return [max(0., center-half), min(1., center+half)]


def eligible(job, instant):
    return instant < min(job.last_trading_utc_us, job.fixing_utc_us) and any(a <= instant < b for a, b in job.eligible_intervals)


def observation_grid(job, seconds=60):
    step = int(round(seconds*1_000_000))
    points = {job.t0_utc_us, job.fixing_utc_us, *job.reports}
    for a, b in job.eligible_intervals:
        lo, hi = max(a, job.t0_utc_us), min(b, job.last_trading_utc_us, job.fixing_utc_us)
        if lo >= hi:
            continue
        if a >= job.t0_utc_us:
            points.add(a)
        points.add(hi-1)
        points.update(range(((lo+step-1)//step)*step, hi, step))
        if lo <= job.t0_utc_us < hi:
            points.add(job.t0_utc_us)
    times = np.array(sorted(t for t in points if job.t0_utc_us <= t <= job.fixing_utc_us), dtype=np.int64)
    if len(times) > job.numerical_settings.max_grid_points:
        raise RiskError("RESOURCE_LIMIT", "Observation grid exceeds explicit point budget", points=len(times))
    return times, np.array([eligible(job, int(t)) for t in times], dtype=bool)


def domain_bound(job, lower, upper):
    horizon = (job.fixing_utc_us-job.t0_utc_us)/YEAR_US
    mu = (job.rate_cc-job.dividend_yield_cc if job.state_kind == "SPOT" else 0)-.5*job.sigma_path**2
    l, u = math.log(job.x0/lower)-abs(mu)*horizon, math.log(upper/job.x0)-abs(mu)*horizon
    if job.sigma_path == 0:
        end = job.x0*math.exp(mu*horizon)
        return 0. if lower < min(job.x0, end) and max(job.x0, end) < upper else 1.
    if l <= 0 or u <= 0:
        return 1.
    sd = job.sigma_path*math.sqrt(horizon)
    return min(1., 2*float(ndtr(-l/sd))+2*float(ndtr(-u/sd)))


def initial_domain(job):
    horizon = (job.fixing_utc_us-job.t0_utc_us)/YEAR_US
    sig = max(job.sigma_path, *(leg.sigma for leg in job.signed_legs))
    pad = max(.5, 8*sig*math.sqrt(horizon)+abs(job.rate_cc-job.dividend_yield_cc-sig*sig/2)*horizon)
    lo = min(job.x0, *(float(x.strike) for x in job.signed_legs))*math.exp(-pad)
    hi = max(job.x0, *(float(x.strike) for x in job.signed_legs))*math.exp(pad)
    for _ in range(20):
        if domain_bound(job, lo, hi) <= 1e-8:
            return lo, hi
        expansion = max(.5, math.log(hi/lo)/4)
        lo, hi = lo*math.exp(-expansion), hi*math.exp(expansion)
    raise RiskError("NUMERICAL_DOMAIN_UNRESOLVED")


def _portfolio(job, spot, tau, grids=None):
    value = np.full(np.asarray(spot).shape, float(job.entry_cash_usd))
    error = np.zeros_like(value)
    for leg in job.signed_legs:
        factor = leg.quantity*leg.multiplier
        if leg.sigma == 0:
            price = deterministic_american(leg.right, spot, leg.strike, tau, job.rate_cc, job.dividend_yield_cc) if leg.exercise_style == "AMERICAN" else european_price(leg.right, spot, leg.strike, tau, job.rate_cc, job.dividend_yield_cc, 0, job.state_kind)
            uncertainty = 0.
        elif leg.exercise_style == "EUROPEAN" or leg.right == "CALL" and job.dividend_yield_cc == 0 and job.rate_cc >= 0:
            price = european_price(leg.right, spot, leg.strike, tau, job.rate_cc, job.dividend_yield_cc, leg.sigma, job.state_kind)
            uncertainty = 0.
        else:
            grid = grids[leg.contract_id]
            grid.advance(tau)
            price, uncertainty = grid.evaluate(spot)
        value += factor*price
        error += abs(factor)*uncertainty
    return value, error


def build_regions(job, times, *, nodes=None, time_scale=1, started=None):
    started = time.monotonic() if started is None else started
    domain = initial_domain(job)
    nodes = nodes or job.numerical_settings.space_nodes
    spots = np.unique(np.concatenate((np.exp(np.linspace(math.log(domain[0]), math.log(domain[1]), nodes)), [job.x0], [float(x.strike) for x in job.signed_legs])))
    # Numerical log-grid endpoints round by an ulp: use each grid's actual bounds.
    domain = (float(spots[0]), float(spots[-1]))
    eps_port = min(.005, 1e-4*max(1, abs(float(job.target_usd))))
    weight = sum(abs(x.quantity)*x.multiplier for x in job.signed_legs)
    grids = {x.contract_id: AmericanGrid(x.right, x.strike, x.sigma, job.rate_cc, job.dividend_yield_cc, domain, nodes, eps_port/max(weight, 1), time_scale,started+job.numerical_settings.max_work_seconds)
             for x in job.signed_legs if x.exercise_style == "AMERICAN" and x.sigma > 0 and not (x.right == "CALL" and job.dividend_yield_cc == 0 and job.rate_cc >= 0)}
    if grids:
        actual = next(iter(grids.values())).spots
        spots[0], spots[-1] = actual[0], actual[-1]
        domain = (float(actual[0]), float(actual[-1]))
    store_identity={"job":job.computation_hash,"times":stable_hash(times.tolist()),"domain":domain,"nodes":nodes,"time_scale":time_scale}
    region_block_bytes=min(8*1024*1024,max(1024,int(job.numerical_settings.memory_budget_mb*1024*1024/24)))
    regions, certain, possible = (RegionStore({**store_identity,"kind":kind},block_bytes=region_block_bytes,
        max_record_bytes=max(1024,int(job.numerical_settings.memory_budget_mb*1024*1024/6))) for kind in ("regions","certain","possible"))
    endpoints = {}
    leg_checks=[]
    max_boundary = 0.
    targets = tuple(map(float, job.targets))
    analytic_regions=all(x.exercise_style=="EUROPEAN" or x.right=="CALL" and job.dividend_yield_cc==0 and job.rate_cc>=0 for x in job.signed_legs)
    region_subdivision={"method":"EUROPEAN_ANALYTIC_LIPSCHITZ_SUBDIVISION" if analytic_regions else "AMERICAN_REFINED_LINEAR_INTERPOLANT","evaluations":0,"unresolved_tolerance_intervals":0}
    for instant in reversed(times):
        if time.monotonic()-started > job.numerical_settings.max_work_seconds:
            raise RiskError("RESOURCE_LIMIT", "Reference pricing work budget exceeded", last_time=int(instant))
        tau = (job.fixing_utc_us-int(instant))/YEAR_US
        if instant == job.fixing_utc_us:
            for target in targets:
                terminal=terminal_regions(job.signed_legs, job.entry_cash_usd, target)
                regions[(int(instant), target)] = terminal
                certain[(int(instant), target)] = terminal
                possible[(int(instant), target)] = terminal
            continue
        marks, boundary = _portfolio(job, spots, tau, grids)
        # Restrict boundary assessment to utilized neighborhood, tails separately bounded.
        inner = (spots > domain[0]*1.2) & (spots < domain[1]/1.2)
        max_boundary = max(max_boundary, float(np.max(boundary[inner])) if np.any(inner) else math.inf)
        for target in targets:
            key = (int(instant), target)
            if analytic_regions:
                from .european_regions import adaptive_regions,portfolio_lipschitz
                # Exact formulas allow interval refinement beyond the initial
                # knots; all strikes and deterministic discounted kinks are seeds.
                kinks=[float(x.strike)*math.exp(-(job.rate_cc-job.dividend_yield_cc)*tau) if job.state_kind=="SPOT" else float(x.strike) for x in job.signed_legs if x.sigma==0]
                knots=[domain[0],domain[1],job.x0,*[float(x.strike) for x in job.signed_legs],*[x for x in kinks if domain[0]<x<domain[1]]]
                roundoff=lambda s:64*np.finfo(float).eps*(1+abs(float(job.entry_cash_usd))+sum(abs(x.quantity)*x.multiplier*(np.abs(s)+float(x.strike))*math.exp(max(abs(job.rate_cc),abs(job.dividend_yield_cc))*tau) for x in job.signed_legs))
                adaptive=adaptive_regions(lambda s:_portfolio(job,s,tau,grids)[0]-target,knots,
                                          lambda a,b:portfolio_lipschitz(job,a,b,tau),error=roundoff,
                                          deadline=started+job.numerical_settings.max_work_seconds,
                                          max_evaluations=max(1024,int(job.numerical_settings.memory_budget_mb*1024*1024/128)))
                for name in ("evaluations","unresolved_tolerance_intervals"):
                    region_subdivision[name]+=adaptive[name]
                regions[key],certain[key],possible[key]=adaptive["regions"],adaptive["certain"],adaptive["possible"]
            else:
                regions[key] = piecewise_regions(spots, marks-target)
                certain[key] = piecewise_regions(spots, marks-target-boundary-eps_port)
                possible[key] = piecewise_regions(spots, marks-target+boundary+eps_port)
        if instant in job.reports or instant == job.t0_utc_us:
            endpoints[int(instant)] = {"model_pnl": float(np.interp(job.x0, spots, marks)), "boundary_sensitivity": float(np.interp(job.x0, spots, boundary))}
        if instant==job.t0_utc_us:
            for leg in job.signed_legs:
                if leg.bid is None:
                    continue
                single=dataclasses.replace(job,signed_legs=(leg,),entry_cash_usd=0)
                marked,uncertainty=_portfolio(single,np.array([job.x0]),tau,grids)
                price=float(marked[0])/(leg.quantity*leg.multiplier)
                error=float(uncertainty[0])/(abs(leg.quantity)*leg.multiplier)
                midpoint=(leg.bid+leg.ask)/2
                tolerance=max((leg.ask-leg.bid)/2,eps_port/max(weight,1))+error
                leg_checks.append({"contract_id":leg.contract_id,"model_price":price,"midpoint":midpoint,"residual":price-midpoint,"tolerance":tolerance})
                if abs(price-midpoint)>tolerance:
                    raise RiskError("MODEL_BASIS_TOO_LARGE","An individual leg's IV fails its own quote check",contract_id=leg.contract_id,model_price=price,midpoint=midpoint,tolerance=tolerance)
    for store in (regions,certain,possible):
        store.freeze()
    return {"regions": regions, "certain": certain, "possible": possible, "domain": domain,
            "domain_exit_bound": domain_bound(job, *domain), "max_boundary_sensitivity": max_boundary,
            "nodes": nodes, "time_scale": time_scale, "endpoints": endpoints, "eps_port": eps_port,
            "solver_residual": max((x.residual for x in grids.values()), default=0.),"per_leg_model_checks":leg_checks,
            "region_storage":[store.statistics() for store in (regions,certain,possible)],"region_subdivision":region_subdivision}


def endpoint_probability(job, time_us, regions):
    horizon = (time_us-job.t0_utc_us)/YEAR_US
    drift = (job.rate_cc-job.dividend_yield_cc if job.state_kind == "SPOT" else 0)-job.sigma_path**2/2
    mu = math.log(job.x0)+drift*horizon
    sd = job.sigma_path*math.sqrt(max(0., horizon))
    if sd == 0:
        return float(region_membership(regions, math.exp(mu)))
    masses = []
    for interval in regions:
        lo = -math.inf if interval.lower <= 0 else (math.log(interval.lower)-mu)/sd
        hi = math.inf if math.isinf(interval.upper) else (math.log(interval.upper)-mu)/sd
        if lo >= 0:
            masses.append(float(ndtr(-lo)-ndtr(-hi)))
        else:
            masses.append(float(ndtr(hi)-ndtr(lo)))
    return math.fsum(masses)


def _simulate(job, times, region_variants, paths, phase, monitoring_sets=None, checkpoint=None, domain=None,deadline=None):
    """Complete logical shards, immutable seed slots, integer aggregation."""
    targets = tuple(map(float, job.targets))
    reports = job.reports
    monitors = monitoring_sets or [set(int(t) for t in times if eligible(job, int(t)))]
    variant_count = len(region_variants)*len(monitors)
    counts = np.zeros((variant_count, len(targets), len(reports)), dtype=np.int64)
    terminal = np.zeros((variant_count, len(targets)), dtype=np.int64)
    histograms = [Counter() for _ in range(variant_count*len(targets))]
    disagreements = np.zeros((max(0, variant_count-1), len(targets), len(reports)), dtype=np.int64)
    settings = job.numerical_settings
    region_identity = stable_hash([region_digest(variant) for variant in region_variants])
    kernel_hash=implementation_identity()["sha256"]
    identity = stable_hash({"job": job.computation_hash, "times": times.tolist(), "N": paths, "phase": phase, "numpy": np.__version__, "regions": region_identity, "monitoring_sets": [sorted(x) for x in monitors], "domain":domain,"kernel_source_sha256":kernel_hash})
    completed = set()
    if checkpoint and Path(checkpoint).exists():
        try:
            saved = json.loads(Path(checkpoint).read_text(encoding="utf-8"))
        except (ValueError,UnicodeError) as exc:
            raise RiskError('CHECKPOINT_MISMATCH','CHECKPOINT_MISMATCH: reference shard checkpoint is not valid JSON') from exc
        def invalid_checkpoint(reason):
            raise RiskError("CHECKPOINT_MISMATCH", "CHECKPOINT_MISMATCH: reference shard checkpoint failed integrity or conservation checks", validation=reason)
        if not isinstance(saved,dict):
            invalid_checkpoint('ROOT_TYPE')
        payload={key:value for key,value in saved.items() if key!='aggregate_sha256'}
        if saved.get('schema')!='QQQ_REFERENCE_SHARDS_V2' or saved.get('identity')!=identity:
            invalid_checkpoint('SCHEMA_OR_INPUT_IDENTITY')
        if saved.get('aggregate_sha256')!=stable_hash(payload):
            invalid_checkpoint('AGGREGATE_CHECKSUM')
        effective_paths=1 if job.sigma_path==0 else paths
        shard_count=(effective_paths+settings.shard_size-1)//settings.shard_size
        ids=saved.get('completed')
        if not isinstance(ids,list) or any(isinstance(value,bool) or not isinstance(value,int) or not 0<=value<shard_count for value in ids):
            invalid_checkpoint('COMPLETED_SHARD_RANGE')
        if ids!=sorted(set(ids)):
            invalid_checkpoint('DUPLICATE_OR_UNORDERED_COMPLETED_SHARD')
        completed=set(ids)
        completed_paths=sum(min(settings.shard_size,effective_paths-shard*settings.shard_size) for shard in ids)
        def restore_array(name,shape):
            raw=saved.get(name)
            try:
                array=np.asarray(raw)
            except (ValueError,TypeError,OverflowError):
                invalid_checkpoint(name.upper()+'_SHAPE_OR_TYPE')
            # JSON floats, booleans, ragged/object arrays and implicit NumPy
            # broadcasting must never become apparently valid path counts.
            if array.shape!=shape or array.dtype.kind not in ('i','u') or np.any(array<0) or np.any(array>completed_paths):
                invalid_checkpoint(name.upper()+'_SHAPE_TYPE_OR_RANGE')
            return array.astype(np.int64,copy=False)
        counts=restore_array('counts',counts.shape)
        terminal=restore_array('terminal',terminal.shape)
        # A zero-length leading dimension serializes as [] and loses its other
        # dimensions; accept only that exact empty representation in this case.
        if disagreements.size==0:
            if saved.get('disagreements')!=[]:
                invalid_checkpoint('DISAGREEMENTS_EMPTY_SHAPE')
        else:
            disagreements=restore_array('disagreements',disagreements.shape)
        if np.any(np.diff(counts,axis=2)<0) or np.any(np.diff(counts,axis=1)>0):
            invalid_checkpoint('HIT_COUNT_MONOTONICITY')
        raw_histograms=saved.get('histograms')
        if not isinstance(raw_histograms,list) or len(raw_histograms)!=len(histograms):
            invalid_checkpoint('HISTOGRAM_COUNT')
        for position,(dst,src) in enumerate(zip(histograms,raw_histograms)):
            if not isinstance(src,dict):
                invalid_checkpoint('HISTOGRAM_TYPE')
            values={}
            variant,target_index=divmod(position,len(targets))
            monitor=monitors[variant%len(monitors)]
            for key,value in src.items():
                try:
                    instant=int(key)
                except (ValueError,TypeError,OverflowError):
                    invalid_checkpoint('HISTOGRAM_TIME_TYPE')
                offset=int(np.searchsorted(times,instant))
                if str(instant)!=str(key) or offset>=len(times) or int(times[offset])!=instant or instant not in monitor:
                    invalid_checkpoint('HISTOGRAM_TIME_NOT_MONITORED')
                if isinstance(value,bool) or not isinstance(value,int) or not 0<value<=completed_paths:
                    invalid_checkpoint('HISTOGRAM_COUNT_RANGE')
                values[instant]=value
            if sum(values.values())+int(terminal[variant,target_index])>completed_paths:
                invalid_checkpoint('HIT_AND_TERMINAL_CONSERVATION')
            for report_index,report in enumerate(reports):
                if sum(value for instant,value in values.items() if instant<=report)!=counts[variant,target_index,report_index]:
                    invalid_checkpoint('HISTOGRAM_CUMULATIVE_COUNT_MISMATCH')
            dst.update(values)
        if np.any(np.diff(counts[:,:,-1]+terminal,axis=1)>0):
            invalid_checkpoint('TERMINAL_INCLUSIVE_TARGET_MONOTONICITY')
        if disagreements.size:
            left,right=counts[:-1],counts[1:]
            if (np.any(disagreements<np.abs(left-right)) or np.any(disagreements>np.minimum(left+right,2*completed_paths-left-right))
                    or np.any((disagreements-np.abs(left-right))%2)):
                invalid_checkpoint('PAIRED_DISAGREEMENT_CONSERVATION')
    started = time.monotonic()
    deadline=started+settings.max_work_seconds if deadline is None else deadline
    estimated_bytes=settings.shard_size*(variant_count*len(targets)*8+64)
    if estimated_bytes>settings.memory_budget_mb*1024*1024:
        raise RiskError("RESOURCE_LIMIT","Path shard working set exceeds declared memory budget",estimated_bytes=estimated_bytes)
    deterministic = job.sigma_path == 0
    if deterministic:
        paths = 1
    from .shards import run_shard,initialize,run_initialized
    from concurrent.futures import ProcessPoolExecutor,TimeoutError
    from multiprocessing import get_context
    from .resource_metrics import process_memory
    memory_before=process_memory()
    process_peaks={}
    context=(job,times,region_variants,paths,phase,monitors,domain,deadline)
    remaining=[i for i in range((paths+settings.shard_size-1)//settings.shard_size) if i not in completed]
    # Worker count changes resource usage only, never mathematical shards/seeds.
    per_worker=estimated_bytes+64*1024*1024
    available_memory=max(0,settings.memory_budget_mb*1024*1024-(memory_before.get("rss_bytes") or 64*1024*1024))
    workers=min(settings.workers,6,max(1,int(available_memory//per_worker)),len(remaining) or 1)
    if deterministic:
        workers=1
    def merge(result):
        nonlocal counts,terminal,disagreements
        if result.get("error"):
            error=result["error"]
            raise RiskError(error["reason"],error["message"],**error.get("details",{}))
        counts+=result["counts"]
        terminal+=result["terminal"]
        disagreements+=result["disagreements"]
        for dst,src in zip(histograms,result["histograms"]):
            dst.update(src)
        memory=result.get("process_memory",{})
        if memory.get("peak_rss_bytes"):
            process_peaks[memory["pid"]]=max(process_peaks.get(memory["pid"],0),memory["peak_rss_bytes"])
        completed.add(result["shard"])
        if checkpoint:
            payload={"schema":"QQQ_REFERENCE_SHARDS_V2","identity":identity,"completed":sorted(completed),"counts":counts.tolist(),"terminal":terminal.tolist(),"histograms":[dict(x) for x in histograms],"disagreements":disagreements.tolist()}
            atomic_json(checkpoint,{**payload,'aggregate_sha256':stable_hash(payload)})
    if workers==1:
        for shard in remaining:
            if time.monotonic()>deadline:
                raise RiskError("RESOURCE_LIMIT","Path work budget exceeded",completed_shards=len(completed),intended_paths=paths)
            merge(run_shard(context,shard,clock=time.monotonic))
    elif remaining:
        executor=ProcessPoolExecutor(max_workers=workers,mp_context=get_context("spawn"),initializer=initialize,initargs=(context,))
        outstanding=[]
        iterator=iter(remaining)
        try:
            for _ in range(workers):
                shard=next(iterator,None)
                if shard is not None:
                    outstanding.append((shard,executor.submit(run_initialized,shard)))
            while outstanding:
                shard,future=outstanding.pop(0)
                budget=deadline-time.monotonic()
                if budget<=0:
                    raise RiskError("RESOURCE_LIMIT","Path work budget exceeded",completed_shards=len(completed),intended_paths=paths)
                try:
                    result=future.result(timeout=budget)
                except TimeoutError as exc:
                    raise RiskError("RESOURCE_LIMIT","Path work budget exceeded",completed_shards=len(completed),intended_paths=paths) from exc
                if result.get("shard",shard)!=shard:
                    raise RiskError("SHARD_IDENTITY_MISMATCH")
                merge(result)
                following=next(iterator,None)
                if following is not None:
                    outstanding.append((following,executor.submit(run_initialized,following)))
        finally:
            for _,future in outstanding:
                future.cancel()
            # Read every live worker, including an initialized worker that did
            # not win any very short shard before another worker finished it.
            for process in (getattr(executor,"_processes",None) or {}).values():
                memory=process_memory(process.pid)
                if memory.get("peak_rss_bytes"):
                    process_peaks[memory["pid"]]=max(process_peaks.get(memory["pid"],0),memory["peak_rss_bytes"])
            executor.shutdown(wait=True,cancel_futures=True)
    parent_memory=process_memory()
    if parent_memory.get("peak_rss_bytes"):
        process_peaks[parent_memory["pid"]]=max(process_peaks.get(parent_memory["pid"],0),parent_memory["peak_rss_bytes"])
    return {"paths":paths,"counts":counts,"terminal":terminal,"histograms":[dict(x) for x in histograms],"shards":len(completed),"identity":identity,"deterministic":deterministic,"disagreements":disagreements,"workers_used":workers,"worker_working_set_estimate_bytes":per_worker,
            "process_peak_rss_bytes":process_peaks,"sum_process_peaks_bytes":sum(process_peaks.values()),"memory_interpretation":"SUM_OF_PROCESS_LIFETIME_PEAKS_IS_A_CONSERVATIVE_AGGREGATE_BOUND_NOT_SIMULTANEOUS_SAMPLING"}


def qualify_american_prices(job, times, regions, started):
    """Two separate space and time refinements, checked at every used slice."""
    legs = [x for x in job.signed_legs if x.exercise_style == "AMERICAN" and x.sigma > 0 and not (x.right == "CALL" and job.dividend_yield_cc == 0 and job.rate_cc >= 0)]
    if not legs:
        return {"qualified": True, "method": "EXACT_ANALYTICAL_LEG_PRICES", "max_price_sensitivity": 0.}
    eps = regions["eps_port"]/max(1, sum(abs(x.quantity)*x.multiplier for x in job.signed_legs))
    base = job.numerical_settings.space_nodes
    records = []
    for refinement in range(job.numerical_settings.max_refinements):
        fine = 4*(base-1)+1
        if fine > job.numerical_settings.max_space_nodes:
            break
        scale = 2**refinement
        max_space = np.zeros(2)
        max_time = np.zeros(2)
        max_boundary = max_residual = 0.
        configurations = ((base,4*scale),(2*base-1,4*scale),(fine,4*scale),(fine,scale),(fine,2*scale))
        solvers = {leg.contract_id: [AmericanGrid(leg.right,leg.strike,leg.sigma,job.rate_cc,job.dividend_yield_cc,regions["domain"],nodes,eps,step,started+job.numerical_settings.max_work_seconds) for nodes,step in configurations] for leg in legs}
        lo,hi=regions["domain"]
        width=math.log(hi/lo)
        check_lo,check_hi=lo*math.exp(width*.1),hi*math.exp(-width*.1)
        while domain_bound(job,check_lo,check_hi)>1e-8:
            check_lo=math.sqrt(check_lo*lo);check_hi=math.sqrt(check_hi*hi)
        base_points=np.unique(np.concatenate((np.exp(np.linspace(math.log(check_lo),math.log(check_hi),257)),[job.x0],*[np.array([float(x.strike)*(1-1e-6),float(x.strike),float(x.strike)*(1+1e-6)]) for x in job.signed_legs])))
        for instant in reversed(times):
            if time.monotonic()-started>job.numerical_settings.max_work_seconds:
                raise RiskError("RESOURCE_LIMIT", "Independent American space/time qualification budget reached", refinements=records)
            tau=(job.fixing_utc_us-int(instant))/YEAR_US
            roots=[endpoint for target in map(float,job.targets) for interval in regions["regions"].get((int(instant),target),()) for endpoint in (interval.lower,interval.upper) if check_lo <= endpoint <= check_hi and math.isfinite(endpoint)]
            points=np.unique(np.concatenate((base_points,roots)))
            for leg in legs:
                values=[]
                for solver in solvers[leg.contract_id]:
                    solver.advance(tau)
                    value,bound=solver.evaluate(points)
                    values.append(value)
                    max_boundary=max(max_boundary,float(np.max(bound)))
                    max_residual=max(max_residual,solver.residual)
                max_space=np.maximum(max_space,[np.max(np.abs(values[1]-values[0])),np.max(np.abs(values[2]-values[1]))])
                max_time=np.maximum(max_time,[np.max(np.abs(values[4]-values[3])),np.max(np.abs(values[2]-values[4]))])
        passed=max(max_space)<=eps/4 and max(max_time)<=eps/4 and max_boundary<=eps/4 and max_residual<=eps/4
        records.append({"nodes": [base,2*base-1,fine],"space_differences":max_space.tolist(),"time_differences":max_time.tolist(),"boundary_halfwidth":max_boundary,"solver_residual":max_residual,"per_component_budget":eps/4,"passed":bool(passed)})
        if passed:
            return {"qualified":True,"method":"INDEPENDENT_TWO_SPACE_AND_TIME_REFINEMENTS","refinements":records,"error_interpretation":"MEASURED_NUMERICAL_SENSITIVITY"}
        base=2*base-1
    raise RiskError("PRICER_NOT_CONVERGED", "Independent price refinement gates failed", refinements=records)


def atomic_json(path, data):
    from ...config import writable_path
    path = writable_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(canonical(data), stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except OSError as exc:
                if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 5:
                    raise
                time.sleep(.01*(2**attempt))
    finally:
        # Preserve the original error if Windows also temporarily denies
        # cleanup. The previous committed target is never deleted or edited.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def calculate_reference(job: ReferenceJob | dict, checkpoint=None, *, allow_preliminary=False):
    job = ReferenceJob.from_dict(job) if isinstance(job, dict) else job
    started = time.monotonic()
    record = {"job": canonical(job), "job_hash": job.hash, "model_event": "UNCHANGED_THEORETICAL_FIRST_HIT", "model_profile": job.model_profile, "numpy_version": np.__version__, "implementation":implementation_identity(),"failed_numerical_gates": []}
    try:
        if not job.signed_legs:
            raise RiskError("NO_OPEN_POSITION_HOLD_EVENT","A flat managed campaign has no unchanged-option HOLD event")
        grids = [observation_grid(job, delta)[0] for delta in job.numerical_settings.monitoring_seconds]
        union = np.unique(np.concatenate(grids))
        region = build_regions(job, union, started=started)
        record["per_leg_model_checks"]=region["per_leg_model_checks"]
        record["region_storage"]=region["region_storage"]
        record["region_subdivision"]=region["region_subdivision"]
        narrowest=None
        deterministic_static = job.sigma_path == 0 and job.rate_cc == 0 and job.dividend_yield_cc == 0 and all(leg.sigma == 0 for leg in job.signed_legs)
        if not deterministic_static:
            initially_satisfied={float(threshold) for threshold in job.targets if eligible(job,job.t0_utc_us) and region_membership(region["regions"].get((job.t0_utc_us,float(threshold)),()),job.x0)}
            items=region["regions"].ordered_items() if hasattr(region["regions"],"ordered_items") else region["regions"].items()
            for (instant,threshold),intervals in items:
                if threshold in initially_satisfied or not eligible(job,int(instant)):continue
                singleton=next((interval for interval in intervals if interval.lower==interval.upper and interval.lower_closed and interval.upper_closed),None)
                if singleton is not None:
                    raise RiskError("SINGLETON_CLOSING_EVENT_UNRESOLVED","Endpoint monitoring cannot qualify a between-observation hit of an isolated target point",target=threshold,instant_utc_us=int(instant),spot=singleton.lower,path_kind="STOCHASTIC" if job.sigma_path>0 else "DETERMINISTIC",required_method="INDEPENDENT_CONTINUOUS_CROSSING_OR_BRIDGE_PROOF")
            for variant in ("regions","possible"):
                mapping=region[variant]
                items=mapping.ordered_items() if hasattr(mapping,"ordered_items") else mapping.items()
                for (instant,threshold),intervals in items:
                    if threshold in initially_satisfied or not eligible(job,int(instant)):continue
                    for interval in intervals:
                        # A component meeting a truncated numerical-domain edge
                        # is a ray within that domain, whose discarded mass has
                        # its separate bound. Only interior islands can be
                        # jumped completely across between endpoint samples.
                        if not region["domain"][0]<interval.lower<interval.upper<region["domain"][1]:continue
                        width=math.log(interval.upper/interval.lower)
                        if narrowest is None or width<narrowest["log_width"]:
                            narrowest={"target":threshold,"instant_utc_us":int(instant),"bounds":[interval.lower,interval.upper],"log_width":width,"region_variant":variant}
        if job.sigma_path == 0 and narrowest is not None and not deterministic_static:
            raise RiskError("DETERMINISTIC_CLOSING_TOPOLOGY_UNRESOLVED","A moving deterministic path or value surface can cross a finite target island between observations; exact time-crossing evidence is required",**narrowest,required_method="EXACT_DETERMINISTIC_TIME_CROSSING")
        def island_resolved(seconds):
            if narrowest is None:return True
            dt=float(seconds)/31536000
            mu=(job.rate_cc-job.dividend_yield_cc if job.state_kind=="SPOT" else 0)-job.sigma_path**2/2
            return narrowest["log_width"]>=8*job.sigma_path*math.sqrt(dt)+abs(mu)*dt
        if not island_resolved(min(job.numerical_settings.monitoring_seconds)):
            raise RiskError("NARROW_CLOSING_ISLAND_UNRESOLVED","Endpoint observations are too far apart to resolve a finite target island; an apparent zero gained-hit plateau cannot qualify it",**narrowest,finest_seconds=min(job.numerical_settings.monitoring_seconds),required_resolution="LOG_WIDTH_AT_LEAST_EIGHT_STEP_STANDARD_DEVIATIONS_PLUS_ABSOLUTE_DRIFT")
        record["closing_topology_resolution"]={"narrowest_interior_island":narrowest,"singleton_bridge_required":False,"criterion":"CONSERVATIVE_ENDPOINT_MONITORING_GUARD_NOT_A_BRIDGE_PROOF"}
        target = float(job.target_usd)
        if any(x.bid is not None for x in job.signed_legs):
            quoted = float(job.entry_cash_usd)+sum(x.quantity*x.multiplier*(x.bid+x.ask)/2 for x in job.signed_legs if x.bid is not None)
            if all(x.bid is not None for x in job.signed_legs):
                tolerance = sum(abs(x.quantity)*x.multiplier*(x.ask-x.bid)/2 for x in job.signed_legs)+region["eps_port"]
                residual = region["endpoints"][job.t0_utc_us]["model_pnl"]-quoted
                record["model_quote_residual_usd"] = residual
                if abs(residual) > tolerance:
                    raise RiskError("MODEL_BASIS_TOO_LARGE", residual=residual, tolerance=tolerance)
        monitors = [set(int(t) for t in grid if eligible(job, int(t))) for grid in grids]
        pilot = _simulate(job, union, [region["regions"]], job.numerical_settings.pilot_paths, 20, monitors, domain=region["domain"],deadline=started+job.numerical_settings.max_work_seconds)
        comparisons = max(1, (len(grids)-1)*len(job.targets)*len(job.reports))
        z = max(1.6448536269514722, float(ndtri(1-.05/comparisons)))
        passes, chosen, pilot_checks = 0, None, []
        for i in range(len(grids)-1):
            gains = pilot["counts"][i+1]-pilot["counts"][i]
            # Closing-only and terminal-inclusive union are separate required
            # comparisons; a newly earlier hit may replace a terminal-only atom.
            union_gains=(pilot["counts"][i+1,:,-1]+pilot["terminal"][i+1])-(pilot["counts"][i,:,-1]+pilot["terminal"][i])
            gains=np.concatenate((gains.ravel(),union_gains.ravel()))
            if np.min(gains) < 0:
                raise RiskError("MONITORING_NESTING_FAILURE")
            upper = float(np.max(gains))/pilot["paths"] if pilot["deterministic"] else max(wilson(int(k), pilot["paths"], z)[1] for k in gains.flat)
            passed = upper <= .0025
            topology_passed=island_resolved(job.numerical_settings.monitoring_seconds[i+1])
            pilot_checks.append({"coarse_seconds": job.numerical_settings.monitoring_seconds[i], "fine_seconds": job.numerical_settings.monitoring_seconds[i+1], "gained_hit_upper": upper, "passed": passed,"finite_island_resolution_passed":topology_passed})
            passes = passes+1 if passed else 0
            if passes >= 2 and topology_passed and chosen is None:
                chosen = i+1
        record["monitoring_pilot"] = pilot_checks
        if job.sigma_path == 0:
            chosen = len(grids)-1
        if chosen is None:
            if not allow_preliminary:
                raise RiskError("NUMERICAL_CONVERGENCE_UNRESOLVED", "No two successive monitoring refinements passed", pilot=pilot_checks)
            chosen = len(grids)-1
            record["failed_numerical_gates"].append("NUMERICAL_CONVERGENCE_UNRESOLVED")
        chosen_times = grids[chosen]
        sensitivity = _simulate(job, chosen_times, [region["certain"], region["possible"]], job.numerical_settings.region_paths, 30, domain=region["domain"],deadline=started+job.numerical_settings.max_work_seconds)
        differences = sensitivity["counts"][1]-sensitivity["counts"][0]
        uncertainty = float(np.max(differences))/sensitivity["paths"] if sensitivity["deterministic"] else max(wilson(int(k), sensitivity["paths"], float(ndtri(1-.05/max(1, differences.size))))[1] for k in differences.flat)
        if uncertainty > .0005:
            if not allow_preliminary:
                raise RiskError("TARGET_REGION_UNRESOLVED", region_uncertainty_upper=uncertainty)
            record["failed_numerical_gates"].append("TARGET_REGION_UNRESOLVED")
        # Independently refine spatial representation and compare on common paths.
        refined = build_regions(job, union, nodes=2*region["nodes"]-1, time_scale=2, started=started)
        stability = _simulate(job, chosen_times, [region["regions"], refined["regions"]], job.numerical_settings.region_paths, 30, domain=refined["domain"],deadline=started+job.numerical_settings.max_work_seconds)
        change = float(np.max(stability["disagreements"]))/stability["paths"] if stability["deterministic"] else max(wilson(int(k),stability["paths"],float(ndtri(1-.05/max(1,stability["disagreements"].size))))[1] for k in stability["disagreements"].flat)
        if change > .0005:
            if not allow_preliminary:
                raise RiskError("TARGET_REGION_UNRESOLVED", refinement_probability_difference=float(change))
            record["failed_numerical_gates"].append("REFINED_TARGET_REGION_UNRESOLVED")
        if not allow_preliminary:
            twice_refined = build_regions(job, union, nodes=4*(region["nodes"]-1)+1, time_scale=4, started=started)
            repeated = _simulate(job,chosen_times,[refined["regions"],twice_refined["regions"]],job.numerical_settings.region_paths,30,domain=twice_refined["domain"],deadline=started+job.numerical_settings.max_work_seconds)
            repeated_bound = float(np.max(repeated["disagreements"]))/repeated["paths"] if repeated["deterministic"] else max(wilson(int(k),repeated["paths"],float(ndtri(1-.05/max(1,repeated["disagreements"].size))))[1] for k in repeated["disagreements"].flat)
            if repeated_bound > .0005:
                raise RiskError("TARGET_REGION_UNRESOLVED", second_refinement_probability_bound=repeated_bound)
            refined=twice_refined
            price_qualification=qualify_american_prices(job,union,refined,started)
            record["price_qualification"]=price_qualification
            record["second_region_refinement_probability_bound"]=repeated_bound
        production = _simulate(job, chosen_times, [refined["regions"]], job.numerical_settings.production_paths, 10, checkpoint=checkpoint, domain=refined["domain"],deadline=started+job.numerical_settings.max_work_seconds)
        n = production["paths"]
        ti = tuple(map(float, job.targets)).index(target)
        di = job.reports.index(job.deadline_utc_us)
        k = int(production["counts"][0, ti, di])
        close_all = int(production["counts"][0, ti, -1])
        settlement = int(production["terminal"][0, ti])
        headline = k+settlement if job.deadline_utc_us == job.fixing_utc_us else k
        endpoint = endpoint_probability(job, job.deadline_utc_us, refined["regions"][(job.deadline_utc_us, target)])
        pop = endpoint_probability(job, job.fixing_utc_us, terminal_regions(job.signed_legs, job.entry_cash_usd, .01))
        strict = endpoint_probability(job, job.fixing_utc_us, terminal_regions(job.signed_legs, job.entry_cash_usd, 0, strict=True))
        record.update(status="PRELIMINARY" if record["failed_numerical_gates"] else "OK", hold_p50=headline/n, closing_only=close_all/n, terminal_inclusive_reference=(close_all+settlement)/n, settlement_only=settlement/n, deadline_endpoint=endpoint, terminal_cent_pop=pop, terminal_strict_positive_pop=strict,
                      sampling_interval=None if production["deterministic"] else wilson(headline, n), sampling_se=0. if production["deterministic"] else math.sqrt((headline/n)*(1-headline/n)/n),
                      production_paths=n, production_shards=production["shards"], production_workers=production["workers_used"], production_memory={k:production[k] for k in ("process_peak_rss_bytes","sum_process_peaks_bytes","memory_interpretation")},first_hit_histograms=production["histograms"], grid_seconds=job.numerical_settings.monitoring_seconds[chosen], grid_hash=stable_hash(chosen_times.tolist()),
                      numerical_sensitivity={"region_probability_upper": uncertainty, "refinement_probability_difference": float(change), "domain_exit_bound": refined["domain_exit_bound"], "solver_residual": refined["solver_residual"]},
                      assumptions=["GBM_Q_PATH", "FROZEN_INDIVIDUAL_IV", "CONTINUOUS_Q", "DISCRETE_ELIGIBLE_OBSERVATIONS", "THEORETICAL_MARK_TOUCH_IS_NOT_EXECUTED_PROFIT"])
        record["release_qualified"] = False
        record["target_reports"]=[{"target_usd":str(h),"deadline_utc_us":when,"closing_first_hit":int(production["counts"][0,target_i,report_i])/n,"terminal_inclusive_reference":(int(production["counts"][0,target_i,report_i])+int(production["terminal"][0,target_i]))/n if when==job.fixing_utc_us else None,"settlement_only":int(production["terminal"][0,target_i])/n if when==job.fixing_utc_us else None} for target_i,h in enumerate(job.targets) for report_i,when in enumerate(job.reports)]
        record["release_reason"] = "PRELIMINARY_REFERENCE_EXPLICITLY_REQUESTED" if allow_preliminary else "INDEPENDENT_TRANSITION_VALIDATION_PENDING"
        if not allow_preliminary:
            independent_records=[]
            record["independent_probability_validation"] = independent_records
            comparisons=[(target_i,float(h),report_i,when,when==job.fixing_utc_us) for target_i,h in enumerate(job.targets) for report_i,when in enumerate(job.reports)]
            comparisons.extend((target_i,float(h),len(job.reports)-1,job.fixing_utc_us,False) for target_i,h in enumerate(job.targets))
            for target_i,threshold,report_i,through,terminal_union in comparisons:
                previous_independent,consecutive=None,0
                refinements=[]
                independent_records.append({"target":threshold,"deadline":through,"terminal_union":terminal_union,"refinements":refinements})
                for independent_nodes in independent_node_schedule(job.numerical_settings.max_space_nodes):
                    if time.monotonic()-started>job.numerical_settings.max_work_seconds:
                        raise RiskError("RESOURCE_LIMIT", "Independent transition solver budget reached", comparisons=independent_records)
                    independent=independent_first_hit(job,chosen_times,refined["regions"],threshold,nodes=independent_nodes,deadline=started+job.numerical_settings.max_work_seconds,through_utc_us=through,terminal_inclusive=terminal_union)
                    midpoint=(independent["lower"]+independent["upper"])/2
                    difference=None if previous_independent is None else abs(midpoint-previous_independent)
                    passed=difference is not None and difference<=.0005 and independent["upper"]-independent["lower"]<=.0005
                    refinements.append({**independent,"difference":difference,"passed":passed})
                    consecutive=consecutive+1 if passed else 0
                    if consecutive>=2:
                        observed_count=int(production["counts"][0,target_i,report_i])+(int(production["terminal"][0,target_i]) if terminal_union else 0)
                        observed=observed_count/n
                        interval=[observed,observed] if production["deterministic"] else wilson(observed_count,n)
                        if midpoint<interval[0]-.001 or midpoint>interval[1]+.001:
                            raise RiskError("INDEPENDENT_SOLVER_DISAGREEMENT",production_interval=interval,independent=independent,target=threshold,deadline=through)
                        break
                    previous_independent=midpoint
                if consecutive<2:
                    raise RiskError("NUMERICAL_CONVERGENCE_UNRESOLVED", "Independent probability grid did not converge twice", comparisons=independent_records)
            record["independent_probability_validation"]=independent_records
            record["release_qualified"]=True
            record["release_reason"]=None
    except RiskError as exc:
        record.update(exc.as_dict())
        record["hold_p50"] = None
    record["elapsed_seconds"] = time.monotonic()-started
    return record


def continuous_barrier_probability(log_distance, horizon, sigma, mu):
    if log_distance <= 0:
        return 1.
    if horizon <= 0:
        return 0.
    if sigma == 0:
        return float(mu*horizon >= log_distance)
    sd = sigma*math.sqrt(horizon)
    return float(ndtr((mu*horizon-log_distance)/sd)+math.exp(2*mu*log_distance/sigma**2+float(log_ndtr((-mu*horizon-log_distance)/sd))))


def transition_matrix(log_grid, dt, sigma, mu):
    """Exact normal integrals of piecewise linear interpolation; no discount."""
    grid = np.asarray(log_grid, dtype=np.float64)
    mean = grid[:, None]+mu*dt
    sd = sigma*math.sqrt(dt)
    if sd == 0:
        matrix = np.zeros((len(grid), len(grid)))
        shifted = grid+mu*dt
        for row, value in enumerate(shifted):
            if grid[0] <= value <= grid[-1]:
                idx = min(len(grid)-2, max(0, np.searchsorted(grid, value)-1))
                alpha = (value-grid[idx])/(grid[idx+1]-grid[idx])
                matrix[row, idx:idx+2] = [1-alpha, alpha]
        return matrix, (shifted < grid[0]).astype(float)+(shifted > grid[-1]).astype(float)
    zl, zr = (grid[:-1]-mean)/sd, (grid[1:]-mean)/sd
    p = np.where(zl >= 0, ndtr(-zl)-ndtr(-zr), ndtr(zr)-ndtr(zl))
    phi_l, phi_r = np.exp(-.5*zl*zl)/math.sqrt(2*math.pi), np.exp(-.5*zr*zr)/math.sqrt(2*math.pi)
    moment = mean*p+sd*(phi_l-phi_r)
    left = (grid[1:]*p-moment)/np.diff(grid)
    right = (moment-grid[:-1]*p)/np.diff(grid)
    if min(float(left.min()), float(right.min())) < -1e-10:
        raise RiskError("TRANSITION_WEIGHT_NEGATIVITY")
    matrix = np.zeros((len(grid), len(grid)))
    matrix[:, :-1] += left
    matrix[:, 1:] += right
    tail = ndtr((grid[0]-mean[:, 0])/sd)+ndtr((mean[:, 0]-grid[-1])/sd)
    if np.max(np.abs(matrix.sum(axis=1)+tail-1)) > 1e-10:
        raise RiskError("TRANSITION_MASS_DEFICIT")
    return matrix, tail


def independent_node_schedule(max_nodes):
    """Refine to the declared cap; the dense allocation guard remains binding."""
    maximum = int(max_nodes)
    if maximum < 3:
        raise RiskError("INVALID_NUMERICAL_SETTINGS", max_space_nodes=maximum)
    nodes = min(129, maximum)
    while True:
        yield nodes
        if nodes == maximum:
            return
        nodes = min(2*nodes-1, maximum)


def independent_path_domain(job, exit_bound_target=1e-8):
    """Path-law domain, separate from the price PDE's strike/leg-IV domain.

    The reflection/union bound controls the whole horizon. Lower and upper
    transition recursions still retain every outside-domain tail contribution.
    """
    if not 0 < exit_bound_target < 1 or job.sigma_path <= 0:
        raise RiskError("INVALID_STOCHASTIC_PATH_DOMAIN")
    horizon = (job.fixing_utc_us-job.t0_utc_us)/YEAR_US
    if horizon <= 0:
        raise RiskError("INVALID_STOCHASTIC_PATH_DOMAIN")
    drift = (job.rate_cc-job.dividend_yield_cc if job.state_kind == "SPOT" else 0)-.5*job.sigma_path**2
    # Half of the permitted bound leaves room for floating-point endpoint
    # construction; this never introduces the pricing domain's 0.5 log pad.
    z = float(ndtri(1-exit_bound_target/8))
    width = abs(drift)*horizon+z*job.sigma_path*math.sqrt(horizon)
    center = math.log(job.x0)
    for _ in range(20):
        lower = float(np.nextafter(math.exp(center-width), 0.))
        upper = float(np.nextafter(math.exp(center+width), math.inf))
        bound = domain_bound(job, lower, upper)
        if 0 < lower < job.x0 < upper and bound <= exit_bound_target:
            return (lower, upper), bound
        width *= 1.1
    raise RiskError("NUMERICAL_DOMAIN_UNRESOLVED", path_exit_bound=bound, required_bound=exit_bound_target)


def independent_first_hit(job, times, regions, target, nodes=401,deadline=None,through_utc_us=None,terminal_inclusive=True):
    through=job.fixing_utc_us if through_utc_us is None else through_utc_us
    times=np.asarray([t for t in times if t<=through],dtype=np.int64)
    if not len(times) or times[-1]!=through:
        times=np.append(times,through)
    if job.sigma_path==0:
        value=0.
        drift=job.rate_cc-job.dividend_yield_cc if job.state_kind=="SPOT" else 0.
        for instant in times:
            spot=job.x0*math.exp(drift*(int(instant)-job.t0_utc_us)/YEAR_US)
            if eligible(job,int(instant)) and region_membership(regions[(int(instant),float(target))],spot):
                value=1.;break
            if instant==job.fixing_utc_us and terminal_inclusive and payoff_array(job.signed_legs,spot,job.entry_cash_usd)>=target:
                value=1.;break
        return {"lower":value,"upper":value,"nodes":1,"status":"EXACT_DETERMINISTIC_INDEPENDENT"}
    # zl/zr, two CDF branches, densities, interval moments/weights and final
    # dense matrix coexist during construction. Guard their conservative peak
    # before allocating even the grid; the prior cached matrix is released
    # before construction whenever dt changes.
    estimated_dense_peak_bytes=12*int(nodes)*int(nodes)*8+32*int(nodes)*8
    memory_budget_bytes=int(job.numerical_settings.memory_budget_mb)*1024*1024
    if estimated_dense_peak_bytes>memory_budget_bytes:
        raise RiskError("RESOURCE_LIMIT","Independent dense transition temporary arrays exceed the declared working-memory budget",resource="memory",nodes=nodes,estimated_additional_peak_bytes=estimated_dense_peak_bytes,memory_budget_bytes=memory_budget_bytes,estimate="TWELVE_FLOAT64_DENSE_ARRAYS_PLUS_LINEAR_WORKSPACE_EXCLUDES_INTERPRETER")
    (lower, upper), analytic_path_exit_bound = independent_path_domain(job)
    log_grid = np.linspace(math.log(lower), math.log(upper), nodes)
    spots = np.exp(log_grid)
    lo = region_membership(terminal_regions(job.signed_legs, job.entry_cash_usd, target), spots).astype(float) if through==job.fixing_utc_us and terminal_inclusive else np.zeros(len(spots))
    if eligible(job,int(times[-1])):
        lo[region_membership(regions[(int(times[-1]),float(target))],spots)]=1.
    hi = lo.copy()
    mu = (job.rate_cc-job.dividend_yield_cc if job.state_kind == "SPOT" else 0)-job.sigma_path**2/2
    cached_step_us=None
    transition=tails=None
    matrix_builds=matrix_reuses=0
    roundoff_tolerance=1e-9
    extrema={"minimum_lower":float(lo.min()),"maximum_upper":float(hi.max()),"minimum_bracket_width":0.,"maximum_bracket_width":0.,"maximum_transition_mass_error":0.,"minimum_transition_weight":0.}
    def check_probabilities(instant):
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise RiskError("TRANSITION_PROBABILITY_NONFINITE",instant_utc_us=int(instant))
        minimum,maximum=float(lo.min()),float(hi.max())
        widths=hi-lo
        smallest,largest=float(widths.min()),float(widths.max())
        extrema["minimum_lower"]=min(extrema["minimum_lower"],minimum)
        extrema["maximum_upper"]=max(extrema["maximum_upper"],maximum)
        extrema["minimum_bracket_width"]=min(extrema["minimum_bracket_width"],smallest)
        extrema["maximum_bracket_width"]=max(extrema["maximum_bracket_width"],largest)
        if minimum < -roundoff_tolerance or maximum > 1+roundoff_tolerance or smallest < -roundoff_tolerance:
            raise RiskError("TRANSITION_PROBABILITY_BOUNDS_FAILURE",instant_utc_us=int(instant),roundoff_tolerance=roundoff_tolerance,extrema=extrema)
    check_probabilities(times[-1])
    for before, after in zip(times[-2::-1], times[:0:-1]):
        if deadline is not None and time.monotonic()>=deadline:
            raise RiskError("RESOURCE_LIMIT", "Independent probability solver checkpoint required")
        step_us=int(after)-int(before)
        if cached_step_us!=step_us:
            # Grid and law are fixed throughout this invocation. Retain one
            # exact transition only; discard it before allocating a gap step.
            transition=tails=None
            transition, tails = transition_matrix(log_grid, step_us/YEAR_US, job.sigma_path, mu)
            extrema["maximum_transition_mass_error"]=max(extrema["maximum_transition_mass_error"],float(np.max(np.abs(transition.sum(axis=1)+tails-1))))
            extrema["minimum_transition_weight"]=min(extrema["minimum_transition_weight"],float(transition.min()))
            cached_step_us=step_us
            matrix_builds+=1
        else:
            matrix_reuses+=1
        lo, hi = transition@lo, transition@hi+tails
        check_probabilities(before)
        if eligible(job, int(before)):
            hit = region_membership(regions[(int(before), float(target))], spots)
            lo[hit], hi[hit] = 1., 1.
    return {"lower": float(np.interp(math.log(job.x0), log_grid, lo)), "upper": float(np.interp(math.log(job.x0), log_grid, hi)), "nodes": nodes, "status": "INDEPENDENT_NUMERICAL_DIAGNOSTIC",
            "path_domain":[lower,upper],"analytic_path_exit_bound":analytic_path_exit_bound,"log_grid_spacing":float(log_grid[1]-log_grid[0]),
            "transition_matrix_builds":matrix_builds,"transition_matrix_reuses":matrix_reuses,
            "probability_preservation":{**extrema,"roundoff_tolerance":roundoff_tolerance,"checked_every_transition":True,"clipping_applied":False},
            "maximum_cached_transition_bytes":0 if transition is None else transition.nbytes+tails.nbytes,"estimated_dense_peak_bytes":estimated_dense_peak_bytes}
