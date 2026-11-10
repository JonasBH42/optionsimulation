"""Measure 100 distinct fast queries, explicitly distinguishing abstention paths."""
from __future__ import annotations
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import argparse
import json
import sys
import time
import tracemalloc

ROOT=Path(r"D:\optionsimulation")
sys.path.insert(0,str(ROOT/"src"))
from qqq_sim.config import configure_environment
configure_environment()
import numpy as np
from qqq_sim.risk.contracts import ReferenceJob
from qqq_sim.risk.hold import evaluate_hold
from qqq_sim.risk.pack_loader import ModelPack
from qqq_sim.serialization import save_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("pack")
    parser.add_argument("--job",required=True)
    parser.add_argument("--output",default="runtime/results/query-benchmark.json")
    args=parser.parse_args()
    before=set(sys.modules)
    start=time.perf_counter()
    pack=ModelPack.load(args.pack)
    load_seconds=time.perf_counter()-start
    base=ReferenceJob.from_dict(json.loads(Path(args.job).read_text(encoding="utf-8")))
    results=[]
    tracemalloc.start()
    for i in range(100):
        shift=(i-50)/10000
        legs=tuple(replace(x,sigma=x.sigma*(1+shift)) for x in base.signed_legs)
        job=replace(base,x0=base.x0*(1+shift),sigma_path=base.sigma_path*(1+shift),signed_legs=legs,
                    target_usd=base.target_usd+Decimal(i)/100)
        start=time.perf_counter()
        result=evaluate_hold(job,pack)
        elapsed=time.perf_counter()-start
        results.append({"job_hash":job.hash,"seconds":elapsed,"status":result.get("status"),"reason":result.get("reason"),"p50":result.get("hold_p50")})
    _,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    offline_imports=sorted(name for name in set(sys.modules)-before if name.startswith("qqq_sim.risk.offline") or name.startswith("torch"))
    timing=np.array([x["seconds"] for x in results])
    covered=sum(x["p50"] is not None for x in results)
    report={"pack_id":pack.pack_id,"queries":100,"unique_inputs":len({x["job_hash"] for x in results}),
            "covered":covered,"abstained":100-covered,"interpretation":"SUPPORTED_INFERENCE" if covered==100 else "ABSTENTION_PATH_ONLY_NOT_A_SUPPORTED_QUERY_PERFORMANCE_PASS",
            "load_seconds":load_seconds,"p50_seconds":float(np.quantile(timing,.5)),"p95_seconds":float(np.quantile(timing,.95)),
            "p99_seconds":float(np.quantile(timing,.99)),"python_traced_peak_bytes":peak,"offline_imports":offline_imports,
            "offline_import_boundary_passed":not offline_imports,"records":results,"production_latency_gate_passed":covered==100 and float(np.quantile(timing,.95))<=1}
    save_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!="records"},indent=2))


if __name__=="__main__":
    main()
