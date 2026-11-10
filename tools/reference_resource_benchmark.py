"""Executed resource evidence for immutable region blocks and spawned shards."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT=Path(r"D:\optionsimulation")
sys.path.insert(0,str(ROOT/"src"))


def main():
    from qqq_sim.config import configure_environment
    configure_environment()
    from qqq_sim.serialization import save_json
    from qqq_sim.risk.contracts import ReferenceJob
    from qqq_sim.risk.payoff import Interval
    from qqq_sim.risk.offline.region_store import RegionStore
    from qqq_sim.risk.offline.resource_metrics import process_memory
    from qqq_sim.risk.offline.hit_oracle import calculate_reference
    started=time.perf_counter()
    store=RegionStore({"purpose":"EXPLICIT_SYNTHETIC_STORAGE_BENCHMARK"})
    for t in reversed(range(40000)):
        store[(t,50.)]=tuple(Interval(float(i),float(i)+.5) for i in range(16))
    store.freeze()
    for t in range(0,40000,317):
        assert store[(t,50.)]==tuple(Interval(float(i),float(i)+.5) for i in range(16))
    record={"storage":store.statistics(),"storage_seconds":time.perf_counter()-started,
            "storage_process_memory":process_memory(),"roundtrip_sampled_records":len(range(0,40000,317)),
            "max_block_bytes":max(x["bytes"] for x in store._blocks)}
    store.close()
    old=json.loads((ROOT/"runtime/risk_validation/qualified_far_otm_reference.json").read_text(encoding="utf-8"))
    job=ReferenceJob.from_dict(old["job"])
    job=replace(job,numerical_settings=replace(job.numerical_settings,workers=2,memory_budget_mb=512,max_work_seconds=120))
    result=calculate_reference(job)
    save_json("runtime/risk_validation/qualified_far_otm_reference_parallel.json",result)
    record["reference"]={k:result.get(k) for k in ("status","reason","release_qualified","production_paths","production_workers","production_memory","hold_p50","elapsed_seconds","region_subdivision")}
    record["scope"]="Synthetic storage and one easy reference case only; not a broad production-model qualification"
    save_json("runtime/results/reference-resource-benchmark.json",record)
    print(json.dumps(record,indent=2))


if __name__=="__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
