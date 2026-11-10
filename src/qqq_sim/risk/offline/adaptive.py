"""P50-M11 deterministic calibration-focused training-design augmentation."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from ..contracts import RiskError,stable_hash
from .design import latin_hypercube,design_record
from .hit_oracle import atomic_json


def failing_calibration_regimes(rows,tolerance=.01):
    groups={}
    for row in rows:
        groups.setdefault(row["regime"],[]).append(row)
    failed=[]
    for regime,cases in sorted(groups.items()):
        errors=np.array([abs(c["prediction"]-c["reference"])+c.get("reference_uncertainty",0) for c in cases])
        if float(np.quantile(errors,.99))>tolerance:
            failed.append({"regime":regime,"p99_error":float(np.quantile(errors,.99)),"case_indices":[c["case_index"] for c in cases]})
    return failed


def augmented_records(records,failed_regimes,cycle,*,count=8192):
    if not 1<=cycle<=6:
        raise RiskError("ADAPTIVE_CYCLE_LIMIT")
    if count!=8192:
        raise RiskError("ADAPTIVE_DESIGN_SIZE_MISMATCH","Production adaptive cycles contain exactly8192 training states")
    if not failed_regimes:
        return []
    by_index={r["index"]:r for r in records}
    centers=[]
    for failure in sorted(failed_regimes,key=lambda x:(-x["p99_error"],x["regime"])):
        centers.extend(by_index[i] for i in failure["case_indices"] if i in by_index and by_index[i].get("augmentation_seed_eligible",True) and not by_index[i].get("heldout_edge_family"))
    if not centers:
        raise RiskError("ADAPTIVE_CALIBRATION_CENTERS_MISSING")
    lhs=latin_hypercube(count,dimensions=len(centers[0]["coordinates"]),cycle=cycle)
    start=max(by_index)+1
    output=[]
    for offset,uniform in enumerate(lhs["coordinates"]):
        center=centers[offset%len(centers)]
        point=np.asarray(center["coordinates"])
        # Sample inside an explicitly bounded local box; do not clip a market
        # state or mutate the original campaign's frozen objective.
        width=.10/(1+.25*(cycle-1))
        lo=np.maximum(0,point-width)
        hi=np.minimum(1,point+width)
        coordinate=lo+(hi-lo)*uniform
        record=design_record(start+offset,coordinate,cycle)
        record.update(split="train",shape_group=f"ADAPTIVE_TRAIN:{cycle}:{offset}",adaptive_parent_index=center["index"],adaptive_cycle=cycle,adaptive_source="CALIBRATION_ONLY",objective_basis="NEW_SYNTHETIC_ORIGINAL_CAMPAIGN_NOT_MUTATED_H0")
        output.append(record)
    return output


def prepare_next_cycle(folder,records,calibration_rows,cycle,*,tolerance=.01):
    from ...config import writable_path
    folder=writable_path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    failures=failing_calibration_regimes(calibration_rows,tolerance)
    if not failures:
        return {"status":"CALIBRATION_ERROR_TARGET_PASSED","next_cycle":None}
    if cycle>=6:
        return {"status":"ADAPTIVE_BUDGET_EXHAUSTED","next_cycle":None,"excluded_calibration_regimes":failures}
    next_cycle=cycle+1
    new=augmented_records(records,failures,next_cycle)
    base_file=folder/f"cycle_{next_cycle}_base_records.json"
    extra_file=folder/f"cycle_{next_cycle}_additional_records.json"
    atomic_json(base_file,records)
    atomic_json(extra_file,new)
    result={"status":"NEXT_TRAINING_CYCLE_PREPARED","next_cycle":next_cycle,"base_records_file":str(base_file),"additional_records_file":str(extra_file),"base_records_hash":stable_hash(records),"additional_records_hash":stable_hash(new),"added_training_cases":len(new),"failed_calibration_regimes":failures,"validation_data_used_for_adaptation":False}
    atomic_json(folder/f"cycle_{next_cycle}_manifest.json",result)
    return result
