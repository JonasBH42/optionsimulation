"""Executable, checkpointed B0-B9 build stages with explicit resource exits.

Production never trains from unresolved numerical labels. The separately named
research mode permits preliminary labels and records that exclusion permanently.
"""
from __future__ import annotations
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ...config import writable_path
from ..contracts import ReferenceJob, RiskError, NumericalSettings, canonical, stable_hash, YEAR_US
from ..features import hold_features, managed_state_features, FEATURE_SCHEMA
from ..feature_layout import MANAGED_FEATURE_COUNT,managed_masks,feature_contract
from ..hold import survival_bins, survival_cdf, network
from ..pack_loader import file_hash
from ..payoff import terminal_regions
from ..policy import generate_actions, action_features
from .design import latin_hypercube, design_record, materialize_case, SEED_KEYS
from .hit_oracle import calculate_reference, endpoint_probability, atomic_json
from .price_basis import build_price_basis, build_price_tree
from .train import train_network, calibration_report, within_bin_validation, HOLD_LABEL_SCHEMA, HOLD_LOSS_SCHEMA
from .control_oracle import ManagedState, transition, converged_expectation, rollout,fresh_account_limits
from .validate import reference_fixtures, online_import_audit
from ..qualification import regime_key,qualify_regimes
from .control_layers import complete_minute_layers,decision_schedule,finite_control_benchmark
from .adaptive import prepare_next_cycle
from .training_store import NumericRows,UniverseRows,ControlCases,SplitRows,ControlLabelCache,require_disk_space


def _array(folder, name, value):
    file = folder/(name+".npy")
    require_disk_space(file,np.asarray(value).nbytes+1024)
    with file.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {"file": file.name, "bytes": file.stat().st_size, "sha256": file_hash(file)}


def _saved_training(folder, prefix):
    return {file.stem: np.load(file, allow_pickle=False) for file in folder.glob(prefix+"_*.npy")}


def _model_hash(arrays):
    return stable_hash({key:hashlib.sha256(np.asarray(value).tobytes()).hexdigest() for key,value in arrays.items() if not key.endswith("training_encodings")})


def _sample_policy_pairs(cases,per_state=8):
    """Keep complete grammar for scoring and stratify sampled rollout pairs."""
    selected=[]
    group=[]
    previous=None
    def flush():
        mandatory=[]
        by_family={}
        for index,action in group:
            by_family.setdefault(action["family"],[]).append((index,action))
        for family in sorted(by_family,key=lambda name:(name not in ("HOLD","FULL_CLOSE"),name)):
            index,_=min(by_family[family],key=lambda item:item[1]["id"])
            mandatory.append(index)
        chosen=mandatory[:per_state]
        for index,action in sorted(group,key=lambda item:item[1]["id"]):
            if len(chosen)>=per_state:break
            if index not in chosen:chosen.append(index)
        selected.extend(sorted(chosen))
    for index,(state,action) in enumerate(cases):
        identity=state.job.hash
        if previous is not None and identity!=previous:
            flush();group=[]
        group.append((index,action));previous=identity
    if group:flush()
    return selected


def _hold_label(job, result):
    from .train import HOLD_LABEL_SIZE,HOLD_LABEL_BASE_SIZE,HOLD_WITHIN_BIN_SLOTS
    from .hit_oracle import observation_grid
    n = result["production_paths"]
    if isinstance(n,bool) or not isinstance(n,int) or n<=0:
        raise RiskError("INVALID_LABEL_SAMPLE_COUNT")
    frozen_grid,_ = observation_grid(job,result["grid_seconds"])
    if result.get("grid_hash") != stable_hash(frozen_grid.tolist()):
        raise RiskError("HOLD_LABEL_GRID_MISMATCH","The saved histogram must use this exact requested-time grid")
    bins = survival_bins(job, result["grid_seconds"])
    index = tuple(map(float, job.targets)).index(float(job.target_usd))
    histogram = result["first_hit_histograms"][index]
    counts = np.zeros(576)
    lookup = {int(t):b for b,observations in enumerate(bins) for t in observations}
    times,first_counts = [],[]
    for instant, count in sorted(histogram.items(),key=lambda item:int(item[0])):
        instant = int(instant)
        if instant not in lookup:
            raise RiskError("HOLD_LABEL_GRID_MISMATCH",instant_utc_us=instant)
        if isinstance(count,bool) or not isinstance(count,int) or count<0:
            raise RiskError("INVALID_LABEL_SAMPLE_COUNT")
        counts[lookup[instant]] += count
        times.append(instant);first_counts.append(count)
    survivors = n-counts.sum()
    if survivors<0:
        raise RiskError("INVALID_LABEL_SAMPLE_COUNT")
    conditional = result["settlement_only"]*n/survivors if survivors else 0.
    if survivors==0 and result["settlement_only"]!=0:
        raise RiskError("INVALID_LABEL_SAMPLE_COUNT")
    terminal = endpoint_probability(job, job.fixing_utc_us, terminal_regions(job.signed_legs, job.entry_cash_usd, job.target_usd))
    label = np.zeros(HOLD_LABEL_SIZE,dtype=np.float64)
    label[:HOLD_LABEL_BASE_SIZE] = np.concatenate((counts, [survivors, result["terminal_cent_pop"], terminal, conditional], [float(len(x) > 0) for x in bins]))
    # Select only already observed instants. Sampling these optional labels does
    # not insert a reporting time into the frozen reference job or histogram.
    seed_hash=stable_hash({"job":job.computation_hash,"grid":result["grid_hash"],"schema":"QQQ_HOLD_LABEL_V2"})
    seed_words=tuple(int(seed_hash[offset:offset+8],16) for offset in range(0,32,8))
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261109,spawn_key=(1,10,*seed_words))))
    candidates=[b for b,observations in enumerate(bins) if len(observations)>1]
    selected=sorted(rng.choice(candidates,size=min(len(candidates),HOLD_WITHIN_BIN_SLOTS),replace=False).tolist()) if candidates else []
    cumulative=np.cumsum(first_counts,dtype=np.int64)
    samples=label[HOLD_LABEL_BASE_SIZE:].reshape(HOLD_WITHIN_BIN_SLOTS,5)
    for slot,b in enumerate(selected):
        rank=int(rng.integers(1,len(bins[b])))
        instant=int(bins[b][rank-1])
        hit_index=int(np.searchsorted(times,instant,side="right"))
        hits=int(cumulative[hit_index-1]) if hit_index else 0
        samples[slot]=[1,b,rank/len(bins[b]),hits,instant]
    return label


def _policy_selector(arrays, pack, limits, *, error=0, candidate_cap=2000, deadline=None):
    scoring_arrays=dict(pack.arrays)
    for key in ("universe_w0","universe_b0","universe_w1","universe_b1"):
        if "q_score_"+key in arrays:
            scoring_arrays[key]=arrays["q_score_"+key]
    score_pack=SimpleNamespace(price_basis=pack.price_basis,arrays=scoring_arrays)
    def selector(state):
        if deadline is not None and time.monotonic() >= deadline:
            raise RiskError("RESOURCE_LIMIT")
        search = generate_actions(state.job, state.campaign, state.universe, fresh_account_limits(state,limits), max_candidates=candidate_cap)
        from ..policy import mandatory_last_opportunity
        mandatory=mandatory_last_opportunity(state.job,state.campaign,search["actions"])
        if mandatory is not None:
            return mandatory
        choices = []
        from ..universe import universe_features
        state_features=managed_state_features(state.job,score_pack,state.campaign)
        universe_encoding=universe_features(state.job,state.universe,score_pack)
        for action in search["actions"]:
            if deadline is not None and time.monotonic() >= deadline:
                raise RiskError("RESOURCE_LIMIT")
            raw = action_features(state.job, state.campaign, action, score_pack,state.universe,state_features=state_features,universe_encoding=universe_encoding)
            scaled = (raw-arrays["q_score_mean"])/arrays["q_score_std"]
            score = float(np.exp(-np.logaddexp(0, -network(arrays, "q_score", scaled)[0])))
            risk = float(action.get("certificate", {}).get("operational_loss_bound_usd", 0))
            count = sum(abs(int(x["quantity"])) for x in action["legs"])
            choices.append((score, risk, float(action["premium_cost_usd"]), count, action["id"], action))
        if not choices:
            raise RiskError("NO_FEASIBLE_SUPPORTED_ACTION")
        top = max(x[0] for x in choices)
        tied = [x for x in choices if top-x[0] <= error]
        return min(tied, key=lambda x: (x[1], x[2], x[3], x[4]))[-1]
    return selector


def run_pipeline(config, model_config):
    research = bool(config.get("research", False))
    count = model_config["case_count"]
    transient=("budget_seconds","max_work_seconds","output_dir","work_dir","base_records_file","additional_records_file","parent_case_cache","price_basis_work_dir")
    semantic={key:value for key,value in config.items() if key not in transient}
    semantic_defaults={"epochs":20 if research else 300,"control_epochs":5 if research else 300,"research_paths":4096,"research_policy_paths":64,"research_candidate_cap":32,"price_initial_cell_limit":None,"price_stage_budget_fraction":1.0,"price_solve_budget_seconds":None,"quote_price_budget_seconds":None,"reference_job_budget_seconds":None}
    semantic={**semantic_defaults,**semantic}
    for key in ("base_records_file","additional_records_file"):
        if config.get(key):
            semantic[key+"_sha256"]=file_hash(Path(config[key]))
    sources={file.relative_to(Path(__file__).parent.parent).as_posix():file_hash(file) for file in sorted(Path(__file__).parent.parent.rglob("*.py"))}
    package_root=Path(__file__).parents[2]
    for file in sorted((package_root/"broker").rglob("*.py")):
        sources[file.relative_to(package_root).as_posix()]=file_hash(file)
    calendar_source=package_root/"replay"/"calendar.py"
    if calendar_source.is_file():sources["replay/calendar.py"]=file_hash(calendar_source)
    if not sources:
        from ..._build_info import SOURCE_HASHES
        sources={key.removeprefix("src/qqq_sim/risk/") if key.startswith("src/qqq_sim/risk/") else key.removeprefix("src/qqq_sim/"):value for key,value in SOURCE_HASHES.items() if key.startswith(("src/qqq_sim/risk/","src/qqq_sim/broker/")) or key=="src/qqq_sim/replay/calendar.py"}
    if not sources:
        raise RiskError("IMPLEMENTATION_IDENTITY_UNAVAILABLE")
    dependencies={}
    for name in ("numpy","scipy","torch","exchange_calendars","tzdata"):
        try:dependencies[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:dependencies[name]=None
    normalized = {**model_config, "research": research, "research_domain": "Q0_NONNEGATIVE_RATE_CALLS" if research else None,"semantic_configuration":semantic,"implementation_fingerprint":stable_hash(sources),"dependency_versions":dependencies}
    build_id = stable_hash(normalized)[:20]
    folder = writable_path(config.get("work_dir", f"runtime/builds/{build_id}"))
    folder.mkdir(parents=True, exist_ok=True)
    destination = writable_path(config.get("output_dir", f"runtime/model_packs/{build_id}"))
    checkpoint = folder/"pipeline_checkpoint.json"
    identity = stable_hash(normalized)
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["identity"] != identity:
            raise RiskError("CHECKPOINT_MISMATCH")
        for relative, digest in state.get("case_hashes", {}).items():
            if not (folder/relative).is_file() or file_hash(folder/relative) != digest:
                raise RiskError("CHECKPOINT_MISMATCH", artifact=relative)
    else:
        state = {"identity": identity, "stages": {}, "case_hashes": {}, "rejected": {}, "labels": [], "policy_rollouts": []}
    budget = float(config.get("budget_seconds", config.get("max_work_seconds", 600)))
    if state.get("adaptive_child"):
        child_config={**state["adaptive_child"],"budget_seconds":budget}
        child_model={**model_config,"case_count":child_config["case_count"]}
        report=run_pipeline(child_config,child_model)
        atomic_json(folder/"adaptive_resume_report.json",report)
        return {**report,"resumed_from_parent":str(checkpoint)}
    deadline = time.monotonic()+budget
    started = time.perf_counter()
    arrays, artifacts = {}, {}
    stores=[]
    coverage={}
    capabilities={name:{"qualified":False,"reason":"STAGE_NOT_COMPLETED","independent_cases":0} for name in ("price","hold","managed")}
    report_path = folder/"build_report.json"

    def remaining():
        return max(0., deadline-time.monotonic())

    def save():
        atomic_json(checkpoint, state)

    def require_time(stage):
        if remaining() <= 0:
            raise RiskError("RESOURCE_LIMIT", stage=stage, checkpoint=str(checkpoint))

    def save_case(index, name, value):
        file = folder/"cases"/f"{index:06d}_{name}.json"
        atomic_json(file, value)
        state["case_hashes"][str(file.relative_to(folder))] = file_hash(file)
        save()

    state["stages"]["B0"] = {"status": "COMPLETE", "config": normalized, "seeds": SEED_KEYS, "numpy": np.__version__}
    state.pop("last_stop", None)
    save()
    try:
        require_time("B1")
        design_path = folder/"design.npz"
        if not design_path.exists():
            design = latin_hypercube(count)
            np.savez(design_path, **design)
        else:
            with np.load(design_path, allow_pickle=False) as loaded:
                design = {key: loaded[key] for key in loaded.files}
        records = [design_record(i, design["coordinates"][i]) for i in range(count)]
        if config.get("base_records_file"):
            records=json.loads(Path(config["base_records_file"]).read_text(encoding="utf-8"))
            if config.get("additional_records_file"):
                records+=json.loads(Path(config["additional_records_file"]).read_text(encoding="utf-8"))
            if len(records)!=count or [x["index"] for x in records]!=list(range(count)):
                raise RiskError("ADAPTIVE_DESIGN_SCHEMA_MISMATCH")
        elif not research:
            from .edge_design import heldout_design_records
            records+=heldout_design_records(len(records))
            count=len(records)
        if config.get("parent_case_cache"):
            # Verify immutable parent receipts before copying reusable work.
            import shutil
            parent=Path(config["parent_case_cache"])
            parent_state=json.loads((parent/"pipeline_checkpoint.json").read_text(encoding="utf-8"))
            parent_config=parent_state.get("stages",{}).get("B0",{}).get("config",{})
            if parent_config.get("implementation_fingerprint")!=normalized["implementation_fingerprint"] or parent_config.get("research")!=research:
                raise RiskError("CHECKPOINT_MISMATCH","Parent cached labels belong to another numerical implementation or research scope")
            for relative,digest in parent_state.get("case_hashes",{}).items():
                if not relative.endswith(("_input.json","_label.json")):
                    continue
                original=parent/relative
                if file_hash(original)!=digest:
                    raise RiskError("CHECKPOINT_MISMATCH",artifact=str(original))
                target=folder/relative
                if not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(original,target)
                    state["case_hashes"][relative]=digest
        state["stages"]["B1"] = {"status": "DESIGN_COORDINATES_PERSISTED", "count": count,"base_design_count":model_config["case_count"],"mandatory_heldout_cases":sum(bool(row.get("heldout_edge_family")) for row in records), "shape_group_splits": True,"heldout_families_excluded_from_training_calibration_and_augmentation":True}
        save()
        if not research:
            require_time("B2")
            # One pass is budgeted, and resumed cells retain all accepted evidence.
            fraction=float(config.get("price_stage_budget_fraction",1.0))
            if not 0<fraction<=1:raise RiskError("INVALID_WORK_BUDGET")
            basis_report = build_price_basis(config.get("price_basis_work_dir",folder/"price_basis"), budget_seconds=remaining()*fraction, max_initial_cells=config.get("price_initial_cell_limit"),max_solve_seconds=config.get("price_solve_budget_seconds"))
            state["stages"]["B2"] = basis_report
            save()
            from ..pricing_fast import PriceBasis
            bounds, coefficients, errors,greek_errors = [], [], [],[]
            for cell in basis_report["accepted"]:
                with np.load(Path(basis_report["directory"])/cell["file"], allow_pickle=False) as data:
                    bounds.append(data["bounds"]); coefficients.append(data["coefficients"]); errors.append(data["errors"]);greek_errors.append(data["greek_errors"])
            price_tree = build_price_tree(np.asarray(bounds)) if bounds else {}
            basis = PriceBasis(np.asarray(bounds), np.asarray(coefficients), np.asarray(errors),np.asarray(greek_errors),tree_nodes=price_tree["price_tree_nodes"],tree_bounds=price_tree["price_tree_bounds"]) if bounds else None
            if basis is not None:
                arrays.update(price_bounds=np.asarray(bounds),price_coefficients=np.asarray(coefficients),price_errors=np.asarray(errors),price_greek_errors=np.asarray(greek_errors))
                arrays.update(price_tree)
                capabilities["price"]={"qualified":bool(bounds),"independent_cases":sum(x["validation_count"] for x in basis_report["accepted"]),"pass_fraction":1.,"accepted_cells":len(bounds),"excluded_cells":len(basis_report["excluded"]),"greeks_qualified":all(x.get("greeks_qualified",False) for x in basis_report["accepted"]),"reason":None if bounds else "NO_ACCEPTED_PRICE_CELLS"}
        else:
            # This is an exact American special case, explicitly excluded from
            # claiming a price basis qualified for general American QQQ options.
            state["stages"]["B2"] = {"status": "EXACT_RESEARCH_SPECIAL_CASE", "profile": "Q0_NONNEGATIVE_RATE_AMERICAN_CALL_EQUALS_EUROPEAN", "general_price_basis_qualified": False}
            basis = None
        from .train_checkpoint import numeric_identity
        basis_identity=stable_hash({name:numeric_identity(value) for name,value in arrays.items() if name.startswith("price_")})
        pack = SimpleNamespace(price_basis=basis, arrays=arrays, manifest={"monitoring_seconds": 15}, pack_id=build_id,basis_identity=basis_identity)
        state["stages"]["B3"] = {"status": "EXACT_GBM_AND_TRUNCATED_NORMAL_KERNEL_IMPLEMENTED", "mass_discarding": False}
        cases = {}
        job_budget=lambda:max(.01,min(float(config["reference_job_budget_seconds"]),remaining()) if config.get("reference_job_budget_seconds") is not None else remaining())
        settings = NumericalSettings(production_paths=int(config.get("research_paths", 4096)) if research else 200000, max_work_seconds=job_budget(), space_nodes=1601)
        for record in records:
            require_time("B4")
            index = record["index"]
            input_file = folder/"cases"/f"{index:06d}_input.json"
            label_file = folder/"cases"/f"{index:06d}_label.json"
            try:
                if input_file.exists():
                    saved = json.loads(input_file.read_text(encoding="utf-8"))
                    from ..contracts import RiskLeg
                    case = {"job": ReferenceJob.from_dict(saved["job"]), "campaign": saved["campaign"], "universe": tuple(RiskLeg.from_dict(x) for x in saved["universe"]), "record": saved.get("record",record), "scope": saved["scope"], "generator_limitations": saved["generator_limitations"],"account_context":saved.get("account_context")}
                else:
                    modified_record = dict(record)
                    if research:
                        modified_record["r"] = abs(modified_record["r"])
                    def provider(right,s,k,tau,r,q,sigma):
                        if basis is not None:
                            try:return basis.evaluate(right,s,k,tau,r,q,sigma)["price"]
                            except RiskError as exc:
                                if exc.code not in ("PRICE_BASIS_UNVALIDATED_CELL","PRICE_BASIS_OUTSIDE_DOMAIN","PRICE_BASIS_MISSING"):raise
                        from .pricing_oracle import american_price
                        budget=remaining()
                        if config.get("quote_price_budget_seconds") is not None:budget=min(budget,float(config["quote_price_budget_seconds"]))
                        if budget<=0:raise RiskError("RESOURCE_LIMIT","Synthetic quote pricing exhausted the shared build budget")
                        return american_price(right,s,k,tau,r,q,sigma,eps=1e-7,max_seconds=budget).price
                    if research:provider=None
                    case = materialize_case(modified_record, research=research, price_provider=provider, numerical_settings=settings,deadline=deadline)
                    save_case(index, "input", canonical(case))
                if not case["job"].signed_legs:
                    # A flat campaign still has a managed reentry objective,
                    # but no HOLD first-hit option event to label or train.
                    cases[index]=case
                    state.setdefault("managed_only_cases",[])
                    if index not in state["managed_only_cases"]:
                        state["managed_only_cases"].append(index)
                    state["rejected"].pop(str(index),None)
                    save()
                    continue
                if label_file.exists():
                    label = json.loads(label_file.read_text(encoding="utf-8"))
                else:
                    label=None
                if label is None or label.get("reason") in ("RESOURCE_LIMIT","CANCELLED","CANCELED"):
                    if label is not None:
                        atomic_json(folder/"cases"/"attempts"/(f"{index:06d}_"+stable_hash(label)[:16]+".json"),label)
                    job = dataclasses.replace(case["job"], numerical_settings=dataclasses.replace(case["job"].numerical_settings, max_work_seconds=job_budget()))
                    label = calculate_reference(job, checkpoint=folder/"cases"/f"{index:06d}_paths_checkpoint.json", allow_preliminary=research)
                    save_case(index, "label", label)
                if label.get("status") not in (("OK", "PRELIMINARY") if research else ("OK",)) or (not research and not label.get("release_qualified")):
                    raise RiskError(label.get("reason", "REFERENCE_LABEL_NOT_QUALIFIED"))
                cases[index] = case
                state["rejected"].pop(str(index),None)
                if index not in state["labels"]:
                    state["labels"].append(index)
                save()
            except RiskError as exc:
                if exc.code == "RESOURCE_LIMIT":
                    raise
                state["rejected"][str(index)] = {"reason": exc.code, "details": exc.details}
                save()
        state["stages"]["B4"] = {"status": "COMPLETE" if len(cases) == count else "PARTIAL_COVERAGE", "labels": len(cases), "requested": count, "preliminary_labels_permitted": research, "rejections": state["rejected"]}
        require_time("B5")
        if len(cases) < 3:
            raise RiskError("TRAINING_LABEL_COVERAGE_INSUFFICIENT", qualified_labels=len(cases))
        ordered = sorted(index for index,case in cases.items() if case["job"].signed_legs)
        if len(ordered)<3:
            raise RiskError("HOLD_TRAINING_LABEL_COVERAGE_INSUFFICIENT",hold_cases=len(ordered),managed_cases=len(cases))
        features, labels = [], []
        for index in ordered:
            case = cases[index]
            result = json.loads((folder/"cases"/f"{index:06d}_label.json").read_text(encoding="utf-8"))
            features.append(hold_features(case["job"], pack))
            labels.append(_hold_label(case["job"], result))
        hold_label_artifact=_array(folder,"hold_labels_v2",np.asarray(labels))
        train_indices = [i for i, index in enumerate(ordered) if records[index]["split"] == "train"]
        calibration_indices = [i for i, index in enumerate(ordered) if records[index]["split"] == "calibration"]
        validation_indices = [i for i, index in enumerate(ordered) if records[index]["split"] == "validation"]
        if research and (not train_indices or not calibration_indices):
            # Only diagnostic runs may make a predeclared alternate split.
            train_indices, calibration_indices, validation_indices = list(range(max(1, len(ordered)-2))), [len(ordered)-2], [len(ordered)-1]
        from ..universe import hold_flag_indices,universe_tokens,universe_feature_mask_indices
        training = train_network(np.asarray(features), np.asarray(labels), train_indices, calibration_indices, epochs=int(config.get("epochs", 20 if research else 300)),mask_indices=hold_flag_indices(),deadline=deadline,training_directory=folder/"training"/"hold")
        arrays.update(training["arrays"])
        for name, value in arrays.items():
            artifacts[name] = _array(folder, name, value)
        atomic_json(folder/"hold_training.json", {k: v for k, v in training.items() if k != "arrays"})
        state["stages"]["B5"] = {"status": "TRAINED", "epochs": len(training["history"]), "train_cases": len(train_indices), "calibration_cases": len(calibration_indices), "untouched_validation_cases": len(validation_indices)}
        state["stages"]["B5"].update(label_schema=HOLD_LABEL_SCHEMA,loss_schema=HOLD_LOSS_SCHEMA,frozen_label_artifact=hold_label_artifact)
        save()
        require_time("B6")
        from .universe_train import initialize_universe_arrays
        arrays.update(initialize_universe_arrays())
        # Exact candidate generation precedes integration and preserves the full
        # grammar; research case/action caps are disclosed as reduced scope.
        limits = {"available_cash_usd": 5000, "loss_cap_usd": 1000,"explicit_unqualified_research_account":research}
        control_inputs=NumericRows(folder/"control_rows"/"features.bin",MANAGED_FEATURE_COUNT)
        control_targets=NumericRows(folder/"control_rows"/"targets.bin",1)
        control_cases=ControlCases(folder/"control_rows"/"cases")
        control_tokens=UniverseRows(folder/"control_rows"/"tokens")
        control_splits=SplitRows(folder/"control_rows"/"splits.bin")
        label_cache=ControlLabelCache(folder/"control_target_cache.sqlite3")
        stores.extend((control_inputs,control_targets,control_cases,control_tokens,control_splits,label_cache))
        action_count = 0
        current_score = None
        continuation_hash="EXACT_ABSORBING_TERMINAL_LAYER"
        eligible_cases=[]
        for case in cases.values():
            if not research and not case.get("account_context"):
                from ..account_adapter import create_isolated_account_context
                try:
                    case["account_context"]=create_isolated_account_context(case["job"],case["campaign"],case["universe"])
                except RiskError as exc:
                    state["rejected"][f"account:{case['record']['index']}"]={"reason":exc.code}
                    continue
            eligible_cases.append(case)
        groups,layer_report=complete_minute_layers(eligible_cases,pack,deadline=deadline,checkpoint_dir=folder/"control_layers")
        state["control_layers"]=layer_report
        save()
        for case_index,case in enumerate(groups):
            require_time("B7_SCORING")
            context=case.get("account_context") if not research else None
            initial = ManagedState(case["job"], case["campaign"], case["universe"],account_context=context)
            case={**case,'campaign':initial.campaign}
            search = generate_actions(case["job"], case["campaign"], case["universe"], fresh_account_limits(initial,limits), max_candidates=int(config.get("research_candidate_cap", 32)) if research else 2000)
            case_tokens=universe_tokens(case["job"],case["universe"])
            source_state_hash=stable_hash([initial.job.hash,initial.campaign,initial.universe,initial.account_context])
            for action in search["actions"]:
                require_time("B7_SCORING")
                def continuation(normal):
                    require_time("B7_SCORING")
                    next_state = transition(initial, action, normal, pack, limits)
                    if next_state.success:
                        return 1.
                    if next_state.failed:
                        return 0.
                    if current_score is None:
                        # Only the exact terminal layer can bootstrap without
                        # a fitted previous layer. Missing continuation is not 0.
                        if len(decision_schedule(next_state.job,next_state.campaign)):
                            raise RiskError("CONTROL_PREVIOUS_LAYER_UNAVAILABLE")
                        return 0.
                    next_search = generate_actions(next_state.job, next_state.campaign, next_state.universe, fresh_account_limits(next_state,limits), max_candidates=int(config.get("research_candidate_cap", 32)) if research else 2000)
                    values = []
                    from ..universe import universe_features
                    score_arrays={**arrays,**{k:current_score.get("q_score_"+k,arrays[k]) for k in ("universe_w0","universe_b0","universe_w1","universe_b1")}}
                    score_pack=SimpleNamespace(price_basis=pack.price_basis,arrays=score_arrays)
                    next_features=managed_state_features(next_state.job,score_pack,next_state.campaign)
                    next_universe=universe_features(next_state.job,next_state.universe,score_pack)
                    for candidate in next_search["actions"]:
                        require_time("B7_SCORING")
                        raw = action_features(next_state.job, next_state.campaign, candidate, score_pack,next_state.universe,state_features=next_features,universe_encoding=next_universe)
                        scaled = (raw-current_score["q_score_mean"])/current_score["q_score_std"]
                        values.append(float(np.exp(-np.logaddexp(0, -network(current_score, "q_score", scaled)[0]))))
                    return max(values, default=0.)
                try:
                    label_key=stable_hash([source_state_hash,action["id"],continuation_hash,basis_identity])
                    target=label_cache.get(label_key)
                    if target is None:
                        target = converged_expectation(continuation)
                        label_cache.put(label_key,target)
                except RiskError as exc:
                    if exc.code == "RESOURCE_LIMIT":
                        raise
                    state["rejected"][f"control:{case['record']['index']}:{action['id']}"] = {"reason": exc.code}
                    continue
                raw = action_features(case["job"], case["campaign"], action, pack,case["universe"])
                control_inputs.append(raw); control_targets.append([target["value"]]); control_cases.append((initial, action))
                control_tokens.append(case_tokens)
                control_splits.append(case["record"]["split"])
                action_count += 1
            layer=len(decision_schedule(case["job"],case["campaign"]))
            last_in_layer=case_index+1==len(groups) or len(decision_schedule(groups[case_index+1]["job"],groups[case_index+1]["campaign"]))!=layer
            if len(control_inputs) >= 3 and last_in_layer:
                n = len(control_inputs)
                flags=managed_masks()
                score_train=[i for i,split in enumerate(control_splits) if split=="train"]
                score_calibration=[i for i,split in enumerate(control_splits) if split=="calibration"]
                if research and (not score_train or not score_calibration):
                    score_train,score_calibration=list(range(n-1)),[n-1]
                if not score_train or not score_calibration:
                    continue
                fitted = train_network(control_inputs, control_targets, score_train, score_calibration, output_size=1, kind="q_score", epochs=int(config.get("control_epochs", 5 if research else 300)),mask_indices=flags,universe_tokens=control_tokens,universe_arrays={k:current_score.get("q_score_"+k,arrays[k]) if current_score is not None else arrays[k] for k in ("universe_w0","universe_b0","universe_w1","universe_b1")},deadline=deadline,training_directory=folder/"training"/f"score_layer_{layer}",export_training_encodings=False)
                current_score = fitted["arrays"]
                continuation_hash=_model_hash(current_score)
        if current_score is None:
            raise RiskError("CONTROL_LABEL_COVERAGE_INSUFFICIENT")
        arrays.update(current_score)
        state["stages"]["B6"] = {"status": "GENERATED_AND_EXACT_FILTERED", "scoring_pairs": action_count, "candidate_cap": int(config.get("research_candidate_cap", 32)) if research else 2000}
        # Independently evaluate the frozen selector, never relabel Bellman-max
        # scores as probabilities of this returned policy.
        selector = _policy_selector(current_score, pack, limits, candidate_cap=int(config.get("research_candidate_cap", 32)) if research else 2000, deadline=deadline)
        policy_id = stable_hash({"weights": _model_hash(current_score),"price_basis":basis_identity, "selector": "LOWER_RISK_COST_SIZE_ID_V1", "action_grammar": "P50_M07","selector_and_generator_source":sources.get("policy.py")})
        eval_inputs=NumericRows(folder/"eval_rows"/"features.bin",MANAGED_FEATURE_COUNT)
        eval_targets=NumericRows(folder/"eval_rows"/"targets.bin",1)
        eval_tokens=UniverseRows(folder/"eval_rows"/"tokens")
        eval_source_indices=NumericRows(folder/"eval_rows"/"sources.bin",1,np.int64,scalar=True)
        stores.extend((eval_inputs,eval_targets,eval_tokens,eval_source_indices))
        rollout_pair_indices=_sample_policy_pairs(control_cases,int(config.get("policy_pairs_per_state",8)))
        for i in rollout_pair_indices:
            initial,action=control_cases[i]
            require_time("B7_POLICY_EVALUATION")
            evaluated = rollout(initial, action, selector, pack, paths=int(config.get("research_policy_paths", 64)) if research else 20000, limits=limits, deadline=deadline,checkpoint=folder/"policy_paths"/f"train_{i:09d}.json",policy_identity=policy_id)
            save_case(i, "policy_rollout", evaluated)
            if evaluated["status"] == "OK":
                eval_inputs.append(control_inputs[i]); eval_targets.append([evaluated["probability_lower"]])
                eval_tokens.append(control_tokens[i])
                eval_source_indices.append(i)
        if len(eval_inputs) < 3:
            raise RiskError("POLICY_EVALUATION_COVERAGE_INSUFFICIENT")
        n = len(eval_inputs)
        eval_train=[i for i,source in enumerate(eval_source_indices) if control_splits[source]=="train"]
        eval_calibration=[i for i,source in enumerate(eval_source_indices) if control_splits[source]=="calibration"]
        if research and (not eval_train or not eval_calibration):
            eval_train,eval_calibration=list(range(n-1)),[n-1]
        fitted_eval = train_network(eval_inputs, eval_targets, eval_train, eval_calibration, output_size=1, kind="q_eval", epochs=int(config.get("control_epochs", 5 if research else 300)), seed=20261115,mask_indices=flags,universe_tokens=eval_tokens,universe_arrays={k:current_score["q_score_"+k] for k in ("universe_w0","universe_b0","universe_w1","universe_b1")},deadline=deadline,training_directory=folder/"training"/"policy_eval")
        arrays.update(fitted_eval["arrays"])
        arrays.update({key:arrays["q_eval_"+key] for key in ("universe_w0","universe_b0","universe_w1","universe_b1")})
        arrays["managed_mean"] = arrays["q_eval_mean"]
        arrays["managed_std"] = arrays["q_eval_std"]
        arrays["managed_training_encodings"] = arrays["q_eval_training_encodings"]
        state["stages"]["B7"] = {"status": "SCORER_FROZEN_AND_POLICY_EVALUATION_TRAINED", "policy_id": policy_id, "independent_rollout_cases": len(eval_inputs), "paths_per_state": int(config.get("research_policy_paths", 64)) if research else 20000,"minute_layers":layer_report,"within_layer_targets_frozen":True,"rollout_pair_sampling":"UP_TO_EIGHT_FAMILY_STRATIFIED_PER_STATE_FROM_FULL_SCORING_GRAMMAR","training_storage":"DISK_BACKED_ROWS_SHARED_UNIVERSE_POOL_256_ROW_TORCH_BATCHES","disk_reserve":"MAX_1_GIB_OR_5_PERCENT_CAPACITY"}
        save()
        # Full regime maps, held-out family counts, price/Greek and all stress
        # evidence must pass separately; a trained artifact is never a pass.
        require_time("B8_HOLD_VALIDATION")
        hold_pred, hold_truth, hold_uncertainty = [], [], []
        validation_rows,calibration_rows = [],[]
        validation_features,validation_fractions,validation_terminal,validation_reference,validation_errors=[],[],[],[],[]
        within_outputs,within_labels,within_numerical,within_deterministic,within_families=[],[],[],[],[]
        for row in sorted(set(calibration_indices+validation_indices)):
            case = cases[ordered[row]]
            observed = json.loads((folder/"cases"/f"{ordered[row]:06d}_label.json").read_text(encoding="utf-8"))
            scaled = (features[row]-arrays["hold_mean"])/arrays["hold_std"]
            output = network(arrays, "hold", scaled)
            hazards = np.logaddexp(0, output[:576])
            bins = survival_bins(case["job"], observed["grid_seconds"])
            closing = survival_cdf(hazards, bins, case["job"].deadline_utc_us)
            predicted = closing
            if case["job"].deadline_utc_us == case["job"].fixing_utc_us:
                terminal_conditional = float(np.exp(-np.logaddexp(0, -output[-1])))
                predicted += (1-closing)*terminal_conditional
            interval = observed.get("sampling_interval") or [observed["hold_p50"]]*2
            uncertainty = (interval[1]-interval[0])/2+observed.get("numerical_sensitivity", {}).get("region_probability_upper", 0)
            item={"case_index": ordered[row],"regime":regime_key(case["job"],features=features[row]),"encoding":scaled, "prediction": predicted, "reference": observed["hold_p50"], "reference_uncertainty": uncertainty, "reference_qualified": observed.get("release_qualified", False)}
            case_sensitivity=observed.get("numerical_sensitivity",{})
            case_numeric=sum(float(case_sensitivity.get(key,0.)) for key in ("region_probability_upper","refinement_probability_difference","domain_exit_bound"))
            within_case=within_bin_validation(np.asarray(output)[None,:],np.asarray(labels[row])[None,:],[case_numeric],deterministic=[case["job"].sigma_path==0])
            item["auxiliary_error"]=within_case["case_errors"][0]
            item["auxiliary_error_scope"]="SAMPLED_WITHIN_BIN_CDF_INCLUDING_REFERENCE_UNCERTAINTY"
            if row in validation_indices:
                hold_pred.append(predicted); hold_truth.append(observed["hold_p50"]); hold_uncertainty.append(uncertainty)
                validation_rows.append(item)
                validation_features.append(features[row])
                validation_fractions.append([int(np.searchsorted(bin_times,case["job"].deadline_utc_us,side="right"))/len(bin_times) if len(bin_times) else 0. for bin_times in bins])
                validation_terminal.append(float(case["job"].deadline_utc_us==case["job"].fixing_utc_us))
                validation_reference.append(observed["hold_p50"])
                validation_errors.append(uncertainty)
                within_outputs.append(output)
                within_labels.append(labels[row])
                sensitivity=observed.get("numerical_sensitivity",{})
                within_numerical.append(sum(float(sensitivity.get(key,0.)) for key in ("region_probability_upper","refinement_probability_difference","domain_exit_bound")))
                within_deterministic.append(bool(case["job"].sigma_path==0))
                within_families.append(case["record"].get("heldout_edge_family"))
            else:
                calibration_rows.append(item)
        hold_validation = calibration_report(hold_pred, hold_truth, hold_uncertainty) if hold_pred else {"qualified": False, "reason": "NO_UNTOUCHED_VALIDATION_CASES", "count": 0}
        within_validation=within_bin_validation(within_outputs,within_labels,within_numerical,deterministic=within_deterministic) if within_labels else {"qualified":False,"reason":"NO_UNTOUCHED_VALIDATION_CASES","independent_cases":0}
        within_family_gates={}
        for family in sorted(set(within_families)-{None}):
            chosen=np.array([i for i,value in enumerate(within_families) if value==family])
            gate=within_bin_validation(np.asarray(within_outputs)[chosen],np.asarray(within_labels)[chosen],np.asarray(within_numerical)[chosen],deterministic=np.asarray(within_deterministic)[chosen])
            within_family_gates[family]={key:value for key,value in gate.items() if key not in ("case_errors","applicable_cases")}
        within_validation["heldout_family_gates"]=within_family_gates
        within_validation["qualified"]=within_validation["qualified"] and all(gate["qualified"] or gate["singleton_only_cases"]==gate["total_cases"] for gate in within_family_gates.values())
        atomic_json(folder/"hold_validation.json", {"summary": hold_validation,"within_bin":within_validation,"label_schema":HOLD_LABEL_SCHEMA,"loss_schema":HOLD_LOSS_SCHEMA, "cases": [{key:value for key,value in row.items() if key!="encoding"} for row in validation_rows]})
        if validation_features:
            arrays.update(hold_validation_features=np.asarray(validation_features),hold_validation_bin_fractions=np.asarray(validation_fractions),hold_validation_terminal=np.asarray(validation_terminal),hold_validation_reference=np.asarray(validation_reference),hold_validation_uncertainty=np.asarray(validation_errors))
            arrays.update(hold_within_validation_labels=np.asarray(within_labels),hold_within_validation_numerical_uncertainty=np.asarray(within_numerical),hold_within_validation_deterministic=np.asarray(within_deterministic,dtype=np.uint8))
        coverage["hold"]=qualify_regimes(arrays["hold_training_encodings"],[regime_key(cases[ordered[i]]["job"],features=features[i]) for i in train_indices],calibration_rows,validation_rows,tolerance=.01)
        capabilities["hold"]={**{k:v for k,v in coverage["hold"].items() if k!="regimes"},"qualified":coverage["hold"]["qualified"] and not research and capabilities["price"].get("qualified",False)}
        capabilities["hold"]["within_bin_validation"]={key:value for key,value in within_validation.items() if key not in ("case_errors","applicable_cases")}
        capabilities["hold"]["qualified"]=capabilities["hold"]["qualified"] and within_validation["qualified"]
        if not within_validation["qualified"]:capabilities["hold"]["reason"]=within_validation.get("reason") or "WITHIN_BIN_HELDOUT_FAMILY_GATE_FAILED"
        require_time("B8_POLICY_VALIDATION")
        policy_pred, policy_truth, policy_uncertainty, policy_rows = [], [], [], []
        policy_calibration_rows,policy_validation_rows=[],[]
        policy_features,policy_references,policy_errors=[],[],[]
        for i in rollout_pair_indices:
            initial,action=control_cases[i]
            require_time("B8_POLICY_VALIDATION")
            independent = rollout(initial, action, selector, pack, paths=int(config.get("research_policy_paths", 64)) if research else 20000, seed=20261114, limits=limits, deadline=deadline,checkpoint=folder/"policy_paths"/f"validation_{i:09d}.json",policy_identity=policy_id)
            if independent["status"] != "OK":
                policy_rows.append({"case": i, "status": independent["status"], "unknown_mass": independent["unknown"]/independent["paths"]})
                continue
            raw = action_features(initial.job, initial.campaign, action, pack,initial.universe)
            scaled = (raw-arrays["q_eval_mean"])/arrays["q_eval_std"]
            predicted = float(np.exp(-np.logaddexp(0, -network(arrays, "q_eval", scaled)[0])))
            interval = independent["wilson_success"]
            policy_pred.append(predicted); policy_truth.append(independent["probability_lower"]); policy_uncertainty.append((interval[1]-interval[0])/2)
            policy_rows.append({"case": i, "prediction": predicted, "independent_probability": independent["probability_lower"], "sampling_interval": interval, "paths": independent["paths"], "seed": 20261114})
            source_case=next((case for case in cases.values() if case["job"].snapshot_id==initial.job.snapshot_id),None)
            encoded={"case_index":source_case["record"]["index"] if source_case else -1,"regime":regime_key(initial.job,initial.campaign,features=raw),"encoding":scaled,"prediction":predicted,"reference":independent["probability_lower"],"reference_uncertainty":(interval[1]-interval[0])/2,"reference_qualified":independent["paths"]>=20000 and not research}
            origin=control_splits[i]
            (policy_calibration_rows if origin=="calibration" else policy_validation_rows if origin=="validation" else []).append(encoded)
            if origin=="validation":
                policy_features.append(raw);policy_references.append(independent["probability_lower"]);policy_errors.append((interval[1]-interval[0])/2)
        policy_validation = calibration_report(policy_pred, policy_truth, policy_uncertainty, tolerance=.02) if policy_pred else {"qualified": False, "reason": "NO_VALID_POLICY_HOLDOUT", "count": 0}
        atomic_json(folder/"policy_validation.json", {"summary": policy_validation, "cases": policy_rows})
        if policy_features:
            arrays.update(managed_validation_features=np.asarray(policy_features),managed_validation_reference=np.asarray(policy_references),managed_validation_uncertainty=np.asarray(policy_errors))
        policy_training_keys=[regime_key(control_cases[eval_source_indices[i]][0].job,control_cases[eval_source_indices[i]][0].campaign,features=eval_inputs[i]) for i in eval_train]
        coverage["managed"]=qualify_regimes(arrays["q_eval_training_encodings"],policy_training_keys,policy_calibration_rows,policy_validation_rows,tolerance=.02)
        benchmark_rows=[]
        benchmark_limit=int(config.get("control_benchmark_cases",2 if research else 200))
        seen_benchmarks=set()
        for initial,_ in control_cases:
            if initial.job.hash in seen_benchmarks or len(benchmark_rows)>=benchmark_limit or len(decision_schedule(initial.job,initial.campaign))>3:
                continue
            seen_benchmarks.add(initial.job.hash)
            require_time("B8_CONTROL_BENCHMARK")
            benchmark=finite_control_benchmark(initial,pack,limits=limits,deadline=min(deadline,time.monotonic()+float(config.get("control_benchmark_seconds",5 if research else 120))))
            item={"job_hash":initial.job.hash,"benchmark":benchmark}
            if benchmark["status"]=="CONVERGED_FINITE_SUBSET_BENCHMARK":
                chosen=selector(initial)
                actual=rollout(initial,chosen,selector,pack,paths=int(config.get("research_policy_paths",64)) if research else 20000,seed=20261116,limits=limits,deadline=deadline,checkpoint=folder/"benchmark_policy_paths"/(initial.job.hash+".json"),policy_identity=policy_id)
                item.update(returned_policy=actual,measured_regret=max(0.,benchmark["value"]-actual["probability_lower"]),regret_upper=max(0.,benchmark["value"]+.0025-actual["wilson_success"][0]))
            benchmark_rows.append(item)
        atomic_json(folder/"control_benchmarks.json",{"cases":benchmark_rows,"global_optimality_claim":False,"unavailable_benchmarks_excluded":True})
        from .stress import stress_report,stress_policy_report
        stresses=stress_report(list(cases.values()))
        atomic_json(folder/"stress_validation.json",stresses)
        stress_probabilities=stress_policy_report(list(cases.values())[:int(config.get("stress_case_count",1 if research else 200))],selector,pack,paths=int(config.get("research_policy_paths",64)) if research else 20000,limits=limits,deadline=deadline,checkpoint_dir=folder/"stress_policy_paths",policy_identity=policy_id)
        atomic_json(folder/"stress_policy_validation.json",stress_probabilities)
        capabilities["managed"]={**{k:v for k,v in coverage["managed"].items() if k!="regimes"},"qualified":coverage["managed"]["qualified"] and not research and capabilities["price"].get("greeks_qualified",False) and stresses["all_required_scenarios_exercised"] and stresses["accepted_known_constraint_violations"]==0 and stress_probabilities["all_required_stress_outcomes_classified"] and layer_report["status"]=="COMPLETE","independent_policy_rollouts":True,"policy_id":policy_id,"control_benchmarks":benchmark_rows,"stress_probability_evidence":stress_probabilities,"stress_outside_coverage_allowed_by":"P50-M15"}
        # State-family/regime qualification and stress coverage remain stricter
        # than a pooled score: tiny research runs are permanently unqualified.
        all_qualified=all(x.get("qualified",False) for x in capabilities.values())
        state["stages"]["B8"] = {"status": "QUALIFIED" if all_qualified else "MEASURED_UNQUALIFIED", "reason":None if all_qualified else "REGIME_COUNTS_PRICE_GREEK_AND_STRESS_GATES_NOT_MET", "research": research, "hold_validation": hold_validation, "policy_validation": policy_validation,"capabilities":capabilities, "independent_policy_seed": 20261114, "regime_minimum_independent_cases": 200}
        save()
        if not research:
            # Adapt only to calibration errors. Original held-out groups and
            # labels remain untouched throughout at most six added designs.
            adaptive=prepare_next_cycle(folder/"adaptive",records,calibration_rows,int(config.get("adaptive_cycle",0)))
            state["adaptive"]=adaptive
            if adaptive.get("next_cycle") is not None:
                cycle=adaptive["next_cycle"]
                child={**config,"adaptive_cycle":cycle,"base_records_file":adaptive["base_records_file"],"additional_records_file":adaptive["additional_records_file"],"parent_case_cache":str(folder),"price_basis_work_dir":str(config.get("price_basis_work_dir",folder/"price_basis")),"case_count":count+8192,"work_dir":str(folder/f"adaptive_cycle_{cycle}"),"output_dir":str(destination),"budget_seconds":remaining()}
                state["adaptive_child"]=child
            save()
    except RiskError as exc:
        state["last_stop"] = exc.as_dict()
        save()
    finally:
        for name, value in arrays.items():
            try:
                artifacts[name] = _array(folder, name, value)
            except RiskError as exc:
                state["last_stop"]=exc.as_dict()
                artifacts.pop(name,None)
        # Export an immutable timestamp-free revision keyed by current completed
        # work. Resuming creates a new revision, preserving previously loaded bytes.
        evidence_hash = stable_hash({"stages": state["stages"], "cases": state["case_hashes"], "artifacts": artifacts})[:12]
        pack_dir = destination/evidence_hash
        staging_dir=destination/(".building_"+evidence_hash)
        staging_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for artifact in artifacts.values():
            target = staging_dir/artifact["file"]
            if not target.exists():
                # Same-workspace atomic move avoids a second huge numeric copy.
                # Subsequent builds create new source files; published bytes
                # share no writable file with the resumed working directory.
                os.replace(folder/artifact["file"],target)
        versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch")}
        qualified=not research and all(x.get("qualified",False) for x in capabilities.values())
        manifest = {"schema_version": "QQQ_P50_PACK_V1", "pack_id": build_id+"_"+evidence_hash, "pack_kind": "RESEARCH_PIPELINE" if research else "PRODUCTION" if qualified else "PRODUCTION_BUILD_CHECKPOINT", "model_config": normalized, "dtype": "float64", "profiles": {"reference": "P50_REFERENCE_V1", "managed": "FROZEN_KNOWN_SURFACE_QQQ_V1", "execution": "SIM_CONDITIONAL_RISK_CHECK_V1", "risk": "SIMULTANEOUS_DELIVERY_HEDGE_V1"}, "artifacts": artifacts, "feature_schema": FEATURE_SCHEMA, "monitoring_seconds": 15, "policy_id": state["stages"].get("B7", {}).get("policy_id"), "versions": versions, "build_stages": state["stages"], "coverage": coverage, "excluded_regimes": sorted(set(k for c in coverage.values() for k in c.get("excluded_regimes",[]))) or ([] if qualified else ["UNVALIDATED_REGIMES"]), "validation_status": "QUALIFIED" if qualified else "UNQUALIFIED", "historical_market_observations_used": False}
        if "price_tree_nodes" in arrays:
            from ..pricing_fast import PRICE_TREE_SCHEMA
            manifest["price_tree_schema"] = PRICE_TREE_SCHEMA
        manifest.update(hold_label_schema=HOLD_LABEL_SCHEMA,hold_loss_schema=HOLD_LOSS_SCHEMA,feature_contract=feature_contract())
        atomic_json(staging_dir/"manifest.json", manifest)
        (staging_dir/"manifest.sha256").write_text(file_hash(staging_dir/"manifest.json")+"\n", encoding="ascii")
        validation = {"manifest_hash": stable_hash(manifest), "qualified": qualified, "capabilities": capabilities, "online_import_boundary": online_import_audit()}
        atomic_json(staging_dir/"validation.json", validation)
        (staging_dir/"validation.sha256").write_text(file_hash(staging_dir/"validation.json")+"\n", encoding="ascii")
        from .publish import validate_and_publish
        try:
            persisted_validation=validate_and_publish(staging_dir,pack_dir,folder/"clean_pack_validation.json")
            qualified=qualified and persisted_validation.get("qualified",False)
        except RiskError as exc:
            state["last_stop"]=exc.as_dict()
            qualified=False
            pack_dir=staging_dir
        state["stages"]["B9"] = {"status": "QUALIFIED_PACK_WRITTEN" if qualified else "CHECKPOINT_PACK_WRITTEN", "production_promoted": qualified}
        completed_status="RESEARCH_COMPLETE_UNQUALIFIED" if research else "CALIBRATION_COMPLETE_UNQUALIFIED"
        report = {"status": "QUALIFIED" if qualified else "CHECKPOINTED" if state.get("last_stop", {}).get("reason") == "RESOURCE_LIMIT" else completed_status if state["stages"].get("B7", {}).get("status") == "SCORER_FROZEN_AND_POLICY_EVALUATION_TRAINED" else "INCOMPLETE", "production_qualified": qualified, "pack_path": str(pack_dir), "work_path": str(folder), "checkpoint": str(checkpoint), "stages": state["stages"], "stop": state.get("last_stop"), "actual_reference_labels": len(state["labels"]), "arrays_exported": len(arrays), "elapsed_seconds": time.perf_counter()-started}
        atomic_json(report_path, report)
        save()
        for store in stores:
            store.close()
    if state.get("adaptive_child") and remaining()>1:
        child_config={**state["adaptive_child"],"budget_seconds":remaining()}
        child_report=run_pipeline(child_config,{**model_config,"case_count":child_config["case_count"]})
        atomic_json(folder/"adaptive_resume_report.json",child_report)
        return {**child_report,"adaptive_parent_report":str(report_path)}
    return report
