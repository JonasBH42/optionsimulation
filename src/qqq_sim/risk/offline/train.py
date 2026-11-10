"""CPU float64 AdamW training; only exported numeric arrays are used online."""
from __future__ import annotations
import copy
import time
import numpy as np
from ..contracts import RiskError,canonical


HOLD_WITHIN_BIN_SLOTS = 32
HOLD_LABEL_BASE_SIZE = 1156
HOLD_LABEL_SIZE = HOLD_LABEL_BASE_SIZE + HOLD_WITHIN_BIN_SLOTS * 5
HOLD_LABEL_SCHEMA = {"version":"QQQ_HOLD_LABEL_V2", "size":HOLD_LABEL_SIZE,
    "first_hit_counts":[0,576],"survivor_count":576,"terminal_companions":[577,580],"bin_masks":[580,1156],
    "within_bin_offset":HOLD_LABEL_BASE_SIZE,"within_bin_slots":HOLD_WITHIN_BIN_SLOTS,
    "within_bin_fields":["mask","bin_index","observation_fraction","cumulative_first_hit_count","deadline_utc_us"],
    "sampling":"DESIGN_SEED_CHILD_1_10_JOB_AND_FROZEN_GRID_EXISTING_OBSERVATIONS_ONLY"}
HOLD_LOSS_SCHEMA = {"version":"QQQ_HOLD_LOSS_V2","multinomial":"SUM_COUNT_NLL_DIVIDED_BY_TOTAL_INDEPENDENT_PATHS",
    "within_bin":"PATH_WEIGHTED_MEAN_SQUARED_ERROR_OVER_VALID_SAMPLED_DEADLINES",
    "companions":"PATH_WEIGHTED_MSE_WITH_SURVIVOR_WEIGHTED_CONDITIONAL_TERMINAL",
    "term_coefficients":{"multinomial":1.,"within_bin":1.,"companions":1.}}


def hold_loss_components(output, expected, *, allow_legacy=False):
    """Per-case terms; numerical reference counts are never smoothed or invented."""
    import torch
    if output.ndim != 2 or output.shape[1] != 579 or expected.ndim != 2 or len(output) != len(expected):
        raise RiskError("HOLD_LABEL_SCHEMA_MISMATCH")
    legacy = expected.shape[1] in (580,1156)
    if expected.shape[1] != HOLD_LABEL_SIZE and not (allow_legacy and legacy):
        raise RiskError("HOLD_LABEL_SCHEMA_MISMATCH")
    if not torch.all(torch.isfinite(expected)):
        raise RiskError("HOLD_LABEL_NONFINITE")
    counts, survivors = expected[:,:576], expected[:,576]
    total = counts.sum(dim=1)+survivors
    mask = expected[:,580:1156] if expected.shape[1] != 580 else torch.ones_like(counts)
    if torch.any(total<=0) or torch.any(expected[:,:577]<0) or torch.any(expected[:,:577] != torch.round(expected[:,:577])):
        raise RiskError("INVALID_LABEL_SAMPLE_COUNT")
    if torch.any((mask!=0)&(mask!=1)) or torch.any(counts*(1-mask)!=0):
        raise RiskError("HOLD_LABEL_MASK_MISMATCH")
    companion_targets = expected[:,577:580]
    if torch.any((companion_targets<0)|(companion_targets>1)):
        raise RiskError("HOLD_LABEL_PROBABILITY_INVALID")
    hazards = torch.nn.functional.softplus(output[:,:576])*mask
    prefix = torch.cumsum(hazards,dim=1)-hazards
    log_mass = -prefix+torch.nn.functional.logsigmoid(output[:,:576])
    nll = -(counts*log_mass).sum(dim=1)+survivors*hazards.sum(dim=1)
    companion_errors = (torch.sigmoid(output[:,576:])-companion_targets)**2
    # An empty survivor set supplies no conditional-terminal observation.
    companion = (companion_errors[:,0]+companion_errors[:,1]+companion_errors[:,2]*survivors/total)/3
    within = torch.zeros_like(total)
    if not legacy:
        samples = expected[:,HOLD_LABEL_BASE_SIZE:].reshape(-1,HOLD_WITHIN_BIN_SLOTS,5)
        valid, bin_index, fractions, hit_counts, instants = samples.unbind(dim=2)
        selected = valid.bool()
        if torch.any((valid!=0)&(valid!=1)) or torch.any(samples[~selected]!=0):
            raise RiskError("HOLD_WITHIN_BIN_LABEL_INVALID")
        if (torch.any(bin_index[selected]!=torch.round(bin_index[selected])) or torch.any((bin_index[selected]<0)|(bin_index[selected]>=576))
                or torch.any((fractions[selected]<=0)|(fractions[selected]>=1)) or torch.any(hit_counts[selected]!=torch.round(hit_counts[selected]))
                or torch.any(hit_counts[selected]<0) or torch.any(hit_counts>total[:,None])
                or torch.any(instants[selected]!=torch.round(instants[selected]))):
            raise RiskError("HOLD_WITHIN_BIN_LABEL_INVALID")
        indices = bin_index.to(torch.int64)
        if torch.any(torch.gather(mask,1,indices)*valid != valid):
            raise RiskError("HOLD_WITHIN_BIN_LABEL_INVALID")
        prior_counts=torch.gather(torch.cumsum(counts,dim=1)-counts,1,indices)
        through_counts=prior_counts+torch.gather(counts,1,indices)
        if torch.any((hit_counts<prior_counts)&selected) or torch.any((hit_counts>through_counts)&selected):
            raise RiskError("HOLD_WITHIN_BIN_LABEL_INVALID")
        cumulative = torch.gather(prefix,1,indices)+torch.gather(hazards,1,indices)*fractions
        predicted = -torch.expm1(-cumulative)
        squared = (predicted-hit_counts/total[:,None])**2*valid
        within = squared.sum(dim=1)/valid.sum(dim=1).clamp_min(1)
    return {"multinomial_nll":nll,"within_bin_mse":within,"companion_mse":companion,"independent_paths":total}


def hold_training_loss(output, expected, *, allow_legacy=False):
    terms = hold_loss_components(output,expected,allow_legacy=allow_legacy)
    paths = terms["independent_paths"]
    return (terms["multinomial_nll"]+paths*(terms["within_bin_mse"]+terms["companion_mse"])).sum()/paths.sum()


def within_bin_validation(output, labels, numerical_uncertainty=None, *, tolerance=.01, deterministic=None):
    """Held-out cases count once; correlated deadlines never inflate sample size."""
    output,labels=np.asarray(output,dtype=np.float64),np.asarray(labels,dtype=np.float64)
    if output.shape!=(len(labels),579) or labels.ndim!=2 or labels.shape[1]!=HOLD_LABEL_SIZE:
        raise RiskError("HOLD_LABEL_SCHEMA_MISMATCH")
    if not np.all(np.isfinite(output)) or not np.all(np.isfinite(labels)):
        raise RiskError("HOLD_LABEL_NONFINITE")
    n=labels[:,:577].sum(axis=1)
    if np.any(n<=0):raise RiskError("INVALID_LABEL_SAMPLE_COUNT")
    samples=labels[:,HOLD_LABEL_BASE_SIZE:].reshape(-1,HOLD_WITHIN_BIN_SLOTS,5)
    if np.any((samples[:,:,0]!=0)&(samples[:,:,0]!=1)) or np.any(labels[:,:577]<0) or np.any(labels[:,:577]!=np.rint(labels[:,:577])):
        raise RiskError("HOLD_WITHIN_BIN_LABEL_INVALID")
    valid=samples[:,:,0].astype(bool)
    indices=samples[:,:,1].astype(np.int64)
    fractions=samples[:,:,2]
    if (np.any((indices<0)|(indices>=576)) or np.any(samples[:,:,1]!=indices) or np.any((fractions[valid]<=0)|(fractions[valid]>=1))
            or np.any(samples[~valid]!=0) or np.any((samples[:,:,3]<0)|(samples[:,:,3]>n[:,None]))
            or np.any(samples[:,:,3]!=np.rint(samples[:,:,3]))):
        raise RiskError("HOLD_WITHIN_BIN_LABEL_INVALID")
    hazards=np.logaddexp(0,output[:,:576])*labels[:,580:1156]
    masks=labels[:,580:1156]
    prior_counts=np.take_along_axis(np.cumsum(labels[:,:576],axis=1)-labels[:,:576],indices,axis=1)
    through_counts=prior_counts+np.take_along_axis(labels[:,:576],indices,axis=1)
    if (np.any((masks!=0)&(masks!=1)) or np.any(labels[:,:576]*(1-masks)!=0)
            or np.any(np.take_along_axis(masks,indices,axis=1)[valid]!=1)
            or np.any((samples[:,:,3]<prior_counts)&valid) or np.any((samples[:,:,3]>through_counts)&valid)):
        raise RiskError("HOLD_WITHIN_BIN_LABEL_INVALID")
    prefix=np.cumsum(hazards,axis=1)-hazards
    cumulative=np.take_along_axis(prefix,indices,axis=1)+np.take_along_axis(hazards,indices,axis=1)*fractions
    predicted=-np.expm1(-cumulative)
    reference=samples[:,:,3]/n[:,None]
    # Wilson intervals with a Bonferroni allocation across each case's sampled
    # deadlines retain case-level coverage despite correlated first-hit labels.
    from scipy.special import ndtri
    z=ndtri(1-.05/(2*np.maximum(1,valid.sum(axis=1))))[:,None]
    denominator=1+z*z/n[:,None]
    center=(reference+z*z/(2*n[:,None]))/denominator
    radius=z*np.sqrt(reference*(1-reference)/n[:,None]+z*z/(4*n[:,None]**2))/denominator
    sampling=np.maximum(center+radius-reference,reference-(center-radius))
    if deterministic is not None:
        exact=np.asarray(deterministic,dtype=bool)
        if exact.shape!=(len(labels),):raise RiskError("HOLD_LABEL_SCHEMA_MISMATCH")
        sampling[exact]=0.
    numeric=np.zeros(len(labels)) if numerical_uncertainty is None else np.asarray(numerical_uncertainty,dtype=np.float64)
    if numeric.shape!=(len(labels),) or not np.all(np.isfinite(numeric)) or np.any(numeric<0):
        raise RiskError("HOLD_LABEL_UNCERTAINTY_INVALID")
    conservative=np.abs(predicted-reference)+sampling+numeric[:,None]
    errors=np.max(np.where(valid,conservative,0.),axis=1)
    applicable=valid.any(axis=1)
    measured=errors[applicable]
    count=int(applicable.sum())
    passed=float(np.mean(measured<=tolerance)) if count else 0.
    return {"qualified":bool(count>=200 and passed>=.99),"independent_cases":count,
        "total_cases":len(labels),"singleton_only_cases":int((~applicable).sum()),"sampled_deadlines":int(valid.sum()),
        "pass_fraction":passed,"maximum_absolute_error_plus_reference_uncertainty":float(measured.max()) if count else None,
        "tolerance":tolerance,"case_errors":errors.tolist(),"applicable_cases":applicable.tolist(),
        "sampling_uncertainty":"PER_CASE_BONFERRONI_WILSON_95_PERCENT_UNLESS_EXACT_DETERMINISTIC",
        "reason":None if count>=200 and passed>=.99 else "WITHIN_BIN_INDEPENDENT_CASES_INSUFFICIENT" if count<200 else "WITHIN_BIN_CDF_ERROR_GATE_FAILED"}


def train_network(features, labels, train_indices, calibration_indices, *, output_size=579, kind="hold", epochs=300, seed=20261109, mask_indices=(),universe_tokens=None,universe_arrays=None,deadline=None,training_directory=None,export_training_encodings=True,allow_legacy_hold_labels=False):
    import torch
    from torch import nn
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if not len(train_indices) or not len(calibration_indices):
        raise RiskError("TRAINING_SPLIT_INSUFFICIENT")
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if kind == "hold" and (y.ndim!=2 or (y.shape[1]!=HOLD_LABEL_SIZE and not (allow_legacy_hold_labels and y.shape[1] in (580,1156)))):
        raise RiskError("HOLD_LABEL_SCHEMA_MISMATCH")
    train_indices=np.asarray(train_indices,dtype=np.int64)
    calibration_indices=np.asarray(calibration_indices,dtype=np.int64)
    if universe_tokens is not None and not hasattr(universe_tokens,"shape"):
        universe_tokens=np.asarray(universe_tokens,dtype=np.float64)
    checkpoint=None
    restored=None
    if training_directory is not None:
        from .train_checkpoint import TrainingCheckpoint,training_identity,numeric_identity
        identity=training_identity(features,labels,train_indices,calibration_indices,universe_tokens,{"output_size":output_size,"kind":kind,"epochs":min(300,epochs),"seed":seed,"masks":list(mask_indices),"encoder_initialization":None if universe_arrays is None else {name:numeric_identity(value) for name,value in universe_arrays.items()},"torch":torch.__version__,"numpy":np.__version__,"architecture":[256,256,128],"batch":256,"adamw":[.001,.9,.999,1e-8,1e-5],"patience":25,"improvement":1e-5})
        checkpoint=TrainingCheckpoint(training_directory,identity)
        restored=checkpoint.load()
    mean=np.zeros(x.shape[1]);squares=np.zeros_like(mean);seen=0
    scaler_complete=False
    if restored is not None:
        metadata,saved_arrays=restored
        mean=saved_arrays["mean"].copy()
        if metadata["phase"]=="SCALER":
            squares=saved_arrays["squares"].copy();seen=int(metadata["seen"])
        else:
            std=saved_arrays["std"].copy();scaler_complete=True
    for offset in range(seen,len(train_indices) if not scaler_complete else seen,1024):
        if deadline is not None and time.monotonic()>=deadline:
            if checkpoint:checkpoint.save({"phase":"SCALER","seen":seen},{"mean":mean,"squares":squares})
            raise RiskError("RESOURCE_LIMIT","Training scaler checkpoint saved",stage=kind,completed_rows=seen)
        block=x[train_indices[offset:offset+1024]]
        block_mean=block.mean(axis=0)
        change=block_mean-mean
        count=len(block)
        squares+=np.sum((block-block_mean)**2,axis=0)+change**2*seen*count/(seen+count)
        mean+=change*count/(seen+count)
        seen+=count
        if checkpoint and seen%(1024*128)==0:checkpoint.save({"phase":"SCALER","seen":seen},{"mean":mean,"squares":squares})
    if not scaler_complete:
        std=np.maximum(np.sqrt(squares/seen),1e-8)
        if len(mask_indices):
            mean[list(mask_indices)], std[list(mask_indices)] = 0, 1
        if checkpoint:checkpoint.save({"phase":"INITIALIZED"},{"mean":mean,"std":std})
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    layers = []
    sizes = (x.shape[1], 256, 256, 128, output_size)
    for index, (left, right) in enumerate(zip(sizes, sizes[1:])):
        layer = nn.Linear(left, right, dtype=torch.float64)
        with torch.no_grad():
            layer.weight.copy_(torch.randn(right, left, generator=generator, dtype=torch.float64)*(2/left)**.5)
            layer.bias.zero_()
        layers.append(layer)
        if index < 3:
            layers.append(nn.SiLU())
    model = nn.Sequential(*layers)
    encoder=None
    if universe_tokens is not None:
        from .universe_train import make_trainable_encoder
        encoder=make_trainable_encoder(universe_arrays)
        if not hasattr(universe_tokens,"shape"):
            universe_tokens=np.asarray(universe_tokens,dtype=np.float64)
        if universe_tokens.shape!=(len(x),512,16):
            raise RiskError("UNIVERSE_TRAINING_SCHEMA_MISMATCH")
    parameters=list(model.named_parameters())+(list(encoder.named_parameters()) if encoder is not None else [])
    weights = [p for name, p in parameters if name.endswith("weight")]
    biases = [p for name, p in parameters if name.endswith("bias")]
    optimizer = torch.optim.AdamW([{"params": weights, "weight_decay": 1e-5}, {"params": biases, "weight_decay": 0}], lr=.001, betas=(.9, .999), eps=1e-8)
    mean_tensor=torch.tensor(mean,dtype=torch.float64)
    std_tensor=torch.tensor(std,dtype=torch.float64)
    def forward(indices):
        prepared=torch.tensor((x[indices]-mean)/std,dtype=torch.float64)
        if encoder is not None:
            encoded=encoder(torch.tensor(np.asarray(universe_tokens[indices]),dtype=torch.float64))
            prepared=torch.cat((prepared[:,:-154],(encoded-mean_tensor[-154:])/std_tensor[-154:]),dim=1)
        return model(prepared)

    def loss_fn(output, expected):
        if kind != "hold":
            return ((torch.sigmoid(output)-expected)**2).mean()
        return hold_training_loss(output,expected,allow_legacy=allow_legacy_hold_labels)

    best, best_loss, stale = None, float("inf"), 0
    history = []
    shuffle = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(3, 0, 0))))
    train_indices = np.asarray(train_indices)
    start_epoch=0
    if restored is not None and restored[0]["phase"]=="EPOCH":
        metadata,saved_arrays=restored
        def tensor_state(prefix,names):
            return {name:torch.from_numpy(saved_arrays[f"{prefix}_{index}"].copy()) for index,name in enumerate(names)}
        model.load_state_dict(tensor_state("model",metadata["model_names"]),strict=True)
        if encoder is not None:encoder.load_state_dict(tensor_state("encoder",metadata["encoder_names"]),strict=True)
        best=(tensor_state("best_model",metadata["model_names"]),tensor_state("best_encoder",metadata["encoder_names"]) if encoder is not None else None)
        optimizer_state={}
        for parameter_id,entry in metadata["optimizer_states"].items():
            optimizer_state[int(parameter_id)]={"step":torch.tensor(entry["step"],dtype=torch.float32),**{name:torch.from_numpy(saved_arrays[array_key].copy()) for name,array_key in entry["moments"].items()}}
        optimizer.load_state_dict({"state":optimizer_state,"param_groups":metadata["optimizer_groups"]})
        best_loss=float(metadata["best_loss"]);stale=int(metadata["stale"]);history=metadata["history"]
        shuffle.bit_generator.state=metadata["shuffle_state"]
        start_epoch=int(metadata["next_epoch"])

    def save_epoch(next_epoch):
        if checkpoint is None:return
        values={"mean":mean,"std":std}
        model_names=list(model.state_dict());encoder_names=list(encoder.state_dict()) if encoder is not None else []
        for prefix,source in (("model",model.state_dict()),("encoder",encoder.state_dict() if encoder is not None else {}),("best_model",best[0]),("best_encoder",best[1] or {})):
            values.update({f"{prefix}_{index}":tensor.detach().cpu().numpy().copy() for index,tensor in enumerate(source.values())})
        optimizer_saved=optimizer.state_dict()
        optimizer_states={}
        for parameter_id,entry in optimizer_saved["state"].items():
            moments={}
            for name,tensor in entry.items():
                if name=="step":continue
                key=f"optimizer_{parameter_id}_{name}"
                values[key]=tensor.detach().cpu().numpy().copy();moments[name]=key
            optimizer_states[str(parameter_id)]={"step":float(entry["step"].item()),"moments":moments}
        checkpoint.save({"phase":"EPOCH","next_epoch":next_epoch,"best_loss":best_loss,"stale":stale,"history":history,"shuffle_state":canonical(shuffle.bit_generator.state),"model_names":model_names,"encoder_names":encoder_names,"optimizer_states":optimizer_states,"optimizer_groups":canonical(optimizer_saved["param_groups"])},values)

    for epoch in range(start_epoch,min(300, epochs) if stale<25 else start_epoch):
        if deadline is not None and time.monotonic()>=deadline:
            raise RiskError("RESOURCE_LIMIT","Training stage budget exhausted",stage=kind,completed_epochs=epoch)
        order = train_indices[shuffle.permutation(len(train_indices))]
        model.train()
        for offset in range(0, len(order), 256):
            if deadline is not None and time.monotonic()>=deadline:
                raise RiskError("RESOURCE_LIMIT","Training stage budget exhausted",stage=kind,completed_epochs=epoch)
            batch = order[offset:offset+256]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(forward(batch), torch.tensor(y[batch],dtype=torch.float64))
            if not torch.isfinite(loss):
                raise RiskError("TRAINING_NONFINITE_LOSS")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _,p in parameters], 5.)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            total_loss=0.
            total_weight=0.
            for offset in range(0,len(calibration_indices),256):
                if deadline is not None and time.monotonic()>=deadline:
                    raise RiskError("RESOURCE_LIMIT","Calibration stage budget exhausted",stage=kind,completed_epochs=epoch)
                batch=calibration_indices[offset:offset+256]
                expected=torch.tensor(y[batch],dtype=torch.float64)
                weight=float(expected[:,:577].sum()) if kind=="hold" else len(batch)
                total_loss+=float(loss_fn(forward(batch),expected))*weight
                total_weight+=weight
            cal_loss=total_loss/total_weight
        history.append({"epoch": epoch, "calibration_loss": cal_loss})
        if best_loss-cal_loss >= 1e-5:
            best, best_loss, stale = (copy.deepcopy(model.state_dict()),copy.deepcopy(encoder.state_dict()) if encoder is not None else None), cal_loss, 0
        else:
            stale += 1
        save_epoch(epoch+1)
        if stale >= 25:
            break
    if best is None:
        raise RiskError("TRAINING_NO_VALID_CHECKPOINT")
    model.load_state_dict(best[0])
    if encoder is not None:
        encoder.load_state_dict(best[1])
    prefix = "hold" if kind == "hold" else kind
    if not export_training_encodings:
        encodings=None
        row_store=None
    elif training_directory is None:
        encodings=np.empty((len(train_indices),x.shape[1]),dtype=np.float64)
        row_store=None
    else:
        from pathlib import Path
        from .training_store import NumericRows
        row_store=NumericRows(Path(training_directory)/(prefix+"_encodings.bin"),x.shape[1])
        encodings=None
    with torch.no_grad():
        for offset in range(0,len(train_indices) if export_training_encodings else 0,256):
            if deadline is not None and time.monotonic()>=deadline:
                raise RiskError("RESOURCE_LIMIT","Training-encoding export budget exhausted",stage=kind)
            batch=train_indices[offset:offset+256]
            encoded=x[batch].copy()
            if encoder is not None:
                encoded[:,-154:]=encoder(torch.tensor(np.asarray(universe_tokens[batch]),dtype=torch.float64)).numpy()
            normalized=(encoded-mean)/std
            if row_store is None:encodings[offset:offset+len(batch)]=normalized
            else:
                for row in normalized:row_store.append(row)
    if row_store is not None:
        encodings=row_store.array()
        row_store.close()
    arrays = {prefix+"_mean": mean, prefix+"_std": std}
    if encodings is not None:arrays[prefix+"_training_encodings"]=encodings
    if encoder is not None:
        arrays.update({prefix+"_"+name:value for name,value in encoder.export_arrays().items()})
    for index, layer in enumerate(layer for layer in model if isinstance(layer, nn.Linear)):
        arrays[f"{prefix}_w{index}"] = layer.weight.detach().numpy().T.copy()
        arrays[f"{prefix}_b{index}"] = layer.bias.detach().numpy().copy()
    return {"arrays": arrays, "history": history, "best_calibration_loss": best_loss, "torch_version": torch.__version__, "dtype": "float64", "selection_split": "calibration", "untouched_validation_used_for_selection": False,"resumed_completed_epochs":start_epoch,"checkpoint_format":"SAFE_NUMERIC_NPZ_JSON_V1" if checkpoint else None,
        "hold_label_schema":HOLD_LABEL_SCHEMA if kind=="hold" and y.shape[1]==HOLD_LABEL_SIZE else None,
        "hold_loss_schema":HOLD_LOSS_SCHEMA if kind=="hold" and y.shape[1]==HOLD_LABEL_SIZE else None,
        "legacy_hold_labels_unqualified":bool(kind=="hold" and y.shape[1]!=HOLD_LABEL_SIZE)}


def calibration_report(predictions, reference, reference_uncertainty=None, tolerance=.01):
    error = np.abs(np.asarray(predictions)-np.asarray(reference))
    uncertainty = np.zeros_like(error) if reference_uncertainty is None else np.asarray(reference_uncertainty)
    if not len(error) or not np.all(np.isfinite(error)):
        raise RiskError("CALIBRATION_DATA_INVALID")
    conservative = error+uncertainty
    return {"count": len(error), "p50_absolute_error": float(np.quantile(error, .5)), "p95_absolute_error": float(np.quantile(error, .95)), "p99_absolute_error": float(np.quantile(error, .99)), "maximum_absolute_error": float(error.max()), "reference_uncertainty": float(np.max(uncertainty)), "pass_fraction": float(np.mean(conservative <= tolerance)), "tolerance": tolerance, "qualified": len(error) >= 200 and float(np.mean(conservative <= tolerance)) >= .99}
