"""Resumable numeric-only training checkpoints; no torch/pickle deserialization."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from ...config import writable_path
from ..contracts import RiskError,stable_hash
from ..pack_loader import file_hash
from .hit_oracle import atomic_json
from .training_store import require_disk_space


def numeric_identity(value):
    if hasattr(value,"identity"):return value.identity
    array=np.asarray(value)
    digest=hashlib.sha256()
    for offset in range(0,len(array),1024):digest.update(np.ascontiguousarray(array[offset:offset+1024]).tobytes())
    return {"sha256":digest.hexdigest(),"shape":list(array.shape),"dtype":array.dtype.str}


def training_identity(features,labels,train_indices,calibration_indices,tokens,configuration):
    from qqq_sim._build_info import SOURCE_HASHES
    sources={}
    for name in ("train.py","train_checkpoint.py","universe_train.py"):
        path=Path(__file__).parent/name
        key="src/qqq_sim/risk/offline/"+name
        if path.is_file():sources[key]=file_hash(path)
        elif key in SOURCE_HASHES:sources[key]=SOURCE_HASHES[key]
        else:raise RiskError("IMPLEMENTATION_IDENTITY_UNAVAILABLE",source=key)
    return stable_hash({"features":numeric_identity(features),"labels":numeric_identity(labels),"train_indices":numeric_identity(train_indices),"calibration_indices":numeric_identity(calibration_indices),"universe_tokens":None if tokens is None else numeric_identity(tokens),"configuration":configuration,"sources":sources})


class TrainingCheckpoint:
    def __init__(self,folder,identity):
        self.folder=writable_path(folder)
        self.folder.mkdir(parents=True,exist_ok=True)
        self.identity=identity
        self.manifest=self.folder/"training_checkpoint.json"

    def load(self):
        if not self.manifest.exists():return None
        metadata=json.loads(self.manifest.read_text(encoding="utf-8"))
        if metadata.get("identity")!=self.identity:raise RiskError("TRAINING_CHECKPOINT_MISMATCH")
        artifact=self.folder/metadata["artifact"]
        if not artifact.is_file() or artifact.stat().st_size!=metadata["bytes"] or file_hash(artifact)!=metadata["sha256"]:
            raise RiskError("TRAINING_CHECKPOINT_MISMATCH",artifact=str(artifact))
        with np.load(artifact,allow_pickle=False) as loaded:
            arrays={key:loaded[key].copy() for key in loaded.files}
        if any(array.dtype.hasobject or not np.all(np.isfinite(array)) for array in arrays.values()):
            raise RiskError("TRAINING_CHECKPOINT_INVALID_ARRAY")
        return metadata,arrays

    def save(self,metadata,arrays):
        required=sum(np.asarray(value).nbytes for value in arrays.values())+65536
        require_disk_space(self.folder/"numeric.pending",required)
        temporary=self.folder/"numeric.pending"
        with temporary.open("wb") as stream:
            np.savez(stream,**{key:np.asarray(value) for key,value in arrays.items()})
            stream.flush();os.fsync(stream.fileno())
        digest=file_hash(temporary)
        artifact=self.folder/("numeric_"+digest+".npz")
        if artifact.exists():temporary.unlink()
        else:os.replace(temporary,artifact)
        old=None
        if self.manifest.exists():old=json.loads(self.manifest.read_text(encoding="utf-8")).get("artifact")
        atomic_json(self.manifest,{**metadata,"identity":self.identity,"artifact":artifact.name,"sha256":digest,"bytes":artifact.stat().st_size,"format":"SAFE_NUMERIC_NPZ_JSON_V1"})
        if old and old!=artifact.name:
            previous=self.folder/old
            if previous.resolve().parent==self.folder.resolve() and previous.name.startswith("numeric_") and previous.suffix==".npz" and previous.exists():previous.unlink()
