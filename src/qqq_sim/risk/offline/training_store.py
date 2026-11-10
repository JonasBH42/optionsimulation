"""Append-only numeric training rows and compact repeated state/token storage."""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import numpy as np
from ...config import writable_path
from ..contracts import RiskError,canonical,ReferenceJob,RiskLeg,stable_hash
from .control_oracle import ManagedState


def require_disk_space(path,additional_bytes=0):
    usage=shutil.disk_usage(Path(path).parent)
    reserve=max(1<<30,int(usage.total*.05))
    if usage.free-additional_bytes<reserve:
        raise RiskError("RESOURCE_LIMIT","Training disk reserve would be crossed",resource="disk",available_bytes=usage.free,reserved_bytes=reserve,requested_bytes=additional_bytes)


class NumericRows:
    def __init__(self,path,width,dtype=np.float64,*,scalar=False):
        self.path=writable_path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        require_disk_space(self.path)
        self.stream=self.path.open("wb")
        self.width=width
        self.dtype=np.dtype(dtype)
        self.scalar=scalar
        self.count=0
        self._map=None
        self.digest=hashlib.sha256()

    def append(self,value):
        row=np.asarray(value,dtype=self.dtype).reshape(-1)
        if row.shape!=(self.width,):raise RiskError("TRAINING_ROW_SCHEMA_MISMATCH")
        if self.count%256==0:require_disk_space(self.path,256*row.nbytes)
        data=row.tobytes()
        self.stream.write(data)
        self.digest.update(data)
        self.count+=1
        self._map=None

    def array(self):
        self.stream.flush()
        if not self.count:return np.empty((0,self.width),dtype=self.dtype)
        if self._map is None:self._map=np.memmap(self.path,mode="r",dtype=self.dtype,shape=(self.count,self.width))
        return self._map

    def __len__(self):return self.count
    @property
    def identity(self):return {"sha256":self.digest.hexdigest(),"shape":[self.count,self.width],"dtype":self.dtype.str}
    def __array__(self,dtype=None,copy=None):
        result=self.array()
        return result.astype(dtype,copy=True) if dtype is not None and np.dtype(dtype)!=self.dtype else result.copy() if copy else result
    def __getitem__(self,index):
        value=self.array()[index]
        return value.item() if self.scalar and np.ndim(index)==0 and isinstance(index,(int,np.integer)) else value
    def close(self):self.stream.close()


class UniverseRows:
    def __init__(self,folder):
        self.pool=NumericRows(Path(folder)/"pool.bin",512*16)
        self.indices=NumericRows(Path(folder)/"indices.bin",1,np.int64,scalar=True)
        self.identities={}
        self.last_object=None
        self.last_index=None

    def append(self,value):
        if value is self.last_object:
            index=self.last_index
        else:
            value=np.asarray(value,dtype=np.float64)
            key=hashlib.sha256(value.tobytes()).hexdigest()
            if key not in self.identities:
                self.identities[key]=len(self.pool)
                self.pool.append(value)
            index=self.identities[key]
            self.last_object,self.last_index=value,index
        self.indices.append(index)

    @property
    def shape(self):return (len(self.indices),512,16)
    @property
    def identity(self):return {"pool":self.pool.identity,"indices":self.indices.identity,"shape":self.shape}
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        mapping=np.asarray(self.indices.array()[index]).reshape(-1)
        values=np.asarray(self.pool.array()[mapping]).reshape(-1,512,16)
        return values[0] if isinstance(index,(int,np.integer)) else values
    def close(self):self.pool.close();self.indices.close()


class SplitRows:
    names=("train","calibration","validation")
    def __init__(self,path):self.rows=NumericRows(path,1,np.uint8,scalar=True)
    def append(self,value):self.rows.append(self.names.index(value))
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.names[int(self.rows[index])]
    def close(self):self.rows.close()


class ControlCases:
    def __init__(self,folder):
        self.folder=writable_path(folder)
        self.folder.mkdir(parents=True,exist_ok=True)
        self.records=(self.folder/"pairs.jsonl").open("w+b")
        self.offsets=NumericRows(self.folder/"offsets.bin",1,np.int64,scalar=True)
        self.last_state=None
        self.last_key=None
        self.cache=OrderedDict()

    def append(self,pair):
        state,action=pair
        if state is not self.last_state:
            value=canonical(state)
            key=stable_hash(value)
            path=self.folder/(key+".json")
            if not path.exists():
                encoded=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
                require_disk_space(path,len(encoded))
                path.write_bytes(encoded)
            self.last_state,self.last_key=state,key
        encoded=(json.dumps({"state":self.last_key,"action":canonical(action)},separators=(",",":"),allow_nan=False)+"\n").encode()
        require_disk_space(self.folder/"pairs.jsonl",len(encoded))
        self.offsets.append(self.records.seek(0,2))
        self.records.write(encoded)

    def __len__(self):return len(self.offsets)
    def __getitem__(self,index):
        if index<0:index+=len(self)
        if index>=len(self) or index<0:raise IndexError(index)
        self.records.flush()
        self.records.seek(int(self.offsets[index]))
        record=json.loads(self.records.readline())
        key=record["state"]
        if key not in self.cache:
            value=json.loads((self.folder/(key+".json")).read_text(encoding="utf-8"))
            if stable_hash(value)!=key:raise RiskError("CHECKPOINT_MISMATCH",artifact=key)
            self.cache[key]=ManagedState(ReferenceJob.from_dict(value["job"]),value["campaign"],tuple(RiskLeg.from_dict(x) for x in value["universe"]),value["success"],value["failed"],tuple(value["events"]),value["account_context"])
            if len(self.cache)>8:self.cache.popitem(last=False)
        return self.cache[key],record["action"]
    def close(self):self.records.close();self.offsets.close()


class ControlLabelCache:
    def __init__(self,path):
        self.path=writable_path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        require_disk_space(self.path)
        self.connection=sqlite3.connect(self.path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS labels (identity TEXT PRIMARY KEY, body TEXT NOT NULL, sha256 TEXT NOT NULL)")

    def get(self,key):
        row=self.connection.execute("SELECT body,sha256 FROM labels WHERE identity=?",(key,)).fetchone()
        if row is None:return None
        if hashlib.sha256(row[0].encode()).hexdigest()!=row[1]:raise RiskError("CHECKPOINT_MISMATCH",artifact="control_label:"+key)
        return json.loads(row[0])

    def put(self,key,value):
        body=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
        require_disk_space(self.path,len(body)+16384)
        self.connection.execute("INSERT OR REPLACE INTO labels VALUES (?,?,?)",(key,body,hashlib.sha256(body.encode()).hexdigest()))
        self.connection.commit()

    def close(self):self.connection.close()
