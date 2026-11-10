"""Bounded immutable interval records, with hashed 8 MiB disk blocks when needed."""
from __future__ import annotations
from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import bisect
import json
from pathlib import Path
import sqlite3
import sys
from uuid import uuid4
import numpy as np

from ...config import writable_path
from ..contracts import RiskError, canonical
from ..payoff import Interval

BLOCK_BYTES=8*1024*1024
DTYPE=np.dtype([("lower","<f8"),("upper","<f8"),("lower_closed","u1"),("upper_closed","u1")])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def region_digest(mapping):
    h=hashlib.sha256()
    for key,value in mapping.ordered_items() if hasattr(mapping,"ordered_items") else sorted(mapping.items()):
        h.update(json.dumps([key,canonical(value)],sort_keys=True,separators=(",",":"),allow_nan=False).encode())
        h.update(b"\n")
    return h.hexdigest()


class RegionStore(Mapping):
    """Small sets stay in memory; large sets spill without discarding intervals.

    The SQLite index and NumPy blocks contain data only. Frozen handles can be
    passed to spawned workers, which open their own read-only connections/maps.
    """
    def __init__(self,identity,*,block_bytes=BLOCK_BYTES,root="runtime/builds/regions",max_record_bytes=64*1024*1024):
        self.identity=canonical(identity)
        self.block_bytes=min(BLOCK_BYTES,max(DTYPE.itemsize,int(block_bytes)))
        self.root=str(writable_path(root))
        self.folder=None
        self._memory={}
        self._memory_bytes=0
        self.max_record_bytes=max(256,int(max_record_bytes))
        self._record_interval_limit=max(1,self.max_record_bytes//256)
        self._buffer=np.empty(0,dtype=DTYPE)
        self._buffer_count=0
        self._buffer_capacity=max(1,(self.block_bytes-512)//DTYPE.itemsize)
        self._pending=[]
        self._blocks=[]
        self._db=None
        self._cache=OrderedDict()
        self._frozen=False
        self._count=0
        self._digest=None
        self._written_intervals=0

    def __setitem__(self,key,value):
        if self._frozen:
            raise RiskError("IMMUTABLE_REGION_STORE")
        key=(int(key[0]),float(key[1]))
        if hasattr(value,'__len__') and len(value)>self._record_interval_limit:
            raise RiskError('RESOURCE_LIMIT','One target-region record exceeds its decoded working-set budget',
                            intervals=len(value),max_record_bytes=self.max_record_bytes)
        if not isinstance(value,tuple):
            from itertools import islice
            value=tuple(islice(iter(value),self._record_interval_limit+1))
        if len(value)>self._record_interval_limit:
            raise RiskError('RESOURCE_LIMIT','One target-region record exceeds its decoded working-set budget',
                            intervals=len(value),max_record_bytes=self.max_record_bytes)
        if self.folder is None:
            if key in self._memory:
                raise RiskError("DUPLICATE_REGION_RECORD")
            self._memory[key]=value
            self._memory_bytes+=sys.getsizeof(key)+sys.getsizeof(value)+sum(sys.getsizeof(x)+sys.getsizeof(x.__dict__) for x in value)+64
            self._count+=1
            if self._memory_bytes+sys.getsizeof(self._memory)>self.block_bytes:
                self._spill()
        else:
            self._append(key,value)
            self._count+=1

    def _spill(self):
        folder=writable_path(Path(self.root)/uuid4().hex)
        folder.mkdir(parents=True)
        self.folder=str(folder)
        self._buffer=np.empty(self._buffer_capacity,dtype=DTYPE)
        self._db=sqlite3.connect(folder/"index.sqlite")
        self._db.execute("PRAGMA journal_mode=DELETE")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("CREATE TABLE regions(t INTEGER NOT NULL,h REAL NOT NULL,block INTEGER NOT NULL,start INTEGER NOT NULL,n INTEGER NOT NULL,PRIMARY KEY(t,h)) WITHOUT ROWID")
        memory,self._memory=self._memory,{}
        for key,value in memory.items():
            self._append(key,value)
        self._memory_bytes=0

    def _append(self,key,value):
        # A single large region is split across adjacent blocks; index rows point
        # to logical interval offsets, so block limits also hold for that case.
        start=self._written_intervals+self._buffer_count
        self._pending.append((key[0],key[1],0,start,len(value)))
        for interval in value:
            self._buffer[self._buffer_count]=(interval.lower,interval.upper,interval.lower_closed,interval.upper_closed)
            self._buffer_count+=1
            if self._buffer_count==self._buffer_capacity:
                self._flush_block()
        if len(self._pending)*256>=self.block_bytes:
            self._flush_index()

    def _flush_index(self):
        if self._pending:
            try:
                self._db.executemany("INSERT INTO regions VALUES(?,?,?,?,?)",self._pending)
            except sqlite3.IntegrityError as exc:
                raise RiskError("DUPLICATE_REGION_RECORD") from exc
            self._pending=[]
            self._db.commit()

    def _flush_block(self):
        if not self._buffer_count:
            return
        index=len(self._blocks)
        file=Path(self.folder)/f"intervals-{index:06d}.npy"
        values=self._buffer[:self._buffer_count]
        with file.open("wb") as stream:
            np.save(stream,values,allow_pickle=False)
            stream.flush()
            __import__("os").fsync(stream.fileno())
        self._blocks.append({"file":file.name,"sha256":sha(file),"start":self._written_intervals,"count":len(values),"bytes":file.stat().st_size})
        self._written_intervals+=len(values)
        self._buffer_count=0

    def freeze(self):
        if self._frozen:
            return self
        if self.folder:
            self._flush_block()
            self._flush_index()
            self._db.close()
            self._db=None
            self._buffer=np.empty(0,dtype=DTYPE)
            manifest={"schema":"QQQ_REGION_BLOCKS_V1","identity":self.identity,"records":self._count,
                      "index_sha256":sha(Path(self.folder)/"index.sqlite"),"blocks":self._blocks,
                      "block_target_bytes":self.block_bytes,"max_decoded_record_bytes":self.max_record_bytes,"dtype":DTYPE.descr}
            (Path(self.folder)/"manifest.json").write_text(json.dumps(manifest,sort_keys=True),encoding="utf-8")
            (Path(self.folder)/"manifest.sha256").write_text(sha(Path(self.folder)/"manifest.json"),encoding="ascii")
        self._frozen=True
        self._digest=region_digest(self)
        return self

    def _reader(self):
        if self._db is None:
            if (Path(self.folder)/"manifest.sha256").read_text(encoding="ascii")!=sha(Path(self.folder)/"manifest.json"):
                raise RiskError("REGION_STORE_INTEGRITY_FAILURE")
            manifest=json.loads((Path(self.folder)/"manifest.json").read_text(encoding="utf-8"))
            if manifest["identity"]!=self.identity or manifest["index_sha256"]!=sha(Path(self.folder)/"index.sqlite"):
                raise RiskError("REGION_STORE_INTEGRITY_FAILURE")
            self._blocks=manifest["blocks"]
            self._db=sqlite3.connect((Path(self.folder)/"index.sqlite").as_uri()+"?mode=ro&immutable=1",uri=True)
        return self._db

    def _intervals(self,start,count):
        if count>self._record_interval_limit:
            raise RiskError('RESOURCE_LIMIT','One target-region record exceeds its decoded working-set budget',
                            intervals=count,max_record_bytes=self.max_record_bytes)
        if not count:
            return ()
        result=[]
        first=max(0,bisect.bisect_right([b["start"] for b in self._blocks],start)-1)
        for index in range(first,len(self._blocks)):
            block=self._blocks[index]
            offset=block["start"]
            end=offset+block["count"]
            if end>start and offset<start+count:
                array=self._cache.pop(index,None)
                if array is None:
                    file=Path(self.folder)/block["file"]
                    if not file.resolve().is_relative_to(Path(self.folder).resolve()) or sha(file)!=block["sha256"]:
                        raise RiskError("REGION_STORE_INTEGRITY_FAILURE")
                    array=np.load(file,mmap_mode="r",allow_pickle=False)
                    if array.dtype!=DTYPE or array.ndim!=1:
                        raise RiskError("REGION_STORE_SCHEMA_MISMATCH")
                self._cache[index]=array
                while len(self._cache)>1:
                    self._cache.popitem(last=False)
                lo,hi=max(0,start-offset),min(block["count"],start+count-offset)
                result.extend(Interval(float(x["lower"]),float(x["upper"]),bool(x["lower_closed"]),bool(x["upper_closed"])) for x in array[lo:hi])
            offset=end
            if offset>=start+count:
                break
        if len(result)!=count:
            raise RiskError("REGION_STORE_TRUNCATED")
        return tuple(result)

    def __getitem__(self,key):
        if self.folder is None:
            return self._memory[key]
        if not self._frozen:
            raise RiskError("REGION_STORE_NOT_FROZEN")
        row=self._reader().execute("SELECT start,n FROM regions WHERE t=? AND h=?",(int(key[0]),float(key[1]))).fetchone()
        if row is None:
            raise KeyError(key)
        return self._intervals(*row)

    def __iter__(self):
        if self.folder is None:
            yield from sorted(self._memory)
        else:
            yield from ((int(t),float(h)) for t,h in self._reader().execute("SELECT t,h FROM regions ORDER BY t,h"))

    def ordered_items(self):
        if self.folder is None:
            yield from sorted(self._memory.items())
        else:
            for t,h,start,n in self._reader().execute("SELECT t,h,start,n FROM regions ORDER BY t,h"):
                yield (int(t),float(h)),self._intervals(start,n)

    def __len__(self):
        return self._count

    def statistics(self):
        return {"records":len(self),"disk_backed":bool(self.folder),"directory":self.folder,
                "block_target_bytes":self.block_bytes,"blocks":len(self._blocks),
                "disk_bytes":sum(p.stat().st_size for p in Path(self.folder).iterdir()) if self.folder else 0,
                "estimated_in_memory_bytes":self._memory_bytes+sys.getsizeof(self._memory)+self._buffer.nbytes,
                "write_buffer_capacity_bytes":self._buffer_capacity*DTYPE.itemsize,
                "max_decoded_record_bytes":self.max_record_bytes,"content_sha256":self._digest}

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db=None
        self._cache.clear()

    def __getstate__(self):
        if not self._frozen:
            raise RiskError("REGION_STORE_NOT_FROZEN")
        return {k:v for k,v in self.__dict__.items() if k not in ("_db","_cache")}

    def __setstate__(self,state):
        self.__dict__.update(state)
        self._db=None
        self._cache=OrderedDict()
