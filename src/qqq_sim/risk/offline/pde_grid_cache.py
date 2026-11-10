"""Shared, resumable normalized American grids for one exact (w,a,b) law.

Only grids with independent spatial, temporal and boundary evidence supply
labels. Interpolation is tested on the requested x domain; its measured error
shares the spatial budget. These are empirical sensitivities, not uniform proofs.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import scipy

from ...config import writable_path
from ..contracts import RiskError, stable_hash
from ..pack_loader import file_hash
from .hit_oracle import atomic_json
from .pricing_oracle import AmericanGrid, PriceResult


SCHEMA = "NORMALIZED_SHARED_PDE_GRID_V1"


@contextmanager
def _law_lock(folder):
    """An OS-owned lock is released even when a build process is killed."""
    with (folder/"writer.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RiskError("RESOURCE_LIMIT", "The shared normalized law is being built by another process", resource="normalized_pde_cache_busy") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _source_digest(path, key):
    path = Path(path)
    if path.is_file():
        return file_hash(path)
    from ..._build_info import SOURCE_HASHES
    value = SOURCE_HASHES.get(key)
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise RiskError("IMPLEMENTATION_IDENTITY_UNAVAILABLE", source=key)
    return value


def normalized_step_schedule(scale, rate):
    """Exact same initial segment partitions as AmericanGrid.advance(1)."""
    start = 0.
    for end, nominal in ((1/365, 60/(365*86400)), (1., 3600/(365*86400))):
        if rate < 0:
            nominal = min(nominal, .25/-rate)
        count = max(1, math.ceil((end-start)/nominal))*scale
        for index in range(count):
            yield end if index == count-1 else start+(end-start)*(index+1)/count
        start = end


class NormalizedGridCache:
    def __init__(self, directory, *, epsilon, kernel_hash, initial_nodes=1601,
                 max_nodes=25601, max_time_scale=1024, memory_bytes=64<<20):
        self.directory = writable_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.epsilon, self.kernel_hash = float(epsilon), kernel_hash
        self.initial_nodes, self.max_nodes = int(initial_nodes), int(max_nodes)
        self.max_time_scale, self.memory_bytes = int(max_time_scale), int(memory_bytes)
        if not 0 < self.epsilon or initial_nodes < 3 or max_nodes < initial_nodes or memory_bytes <= 0:
            raise RiskError("INVALID_PRICING_CONFIGURATION")
        self.source_hash = _source_digest(__file__, "src/qqq_sim/risk/offline/pde_grid_cache.py")
        self._memory = OrderedDict()
        self._memory_size = 0
        self.grid_solves = self.memory_hits = self.disk_hits = self.resumed_steps = 0

    def _read_json(self, path):
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            digest = record.pop("payload_sha256", None)
            if digest != stable_hash(record):
                raise RiskError("CHECKPOINT_MISMATCH", artifact=str(path))
            return record
        except (ValueError, OSError) as exc:
            raise RiskError("CHECKPOINT_MISMATCH", artifact=str(path)) from exc

    def _write_json(self, path, value):
        # Digest and metadata have one atomic commit point. A process crash
        # cannot leave a new manifest paired with an old separate receipt.
        atomic_json(path, {**value, "payload_sha256":stable_hash(value)})

    def _remember(self, key, arrays):
        size = sum(value.nbytes for value in arrays.values())
        limit = self.memory_bytes//4
        if key in self._memory:
            old = self._memory.pop(key)
            self._memory_size -= sum(value.nbytes for value in old.values())
        while self._memory and self._memory_size+size > limit:
            _, old = self._memory.popitem(last=False)
            self._memory_size -= sum(value.nbytes for value in old.values())
        if size <= limit:
            self._memory[key] = arrays
            self._memory_size += size

    def _write_arrays(self, folder, arrays):
        from .training_store import require_disk_space
        temporary = folder/"numeric.tmp"
        require_disk_space(temporary, sum(np.asarray(v).nbytes for v in arrays.values())+4096)
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        digest = file_hash(temporary)
        target = folder/(digest+".npz")
        os.replace(temporary, target)
        return {"file": target.name, "sha256": digest, "bytes": target.stat().st_size}

    def _read_arrays(self, folder, record):
        name = record.get("file", "")
        if Path(name).name != name or name != record.get("sha256", "")+".npz":
            raise RiskError("CHECKPOINT_MISMATCH", artifact=name)
        path = folder/name
        # Verify disk bytes even for a memory hit: mutable local artifacts are
        # never trusted merely because an earlier request loaded them.
        if not path.is_file() or path.stat().st_size != record.get("bytes") or file_hash(path) != record.get("sha256"):
            raise RiskError("CHECKPOINT_MISMATCH", artifact=str(path))
        if str(path) in self._memory:
            self.memory_hits += 1
            arrays = self._memory.pop(str(path))
            self._memory[str(path)] = arrays
            return arrays
        self.disk_hits += 1
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"spots", "lower", "upper"}:
                raise RiskError("CHECKPOINT_MISMATCH", artifact=str(path))
            arrays = {name: np.asarray(data[name]) for name in data.files}
        shape = arrays["spots"].shape
        if (len(shape) != 1 or not 3 <= shape[0] <= self.max_nodes or
            any(value.dtype != np.dtype("float64") or value.shape != shape or not np.all(np.isfinite(value)) for value in arrays.values()) or
            np.any(arrays["spots"] <= 0) or np.any(np.diff(arrays["spots"]) <= 0) or
            np.any(arrays["upper"] < arrays["lower"]-1e-8)):
            raise RiskError("CHECKPOINT_MISMATCH", artifact=str(path))
        for value in arrays.values():
            value.flags.writeable = False
        self._remember(str(path), arrays)
        return arrays

    def _solve_grid(self, folder, law, domain, nodes, scale, deadline):
        identity = stable_hash({"law": law, "domain": domain, "nodes": nodes, "scale": scale})
        path = folder/(identity+".json")
        saved = self._read_json(path)
        if saved and saved.get("identity") != identity:
            raise RiskError("CHECKPOINT_MISMATCH", artifact=str(path))
        if saved and saved.get("complete"):
            return self._read_arrays(folder, saved["numeric"]), saved
        # Includes solver temporaries, retained three-grid comparisons and LRU.
        estimate = nodes*8*80+self._memory_size
        if estimate > self.memory_bytes:
            raise RiskError("RESOURCE_LIMIT", "Normalized PDE workspace exceeds memory budget", resource="normalized_pde_memory", estimated_bytes=estimate, budget_bytes=self.memory_bytes)
        grid = AmericanGrid(law["right"], 1., law["w"], law["a"], law["b"], domain,
                            nodes, self.epsilon, scale, deadline)
        offset = 0
        if saved:
            arrays = self._read_arrays(folder, saved["numeric"])
            if not np.array_equal(arrays["spots"], grid.spots):
                raise RiskError("CHECKPOINT_MISMATCH", artifact=str(path))
            grid.lower, grid.upper = arrays["lower"].copy(), arrays["upper"].copy()
            offset, grid.tau, grid.steps = int(saved["completed_schedule_steps"]), saved["tau"], saved["steps"]
            grid.residual, grid.max_iterations = saved["residual"], saved["max_iterations"]
            self.resumed_steps += offset
        previous_artifact = saved.get("numeric") if saved else None
        def save(complete=False):
            nonlocal previous_artifact
            numeric = self._write_arrays(folder, {"spots": grid.spots, "lower": grid.lower, "upper": grid.upper})
            record = {"identity": identity, "complete": complete, "completed_schedule_steps": offset,
                      "tau": grid.tau, "steps": grid.steps, "residual": grid.residual,
                      "max_iterations": grid.max_iterations, "numeric": numeric,
                      "nodes": nodes, "time_scale": scale, "domain": list(domain)}
            self._write_json(path, record)
            if previous_artifact and previous_artifact["file"] != numeric["file"]:
                old = writable_path(folder/previous_artifact["file"])
                referenced = any((self._read_json(other) or {}).get("numeric", {}).get("file") == old.name for other in folder.glob("*.json"))
                if not referenced:
                    self._memory.pop(str(old), None)
                    self._memory_size = sum(sum(v.nbytes for v in values.values()) for values in self._memory.values())
                    old.unlink(missing_ok=True)
            previous_artifact = numeric
            return record
        self.grid_solves += 1
        try:
            for index, next_tau in enumerate(normalized_step_schedule(scale, law["a"])):
                if index < offset:
                    if index == offset-1 and next_tau != grid.tau:
                        raise RiskError("CHECKPOINT_MISMATCH", artifact=str(path), reason="time_partition")
                    continue
                if time.monotonic() >= deadline:
                    raise RiskError("RESOURCE_LIMIT", "Shared normalized PDE time-step budget reached", resource="normalized_pde", nodes=nodes, time_scale=scale, completed_schedule_steps=offset, tau=grid.tau)
                # Preserve the deterministic partition across interruptions.
                grid._step(next_tau-grid.tau)
                grid.tau = next_tau
                offset = index+1
                if offset % 256 == 0:
                    save()
        except RiskError as exc:
            if exc.code == "RESOURCE_LIMIT":
                save()
            raise
        record = save(True)
        return self._read_arrays(folder, record["numeric"]), record

    @staticmethod
    def _evaluate(arrays, spots):
        lower = np.interp(spots, arrays["spots"], arrays["lower"])
        upper = np.interp(spots, arrays["spots"], arrays["upper"])
        return (lower+upper)*.5, (upper-lower)*.5

    def value(self, right, point, requested_x, *, max_seconds):
        x, w, a, b = map(float, point)
        low, high = map(float, requested_x)
        if not low <= x <= high or not all(map(math.isfinite, (x, w, a, b, low, high))) or w <= 0:
            raise RiskError("INVALID_NORMALIZED_GRID_REQUEST")
        law = {"right": right, "w": w, "a": a, "b": b, "requested_x": [low, high],
               "epsilon": self.epsilon, "initial_nodes": self.initial_nodes, "max_nodes": self.max_nodes,
               "max_time_scale": self.max_time_scale, "kernel": self.kernel_hash,
               "cache_source": self.source_hash, "schema": SCHEMA, "numpy": np.__version__, "scipy":scipy.__version__}
        identity = stable_hash(law)
        folder = self.directory/identity
        folder.mkdir(parents=True, exist_ok=True)
        with _law_lock(folder):
            return self._value_locked(law, identity, folder, x, low, high, max_seconds)

    def _value_locked(self, law, identity, folder, x, low, high, max_seconds):
        w, a, b = law["w"], law["a"], law["b"]
        accepted_path = folder/"accepted.json"
        accepted = self._read_json(accepted_path)
        if accepted:
            if accepted.get("identity") != identity or accepted.get("status") != "CONVERGED":
                raise RiskError("CHECKPOINT_MISMATCH", artifact=str(accepted_path))
            arrays = self._read_arrays(folder, accepted["numeric"])
            value, _ = self._evaluate(arrays, math.exp(x))
            return PriceResult(float(value), accepted["sensitivity"], "CONVERGED", accepted["diagnostics"])
        deadline = time.monotonic()+max(0., float(max_seconds))
        pad = max(.5, 8*w+abs(a-b-w*w/2))
        domain = (math.exp(min(low, 0.)-pad), math.exp(max(high, 0.)+pad))
        base_nodes = self.initial_nodes
        records = []
        for expansion in range(20):
            nlevels = [base_nodes, 2*base_nodes-1, 4*base_nodes-3]
            scales = [1, 2, 4]
            while nlevels[-1] <= self.max_nodes and scales[-1] <= self.max_time_scale:
                current, final_record = None, None
                for nodes in nlevels[-3:]:
                    for scale in scales[-3:]:
                        current, final_record = self._solve_grid(folder, law, domain, nodes, scale, deadline)
                spots = current["spots"]
                inside = spots[(spots >= math.exp(low)) & (spots <= math.exp(high))]
                # Knot and midpoint extraction tests resolve interpolation on
                # the actual requested cell, not just the query's single x.
                checks = np.unique(np.concatenate((np.exp(np.linspace(low, high, 257)), inside,
                          (inside[:-1]+inside[1:])*.5, [math.exp(low), math.exp(high)],
                          np.array([1.-1e-6, 1., 1.+1e-6]))))
                checks = checks[(checks >= domain[0]) & (checks <= domain[1])]
                values, boundary = self._evaluate(current, checks)
                spatial_values, temporal_values = [], []
                for nodes in nlevels[-3:]:
                    arrays, _ = self._solve_grid(folder, law, domain, nodes, scales[-1], deadline)
                    spatial_values.append(self._evaluate(arrays, checks)[0])
                for scale in scales[-3:]:
                    arrays, _ = self._solve_grid(folder, law, domain, nlevels[-1], scale, deadline)
                    temporal_values.append(self._evaluate(arrays, checks)[0])
                spatial = [float(np.max(np.abs(v-u))) for u, v in zip(spatial_values[:-1], spatial_values[1:])]
                temporal = [float(np.max(np.abs(v-u))) for u, v in zip(temporal_values[:-1], temporal_values[1:])]
                bound = float(np.max(boundary))
                records.append({"nodes": nlevels[-1], "time_scale": scales[-1], "space_differences": spatial,
                                "time_differences": temporal, "boundary_halfwidth": bound, "validation_points": len(checks)})
                self._write_json(folder/"refinement_report.json", {"identity": identity, "records": records})
                if bound > self.epsilon/4:
                    width = math.log(domain[1]/domain[0])
                    extra = max(.5, .25*width)
                    base_nodes = math.ceil((base_nodes-1)*(width+2*extra)/width)+1
                    domain = (domain[0]*math.exp(-extra), domain[1]*math.exp(extra))
                    break
                if max(spatial) <= self.epsilon/4 and max(temporal) <= self.epsilon/4:
                    sensitivity = bound+max(spatial)+max(temporal)+final_record["residual"]
                    if sensitivity > self.epsilon:
                        raise RiskError("PRICE_REFERENCE_UNRESOLVED", sensitivity=sensitivity, epsilon=self.epsilon)
                    diagnostics = {"method": SCHEMA, "requested_x": [low, high], "domain": list(domain),
                                   "refinements": records, "interpolation": "LINEAR_IN_SPOT",
                                   "interpolation_budget": "INCLUDED_IN_SPATIAL_KNOT_AND_MIDPOINT_REFINEMENT",
                                   "validation_domain": "REQUESTED_CELL_AND_STRIKE_NEIGHBORHOOD",
                                   "error_interpretation": "MEASURED_NUMERICAL_SENSITIVITY", "grid_identity": identity}
                    accepted = {"identity": identity, "status": "CONVERGED", "numeric": final_record["numeric"],
                                "sensitivity": sensitivity, "diagnostics": diagnostics}
                    self._write_json(accepted_path, accepted)
                    value, _ = self._evaluate(current, math.exp(x))
                    return PriceResult(float(value), sensitivity, "CONVERGED", diagnostics)
                if max(spatial) >= max(temporal):
                    nlevels.append(2*nlevels[-1]-1)
                else:
                    scales.append(2*scales[-1])
            else:
                raise RiskError("PRICER_NOT_CONVERGED", "Shared grid exhausted independent spatial/temporal refinement caps", refinements=records, max_nodes=self.max_nodes, max_time_scale=self.max_time_scale)
        raise RiskError("NUMERICAL_DOMAIN_UNRESOLVED", refinements=records)
