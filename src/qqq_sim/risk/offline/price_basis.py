"""Adaptive four-dimensional American-price tensor builder (explicit offline).

Every cell carries measured evidence. Failed oracle nodes exclude the cell;
they are never replaced by zero or an asymptotic guess.
"""
from __future__ import annotations
import itertools
import hashlib
import math
import time
from pathlib import Path
import numpy as np
import scipy
from ..contracts import RiskError, stable_hash
from ..pricing_fast import tensor_chebyshev, european_price
from .pricing_oracle import american_price
from .hit_oracle import atomic_json

AXES = (
    (-math.log(8), -math.log(4), -math.log(2), -.25, -.10, 0, .10, .25, math.log(2), math.log(4), math.log(8)),
    (0, .01, .02, .04, .08, .16, .32, .64, 1),
    (-.002, 0, .002, .005, .01, .02),
    (0, .002, .005, .01, .02),
)


def build_price_tree(bounds):
    """Offline deterministic balanced spatial tree over accepted cell indices."""
    from ..pricing_fast import PriceDecisionTree,MAX_PRICE_CELLS
    bounds=np.asarray(bounds,dtype=np.float64)
    count=bounds.shape[0] if bounds.ndim else 0
    if (bounds.shape!=(count,4,2) or not 1<=count<=MAX_PRICE_CELLS
        or not np.all(np.isfinite(bounds)) or np.any(bounds[:,:,0]>=bounds[:,:,1])):
        raise RiskError('INVALID_PRICE_BASIS')
    nodes=np.full((2*count-1,3),-1,dtype=np.int64)
    node_bounds=np.empty((2*count-1,4,2),dtype=np.float64)
    cursor=0
    def add(indices):
        nonlocal cursor
        index=cursor; cursor+=1
        selected=bounds[indices]
        lo,hi=selected[:,:,0].min(axis=0),selected[:,:,1].max(axis=0)
        node_bounds[index,:,0],node_bounds[index,:,1]=lo,hi
        if len(indices)==1:
            nodes[index,2]=indices[0]
            return index
        centers=selected[:,:,0]+(selected[:,:,1]-selected[:,:,0])*.5
        spread=np.ptp(centers,axis=0)/(hi-lo)
        axis=int(np.argmax(spread))
        ordered=indices[np.lexsort((indices,centers[:,axis]))]
        middle=len(ordered)//2
        nodes[index,:2]=add(ordered[:middle]),add(ordered[middle:])
        return index
    add(np.arange(count,dtype=np.int64))
    PriceDecisionTree(bounds,nodes,node_bounds)
    return {'price_tree_nodes':nodes,'price_tree_bounds':node_bounds}


def _source_hash(path, key):
    """Use live bytes or the exact source digest embedded by the build step."""
    source = Path(path)
    if source.is_file():
        return hashlib.sha256(source.read_bytes()).hexdigest()
    from ..._build_info import SOURCE_HASHES
    digest = SOURCE_HASHES.get(key)
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise RiskError("IMPLEMENTATION_IDENTITY_UNAVAILABLE", source=key)
    return digest


def initial_cells():
    return [np.array(bounds, dtype=np.float64) for bounds in itertools.product(*[list(zip(axis[:-1], axis[1:])) for axis in AXES])]


def transform_lobatto(values):
    n = 3
    alpha, beta = np.array([.5, 1, 1, .5]), np.array([.5, 1, 1, .5])
    transform = (2/n)*alpha[:, None]*np.cos(np.pi*np.arange(4)[:, None]*np.arange(4)/n)*beta
    result = np.asarray(values, dtype=np.float64)
    for axis in range(4):
        result = np.moveaxis(np.tensordot(transform, result, axes=(1, axis)), 0, axis)
    return result


def normalized_cell_value_and_greeks(cell, point, right_index):
    """K=tau=1 physical derivatives, with actual coordinate scale factors."""
    bounds = np.asarray(cell["bounds"], dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    scale = 2/np.diff(bounds, axis=1)[:, 0]
    z = (point-bounds[:, 0])*scale-1
    coefficients = cell["coefficients"][right_index]
    value = tensor_chebyshev(coefficients, z)
    derivative = np.array([tensor_chebyshev(coefficients, z, axis)*scale[axis] for axis in range(4)])
    second = tensor_chebyshev(coefficients, z, 0, True)*scale[0]**2
    x, w, a, b = point
    spot = math.exp(x)
    greeks = np.array([derivative[0]/spot, (second-derivative[0])/spot**2,
                       .01*derivative[1], -(w*.5*derivative[1]+a*derivative[2]+b*derivative[3])/365])
    return value, greeks


def independent_normalized_greeks(oracle, right, point, deadline=None):
    """Three finite-difference grids with propagated reference uncertainty."""
    x, w, a, b = map(float, point)
    spot = math.exp(x)
    estimates, uncertainties = [], []
    def price(coordinate):
        if deadline is not None and time.monotonic() >= deadline:
            raise RiskError("RESOURCE_LIMIT", "Independent Greek validation budget reached")
        if deadline is not None:
            oracle.max_solve_seconds = min(getattr(oracle,"max_solve_seconds",120), max(.01,deadline-time.monotonic()))
        return oracle.value(right, coordinate)
    middle, middle_error = price(point)
    for refinement in range(3):
        hs, hv, ht = spot*1e-3/(2**refinement), 1e-3/(2**refinement), min(60/(365*86400), .25)/(2**refinement)
        lower, el = price([math.log(spot-hs), w, a, b])
        upper, eu = price([math.log(spot+hs), w, a, b])
        delta = (upper-lower)/(2*hs)
        gamma = (upper-2*middle+lower)/hs**2
        if w >= hv:
            lowv, evl = price([x, w-hv, a, b]); highv, evu = price([x, w+hv, a, b])
            vega, vega_error = .01*(highv-lowv)/(2*hv), .01*(evu+evl)/(2*hv)
        else:
            highv, evu = price([x, w+hv, a, b])
            vega, vega_error = .01*(highv-middle)/hv, .01*(evu+middle_error)/hv
        lowt, etl = price([x, w*math.sqrt(1-ht), a*(1-ht), b*(1-ht)])
        hight, etu = price([x, w*math.sqrt(1+ht), a*(1+ht), b*(1+ht)])
        theta = -(hight-lowt)/(2*ht*365)
        estimates.append(np.array([delta, gamma, vega, theta]))
        uncertainties.append(np.array([(eu+el)/(2*hs), (eu+2*middle_error+el)/hs**2, vega_error, (etu+etl)/(2*ht*365)]))
    sensitivity = np.maximum(np.abs(estimates[1]-estimates[0]), np.abs(estimates[2]-estimates[1]))
    return estimates[-1], sensitivity+np.max(uncertainties, axis=0)


def shared_cell_faces(left_bounds, right_bounds):
    left, right = np.asarray(left_bounds), np.asarray(right_bounds)
    for axis in range(4):
        coordinate = left[axis, 1] if left[axis, 1] == right[axis, 0] else left[axis, 0] if left[axis, 0] == right[axis, 1] else None
        if coordinate is None:
            continue
        overlap = np.column_stack((np.maximum(left[:, 0], right[:, 0]), np.minimum(left[:, 1], right[:, 1])))
        tangential = [index for index in range(4) if index != axis]
        if all(overlap[index, 0] < overlap[index, 1] for index in tangential):
            overlap[axis] = coordinate
            yield axis, overlap


def validate_neighbor_faces(left, right, *, oracle=None, price_tolerance=1e-10, greek_tolerance=1e-7, deadline=None):
    """Every common face, including coarse/fine subfaces, on its5^3 grid."""
    faces = list(shared_cell_faces(left["bounds"], right["bounds"]))
    price_errors = np.array([left["errors"], right["errors"]], dtype=np.float64)
    greek_errors = np.array([left["greek_errors"], right["greek_errors"]], dtype=np.float64)
    greek_cross = np.zeros((2, 4))
    price_cross = np.zeros(2)
    failures, records = set(), []
    residual_variation = np.zeros(4)
    for axis, face in faces:
        tangential = [index for index in range(4) if index != axis]
        count = 0
        residual_samples = []
        for coordinate in itertools.product((0., .25, .5, .75, 1.), repeat=3):
            if deadline is not None and time.monotonic() >= deadline:
                raise RiskError("RESOURCE_LIMIT", "Cross-face validation budget reached")
            point = face.mean(axis=1)
            point[tangential] = face[tangential, 0]+np.array(coordinate)*np.diff(face[tangential], axis=1)[:, 0]
            sample_residual = np.zeros(2)
            for ri, option_right in enumerate(("CALL", "PUT")):
                lv, lg = normalized_cell_value_and_greeks(left, point, ri)
                rv, rg = normalized_cell_value_and_greeks(right, point, ri)
                price_cross[ri] = max(price_cross[ri], abs(lv-rv))
                greek_cross[ri] = np.maximum(greek_cross[ri], np.abs(lg-rg))
                if oracle is not None:
                    if deadline is not None:
                        oracle.max_solve_seconds = min(getattr(oracle,"max_solve_seconds",120), max(.01,deadline-time.monotonic()))
                    expected, uncertainty = oracle.value(option_right, point)
                    reference, greek_uncertainty = independent_normalized_greeks(oracle, option_right, point, deadline)
                    for index, (value, derivative) in enumerate(((lv, lg), (rv, rg))):
                        measured = abs(value-expected)+uncertainty
                        sample_residual[index] = max(sample_residual[index], measured)
                        price_errors[index, ri] = max(price_errors[index, ri], measured)
                        greek_errors[index, ri] = np.maximum(greek_errors[index, ri], np.abs(derivative-reference)+greek_uncertainty)
                        if measured > price_tolerance:
                            failures.add(index)
                elif abs(lv-rv) > 2*price_tolerance:
                    failures.update((0, 1))
            count += 1
            residual_samples.append((coordinate, sample_residual))
        # Midplane residual variation selects a deterministic refinement axis;
        # a constant residual has the specified x,w,a,b tie order.
        for local_axis, physical_axis in enumerate(tangential):
            selected = [residual for coordinate, residual in residual_samples if all(coordinate[index] == .5 for index in range(3) if index != local_axis)]
            if selected:
                residual_variation[physical_axis] = max(residual_variation[physical_axis], float(np.max(np.ptp(selected, axis=0)))/(face[physical_axis, 1]-face[physical_axis, 0]))
        records.append({"axis": axis, "bounds": face.tolist(), "validation_points_per_right": count})
    # A derivative mismatch is measured evidence of an unavailable Greek, even
    # when both independently accepted price polynomials agree on the face.
    greek_errors = np.maximum(greek_errors, greek_cross[None, :, :])
    return {"adjacent": bool(faces), "faces": records, "price_ok": not failures,
            "failed_cells": sorted(failures), "greeks_qualified": bool(np.all(greek_errors <= greek_tolerance)),
            "cell_greeks_qualified": [bool(np.all(item <= greek_tolerance)) for item in greek_errors],
            "price_errors": price_errors, "greek_errors": greek_errors,
            "maximum_price_disagreement": price_cross, "maximum_greek_disagreement": greek_cross,
            "split_axis": int(np.argmax(residual_variation)), "independent_reference_checked": oracle is not None}


class OraclePriceCache:
    def __init__(self, directory, *, epsilon=1e-10, max_solve_seconds=120):
        from ...config import writable_path
        self.directory = writable_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.epsilon, self.max_solve_seconds = epsilon, max_solve_seconds
        self.calls = 0
        from . import pricing_oracle
        self.kernel_hash = _source_hash(pricing_oracle.__file__, "src/qqq_sim/risk/offline/pricing_oracle.py")
        from .pde_grid_cache import NormalizedGridCache
        self.grid_cache = NormalizedGridCache(self.directory/"shared_grids", epsilon=epsilon, kernel_hash=self.kernel_hash)
        self.requested_x = None

    def set_requested_domain(self, x_bounds):
        low, high = map(float, x_bounds)
        if not math.isfinite(low+high) or low >= high:
            raise RiskError("INVALID_NORMALIZED_GRID_REQUEST")
        # The independent Greek check shifts spot by at most 0.1 percent.
        self.requested_x = (low-.002, high+.002)

    def value(self, right, point):
        import json
        x, w, a, b = map(float, point)
        key = stable_hash({"right": right, "point": list(point), "eps": self.epsilon, "version": "P50_R_SHARED_GRID_V1", "kernel_hash": self.kernel_hash, "grid_source":self.grid_cache.source_hash, "numpy": np.__version__,"scipy":scipy.__version__})
        file = self.directory/(key+".json")
        if file.exists():
            from ..pack_loader import file_hash
            receipt=file.with_suffix(".sha256")
            if not receipt.is_file() or receipt.read_text(encoding="ascii").strip()!=file_hash(file):
                raise RiskError("CHECKPOINT_MISMATCH",artifact=str(file))
            cached = json.loads(file.read_text(encoding="utf-8"))
            if cached.get("status") not in ("CONVERGED", "EXACT", "EXACT_FORMULA"):
                raise RiskError("PRICE_REFERENCE_UNRESOLVED", cached=cached)
            return cached["price"], cached["sensitivity"]
        self.calls += 1
        try:
            if w == 0 or (right == "CALL" and b == 0 and a >= 0):
                result = american_price(right, math.exp(x), 1, 1, a, b, w, eps=self.epsilon, max_seconds=self.max_solve_seconds, max_nodes=25601)
            else:
                bounds = self.requested_x
                if bounds is None or not bounds[0] <= x <= bounds[1]:
                    bounds = (x-.002, x+.002)
                result = self.grid_cache.value(right, (x,w,a,b), bounds, max_seconds=self.max_solve_seconds)
        except RiskError as exc:
            raise RiskError(exc.code,str(exc),**{**exc.details,"normalized_point":{"right":right,"x":x,"w":w,"a":a,"b":b},"requested_price_epsilon":self.epsilon,"per_oracle_seconds":self.max_solve_seconds}) from exc
        record = {"status": result.status, "price": result.price, "sensitivity": result.sensitivity, "diagnostics": result.diagnostics}
        atomic_json(file, record)
        from ..pack_loader import file_hash
        file.with_suffix(".sha256").write_text(file_hash(file)+"\n",encoding="ascii")
        if result.status not in ("CONVERGED", "EXACT", "EXACT_FORMULA"):
            raise RiskError("PRICE_REFERENCE_UNRESOLVED")
        return result.price, result.sensitivity


def midplane_residual_axis(bounds, coordinates, residuals):
    """Five-point signed residual variation on each central axis, divided by width."""
    bounds = np.asarray(bounds, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.ndim == 1:
        residuals = residuals[:, None]
    variations = np.zeros(4)
    for axis in range(4):
        other = [i for i in range(4) if i != axis]
        selected = np.all(coordinates[:, other] == 0., axis=1)
        samples = residuals[selected]
        if samples.shape[0] != 5 or set(coordinates[selected, axis]) != {-1., -.5, 0., .5, 1.}:
            raise RiskError("PRICE_BASIS_VALIDATION_GRID_INVALID", axis=axis)
        variations[axis] = float(np.max(np.ptp(samples, axis=0))) / (bounds[axis, 1] - bounds[axis, 0])
    # np.argmax retains the first maximum: the prescribed x,w,a,b tie order.
    return int(np.argmax(variations)), variations


def polynomial_roundoff_envelope(bounds, coefficients, point):
    """Conservative float64 evaluation allowance, propagated through derivatives."""
    # The degree-three tensor has 256 terms. This allows several rounded
    # operations per term and uses |T|<=1, |T'|<=9, |T''|<=24 on [-1,1].
    base = 1024 * np.finfo(np.float64).eps * float(np.sum(np.abs(coefficients)))
    scale = 2 / np.diff(np.asarray(bounds), axis=1)[:, 0]
    x, w, a, b = point
    spot = math.exp(x)
    first = base * 9 * scale
    second_x = base * 24 * scale[0]**2
    greek = np.array([first[0]/spot, (second_x+first[0])/spot**2, .01*first[1],
                      (abs(w)*.5*first[1]+abs(a)*first[2]+abs(b)*first[3])/365])
    return base, greek


def financial_inequality_checks(right, point, value, greeks, *, price_uncertainty=0., greek_uncertainty=None):
    """Check the sampled polynomial itself; never repair or clip its outputs."""
    x, w, a, b = map(float, point)
    spot = math.exp(x)
    eta = 1 if right == "CALL" else -1
    delta, gamma, vega = map(float, np.asarray(greeks)[:3])
    uncertainty = np.zeros(4) if greek_uncertainty is None else np.asarray(greek_uncertainty)
    if not np.all(np.isfinite([value, price_uncertainty, delta, gamma, vega, *uncertainty])) or price_uncertainty < 0 or np.any(uncertainty < 0):
        raise RiskError("PRICE_BASIS_VALIDATION_NONFINITE")
    intrinsic = max(eta*(spot-1), 0.)
    european = float(european_price(right, spot, 1., 1., a, b, w))
    upper = spot if right == "CALL" else math.exp(max(-a, 0.))
    # At fixed S,tau,sigma,r,q, homogeneity gives dV/dK=f-f_x=f-S*Delta.
    strike_derivative = value-spot*delta
    margins = {
        "INTRINSIC_LOWER_BOUND": (value-intrinsic, price_uncertainty),
        "EUROPEAN_LOWER_BOUND": (value-european, price_uncertainty),
        "FINANCIAL_UPPER_BOUND": (upper-value, price_uncertainty),
        "SPOT_MONOTONICITY": (eta*delta, float(uncertainty[0])),
        "STRIKE_MONOTONICITY": (-eta*strike_derivative, price_uncertainty+spot*float(uncertainty[0])),
        "SPOT_CONVEXITY": (gamma, float(uncertainty[1])),
        "NONNEGATIVE_VEGA": (vega, float(uncertainty[2])),
    }
    return {name: {"passed": bool(margin >= -error), "margin": float(margin), "uncertainty": float(error),
                   "violation_beyond_uncertainty": max(0., float(-margin-error))}
            for name, (margin, error) in margins.items()}


def build_cell(bounds, oracle, *, index=0, price_tolerance=1e-10, random_count=256, deadline=None):
    if hasattr(oracle, "set_requested_domain"):
        oracle.set_requested_domain(bounds[0])
    def check_budget():
        if deadline is not None and time.monotonic() >= deadline:
            raise RiskError("RESOURCE_LIMIT", "Price-cell build checkpoint required")
        if deadline is not None:
            oracle.max_solve_seconds=min(oracle.max_solve_seconds,max(.01,deadline-time.monotonic()))
    lobatto = np.cos(np.pi*np.arange(4)/3)
    coordinates = list(itertools.product(range(4), repeat=4))
    coefficients, maximum_errors, reference_errors,greek_errors = [], [], [],[]
    signed_residuals, roundoff_errors = [], []
    financial_audit = {}
    random = np.random.Generator(np.random.PCG64(np.random.SeedSequence(20261109, spawn_key=(4, index))))
    validation = [*itertools.product((-1, -.5, 0, .5, 1), repeat=4), *random.uniform(-1, 1, size=(random_count, 4))]
    for right in ("CALL", "PUT"):
        values = np.empty((4, 4, 4, 4), dtype=np.float64)
        reference_error = 0.
        for coordinate in coordinates:
            check_budget()
            z = lobatto[list(coordinate)]
            point = bounds[:, 0]+(z+1)/2*np.diff(bounds, axis=1)[:, 0]
            values[coordinate], e = oracle.value(right, point)
            reference_error = max(reference_error, e)
        coefs = transform_lobatto(values)
        error = 0.
        roundoff_error = 0.
        reference_uncertainties, residual_column = [], []
        maximum_greek=np.zeros(4)
        for z in validation:
            check_budget()
            z = np.array(z)
            point = bounds[:, 0]+(z+1)/2*np.diff(bounds, axis=1)[:, 0]
            expected, e = oracle.value(right, point)
            residual = tensor_chebyshev(coefs, z)-expected
            error = max(error, abs(residual))
            reference_error = max(reference_error, e)
            residual_column.append(residual)
            reference_uncertainties.append(e)
        # Every prescribed validation point receives an independent derivative
        # comparison, including all face grid points. Price agreement alone
        # cannot qualify a derivative, especially Gamma.
        for point_index, z in enumerate(validation):
            check_budget()
            point=bounds[:,0]+(np.asarray(z)+1)/2*np.diff(bounds,axis=1)[:,0]
            value, polynomial = normalized_cell_value_and_greeks({"bounds":bounds,"coefficients":(coefs,coefs)},point,0)
            reference, uncertainty = independent_normalized_greeks(oracle,right,point,deadline)
            arithmetic_price, arithmetic_greeks = polynomial_roundoff_envelope(bounds,coefs,point)
            maximum_greek=np.maximum(maximum_greek,np.abs(polynomial-reference)+uncertainty+arithmetic_greeks)
            roundoff_error = max(roundoff_error, arithmetic_price)
            checks = financial_inequality_checks(right,point,value,polynomial,
                price_uncertainty=reference_uncertainties[point_index]+arithmetic_price,
                greek_uncertainty=uncertainty+arithmetic_greeks)
            for name, check in checks.items():
                audit = financial_audit.setdefault(name,{"passed":True,"checks":0,"maximum_violation_beyond_uncertainty":0.,"maximum_uncertainty":0.,"first_failure":None})
                audit["checks"] += 1
                audit["passed"] = audit["passed"] and check["passed"]
                audit["maximum_violation_beyond_uncertainty"] = max(audit["maximum_violation_beyond_uncertainty"],check["violation_beyond_uncertainty"])
                audit["maximum_uncertainty"] = max(audit["maximum_uncertainty"],check["uncertainty"])
                if not check["passed"] and audit["first_failure"] is None:
                    audit["first_failure"] = {"right":right,"point":point.tolist(),**check}
        coefficients.append(coefs)
        maximum_errors.append(error)
        reference_errors.append(reference_error)
        greek_errors.append(maximum_greek)
        signed_residuals.append(residual_column)
        roundoff_errors.append(roundoff_error)
    # Conservative normalized per-contract engineering thresholds. Online
    # portfolio checks must still propagate strike and contract multiplicities.
    greek_qualified=bool(np.all(np.asarray(greek_errors)<=1e-7))
    split_axis, residual_variation = midplane_residual_axis(bounds,validation[:625],np.asarray(signed_residuals)[:,:625].T)
    financial_passed = all(item["passed"] for item in financial_audit.values())
    total_errors = np.array(maximum_errors)+reference_errors+roundoff_errors
    return {"accepted": bool(max(total_errors) <= price_tolerance and financial_passed), "bounds": bounds, "coefficients": np.array(coefficients), "errors": total_errors, "validation_count": len(validation), "split_axis": split_axis, "midplane_residual_variation":residual_variation.tolist(), "price_errors": maximum_errors, "reference_errors": reference_errors, "roundoff_errors":roundoff_errors, "greek_errors":np.asarray(greek_errors),"greeks_status":"VALIDATED" if greek_qualified else "GREEK_DISCONTINUITY_OR_ERROR_GATE_FAILED","greeks_qualified":greek_qualified,"financial_validation":{"passed":financial_passed,"checks":financial_audit,"scope":"PRESCRIBED_GRID_AND_FIXED_SEED_INTERIOR_POINTS"}}


def build_price_basis(directory, *, budget_seconds=600, max_cells=262144, max_initial_cells=None, epsilon=1e-10,max_solve_seconds=None):
    import json
    from ...config import writable_path
    folder = writable_path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder/"basis_checkpoint.json"
    from . import pricing_oracle, pde_grid_cache
    identity = stable_hash({"axes": AXES, "epsilon": epsilon, "max_cells": max_cells, "max_initial_cells": max_initial_cells,"declared_per_oracle_seconds":max_solve_seconds,
                            "builder_sha256": _source_hash(__file__, "src/qqq_sim/risk/offline/price_basis.py"),
                            "pde_kernel_sha256":_source_hash(pricing_oracle.__file__,"src/qqq_sim/risk/offline/pricing_oracle.py"),
                            "grid_cache_sha256":_source_hash(pde_grid_cache.__file__,"src/qqq_sim/risk/offline/pde_grid_cache.py"),
                            "numpy":np.__version__,"scipy":scipy.__version__,"face_gate": "ALL_SHARED_FACES_5CUBED_REFERENCE_GREEKS_V1"})
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise RiskError("CHECKPOINT_MISMATCH")
        pending, accepted, excluded = saved["pending"], saved["accepted"], saved["excluded"]
        from ..pack_loader import file_hash
        for item in accepted:
            if not (folder/item["file"]).is_file() or file_hash(folder/item["file"])!=item.get("sha256"):
                raise RiskError("CHECKPOINT_MISMATCH",artifact=item["file"])
    else:
        cells = initial_cells()
        if max_initial_cells is not None:
            cells = cells[:max_initial_cells]
        pending = [{"bounds": b.tolist(), "depth": 0, "id": str(i)} for i, b in enumerate(cells)]
        accepted, excluded = [], []
    deadline = time.monotonic()+budget_seconds
    oracle_budget=budget_seconds if max_solve_seconds is None else min(float(max_solve_seconds),budget_seconds)
    if oracle_budget<=0:raise RiskError("INVALID_WORK_BUDGET")
    oracle = OraclePriceCache(folder/"oracle_cache", epsilon=epsilon, max_solve_seconds=oracle_budget)
    last_stop=None
    def split(item, axis, reason):
        if item["depth"] >= 20:
            excluded.append({**item, "reason": "PRICE_BASIS_UNVALIDATED_CELL", "failed_gate": reason})
            return
        bounds = np.asarray(item["bounds"])
        middle = float(bounds[axis].mean())
        left, right = bounds.copy(), bounds.copy()
        left[axis, 1], right[axis, 0] = middle, middle
        pending[:0] = [{"bounds": b.tolist(), "depth": item["depth"]+1, "id": item["id"]+"/"+str(i), "refinement_reason": reason} for i, b in enumerate((left, right))]

    def write_cell(item, result):
        import os, uuid
        stem = "cell_"+item["id"].replace("/", "_")
        temporary = folder/(stem+"."+uuid.uuid4().hex+".part")
        with temporary.open("wb") as stream:
            np.savez(stream, bounds=result["bounds"], coefficients=result["coefficients"], errors=result["errors"], greek_errors=result["greek_errors"])
            stream.flush(); os.fsync(stream.fileno())
        from ..pack_loader import file_hash
        fingerprint = file_hash(temporary)
        file = folder/(stem+"_"+fingerprint[:16]+".npz")
        os.replace(temporary,file)
        return {**item,"file":file.name,"sha256":fingerprint,"validation_count":result.get("validation_count",item.get("validation_count",0)),
                "financial_validation":result.get("financial_validation",item.get("financial_validation")),
                "midplane_residual_variation":result.get("midplane_residual_variation",item.get("midplane_residual_variation")),
                "greeks_status": "VALIDATED" if np.all(result["greek_errors"]<=1e-7) else "GREEK_DISCONTINUITY_OR_ERROR_GATE_FAILED",
                "greeks_qualified":bool(np.all(result["greek_errors"]<=1e-7)),
                "cross_face_status":result.get("cross_face_status",item.get("cross_face_status","NO_ACCEPTED_NEIGHBOR")),
                "cross_face_validation":result.get("cross_face_validation",item.get("cross_face_validation",[]))}

    def cross_faces(candidate):
        updates, invalid, audits = [], [], []
        candidate_failed, candidate_axis = False, candidate.get("split_axis",0)
        for neighbor in accepted:
            if time.monotonic() >= deadline:
                raise RiskError("RESOURCE_LIMIT", "Cross-face work checkpoint required")
            if not any(shared_cell_faces(candidate["bounds"], neighbor["bounds"])):
                continue
            with np.load(folder/neighbor["file"],allow_pickle=False) as saved_cell:
                other = {key:saved_cell[key].copy() for key in ("bounds","coefficients","errors","greek_errors")}
            gate = validate_neighbor_faces(candidate,other,oracle=oracle,price_tolerance=epsilon,deadline=deadline)
            audit={"neighbor_id":neighbor["id"],"faces":gate["faces"],"price_ok":gate["price_ok"],
                   "maximum_price_disagreement":gate["maximum_price_disagreement"].tolist(),"maximum_greek_disagreement":gate["maximum_greek_disagreement"].tolist(),
                   "independent_reference_checked":gate["independent_reference_checked"]}
            audits.append(audit)
            if not gate["price_ok"]:
                if 0 in gate["failed_cells"]:
                    candidate_failed, candidate_axis = True,gate["split_axis"]
                if 1 in gate["failed_cells"]:
                    invalid.append((neighbor,gate["split_axis"]))
                continue
            candidate["errors"] = np.maximum(candidate["errors"],gate["price_errors"][0])
            candidate["greek_errors"] = np.maximum(candidate["greek_errors"],gate["greek_errors"][0])
            other["errors"] = np.maximum(other["errors"],gate["price_errors"][1])
            other["greek_errors"] = np.maximum(other["greek_errors"],gate["greek_errors"][1])
            other["cross_face_status"]="VALIDATED"
            other["cross_face_validation"]=[*neighbor.get("cross_face_validation",[]),{**audit,"neighbor_id":"CANDIDATE_PENDING_ID"}]
            updates.append((neighbor,other))
        candidate["cross_face_status"]="VALIDATED" if audits else "NO_ACCEPTED_NEIGHBOR"
        candidate["cross_face_validation"]=audits
        return candidate_failed,candidate_axis,updates,invalid

    while pending and len(accepted) < max_cells and time.monotonic() < deadline:
        item = pending[0]
        try:
            result = build_cell(np.asarray(item["bounds"]), oracle, index=int(stable_hash(item["id"])[:8], 16), price_tolerance=epsilon, deadline=deadline)
            face_state = cross_faces(result) if result["accepted"] else None
        except RiskError as exc:
            if exc.code == "RESOURCE_LIMIT":
                last_stop={**exc.as_dict(),"cell_id":item["id"],"unfinished_single_pde_solve_restarts":False,"partial_pde_state_checkpointed":exc.details.get("resource")=="normalized_pde"}
                break
            pending.pop(0)
            excluded.append({**item, "reason": exc.code, "details": exc.details})
        else:
            pending.pop(0)
            if result["accepted"]:
                failed,axis,updates,invalid=face_state
                invalid_ids={neighbor["id"] for neighbor,_ in invalid}
                for neighbor,other in updates:
                    if neighbor["id"] in invalid_ids:
                        continue
                    other["cross_face_validation"][-1]["neighbor_id"]=item["id"]
                    accepted[accepted.index(neighbor)]=write_cell(neighbor,other)
                for neighbor,split_axis in invalid:
                    accepted[:]=[old for old in accepted if old["id"]!=neighbor["id"]]
                    split(neighbor,split_axis,"CROSS_FACE_PRICE_REFERENCE_FAILED")
                if failed:
                    split(item,axis,"CROSS_FACE_PRICE_REFERENCE_FAILED")
                else:
                    accepted.append(write_cell(item,result))
            elif item["depth"] >= 20:
                excluded.append({**item, "reason": "PRICE_BASIS_UNVALIDATED_CELL", "errors": result["errors"].tolist(),"financial_validation":result.get("financial_validation")})
            else:
                failed_gate = "CELL_FINANCIAL_INEQUALITY_FAILED" if not result.get("financial_validation",{}).get("passed",True) else "CELL_PRICE_ERROR_GATE_FAILED"
                split(item,result["split_axis"],failed_gate)
        atomic_json(checkpoint, {"identity": identity, "pending": pending, "accepted": accepted, "excluded": excluded})
    atomic_json(checkpoint, {"identity": identity, "pending": pending, "accepted": accepted, "excluded": excluded})
    return {"status": "COMPLETE" if not pending else "CHECKPOINTED", "accepted": accepted, "excluded": excluded, "pending_cells": len(pending), "oracle_calls_this_run": oracle.calls, "shared_grid_solves_this_run":getattr(getattr(oracle,"grid_cache",None),"grid_solves",0),"resumed_pde_schedule_steps":getattr(getattr(oracle,"grid_cache",None),"resumed_steps",0),"shared_grid_memory_hits":getattr(getattr(oracle,"grid_cache",None),"memory_hits",0),"shared_grid_disk_hits":getattr(getattr(oracle,"grid_cache",None),"disk_hits",0),"checkpoint": str(checkpoint), "directory": str(folder),"last_stop":last_stop,"per_oracle_budget_seconds":oracle_budget,"resume_guidance":"Repeat the same source/configuration to resume partial PDE time steps, completed grid refinements, oracle values and accepted cells; a larger wall-clock budget is compatible." if last_stop else None}
