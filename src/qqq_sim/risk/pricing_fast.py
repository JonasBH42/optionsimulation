"""Deterministic pricing formulas and stored Chebyshev evaluation only.

No PDE, simulation, training or iterative grid construction is imported here.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import ndtr
from .contracts import RiskError


def european_price(kind, spot, strike, tau, r, q, sigma, state_kind="SPOT"):
    s = np.asarray(spot, dtype=np.float64)
    k = float(strike)
    if np.any(s < 0) or not np.all(np.isfinite(s)) or k <= 0 or tau < 0 or sigma < 0:
        raise RiskError("MODEL_MARKS_INVALID")
    eta = 1 if kind.upper() in ("CALL", "C") else -1
    if tau == 0:
        return np.maximum(eta*(s-k), 0)
    discount = math.exp(-r*tau)
    forward = s if state_kind == "TERMINAL_FORWARD" else s*math.exp((r-q)*tau)
    if sigma == 0:
        return discount*np.maximum(eta*(forward-k), 0)
    vol = sigma*math.sqrt(tau)
    with np.errstate(divide="ignore"):
        d1 = np.log(forward/k)/vol+0.5*vol
    d2 = d1-vol
    # Evaluate the out-of-money side, then use parity on the ITM side.
    otm_call = discount*(forward*ndtr(d1)-k*ndtr(d2))
    otm_put = discount*(k*ndtr(-d2)-forward*ndtr(-d1))
    value = np.where(forward >= k, otm_put+discount*(forward-k), otm_call) if eta == 1 else np.where(forward <= k, otm_call+discount*(k-forward), otm_put)
    if np.any(value < -1e-11):
        raise RiskError("MODEL_MARKS_INVALID", "Negative European value")
    return value


def deterministic_american(kind, spot, strike, tau, r, q):
    s = np.asarray(spot, dtype=np.float64)
    k = float(strike)
    eta = 1 if kind.upper() in ("CALL", "C") else -1
    if tau < 0 or k <= 0 or np.any(s < 0):
        raise RiskError("MODEL_MARKS_INVALID")
    value = np.maximum(eta*(s-k), 0)
    value = np.maximum(value, eta*(s*math.exp(-q*tau)-k*math.exp(-r*tau)))
    if r and q and r != q:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = r*k/(q*s)
            u = np.log(ratio)/(r-q)
        valid = (ratio > 0) & (u > 0) & (u < tau)
        safe_u = np.where(valid, u, 0)
        candidate = eta*(s*np.exp(-q*safe_u)-k*np.exp(-r*safe_u))
        value = np.maximum(value, np.where(valid, candidate, 0))
    return value


def chebyshev_basis(z):
    z = np.asarray(z, dtype=np.float64)
    return np.stack((np.ones_like(z), z, 2*z*z-1, 4*z*z*z-3*z), axis=-1)


def tensor_chebyshev(coefficients, coordinates, derivative_axis=None, second=False):
    """Degree-three tensor, with analytic normalized-coordinate derivatives."""
    bases = []
    for axis, z in enumerate(coordinates):
        if axis == derivative_axis:
            bases.append(np.array([0, 0, 4, 24*z]) if second else np.array([0, 1, 4*z, 12*z*z-3]))
        else:
            bases.append(chebyshev_basis(z))
    return float(np.einsum("ijkl,i,j,k,l->", coefficients, *bases, optimize=True))


PRICE_TREE_SCHEMA = "PRICE_AABB_TREE_V1"
MAX_PRICE_CELLS = 262144
MAX_PRICE_TREE_DEPTH = 64
MAX_PRICE_LOOKUP_NODES = 4096
MAX_PRICE_LOOKUP_MATCHES = 256


class PriceDecisionTree:
    """Validated saved binary spatial decisions, including inclusive faces."""
    def __init__(self,cell_bounds,nodes,bounds):
        self.nodes=np.asarray(nodes)
        self.bounds=np.asarray(bounds)
        count=len(cell_bounds); size=2*count-1
        def invalid(reason):
            raise RiskError('PRICE_TREE_INVALID',detail=reason)
        if (not 1<=count<=MAX_PRICE_CELLS or self.nodes.dtype!=np.dtype('<i8')
            or self.bounds.dtype!=np.dtype('<f8') or self.nodes.shape!=(size,3)
            or self.bounds.shape!=(size,4,2)):
            invalid('Invalid saved tree dtype, shape or cell budget')
        if not np.all(np.isfinite(self.bounds)) or np.any(self.bounds[:,:,0]>=self.bounds[:,:,1]):
            invalid('Nonfinite or empty node bounds')
        leaf=self.nodes[:,2]>=0
        leaves=np.flatnonzero(leaf); branches=np.flatnonzero(~leaf)
        if len(leaves)!=count or np.any(self.nodes[leaves,:2]!=-1) or np.any(self.nodes[branches,2]!=-1):
            invalid('Invalid branch/leaf encoding')
        ids=self.nodes[leaves,2]
        if not np.array_equal(np.sort(ids),np.arange(count,dtype=np.int64)):
            invalid('Each accepted cell must occur exactly once')
        if not np.array_equal(self.bounds[leaves],cell_bounds[ids]):
            invalid('Leaf bounds differ from their accepted cell')
        children=self.nodes[branches,:2]
        # Canonical preorder: descendants have larger indices. Along with one
        # parent for each nonroot node this proves reachability and no cycles.
        if (np.any(children<=branches[:,None]) or np.any(children>=size)
            or not np.array_equal(np.sort(children.ravel()),np.arange(1,size,dtype=np.int64))):
            invalid('Invalid, cyclic, shared or unreachable child index')
        if len(branches):
            left,right=self.bounds[children[:,0]],self.bounds[children[:,1]]
            expected=np.stack((np.minimum(left[:,:,0],right[:,:,0]),np.maximum(left[:,:,1],right[:,:,1])),axis=-1)
            if not np.array_equal(self.bounds[branches],expected):
                invalid('Branch bounds do not exactly cover both children')
        depth=np.zeros(size,dtype=np.int16)
        for index in branches:
            child_depth=int(depth[index])+1
            if child_depth>MAX_PRICE_TREE_DEPTH:
                invalid('Tree depth exceeds the online budget')
            depth[self.nodes[index,:2]]=child_depth
        self.max_depth=int(depth.max())
        self.nodes.flags.writeable=False
        self.bounds.flags.writeable=False

    def locate(self,coordinates):
        point=np.asarray(coordinates,dtype=np.float64)
        if point.shape!=(4,) or not np.all(np.isfinite(point)):
            raise RiskError('PRICE_BASIS_OUTSIDE_DOMAIN')
        stack=[0]; found=[]; visited=0
        while stack:
            index=stack.pop(); visited+=1
            if visited>MAX_PRICE_LOOKUP_NODES:
                raise RiskError('PRICE_TREE_LOOKUP_LIMIT',visited_nodes=visited)
            bounds=self.bounds[index]
            if np.any(point<bounds[:,0]) or np.any(point>bounds[:,1]):
                continue
            left,right,cell=self.nodes[index]
            if cell>=0:
                found.append(int(cell))
                if len(found)>MAX_PRICE_LOOKUP_MATCHES:
                    raise RiskError('PRICE_TREE_LOOKUP_LIMIT',matched_cells=len(found))
            else:
                stack.extend((int(right),int(left)))
        # All shared-face cells remain available for consistency checks, while
        # the original lower cell index determines the chosen interpolant.
        return np.asarray(sorted(found),dtype=np.int64)


class PriceBasis:
    """Read-only measured cells. Face order is stable; extrapolation is forbidden."""
    def __init__(self, bounds, coefficients, errors, greek_errors=None, *, tree_nodes=None, tree_bounds=None):
        self.bounds = np.asarray(bounds, dtype=np.float64)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.errors = np.asarray(errors, dtype=np.float64)
        self.greek_errors = None if greek_errors is None else np.asarray(greek_errors,dtype=np.float64)
        count=self.bounds.shape[0] if self.bounds.ndim else 0
        if (self.bounds.ndim != 3 or self.bounds.shape[1:] != (4, 2) or not 1<=count<=MAX_PRICE_CELLS
            or self.coefficients.shape!=(count,2,4,4,4,4) or self.errors.shape!=(count,2)
            or np.any(self.bounds[:,:,0]>=self.bounds[:,:,1]) or np.any(self.errors<0)):
            raise RiskError("INVALID_PRICE_BASIS")
        for arr in (self.bounds, self.coefficients, self.errors):
            if not np.all(np.isfinite(arr)):
                raise RiskError("INVALID_PRICE_BASIS")
            arr.flags.writeable = False
        if self.greek_errors is not None:
            if self.greek_errors.shape!=(count,2,4) or not np.all(np.isfinite(self.greek_errors)) or np.any(self.greek_errors<0):
                raise RiskError('INVALID_PRICE_BASIS')
            self.greek_errors.flags.writeable=False
        if (tree_nodes is None)!=(tree_bounds is None):
            raise RiskError('PRICE_TREE_INVALID',detail='Both saved tree arrays are required')
        self.tree=None if tree_nodes is None else PriceDecisionTree(self.bounds,tree_nodes,tree_bounds)

    def locate(self,coordinates):
        if self.tree is not None:
            return self.tree.locate(coordinates)
        # Compatibility for explicitly unqualified legacy/research fixtures.
        point=np.asarray(coordinates,dtype=np.float64)
        return np.flatnonzero(np.all((point>=self.bounds[:,:,0]) & (point<=self.bounds[:,:,1]),axis=1))

    def evaluate(self, right, spot, strike, tau, r, q, sigma, greeks=False):
        if tau == 0:
            return {"price": float(european_price(right, spot, strike, tau, r, q, sigma)), "error": 0.0, "greeks_reason": "GREEK_DISCONTINUITY"}
        if sigma == 0:
            return {"price": float(deterministic_american(right, spot, strike, tau, r, q)), "error": 0.0, "greeks_reason": "GREEK_DISCONTINUITY"}
        if spot <= 0 or strike <= 0 or tau < 0:
            raise RiskError("PRICE_BASIS_OUTSIDE_DOMAIN")
        x = np.array([math.log(spot/strike), sigma*math.sqrt(tau), r*tau, q*tau])
        valid = self.locate(x)
        if not len(valid):
            raise RiskError("PRICE_BASIS_UNVALIDATED_CELL", coordinates=x.tolist())
        index = int(valid[0])
        lo, hi = self.bounds[index, :, 0], self.bounds[index, :, 1]
        z = 2*(x-lo)/(hi-lo)-1
        scale = 2/(hi-lo)
        right_index = 0 if right in ("CALL", "C") else 1
        coefs = self.coefficients[index, right_index]
        f = tensor_chebyshev(coefs, z)
        result = {"price": strike*f, "error": strike*float(self.errors[index, right_index]), "cell": index}
        intrinsic = max((1 if right_index == 0 else -1)*(spot-strike), 0)
        if result["price"]+result["error"] < intrinsic:
            raise RiskError("PRICE_BASIS_OBSTACLE_FAILURE")
        if greeks:
            if self.greek_errors is None:
                result["greeks_reason"] = "GREEKS_UNVALIDATED"
            else:
                d = [tensor_chebyshev(coefs, z, axis)*scale[axis] for axis in range(4)]
                dxx = tensor_chebyshev(coefs, z, 0, True)*scale[0]**2
                result["greeks"] = {"delta": strike/spot*d[0], "gamma": strike/spot**2*(dxx-d[0]), "vega_per_1pct": .01*strike*math.sqrt(tau)*d[1], "theta_per_day": -strike/365*(sigma/(2*math.sqrt(tau))*d[1]+r*d[2]+q*d[3])}
                normalized=np.asarray(self.greek_errors[index,right_index],dtype=np.float64)
                result["greek_errors"] = (normalized*np.array([1,1/strike,strike*math.sqrt(tau),strike/tau])).tolist()
                unavailable = [name for name,error in zip(("delta","gamma","vega_per_1pct","theta_per_day"),normalized) if error>1e-7]
                if unavailable:
                    result["greeks_reason"]="GREEK_DISCONTINUITY_OR_ERROR_GATE_FAILED"
                    result["unavailable_greeks"]=unavailable
                    for name in unavailable:
                        result["greeks"].pop(name,None)
        # A shared-face lookup chooses the first stored cell deterministically,
        # but its neighbors must agree within their stored error envelopes.
        if len(valid)>1:
            result["shared_face_cells"]=valid.tolist()
            for neighbor in valid[1:]:
                nlo,nhi=self.bounds[neighbor,:,0],self.bounds[neighbor,:,1]
                nz=2*(x-nlo)/(nhi-nlo)-1
                nscale=2/(nhi-nlo)
                nc=self.coefficients[neighbor,right_index]
                nprice=strike*tensor_chebyshev(nc,nz)
                tolerance=strike*float(self.errors[index,right_index]+self.errors[neighbor,right_index])+16*np.finfo(float).eps*max(1.,abs(nprice),abs(result["price"]))
                if abs(nprice-result["price"])>tolerance:
                    raise RiskError("PRICE_BASIS_CROSS_FACE_FAILURE",cells=[index,int(neighbor)],price_difference=abs(nprice-result["price"]),allowed=tolerance)
                if greeks and self.greek_errors is not None:
                    nd=[tensor_chebyshev(nc,nz,axis)*nscale[axis] for axis in range(4)]
                    ndxx=tensor_chebyshev(nc,nz,0,True)*nscale[0]**2
                    neighboring=np.array([strike/spot*nd[0],strike/spot**2*(ndxx-nd[0]),.01*strike*math.sqrt(tau)*nd[1],-strike/365*(sigma/(2*math.sqrt(tau))*nd[1]+r*nd[2]+q*nd[3])])
                    local=np.array([strike/spot*d[0],strike/spot**2*(dxx-d[0]),.01*strike*math.sqrt(tau)*d[1],-strike/365*(sigma/(2*math.sqrt(tau))*d[1]+r*d[2]+q*d[3])])
                    factor=np.array([1,1/strike,strike*math.sqrt(tau),strike/tau])
                    allowed=(np.asarray(self.greek_errors[index,right_index])+np.asarray(self.greek_errors[neighbor,right_index]))*factor+1e-12
                    for offset,name in enumerate(("delta","gamma","vega_per_1pct","theta_per_day")):
                        if abs(local[offset]-neighboring[offset])>allowed[offset] or self.greek_errors[neighbor,right_index,offset]>1e-7:
                            result["greeks"].pop(name,None)
                            result["greeks_reason"]="GREEK_DISCONTINUITY"
                            unavailable=result.setdefault("unavailable_greeks",[])
                            if name not in unavailable:
                                unavailable.append(name)
                            result.setdefault("greek_intervals",{})[name]=[float(min(local[offset],neighboring[offset])-allowed[offset]),float(max(local[offset],neighboring[offset])+allowed[offset])]
        return result

    def inverse_iv(self, right, spot, strike, tau, r, q, mark):
        lo, hi = 0.0, 2.0
        left = self.evaluate(right, spot, strike, tau, r, q, lo)
        right_value = self.evaluate(right, spot, strike, tau, r, q, hi)
        if mark < left["price"]-left["error"] or mark > right_value["price"]+right_value["error"]:
            raise RiskError("IV_PRICE_OUT_OF_RANGE")
        for _ in range(64):
            mid = (lo+hi)/2
            value = self.evaluate(right, spot, strike, tau, r, q, mid)
            if value["price"] < left["price"]-value["error"]-left["error"] or value["price"] > right_value["price"]+value["error"]+right_value["error"]:
                raise RiskError("PRICE_BASIS_MONOTONICITY_FAILURE")
            if mark < value["price"]-value["error"]:
                hi, right_value = mid, value
            elif mark > value["price"]+value["error"]:
                lo, left = mid, value
            else:
                return {"status": "OK" if hi-lo <= 1e-4 else "IV_INTERVAL_ONLY", "sigma": mid if hi-lo <= 1e-4 else None, "interval": [lo, hi]}
        return {"status": "IV_INTERVAL_ONLY", "sigma": None, "interval": [lo, hi]}
