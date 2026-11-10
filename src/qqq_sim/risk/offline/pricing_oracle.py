"""P50-R monotone log-grid American complementarity solver."""
from __future__ import annotations
import math
import time
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_banded
from ..contracts import RiskError
from ..pricing_fast import european_price, deterministic_american


@dataclass(frozen=True)
class PriceResult:
    price: float
    sensitivity: float
    status: str
    diagnostics: dict


def generator_coefficients(sigma, r, q, h):
    a, mu = sigma*sigma/2, r-q-sigma*sigma/2
    lower, upper = a/h**2-mu/(2*h), a/h**2+mu/(2*h)
    if lower < 0 or upper < 0:
        lower, upper = a/h**2+max(-mu, 0)/h, a/h**2+max(mu, 0)/h
    return lower, upper


def solve_lcp(later, obstacle, dt, lower, upper, r, boundaries, tolerance=1e-9):
    n = len(later)-2
    rhs = later[1:-1].copy()
    dl, du, diagonal = -dt*lower, -dt*upper, 1+dt*(lower+upper+r)
    rhs[0] -= dl*boundaries[0]
    rhs[-1] -= du*boundaries[1]
    payoff = obstacle[1:-1]
    current = later[1:-1].copy()
    max_residual = math.inf
    for iteration in range(100):
        continuation = diagonal*current-rhs
        continuation[1:] += dl*current[:-1]
        continuation[:-1] += du*current[1:]
        exercise = current-payoff <= continuation
        ab = np.zeros((3, n))
        ab[1] = np.where(exercise, 1.0, diagonal)
        ab[0, 1:] = np.where(exercise[:-1], 0.0, du)
        ab[2, :-1] = np.where(exercise[1:], 0.0, dl)
        current = solve_banded((1, 1), ab, np.where(exercise, payoff, rhs), check_finite=False)
        continuation = diagonal*current-rhs
        continuation[1:] += dl*current[:-1]
        continuation[:-1] += du*current[1:]
        e = current-payoff
        max_residual = max(float(np.max(np.maximum(-e, 0))), float(np.max(np.maximum(-continuation, 0))), float(np.max(np.abs(np.minimum(e, continuation)))))
        if max_residual <= tolerance+1e-12*max(1, float(np.max(np.abs(current)))):
            return np.concatenate(([boundaries[0]], current, [boundaries[1]])), max_residual, iteration+1
    raise RiskError("PRICER_NOT_CONVERGED", residual=max_residual)


class AmericanGrid:
    """One active lower/upper bracket; requested slices may be consumed streaming."""
    def __init__(self, right, strike, sigma, r, q, domain, nodes=1601, eps=1e-9, time_scale=1,deadline=None):
        self.right, self.strike, self.sigma, self.r, self.q = right, float(strike), sigma, r, q
        self.log_grid = np.linspace(math.log(domain[0]), math.log(domain[1]), nodes)
        self.spots = np.exp(self.log_grid)
        self.obstacle = np.maximum((1 if right == "CALL" else -1)*(self.spots-self.strike), 0)
        self.lower = self.obstacle.copy()
        self.upper = self.obstacle.copy()
        self.l, self.u = generator_coefficients(sigma, r, q, self.log_grid[1]-self.log_grid[0])
        self.tau, self.eps, self.time_scale = 0., min(1e-9, eps/100), time_scale
        self.residual, self.steps, self.max_iterations = 0., 0, 0
        self.deadline=deadline

    def _boundary(self, tau):
        s = self.spots[[0, -1]]
        lower = np.maximum(self.obstacle[[0, -1]], european_price(self.right, s, self.strike, tau, self.r, self.q, self.sigma))
        upper = s*math.exp(max(-self.q, 0)*tau) if self.right == "CALL" else np.full(2, self.strike*math.exp(max(-self.r, 0)*tau))
        return lower, upper

    def _step(self, dt, halvings=0):
        lower_boundary, upper_boundary = self._boundary(self.tau+dt)
        try:
            low, rl, il = solve_lcp(self.lower, self.obstacle, dt, self.l, self.u, self.r, lower_boundary, self.eps)
            up, ru, iu = solve_lcp(self.upper, self.obstacle, dt, self.l, self.u, self.r, upper_boundary, self.eps)
        except RiskError:
            if halvings >= 8:
                raise
            self._step(dt/2, halvings+1)
            self._step(dt/2, halvings+1)
            return
        if np.min(up-low) < -1e-8:
            raise RiskError("PRICER_NOT_CONVERGED", "Boundary comparison principle failed")
        self.lower, self.upper = low, up
        self.tau += dt
        self.residual, self.max_iterations = max(self.residual, rl, ru), max(self.max_iterations, il, iu)
        self.steps += 1

    def advance(self, tau):
        if tau < self.tau-1e-15:
            raise RiskError("INVALID_PRICING_ORDER")
        while self.tau < tau-1e-15:
            early=self.tau < 1/365
            segment_end=min(tau,1/365) if early else tau
            segment_start=self.tau
            duration=segment_end-segment_start
            base_step=(60 if early else 3600)/(365*86400)
            if self.r < 0:
                base_step=min(base_step,.25/(-self.r))
            # Refine every coarse interval, including a short final interval.
            # min(remaining, nominal/time_scale) would produce identical
            # single steps at scales1/2/4 for horizons below15seconds.
            coarse_steps=max(1,math.ceil(duration/base_step))
            step_count=coarse_steps*self.time_scale
            for index in range(step_count):
                if self.deadline is not None and time.monotonic()>=self.deadline:
                    raise RiskError("RESOURCE_LIMIT","American PDE time-step budget reached",steps=self.steps)
                next_tau=segment_end if index==step_count-1 else segment_start+duration*(index+1)/step_count
                self._step(next_tau-self.tau)
                self.tau=next_tau
        return (self.lower+self.upper)/2

    def evaluate(self, spot):
        s = np.asarray(spot)
        if np.any(s < self.spots[0]) or np.any(s > self.spots[-1]):
            raise RiskError("NUMERICAL_DOMAIN_UNRESOLVED")
        lo, hi = np.interp(s, self.spots, self.lower), np.interp(s, self.spots, self.upper)
        return (lo+hi)/2, (hi-lo)/2


def american_price(right, spot, strike, tau, r, q, sigma, *, nodes=1601, max_nodes=12801, eps=1e-5, max_seconds=120, require_convergence=True):
    if tau < 0 or spot < 0 or strike <= 0:
        raise RiskError("MODEL_MARKS_INVALID")
    if tau == 0 or sigma == 0:
        return PriceResult(float(deterministic_american(right, spot, strike, tau, r, q)), 0, "EXACT", {"method": "DETERMINISTIC_STOPPING"})
    if spot == 0:
        return PriceResult(0.0 if right == "CALL" else strike*max(1, math.exp(-r*tau)), 0, "EXACT", {"method": "ZERO_SPOT_LIMIT"})
    if right == "CALL" and q == 0 and r >= 0:
        return PriceResult(float(european_price(right, spot, strike, tau, r, q, sigma)), 0., "EXACT_FORMULA", {"method": "NO_DIVIDEND_NONNEGATIVE_RATE_CALL"})
    started = time.monotonic()
    pad = max(.5, 8*sigma*math.sqrt(tau)+abs(r-q-sigma*sigma/2)*tau)
    domain = (min(spot, strike)*math.exp(-pad), max(spot, strike)*math.exp(pad))
    records, history = [], {}
    time_scale = 1
    last = None
    while nodes <= max_nodes:
        if time.monotonic()-started > max_seconds:
            raise RiskError("RESOURCE_LIMIT", "American pricing budget reached", refinements=records)
        coarse = AmericanGrid(right, strike, sigma, r, q, domain, nodes, eps, time_scale,started+max_seconds)
        coarse.advance(tau)
        checks = np.unique(np.concatenate(([spot, strike, strike*(1-1e-6), strike*(1+1e-6)], np.exp(np.linspace(math.log(max(domain[0]*1.01, min(spot, strike)*.8)), math.log(min(domain[1]/1.01, max(spot, strike)*1.2)), 257)))))
        values, boundary = coarse.evaluate(checks)
        value, bound = coarse.evaluate(spot)
        history[(nodes,time_scale)]=values
        spatial = []
        temporal = []
        if (nodes,time_scale//2) in history and (nodes,time_scale//4) in history:
            temporal=[float(np.max(np.abs(values-history[nodes,time_scale//2]))),float(np.max(np.abs(history[nodes,time_scale//2]-history[nodes,time_scale//4])))]
        coarse_nodes=(nodes+1)//2
        quarter_nodes=(coarse_nodes+1)//2
        if (coarse_nodes,time_scale) in history and (quarter_nodes,time_scale) in history:
            spatial=[float(np.max(np.abs(values-history[coarse_nodes,time_scale]))),float(np.max(np.abs(history[coarse_nodes,time_scale]-history[quarter_nodes,time_scale])))]
        difference=max((*spatial,*temporal),default=math.inf)
        max_boundary = float(np.max(boundary))
        last = PriceResult(float(value), max(float(bound), difference), "PRELIMINARY", {"refinements": records, "domain": domain})
        records.append({"nodes": nodes, "time_scale": time_scale, "max_difference": None if not math.isfinite(difference) else difference, "boundary_halfwidth": max_boundary, "residual": coarse.residual, "steps": coarse.steps})
        if max_boundary > eps/4:
            width = math.log(domain[1]/domain[0])
            expand = max(.5, .25*width)
            factor = (width+2*expand)/width
            nodes = int((nodes-1)*factor)+1
            domain = (domain[0]*math.exp(-expand), domain[1]*math.exp(expand))
            history={}
            continue
        passed=len(spatial)==2 and len(temporal)==2 and difference<=eps/4
        if passed or not require_convergence:
            return PriceResult(float(value), max(float(bound), difference if math.isfinite(difference) else 0.), "CONVERGED" if passed else "PRELIMINARY", {"refinements": records, "domain": domain, "space_differences":spatial,"time_differences":temporal,"error_interpretation": "MEASURED_NUMERICAL_SENSITIVITY"})
        # Complete a Cartesian refinement block so two space differences share
        # one time grid and two time differences share one spatial grid.
        if time_scale < 4:
            time_scale *= 2
        else:
            nodes = 2*nodes-1
            time_scale = 1
    raise RiskError("PRICER_NOT_CONVERGED", "Required two successive refinements were not demonstrated", refinements=records)


def crr_price(right, spot, strike, tau, r, q, sigma, steps=2000):
    if tau == 0 or sigma == 0:
        return float(deterministic_american(right, spot, strike, tau, r, q))
    dt = tau/steps
    up = math.exp(sigma*math.sqrt(dt))
    p = (math.exp((r-q)*dt)-1/up)/(up-1/up)
    if not 0 <= p <= 1:
        raise RiskError("TREE_TRANSITION_OUTSIDE_DOMAIN")
    discount = math.exp(-r*dt)
    eta = 1 if right == "CALL" else -1
    nodes = spot*np.exp((2*np.arange(steps+1)-steps)*math.log(up))
    value = np.maximum(eta*(nodes-strike), 0)
    for n in range(steps-1, -1, -1):
        nodes = spot*np.exp((2*np.arange(n+1)-n)*math.log(up))
        value = np.maximum(eta*(nodes-strike), discount*((1-p)*value[:-1]+p*value[1:]))
    return float(value[0])


def invert_iv(right, spot, strike, tau, r, q, price, style="EUROPEAN", eps=1e-6, explicit_zero=False):
    def pricer(vol):
        if style == "EUROPEAN":
            return float(european_price(right, spot, strike, tau, r, q, vol))
        return american_price(right, spot, strike, tau, r, q, vol, eps=eps).price
    lower = pricer(0)
    if price < lower-eps:
        raise RiskError("IV_PRICE_OUT_OF_RANGE")
    if abs(price-lower) <= eps:
        if explicit_zero:
            return {"sigma": 0., "status": "EXPLICIT_ZERO_SCENARIO", "interval": [0., 0.]}
        raise RiskError("LEG_IV_NOT_IDENTIFIABLE", "Intrinsic plateau does not identify zero IV")
    lo, hi = 0., .25
    while hi < 8 and pricer(hi) < price:
        hi *= 2
    hi = min(hi, 8.)
    if pricer(hi) < price-eps:
        raise RiskError("LEG_IV_OUTSIDE_MODEL_DOMAIN")
    for _ in range(80):
        mid = (lo+hi)/2
        residual = pricer(mid)-price
        if hi-lo <= 1e-8 and abs(residual) <= eps:
            return {"sigma": mid, "status": "OK", "interval": [lo, hi], "residual": residual}
        if residual > 0:
            hi = mid
        else:
            lo = mid
    raise RiskError("IV_SOLVER_UNRESOLVED")
