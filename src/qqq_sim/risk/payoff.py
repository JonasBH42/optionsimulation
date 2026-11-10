"""Exact piecewise-affine payoff extrema and complete threshold topology."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
import numpy as np
from .contracts import RiskError, RiskLeg


def payoff_decimal(legs, spot, cash=0, shares=0) -> Decimal:
    s = Decimal(str(spot))
    out = Decimal(str(cash)) + Decimal(str(shares)) * s
    for leg in legs:
        eta = 1 if leg.right == "CALL" else -1
        out += leg.quantity * leg.multiplier * max(Decimal(0), eta * (s - leg.strike))
    return out


def payoff_array(legs, spot, cash=0, shares=0):
    s = np.asarray(spot, dtype=np.float64)
    value = float(cash) + shares * s
    for leg in legs:
        value = value + leg.quantity * leg.multiplier * np.maximum((1 if leg.right == "CALL" else -1) * (s - float(leg.strike)), 0)
    return value


def payoff_extrema(legs, entry_cash=0, shares=0):
    strikes = sorted({Decimal(0), *(x.strike for x in legs)})
    values = [payoff_decimal(legs, x, entry_cash, shares) for x in strikes]
    upper_slope = shares + sum(x.quantity*x.multiplier for x in legs if x.right == "CALL")
    lower_slope = shares - sum(x.quantity*x.multiplier for x in legs if x.right == "PUT")
    maximum = Decimal("Infinity") if upper_slope > 0 else max(values)
    minimum = Decimal("-Infinity") if upper_slope < 0 else min(values)
    return {"max_profit": maximum, "max_loss": max(Decimal(0), -minimum), "min_profit": minimum,
            "tail_slopes": (lower_slope, upper_slope), "breakpoints": tuple(zip(strikes, values))}


def original_target(legs, entry_cash, fraction=Decimal("0.5")):
    extrema = payoff_extrema(legs, entry_cash)
    if not extrema["max_profit"].is_finite() or extrema["max_profit"] <= 0:
        raise RiskError("UNDEFINED_MAX_PROFIT_TARGET")
    if not extrema["max_loss"].is_finite():
        raise RiskError("UNBOUNDED_LOSS")
    return extrema["max_profit"] * Decimal(str(fraction))


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    lower_closed: bool = True
    upper_closed: bool = True

    def contains(self, x):
        x = np.asarray(x)
        return ((x >= self.lower) if self.lower_closed else (x > self.lower)) & ((x <= self.upper) if self.upper_closed else (x < self.upper))


def merge_intervals(intervals):
    output = []
    for item in sorted(intervals, key=lambda x: (x.lower, x.upper)):
        if item.lower > item.upper or (item.lower == item.upper and not (item.lower_closed and item.upper_closed)):
            continue
        if output and (item.lower < output[-1].upper or (item.lower == output[-1].upper and (item.lower_closed or output[-1].upper_closed))):
            prev = output.pop()
            top = max(prev.upper, item.upper)
            closed = prev.upper_closed if prev.upper > item.upper else item.upper_closed if item.upper > prev.upper else prev.upper_closed or item.upper_closed
            output.append(Interval(prev.lower, top, prev.lower_closed, closed))
        else:
            output.append(item)
    return tuple(output)


def piecewise_regions(knots, values, strict=False):
    """All components of a declared linear interpolant, including tangencies."""
    regions = []
    for a, b, fa, fb in zip(knots[:-1], knots[1:], values[:-1], values[1:]):
        a, b, fa, fb = map(float, (a, b, fa, fb))
        if a >= b:
            raise RiskError("INVALID_REGION_GRID")
        ia, ib = (fa > 0, fb > 0) if strict else (fa >= 0, fb >= 0)
        if ia and ib:
            regions.append(Interval(a, b))
        elif ia != ib:
            root = a + (b-a)*(-fa)/(fb-fa)
            regions.append(Interval(a, root, True, not strict) if ia else Interval(root, b, not strict, True))
        elif not strict:
            if fa == 0:
                regions.append(Interval(a, a))
            if fb == 0:
                regions.append(Interval(b, b))
    return merge_intervals(regions)


def terminal_regions(legs, entry_cash, threshold, strict=False, shares=0):
    strikes = sorted({Decimal(0), *(x.strike for x in legs)})
    values = [payoff_decimal(legs, x, entry_cash, shares)-Decimal(str(threshold)) for x in strikes]
    regions = list(piecewise_regions(strikes, values, strict))
    a, fa = strikes[-1], values[-1]
    slope = Decimal(str(shares + sum(x.quantity*x.multiplier for x in legs if x.right == "CALL")))
    if slope == 0 and (fa > 0 if strict else fa >= 0):
        regions.append(Interval(float(a), math.inf, not strict or fa != 0, False))
    elif slope > 0:
        root = max(a, a-fa/slope)
        regions.append(Interval(float(root), math.inf, not strict or root == a and fa > 0, False))
    elif slope < 0 and (fa > 0 if strict else fa >= 0):
        regions.append(Interval(float(a), float(a-fa/slope), True, not strict))
    return merge_intervals(regions)


def region_membership(regions, spot):
    out = np.zeros(np.asarray(spot).shape, dtype=bool)
    for region in regions:
        out |= region.contains(spot)
    return out
