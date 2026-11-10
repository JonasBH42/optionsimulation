"""Pure fast risk APIs. Expensive reference/build modules are explicit imports."""

from .contracts import ReferenceJob, RiskLeg, RiskError
from .payoff import payoff_extrema, terminal_regions

__all__ = ["ReferenceJob", "RiskLeg", "RiskError", "payoff_extrema", "terminal_regions"]
