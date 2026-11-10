"""Bounded forward chain summaries; unusual arrival histories use a full read."""
from dataclasses import dataclass
from math import inf

from .models import DataError, stable_hash
from .normalization import canonical_schema, field, resolve_strike


@dataclass(frozen=True)
class QuoteState:
    cutoff: object
    inclusive: bool
    latest: dict
    valid: dict
    scales: frozenset
    spot_lower: float
    spot_upper: float


def chain_quotes(reader, partition, cutoff, spot, mapping, inclusive=False):
    """Keep only the two observations per contract needed by a chain snapshot.

    A late correction can invalidate an older valid quote. Such arrivals, and
    diagnostic histories, deliberately fall back to the complete causal reader.
    Inferred strike units are reused only while every old strike still admits
    exactly the same scale at the current authoritative spot.
    """
    key = ("chain-quote-state-v1", partition.path, partition.size,
           partition.mtime_ns, stable_hash(mapping), reader.calendar.session(partition.key.session_date))
    state = reader.cache.get(key)
    can_extend = (state is not None and cutoff >= state.cutoff
                  and (not state.inclusive or inclusive or cutoff > state.cutoff)
                  and (state.spot_lower == 0 and state.spot_upper == inf
                       or spot is not None and state.spot_lower < spot < state.spot_upper))
    if can_extend and cutoff == state.cutoff and inclusive == state.inclusive:
        return _pairs(state.latest, state.valid), ()
    quotes, diagnostics = reader.quotes(partition, cutoff, spot, mapping, inclusive,
                                        interval_start=state.cutoff if can_extend else None)
    if can_extend and (diagnostics or any(
            quote.contract in state.latest
            and quote.event_at <= state.latest[quote.contract].event_at
            and quote != state.latest[quote.contract] for quote in quotes)):
        quotes, diagnostics = reader.quotes(partition, cutoff, spot, mapping, inclusive)
        can_extend = False
    latest = dict(state.latest) if can_extend else {}
    valid = dict(state.valid) if can_extend else {}
    scales = set(state.scales) if can_extend else set()
    lower, upper = (state.spot_lower, state.spot_upper) if can_extend else (0, inf)
    changed = {}
    for quote in quotes:
        latest[quote.contract] = quote
        changed[quote.contract] = quote
        if quote.valid:
            valid[quote.contract] = quote
    # Every contract has a fixed strike. The reader already verifies consistent
    # divisors within this batch, so unit evidence needs one row per contract.
    schema = canonical_schema(dict(quotes[0].raw_fields)) if quotes else {}
    for quote in changed.values():
        row = dict(quote.raw_fields)
        _, divisor = resolve_strike(field(row, "strike", schema), spot, mapping, schema.get("strike", "strike"))
        scales.add(divisor)
        if mapping.strike_divisor is None and schema.get("strike", "strike").lower() not in ("strike_milli", "strike_millis"):
            lower = max(lower, float(quote.contract.strike / 10))
            upper = min(upper, float(quote.contract.strike * 10))
    if len(scales) > 1:
        raise DataError("INCONSISTENT_STRIKE_SCALE", "Visible rows require conflicting unit mappings")
    if not diagnostics:
        state = QuoteState(cutoff, inclusive, latest, valid, frozenset(scales), lower, upper)
        observations = {id(quote): quote for quote in (*latest.values(), *valid.values())}
        estimated = 512 + 256 * len(latest) + sum(
            1024 + sum(len(name) + len(value) + 128 for name, value in quote.raw_fields)
            for quote in observations.values())
        reader.cache.put(key, state, estimated)
    return _pairs(latest, valid), diagnostics


def _pairs(latest, valid):
    return {contract: (quote, valid.get(contract)) for contract, quote in latest.items()}
