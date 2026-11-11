"""Bounded latest-trade summaries with a causal correction ledger."""
from dataclasses import dataclass, replace
from math import inf
from sys import getsizeof

from .models import DataError, stable_hash
from .normalization import canonical_schema
from .theta import active_trades


@dataclass(frozen=True)
class TradeState:
    cutoff: object
    inclusive: bool
    active_ids: dict
    seen: frozenset
    groups: dict
    last_order: tuple | None
    inferred: bool
    spot_lower: float
    spot_upper: float


def latest_trades(reader, partition, cutoff, spot, mapping, inclusive=False):
    """Select the same latest active observations as the full correction ledger.

    Only current timestamp groups retain trade objects. Older active identities
    remain available for correction validation. Removing a current group member,
    or inserting before an already processed observation, requires a full read.
    """
    from .replay_window import indexed_trade_observations
    indexed = indexed_trade_observations(reader, partition, cutoff, spot, mapping, inclusive)
    if indexed is not None:
        groups = {}
        for item in active_trades(indexed):
            _add_group(groups, item)
        return _selected(groups)
    key=("latest-trade-state-v1",partition.path,partition.size,partition.mtime_ns,
         stable_hash(mapping),reader.calendar.session(partition.key.session_date))
    state=reader.cache.get(key)
    extend=(state is not None and cutoff>=state.cutoff
            and (not state.inclusive or inclusive or cutoff>state.cutoff)
            and (state.spot_lower==0 and state.spot_upper==inf
                 or spot is not None and state.spot_lower<spot<state.spot_upper))
    if extend and cutoff==state.cutoff and inclusive==state.inclusive:
        return _selected(state.groups)
    observations=reader.trades(partition,cutoff,spot,mapping,inclusive=inclusive,
                               interval_start=state.cutoff if extend else None)
    if extend:
        fresh=tuple(item for item in observations if item.source_id not in state.seen)
        updated=_extend(state,fresh)
        if updated is None:
            extend=False
            observations=reader.trades(partition,cutoff,spot,mapping,inclusive=inclusive)
        else:
            active_ids,seen,groups,last_order=updated
            observations=fresh
            inferred=state.inferred
            lower,upper=state.spot_lower,state.spot_upper
    if not extend:
        active=active_trades(observations)
        active_ids={item.trade_id:item.source_id for item in active}
        seen={item.source_id for item in observations}
        groups={}
        for item in active:
            _add_group(groups,item)
        last_order=_order(observations[-1]) if observations else None
        inferred=(mapping.strike_divisor is None
                  and canonical_schema(reader.header(partition)).get("strike","strike").lower()
                  not in ("strike_milli","strike_millis"))
        lower,upper=0,inf
    if inferred:
        # Canceled and corrected observations were also normalized by the full
        # reader. Their original strike inference must remain valid on reuse.
        strikes={item.contract.strike for item in observations}
        if strikes:
            lower=max(lower,float(max(strikes)/10))
            upper=min(upper,float(min(strikes)*10))
    state=TradeState(cutoff,inclusive,active_ids,frozenset(seen),groups,last_order,
                     inferred,lower,upper)
    identity_objects={id(value):value for value in (*state.seen,*active_ids.keys(),*active_ids.values())}
    estimated=(1024+getsizeof(active_ids)+getsizeof(state.seen)+getsizeof(groups)
               +sum(getsizeof(value)+16 for value in identity_objects.values())
               +sum(getsizeof(group)+sum(1024+getsizeof(item.trade_id)+getsizeof(item.source_id)
                                        for item in group) for group in groups.values()))
    reader.cache.put(key,state,estimated)
    return _selected(groups)


def _order(item):
    return (item.known_at,item.event_at,item.sequence if item.sequence is not None else -1,item.source_id)


def _extend(state,observations):
    if observations and state.last_order is not None and _order(observations[0])<=state.last_order:
        return None
    active_ids,seen,groups=dict(state.active_ids),set(state.seen),dict(state.groups)
    latest_sources={item.source_id for group in groups.values() for item in group}
    for item in observations:
        if item.source_id in seen:
            continue
        seen.add(item.source_id)
        if item.action in ("CANCEL","CORRECT"):
            source=active_ids.get(item.correction_of)
            if source is None:
                raise DataError("TRADE_CORRECTION_TARGET_UNKNOWN",str(item.correction_of))
            if source in latest_sources:
                return None
            active_ids.pop(item.correction_of)
        if item.action!="CANCEL":
            if item.trade_id in active_ids and active_ids[item.trade_id]!=item.source_id:
                raise DataError("CONFLICTING_TRADE_ID",item.trade_id)
            active_ids[item.trade_id]=item.source_id
            previous=groups.get(item.contract,())
            _add_group(groups,item)
            current=groups.get(item.contract,())
            if current is not previous:
                latest_sources.difference_update(trade.source_id for trade in previous)
                latest_sources.update(trade.source_id for trade in current)
    last_order=_order(observations[-1]) if observations else state.last_order
    return active_ids,seen,groups,last_order


def _add_group(groups,item):
    if not item.size:
        return
    previous=groups.get(item.contract)
    if previous is None or item.event_at>previous[0].event_at:
        groups[item.contract]=(item,)
    elif item.event_at==previous[0].event_at:
        groups[item.contract]=(*previous,item)


def _selected(groups):
    result={}
    for contract,group in groups.items():
        if all(item.sequence is not None for item in group):
            sequence=max(item.sequence for item in group)
            group=tuple(item for item in group if item.sequence==sequence)
        item=max(group,key=lambda trade:(trade.known_at,trade.source_id))
        if len({trade.price for trade in group})>1:
            item=replace(item,quality_flags=(*item.quality_flags,"AMBIGUOUS_LAST_TRADE_ORDER"))
        result[contract]=item
    return result
