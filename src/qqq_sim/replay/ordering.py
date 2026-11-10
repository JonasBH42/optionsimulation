"""Stable event ordering with the completed-bar decision barrier."""
from heapq import merge
from typing import Iterable, Iterator
from qqq_sim.data.models import SourceEvent

PHASES = {"rule": 10, "cancellation": 20, "candle": 20, "bar_snapshot": 25,
          "bar_command": 26, "quote": 30, "trade": 30, "fixing": 35, "match": 40,
          "posting": 50, "exercise": 60, "settlement": 70, "snapshot": 80, "command": 90}


def merge_events(*streams: Iterable[SourceEvent]) -> Iterator[SourceEvent]:
    yield from merge(*streams, key=lambda event: event.key)


def before_decision_barrier(event: SourceEvent, decision_at) -> bool:
    return event.event_at < decision_at or (event.event_at == decision_at and event.phase <= 25)
