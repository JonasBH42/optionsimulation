from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from .calendar import CalendarService, add_elapsed_seconds, local_datetime


@dataclass(frozen=True)
class ReplayClock:
    decision_at_ny: datetime
    state_version: int = 0
    event_sequence: int = 0
    phase: int = 25
    profile: str = "BAR_END_DECISION_BARRIER_V1"

    def __post_init__(self):
        object.__setattr__(self, "decision_at_ny", local_datetime(self.decision_at_ny))

    @property
    def visible_bar_label_ny(self) -> datetime:
        return self.decision_at_ny - timedelta(minutes=1)

    @property
    def decision_at_utc(self) -> datetime:
        return self.decision_at_ny.astimezone(timezone.utc)

    def trading_target(self, count: int, calendar: CalendarService) -> datetime:
        return calendar.advance_trading_minutes(self.decision_at_ny, count)

    def elapsed_target(self, seconds: float) -> datetime:
        return add_elapsed_seconds(self.decision_at_ny, seconds)

    def committed(self, target: datetime, phase: int = 25) -> "ReplayClock":
        return replace(self, decision_at_ny=target, phase=phase, state_version=self.state_version + 1, event_sequence=self.event_sequence + 1)
