"""Explicit exchange records and the specified conservative calendar scenario."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from calendar import monthrange
from zoneinfo import ZoneInfo
from typing import Iterable

NY = ZoneInfo("America/New_York")


class CalendarError(ValueError):
    def __init__(self, code: str, message: str):
        self.code, self.message = code, message
        super().__init__(f"{code}: {message}")


def local_datetime(value: datetime) -> datetime:
    """Interpret naive wall clock strictly; reject unspecified ambiguous DST folds."""
    if value.tzinfo is not None:
        return value.astimezone(NY)
    early = value.replace(tzinfo=NY, fold=0)
    late = value.replace(tzinfo=NY, fold=1)
    roundtrip = early.astimezone(timezone.utc).astimezone(NY).replace(tzinfo=None)
    if roundtrip != value:
        raise CalendarError("NONEXISTENT_LOCAL_TIME", value.isoformat())
    if early.utcoffset() != late.utcoffset():
        raise CalendarError("AMBIGUOUS_LOCAL_TIME", "Supply an explicit UTC offset")
    return early


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, month, monthrange(year, month)[1])
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def easter(year: int) -> date:
    a, b, c = year % 19, year // 100, year % 100
    d, e, f = b // 4, b % 4, (b + 8) // 25
    g = (b - f + 1) // 3
    h, i, k = (19 * a + b - d - g + 15) % 30, c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    return date(year, (h + l - 7 * m + 114) // 31, (h + l - 7 * m + 114) % 31 + 1)


def observed(day: date, saturday_friday: bool = True) -> date:
    if day.weekday() == 5 and saturday_friday:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def holidays(year: int) -> frozenset[date]:
    values = {observed(date(year, 1, 1), False), nth_weekday(year, 1, 0, 3),
              nth_weekday(year, 2, 0, 3), easter(year) - timedelta(days=2),
              last_weekday(year, 5, 0), observed(date(year, 7, 4)),
              nth_weekday(year, 9, 0, 1), nth_weekday(year, 11, 3, 4),
              observed(date(year, 12, 25))}
    if year >= 2022:
        values.add(observed(date(year, 6, 19)))
    values.update(date.fromisoformat(s) for s in ("2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09") if s.startswith(str(year)))
    return frozenset(values)


@dataclass(frozen=True)
class SessionRecord:
    local_date: date
    exchange_open: datetime
    underlying_close: datetime
    option_close: datetime
    fixing_at: datetime | None = None
    exercise_instruction_cutoff: datetime | None = None
    settlement_business_day: bool = True
    status: str = "OPEN"
    early_close: bool = False
    profile_id: str = "STANDARD_US_CALENDAR_SCENARIO_V1"
    scenario_flag: bool = True


class CalendarService:
    def __init__(self, records: Iterable[SessionRecord] = ()):
        self.records = {record.local_date: record for record in records}

    def session(self, day: date) -> SessionRecord:
        if day in self.records:
            return self.records[day]
        if not 2012 <= day.year <= 2030:
            raise CalendarError("OUTSIDE_CALENDAR_DOMAIN", "Scenario supports years 2012 through 2030; supply explicit SessionRecord")
        closed = day.weekday() >= 5 or day in holidays(day.year)
        thanksgiving = nth_weekday(day.year, 11, 3, 4)
        early = not closed and (day == thanksgiving + timedelta(days=1) or (day.month, day.day) in ((7, 3), (12, 24)))
        close_hour = 13 if early else 16
        return SessionRecord(day, datetime.combine(day, time(9, 30), NY),
                             datetime.combine(day, time(close_hour), NY),
                             datetime.combine(day, time(close_hour, 15), NY),
                             status="CLOSED" if closed else "OPEN", early_close=early,
                             settlement_business_day=self.is_settlement_day(day))

    def minute_labels(self, day: date) -> tuple[datetime, ...]:
        record = self.session(day)
        if record.status != "OPEN":
            return ()
        count = int((record.underlying_close - record.exchange_open).total_seconds() / 60)
        return tuple(record.exchange_open + timedelta(minutes=i) for i in range(count))

    def intervals(self, product: str, start: date, end: date, version: str | None = None) -> tuple[SessionRecord, ...]:
        if product != "QQQ":
            raise CalendarError("UNSUPPORTED_PRODUCT", product)
        return tuple(self.session(start + timedelta(days=i)) for i in range((end - start).days + 1))

    def is_settlement_day(self, day: date) -> bool:
        if day in self.records:
            return self.records[day].settlement_business_day
        return (day.weekday() < 5 and day not in holidays(day.year)
                and day != observed(date(day.year, 11, 11))
                and day != nth_weekday(day.year, 10, 0, 2))

    def settlement_date(self, economic_date: date, lag: int = 1) -> date:
        if lag < 0:
            raise ValueError("Settlement lag cannot be negative")
        current, left = economic_date, lag
        while left:
            current += timedelta(days=1)
            if self.is_settlement_day(current):
                left -= 1
        return current

    def stock_settlement_date(self, day: date, historical: bool = False) -> date:
        lag = 3 if historical and day < date(2017, 9, 5) else 2 if historical and day < date(2024, 5, 28) else 1
        return self.settlement_date(day, lag)

    def next_session(self, day: date) -> SessionRecord:
        for offset in range(1, 370):
            record = self.session(day + timedelta(days=offset))
            if record.status == "OPEN":
                return record
        raise CalendarError("NO_NEXT_SESSION", day.isoformat())

    def advance_trading_minutes(self, decision_at: datetime, count: int) -> datetime:
        if count < 0:
            raise CalendarError("REWIND_REQUIRES_BRANCH", "Only nonnegative advances are accepted")
        current = local_datetime(decision_at)
        if count == 0:
            return current
        day, remaining = current.date(), count
        while remaining:
            for label in self.minute_labels(day):
                available = label + timedelta(minutes=1)
                if available > current:
                    remaining -= 1
                    if not remaining:
                        return available
            day += timedelta(days=1)
        return current

    def scheduled_between(self, start: datetime, end: datetime) -> tuple[datetime, ...]:
        start, end = local_datetime(start), local_datetime(end)
        result, day = [], start.date()
        while day <= end.date():
            result.extend(label for label in self.minute_labels(day) if start < label + timedelta(minutes=1) <= end)
            day += timedelta(days=1)
        return tuple(result)


def add_elapsed_seconds(value: datetime, seconds: float) -> datetime:
    return (local_datetime(value).astimezone(timezone.utc) + timedelta(seconds=seconds)).astimezone(NY)


def add_calendar_days(value: datetime, days: int) -> datetime:
    local = local_datetime(value)
    return local_datetime(local.replace(tzinfo=None) + timedelta(days=days))
